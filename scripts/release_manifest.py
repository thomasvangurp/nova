#!/usr/bin/env python3
"""Build and verify a release from committed bytes, including its test evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_PATHS = (
    'external_tools/fast_nanobody_scorer/fast_score_batch.py',
    'external_tools/fast_nanobody_scorer/resident.py',
    'external_tools/fast_nanobody_scorer/synth_fullatom.py',
    'nanobody_opt', 'nanobody_opt_v2', 'nanobody_dash', 'tools', 'neurons', 'compound_miner',
    'utils', 'config', 'combinatorial_db/reactions.py', 'tests', 'scripts',
    'deploy', 'ansible', '.github/workflows', 'requirements',
    'pixi.toml', 'pixi.lock', 'go_live_v2.sh', 'restart_v2.sh', 'deploy_gate.sh',
)
RUNTIME_DIRS = ('nanobody_opt', 'nanobody_opt_v2', 'nanobody_dash', 'tools', 'neurons',
                'compound_miner', 'utils', 'config', 'combinatorial_db',
                'tests', 'scripts', 'deploy')
LIVE_CONFIG = {'config/config.yaml', 'config/boltzgen_config.yaml'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    if result.returncode:
        raise ValueError('git source verification failed')
    return result.stdout


def runtime_file(name: str) -> bool:
    return name not in LIVE_CONFIG and (
        name.split('/')[0] in RUNTIME_DIRS or name in {'pixi.toml', 'pixi.lock', 'deploy_gate.sh'})


def safe_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('invalid release path')
    if any(part in {'.env', '.npmrc', '.pypirc', 'credentials', '.git'}
           or part.startswith('.env.') or part.endswith(('.pem', '.key')) for part in relative.parts):
        raise ValueError('credential material is forbidden in a release')
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('release paths must stay inside the source tree')
    return path


def stage(root: Path, destination: Path) -> dict:
    revision = git(root, 'rev-parse', 'HEAD').strip()
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('release revision must be a full commit SHA')
    if git(root, 'status', '--porcelain', '--untracked-files=all', '--', *SOURCE_PATHS).strip():
        raise ValueError('commit all release-source changes before building')
    names = git(root, 'ls-files', '-z', '--', *SOURCE_PATHS).split('\0')
    if destination.exists():
        raise ValueError('release destination already exists')
    destination.mkdir(parents=True)
    files = {}
    for name in sorted(filter(None, names)):
        source = safe_path(root, name)
        if not source.is_file():
            raise ValueError('release source is missing or is not a regular file')
        target = safe_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[name] = digest(target)
    if 'nanobody_opt_v2/run_live_producer.py' not in files or 'pixi.lock' not in files:
        raise ValueError('release is missing the shipping producer or lockfile')
    manifest = {'schema': 1, 'git_sha': revision, 'pixi_lock_hash': files['pixi.lock'],
                'files': files, 'runtime_files': [p for p in files if runtime_file(p)]}
    (destination / 'build_stamp.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(root: Path, manifest: dict, *, expected_sha: str,
           runtime_only: bool = False, require_tests: bool = True) -> None:
    if not re.fullmatch('[0-9a-f]{40}', expected_sha) or manifest.get('git_sha') != expected_sha:
        raise ValueError('release does not match the tested revision')
    files = manifest.get('files', {})
    if (manifest.get('schema') != 1 or not files
            or manifest.get('pixi_lock_hash') != files.get('pixi.lock')):
        raise ValueError('invalid release manifest')
    expected_runtime = sorted(p for p in files if runtime_file(p))
    if sorted(manifest.get('runtime_files', [])) != expected_runtime:
        raise ValueError('release runtime inventory is incomplete')
    names = expected_runtime if runtime_only else sorted(files)
    for name in names:
        path = safe_path(root, name)
        if not path.is_file() or digest(path) != files[name]:
            raise ValueError(f'release byte mismatch: {name}')
    if require_tests:
        result = manifest.get('validation', {})
        if (result.get('passed') is not True or result.get('tests', 0) < 1
                or result.get('failures') != 0 or result.get('errors') != 0
                or result.get('skipped') != 0):
            raise ValueError('release has no complete passing regression evidence')
        from scripts.shipping_checks import TEST_FILES
        if result.get('test_files') != list(TEST_FILES):
            raise ValueError('release omitted required regression suites')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'verify'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--expected-sha')
    parser.add_argument('--runtime-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'stage':
            if args.destination is None:
                parser.error('stage requires --destination')
            manifest = stage(args.root, args.destination)
        else:
            if args.manifest is None or args.expected_sha is None:
                parser.error('verify requires --manifest and --expected-sha')
            manifest = json.loads(args.manifest.read_text())
            verify(args.root, manifest, expected_sha=args.expected_sha, runtime_only=args.runtime_only)
        print(json.dumps({'status': 'ok', 'revision': manifest['git_sha'], 'files': len(manifest['files'])}))
        return 0
    except (OSError, ValueError) as error:
        print(f'RELEASE REFUSED: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
