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
    'NOVA-nanobody-filter', 'data/structures/8AOK.cif',
)
RUNTIME_DIRS = ('nanobody_opt', 'nanobody_opt_v2', 'nanobody_dash', 'tools', 'neurons',
                'compound_miner', 'utils', 'config', 'combinatorial_db',
                'tests', 'scripts', 'deploy', 'NOVA-nanobody-filter', 'data')
LIVE_CONFIG = {'config/config.yaml', 'config/boltzgen_config.yaml'}
CONTROL_FILES = frozenset({
    'deploy_gate.sh', 'scripts/release_manifest.py', 'scripts/shipping_checks.py',
    'scripts/check_launch_contract.py', 'scripts/deploy_tested_release.py',
    'scripts/promotion_gate.py', 'scripts/promote_release.py',
    'scripts/release_state.py', 'scripts/shadow_release.py',
    'scripts/pi_shadow_receipt.py', 'scripts/export_promotion_evidence.py',
    'scripts/control_release.py', 'scripts/replay_handoff.py',
    'scripts/pi_control_agent.py',
})


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


def stage(root: Path, destination: Path, *, profile: str = 'application') -> dict:
    if profile not in {'application', 'release-controls'}:
        raise ValueError('unknown deployment profile')
    revision = git(root, 'rev-parse', 'HEAD').strip()
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('release revision must be a full commit SHA')
    if git(root, 'status', '--porcelain', '--untracked-files=all', '--', *SOURCE_PATHS).strip():
        raise ValueError('commit all release-source changes before building')
    names = git(root, 'ls-files', '-z', '--', *SOURCE_PATHS).split('\0')
    submodules = {}
    if 'NOVA-nanobody-filter' in names:
        name = 'NOVA-nanobody-filter'
        entry = git(root, 'ls-tree', 'HEAD', '--', name).split()
        if len(entry) != 4 or entry[:2] != ['160000', 'commit']:
            raise ValueError('filter backend must be a pinned submodule')
        source = safe_path(root, name)
        if (git(source, 'rev-parse', 'HEAD').strip() != entry[2]
                or git(source, 'status', '--porcelain', '--untracked-files=normal').strip()):
            raise ValueError('filter submodule differs from its committed pin')
        submodules[name] = entry[2]
        names.remove(name)
        names.extend(name + '/' + p for p in git(source, 'ls-files', '-z').split('\0') if p)
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
    if submodules:
        manifest['submodules'] = submodules
    if profile == 'release-controls':
        if not CONTROL_FILES <= set(files):
            raise ValueError('release control package is incomplete')
        manifest.update(deployment_profile=profile, runtime_files=sorted(CONTROL_FILES))
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
    profile = manifest.get('deployment_profile', 'application')
    if profile not in {'application', 'release-controls'}:
        raise ValueError('unknown deployment profile')
    if profile == 'release-controls' and not CONTROL_FILES <= set(files):
        raise ValueError('release control package is incomplete')
    expected_runtime = (sorted(CONTROL_FILES) if profile == 'release-controls' else
                        sorted(p for p in files if runtime_file(p)))
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
