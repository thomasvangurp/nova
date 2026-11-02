#!/usr/bin/env python3
"""Pin and verify the PR100 native-reference runtime and its CUDA sidecar.

The native-reference environment is the registered, sealed runtime behind
``nanobody_opt.pr100_native_reference``. It is built off-box and seeded onto each
rental. This script is the repo-owned verification for that seed: it reads the
qualified CUDA registration from ``config/pr100_native_reference.json`` and
refuses a seed that does not match the sealed manifest, the pinned OpenMM CUDA
core (``loader.core_sha256``), or the pinned CUDA sidecar.

It installs nothing, contacts no host, and restarts nothing. Exit 0 only when the
runtime, its OpenMM core, and the CUDA sidecar are byte-identical to the pins.

The previous failure this guards against: a seed verified
``{"verified_local_seed_files": 0}`` and was still accepted, leaving a rental with
no ``OpenMM.libs/lib/libOpenMM.so`` and thousands of missing sealed files, which
only surfaced as ``cuda_loader_identity_mismatch`` at CUDA worker startup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / 'config/pr100_native_reference.json'
DEFAULT_REFERENCE_ROOT = Path('/root/pr100-validator-reference-20261030')
DEFAULT_ENV_NAME = 'ba353b3-openmm850-biopython185-numpy202'
DEFAULT_SIDECAR = 'cuda850-sidecar-v1'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def qualified_cuda_row(registry: dict) -> dict:
    for row in registry.get('registrations', []):
        plugin = row.get('cuda_plugin')
        policy = row.get('policy', {}).get('final_refinement', {})
        if (plugin and plugin.get('loader') and row.get('qualification_status') == 'qualified'
                and policy.get('platform') == 'CUDA'):
            return row
    raise ValueError('no qualified CUDA registration in the registry')


def fail(message: str) -> int:
    print('NO_GO: ' + message, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, default=DEFAULT_REFERENCE_ROOT,
                        help='parent of the env directory and the sidecar directory')
    parser.add_argument('--env-name', default=DEFAULT_ENV_NAME)
    parser.add_argument('--sidecar', default=DEFAULT_SIDECAR)
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())
    row = qualified_cuda_row(registry)
    plugin = row['cuda_plugin']
    core_sha = plugin['loader']['core_sha256']
    env_root = args.reference_root / args.env_name
    env = env_root / 'env'

    manifest_path = env_root / 'runtime-manifest.json'
    if not manifest_path.is_file():
        return fail(f'{manifest_path} is absent; the native-reference env was never seeded')
    if sha(manifest_path) != row['environment_manifest_sha256']:
        return fail('runtime-manifest.json does not match the sealed environment_manifest_sha256')

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files', {})
    if not files:
        return fail('runtime-manifest.json lists no files')

    verified = 0
    missing = 0
    for relative, expected in files.items():
        target = env_root / relative
        if not target.is_file() or sha(target) != expected:
            missing += 1
            if missing <= 10:
                print(f'{target} is missing or changed', file=sys.stderr)
        else:
            verified += 1

    broken_links = 0
    for relative, target_value in manifest.get('symlinks', {}).items():
        link = env_root / relative
        if not link.is_symlink() or os.readlink(link) != target_value:
            broken_links += 1

    core = env / 'lib/python3.12/site-packages/OpenMM.libs/lib/libOpenMM.so'
    core_ok = core.is_file() and sha(core) == core_sha

    sidecar_root = args.reference_root / args.sidecar
    sidecar_manifest = sidecar_root / 'plugin-manifest.json'
    sidecar_ok = sidecar_manifest.is_file() and sha(sidecar_manifest) == plugin['manifest_sha256']
    if sidecar_ok:
        sidecar_json = json.loads(sidecar_manifest.read_text())
        for relative, expected in sidecar_json.get('files', {}).items():
            target = sidecar_root / relative
            if not target.is_file() or sha(target) != expected:
                sidecar_ok = False
                break
        plugin_dir = sidecar_root / sidecar_json.get('plugin_directory', '')
        if not (plugin_dir / 'libOpenMMCUDA.so').is_file():
            sidecar_ok = False

    report = {
        'verified_local_seed_files': verified,
        'manifest_file_count': len(files),
        'missing_or_changed_files': missing,
        'broken_symlinks': broken_links,
        'openmm_cuda_core_sha256': core_sha,
        'openmm_cuda_core_ok': core_ok,
        'sidecar_manifest_sha256': plugin['manifest_sha256'],
        'sidecar_wheel_sha256': plugin['wheel_sha256'],
        'sidecar_ok': sidecar_ok,
        'status': 'OK' if (missing == 0 and broken_links == 0 and core_ok and sidecar_ok) else 'NO_GO',
    }
    print(json.dumps(report, indent=2))

    if missing:
        return fail(f'{missing} sealed files are missing or changed (verified {verified} of {len(files)})')
    if broken_links:
        return fail(f'{broken_links} sealed symlinks are missing or retargeted')
    if not core_ok:
        return fail(f'OpenMM CUDA core {core} does not match pinned core_sha256 {core_sha}')
    if not sidecar_ok:
        return fail(f'CUDA sidecar {sidecar_root} does not match pinned manifest {plugin["manifest_sha256"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
