#!/usr/bin/env python3
"""Deploy CI's passing package; keep the SSH credential inside this process."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_manifest import safe_path, verify
from scripts.promotion_gate import verify as qualify, digest, read, require


def deployment_plan(manifest_path, revision, host, activate, bundle=None, target=None):
    """Resolve the only permitted path before accessing any SSH credential."""
    if not activate:
        return 'ansible/stage_nanobody_release.yml', {}
    require(bundle is not None and bool(target), 'promotion_evidence_required')
    bundle = Path(bundle).resolve()
    qualify(bundle, candidate_sha=revision, host=host, target=target)
    plan = read(bundle / 'promotion.json')
    receipt = read(safe_path(bundle, plan['shadow_receipt']['path']))
    require(receipt['runs']['candidate']['manifest_sha256'] == digest(manifest_path),
            'candidate_manifest_mismatch')
    # Copy only checked, explicitly referenced documents, never an entire
    # operator-supplied directory that could contain credentials or symlinks.
    paths = ['promotion.json'] + [plan[k]['path'] for k in (
        'fixture', 'baseline_report', 'candidate_report', 'shadow_receipt',
        'rollback_receipt', 'rollback_report', 'pi_receipt', 'state_snapshot')]
    require(len(paths) == len(set(paths)), 'duplicate_evidence_paths')
    for name in paths:
        require(safe_path(bundle, name).is_file(), 'missing_evidence_document')
    return 'ansible/promote_nanobody_release.yml', {
        'sn68_promotion_dir': str(bundle), 'sn68_promotion_files': paths,
        'sn68_deploy_host': host, 'sn68_target': target}


def unpack_release(archive: Path, destination: Path, manifest: dict, revision: str) -> None:
    with tarfile.open(archive, 'r:gz') as package:
        expected = set(manifest['files']) | {'build_stamp.json'}
        members = package.getmembers()
        if {m.name for m in members} != expected or len(members) != len(expected):
            raise ValueError('release archive inventory differs from its manifest')
        for member in members:
            safe_path(destination, member.name)
            if not member.isfile():
                raise ValueError('release archives may contain only regular files')
        package.extractall(destination, filter='data')
    if json.loads((destination / 'build_stamp.json').read_text()) != manifest:
        raise ValueError('release archive has a different test manifest')
    verify(destination, manifest, expected_sha=revision)


def main() -> int:
    phase = 'release verification'
    try:
        release = Path(os.environ['SN68_RELEASE_DIR'])
        revision = os.environ['SN68_TESTED_SHA']
        host, port = os.environ['SN68_DEPLOY_HOST'], int(os.environ['SN68_DEPLOY_PORT'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.:-]*', host) or not 1 <= port <= 65535:
            raise ValueError('invalid deployment endpoint')
        manifest = json.loads((release / 'build_stamp.json').read_text())
        verify(ROOT, manifest, expected_sha=revision)
        activate = os.environ.get('SN68_DEPLOY_ACTIVATE') == 'true'
        phase = 'promotion qualification' if activate else 'staging verification'
        playbook, promotion = deployment_plan(release / 'build_stamp.json', revision, host,
            activate, os.environ.get('SN68_PROMOTION_DIR'), os.environ.get('SN68_TARGET'))
        # Ansible needs a key file. It exists only in this private temporary
        # directory and is removed on success, exception, or subprocess failure.
        with tempfile.TemporaryDirectory(prefix='sn68-deploy-') as temporary:
            scratch = Path(temporary)
            unpack_release(release / 'sn68-release.tar.gz', scratch / 'release', manifest, revision)
            phase = 'SSH setup'
            key = scratch / 'identity'
            descriptor = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w') as handle:
                handle.write(os.environ['SSH_PRIVATE_KEY'])
                handle.write('\n')
            host_keys = subprocess.run(['ssh-keyscan', '-p', str(port), '-H', host],
                                       capture_output=True, timeout=30)
            if host_keys.returncode or not host_keys.stdout.strip():
                raise ValueError('SSH host key discovery failed')
            known_hosts = scratch / 'known_hosts'
            known_hosts.write_bytes(host_keys.stdout)
            inventory = scratch / 'inventory.json'
            inventory.write_text(json.dumps({'all': {'children': {'nanobody_producer': {
                'hosts': {'tested-producer': {'ansible_host': host, 'ansible_port': port}}}}}}))
            variables = scratch / 'variables.json'
            variables.write_text(json.dumps({
                'sn68_release_archive': str((release / 'sn68-release.tar.gz').resolve()),
                'sn68_release_manifest': str((release / 'build_stamp.json').resolve()),
                'sn68_tested_sha': revision,
                **promotion,
            }))
            phase = 'Ansible deployment'
            environment = os.environ.copy()
            environment.pop('SSH_PRIVATE_KEY', None)
            environment['ANSIBLE_HOST_KEY_CHECKING'] = 'True'
            result = subprocess.run([
                'ansible-playbook', '-i', str(inventory), playbook,
                '--private-key', str(key), '-e', '@' + str(variables),
                '--ssh-common-args', f'-o StrictHostKeyChecking=yes -o UserKnownHostsFile={known_hosts}',
            ], cwd=ROOT, env=environment, capture_output=True, timeout=7000)
            # Ansible dependencies can emit authentication diagnostics. Only
            # the phase and exit status are allowed into CI output.
            print(json.dumps({'phase': phase, 'exit': result.returncode, 'revision': revision}))
            return result.returncode
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        print(json.dumps({'status': 'deployment_refused', 'phase': phase}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
