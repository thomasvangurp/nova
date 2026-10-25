#!/usr/bin/env python3
"""Run an approved recorded-workload driver in three credential-free containers.

The same driver/fixtures run baseline, candidate, then baseline against the
candidate's resulting state. A driver must execute the production pipeline and
emit the report contract in docs/RELEASE_PROMOTION.md. There is no shell command,
network, GPU, live runtime, wallet, or Docker socket inside these containers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.promotion_gate import digest, read, require
from scripts.release_manifest import safe_path, verify


def inventory(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative)
        require(not any(x in {'.ssh', '.aws', '.bittensor', 'wallets', 'secrets', 'keychains'}
                        for x in path.relative_to(root).parts), 'credentials_in_trial_input')
        if path.is_file():
            result[relative] = digest(path)
    return result


def fingerprint(root):
    import hashlib
    return hashlib.sha256(json.dumps(inventory(root), sort_keys=True).encode()).hexdigest()


def validate_release(root):
    manifest = read(root / 'build_stamp.json')
    verify(root, manifest, expected_sha=manifest['git_sha'], require_tests=False)
    validation = manifest['validation']
    require(validation['passed'] is True and validation['tests'] > 0 and
            all(validation[k] == 0 for k in ['failures', 'errors', 'skipped']), 'untested_trial_release')
    return manifest


def container_args(*, name, image, release, harness, fixture, output, driver, resume=None):
    command = ['docker', 'create', '--name', name, '--pull=never', '--network=none',
        '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
        '--user=65534:65534', '--pids-limit=256', '--cpus=4', '--memory=8g',
        '--tmpfs=/tmp:rw,noexec,nosuid,size=1g', '--workdir=/output',
        '--env=PYTHONPATH=/release', '--env=PYTHONDONTWRITEBYTECODE=1',
        '--mount', f'type=bind,src={release},dst=/release,readonly',
        '--mount', f'type=bind,src={harness},dst=/harness,readonly',
        '--mount', f'type=bind,src={fixture},dst=/fixtures,readonly',
        '--mount', f'type=bind,src={output},dst=/output']
    if resume:
        command += ['--mount', f'type=bind,src={resume},dst=/resume,readonly']
    command += ['--entrypoint=python3', image, '/harness/' + driver,
                '--release-root', '/release', '--fixtures', '/fixtures',
                '--output', '/output/report.json', '--state-output', '/output/state']
    if resume:
        command += ['--resume-state', '/resume']
    return command


def run(*, baseline, candidate, fixtures, output, driver, image, timeout=3600):
    baseline, candidate, fixtures, output = (Path(p).resolve() for p in
                                             [baseline, candidate, fixtures, output])
    require(re.fullmatch(r'[^\s,]+@sha256:[0-9a-f]{64}', image) is not None, 'image_must_be_pinned')
    require(30 <= timeout <= 7200, 'invalid_trial_timeout')
    for path in [baseline, candidate, fixtures, output]:
        require(',' not in str(path), 'invalid_mount_path')
    require(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output)
                for p in [baseline, candidate, fixtures]), 'trial_output_overlap')
    old, new = validate_release(baseline), validate_release(candidate)
    require(driver in new['files'] and safe_path(candidate, driver).suffix == '.py', 'untracked_driver')
    originals = {str(p): inventory(p) for p in [baseline, candidate, fixtures]}
    fixture_hash = digest(fixtures / 'fixture.json')
    output.mkdir(parents=True)
    receipt = {'schema': 'sn68-isolated-trial-v1', 'fixture_sha256': fixture_hash,
        'image_digest': image.rsplit(':', 1)[1], 'driver_sha256': new['files'][driver],
        'network': 'none', 'root_read_only': True, 'host_credentials_mounted': False,
        'capabilities': [], 'live_submission_access': False, 'runs': {}}
    for kind, release, manifest in [('baseline', baseline, old), ('candidate', candidate, new),
                                     ('rollback', baseline, old)]:
        destination = output / kind
        destination.mkdir(mode=0o777)
        destination.chmod(0o777)  # Container's unprivileged UID can write only this scratch directory.
        resume = output / 'candidate/state' if kind == 'rollback' else None
        if resume:
            require(resume.is_dir() and bool(inventory(resume)), 'candidate_did_not_emit_state')
        name = 'sn68-shadow-' + uuid.uuid4().hex
        started = time.time()
        try:
            created = subprocess.run(container_args(name=name, image=image, release=release,
                harness=candidate, fixture=fixtures, output=destination, driver=driver, resume=resume),
                capture_output=True, timeout=30)
            require(created.returncode == 0, 'shadow_container_creation_failed')
            executed = subprocess.run(['docker', 'start', '--attach', name],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
            require(executed.returncode == 0, 'shadow_execution_failed')
            report_path = destination / 'report.json'
            report = read(report_path)
            require(report['revision'] == manifest['git_sha'] and report['fixture_sha256'] == fixture_hash,
                    'shadow_output_scope')
            receipt['runs'][kind] = {'revision': manifest['git_sha'], 'exit_code': 0,
                'manifest_sha256': digest(release / 'build_stamp.json'), 'report_sha256': digest(report_path),
                'started_at': started, 'finished_at': time.time()}
            if resume:
                receipt['runs'][kind]['resume_state_sha256'] = fingerprint(resume)
        finally:
            # Timeout includes descendants: remove the container, not only the docker client.
            subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
        require(all(inventory(Path(p)) == before for p, before in originals.items()), 'trial_mutated_input')
    (output / 'shadow_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'candidate', 'fixtures', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--driver', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    try:
        result = run(**vars(args))
        print(json.dumps({'status': 'trial_complete', 'runs': list(result['runs'])}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print(json.dumps({'status': 'trial_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
