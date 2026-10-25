#!/usr/bin/env python3
"""Qualify and install deployment controls without replacing application code.

An observed application snapshot is explicitly not a claim that the application
passed this control release's CI. The installed control payload is a fixed,
separately tested allowlist. Both sides of installation verify application and
launcher bytes; no producer, scorer, publisher or Pi service is restarted.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.promotion_gate import digest, read, require, referenced, sha
from scripts.release_manifest import CONTROL_FILES, safe_path, verify

FIXTURES = 'tests/fixtures/production_handoff'
REQUIRED_APPLICATION = frozenset({
    'nanobody_opt_v2/run_live_producer.py', 'nanobody_opt_v2/sequential_exact_race.py',
    'nanobody_opt_v2/select_elite_parents.py', 'nanobody_opt/rank_exact.py',
    'nanobody_opt/score_boltzgen_fast_adapter.py', 'nanobody_opt/scorer_daemon.py',
    'tools/publish_current_nanobody.py', 'compound_miner/nanobody_provider.py',
    'pixi.toml', 'pixi.lock',
})
LAUNCH_BINDINGS = ('artifacts/producer_argv.txt', 'producer_relaunch.sh')
GUARD_BEGIN = '# BEGIN verified release controls\n'
GUARD_END = '# END verified release controls\n'


def guard_launcher(raw, revision):
    """Insert one checked gate; preserve every original command/argument byte."""
    require(sha(revision, 40), 'invalid_guard_revision')
    text = raw.decode('utf-8')
    head, separator, body = text.partition('\n')
    require(separator and head in {'#!/usr/bin/env bash', '#!/bin/bash', '#!/bin/sh'},
            'unsupported_launcher_shell')
    if GUARD_BEGIN in body or GUARD_END in body:
        require(body.startswith(GUARD_BEGIN) and body.count(GUARD_BEGIN) == 1
                and body.count(GUARD_END) == 1, 'unexpected_existing_launch_guard')
        previous, body = body[len(GUARD_BEGIN):].split(GUARD_END, 1)
        require(re.fullmatch(r'SN68_TESTED_SHA=[0-9a-f]{40}'
                             + re.escape(' bash /root/deploy_gate.sh || exit $?\n'), previous),
                'unexpected_existing_launch_guard')
    gate = f'SN68_TESTED_SHA={revision} bash /root/deploy_gate.sh || exit $?\n'
    return (head + separator + GUARD_BEGIN + gate + GUARD_END + body).encode('utf-8')


def application_source(name):
    parts = Path(name).parts
    return (name in {'pixi.toml', 'pixi.lock'} or
            (parts[0] in {'nanobody_opt', 'nanobody_opt_v2', 'tools', 'neurons',
                          'compound_miner', 'utils', 'config', 'combinatorial_db'}
             and 'tests' not in parts and Path(name).suffix in {'.py', '.sh'}))


def save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.next-' + uuid.uuid4().hex)
    with temporary.open('x') as handle:
        handle.write(json.dumps(payload, indent=2) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def capture_application(root, source_manifest, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    source = read(source_manifest)
    require(sha(source['git_sha'], 40) and not output.exists(), 'invalid_application_capture')
    files, absent = {}, []
    for name, expected in source['files'].items():
        if not application_source(name):
            continue
        path = safe_path(root, name)
        if not path.is_file():
            absent.append(name)
            continue
        actual = digest(path)
        require(actual == expected, 'observed_application_changed')
        files[name] = actual
    require(REQUIRED_APPLICATION <= set(files), 'missing_application_entrypoint')
    bindings = {name: digest(safe_path(root, name)) for name in LAUNCH_BINDINGS}
    # Record the running launch contract, not an old deployment template.
    args = (root / LAUNCH_BINDINGS[0]).read_text().splitlines()
    require(args[:3] == ['/root/.pixi/envs/bgen/bin/python', '-m', 'nanobody_opt_v2.run_live_producer'],
            'unsupported_running_launcher')
    for flag, expected in [('--epoch-candidate-budget', '4800'), ('--epoch-proposal-budget', '8000')]:
        require(args.count(flag) == 1 and args[args.index(flag) + 1] == expected, 'running_budget_contract')
    snapshot = {'schema': 'sn68-observed-application-v1', 'source_revision': source['git_sha'],
        'source_manifest_sha256': digest(source_manifest), 'captured_at': time.time(),
        'files': files, 'absent_nonrequired_files': sorted(absent), 'launch_bindings': bindings,
        'application_ci_attested': False, 'new_evaluation_cap': 4800, 'proposal_cap': 8000}
    save(output, snapshot)
    return snapshot


def verify_application(root, snapshot):
    root = Path(root)
    require(snapshot['schema'] == 'sn68-observed-application-v1'
            and sha(snapshot['source_revision'], 40)
            and REQUIRED_APPLICATION <= set(snapshot['files']), 'invalid_observed_application')
    require(set(snapshot['launch_bindings']) == set(LAUNCH_BINDINGS), 'missing_launch_bindings')
    require(snapshot['new_evaluation_cap'] == 4800 and snapshot['proposal_cap'] == 8000,
            'observed_budget_contract')
    for name, expected in {**snapshot['files'], **snapshot['launch_bindings']}.items():
        require(name in LAUNCH_BINDINGS or application_source(name), 'unexpected_application_path')
        require(sha(expected) and digest(safe_path(root, name)) == expected, 'application_or_launcher_drift')
    for name in snapshot['absent_nonrequired_files']:
        require(application_source(name) and not safe_path(root, name).exists(), 'application_inventory_drift')


def verify_evidence(stage, evidence):
    stage, evidence = Path(stage), Path(evidence)
    manifest = read(stage / 'build_stamp.json')
    verify(stage, manifest, expected_sha=manifest['git_sha'])
    require(manifest.get('deployment_profile') == 'release-controls', 'control_profile_required')
    qualification = read(evidence / 'qualification.json')
    require(qualification['schema'] == 'sn68-control-qualification-v1'
            and qualification['revision'] == manifest['git_sha']
            and qualification['manifest_sha256'] == digest(stage / 'build_stamp.json')
            and qualification['driver_sha256'] == manifest['files']['scripts/replay_handoff.py'],
            'control_qualification_scope')
    require(0 <= time.time() - qualification['created_at'] <= 86400,
            'expired_control_qualification')
    isolated = qualification['isolation']
    require(isolated['network'] == 'none' and isolated['read_only'] is True
            and isolated['capabilities'] == [] and isolated['credentials_mounted'] is False
            and sha(isolated['image_digest']) and isolated['exit_code'] == 0, 'unsafe_control_trial')
    fixture = read(stage / FIXTURES / 'fixture.json')
    report = referenced(evidence, qualification['report'])
    require(report['schema'] == 'sn68-control-replay-v1' and report['status'] == 'passed'
            and report['fixture_sha256'] == digest(stage / FIXTURES / 'fixture.json')
            and report['consumer_sha256'] == fixture['consumer']['sha256']
            and report['installed_consumer_sha256'] == fixture['installed_consumer_sha256']
            and report['submission_attempts'] == 0 and report['scoring_calls'] == 0,
            'recorded_replay_scope')
    require(len({c['epoch'] for c in report['cases']}) >= 3
            and len(report['cases']) == len(fixture['cases']), 'incomplete_recorded_replay')
    for actual, expected in zip(report['cases'], fixture['cases']):
        require(actual['epoch'] == expected['epoch'] and actual['entries'] == len(expected['uids'])
                and actual['allocation_sha256'] == expected['expected_allocation_sha256']
                and actual['artifact_sha256'] == expected['artifact']['sha256']
                and actual['wrong_epoch_refused'] is True, 'recorded_replay_changed')
    return qualification, report


def qualify(stage, output, image='python:3.12-slim'):
    stage, output = Path(stage).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(stage), 'qualification_output_exists')
    manifest = read(stage / 'build_stamp.json')
    verify(stage, manifest, expected_sha=manifest['git_sha'])
    require(manifest.get('deployment_profile') == 'release-controls', 'control_profile_required')
    # Pull only the explicit official runtime; resolve it to an immutable digest
    # before creating the isolated container. Never pass host environment values.
    require(image == 'python:3.12-slim' or re.fullmatch(r'python@sha256:[0-9a-f]{64}', image),
            'unsupported_control_image')
    result = subprocess.run(['docker', 'pull', image], capture_output=True, timeout=180)
    require(result.returncode == 0, 'control_image_unavailable')
    result = subprocess.run(['docker', 'image', 'inspect', image, '--format', '{{json .RepoDigests}}'],
                            capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, 'control_image_unavailable')
    pinned = next((x for x in json.loads(result.stdout) if re.fullmatch(r'python@sha256:[0-9a-f]{64}', x)), None)
    require(pinned is not None, 'control_image_not_pinned')
    from scripts.shadow_release import inventory
    inventory(stage)  # rejects credentials/symlinks before any mount
    output.mkdir(parents=True, mode=0o700)
    name = 'sn68-control-replay-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='sn68-control-output-') as scratch:
        scratch_path = Path(scratch); scratch_path.chmod(0o777)
        command = ['docker', 'create', '--name', name, '--pull=never', '--network=none', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user=65534:65534',
            '--pids-limit=128', '--memory=512m', '--cpus=1',
            '--env=PYTHONDONTWRITEBYTECODE=1', '--env=PYTHONPATH=/release',
            '--mount', f'type=bind,src={stage},dst=/release,readonly',
            '--mount', f'type=bind,src={scratch_path},dst=/output',
            '--entrypoint=python3', pinned, '/release/scripts/replay_handoff.py',
            '--fixtures', '/release/' + FIXTURES, '--output', '/output/report.json']
        try:
            result = subprocess.run(command, capture_output=True, timeout=30)
            require(result.returncode == 0, 'control_container_failed')
            result = subprocess.run(['docker', 'start', '--attach', name],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
            require(result.returncode == 0, 'control_replay_failed')
            shutil.copyfile(scratch_path / 'report.json', output / 'report.json')
        finally:
            subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=30)
    qualification = {'schema': 'sn68-control-qualification-v1', 'revision': manifest['git_sha'],
        'manifest_sha256': digest(stage / 'build_stamp.json'),
        'driver_sha256': manifest['files']['scripts/replay_handoff.py'], 'created_at': time.time(),
        'ci_run_id': os.environ.get('GITHUB_RUN_ID'),
        'isolation': {'network': 'none', 'read_only': True, 'capabilities': [],
                      'credentials_mounted': False, 'image_digest': pinned.rsplit(':', 1)[1], 'exit_code': 0},
        'report': {'path': 'report.json', 'sha256': digest(output / 'report.json')}}
    save(output / 'qualification.json', qualification)
    verify_evidence(stage, output)
    return qualification


def install_controls(stage, evidence, snapshot_path, pi_receipt, root):
    stage, root = Path(stage).resolve(), Path(root).resolve()
    qualification, report = verify_evidence(stage, evidence)
    manifest = read(stage / 'build_stamp.json')
    snapshot = read(snapshot_path)
    verify_application(root, snapshot)
    pi = read(pi_receipt)
    clock = read(root / 'artifacts/chain_state.json')
    request = read(root / 'artifacts/control_qualification_request.json')
    require(0 <= time.time() - clock['updated_at_unix'] <= 90, 'stale_live_clock')
    require(pi['schema'] == 'sn68-pi-shadow-v1' and pi['host'] == 'pi'
            and pi['candidate_sha'] == manifest['git_sha']
            and pi['target'] == clock['nanobody_target'] and pi['epoch'] == clock['epoch']
            and pi['read_only'] is True and pi['submission_attempts'] == 0
            and pi['accepted_entries'] >= 1 and pi['rejected_entries'] == 0
            and pi['consumer_sha256'] == report['installed_consumer_sha256']
            and pi['request_nonce'] == request['nonce']
            and request['candidate_sha'] == manifest['git_sha']
            and request['schema'] == 'sn68-control-pi-request-v1'
            and request['epoch'] == clock['epoch'] and request['target'] == clock['nanobody_target']
            and request['artifact_sha256'] == pi['artifact_sha256']
            and request['created_at'] <= pi['verified_at']
            and 0 <= time.time() - request['publication_mtime'] <= 900
            and request['expires_at'] > time.time()
            and 0 <= time.time() - pi['verified_at'] <= 900, 'current_pi_receipt_required')
    require(sha(request['nonce'], 32), 'invalid_pi_request')
    artifact = stage / 'pi_trials' / (request['nonce'] + '.json')
    require(digest(artifact) == pi['artifact_sha256'], 'pi_checked_different_live_artifact')
    payload = read(artifact)
    require(payload['epoch'] == clock['epoch'] and len(payload['submissions']) == pi['accepted_entries'],
            'pi_current_fleet_shortfall')
    attempt = stage / ('control-install-' + uuid.uuid4().hex)
    attempt.mkdir(mode=0o700)
    before = {}
    owned = sorted(CONTROL_FILES) + ['build_stamp.json', 'artifacts/release_controls.json',
                                    'artifacts/observed_application.json', 'producer_relaunch.sh']
    for name in owned:
        current = safe_path(root, name)
        before[name] = current.is_file()
        if before[name]:
            saved = safe_path(attempt, name); saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, saved)
    try:
        for name in sorted(CONTROL_FILES):
            destination = safe_path(root, name); destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_name(destination.name + '.control-next')
            shutil.copy2(safe_path(stage, name), pending); pending.replace(destination)
        # This stamp attests ONLY the tested control payload. Its role prevents
        # application deployment or claims that the observed application passed CI.
        shutil.copy2(stage / 'build_stamp.json', root / 'build_stamp.control-next')
        (root / 'build_stamp.control-next').replace(root / 'build_stamp.json')
        launcher = root / 'producer_relaunch.sh'
        guarded = launcher.with_name(launcher.name + '.control-next')
        guarded.write_bytes(guard_launcher(launcher.read_bytes(), manifest['git_sha']))
        shutil.copymode(launcher, guarded); guarded.replace(launcher)
        installed_snapshot = copy.deepcopy(snapshot)
        installed_snapshot['launch_bindings']['producer_relaunch.sh'] = digest(launcher)
        installed_snapshot['launcher_guard_revision'] = manifest['git_sha']
        save(root / 'artifacts/observed_application.json', installed_snapshot)
        verify(root, manifest, expected_sha=manifest['git_sha'], runtime_only=True)
        verify_application(root, installed_snapshot)
        receipt = {'schema': 'sn68-control-install-v1', 'status': 'active',
            'revision': manifest['git_sha'], 'manifest_sha256': digest(root / 'build_stamp.json'),
            'qualification_sha256': digest(Path(evidence) / 'qualification.json'),
            'application_snapshot_sha256': digest(root / 'artifacts/observed_application.json'),
            'application_source_revision': snapshot['source_revision'], 'application_ci_attested': False,
            'application_files_unchanged': True, 'services_restarted': False,
            'launch_arguments_unchanged': True, 'launcher_guard_installed': True,
            'previous_launcher_sha256': snapshot['launch_bindings']['producer_relaunch.sh'],
            'pi_artifact_sha256': pi['artifact_sha256'], 'pi_entries': pi['accepted_entries'],
            'activation_epoch': clock['epoch'], 'activated_at': time.time()}
        save(root / 'artifacts/release_controls.json', receipt)
        return receipt
    except BaseException:
        from scripts.promote_release import ignore_cancellation
        with ignore_cancellation():
            for name in owned:
                path = safe_path(root, name)
                if before[name]:
                    shutil.copy2(safe_path(attempt, name), path)
                else:
                    path.unlink(missing_ok=True)
        raise


def request_pi(stage, evidence, root):
    stage, root = Path(stage).resolve(), Path(root).resolve()
    qualification, report = verify_evidence(stage, evidence)
    require(stage == root / 'releases' / qualification['revision'], 'invalid_pi_trial_stage')
    clock = read(root / 'artifacts/chain_state.json')
    require(0 <= time.time() - clock['updated_at_unix'] <= 90, 'stale_live_clock')
    original = root / 'artifacts/nanobody_result.json'
    with original.open('rb') as handle:
        raw = handle.read()
        publication_mtime = os.fstat(handle.fileno()).st_mtime
    require(0 <= time.time() - publication_mtime <= 900, 'stale_original_publication')
    artifact = json.loads(raw)
    fixture = read(stage / FIXTURES / 'fixture.json')
    require(artifact['epoch'] == clock['epoch'] and artifact['target'] == clock['nanobody_target']
            and fixture['target'] == artifact['target'], 'pi_trial_scope')
    nonce = uuid.uuid4().hex
    trials = stage / 'pi_trials'; trials.mkdir(exist_ok=True, mode=0o700)
    trial = trials / (nonce + '.json')
    trial.write_bytes(raw); trial.chmod(0o600)
    request = {'schema': 'sn68-control-pi-request-v1', 'candidate_sha': qualification['revision'],
        'nonce': nonce, 'epoch': clock['epoch'], 'target': artifact['target'],
        'publication_mtime': publication_mtime,
        'artifact_sha256': hashlib.sha256(raw).hexdigest(), 'created_at': time.time(),
        'expires_at': time.time() + 900}
    save(root / 'artifacts/control_qualification_request.json', request)
    return request


def verify_installed(root, revision):
    root = Path(root)
    manifest = read(root / 'build_stamp.json')
    require(manifest.get('deployment_profile') == 'release-controls', 'control_profile_required')
    verify(root, manifest, expected_sha=revision, runtime_only=True)
    receipt = read(root / 'artifacts/release_controls.json')
    snapshot_path = root / 'artifacts/observed_application.json'
    require(receipt['status'] == 'active' and receipt['revision'] == revision
            and receipt['manifest_sha256'] == digest(root / 'build_stamp.json')
            and receipt['application_snapshot_sha256'] == digest(snapshot_path), 'control_installation_drift')
    snapshot = read(snapshot_path)
    verify_application(root, snapshot)
    require(snapshot.get('launcher_guard_revision') == revision
            and guard_launcher((root / 'producer_relaunch.sh').read_bytes(), revision)
                == (root / 'producer_relaunch.sh').read_bytes(), 'restart_guard_missing')
    return {'status': 'verified', 'revision': revision, 'deployment_profile': 'release-controls'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['capture', 'qualify', 'request-pi', 'install', 'verify'])
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--root', type=Path, default=Path('/root'))
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--pi-receipt', type=Path)
    parser.add_argument('--revision')
    args = parser.parse_args()
    try:
        if args.action == 'capture':
            result = capture_application(args.root, args.source_manifest, args.output)
        elif args.action == 'qualify':
            result = qualify(args.stage, args.output)
        elif args.action == 'verify':
            result = verify_installed(args.root, args.revision)
        elif args.action == 'request-pi':
            result = request_pi(args.stage, args.evidence, args.root)
        else:
            require(args.root.resolve() == Path('/root'), 'production_root_required')
            from scripts.promote_release import cleanup_on_disconnect
            with cleanup_on_disconnect(), (args.root / 'artifacts/release_controls.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = install_controls(args.stage, args.evidence, args.snapshot, args.pi_receipt, args.root)
        print(json.dumps({k: result[k] for k in ['status', 'revision', 'source_revision', 'activation_epoch',
                                              'services_restarted', 'application_files_unchanged'] if k in result}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, subprocess.SubprocessError):
        print(json.dumps({'status': 'control_release_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
