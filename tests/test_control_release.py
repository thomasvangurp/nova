"""Control activation must not replace or pause the application it protects."""
import base64
import copy
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import pytest
import yaml

from scripts import control_release as control
from scripts import release_manifest as release
from scripts import pi_control_agent as agent
from scripts import promotion_gate as gate
from scripts.replay_handoff import replay
from scripts.shipping_checks import TEST_FILES
from sn68_source import ROOT

OLD, NEW = 'a'*40, 'b'*40
NOW = 1000000.0


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def application(root, source_manifest, snapshot):
    files = {}
    for name in control.REQUIRED_APPLICATION | {'nanobody_opt/fleet.py'}:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# existing application\n'); files[name] = gate.digest(path)
    files['tools/unused_optional_tool.py'] = 'e'*64
    files['nanobody_dash/build.py'] = 'f'*64
    argv = root / 'artifacts/producer_argv.txt'; argv.parent.mkdir(parents=True, exist_ok=True)
    argv.write_text('/root/.pixi/envs/bgen/bin/python\n-m\nnanobody_opt_v2.run_live_producer\n'
                    '--epoch-candidate-budget\n4800\n--epoch-proposal-budget\n8000\n')
    (root / 'producer_relaunch.sh').write_text('#!/bin/bash\n# unchanged launcher\n')
    write(source_manifest, {'git_sha': OLD, 'files': files})
    return control.capture_application(root, source_manifest, snapshot)


def package(root):
    for name in release.CONTROL_FILES | {'pixi.lock'}:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# tested control ' + name + '\n')
    shutil.copytree(ROOT / control.FIXTURES, root / control.FIXTURES)
    files = {p.relative_to(root).as_posix(): gate.digest(p) for p in root.rglob('*') if p.is_file()}
    manifest = {'schema': 1, 'git_sha': NEW, 'deployment_profile': 'release-controls',
        'submodules': {'NOVA-nanobody-filter': 'fe2d5868126406a0de8953d26ecd8bdccbdbd029'},
        'files': files, 'runtime_files': sorted(release.CONTROL_FILES), 'pixi_lock_hash': files['pixi.lock'],
        'validation': {'passed': True, 'tests': 100, 'failures': 0, 'errors': 0, 'skipped': 0,
                       'test_files': list(TEST_FILES)}}
    write(root / 'build_stamp.json', manifest)
    return manifest


def evidence(stage, output):
    report = replay(stage / control.FIXTURES)
    write(output / 'report.json', report)
    qualification = {'schema': 'sn68-control-qualification-v1', 'revision': NEW,
        'created_at': time.time(),
        'manifest_sha256': gate.digest(stage / 'build_stamp.json'),
        'driver_sha256': gate.digest(stage / 'scripts/replay_handoff.py'),
        'isolation': {'network': 'none', 'read_only': True, 'capabilities': [],
                      'credentials_mounted': False, 'image_digest': 'd'*64, 'exit_code': 0},
        'report': {'path': 'report.json', 'sha256': gate.digest(output / 'report.json')}}
    write(output / 'qualification.json', qualification)
    return qualification, report


def setup_install(tmp_path, monkeypatch, entries=12):
    root = tmp_path / 'runtime'; stage = root / 'releases' / NEW
    snapshot = tmp_path / 'observed.json'; source = tmp_path / 'source.json'
    application(root, source, snapshot)
    package(stage)
    monkeypatch.setattr(control.time, 'time', lambda: NOW)
    proof = tmp_path / 'evidence'; _, report = evidence(stage, proof)
    write(root / 'artifacts/chain_state.json', {'epoch': 26000, 'nanobody_target': 'P20809',
                                               'updated_at_unix': NOW})
    artifact = {'epoch': 26000, 'target': 'P20809', 'submissions': [{'uid': i} for i in range(entries)]}
    write(root / 'artifacts/nanobody_result.json', artifact)
    os.utime(root / 'artifacts/nanobody_result.json', (NOW, NOW))
    request = control.request_pi(stage, proof, root)
    pi = tmp_path / 'pi.json'
    write(pi, {'schema': 'sn68-pi-shadow-v1', 'host': 'pi', 'candidate_sha': NEW, 'target': 'P20809',
        'epoch': 26000, 'read_only': True, 'submission_attempts': 0,
        'accepted_entries': entries, 'rejected_entries': 0, 'verified_at': NOW,
        'consumer_sha256': report['installed_consumer_sha256'], 'request_nonce': request['nonce'],
        'artifact_sha256': request['artifact_sha256']})
    return root, stage, snapshot, proof, pi


def test_actual_recorded_handoffs_are_not_synthetic_fleets():
    report = replay(ROOT / control.FIXTURES)
    assert [r['epoch'] for r in report['cases']] == [25913, 25916, 25918, 25921, 25926, 25927]
    assert [r['entries'] for r in report['cases']] == [12, 13, 13, 13, 13, 13]
    assert report['submission_attempts'] == report['scoring_calls'] == 0
    assert all(row['wrong_epoch_refused'] for row in report['cases'])


@pytest.mark.parametrize('epoch', [25913, 25916, 25918, 25921, 25926, 25927])
def test_receipt_wrapper_accepts_real_archived_publication_formats(tmp_path, monkeypatch, epoch):
    from scripts import pi_shadow_receipt as pi
    fixtures = ROOT / control.FIXTURES
    fixture = gate.read(fixtures / 'fixture.json')
    case = next(row for row in fixture['cases'] if row['epoch'] == epoch)
    consumer = tmp_path / 'consumer'; consumer.mkdir()
    shutil.copyfile(fixtures / fixture['consumer']['path'], consumer / 'pi_submit.py')
    original_require = pi.require
    monkeypatch.setattr(pi, 'require', lambda condition, reason:
                        None if reason == 'receipt_requires_pi' else original_require(condition, reason))
    monkeypatch.setattr(pi.time, 'time', lambda: case['replay_time'])
    receipt = pi.collect(fixtures / case['artifact']['path'], consumer,
                         revision=NEW, target=fixture['target'], epoch=epoch, uids=case['uids'],
                         publication_mtime=case['replay_time'])
    assert receipt['accepted_entries'] == len(case['uids'])
    assert receipt['submission_attempts'] == 0


def test_observed_application_capture_is_honest_and_role_specific(tmp_path):
    root = tmp_path / 'runtime'
    snapshot = application(root, tmp_path / 'source.json', tmp_path / 'observed.json')
    assert snapshot['application_ci_attested'] is False
    assert 'nanobody_dash/build.py' not in snapshot['files']
    assert snapshot['absent_nonrequired_files'] == ['tools/unused_optional_tool.py']
    control.verify_application(root, snapshot)
    (root / 'tools/unused_optional_tool.py').write_text('new unreviewed application code')
    with pytest.raises(ValueError, match='application_inventory_drift'):
        control.verify_application(root, snapshot)


@pytest.mark.parametrize('name', ['nanobody_opt/fleet.py', 'artifacts/producer_argv.txt', 'producer_relaunch.sh'])
def test_observed_snapshot_never_waives_application_or_launcher_drift(tmp_path, name):
    root = tmp_path / 'runtime'
    snapshot = application(root, tmp_path / 'source.json', tmp_path / 'observed.json')
    (root / name).write_text('changed')
    with pytest.raises(ValueError, match='application_or_launcher_drift'):
        control.verify_application(root, snapshot)


def test_control_profile_cannot_install_application_files(tmp_path):
    manifest = package(tmp_path)
    release.verify(tmp_path, manifest, expected_sha=NEW)
    assert not any(name.startswith(('nanobody_opt/', 'nanobody_opt_v2/', 'tools/', 'config/'))
                   for name in release.CONTROL_FILES)
    manifest['runtime_files'].append('nanobody_opt_v2/run_live_producer.py')
    with pytest.raises(ValueError, match='runtime inventory'):
        release.verify(tmp_path, manifest, expected_sha=NEW)


@pytest.mark.parametrize('entries', [12, 13])
def test_control_install_protects_application_and_actual_current_fleet(tmp_path, monkeypatch, entries):
    root, stage, snapshot, proof, pi = setup_install(tmp_path, monkeypatch, entries)
    before = gate.read(snapshot)
    # A newer publication does not erase the immutable artifact checked by Pi.
    write(root / 'artifacts/nanobody_result.json', {'epoch': 26000, 'newer_publication': True})
    result = control.install_controls(stage, proof, snapshot, pi, root)
    assert result['application_files_unchanged'] is True and result['services_restarted'] is False
    assert result['application_ci_attested'] is False and result['pi_entries'] == entries
    assert result['launcher_guard_installed'] is True and result['launch_arguments_unchanged'] is True
    installed = gate.read(root / 'artifacts/observed_application.json')
    assert installed['files'] == before['files']
    assert installed['launch_bindings']['artifacts/producer_argv.txt'] == before['launch_bindings']['artifacts/producer_argv.txt']
    control.verify_application(root, installed)
    assert control.verify_installed(root, NEW)['status'] == 'verified'
    assert gate.verify_restart(root, NEW)['status'] == 'verified'
    from scripts.check_launch_contract import check
    check(root, installed=True)  # no template is present; check the actual launcher


def test_restart_guard_preserves_exact_commands_and_does_not_accumulate():
    original = b'#!/bin/bash\nset -eu\nexec python3 worker.py "$@"\n'
    first = control.guard_launcher(original, OLD)
    second = control.guard_launcher(first, NEW)
    assert second.endswith(original.split(b'\n', 1)[1])
    assert second.count(control.GUARD_BEGIN.encode()) == 1
    assert OLD.encode() not in second and NEW.encode() in second
    assert control.guard_launcher(second, NEW) == second


@pytest.mark.parametrize('problem', ['expired', 'wrong_host', 'consumer_changed', 'wrong_nonce', 'shortfall', 'wrong_artifact', 'wrong_epoch', 'wrote_submission'])
def test_control_install_refuses_bad_pi_evidence_without_writing(tmp_path, monkeypatch, problem):
    root, stage, snapshot, proof, pi_path = setup_install(tmp_path, monkeypatch)
    pi = gate.read(pi_path)
    if problem == 'expired': pi['verified_at'] -= 901
    if problem == 'wrong_host': pi['host'] = 'mac'
    if problem == 'consumer_changed': pi['consumer_sha256'] = 'e'*64
    if problem == 'wrong_nonce': pi['request_nonce'] = 'e'*32
    if problem == 'shortfall': pi['accepted_entries'] = 11
    if problem == 'wrong_artifact': pi['artifact_sha256'] = 'e'*64
    if problem == 'wrong_epoch': pi['epoch'] -= 1
    if problem == 'wrote_submission': pi['submission_attempts'] = 1
    write(pi_path, pi)
    with pytest.raises(ValueError): control.install_controls(stage, proof, snapshot, pi_path, root)
    assert not (root / 'build_stamp.json').exists()
    assert not (root / 'scripts/control_release.py').exists()


@pytest.mark.parametrize('age', [901, -1])
def test_old_unstamped_publication_cannot_be_freshened_by_copying_it(tmp_path, monkeypatch, age):
    root, stage, snapshot, proof, pi = setup_install(tmp_path, monkeypatch)
    request = root / 'artifacts/control_qualification_request.json'
    previous = request.read_bytes()
    original = root / 'artifacts/nanobody_result.json'
    os.utime(original, (NOW-age, NOW-age))
    with pytest.raises(ValueError, match='stale_original_publication'):
        control.request_pi(stage, proof, root)
    assert request.read_bytes() == previous


def test_failed_control_copy_restores_controls_without_rewinding_concurrent_work(tmp_path, monkeypatch):
    root, stage, snapshot, proof, pi = setup_install(tmp_path, monkeypatch)
    old = root / 'scripts/shipping_checks.py'; old.parent.mkdir(parents=True)
    old.write_text('previous control')
    ledger = root / 'artifacts/submitted.txt'; ledger.write_text('before\n')
    original_copy = shutil.copy2
    def fail(source, destination, *args, **kwargs):
        if Path(source) == stage / 'scripts/control_release.py':
            ledger.write_text('before\nconcurrent-new-submission\n')
            raise OSError('injected_copy_failure')
        return original_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(control.shutil, 'copy2', fail)
    with pytest.raises(OSError, match='injected_copy_failure'):
        control.install_controls(stage, proof, snapshot, pi, root)
    assert old.read_text() == 'previous control'
    assert not (root / 'build_stamp.json').exists()
    assert 'concurrent-new-submission' in ledger.read_text()
    control.verify_application(root, gate.read(snapshot))


@pytest.mark.parametrize('problem', ['expired', 'host_network', 'credentials', 'unpassed', 'different_manifest', 'different_mapping'])
def test_control_qualification_refuses_unisolated_or_mismatched_results(tmp_path, problem):
    stage, proof = tmp_path / 'stage', tmp_path / 'proof'
    package(stage); qualification, report = evidence(stage, proof)
    if problem == 'expired': qualification['created_at'] -= 86401
    if problem == 'host_network': qualification['isolation']['network'] = 'host'
    if problem == 'credentials': qualification['isolation']['credentials_mounted'] = True
    if problem == 'unpassed': qualification['isolation']['exit_code'] = 1
    if problem == 'different_manifest': qualification['manifest_sha256'] = 'e'*64
    if problem == 'different_mapping':
        report['cases'][0]['allocation_sha256'] = 'e'*64
        write(proof / 'report.json', report)
        qualification['report']['sha256'] = gate.digest(proof / 'report.json')
    write(proof / 'qualification.json', qualification)
    with pytest.raises(ValueError): control.verify_evidence(stage, proof)


def test_pi_agent_uses_private_scratch_and_only_delivers_a_read_only_receipt(tmp_path, monkeypatch):
    artifact = {'epoch': 26000, 'target': 'P20809', 'submissions': [{'uid': i} for i in range(12)]}
    raw = json.dumps(artifact).encode()
    import hashlib
    request = {'schema': 'sn68-control-pi-request-v1', 'candidate_sha': NEW, 'nonce': 'a'*32,
        'epoch': 26000, 'target': 'P20809', 'created_at': NOW-1, 'expires_at': NOW+60,
        'publication_mtime': NOW-2,
        'artifact_sha256': hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(agent.time, 'time', lambda: NOW)
    monkeypatch.setattr(agent, 'endpoint', lambda *a: ('producer.example', 22))
    paths, delivered = [], []
    def remote(host, port, script, payload=None):
        assert (host, port) == ('producer.example', 22)
        if script == agent.FETCH:
            assert payload is None
            return {'status': 'requested', 'request': request, 'artifact': base64.b64encode(raw).decode()}
        assert script == agent.DELIVER
        delivered.append(json.loads(payload)); return {'status': 'delivered'}
    monkeypatch.setattr(agent, 'remote', remote)
    def collect(path, consumer_root, **kwargs):
        assert path.read_bytes() == raw and not path.is_relative_to(consumer_root)
        assert kwargs['uids'] == list(range(12)); paths.append(path)
        assert kwargs['publication_mtime'] == NOW-2
        return {'accepted_entries': 12, 'submission_attempts': 0, 'read_only': True}
    monkeypatch.setattr(agent, 'collect', collect)
    assert agent.tick()['status'] == 'verified'
    assert delivered[0]['submission_attempts'] == 0 and delivered[0]['request_nonce'] == request['nonce']
    assert not paths[0].exists()


def test_qualification_runs_on_available_hosted_infrastructure_without_secrets():
    workflow = yaml.load((ROOT / '.github/workflows/qualify-nanobody-release.yml').read_text(), Loader=yaml.BaseLoader)
    job = workflow['jobs']['qualification']
    assert job['runs-on'] == 'ubuntu-latest' and job['needs'] == 'checks'
    assert workflow['jobs']['checks']['with']['release_profile'] == 'release-controls'
    assert 'secrets.' not in json.dumps(workflow)
    assert 'control_release.py qualify' in json.dumps(job)
    play = yaml.safe_load((ROOT / 'ansible/promote_release_controls.yml').read_text())
    assert 'kill-session' not in json.dumps(play) and 'restart' not in json.dumps(play)
