"""Release failures must never become live policy or rewind completed work."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from scripts import promotion_gate as gate
from scripts import release_state as state
from scripts import shadow_release as shadow
from scripts import promote_release as transaction
from scripts import deploy_tested_release as deploy
from scripts.shipping_checks import TEST_FILES
from sn68_source import ROOT

OLD, NEW = 'a' * 40, 'b' * 40
NOW = 100000


def ident(i):
    return f'{i:064x}'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def evidence():
    proposals = [ident(i) for i in range(1, 8001)]
    selected = [{'id': ident(i), 'selection_pass_ids': [ident(20000+j) for j in range(10)]}
                for i in range(1, 14)]
    cases = []
    for n, scenario in enumerate(['normal'] * 3 + sorted(gate.SCENARIOS - {'normal'})):
        cases.append({'name': f'case-{n}', 'scenario': scenario, 'epoch': 25900+n,
            'proposal_ids': proposals, 'previously_evaluated_ids': [ident(99999)],
            'new_budget': 4800, 'fleet_size': 13, 'currently_eligible_ids': proposals[:26],
            'field_sha256': ident(70000), 'held_out': {
                ident(i): {'pass_ids': [ident(30000+j) for j in range(10)],
                           'scores': [1000 if i < 14 else 2000] * 10} for i in range(1, 27)},
            'scoring_cutoff': 285, 'publication_cutoff': 315, 'pi_pull_cutoff': 320,
            'commit_cutoff': 345, 'epoch_length': 361})
    report = {'revision': OLD, 'fixture_sha256': '', 'handoff_artifact_sha256': ident(70001),
              'cases': [{'name': c['name'], 'epoch': c['epoch'], 'status': 'complete',
                'confirmation_health': 'healthy', 'dispatched_new_ids': proposals[:4800],
                'completed_new_ids': proposals[:4800], 'selected': copy.deepcopy(selected),
                'handoff': {'artifact_sha256': ident(70001), 'pi_artifact_sha256': ident(70001),
                    'confirmed_offset': 280, 'published_offset': 310, 'pi_pull_offset': 319},
                'old_epoch_work_running': False, 'unavailable_gates_admitted': 0} for c in cases]}
    documents = {'fixture': {'schema': 'sn68-production-trial-v1', 'target': 'TEST', 'cases': cases},
                 'baseline_report': report, 'candidate_report': copy.deepcopy(report),
                 'rollback_report': copy.deepcopy(report)}
    documents['candidate_report']['revision'] = NEW
    documents['rollback_report'].update(protected_state_before={'ledger': ident(8)},
                                        protected_state_after={'ledger': ident(8)})
    documents['shadow_receipt'] = {'schema': 'sn68-isolated-trial-v1', 'fixture_sha256': '',
        'network': 'none', 'root_read_only': True, 'host_credentials_mounted': False,
        'capabilities': [], 'live_submission_access': False, 'image_digest': ident(6),
        'runs': {k: {'revision': NEW if k == 'candidate' else OLD, 'exit_code': 0,
                    'report_sha256': '', 'manifest_sha256': ident(2 if k == 'candidate' else 1),
                    'resume_state_sha256': ident(5)} for k in ['baseline', 'candidate', 'rollback']}}
    documents['state_snapshot'] = {'schema': 'sn68-state-snapshot-v1', 'baseline_sha': OLD,
        'candidate_sha': NEW, 'restores_durable_state': False, 'files': {'ledger': ident(3)}}
    documents['rollback_receipt'] = {'baseline_sha': OLD, 'candidate_sha': NEW,
        'policy': 'preserve-durable-invalidate-derived-v1', 'restores_durable_state': False,
        'snapshot_manifest_sha256': '', 'post_candidate_state_sha256': ident(5)}
    documents['pi_receipt'] = {'schema': 'sn68-pi-shadow-v1', 'candidate_sha': NEW, 'target': 'TEST',
        'epoch': 26000, 'host': 'pi', 'read_only': True, 'submission_attempts': 0,
        'accepted_entries': 13, 'rejected_entries': 0, 'artifact_sha256': ident(70001),
        'consumer_sha256': ident(4), 'verified_at': NOW - 10}
    plan = {'schema': gate.SCHEMA, 'candidate_sha': NEW, 'baseline_sha': OLD, 'change_kind': 'repair',
        'host': 'producer.example', 'target': 'TEST', 'created_at': NOW - 30, 'expires_at': NOW + 60,
        'activation_epoch': 26000, 'baseline_manifest_sha256': ident(1)}
    return documents, plan


def save_bundle(root, docs, plan):
    """Rehash changed records so negative tests reach the semantic check."""
    for key in ['fixture', 'state_snapshot']:
        write(root / (key + '.json'), docs[key])
    fixture_hash = gate.digest(root / 'fixture.json')
    docs['rollback_receipt']['snapshot_manifest_sha256'] = gate.digest(root / 'state_snapshot.json')
    for key in ['baseline_report', 'candidate_report', 'rollback_report']:
        docs[key]['fixture_sha256'] = fixture_hash
        write(root / (key + '.json'), docs[key])
    docs['shadow_receipt']['fixture_sha256'] = fixture_hash
    for kind in ['baseline', 'candidate', 'rollback']:
        docs['shadow_receipt']['runs'][kind]['report_sha256'] = gate.digest(root / (kind + '_report.json'))
    for key, doc in docs.items():
        write(root / (key + '.json'), doc)
        plan[key] = {'path': key + '.json', 'sha256': gate.digest(root / (key + '.json'))}
    write(root / 'promotion.json', plan)


def qualify(root):
    return gate.verify(root, candidate_sha=NEW, host='producer.example', target='TEST', epoch=26000, now=NOW)


def test_complete_production_evidence_qualifies(tmp_path):
    docs, plan = evidence()
    save_bundle(tmp_path, docs, plan)
    assert qualify(tmp_path)['cases'] == 13


@pytest.mark.parametrize('failure', [
    'mixed_change', 'expired', 'wrong_epoch', 'wrong_host', 'wrong_revision', 'missing_scenario',
    'small_canary', 'cached_charged_new', 'overspend', 'unfinished_budget', 'fleet_shortfall',
    'one_pass', 'leakage', 'quality', 'late', 'relaxed_window', 'torn_handoff', 'old_work',
    'gates', 'degraded', 'network', 'credentials', 'skipped_trial', 'rewind_state', 'lost_state',
    'pi_stale', 'pi_wrong_artifact', 'pi_submission', 'pi_shortfall', 'baseline_failed', 'no_gain',
    'state_revision', 'rollback_state', 'manifest', 'duplicate_ids', 'nan_score'])
def test_promotion_refuses_each_incident_or_missing_proof(tmp_path, failure):
    docs, plan = evidence()
    case = docs['fixture']['cases'][0]
    after = docs['candidate_report']['cases'][0]
    pi = docs['pi_receipt']
    if failure == 'mixed_change': plan['change_kind'] = 'repair+experiment'
    if failure == 'expired': plan['expires_at'] = NOW - 1
    if failure == 'wrong_epoch': plan['activation_epoch'] += 1
    if failure == 'wrong_host': plan['host'] = 'other.example'
    if failure == 'wrong_revision': docs['candidate_report']['revision'] = OLD
    if failure == 'missing_scenario': docs['fixture']['cases'][-1]['scenario'] = 'normal'
    if failure == 'small_canary': case['proposal_ids'] = case['proposal_ids'][:32]
    if failure == 'cached_charged_new': case['previously_evaluated_ids'] = [ident(1)]
    if failure == 'overspend': after['dispatched_new_ids'] = case['proposal_ids'][:4801]
    if failure == 'unfinished_budget': after['completed_new_ids'] = after['completed_new_ids'][:4799]
    if failure == 'fleet_shortfall': after['selected'].pop()
    if failure == 'one_pass': after['selected'][0]['selection_pass_ids'] = [ident(20000)]
    if failure == 'leakage': after['selected'][0]['selection_pass_ids'] = case['held_out'][ident(1)]['pass_ids']
    if failure == 'quality': after['selected'][0]['id'] = ident(14)
    if failure == 'late': after['handoff']['pi_pull_offset'] = 321
    if failure == 'relaxed_window': case['scoring_cutoff'] = 350
    if failure == 'torn_handoff': after['handoff']['pi_artifact_sha256'] = ident(10)
    if failure == 'old_work': after['old_epoch_work_running'] = True
    if failure == 'gates': after['unavailable_gates_admitted'] = 1
    if failure == 'degraded': after['confirmation_health'] = 'degraded'
    if failure == 'network': docs['shadow_receipt']['network'] = 'host'
    if failure == 'credentials': docs['shadow_receipt']['host_credentials_mounted'] = True
    if failure == 'skipped_trial': docs['shadow_receipt']['runs']['candidate']['exit_code'] = 1
    if failure == 'rewind_state': docs['rollback_receipt']['restores_durable_state'] = True
    if failure == 'lost_state': docs['rollback_report']['protected_state_after'] = {}
    if failure == 'pi_stale': pi['verified_at'] = NOW - 901
    if failure == 'pi_wrong_artifact': pi['artifact_sha256'] = ident(99)
    if failure == 'pi_submission': pi['submission_attempts'] = 1
    if failure == 'pi_shortfall': pi['accepted_entries'] = 12
    if failure == 'baseline_failed': docs['baseline_report']['cases'][0]['status'] = 'failed'
    if failure == 'no_gain': plan['change_kind'] = 'experiment'
    if failure == 'state_revision': docs['state_snapshot']['candidate_sha'] = OLD
    if failure == 'rollback_state': docs['shadow_receipt']['runs']['rollback']['resume_state_sha256'] = ident(123)
    if failure == 'manifest': plan['baseline_manifest_sha256'] = ident(999)
    if failure == 'duplicate_ids': after['selected'][1]['id'] = after['selected'][0]['id']
    if failure == 'nan_score': case['held_out'][ident(1)]['scores'][0] = float('nan')
    save_bundle(tmp_path, docs, plan)
    with pytest.raises(ValueError): qualify(tmp_path)


def test_fixes_must_restore_failed_incidents_and_experiments_must_improve(tmp_path):
    docs, plan = evidence()
    docs['baseline_report']['cases'][-1]['status'] = 'failed'
    save_bundle(tmp_path, docs, plan)
    qualify(tmp_path)
    plan['change_kind'] = 'experiment'
    for case in docs['fixture']['cases']:
        case['held_out'][ident(14)]['scores'] = [900] * 10
    for result in docs['candidate_report']['cases']:
        result['selected'][0]['id'] = ident(14)
    save_bundle(tmp_path, docs, plan)
    qualify(tmp_path)


def test_discovery_regression_in_partial_wave_is_rejected(tmp_path):
    docs, plan = evidence()
    partial = next(r for c, r in zip(docs['fixture']['cases'], docs['candidate_report']['cases'])
                   if c['scenario'] == 'partial_wave')
    partial['completed_new_ids'] = partial['completed_new_ids'][:700]
    save_bundle(tmp_path, docs, plan)
    with pytest.raises(ValueError, match='discovery_regression'): qualify(tmp_path)


def test_evidence_bytes_and_duplicate_json_keys_are_checked(tmp_path):
    docs, plan = evidence()
    save_bundle(tmp_path, docs, plan)
    (tmp_path / 'pi_receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence_hash_mismatch'): qualify(tmp_path)
    (tmp_path / 'duplicate.json').write_text('{"status": true, "status": false}')
    with pytest.raises(ValueError, match='duplicate_json_key'): gate.read(tmp_path / 'duplicate.json')


def test_activation_refused_before_ssh_and_staging_has_no_live_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, 'ROOT', tmp_path)
    monkeypatch.setattr(deploy, 'verify', lambda *a, **k: None)
    monkeypatch.setenv('SN68_RELEASE_DIR', str(tmp_path))
    monkeypatch.setenv('SN68_TESTED_SHA', NEW)
    monkeypatch.setenv('SN68_DEPLOY_HOST', 'producer.example')
    monkeypatch.setenv('SN68_DEPLOY_PORT', '22')
    monkeypatch.setenv('SN68_DEPLOY_ACTIVATE', 'true')
    monkeypatch.delenv('SN68_PROMOTION_DIR', raising=False)
    monkeypatch.delenv('SSH_PRIVATE_KEY', raising=False)
    write(tmp_path / 'build_stamp.json', {})
    def forbidden(*a, **k): pytest.fail('No SSH or subprocess is permitted before qualification')
    monkeypatch.setattr(deploy.subprocess, 'run', forbidden)
    assert deploy.main() == 1
    play, variables = deploy.deployment_plan(tmp_path / 'build_stamp.json', NEW, 'producer.example', False)
    assert play == 'ansible/stage_nanobody_release.yml' and variables == {}
    tasks = yaml.safe_load((ROOT / play).read_text())[0]['tasks']
    assert tasks == [{'name': 'Stage and verify immutable release bytes',
                      'ansible.builtin.include_tasks': 'tasks/prepare_tested_release.yml'}]


def test_controller_binds_candidate_manifest(tmp_path, monkeypatch):
    docs, plan = evidence()
    save_bundle(tmp_path, docs, plan)
    monkeypatch.setattr(gate.time, 'time', lambda: NOW)
    manifest = tmp_path / 'build_stamp.json'
    write(manifest, {'git_sha': NEW})
    with pytest.raises(ValueError, match='candidate_manifest_mismatch'):
        deploy.deployment_plan(manifest, NEW, 'producer.example', True, tmp_path, 'TEST')
    docs['shadow_receipt']['runs']['candidate']['manifest_sha256'] = gate.digest(manifest)
    save_bundle(tmp_path, docs, plan)
    play, variables = deploy.deployment_plan(manifest, NEW, 'producer.example', True, tmp_path, 'TEST')
    assert play == 'ansible/promote_nanobody_release.yml'
    assert len(variables['sn68_promotion_files']) == 9


def durable(root):
    ledger = root / state.DURABLE[0]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('published-before\n')
    database = sqlite3.connect(root / state.DURABLE[3])
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('CREATE TABLE measurements (id INTEGER PRIMARY KEY, value INTEGER)')
    database.execute('INSERT INTO measurements VALUES (1, 100)')
    database.commit()
    write(root / 'live/epoch26000/producer/fresh_screen_budget.json', {'reserved': 100})
    return database


def test_snapshot_includes_wal_and_quarantine_never_rewinds_new_work(tmp_path):
    root = tmp_path / 'runtime'
    db = durable(root)
    snapshot = tmp_path / 'snapshot'
    state.capture(root, snapshot, epoch=26000, baseline_sha=OLD, candidate_sha=NEW)
    with sqlite3.connect(snapshot / state.DURABLE[3]) as saved:
        assert saved.execute('SELECT value FROM measurements').fetchall() == [(100,)]
    db.execute('INSERT INTO measurements VALUES (2, 200)'); db.commit()
    (root / state.DURABLE[0]).write_text('published-before\nsubmitted-after\n')
    write(root / 'live/epoch26000/producer/fresh_screen_budget.json', {'reserved': 120})
    cache = root / 'live/epoch26000/producer/full_scan_confirmation.json'
    write(cache, {'derived': True})
    next_cache = root / 'live/epoch26001/comparison_precheck.json'
    write(next_cache, {'prepared_ahead': True})
    next_budget = root / 'live/epoch26001/producer/fresh_screen_budget.json'
    write(next_budget, {'reserved': 5})
    raw = root / 'live/epoch26000/producer/iteration_00/exact_all.json'
    write(raw, {'observations': [1, 2]})
    receipt = state.quarantine_derived(root, tmp_path / 'quarantine', epoch=26000)
    assert receipt['restores_durable_state'] is False and not cache.exists()
    assert not next_cache.exists() and gate.read(next_budget)['reserved'] == 5
    assert raw.is_file() and db.execute('SELECT COUNT(*) FROM measurements').fetchone()[0] == 2
    assert 'submitted-after' in (root / state.DURABLE[0]).read_text()
    assert gate.read(root / 'live/epoch26000/producer/fresh_screen_budget.json')['reserved'] == 120
    db.close()


def release(root, revision, *, extra=False):
    contents = {'pixi.lock': 'locked\n', 'scripts/version.py': revision,
                'deploy/producer/producer_argv.txt': 'python\n-m\nproducer\n'}
    if extra: contents['scripts/new_only.py'] = 'candidate-only\n'
    files = {}
    for name, content in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
        files[name] = gate.digest(path)
    manifest = {'schema': 1, 'git_sha': revision, 'files': files, 'runtime_files': sorted(files),
        'pixi_lock_hash': files['pixi.lock'], 'validation': {'passed': True, 'tests': 100,
        'failures': 0, 'errors': 0, 'skipped': 0, 'test_files': list(TEST_FILES)}}
    write(root / 'build_stamp.json', manifest)
    return manifest


def test_failed_promotion_restores_code_but_keeps_candidate_measurements_and_reservations(tmp_path, monkeypatch):
    root, stage = tmp_path / 'runtime', tmp_path / 'stage'
    release(root, OLD); release(stage, NEW, extra=True)
    (root / 'build_stamp.json').write_text(json.dumps(gate.read(root / 'build_stamp.json')))
    original_manifest_hash = gate.digest(root / 'build_stamp.json')
    db = durable(root)
    (root / 'artifacts/producer_argv.txt').write_text('original argv')
    monkeypatch.setattr(deploy, 'deployment_plan', lambda *a, **k: ('guarded', {}))
    monkeypatch.setattr(transaction, 'verify_host', lambda *a, **k: {'activation_epoch': 26000})
    monkeypatch.setattr(transaction, 'publisher_launch', lambda: 'approved publisher')
    monkeypatch.setattr(transaction, 'processes', lambda: {
        101: (1, ['nanobody_opt_v2.run_live_producer']), 102: (1, ['/root/tools/publish_current_nanobody.py'])})
    monkeypatch.setattr(transaction.os, 'kill', lambda *a: None)
    monkeypatch.setattr(transaction, 'stop', lambda *a: None)
    launches = []
    def launch(_):
        launches.append(gate.read(root / 'build_stamp.json')['git_sha'])
        if len(launches) == 1:
            db.execute('INSERT INTO measurements VALUES (2, 200)'); db.commit()
            (root / state.DURABLE[0]).write_text('published-before\nsubmitted-after\n')
            write(root / 'live/epoch26000/producer/fresh_screen_budget.json', {'reserved': 140})
            write(root / 'live/epoch26000/producer/full_scan_confirmation.json', {'bad': True})
            raise ValueError('injected_startup_failure')
    monkeypatch.setattr(transaction, 'launch', launch)
    with pytest.raises(ValueError, match='injected_startup_failure'):
        transaction.promote(stage, tmp_path / 'proof', root, host='producer.example', target='TEST')
    assert launches == [NEW, OLD]
    assert (root / 'scripts/version.py').read_text() == OLD
    assert gate.digest(root / 'build_stamp.json') == original_manifest_hash
    assert not (root / 'scripts/new_only.py').exists()
    assert (root / 'artifacts/producer_argv.txt').read_text() == 'original argv'
    assert db.execute('SELECT COUNT(*) FROM measurements').fetchone()[0] == 2
    assert 'submitted-after' in (root / state.DURABLE[0]).read_text()
    assert gate.read(root / 'live/epoch26000/producer/fresh_screen_budget.json')['reserved'] == 140
    assert gate.read(next(stage.glob('promotion-attempt-*/promotion_failure.json')))['durable_state_restored'] is False
    db.close()


def test_docker_trial_has_no_live_mounts_network_or_credentials():
    command = shadow.container_args(name='test', image='runtime@sha256:' + ident(1),
        release='/baseline', harness='/candidate', fixture='/recorded', output='/scratch',
        driver='scripts/driver.py', resume='/candidate-state')
    assert {'--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--user=65534:65534'} <= set(command)
    mounts = [command[i+1] for i, arg in enumerate(command) if arg == '--mount']
    assert len(mounts) == 5
    assert all('readonly' in m for m in mounts if 'dst=/output' not in m)
    assert not any(x in ' '.join(mounts) for x in ['docker.sock', '/root', '.ssh', '.aws'])
    assert all(value in {'--env=PYTHONPATH=/release', '--env=PYTHONDONTWRITEBYTECODE=1'}
               for value in command if value.startswith('--env'))


def test_trial_timeout_destroys_container_and_emits_no_passing_receipt(tmp_path, monkeypatch):
    old, new, fixture, output = [tmp_path / p for p in ['old', 'new', 'fixture', 'output']]
    release(old, OLD); release(new, NEW)
    write(fixture / 'fixture.json', {'schema': 'fixture'})
    calls = []
    def docker(argv, **kwargs):
        calls.append(argv[:3])
        if argv[:2] == ['docker', 'start']:
            raise subprocess.TimeoutExpired('docker', 30)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(shadow.subprocess, 'run', docker)
    with pytest.raises(subprocess.TimeoutExpired):
        shadow.run(baseline=old, candidate=new, fixtures=fixture, output=output,
                   driver='scripts/version.py', image='runtime@sha256:' + ident(1), timeout=30)
    assert calls[-1] == ['docker', 'rm', '--force']
    assert not (output / 'shadow_receipt.json').exists()


def test_legacy_paths_and_restart_cannot_bypass_promotion():
    for playbook in ['nanobody.yml', 'nanobody_services.yml']:
        plays = yaml.safe_load((ROOT / 'ansible' / playbook).read_text())
        tasks = next(p['tasks'] for p in plays if 'tasks' in p)
        assert tasks[0]['ansible.builtin.include_tasks'] == 'tasks/refuse_live_reprovision.yml'
    guard = yaml.safe_load((ROOT / 'ansible/tasks/refuse_live_reprovision.yml').read_text())
    # The live incident host had no readable build stamp. Missing metadata must
    # never turn an installed producer into a supposedly fresh machine.
    assert {'/root/build_stamp.json', '/root/nanobody_opt_v2/run_live_producer.py',
            '/root/tools/publish_current_nanobody.py', '/root/artifacts/producer_argv.txt'} <= set(guard[0]['loop'])
    assert guard[1]['ansible.builtin.assert']['that'] == [
        "sn68_existing_runtime.results | selectattr('stat.exists') | list | length == 0"]
    assert 'promotion_gate.py" --restart' in (ROOT / 'deploy_gate.sh').read_text()
    workflow = yaml.load((ROOT / '.github/workflows/deploy-nanobody.yml').read_text(), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['deploy']['steps']
    names = [s.get('name') for s in steps]
    assert names.index('Verify evidence run belongs to this tested revision') < names.index('Deploy the tested release')
    assert 'tests/test_release_promotion.py' in TEST_FILES


def test_live_readiness_uses_nested_checkpoint_and_real_epoch_window(tmp_path, monkeypatch):
    monkeypatch.setattr(transaction, 'processes', lambda: {
        201: (1, ['nanobody_opt_v2.run_live_producer']), 202: (1, ['/root/tools/publish_current_nanobody.py'])})
    monkeypatch.setattr(transaction.time, 'time', lambda: NOW)
    write(tmp_path / 'artifacts/chain_state.json', {'epoch': 26000, 'block': 26000*361+10,
        'epoch_length': 361, 'updated_at_unix': NOW})
    assert transaction.readiness_remaining(tmp_path, 26000) == (315 - 10) * 12
    rows = [{'sequence': str(i), 'checkpoint': {'exact': {'observation_count': 10}}} for i in range(13)]
    artifact = {'epoch': 26000, 'updated_at_unix': NOW, 'submissions': rows}
    write(tmp_path / 'artifacts/nanobody_result.json', artifact)
    assert transaction.healthy(tmp_path, epoch=26000, activated=NOW-1, previous_pids=[101, 102])['entries'] == 13
    artifact['submissions'][0]['checkpoint']['exact']['observation_count'] = 1
    write(tmp_path / 'artifacts/nanobody_result.json', artifact)
    with pytest.raises(ValueError, match='live_fleet_degraded'):
        transaction.healthy(tmp_path, epoch=26000, activated=NOW-1, previous_pids=[101, 102])
    with pytest.raises(ValueError, match='readiness_clock_changed_or_stale'):
        transaction.readiness_remaining(tmp_path, 26001)


def test_restart_requires_active_matching_qualification_and_intact_bytes(tmp_path):
    release(tmp_path, NEW)
    receipt = {'status': 'active', 'candidate_sha': NEW,
        'manifest_sha256': gate.digest(tmp_path / 'build_stamp.json'), 'qualification_sha256': ident(13)}
    with pytest.raises(FileNotFoundError): gate.verify_restart(tmp_path, NEW)
    write(tmp_path / 'artifacts/release_promotion.json', receipt)
    assert gate.verify_restart(tmp_path, NEW)['status'] == 'qualified_restart'
    (tmp_path / 'scripts/version.py').write_text('tampered')
    with pytest.raises(ValueError): gate.verify_restart(tmp_path, NEW)


@pytest.mark.parametrize('problem', ['boundary', 'stale_clock', 'target', 'baseline'])
def test_live_preflight_refuses_wrong_boundary_or_runtime(tmp_path, monkeypatch, problem):
    root, bundle = tmp_path / 'runtime', tmp_path / 'proof'
    release(root, OLD)
    docs, plan = evidence()
    digest = gate.digest(root / 'build_stamp.json')
    plan['baseline_manifest_sha256'] = digest
    docs['shadow_receipt']['runs']['baseline']['manifest_sha256'] = digest
    save_bundle(bundle, docs, plan)
    clock = {'nanobody_target': 'TEST', 'epoch': 26000, 'block': 26000*361+2,
             'epoch_length': 361, 'updated_at_unix': NOW}
    if problem == 'boundary': clock['block'] += 20
    if problem == 'stale_clock': clock['updated_at_unix'] -= 200
    if problem == 'target': clock['nanobody_target'] = 'OTHER'
    if problem == 'baseline': (root / 'scripts/version.py').write_text('changed')
    write(root / 'artifacts/chain_state.json', clock)
    monkeypatch.setattr(gate.time, 'time', lambda: NOW)
    with pytest.raises(ValueError):
        gate.verify_host(bundle, runtime_root=root, candidate_sha=NEW, host='producer.example', target='TEST')


def test_backup_failure_never_pauses_running_services(tmp_path, monkeypatch):
    root, stage = tmp_path / 'runtime', tmp_path / 'stage'
    release(root, OLD); release(stage, NEW)
    db = durable(root)
    (root / 'artifacts/producer_argv.txt').write_text('original argv')
    monkeypatch.setattr(deploy, 'deployment_plan', lambda *a, **k: ('guarded', {}))
    monkeypatch.setattr(transaction, 'verify_host', lambda *a, **k: {'activation_epoch': 26000})
    monkeypatch.setattr(transaction, 'publisher_launch', lambda: 'approved publisher')
    monkeypatch.setattr(transaction, 'processes', lambda: {
        101: (1, ['nanobody_opt_v2.run_live_producer']), 102: (1, ['/root/tools/publish_current_nanobody.py'])})
    def forbidden(*a, **k): pytest.fail('Backup failure must not pause or kill the live services')
    def fail(*a, **k): raise OSError('injected_snapshot_failure')
    monkeypatch.setattr(transaction.os, 'kill', forbidden)
    monkeypatch.setattr(transaction, 'capture', fail)
    with pytest.raises(OSError, match='injected_snapshot_failure'):
        transaction.promote(stage, tmp_path / 'proof', root, host='producer.example', target='TEST')
    assert (root / 'scripts/version.py').read_text() == OLD
    db.close()


@pytest.mark.parametrize('embedded_stamp', [True, False])
def test_pi_receipt_uses_installed_allocation_without_main_or_commits(tmp_path, monkeypatch, embedded_stamp):
    from scripts import pi_shadow_receipt as pi
    consumer_root = tmp_path / 'consumer'
    consumer_root.mkdir()
    shutil.copyfile(ROOT / 'deploy/submit_host/pi_submit.py', consumer_root / 'pi_submit.py')
    # Only host identity is replaced; execute the actual installed consumer.
    original_require = pi.require
    monkeypatch.setattr(pi, 'require', lambda condition, reason:
                        None if reason == 'receipt_requires_pi' else original_require(condition, reason))
    monkeypatch.setattr(pi.time, 'time', lambda: NOW)
    artifact = tmp_path / 'trial.json'
    payload = {'target': 'TEST', 'epoch': 26000, 'sequence_pool': [f'trial-{i}' for i in range(13)]}
    if embedded_stamp: payload['updated_at_unix'] = NOW
    write(artifact, payload)
    receipt = pi.collect(artifact, consumer_root, revision=NEW, target='TEST', epoch=26000,
                         uids=list(range(13)), publication_mtime=NOW)
    assert receipt['accepted_entries'] == 13 and receipt['submission_attempts'] == 0
    assert receipt['consumer_sha256'] == gate.digest(consumer_root / 'pi_submit.py')
    assert receipt['artifact_sha256'] == gate.digest(artifact)
    if not embedded_stamp:
        with pytest.raises(ValueError, match='stale_trial_artifact'):
            pi.collect(artifact, consumer_root, revision=NEW, target='TEST', epoch=26000,
                       uids=list(range(13)), publication_mtime=NOW-901)
        with pytest.raises(ValueError, match='missing_publication_clock'):
            pi.collect(artifact, consumer_root, revision=NEW, target='TEST', epoch=26000,
                       uids=list(range(13)))


def test_shadow_rejects_credentials_symlinks_and_reused_outputs(tmp_path):
    directory = tmp_path / 'input'; directory.mkdir()
    (directory / '.env').write_text('placeholder fixture only')
    with pytest.raises(ValueError): shadow.inventory(directory)
    (directory / '.env').unlink()
    (directory / 'link').symlink_to(tmp_path / 'outside')
    with pytest.raises(ValueError): shadow.inventory(directory)
    with pytest.raises(ValueError, match='trial_output_overlap'):
        shadow.run(baseline=directory, candidate=directory, fixtures=directory, output=directory,
                   driver='scripts/driver.py', image='runtime@sha256:' + ident(1))


def test_disconnect_enters_cleanup_and_repeated_signals_cannot_interrupt_rollback(monkeypatch):
    registered = {}
    sentinel = object()
    def register(sig, handler):
        previous = registered.get(sig, sentinel)
        registered[sig] = handler
        return previous
    monkeypatch.setattr(transaction.signal, 'signal', register)
    cleaned = []
    with pytest.raises(InterruptedError, match='promotion_interrupted'):
        with transaction.cleanup_on_disconnect():
            try:
                registered[transaction.signal.SIGHUP](transaction.signal.SIGHUP, None)
            finally:
                registered[transaction.signal.SIGTERM](transaction.signal.SIGTERM, None)
                cleaned.append('restoration completed')
    assert cleaned == ['restoration completed']
    assert set(registered.values()) == {sentinel}
