"""Independent regressions for qualified novelty execution and ownership."""
import sys
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from nanobody_opt import check_validator_novelty as novelty
from nanobody_opt import pr100_novelty_runtime as runtime
from nanobody_opt import pr100_epoch_continuation as continuation, rules
from tests.test_pr100_epoch_continuation import legacy, write
from tests.test_pr100_live_handoff_contract import replay

QUALIFIED = {'python': '/qualified/python', 'path': '/qualified/bin',
             'python_path': '/qualified/site', 'identity_sha256': 'a' * 64}


@pytest.mark.parametrize('chain', [SimpleNamespace(), SimpleNamespace(batch=None)])
def test_missing_archive_batch_api_is_execution_failure(monkeypatch, chain):
    """A missing API is not evidence that every archive sequence lacks CDRs."""
    monkeypatch.setitem(sys.modules, 'abnumber', SimpleNamespace(Chain=chain))
    with pytest.raises(RuntimeError):
        novelty._pr100_archive_cdrs([('AAAA', 'archive-id')])


def test_qualified_batch_exception_keeps_pinned_failure_semantics(monkeypatch):
    """Do not replace scientific batch failure handling while fixing capability."""
    calls = []

    def batch(sequences, *, scheme):
        calls.append((sequences, scheme))
        raise ValueError('synthetic numbering failure')

    monkeypatch.setitem(sys.modules, 'abnumber', SimpleNamespace(
        Chain=SimpleNamespace(batch=batch)))
    assert novelty._pr100_archive_cdrs([('AAAA', 'archive-id')]) == {'archive-id': None}
    assert calls == [({'archive-id': 'AAAA'}, 'imgt')]


def test_qualified_batch_outputs_are_not_replaced_with_fallback(monkeypatch):
    def batch(sequences, *, scheme):
        assert sequences == {'archive-id': 'AAAA'} and scheme == 'imgt'
        return {'archive-id': SimpleNamespace(cdr1_seq='A', cdr2_seq='B', cdr3_seq='C')}, {}

    monkeypatch.setitem(sys.modules, 'abnumber', SimpleNamespace(
        Chain=SimpleNamespace(batch=batch)))
    assert novelty._pr100_archive_cdrs([('AAAA', 'archive-id')]) == {
        'archive-id': {'cdr1': 'A', 'cdr2': 'B', 'cdr3': 'C'}}


@pytest.mark.parametrize('failure', ['exception', 'absent', 'empty_cdr'])
def test_positive_batch_preflight_rejects_nonfunctional_runtime(monkeypatch, failure):
    monkeypatch.setattr(runtime, 'runtime_identity', lambda: {'measured': True})
    def batch(*_a, **_kw):
        if failure == 'exception':
            raise RuntimeError('synthetic missing executable')
        return ({'qualified-positive': SimpleNamespace(cdr1_seq='', cdr2_seq='B', cdr3_seq='C')}
                if failure == 'empty_cdr' else {}), {}
    monkeypatch.setitem(sys.modules, 'abnumber', SimpleNamespace(Chain=SimpleNamespace(batch=batch)))
    with pytest.raises(RuntimeError, match='preflight failed'):
        runtime.validate_runtime('AAAA')


def reply_for(request):
    return {'schema': runtime.SCHEMA, 'status': 'complete',
        **{k: request[k] for k in ('runtime_sha256', 'archive_sha256', 'input_sha256')},
        'archive_rows': 1, 'verdicts': [
            {'sequence_hash': rules.sequence_hash(row['sequence']), 'validator_novel': True,
             'exact_historical': False, 'coarse_candidates': 0, 'maximum_identity': 0.0}
            for row in request['rows']]}


@pytest.fixture
def checked_input(tmp_path):
    archive = tmp_path / 'archive.csv'; archive.write_text('sequence,sequence_hash\nAAAA,hash\n')
    return [{'sequence': 'BBBB', 'pr100_evidence': {'original': True}}], archive


def test_child_keeps_outer_ownership_and_uses_only_qualified_environment(checked_input, monkeypatch):
    rows, archive = checked_input
    monkeypatch.setenv('LD_LIBRARY_PATH', '/native/libraries')
    monkeypatch.setenv('OPENMM_PLUGIN_DIR', '/native/plugins')
    def run(command, **kwargs):
        assert command[0] == QUALIFIED['python']
        assert kwargs.get('start_new_session', False) is False and 'preexec_fn' not in kwargs
        assert kwargs['capture_output'] is True
        assert 'LD_LIBRARY_PATH' not in kwargs['env'] and 'OPENMM_PLUGIN_DIR' not in kwargs['env']
        assert kwargs['env']['PATH'] == QUALIFIED['path']
        request = json.loads(kwargs['input'])
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply_for(request)), stderr='')
    kept, count = runtime.check(rows, archive, runtime=QUALIFIED, run=run)
    assert count == 1 and kept[0]['pr100_evidence'] == rows[0]['pr100_evidence']
    assert kept[0]['novelty']['runtime_sha256'] == QUALIFIED['identity_sha256']


@pytest.mark.parametrize('field', ['runtime_sha256', 'archive_sha256', 'input_sha256'])
def test_bound_child_reply_rejects_changed_identity(checked_input, field):
    rows, archive = checked_input
    def run(_command, **kwargs):
        reply = reply_for(json.loads(kwargs['input'])); reply[field] = 'b' * 64
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')
    with pytest.raises(ValueError, match='invalid qualified novelty reply'):
        runtime.check(rows, archive, runtime=QUALIFIED, run=run)


def test_changed_archive_during_child_never_gets_retagged(checked_input):
    rows, archive = checked_input
    def run(_command, **kwargs):
        reply = reply_for(json.loads(kwargs['input'])); archive.write_text('changed\n')
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')
    with pytest.raises(ValueError):
        runtime.check(rows, archive, runtime=QUALIFIED, run=run)


def test_changed_native_evidence_during_child_is_not_reused(checked_input):
    rows, archive = checked_input
    def run(_command, **kwargs):
        reply = reply_for(json.loads(kwargs['input']))
        rows[0]['pr100_evidence']['original'] = False
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr='')
    with pytest.raises(ValueError):
        runtime.check(rows, archive, runtime=QUALIFIED, run=run)


@pytest.mark.parametrize('kind', ['nonzero', 'malformed', 'duplicate', 'nonfinite'])
def test_child_failure_is_not_success_or_raw_diagnostic(checked_input, kind):
    rows, archive = checked_input
    def run(_command, **kwargs):
        reply = reply_for(json.loads(kwargs['input']))
        if kind == 'duplicate': reply['verdicts'].append(reply['verdicts'][0])
        if kind == 'nonfinite': reply['verdicts'][0]['maximum_identity'] = float('nan')
        return SimpleNamespace(returncode=1 if kind == 'nonzero' else 0,
            stdout='private child diagnostic' if kind == 'malformed' else json.dumps(reply),
            stderr='private child diagnostic')
    with pytest.raises((RuntimeError, ValueError)) as error:
        runtime.check(rows, archive, runtime=QUALIFIED, run=run)
    assert 'private child diagnostic' not in str(error.value)


def qualified_config(config):
    config['hosts'] = {'master': {'novelty_runtime': dict(QUALIFIED)},
                       'slave': {'novelty_runtime': {**QUALIFIED, 'identity_sha256': 'b' * 64}}}
    return config


def checkpoint(row, challenge, count=3):
    measurement = row['pr100_evidence']['developability']
    return {'sequence': row['sequence'], 'pr100_annotation': row['pr100_evidence']['annotation'],
        'native_resume': {'schema': 'pr100-native-confirmation-checkpoint-v1',
            'sequence_hash': rules.sequence_hash(row['sequence']), 'execution_host': 'master',
            'identity': {'resume_eligible': True, 'challenge_sha256': rules.digest(challenge),
                         'filter_sha': rules.FILTER_SHA,
                         'requirements_sha256': rules.digest(rules.pinned_config(challenge['epoch'])['nanobody_requirements'])},
            'measurements': [copy.deepcopy(measurement) for _ in range(count)],
            'required_cpu_observations': 3}}


def clean_native_artifact(directory, value, records=(), input_rows=None, preserve_admission_proof=False):
    rows = input_rows if input_rows is not None else value.get('candidates', []) + [r for r in records if 'sequence' in r]
    assigned = {rules.sequence_hash(row['sequence']) for row in rows}
    value = {**value, 'attempted_sequence_hashes': sorted(assigned)}
    if not preserve_admission_proof:
        for key in ('fresh_admitted_sequence_hashes', 'fresh_never_started_sequence_hashes',
                    'verified_fresh_never_started_sequence_hashes', 'fresh_admission_uncertain_sequence_hashes'):
            if key in value: value[key] = [digest for digest in value[key] if digest in assigned]
    write(directory / 'native-supervisor.json', {'cleanup': {
        'remaining_descendants': 0, 'fixed_point_reached': True}})
    write(directory / 'native/result.json', value)
    write(directory / 'native/native-confirmations.json', {'challenge': value['challenge'], 'records': list(records)})
    write(directory / 'native-input.json', {**value, 'candidates': rows})


def old_novelty_rejection(directory, validity, rows):
    import base64
    incoming = {**validity, 'candidates': rows}
    write(directory / 'novelty-supervisor.json', {'cleanup': {
        'remaining_descendants': 0, 'fixed_point_reached': True}})
    write(directory / 'novelty-request.json', {'request': {'files': {
        'input.json': base64.b64encode(json.dumps(incoming).encode()).decode()}}})
    write(directory / 'novelty/result.json', {**incoming, 'candidates': []})


def test_migration_restores_novelty_rejects_without_native_redraw_or_score_loss(legacy):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    qualified_config(x.config)
    row = x.rows[1]; full = checkpoint(row, x.replay.challenge)
    clean_native_artifact(x.local / 'attempts/00001', {**x.replay.validity, 'candidates': []}, [full])
    state['invalidated_native_hashes'] = [rules.sequence_hash(row['sequence'])]
    old_novelty_rejection(x.local / 'attempts/00001/novelty-rechecks/old', x.replay.validity, [row])
    before_scores = copy.deepcopy(state['score_rows']); before_native = state['native_rows'][0]['pr100_evidence']
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert state['pending_confirmation'] == [full]
    assert state['invalidated_native_hashes'] == [] and state['score_rows'] == before_scores
    assert state['native_rows'][0]['pr100_evidence'] == before_native
    assert state['pending_publication'] is True
    once = copy.deepcopy(state)
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config) and state == once


def test_later_real_native_failure_cannot_be_resurrected_by_migration(legacy):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    qualified_config(x.config); row = x.rows[0]; digest = rules.sequence_hash(row['sequence'])
    clean_native_artifact(x.local / 'attempts/00001', x.replay.validity, [checkpoint(row, x.replay.challenge)])
    clean_native_artifact(x.local / 'attempts/00002', {**x.replay.validity, 'candidates': [],
        'terminal_native_outcomes': {digest: 'psh_threshold'}}, input_rows=[row])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert row['sequence'] not in {r['sequence'] for r in state['native_rows']}
    assert not state['pending_confirmation'] and digest in state['invalidated_native_hashes']


@pytest.mark.parametrize('kind', ['active', 'remote_unclean'])
def test_migration_never_releases_uncertain_ownership(legacy, kind):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    if kind == 'active': state['active'] = {'number': 1, 'phase': 'native', 'native_unresolved': True}
    else: write(x.local / 'attempts/00001/native-request.json', {'durable_intent': True})
    before = copy.deepcopy(state)
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config) is False
    assert state == before


@pytest.mark.parametrize('field', ['challenge_sha256', 'filter_sha', 'requirements_sha256'])
def test_migration_does_not_trust_mismatched_checkpoint_binding(legacy, field):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    row = checkpoint(x.rows[1], x.replay.challenge); row['native_resume']['identity'][field] = '0' * 64
    clean_native_artifact(x.local / 'attempts/00001', {**x.replay.validity, 'candidates': []}, [row])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert not state['pending_confirmation']


def test_same_archive_old_proof_and_wrong_host_negative_are_not_qualified(legacy):
    x = legacy; qualified_config(x.config); row = copy.deepcopy(x.rows[0]); archive_hash = 'c' * 64
    row['_novelty_archive_sha256'] = archive_hash
    assert not continuation.novelty_proof_reusable(row, archive_hash, x.config)
    row['novelty']['runtime_sha256'] = QUALIFIED['identity_sha256']
    assert continuation.novelty_proof_reusable(row, archive_hash, x.config)
    merged = {'native_host_status': {'master': {'novelty_runtime_sha256': 'a' * 64},
                                    'slave': {'novelty_runtime_sha256': 'a' * 64}}}
    assert not continuation.qualified_novelty_result(merged, x.config)
    merged['native_host_status']['slave']['novelty_runtime_sha256'] = 'b' * 64
    assert continuation.qualified_novelty_result(merged, x.config)


@pytest.mark.parametrize('running', [True, False])
def test_old_active_owner_is_polled_and_never_relaunched(legacy, monkeypatch, running):
    import base64
    from nanobody_opt import pr100_live_controller as c
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    state['active'] = {'number': 2, 'phase': 'native', 'native_unresolved': True}
    directory = x.local / 'attempts/00002/native-hosts/master'
    request = {'spec': {'epoch': x.replay.challenge['epoch']}, 'directory': '/old-release/owned/native',
               'files': {'input.json': base64.b64encode(json.dumps(x.replay.validity).encode()).decode()}}
    write(directory / 'native-request.json', {'host': 'master', 'request': request})
    original = copy.deepcopy(state); calls = []
    def remote(_cfg, host, supplied):
        calls.append(supplied)
        assert host == 'master' and supplied['operation'] == 'poll'
        assert supplied['directory'] == request['directory'] and 'files' not in supplied
        return {'running': running, 'receipt': {'cleanup': {
            'remaining_descendants': 0, 'fixed_point_reached': True}},
            'files': {'result.json': base64.b64encode(json.dumps(x.replay.validity).encode()).decode()}}
    monkeypatch.setattr(c, 'remote', remote)
    assert continuation.recover_owned_attempt_for_novelty(state, x.local, x.config, x.chain, 1e20) is not running
    assert len(calls) == 1
    if running:
        assert state == original
    else:
        assert state['active'] is None
        assert json.loads((directory / 'native/result.json').read_text()) == x.replay.validity
        assert continuation.restore_for_qualified_novelty(state, x.local, x.config)


def test_same_archive_old_proofs_force_recheck_not_native_or_scoring(legacy, monkeypatch):
    from nanobody_opt import pr100_live_controller as c
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    qualified_config(x.config); write(x.local / 'continuation.json', state); calls = []
    archive_hash = hashlib.sha256(x.replay.archive).hexdigest()
    def phase(config, _challenge, name, files, *_a, **_kw):
        calls.append(name)
        assert name == 'novelty'
        assert config['_phase_operation'] != archive_hash
        value = json.loads(files['input.json'])
        value['novelty_runtime_sha256'] = QUALIFIED['identity_sha256']
        for row in value['candidates']:
            row['novelty']['runtime_sha256'] = QUALIFIED['identity_sha256']
        return {'result.json': json.dumps(value).encode()}
    monkeypatch.setattr(c, 'run_phase', phase)
    outcome = c.run_epoch(x.config, x.chain, c.clock_state(x.chain))
    assert outcome['status'] == 'published' and calls == ['novelty']
    stored = json.loads((x.local / 'continuation.json').read_text())
    assert stored['score_rows'] == state['score_rows']
    assert all(r['novelty']['runtime_sha256'] == QUALIFIED['identity_sha256'] for r in stored['native_rows'])


def test_migration_ignores_checkpoint_outside_original_native_input(legacy):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    forged = checkpoint(x.rows[1], x.replay.challenge)
    clean_native_artifact(x.local / 'attempts/00001', {**x.replay.validity, 'candidates': []}, [forged],
                          input_rows=[x.rows[0]])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert state['pending_confirmation'] == []


def test_unknown_invalidation_is_not_erased_by_older_positive_checkpoint(legacy):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    row = x.rows[1]; digest = rules.sequence_hash(row['sequence'])
    state['invalidated_native_hashes'] = [digest]
    clean_native_artifact(x.local / 'attempts/00001', {**x.replay.validity, 'candidates': [row]},
                          [checkpoint(row, x.replay.challenge)])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert not state['pending_confirmation']
    assert digest in state['invalidated_native_hashes']
    assert row['sequence'] not in {r['sequence'] for r in state['native_rows']}


def test_restored_complete_checkpoint_classifies_with_zero_new_draws(legacy, monkeypatch):
    from nanobody_opt import pr100_live_worker as worker, pr100_filter
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    full = checkpoint(x.rows[1], x.replay.challenge)
    clean_native_artifact(x.local / 'attempts/00001', {**x.replay.validity, 'candidates': []}, [full])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    restored = state['pending_confirmation'][0]
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *_a: restored['native_resume']['identity'])
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    directory = x.local / 'inert-resume'
    write(directory / 'input.json', {**x.replay.validity, 'candidates': [restored]})
    result = worker.native_phase({'challenge': x.replay.challenge, 'native_tools': '/inert',
        'workers': 1, 'repeats': 3}, directory,
        measure=lambda *_a, **_kw: pytest.fail('completed native observations were repeated'))
    assert result['fresh_evaluations'] == result['repeat_evaluations'] == 0
    assert result['carried_cpu_observations'] == 3 and len(result['candidates']) == 1


def test_migration_recovers_completed_raw_scoring_before_state_write(legacy):
    from tests.test_pr100_epoch_continuation import score_files
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    row = x.rows[1]
    operation = x.local / 'attempts/00002/score-batches/old-cohort'
    files = score_files(x.replay, [r for r in x.replay.data['rows'] if r['sequence'] == row['sequence']])
    for name, raw in files.items():
        path = operation / 'score' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    write(operation / 'score-supervisor.json', {'cleanup': {
        'remaining_descendants': 0, 'fixed_point_reached': True}})
    identity = copy.deepcopy(state['scoring_identity']); old_scores = copy.deepcopy(state['score_rows'])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert len(state['score_rows']) == 2 and state['scoring_identity'] == identity
    assert all(state['score_rows'][key] == value for key, value in old_scores.items())


def test_completed_current_runtime_novelty_negative_is_not_reopened(legacy):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    full = checkpoint(x.rows[1], x.replay.challenge)
    clean_native_artifact(x.local / 'attempts/00001', {**x.replay.validity, 'candidates': [],
        'native_classification_complete': True, 'novelty_runtime_sha256': QUALIFIED['identity_sha256']}, [full])
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    assert state['pending_confirmation'] == []


def test_cached_clean_owner_without_native_result_still_consumes_uncertain_admissions(legacy, monkeypatch):
    import base64
    from nanobody_opt import pr100_live_controller as c
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    state['active'] = {'number': 2, 'phase': 'native'}
    directory = x.local / 'attempts/00002/native-hosts/master'
    write(directory / 'native-request.json', {'host': 'master', 'request': {
        'spec': {'epoch': x.replay.challenge['epoch']}, 'directory': '/old/owned',
        'files': {'input.json': base64.b64encode(json.dumps(x.replay.validity).encode()).decode()}}})
    write(directory / 'native-supervisor.json', {'reason': 'nonzero_exit', 'cleanup': {
        'remaining_descendants': 0, 'fixed_point_reached': True}})
    write(directory / 'native-collected.json', {'files': []})
    monkeypatch.setattr(c, 'remote', lambda *_a, **_kw: pytest.fail('clean owner was repolled'))
    assert continuation.recover_owned_attempt_for_novelty(state, x.local, x.config, x.chain, 1e20)
    expected = {rules.sequence_hash(row['sequence']) for row in x.rows}
    assert set(state['migration_uncertain_native_admissions']) == expected
    assert expected <= set(state['native_attempted_hashes']) and state['active'] is None


@pytest.mark.parametrize('forged_peer_refund', [False, True])
def test_migration_reconciles_only_bound_cleanup_verified_fresh_budget_proofs(legacy, forged_peer_refund):
    from nanobody_opt import pr100_fresh_budget as budget
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config); qualified_config(x.config)
    state.update(native_rows=[], native_attempted_hashes=[], pending_native=list(x.rows))
    ledger = budget.ensure(state, x.config); budget.reserve(ledger, 2, x.rows)
    first, second = [rules.sequence_hash(row['sequence']) for row in x.rows]
    master = x.local / 'attempts/00002/native-hosts/master'
    clean_native_artifact(master, {**x.replay.validity, 'candidates': [],
        'fresh_admitted_sequence_hashes': [first],
        'fresh_never_started_sequence_hashes': [second] if forged_peer_refund else []},
        input_rows=x.rows[:1], preserve_admission_proof=True)
    if not forged_peer_refund:
        slave = x.local / 'attempts/00002/native-hosts/slave'
        clean_native_artifact(slave, {**x.replay.validity, 'candidates': [],
            'fresh_admitted_sequence_hashes': [], 'fresh_never_started_sequence_hashes': [second],
            'deferred_candidates': x.rows[1:]}, input_rows=x.rows[1:], preserve_admission_proof=True)
        value = json.loads((slave / 'native/result.json').read_text()); value['attempted_sequence_hashes'] = []
        write(slave / 'native/result.json', value)
    assert continuation.restore_for_qualified_novelty(state, x.local, x.config)
    summary = budget.summary(ledger)
    assert summary['capacity_used'] == (2 if forged_peer_refund else 1)
    assert summary['admitted_unique'] == (0 if forged_peer_refund else 1)
    if not forged_peer_refund:
        assert state['pending_native'] == x.rows[1:]


def test_real_annotation_child_and_descendant_remain_under_phase_deadline(tmp_path):
    """Inert processes exercise the real delegate plus existing outer owner."""
    import os
    import subprocess
    import time
    from tests.test_pr100_live_supervisor import sup, spec_for
    wrapper = tmp_path / 'inert-annotation-python'
    pids = tmp_path / 'owned-pids.json'
    wrapper.write_text('#!' + sys.executable + '\n' +
        'import json,os,signal,subprocess,sys,time\n'
        'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
        'child=subprocess.Popen([sys.executable,"-c",'
        '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])\n'
        + f'open({str(pids)!r},"w").write(json.dumps([os.getpid(),child.pid]))\n'
        + 'time.sleep(60)\n')
    wrapper.chmod(0o700)
    archive = tmp_path / 'archive.csv'; archive.write_text('sequence\nAAAA\n')
    saved = tmp_path / 'native-confirmations.json'; saved.write_text('{"durable_before_novelty":true}')
    before = saved.read_bytes()
    qualified = {**QUALIFIED, 'python': str(wrapper), 'path': os.path.dirname(sys.executable)}
    program = ('from nanobody_opt.pr100_novelty_runtime import check\n'
               + f'check([{{"sequence":"BBBB"}}],{str(archive)!r},runtime={qualified!r})\n')
    spec = spec_for(tmp_path, [sys.executable, '-B', '-c', program], wall_seconds=1.0,
                    deadline_unix=time.time() + 10)
    spec['environment']['PYTHONPATH'] = str(runtime.ROOT)
    specification = tmp_path / 'spec.json'; specification.write_text(json.dumps(spec))
    executed = subprocess.run([sys.executable, '-B', str(runtime.ROOT / 'scripts/pr100_live_supervisor.py'),
                               '--spec', str(specification)], capture_output=True, text=True, timeout=20)
    assert executed.returncode == 1
    receipt = json.loads((tmp_path / 'supervisor-receipt.json').read_text())
    assert pids.is_file() and receipt['reason'] == 'wall_seconds'
    if sys.platform == 'linux':
        assert receipt['cleanup']['remaining_descendants'] == 0
        assert receipt['cleanup']['fixed_point_reached'] is True
    assert saved.read_bytes() == before
    for pid in json.loads(pids.read_text()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
