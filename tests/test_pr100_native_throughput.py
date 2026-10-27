"""More admitted work must preserve full profiles, deadlines and host ownership."""
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from nanobody_opt import pr100_filter, pr100_live_worker as worker, rules
from nanobody_opt import pr100_native_dispatch as dispatch
from tests.test_pr100_native_screen_integration import profile, CHALLENGE
from tests.test_pr100_live_worker import observed_backend


def test_private_executor_admits_more_than32_and_joins_all_threads():
    gate = threading.Barrier(40, timeout=3)
    def call(value):
        gate.wait()
        return value
    async def develop(sequences, concurrency, **kwargs):
        return await asyncio.gather(*(asyncio.to_thread(call, s) for s in sequences))
    values, _, _ = worker.measure_batch(list(range(40)), 40, develop=develop)
    assert values == list(range(40))
    assert not any(t.name.startswith('pr100-native') for t in threading.enumerate())


def test_private_executor_error_cleanup_does_not_leak_threads():
    async def develop(sequences, concurrency, **kwargs):
        await asyncio.gather(*(asyncio.to_thread(lambda: None) for _ in sequences))
        raise RuntimeError('inert profile failure')
    with pytest.raises(RuntimeError): worker.measure_batch(list(range(40)), 40, develop=develop)
    assert not any(t.name.startswith('pr100-native') for t in threading.enumerate())


@pytest.mark.parametrize('adaptive,expected', [(False, 48), (True, 96)])
def test_315s_phase_admits_second_measured99s_batch(adaptive, expected, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'worker_count', lambda _: 48)
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *a: {'resume_eligible': False})
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    sequences = ['A' + str(i) for i in range(144)]
    pr100_filter.atomic(tmp_path / 'input.json', {'challenge': CHALLENGE, 'epoch': CHALLENGE['epoch'],
        'target': CHALLENGE['target'], 'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1,
        'candidates': [{'sequence': s, 'pr100_annotation': {}} for s in sequences]})
    now = [0.]
    def measure(sequences, *_args, **kwargs):
        now[0] += 99
        return [profile(s, 160.) for s in sequences], 0., 99.
    job = {'challenge': CHALLENGE, 'native_tools': '/tools', 'candidate_budget_seconds': 300,
           'phase_budget_seconds': 315, 'workers': 48, 'repeats': 3}
    if adaptive:
        job.update(native_minimum_call_window_seconds=130, native_cleanup_reserve_seconds=10)
    result = worker.native_phase(job, tmp_path, measure=measure, clock=lambda: now[0])
    assert result['fresh_evaluations'] == expected
    assert result['repeat_evaluations'] == 0
    if adaptive:
        assert [a['timeout_seconds'] for a in result['native_admission']['allocations']] == [300, 206]
        assert now[0] == 198 and result['not_run'] == 48


def test_measure_batch_passes_actual_bounded_timeout_to_native_service():
    observed = []
    async def develop(sequences, workers, **kwargs):
        observed.append(kwargs['task_timeout_seconds'])
        return [{} for _ in sequences]
    worker.measure_batch(['A'], 1, develop=develop, timeout_seconds=206)
    assert observed == [206]


def test_capacity_partition_keeps_carried_confirmations_on_their_host():
    rows = [{'sequence': str(i)} for i in range(128)]
    groups, blocked = dispatch.partition(rows, ['master', 'slave'], {'master': 96, 'slave': 32})
    assert [len(groups[h]) for h in ('master', 'slave')] == [96, 32] and not blocked
    row = {'sequence': 'carry', 'native_resume': {'execution_host': 'slave'}}
    groups, blocked = dispatch.partition([row], ['master', 'slave'], {'master': 96, 'slave': 32})
    assert groups['slave'] == [row]
    groups, blocked = dispatch.partition([row], ['master'], {'master': 96})
    assert not groups['master'] and blocked[0]['continuation_blocked'] == 'native_checkpoint_host_unavailable'


def test_clean_host_failure_preserves_peer_valid_rows(tmp_path):
    rows = [{'sequence': 'A'}, {'sequence': 'B'}]
    payload = {'input.json': json.dumps({'challenge': CHALLENGE, 'candidates': rows}).encode()}
    config = {'worker': {'workers': 1}, 'native_hosts': ['master', 'slave']}
    gate = threading.Barrier(2, timeout=2)
    def phase(cfg, challenge, phase, files, stop, chain, local, **kwargs):
        gate.wait()
        pr100_filter.atomic(local / 'native-supervisor.json', {'reason': 'completed',
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
        if cfg['_native_host'] == 'slave': raise RuntimeError('inert failed host')
        result = {'challenge': challenge, 'status': 'complete', 'novelty_checked': True,
                  'candidates': [{'sequence': 'A'}], 'fresh_evaluations': 1,
                  'attempted_sequence_hashes': [rules.sequence_hash('A')]}
        return {'result.json': json.dumps(result).encode()}
    result = dispatch.run(config, CHALLENGE, payload, time.time()+1000, object(), tmp_path, run_phase=phase)
    value = json.loads(result['result.json'])
    assert value['candidates'] == [{'sequence': 'A'}]
    assert value['fresh_evaluations'] == 1 and value['unresolved_native_hosts'] == []


def test_ambiguous_host_failure_retains_peer_and_marks_owner_pending(tmp_path):
    payload = {'input.json': json.dumps({'challenge': CHALLENGE,
        'candidates': [{'sequence': 'A'}, {'sequence': 'B'}]}).encode()}
    def phase(cfg, challenge, phase, files, stop, chain, local, **kwargs):
        if cfg['_native_host'] == 'slave': raise RuntimeError('inert lost transport')
        pr100_filter.atomic(local / 'native-supervisor.json', {'reason': 'completed',
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
        return {'result.json': json.dumps({'challenge': challenge, 'status': 'complete',
            'novelty_checked': True, 'candidates': [{'sequence': 'A'}]}).encode()}
    output = dispatch.run({'worker': {'workers': 1}, 'native_hosts': ['master', 'slave']},
        CHALLENGE, payload, time.time()+1000, object(), tmp_path, run_phase=phase)
    result = json.loads(output['result.json'])
    assert result['unresolved_native_hosts'] == ['slave']
    assert result['candidates'] == [{'sequence': 'A'}]
    assert rules.sequence_hash('B') not in result['attempted_sequence_hashes']


@pytest.mark.parametrize('bad', ['json', 'challenge', 'epoch', 'target', 'archive_sha256'])
def test_bad_host_artifact_does_not_erase_valid_peer(tmp_path, bad):
    incoming = {'challenge': CHALLENGE, 'epoch': CHALLENGE['epoch'], 'target': CHALLENGE['target'],
        'archive_sha256': 'a'*64, 'candidates': [{'sequence': 'A'}, {'sequence': 'B'}]}
    def phase(cfg, challenge, phase, files, stop, chain, local, **kwargs):
        pr100_filter.atomic(local / 'native-supervisor.json', {'reason': 'completed',
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
        value = {**incoming, 'status': 'complete', 'novelty_checked': True,
                 'candidates': [{'sequence': 'A' if cfg['_native_host'] == 'master' else 'B'}],
                 'fresh_evaluations': 1}
        if cfg['_native_host'] == 'slave':
            if bad == 'json': return {'result.json': b'invalid-json'}
            value[bad] = 'mismatched'
        return {'result.json': json.dumps(value).encode()}
    output = dispatch.run({'worker': {'workers': 1}, 'native_hosts': ['master', 'slave']},
        CHALLENGE, {'input.json': json.dumps(incoming).encode()}, time.time()+1000,
        object(), tmp_path, run_phase=phase)
    result = json.loads(output['result.json'])
    assert result['candidates'] == [{'sequence': 'A'}]
    assert result['fresh_evaluations'] == 1
    assert result['attempted_sequence_hashes'] == []
    assert result['blocked_confirmations'][0]['sequence'] == 'B'


def test_changed_routing_cannot_start_additional_host_while_retrying(tmp_path):
    config = {'worker': {'workers': 1}, 'native_hosts': ['master']}
    payload = {'input.json': json.dumps({'challenge': CHALLENGE, 'candidates': [{'sequence': 'A'}]}).encode()}
    dispatch.run(config, CHALLENGE, payload, time.time()+1000, object(), tmp_path,
                 run_phase=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('ambiguous')))
    with pytest.raises(ValueError, match='routing'):
        dispatch.run({**config, 'native_hosts': ['master', 'slave']}, CHALLENGE, payload,
                     time.time()+1000, object(), tmp_path,
                     run_phase=lambda *a, **kw: pytest.fail('changed routing reached host'))


def test_shortened_timeout_reaches_original_manager_configuration(observed_backend):
    async def analyze(sequence): return {'passed': True, 'sequence': sequence}
    observed_backend.analyze = analyze
    config = pr100_filter.helper('developability').developability_config()
    asyncio.run(pr100_filter.develop(['A'], 1, task_timeout_seconds=130))
    assert 129 < config.async_config.task_timeout <= 130
    assert observed_backend.shutdowns == 1


def checkpoint_input(tmp_path, monkeypatch, *, mismatch=False):
    identity = {'resume_eligible': True, 'runtime': 'current'}
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *a: identity)
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    row = {'sequence': 'AAAA', 'pr100_annotation': {}, 'native_resume': {
        'schema': 'pr100-native-confirmation-checkpoint-v1', 'sequence_hash': rules.sequence_hash('AAAA'),
        'identity': {'resume_eligible': True, 'runtime': 'other'} if mismatch else identity,
        'measurements': [profile('AAAA'), profile('AAAA')], 'required_cpu_observations': 3,
        'execution_host': 'slave'}}
    pr100_filter.atomic(tmp_path / 'input.json', {'challenge': CHALLENGE, 'epoch': CHALLENGE['epoch'],
        'target': CHALLENGE['target'], 'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1,
        'candidates': [row, {'sequence': 'BBBB', 'pr100_annotation': {}}] if mismatch else [row]})
    return row


def test_wrong_runtime_checkpoint_is_held_while_fresh_peer_runs(tmp_path, monkeypatch):
    row = checkpoint_input(tmp_path, monkeypatch, mismatch=True)
    calls = []
    def measure(sequences, *_a, **kw):
        calls.extend(sequences)
        return [profile(s, 160.) for s in sequences], 0., 1.
    result = worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/tools',
        'workers': 1, 'repeats': 3, 'native_host': 'master'}, tmp_path, measure=measure)
    assert calls == ['BBBB']
    assert result['blocked_confirmations'][0]['native_resume'] == row['native_resume']
    assert rules.sequence_hash('AAAA') not in result['terminal_native_outcomes']
    assert result['fresh_evaluations'] == 1 and result['repeat_evaluations'] == 0


def test_inconclusive_repeat_preserves_two_prior_cpu_confirmations(tmp_path, monkeypatch):
    row = checkpoint_input(tmp_path, monkeypatch)
    result = worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/tools', 'workers': 1,
        'repeats': 3, 'native_host': 'slave'}, tmp_path,
        measure=lambda seqs, *a, **kw: ([{'error': True, 'error_kind': 'timeout'}], 0., 130.))
    assert result['candidates'] == [] and result['terminal_native_outcomes'] == {}
    assert result['repeat_evaluations'] == 1 and result['fresh_evaluations'] == 0
    saved = json.loads((tmp_path / 'native-confirmations.json').read_text())['records'][0]
    assert saved['native_resume']['measurements'] == row['native_resume']['measurements']
    assert saved['native_resume']['execution_host'] == 'slave'
    assert result['deferred_confirmations'][0]['native_resume']['measurements'] == row['native_resume']['measurements']


def test_host_native_overrides_do_not_change_annotation_environment(tmp_path):
    from tests.test_pr100_live_controller import config, challenge, clock
    from nanobody_opt import pr100_live_controller as c
    cfg = config(tmp_path)
    cfg.update(_native_host='slave', native_worker_overrides={'slave': {'workers': 32, 'cpu_slots': 32}})
    cfg['hosts']['slave'].update(native_path='/native/bin:/usr/bin', native_python_path='/native-shim')
    host, request = c.phase_request(cfg, challenge(), 'native', {}, time.time()+2000, clock(), cap=600)
    import base64
    job = json.loads(base64.b64decode(request['files']['job.json']))
    assert host == 'slave' and job['workers'] == 32 and job['cpu_slots'] == 32
    assert request['spec']['environment']['PATH'] == '/native/bin:/usr/bin'
    assert request['spec']['environment']['PYTHONPATH'] == '/release:/native-shim'
    _, annotation = c.phase_request(cfg, challenge(), 'sequence', {}, time.time()+2000, clock(), cap=100)
    assert annotation['spec']['environment']['PATH'] == '/bin'
    assert annotation['spec']['environment']['PYTHONPATH'] == '/release:/extra'
    assert annotation['spec']['cpu_slots'] == 4


def test_elapsed_host_wall_budget_retries_poll_only(tmp_path):
    from tests.test_pr100_continuation_lifecycle import config, Chain, invoke, terminal
    cfg = {**config(tmp_path), '_bounded_phase_collection': True}
    chain = Chain()
    with pytest.raises(RuntimeError):
        invoke(tmp_path, cfg, chain, lambda *a: (_ for _ in ()).throw(RuntimeError('transport')))
    intent = tmp_path / 'attempt/native-request.json'
    saved = json.loads(intent.read_text())
    saved['created_at_unix'] -= saved['request']['spec']['wall_seconds'] + 40
    intent.write_text(json.dumps(saved))
    calls = []
    def recovered(_cfg, _host, request):
        calls.append(request)
        return terminal()
    invoke(tmp_path, cfg, chain, recovered)
    assert calls[0]['operation'] == 'poll'
    assert calls[0]['transport_deadline_unix'] <= time.time()+20


def test_saved_runtime_binding_refusal_never_reuses_cached_host_result(tmp_path):
    rows = [{'sequence': 'A'}]
    incoming = {'challenge': CHALLENGE, 'candidates': rows}
    child = tmp_path / 'native-hosts/master'
    pr100_filter.atomic(child / 'native/result.json', {**incoming, 'status': 'complete', 'novelty_checked': True})
    pr100_filter.atomic(child / 'native-supervisor.json', {'reason': 'completed',
        'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
    output = dispatch.run({'worker': {'workers': 1}, 'native_hosts': ['master']}, CHALLENGE,
        {'input.json': json.dumps(incoming).encode()}, time.time()+1000, object(), tmp_path,
        run_phase=lambda *a, **kw: (_ for _ in ()).throw(ValueError('binding mismatch')))
    value = json.loads(output['result.json'])
    assert value['candidates'] == [] and value['attempted_sequence_hashes'] == []
    assert value['blocked_confirmations'][0]['sequence'] == 'A'
