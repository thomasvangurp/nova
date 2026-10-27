"""The epoch cap limits new admissions while existing evidence can finish."""
import json
from types import SimpleNamespace
import pytest
from nanobody_opt import rules, pr100_filter, pr100_live_worker as worker
from nanobody_opt import pr100_fresh_budget as budget, pr100_native_dispatch as dispatch
from nanobody_opt import pr100_epoch_continuation as continuation, pr100_live_controller as controller
from tests.test_pr100_native_screen_integration import CHALLENGE, profile
from tests.test_pr100_epoch_continuation import legacy, score_files, write
from tests.test_pr100_live_handoff_contract import replay

def row(name): return {'sequence': name, 'pr100_annotation': {}}
def digest(name): return rules.sequence_hash(name)
def ledger(limit=4800): return budget.ensure({}, {'epoch_fresh_native_limit': limit})


@pytest.mark.parametrize('invalid', [0, -1, True, 4801, 2.5, None])
def test_invalid_epoch_limit_rejected(invalid, tmp_path, monkeypatch):
    from tests.test_pr100_live_controller import config
    monkeypatch.setattr(controller, 'verify_build_stamp', lambda _: None)
    with pytest.raises(ValueError, match='fresh native'):
        controller.validate_config({**config(tmp_path), 'epoch_fresh_native_limit': invalid})


def test_generation_is_free_but_legacy_attempts_cannot_be_refunded():
    state = {'generated_hashes': [str(n) for n in range(8000)],
             'native_attempted_hashes': [str(n) for n in range(4800)]}
    value = budget.ensure(state, {})
    assert budget.summary(value) == {'limit': 4800, 'admitted_unique': 0,
        'legacy_charged_unique': 4800, 'reserved_uncertain_unique': 0,
        'capacity_used': 4800, 'remaining': 0, 'legacy_uncertainty': True}
    assert budget.summary(budget.ensure({'generated_hashes': state['generated_hashes']}, {}))['remaining'] == 4800


def test_4799_boundary_reserves_one_fresh_and_all_confirmations_without_replanning():
    value = budget.ensure({'native_attempted_hashes': [str(n) for n in range(4799)]}, {})
    carry = {**row('carry'), 'native_resume': {'measurements': [{}]}}
    rows = [carry, row('A'), row('B')]
    assert budget.reserve(value, 1, rows) == rows[:2]
    assert budget.summary(value)['remaining'] == 0
    assert budget.reserve(value, 1, rows + [row('C')]) == rows[:2]
    assert budget.planned_fresh_count(value, 1) == 1
    assert budget.reserve(value, 2, [row('D'), carry]) == [carry]


def test_only_cleanup_verified_nonstarts_release_capacity():
    value = ledger(2); rows = [row('A'), row('B')]; budget.reserve(value, 1, rows)
    result = {'fresh_admitted_sequence_hashes': [digest('A')],
              'fresh_never_started_sequence_hashes': [digest('B')], 'deferred_candidates': [rows[1]]}
    budget.reconcile(value, 1, result, cleanup_verified=False)
    assert budget.summary(value)['remaining'] == 0 and budget.summary(value)['admitted_unique'] == 1
    budget.reconcile(value, 1, result, cleanup_verified=True)
    assert budget.summary(value)['remaining'] == 1
    assert budget.reserve(value, 2, [row('A'), row('B'), row('C')]) == [row('B')]


def test_inconsistent_never_started_proof_cannot_refund_prior_admission():
    value = ledger(1); budget.reserve(value, 1, [row('A')])
    budget.reconcile(value, 1, {'fresh_admitted_sequence_hashes': [digest('A')]}, cleanup_verified=False)
    with pytest.raises(ValueError, match='conflict'):
        budget.reconcile(value, 1, {'verified_fresh_never_started_sequence_hashes': [digest('A')]}, cleanup_verified=True)
    assert budget.summary(value)['admitted_unique'] == 1
    assert budget.summary(value)['remaining'] == 0


def test_epoch_limit_cannot_be_raised_mid_epoch():
    state = {}; value = budget.ensure(state, {'epoch_fresh_native_limit': 100})
    budget.ensure(state, {'epoch_fresh_native_limit': 4800})
    assert value['limit'] == 100


def setup_worker(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *a: {'resume_eligible': True})
    monkeypatch.setattr(rules, 'validate_evidence', lambda *a: None)
    write(tmp_path / 'input.json', {'challenge': CHALLENGE, 'target': CHALLENGE['target'],
        'epoch': CHALLENGE['epoch'], 'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1, 'candidates': rows})
    return {'challenge': CHALLENGE, 'native_tools': '/inert-tools', 'workers': 2,
            'max_workers': 2, 'repeats': 3, 'fresh_admission_limit': 2}


def test_worker_stops_fresh_calls_at_limit_and_reports_never_started(tmp_path, monkeypatch):
    rows = [row(str(n)) for n in range(5)]; job = setup_worker(tmp_path, monkeypatch, rows); calls = []
    def measure(seqs, *_a, **kw):
        calls.extend(seqs); return [profile(s, 160) for s in seqs], 0, 1
    result = worker.native_phase(job, tmp_path, measure=measure)
    assert calls == ['0', '1']
    assert result['fresh_evaluations'] == result['fresh_admitted_unique'] == 2
    assert result['repeat_evaluations'] == 0 and result['not_run'] == 3
    assert set(result['fresh_never_started_sequence_hashes']) == {digest(str(n)) for n in range(2, 5)}


def test_interrupted_calls_stay_charged_without_fabricated_completions(tmp_path, monkeypatch):
    job = setup_worker(tmp_path, monkeypatch, [row('A'), row('B'), row('C')])
    with pytest.raises(TimeoutError):
        worker.native_phase(job, tmp_path, measure=lambda *a, **kw: (_ for _ in ()).throw(TimeoutError('inert interruption')))
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['fresh_admitted_unique'] == 2 and result['fresh_evaluations'] == 0
    assert set(result['fresh_admitted_sequence_hashes']) == {digest('A'), digest('B')}


def test_zero_fresh_capacity_still_completes_prior_confirmations(tmp_path, monkeypatch):
    carry = {**row('AAAA'), 'native_resume': {'schema': 'pr100-native-confirmation-checkpoint-v1',
        'sequence_hash': digest('AAAA'), 'identity': {'resume_eligible': True},
        'measurements': [profile('AAAA')], 'required_cpu_observations': 3}}
    job = setup_worker(tmp_path, monkeypatch, [row('BBBB'), carry]); job['fresh_admission_limit'] = 0
    calls = []
    def measure(seqs, *_a, **kw):
        calls.extend(seqs); return [profile(s) for s in seqs], 0, 1
    result = worker.native_phase(job, tmp_path, measure=measure)
    assert calls == ['AAAA', 'AAAA']
    assert result['repeat_evaluations'] == 2 and result['fresh_evaluations'] == 0
    assert result['fresh_admitted_sequence_hashes'] == []
    assert len(result['candidates']) == 1 and result['not_run'] == 1


def test_dispatch_refuses_two_hosts_exceeding_one_shared_reservation(tmp_path):
    incoming = {'challenge': CHALLENGE, 'candidates': [row('A'), row('B')]}
    with pytest.raises(ValueError, match='shared fresh'):
        dispatch.run({'worker': {'workers': 1, 'fresh_admission_limit': 1}, 'native_hosts': ['master', 'slave']},
            CHALLENGE, {'input.json': json.dumps(incoming).encode()}, 9999999999, object(), tmp_path,
            run_phase=lambda *a, **kw: pytest.fail('over-budget input reached a host'))


def test_two_host_recovery_charges_starts_and_holds_uncertain_peer(tmp_path):
    value = ledger(4); rows = [row(name) for name in ('A', 'B', 'C', 'D')]
    selected = budget.reserve(value, 1, rows)
    incoming = {'challenge': CHALLENGE, 'candidates': selected}
    config = {'worker': {'workers': 1, 'fresh_admission_limit': 4}, 'native_hosts': ['master', 'slave']}
    recovered = [False]; allocations = []
    def phase(cfg, challenge, name, files, stop, chain, local, **kw):
        assigned = json.loads(files['input.json'])['candidates']; host = cfg['_native_host']
        allocations.append((host, cfg['worker']['fresh_admission_limit']))
        assert cfg['worker']['fresh_admission_limit'] == len(assigned) == 2
        if host == 'master' or recovered[0]:
            write(local / 'native-supervisor.json', {'reason': 'completed',
                'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
        started = assigned[:1] if host == 'master' else assigned if recovered[0] else []
        deferred = assigned[len(started):]
        result = {**incoming, 'status': 'complete', 'novelty_checked': True, 'candidates': [],
            'attempted_sequence_hashes': [digest(r['sequence']) for r in started],
            'fresh_admitted_sequence_hashes': [digest(r['sequence']) for r in started],
            'fresh_never_started_sequence_hashes': [digest(r['sequence']) for r in deferred],
            'deferred_candidates': deferred}
        return {'result.json': json.dumps(result).encode()}
    payload = {'input.json': json.dumps(incoming).encode()}
    result = json.loads(dispatch.run(config, CHALLENGE, payload, 9999999999, object(), tmp_path,
                                    run_phase=phase)['result.json'])
    assert result['unresolved_native_hosts'] == ['slave']
    budget.reconcile(value, 1, result, cleanup_verified=False)
    assert budget.summary(value)['admitted_unique'] == 1
    assert budget.summary(value)['reserved_uncertain_unique'] == 2
    assert budget.summary(value)['remaining'] == 1
    assert budget.reserve(value, 1, rows) == rows  # exact plan, no new reservations
    recovered[0] = True
    result = json.loads(dispatch.run(config, CHALLENGE, payload, 9999999999, object(), tmp_path,
                                    run_phase=phase)['result.json'])
    budget.reconcile(value, 1, result, cleanup_verified=True)
    assert budget.summary(value)['admitted_unique'] == 3
    assert budget.summary(value)['remaining'] == 1 and not result['unresolved_native_hosts']
    assert budget.reserve(value, 2, [row('C'), row('E')]) == [row('C')]


def test_at4800_controller_stops_new_native_work_without_touching_published_fleet(legacy, monkeypatch):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['native_attempted_hashes'] = [digest(x.rows[0]['sequence'])] + [str(n) for n in range(4799)]
    state['pending_native'] = [x.rows[1]]
    write(x.local / 'continuation.json', state)
    publication = __import__('pathlib').Path(x.config['publication_path']).read_bytes()
    monkeypatch.setattr(controller, 'run_phase', lambda *a, **kw: pytest.fail('budget exhausted but new work launched'))
    result = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert result['status'] == 'fresh_budget_exhausted'
    assert result['fresh_native_budget']['capacity_used'] == 4800
    assert __import__('pathlib').Path(x.config['publication_path']).read_bytes() == publication


def test_at4800_unscored_valid_candidate_still_scores_and_upgrades_fleet(legacy, monkeypatch):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['native_rows'].append({**x.rows[1], '_novelty_archive_sha256': x.replay.validity['archive_sha256']})
    state['native_attempted_hashes'] = [digest(r['sequence']) for r in x.rows] + [str(n) for n in range(4798)]
    write(x.local / 'continuation.json', state)
    calls = []
    def phase(cfg, challenge, name, files, *_a, **kw):
        calls.append(name); assert name == 'score'
        assert json.loads(files['input.json'])['sequences'] == [x.rows[1]['sequence']]
        return score_files(x.replay, [r for r in x.replay.data['rows'] if r['sequence'] == x.rows[1]['sequence']])
    monkeypatch.setattr(controller, 'run_phase', phase)
    result = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert result['status'] == 'published' and calls == ['score']
    assert result['fresh_native_budget']['remaining'] == 0
    fleet = json.loads(__import__('pathlib').Path(x.config['publication_path']).read_text())
    assert set(fleet['sequence_pool']) == {r['sequence'] for r in x.rows}


def test_4799_controller_persists_one_slot_before_dispatch(legacy, monkeypatch):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['native_attempted_hashes'] = [digest(x.rows[0]['sequence'])] + [str(n) for n in range(4798)]
    state['pending_native'] = [x.rows[1], row('inert-extra')]
    write(x.local / 'continuation.json', state)
    def phase(cfg, challenge, name, files, stop, chain, local, **kw):
        assert name == 'native' and cfg['worker']['fresh_admission_limit'] == 1
        incoming = json.loads(files['input.json']); assert len(incoming['candidates']) == 1
        saved = continuation.read(x.local / 'continuation.json')
        assert budget.summary(saved['fresh_native_budget'])['remaining'] == 0
        write(local / 'native-supervisor.json', {'reason': 'completed',
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
        return {'result.json': json.dumps({**incoming, 'status': 'complete', 'novelty_checked': True,
            'candidates': [], 'fresh_admitted_sequence_hashes': [digest(x.rows[1]['sequence'])],
            'attempted_sequence_hashes': [digest(x.rows[1]['sequence'])]}).encode()}
    monkeypatch.setattr(controller, 'run_phase', phase)
    result = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert result['status'] == 'published'
    assert result['fresh_native_budget']['capacity_used'] == 4800
    saved = continuation.read(x.local / 'continuation.json')
    assert saved['pending_native'] == [row('inert-extra')]


def test_legacy_active_saved_input_cannot_relaunch_over_remaining_budget(legacy, monkeypatch):
    x = legacy; state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['native_attempted_hashes'] = [digest(x.rows[0]['sequence'])] + [str(n) for n in range(4798)]
    state['pending_native'] = [x.rows[1], row('inert-extra')]
    state['active'] = {'number': 1, 'phase': 'native', 'proposal_offset': 600}
    write(x.local / 'continuation.json', state)
    write(x.local / 'attempts/00001/native-input.json', {**x.replay.validity, 'candidates': state['pending_native']})
    monkeypatch.setattr(controller, 'run_phase', lambda *a, **kw: pytest.fail('legacy oversized input relaunched'))
    result = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert result['status'] == 'recoverable_failure' and result['error_type'] == 'ValueError'
    saved = continuation.read(x.local / 'continuation.json')
    assert saved['active']['number'] == 1
    assert result['fresh_native_budget']['remaining'] == 0
    assert result['fresh_native_budget']['legacy_uncertainty'] is True
