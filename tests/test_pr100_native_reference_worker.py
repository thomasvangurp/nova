"""Worker orchestration with synthetic authenticated draws, never hardware proof.

Real async batch/refill scheduling, journals, counters, checkpoint export and Pi
evidence validation run unchanged. Only native execution/authentication and host
resource discovery are replaced; actual runner/receipt tests live separately.
"""
import asyncio
from collections import Counter
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from nanobody_opt import pr100_live_worker as worker, pr100_filter, rules
from nanobody_opt import pr100_native_reference as real_native
from test_pr100 import EPOCH, SEQ, evidence as legacy_evidence
from test_pr100_fleet_probability_integration import POLICY, authenticated_draws


MODES = ('barriers', 'interleaved', 'refill')
RESUME = {'resume_eligible': True, 'backend': 'native_reference',
          'runtime_identity_sha256': 'b' * 64, 'registration_sha256': 'd' * 64}


@pytest.fixture
def native_worker(monkeypatch, tmp_path, authenticated_draws):
    helper = sys.modules['nanobody_opt.pr100_native_reference']
    helper.Inconclusive = real_native.Inconclusive
    helper.BACKEND = real_native.BACKEND
    helper.settings_for = lambda job: ({'registration_sha256': 'd' * 64,
        'registration': {'policy': {'preparation_threads': 1}, 'runtime_sha256': 'b' * 64},
        'gpus': [0], 'workers_per_gpu': 4} if job.get('native_reference', {}).get('enabled') else None)
    monkeypatch.setattr(worker, 'worker_count', lambda _job: 4)
    monkeypatch.setattr(worker, 'available_cores', lambda: 64)
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *_args: copy.deepcopy(RESUME))
    monkeypatch.setattr(pr100_filter, 'helper', lambda _name: SimpleNamespace(tool_paths=None))
    run_number = [0]

    def run(mode, *, count=4, outcomes=None, carried=None, phase_budget=None, step=0):
        run_number[0] += 1
        directory = tmp_path / str(run_number[0]); directory.mkdir()
        sequences = [SEQ[:-1] + amino for amino in 'ACDEFGHIKLMNPQRSTVWY'[:count]]
        calls, values, now = Counter(), {}, [0.]
        outputs = outcomes or {}

        class SyntheticAdapter:
            capacity = 4

            def __init__(self, *_args):
                pass

            async def develop(self, batch, concurrency, *, on_result=None,
                              on_result_error=None, **_kwargs):
                results = []
                for sequence in batch:
                    await asyncio.sleep(0)
                    index = sequences.index(sequence)
                    number = calls[sequence]; calls[sequence] += 1
                    assert calls[sequence] <= 4, 'worker overran bounded repeat plan'
                    choices = outputs.get(index, [90., 90., 90.])
                    item = choices[min(number, len(choices) - 1)]
                    value = (authenticated_draws(f'{run_number[0]}:{index}:{number}', item,
                                                sequence=sequence)
                             if type(item) in (int, float) else copy.deepcopy(item))
                    values.setdefault(sequence, []).append(value)
                    now[0] += step
                    # Match Adapter.develop's callback-error behavior exactly:
                    # diagnostic callback errors are routed to on_result_error.
                    if on_result:
                        try:
                            on_result(sequence, value)
                        except Exception as error:
                            if on_result_error:
                                on_result_error(type(error).__name__)
                            else:
                                raise
                    results.append(value)
                return results

        helper.Adapter = SyntheticAdapter
        rows = []
        challenge = legacy_evidence()['challenge']
        for index, sequence in enumerate(sequences):
            row = {'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                   'pr100_annotation': legacy_evidence(sequence)['annotation']}
            if carried and index in carried:
                supplied = carried[index]
                history = [authenticated_draws(f'carried:{index}:{n}', psh, sequence=sequence)
                           for n, psh in enumerate(supplied['psh'])]
                row['native_resume'] = {
                    'schema': 'pr100-native-confirmation-checkpoint-v1',
                    'sequence_hash': rules.sequence_hash(sequence), 'identity': copy.deepcopy(RESUME),
                    'measurements': history, 'required_cpu_observations': 3,
                    'fleet_policy': copy.deepcopy(POLICY),
                    'failed_first_rescue_reserved': supplied.get('rescue', False)}
            rows.append(row)
        payload = {'challenge': challenge, 'target': challenge['target'], 'epoch': EPOCH,
                   'archive_sha256': 'a' * 64, 'archive_checked_at_unix': 1, 'candidates': rows}
        (directory / 'input.json').write_text(json.dumps(payload))
        job = {'challenge': challenge, 'source_sha': '1' * 40, 'native_tools': '/synthetic-tools',
               'native_reference': {'enabled': True}, 'native_fleet_policy': copy.deepcopy(POLICY),
               'workers': 4, 'repeats': 3, 'allow_marginal': True,
               'candidate_budget_seconds': 30, 'native_minimum_call_window_seconds': 10,
               'native_cleanup_reserve_seconds': 0,
               'native_interleave_confirmations': mode != 'barriers',
               'native_work_conserving': mode == 'refill'}
        if phase_budget is not None:
            job['phase_budget_seconds'] = phase_budget
        result = worker.native_phase(job, directory, clock=lambda: now[0])
        checkpoint = json.loads((directory / 'native-confirmations.json').read_text())
        return SimpleNamespace(result=result, calls=calls, values=values, sequences=sequences,
                               rows=rows, directory=directory, checkpoint=checkpoint)

    return run


@pytest.mark.parametrize('mode', MODES)
def test_failed_first_rescue_exports_complete_history_and_true_passing_primary(native_worker, mode):
    run = native_worker(mode, outcomes={0: [90., 81., 91.]})
    sequence = run.sequences[0]
    assert run.calls[sequence] == 3
    row = next(r for r in run.result['candidates'] if r['sequence'] == sequence)
    assert row['tier'] == 'B' and row['measurements'] == run.values[sequence]
    evidence = row['pr100_evidence']; history = evidence['native_fleet_selection']
    assert history['primary_passing_index'] == 1
    assert evidence['developability'] == run.values[sequence][1]
    assert history['measurements'] == run.values[sequence]
    assert run.result['fresh_evaluations'] == 4 and run.result['repeat_evaluations'] == 2
    assert rules.sequence_hash(sequence) not in run.result['terminal_native_outcomes']
    exported = next(r for r in run.result['native_complete_histories']
                    if r['sequence_hash'] == rules.sequence_hash(sequence))
    assert exported['measurements'] == run.values[sequence]
    rules.validate_evidence(evidence, sequence, evidence['challenge'], rules.pinned_config(EPOCH))


@pytest.mark.parametrize('mode', MODES)
def test_later_failure_is_retained_without_vetoing_supported_passer(native_worker, mode):
    run = native_worker(mode, outcomes={0: [81., 91., 82.]})
    sequence = run.sequences[0]
    row = next(r for r in run.result['candidates'] if r['sequence'] == sequence)
    assert run.calls[sequence] == 3
    assert [v['passed'] for v in row['measurements']] == [True, False, True]
    assert row['pr100_evidence']['native_fleet_selection']['primary_passing_index'] == 0
    assert rules.sequence_hash(sequence) not in run.result['terminal_native_outcomes']


@pytest.mark.parametrize('mode', MODES)
def test_all_failed_stops_at_two_draws_and_rescue_stays_within_quarter_fresh(native_worker, mode):
    run = native_worker(mode, count=8)
    assert run.result['candidates'] == []
    assert all(n <= 2 for n in run.calls.values())
    assert sum(n - 1 for n in run.calls.values()) <= 2
    rescue = run.result['native_first_failure_rescue']
    assert rescue['new_reservations'] <= int(run.result['fresh_admitted_unique'] * .25)
    for sequence, count in run.calls.items():
        if count == 2:
            assert run.result['terminal_native_outcomes'][rules.sequence_hash(sequence)] == 'no_measured_pass'


@pytest.mark.parametrize('mode', MODES)
def test_typed_inconclusive_retains_prior_pass_and_sidejournal_without_fake_failure(native_worker, mode):
    timeout = real_native.failure('reference_profile_inconclusive', started=True)
    run = native_worker(mode, outcomes={0: [81., timeout]})
    sequence = run.sequences[0]
    row = next(r for r in run.result['candidates'] if r['sequence'] == sequence)
    assert run.calls[sequence] == 2
    assert len(row['measurements']) == 1 and row['measurements'][0]['passed']
    errors = [r for r in run.result['native_inconclusive_observations']
              if r['sequence_hash'] == rules.sequence_hash(sequence)]
    assert len(errors) == 1 and errors[0]['after_complete_draws'] == 1
    assert run.result['measurement_outcomes']['inconclusive'] == 1
    assert rules.sequence_hash(sequence) not in run.result['terminal_native_outcomes']
    saved = next(r['native_resume'] for r in run.result['deferred_confirmations']
                 if r['sequence'] == sequence)
    assert len(saved['measurements']) == 1 and saved['measurements'][0]['passed']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('after_pass', [False, True])
def test_serialized_error_is_not_trusted_inconclusive(native_worker, mode, after_pass):
    forged = dict(real_native.failure('reference_profile_inconclusive', started=True))
    with pytest.raises(ValueError, match='unattested_native_result'):
        native_worker(mode, outcomes={0: [81., forged] if after_pass else [forged]})


@pytest.mark.parametrize('mode', MODES)
def test_carried_reserved_failure_gets_one_rescue_without_new_fresh_quota(native_worker, mode):
    run = native_worker(mode, count=1, outcomes={0: [81., 82.]},
                        carried={0: {'psh': [90.], 'rescue': True}})
    sequence = run.sequences[0]
    assert run.calls[sequence] == 2
    assert run.result['fresh_evaluations'] == 0 and run.result['repeat_evaluations'] == 2
    assert run.result['native_first_failure_rescue']['carried_reservations'] == 1
    assert run.result['native_first_failure_rescue']['new_reservations'] == 0
    assert [v['passed'] for v in run.result['candidates'][0]['measurements']] == [False, True, True]


@pytest.mark.parametrize('mode', MODES)
def test_complete_carried_positive_history_is_not_redrawn(native_worker, mode):
    run = native_worker(mode, count=1, carried={0: {'psh': [81., 90., 82.]}})
    assert not run.calls
    assert len(run.result['candidates']) == 1
    assert len(run.result['native_complete_histories'][0]['measurements']) == 3


@pytest.mark.parametrize('mode', MODES)
def test_carried_failed_first_without_reserved_rescue_cannot_create_fresh_quota(native_worker, mode):
    with pytest.raises(ValueError, match='invalid native confirmation checkpoint'):
        native_worker(mode, count=1, carried={0: {'psh': [90.], 'rescue': False}})


@pytest.mark.parametrize('mode', MODES)
def test_reserved_rescue_at_deadline_is_deferred_not_terminal(native_worker, mode):
    run = native_worker(mode, phase_budget=13., step=1.)
    pending = run.result['deferred_confirmations']
    assert len(pending) == 1
    saved = pending[0]['native_resume']
    assert saved['failed_first_rescue_reserved'] is True and len(saved['measurements']) == 1
    assert saved['measurements'][0]['passed'] is False
    assert saved['sequence_hash'] not in run.result['terminal_native_outcomes']
    persisted = next(r['native_resume'] for r in run.checkpoint['records']
                     if r.get('sequence') == pending[0]['sequence'])
    assert persisted['failed_first_rescue_reserved'] is True
