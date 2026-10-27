"""Accelerated observations must never become CPU publication evidence."""
import json
import types

import pytest

from nanobody_opt import rules, pr100_live_selection as selection, pr100_live_worker as worker

CHALLENGE = {'epoch': 25963, 'target': 'Q9NZQ7'}


def profile(sequence, psh=80., **changes):
    value = {'passed': True, 'red_flags': [], 'total_cdr_length': 30, 'cdr3_length': 15,
             'cdr3_compactness': 1., 'surface_hydrophobic_patches': psh,
             'positive_charge_patches': .2, 'negative_charge_patches': .2,
             'aggrescan3d': {'status': 'ok', 'passed': True, 'a3d_average': -1., 'a3d_maximum': 1.,
                 'sequence_hash': rules.sequence_hash(sequence), 'residue_count': len(sequence),
                 'ph_scale': 'default-scale', 'surface_method': 'FreeSASA-2.0.1-NACCESS-LR100',
                 'mode': 'static-no-minimization', 'radius_angstrom': 10,
                 'input_structure_sha256': 'a' * 64, 'normalized_structure_sha256': 'b' * 64}}
    value.update(changes)
    return value


def accelerated(sequence, psh=80., backend='cuda_refine', **changes):
    return profile(sequence, psh, evidence_backend=backend, authoritative=False,
                   sequence_hash=rules.sequence_hash(sequence), epoch=CHALLENGE['epoch'],
                   challenge_sha256=rules.digest(CHALLENGE), protocol_sha256='c' * 64,
                   runtime_identity_sha256='d' * 64, **changes)


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.pr100_filter, 'helper', lambda _: types.SimpleNamespace(tool_paths=None))
    def evidence_is_cpu(evidence, *_args):
        assert selection.authoritative_cpu_measurement(evidence['developability'])
    monkeypatch.setattr(worker.rules, 'validate_evidence', evidence_is_cpu)
    def run(sequences, screen, measure=None, **overrides):
        payload = {'challenge': CHALLENGE, 'target': 'Q9NZQ7', 'epoch': 25963,
                   'archive_sha256': 'a' * 64, 'archive_checked_at_unix': 1,
                   'candidates': [{'sequence': s, 'sequence_hash': rules.sequence_hash(s),
                                   'pr100_annotation': {}} for s in sequences]}
        (tmp_path / 'input.json').write_text(json.dumps(payload))
        job = {'phase': 'native', 'challenge': CHALLENGE, 'native_tools': '/tools',
               'workers': 4, 'max_workers': 4, 'repeats': 3, 'allow_marginal': True,
               'candidate_budget_seconds': 10, 'native_screen': {'enabled': True}}
        clock = overrides.pop('clock', lambda: 0.)
        job.update(overrides)
        measure = measure or (lambda seqs, *_args, **kw: ([profile(s) for s in seqs], 0., 1.))
        return worker.native_phase(job, tmp_path, screen=screen, measure=measure, clock=clock)
    return run, tmp_path


def screen_with(psh=80., backend='cuda_refine'):
    return lambda seqs, **kw: {'status': 'complete', 'profiles': {
        s: accelerated(s, psh, backend=backend) for s in seqs}}


@pytest.mark.parametrize('backend', ['resident_cpu', 'cuda_refine'])
def test_actual_backend_separate_from_three_cpu_confirmations(native, backend):
    run, directory = native
    result = run(['AAAA'], screen_with(backend=backend))
    row = result['candidates'][0]
    assert row['repeats'] == 3 and len(row['measurements']) == 3
    assert row['accelerated_screen']['evidence_backend'] == backend
    assert row['cpu_confirmation_minimum'] == 3
    assert all(selection.authoritative_cpu_measurement(m) for m in row['measurements'])
    assert result['fresh_evaluations'] == 1 and result['repeat_evaluations'] == 2
    assert result['completed'] == 3 and result['total_profile_evaluations'] == 4
    assert result['unique_profiled_candidates'] == 1
    assert result['native_screen']['actual_backends'] == {backend: 1}
    records = [json.loads(s) for s in (directory / 'native-observations.jsonl').read_text().splitlines()]
    assert sum(r['record_type'] == 'observation' for r in records) == 3
    screen_log = json.loads((directory / 'native-screen/batch-00000/screen-observations.json').read_text())
    assert screen_log['authoritative'] is False and len(screen_log['records']) == 1


def test_allow_marginal_cannot_publish_gpu_plus_one_cpu(native):
    run, _ = native; now = [0.]
    def measure(sequences, *_args, **kw):
        now[0] += 26.
        return [profile(s, 88.) for s in sequences], 0., 26.
    result = run(['AAAA'], screen_with(88.), measure=measure, phase_budget_seconds=50,
                 clock=lambda: now[0])
    # Two CPU calls fit before the last timeout reservation fails; three required.
    assert result['completed'] == 2 and result['candidates'] == []
    assert result['rejection_counts']['cpu_confirmation_required'] == 1


def test_repeats_one_still_requires_two_cpu_profiles_after_acceleration(native):
    run, _ = native
    result = run(['AAAA'], screen_with(88.), repeats=1,
                 measure=lambda seqs, *_a, **kw: ([profile(s, 88.) for s in seqs], 0., 1.))
    assert result['candidates'][0]['repeats'] == 2
    assert result['candidates'][0]['tier'] == 'B'
    assert result['fresh_evaluations'] == 1 and result['repeat_evaluations'] == 1


@pytest.mark.parametrize('mode', ['unavailable', 'exception', 'empty'])
def test_unavailable_screen_preserves_cpu_only_marginal_behavior(native, mode):
    run, _ = native
    def screen(*args, **kwargs):
        if mode == 'exception': raise RuntimeError('inert unavailable backend')
        return {'status': mode, 'profiles': {}}
    result = run(['AAAA'], screen, repeats=1)
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['repeats'] == 1
    assert result['candidates'][0]['accelerated_screen'] is None
    assert result['native_screen']['completed_profiles'] == 0


@pytest.mark.parametrize('change', [
    {'epoch': 25962}, {'challenge_sha256': 'a' * 64}, {'sequence_hash': 'f' * 64},
    {'evidence_backend': 'cpu_native'}, {'authoritative': True},
    {'protocol_sha256': None}, {'runtime_identity_sha256': 'bad'}, {'evidence_backend': []},
])
def test_stale_or_forged_screen_is_cpu_fallback(native, change):
    run, _ = native
    def screen(seqs, **kw):
        return {'profiles': {s: {**accelerated(s, 160.), **change} for s in seqs}}
    result = run(['AAAA'], screen, repeats=1)
    assert len(result['candidates']) == 1 and result['completed'] == 1
    assert result['native_screen']['completed_profiles'] == 0


@pytest.mark.parametrize('psh', [54., 70., 73., 90., 108.])
def test_near_both_boundaries_is_always_rescued(native, psh):
    run, _ = native
    result = run(['AAAA'], screen_with(psh))
    assert len(result['candidates']) == 1 and result['completed'] == 3
    assert result['native_screen']['audit_candidates'] == 0


def test_non_psh_failure_is_always_rescued(native):
    run, _ = native
    def screen(seqs, **kw):
        return {'profiles': {s: accelerated(s, 80., red_flags=['positive_charge']) for s in seqs}}
    result = run(['AAAA'], screen)
    assert len(result['candidates']) == 1 and result['native_screen']['audit_candidates'] == 0


def test_far_negative_audit_is_deterministic_and_does_not_claim_cpu_rejection(native):
    run, _ = native; seen = []
    def fail(sequences, *_args, **kw):
        seen.extend(sequences)
        return [profile(s, 160.) for s in sequences], 0., 1.
    sequences = ['AAAA', 'CCCC', 'DDDD', 'EEEE']
    result = run(sequences, screen_with(160.), measure=fail)
    expected = min(sequences, key=lambda s: rules.digest([CHALLENGE, s, 'negative-audit-v1']))
    assert seen == [expected]
    assert result['native_screen']['audit_candidates'] == 1
    assert result['native_screen']['screen_negative_candidates'] == 3
    assert result['fresh_evaluations'] == 1 and result['repeat_evaluations'] == 0
    assert result['total_profile_evaluations'] == 5 and result['not_run'] == 0
    assert result['rejection_counts']['nonauthoritative_screen_negative'] == 3


def test_audit_disagreement_rescues_held_negatives_and_stops_later_screening(native):
    run, _ = native; calls = []
    def screen(sequences, **kw):
        calls.append(sequences)
        return screen_with(160.)(sequences, **kw)
    seqs = ['AAAA', 'CCCC', 'DDDD', 'EEEE', 'FFFF', 'GGGG']
    result = run(seqs, screen)
    assert len(calls) == 1 and len(result['candidates']) == 6
    assert result['native_screen']['audit_disagreements'] == 1
    assert result['native_screen']['rescued_candidates'] == 3
    assert result['native_screen']['rejection_disabled'] is True
    assert result['native_screen']['screen_negative_candidates'] == 0
    assert result['fresh_evaluations'] == 6 and result['repeat_evaluations'] == 12


@pytest.mark.parametrize('failure', ['timeout', 'missing', 'nonauthoritative'])
def test_inconclusive_audit_rescues_held_negatives_and_disables_filtering(native, failure):
    run, _ = native; screens = []; calls = [0]
    def screen(sequences, **kw):
        screens.append(sequences)
        return screen_with(160.)(sequences, **kw)
    def measure(sequences, *_args, **kw):
        calls[0] += 1
        if calls[0] == 1:
            assert len(sequences) == 1
            value = ({'error': True, 'error_kind': 'timeout'} if failure == 'timeout' else
                     None if failure == 'missing' else accelerated(sequences[0], 160.))
            return [value], 0., 1.
        return [profile(s) for s in sequences], 0., 1.
    result = run(['AAAA', 'CCCC', 'DDDD', 'EEEE', 'FFFF', 'GGGG'], screen, measure=measure)
    assert len(screens) == 1 and len(result['candidates']) == 5
    assert result['native_screen']['audit_inconclusive'] == 1
    assert result['native_screen']['audit_disagreements'] == 0
    assert result['native_screen']['rescued_candidates'] == 3
    assert result['native_screen']['rejection_disabled'] is True
    assert result['native_screen']['screen_negative_candidates'] == 0


def test_cancellation_before_cpu_confirmation_cannot_publish(native):
    run, directory = native
    def cancelled(*args, **kwargs): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): run(['AAAA'], cancelled)
    result = json.loads((directory / 'result.json').read_text())
    assert result['status'] == 'running' and result['candidates'] == [] and result['completed'] == 0


def test_gpu_data_returned_by_cpu_call_is_not_journaled_as_cpu_metrics(native):
    run, directory = native
    result = run(['AAAA'], screen_with(), measure=lambda seqs, *_a, **kw:
                 ([accelerated(s) for s in seqs], 0., 1.))
    assert result['candidates'] == []
    assert result['rejection_counts']['non_authoritative_measurement'] == 1
    records = [json.loads(s) for s in (directory / 'native-observations.jsonl').read_text().splitlines()]
    observations = [r for r in records if r['record_type'] == 'observation']
    assert len(observations) == 1
    assert observations[0]['observation']['metrics'] == {}
    assert observations[0]['observation']['outcome'] == 'inconclusive'


@pytest.mark.parametrize('backend', ['resident_cpu', 'cuda_refine'])
def test_selection_rejects_accelerated_only_and_mixed_evidence(backend):
    for values in ([accelerated('AAAA', backend=backend)],
                   [profile('AAAA'), profile('AAAA'), accelerated('AAAA', backend=backend)]):
        verdict = selection.decide(values, {'psh_strict_rejection_threshold': 89}, sequence='AAAA')
        assert verdict['tier'] is None and verdict['reason'] == 'non_authoritative_measurement'
    assert selection.order_candidates([{'tier': 'B', 'psh_mean': 80.,
        'accelerated_screen': {'evidence_backend': backend}, 'cpu_confirmation_minimum': 3,
        'measurements': [profile('AAAA')]}], allow_marginal=True) == []
