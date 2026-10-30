"""Policy/selection/publication integration, not synthetic hardware qualification.

The only stub is the registered-native authentication boundary. It recognizes
exact fixture bytes recorded by ``authenticated_draws`` and rejects relabeling.
The real authentication helper has separate source/trace/structure tests. Gate
math, selection, annotation checks and publication validation run unchanged here.
"""
import copy
import hashlib
from pathlib import Path
import sys
import types

import pytest

import nanobody_opt
from nanobody_opt import pr100_fleet_probability as probability
from nanobody_opt import pr100_live_selection as selection
from nanobody_opt import rules
from test_pr100 import EPOCH, SEQ, evidence as legacy_evidence


POLICY = {'kind': probability.POLICY, 'minimum_smoothed_fraction': 0.4}
REQUIREMENTS = rules.pinned_config(EPOCH)['nanobody_requirements']


@pytest.fixture
def authenticated_draws(monkeypatch):
    """Clearly synthetic immutable authentication seam for pure integration."""
    known = set()
    helper = types.ModuleType('nanobody_opt.pr100_native_reference')

    def authentic(value):
        try:
            return (value.get('evidence_backend') == 'native_reference'
                    and rules.digest(value) in known)
        except (AttributeError, TypeError, ValueError):
            return False

    def identity(value):
        if not authentic(value):
            return None
        binding = value['native_reference']
        return {'backend': 'native_reference',
                'registration_sha256': binding['registration_sha256'],
                'runtime_identity_sha256': binding['receipt']['runtime_identity_sha256'],
                'policy_sha256': binding['execution_policy_sha256']}

    helper.authoritative_measurement = authentic
    helper.measurement_identity = identity
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    monkeypatch.setattr(nanobody_opt, 'pr100_native_reference', helper, raising=False)

    def draw(index, psh=81., *, sequence=SEQ, runtime='b' * 64,
             execution_policy='c' * 64, registration='d' * 64, peak='1.5'):
        value = legacy_evidence(sequence)['developability']
        value['surface_hydrophobic_patches'] = psh
        value['aggrescan3d']['a3d_average'] = '-0.8'
        value['aggrescan3d']['a3d_maximum'] = peak
        metrics = {key: value[key] for key in (
            'total_cdr_length', 'cdr3_length', 'cdr3_compactness',
            'surface_hydrophobic_patches', 'positive_charge_patches', 'negative_charge_patches')}
        metrics.update(a3d_average='-0.8', a3d_maximum=peak)
        passed = probability.measured_gate(metrics)
        value.update(sequence_hash=rules.sequence_hash(sequence), passed=passed,
                     authoritative=True, evidence_backend='native_reference')
        value['aggrescan3d']['passed'] = float(peak) <= float(rules.MAX_PEAK)
        value['native_reference'] = {
            'registration_sha256': registration, 'execution_policy_sha256': execution_policy,
            'receipt': {'runtime_identity_sha256': runtime,
                        'request_sha256': hashlib.sha256(str(index).encode()).hexdigest(),
                        'observation': {'passed': passed, 'metrics': metrics}}}
        known.add(rules.digest(value))
        return value

    return draw


def decide(rows, **kwargs):
    return selection.decide(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY, **kwargs)


def publish(rows, *, primary=None):
    verdict = decide(rows)
    item = legacy_evidence()
    index = verdict.get('primary_passing_index')
    item['developability'] = rows[index if primary is None else primary]
    item['native_fleet_selection'] = copy.deepcopy(verdict['native_fleet_selection'])
    return item


def validate(item):
    return rules.validate_evidence(item, SEQ, item['challenge'], rules.pinned_config(EPOCH))


def test_failed_first_then_pass_can_enter_binding_with_every_failure_preserved(authenticated_draws):
    rows = [authenticated_draws(0, 89.5169), authenticated_draws(1, 81.9442),
            authenticated_draws(2, 92.67)]
    original = copy.deepcopy(rows)
    verdict = decide(rows)
    assert verdict['tier'] == 'B' and not verdict['margin_safe']
    assert verdict['primary_passing_index'] == 1
    assert verdict['probability_summary']['passing_draws'] == 1
    assert verdict['probability_summary']['failing_draws'] == 2
    assert verdict['probability_summary']['smoothed_pass_fraction'] == pytest.approx(.4)
    assert verdict['native_fleet_selection']['measurements'] == rows == original
    validate(publish(rows))


@pytest.mark.parametrize('count', [1, 2, 3])
def test_all_failed_never_admitted_even_if_more_draws_are_possible(authenticated_draws, count):
    rows = [authenticated_draws(i, 90 + i) for i in range(count)]
    verdict = decide(rows)
    assert verdict['tier'] is None and verdict['reason'] == 'no_measured_pass'
    assert verdict['probability_summary']['failing_draws'] == count


def test_failed_first_rescue_requires_explicit_available_quota(authenticated_draws):
    rows = [authenticated_draws(0, 90)]
    assert not selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY)
    assert selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY,
                                   failed_first_rescue_available=True)
    assert not selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY,
                                       failed_first_rescue_available=True, repeats=1)


def test_two_complete_failures_stop_after_two_draws_even_with_rescue_quota(authenticated_draws):
    rows = [authenticated_draws(0, 90), authenticated_draws(1, 91)]
    assert not selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY,
                                       failed_first_rescue_available=True, repeats=3)


def test_rescued_passer_can_finish_third_draw_without_spending_another_rescue(authenticated_draws):
    rows = [authenticated_draws(0, 90), authenticated_draws(1, 81)]
    assert selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY)
    rows.append(authenticated_draws(2, 92))
    assert not selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY)


@pytest.mark.parametrize('backend', [None, 'cpu_native', 'cuda_refine', 'warm_cpu_screen'])
def test_legacy_or_screen_profiles_cannot_be_relabelled_as_probabilistic_native(authenticated_draws, backend):
    row = authenticated_draws(0)
    row['evidence_backend'] = backend
    assert decide([row])['reason'] == 'non_authoritative_measurement'
    # A caller's native label cannot restore a mutated/unauthenticated profile.
    row['evidence_backend'] = 'native_reference'
    row['surface_hydrophobic_patches'] = 80.5
    assert decide([row])['tier'] is None


def test_legacy_first_failure_still_stops_and_screen_confirmation_cannot_be_waived(authenticated_draws):
    legacy = legacy_evidence()['developability']
    legacy['surface_hydrophobic_patches'] = 90
    assert not selection.should_repeat([legacy], REQUIREMENTS, sequence=SEQ,
                                       failed_first_rescue_available=True)
    verdict = decide([authenticated_draws(0)], cpu_confirmation_minimum=2)
    assert verdict['tier'] is None
    assert verdict['reason'] == 'native_fleet_policy_cannot_waive_screen_confirmation'


@pytest.mark.parametrize('changed', ['runtime', 'execution_policy', 'registration', 'sequence'])
def test_authentic_but_mixed_identity_histories_are_refused(authenticated_draws, changed):
    change = {changed: SEQ[:-1] + 'A' if changed == 'sequence' else 'e' * 64}
    rows = [authenticated_draws(0), authenticated_draws(1, **change)]
    assert decide(rows)['tier'] is None
    assert not selection.should_repeat(rows, REQUIREMENTS, sequence=SEQ, fleet_policy=POLICY,
                                       failed_first_rescue_available=True)


def test_same_authenticated_receipt_is_not_two_independent_draws(authenticated_draws):
    row = authenticated_draws(0)
    assert decide([row, copy.deepcopy(row)])['tier'] is None


@pytest.mark.parametrize('damage', ['missing_history', 'empty_history', 'missing_policy',
                                   'failed_primary', 'wrong_index', 'screen_confirmation'])
def test_publication_requires_full_native_history_and_a_genuine_passing_primary(authenticated_draws, damage):
    rows = [authenticated_draws(0, 90), authenticated_draws(1)]
    item = publish(rows)
    if damage == 'missing_history':
        item.pop('native_fleet_selection')
    elif damage == 'empty_history':
        item['native_fleet_selection']['measurements'] = []
    elif damage == 'missing_policy':
        item['native_fleet_selection'].pop('policy')
    elif damage == 'failed_primary':
        item['developability'] = rows[0]
    elif damage == 'wrong_index':
        item['native_fleet_selection']['primary_passing_index'] = 0
    else:
        item['native_screen_confirmation'] = {'minimum': 2, 'measurements': rows}
    with pytest.raises(ValueError, match='native_fleet'):
        validate(item)


def test_publication_recomputes_four_draws_instead_of_trusting_stale_one_draw_summary(authenticated_draws):
    rows = [authenticated_draws(0)] + [authenticated_draws(i, 90 + i) for i in range(1, 4)]
    item = publish(rows[:1])
    item['native_fleet_selection']['measurements'] = rows
    item['probability_summary'] = decide(rows[:1])['probability_summary']
    with pytest.raises(ValueError, match='native_fleet_selection_invalid'):
        validate(item)
    assert item['native_fleet_selection']['measurements'] == rows


def test_native_history_cannot_be_attached_to_legacy_primary(authenticated_draws):
    item = publish([authenticated_draws(0)])
    item['developability'] = legacy_evidence()['developability']
    with pytest.raises(ValueError, match='native_fleet_selection_invalid'):
        validate(item)


def test_new_policy_still_requires_current_annotation(authenticated_draws):
    item = publish([authenticated_draws(0)])
    item['annotation']['cdr3_length'] += 1
    with pytest.raises(ValueError, match='wrong_epoch_cdr3_length'):
        validate(item)


def test_rule_configuration_disagreement_cannot_be_blessed_by_native_pass(authenticated_draws):
    verdict = selection.decide([authenticated_draws(0, 85)],
        {**REQUIREMENTS, 'psh_strict_rejection_threshold': 84}, sequence=SEQ, fleet_policy=POLICY)
    assert verdict['tier'] is None and verdict['reason'] == 'reference_gate_mismatch'


# Recorded controller/chain fixture; the new native receipts still use the
# explicitly synthetic immutable authentication seam above. The actual exception
# recovery branch, history reconciliation and persisted state run unchanged.
from tests.test_pr100_epoch_continuation import legacy, replay, write


@pytest.mark.parametrize('partial_kind', ['conflicting', 'extended_rejection',
                                         'complete_positive', 'positive_bad_host'])
def test_cleaned_partial_native_failure_cannot_replace_or_erase_authoritative_history(
        legacy, authenticated_draws, monkeypatch, partial_kind):
    import json
    from nanobody_opt import pr100_epoch_continuation as continuation
    from nanobody_opt import pr100_live_controller as controller

    x = legacy; sequence = x.rows[0]['sequence']; key = rules.sequence_hash(sequence)
    challenge = x.replay.challenge
    saved = continuation.migrate(x.local, challenge, x.config)
    initial = [authenticated_draws('initial', 81., sequence=sequence)]

    def history(values):
        return {'challenge': challenge, 'target': challenge['target'], 'epoch': challenge['epoch'],
                'candidates': [], 'terminal_native_outcomes': {},
                'native_complete_histories': [{'sequence_hash': key, 'measurements': values,
                    'fleet_policy': copy.deepcopy(POLICY), 'identity': copy.deepcopy(identity),
                    'execution_host': 'slave',
                    'request_sha256s': [v['native_reference']['receipt']['request_sha256'] for v in values]}]}

    identity = {'resume_eligible': True, 'backend': 'native_reference',
                'registration_sha256': 'd' * 64, 'runtime_identity_sha256': 'b' * 64,
                'policy_sha256': 'c' * 64, 'challenge_sha256': rules.digest(challenge),
                'requirements_sha256': rules.digest(rules.pinned_config(challenge['epoch'])['nanobody_requirements']),
                'filter_sha': rules.FILTER_SHA}

    def checkpoint(values):
        return {**x.rows[0], 'native_resume': {
            'schema': 'pr100-native-confirmation-checkpoint-v1', 'sequence_hash': key,
            'identity': identity, 'measurements': values, 'required_cpu_observations': 3,
            'execution_host': 'slave',
            'fleet_policy': copy.deepcopy(POLICY), 'failed_first_rescue_reserved': False}}

    continuation.reconcile_native_histories(saved, history(initial), {key})
    saved['pending_confirmation'] = [checkpoint(initial)]
    prior_pending = copy.deepcopy(saved['pending_confirmation'])
    prior_history = copy.deepcopy(saved['native_reference_histories'])
    prior_native = copy.deepcopy(saved['native_rows'])
    write(x.local / 'continuation.json', saved)
    if partial_kind == 'conflicting':
        values = [authenticated_draws('changed-prefix', 82., sequence=sequence)]
    elif partial_kind in ('complete_positive', 'positive_bad_host'):
        values = initial + [authenticated_draws(f'complete-{i}', 82., sequence=sequence) for i in range(2)]
    else:
        values = initial + [authenticated_draws(f'later-{i}', 90. + i, sequence=sequence) for i in range(3)]
    partial = {**history(values), 'status': 'partial',
               'attempted_sequence_hashes': [key], 'fresh_admitted_sequence_hashes': [],
               'fresh_never_started_sequence_hashes': [],
               'deferred_confirmations': [checkpoint(values)] if partial_kind == 'conflicting' else []}
    if partial_kind == 'positive_bad_host':
        partial['native_complete_histories'][0]['execution_host'] = 'master'

    def interrupted_phase(_config, _challenge, phase, _files, _stop, _chain, directory, **_kwargs):
        assert phase == 'native'
        write(directory / 'native/result.json', partial)
        write(directory / 'native-supervisor.json', {'cleanup': {
            'remaining_descendants': 0, 'fixed_point_reached': True}})
        raise RuntimeError('inert post-cleanup native transport failure')

    monkeypatch.setattr(controller, 'run_phase', interrupted_phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    restored = json.loads((x.local / 'continuation.json').read_text())
    assert outcome['status'] == 'recoverable_failure'
    if partial_kind in ('conflicting', 'positive_bad_host'):
        assert outcome['native_history_recovery_rejected'] is True
        assert restored['pending_confirmation'] == prior_pending
        assert restored['native_reference_histories'] == prior_history
        assert restored['native_rows'] == prior_native
    elif partial_kind == 'complete_positive':
        assert restored['native_reference_histories'][key]['complete_draws'] == 3
        assert restored['native_reference_histories'][key]['admitted'] is True
        assert restored['native_rows'] == []  # Partial transport is no novelty qualification.
        pending = restored['pending_confirmation']
        assert len(pending) == 1 and pending[0]['sequence'] == sequence
        assert pending[0]['native_resume']['measurements'] == values
        assert pending[0]['native_resume']['identity'] == identity
        assert pending[0]['native_resume']['fleet_policy'] == POLICY
        assert pending[0]['native_resume']['execution_host'] == 'slave'
        assert continuation.mode_for(restored) == 'native'
    else:
        assert restored['native_reference_histories'][key]['complete_draws'] == 4
        assert restored['native_reference_histories'][key]['admitted'] is False
        assert restored['pending_confirmation'] == [] and restored['native_rows'] == []
        # Otherwise the next publication-quality merge can preserve the older
        # successful member despite the authenticated later rejection.
        assert key in restored['invalidated_native_hashes']
        assert continuation.mode_for(restored) == 'publish'
        # Reconciliation must be the next bounded action, not wait behind an
        # unrelated fresh native batch while a disproven member stays published.
        def no_new_science(*_args, **_kwargs):
            pytest.fail('invalidated publication waited for new scientific work')
        monkeypatch.setattr(controller, 'run_phase', no_new_science)
        corrected = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
        assert corrected['status'] == 'withdrawn'
        published = json.loads(Path(x.config['publication_path']).read_text())
        assert published['sequence_pool'] == [] and published['status'] == 'invalidated'


def test_stale_shorter_partial_result_preserves_newer_canonical_pending_checkpoint(authenticated_draws):
    from nanobody_opt import pr100_epoch_continuation as continuation
    challenge = legacy_evidence()['challenge']; key = rules.sequence_hash(SEQ)
    rows = [authenticated_draws('first', 81.), authenticated_draws('second', 90.)]
    saved = {'challenge': challenge, 'native_rows': [], 'pending_confirmation': []}

    def result(values):
        return {'challenge': challenge, 'epoch': challenge['epoch'], 'candidates': [],
                'terminal_native_outcomes': {}, 'native_complete_histories': [{
                    'sequence_hash': key, 'measurements': values, 'fleet_policy': POLICY,
                    'request_sha256s': [m['native_reference']['receipt']['request_sha256'] for m in values]}]}

    continuation.reconcile_native_histories(saved, result(rows), {key})
    canonical = {'sequence': SEQ, 'native_resume': {'measurements': rows, 'fleet_policy': POLICY,
                 'required_cpu_observations': 3, 'identity': {'resume_eligible': True}}}
    saved['pending_confirmation'] = [canonical]
    incoming = result(rows[:1])
    incoming['deferred_confirmations'] = [
        {'sequence': SEQ, 'native_resume': {**canonical['native_resume'], 'measurements': rows[:1]}}]
    continuation.reconcile_native_histories(saved, incoming, {key})
    assert incoming['deferred_confirmations'] == [canonical]
    assert incoming['terminal_native_outcomes'] == {}
    assert saved['native_reference_histories'][key]['complete_draws'] == 2
