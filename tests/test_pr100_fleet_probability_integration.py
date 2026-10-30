"""Policy/selection/publication integration, not synthetic hardware qualification.

The only stub is the registered-native authentication boundary. It recognizes
exact fixture bytes recorded by ``authenticated_draws`` and rejects relabeling.
The real authentication helper has separate source/trace/structure tests. Gate
math, selection, annotation checks and publication validation run unchanged here.
"""
import copy
import hashlib
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
