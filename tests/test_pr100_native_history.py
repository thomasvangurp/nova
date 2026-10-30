"""Controller history monotonicity; receipt authentication is tested separately."""
import copy

import pytest

from nanobody_opt import pr100_epoch_continuation as continuation
from nanobody_opt import pr100_fleet_probability as probability
from nanobody_opt import rules
from test_pr100_fleet_probability import draw

SEQUENCE = 'QVQLVESGGGLVQAGGSLRLSCAAS'
KEY = rules.sequence_hash(SEQUENCE)
CHALLENGE = {'epoch': 26020, 'target': 'fixture'}
POLICY = {'kind': probability.POLICY, 'minimum_smoothed_fraction': 0.4}


@pytest.fixture(autouse=True)
def authenticated_boundary_stub(monkeypatch):
    def validate(values, *, minimum_smoothed_fraction):
        result = probability.summarize(values, minimum_smoothed_fraction=minimum_smoothed_fraction)
        return {**result, 'native_authority_validated': True,
                'registration': {'runtime': result['runtime_identity_sha256'],
                                 'policy': result['execution_policy_sha256']}}
    monkeypatch.setattr(probability, 'from_measurements', validate)


def observations(scores):
    return [{**draw(i, score), 'sequence_hash': KEY} for i, score in enumerate(scores)]


def result(values):
    summary = probability.summarize(values)
    candidate = {'sequence': SEQUENCE, 'measurements': values,
                 'pr100_evidence': {
                     'developability': {'evidence_backend': 'native_reference'},
                     'native_fleet_selection': {'policy': POLICY, 'measurements': values,
                         'primary_passing_index': summary['primary_passing_observation_index']}}}
    return {'epoch': 26020, 'challenge': CHALLENGE,
            'candidates': [candidate] if summary['admit_to_binding_comparison'] else [],
            'native_complete_histories': [{'sequence_hash': KEY, 'measurements': values,
                'fleet_policy': POLICY, 'request_sha256s': [v['request_sha256'] for v in values]}],
            'terminal_native_outcomes': {}}


def state():
    return {'challenge': CHALLENGE, 'native_rows': []}


def apply(saved, incoming):
    continuation.reconcile_native_histories(saved, incoming, {KEY})


def test_later_one_of_four_removes_prior_pass_and_stale_replay_cannot_revive_it():
    saved = state(); first = result(observations([81]))
    apply(saved, first); saved['native_rows'] = first['candidates']
    later = result(observations([81, 90, 91, 92])); apply(saved, later)
    assert saved['native_rows'] == []
    assert later['terminal_native_outcomes'] == {KEY: 'observed_support_too_low'}
    replay = result(observations([81])); apply(saved, replay)
    assert replay['candidates'] == []
    assert replay['terminal_native_outcomes'] == {KEY: 'observed_support_too_low'}
    assert saved['native_reference_histories'][KEY]['complete_draws'] == 4


def test_one_failed_repeat_cannot_override_supported_aggregate():
    saved = state(); first = result(observations([81]))
    apply(saved, first); saved['native_rows'] = first['candidates']
    mixed = result(observations([81, 90]))
    mixed['terminal_native_outcomes'][KEY] = 'psh_threshold'
    apply(saved, mixed)
    assert mixed['terminal_native_outcomes'] == {}
    assert mixed['candidates'][0]['pr100_evidence']['native_fleet_selection']['measurements'] == observations([81, 90])
    assert saved['native_rows'] == []  # Old evidence superseded; caller installs newly checked row.


def test_changed_previous_draw_is_refused_atomically():
    saved = state(); apply(saved, result(observations([81])))
    before = copy.deepcopy(saved)
    with pytest.raises(ValueError, match='conflicts'):
        apply(saved, result(observations([82, 83])))
    assert saved == before


def test_omitting_failure_from_candidate_evidence_is_refused():
    saved = state(); incoming = result(observations([81, 90]))
    incoming['candidates'][0]['pr100_evidence']['native_fleet_selection']['measurements'] = observations([81])
    with pytest.raises(ValueError, match='omits complete'):
        apply(saved, incoming)
    assert 'native_reference_histories' not in saved


def test_missing_history_and_outside_assignment_are_refused():
    incoming = result(observations([81])); incoming.pop('native_complete_histories')
    with pytest.raises(ValueError, match='history missing'):
        apply(state(), incoming)
    with pytest.raises(ValueError, match='outside assignment'):
        continuation.reconcile_native_histories(state(), result(observations([81])), set())


def test_runtime_change_cannot_mix_with_prior_history():
    saved = state(); apply(saved, result(observations([81])))
    values = observations([81, 82])
    for row in values:
        row['runtime_identity_sha256'] = 'd' * 64
    with pytest.raises(ValueError, match='identity changed'):
        apply(saved, result(values))


def test_empty_not_run_history_never_invalidates_a_candidate():
    saved = state(); first = result(observations([81]))
    apply(saved, first); saved['native_rows'] = first['candidates']
    empty = result([]); apply(saved, empty)
    assert saved['native_rows'] == first['candidates']
    assert empty['terminal_native_outcomes'] == {}


def test_legacy_result_does_not_create_new_history_state():
    saved = state(); incoming = {'candidates': [{'sequence': SEQUENCE}], 'terminal_native_outcomes': {}}
    apply(saved, incoming)
    assert 'native_reference_histories' not in saved


def test_reserved_first_failure_rescue_stays_pending_instead_of_terminal():
    saved = state(); values = observations([90]); incoming = result(values)
    incoming['deferred_confirmations'] = [{'sequence': SEQUENCE, 'native_resume': {
        'measurements': values, 'fleet_policy': POLICY, 'required_cpu_observations': 3,
        'failed_first_rescue_reserved': True}}]
    apply(saved, incoming)
    assert incoming['terminal_native_outcomes'] == {}
    assert saved['native_reference_histories'][KEY]['bounded_rescue_pending'] is True
    assert len(incoming['deferred_confirmations']) == 1


def test_stale_partial_confirmation_cannot_restart_old_history():
    saved = state(); apply(saved, result(observations([81, 90, 91, 92])))
    incoming = result(observations([81]))
    incoming['deferred_confirmations'] = [{'sequence': SEQUENCE, 'native_resume': {
        'measurements': observations([81]), 'fleet_policy': POLICY}}]
    apply(saved, incoming)
    assert incoming['deferred_confirmations'] == []


def test_identical_history_cannot_drop_reserved_first_failure_rescue():
    saved = state(); values = observations([90]); first = result(values)
    checkpoint = {'sequence': SEQUENCE, 'native_resume': {
        'measurements': values, 'fleet_policy': POLICY, 'required_cpu_observations': 3,
        'failed_first_rescue_reserved': True}}
    first['deferred_confirmations'] = [checkpoint]
    apply(saved, first)
    saved['pending_confirmation'] = copy.deepcopy(first['deferred_confirmations'])
    replay = result(values)
    apply(saved, replay)
    assert replay['terminal_native_outcomes'] == {}
    assert replay['deferred_confirmations'] == [checkpoint]
    assert saved['native_reference_histories'][KEY]['bounded_rescue_pending'] is True
    exhausted = result(observations([90, 91]))
    apply(saved, exhausted)
    assert exhausted['terminal_native_outcomes'] == {KEY: 'no_measured_pass'}
    assert exhausted['deferred_confirmations'] == []


@pytest.mark.parametrize('records', [None, []])
def test_registered_terminal_claim_requires_complete_history(records):
    saved = state(); apply(saved, result(observations([81])))
    incoming = {'epoch': 26020, 'challenge': CHALLENGE, 'candidates': [],
                'terminal_native_outcomes': {KEY: 'observed_support_too_low'}}
    if records is not None:
        incoming['native_complete_histories'] = records
    with pytest.raises(ValueError, match='history'):
        apply(saved, incoming)
