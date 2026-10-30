"""Realistic native gate outcomes; sampling failures are retained, never hidden."""
import copy
import hashlib

import pytest

from nanobody_opt import pr100_fleet_probability as policy


def draw(index, psh=81.9442, *, peak='1.55142917', average='-0.75857694'):
    # Critical-child measured reference values, varying only the explicitly
    # tested metric. These synthetic receipts test policy, not hardware identity.
    return {'complete': True, 'sequence_hash': 'a' * 64,
            'runtime_identity_sha256': 'b' * 64, 'execution_policy_sha256': 'c' * 64,
            'request_sha256': hashlib.sha256(str(index).encode()).hexdigest(),
            'metrics': {'surface_hydrophobic_patches': psh, 'total_cdr_length': 35,
                        'cdr3_length': 19, 'cdr3_compactness': 1.4124,
                        'positive_charge_patches': 0.3261, 'negative_charge_patches': 1.2126,
                        'a3d_average': average, 'a3d_maximum': peak},
            'passed': 73.4 <= psh <= 89 and float(peak) <= 1.58023066762925
                      and float(average) <= -0.68002724866200}


def test_borderline_repeat_does_not_destroy_a_genuine_pass():
    observations = [draw(0, 89.5169), draw(1, 81.9442), draw(2, 92.67)]
    original = copy.deepcopy(observations)
    result = policy.summarize(observations)
    assert result['admit_to_binding_comparison'] is True
    assert result['passing_draws'] == 1 and result['failing_draws'] == 2
    assert result['smoothed_pass_fraction'] == pytest.approx(0.4)
    assert result['primary_passing_observation_index'] == 1
    assert result['observations'] == observations == original
    assert result['publication_authorized'] is False
    assert result['calibrated_validator_probability'] is False


def test_first_failure_is_not_itself_passing_evidence():
    result = policy.summarize([draw(0, 89.5169)])
    assert result['admit_to_binding_comparison'] is False
    assert result['reason'] == 'no_measured_pass'


def test_single_genuine_pass_can_be_compared_before_deadline():
    result = policy.summarize([draw(0)])
    assert result['admit_to_binding_comparison']
    assert result['smoothed_pass_fraction'] == pytest.approx(2 / 3)


def test_accumulating_failures_reduces_support_without_discarding_data():
    result = policy.summarize([draw(0), draw(1, 90), draw(2, 91), draw(3, 92)])
    assert not result['admit_to_binding_comparison']
    assert result['complete_draws'] == 4 and len(result['observations']) == 4


def test_aggregation_failure_counts_as_a_full_gate_failure():
    result = policy.summarize([draw(0), draw(1, peak='1.61')])
    assert result['passing_draws'] == result['failing_draws'] == 1


@pytest.mark.parametrize('field', ['sequence_hash', 'runtime_identity_sha256', 'execution_policy_sha256'])
def test_cpu_gpu_or_thread_policies_cannot_be_pooled(field):
    rows = [draw(0), draw(1)]
    rows[1][field] = 'd' * 64
    with pytest.raises(ValueError, match='mixed_candidate'):
        policy.summarize(rows)


def test_duplicate_cached_result_is_not_an_independent_repeat():
    row = draw(0)
    with pytest.raises(ValueError, match='duplicate_draw'):
        policy.summarize([row, row])


def test_timeout_does_not_count_as_biological_rejection():
    row = draw(0); row['complete'] = False
    with pytest.raises(ValueError, match='not_a_biological_draw'):
        policy.summarize([row])


def test_caller_cannot_bless_a_failed_gate():
    row = draw(0, 90); row['passed'] = True
    with pytest.raises(ValueError, match='disagrees'):
        policy.summarize([row])


def test_no_data_and_uncalibrated_fleet_expectation():
    assert not policy.summarize([])['admit_to_binding_comparison']
    a = policy.summarize([draw(0), draw(1, 90)])
    b = policy.summarize([{**draw(i), 'sequence_hash': 'd' * 64} for i in range(3)])
    result = policy.expected_valid_count([a, b])
    assert result['expected_valid_count_proxy'] == pytest.approx(1.3)
    assert not result['probability_of_winning_estimated']


def test_fleet_cannot_count_rejected_or_duplicate_candidates():
    a = policy.summarize([draw(0)])
    with pytest.raises(ValueError, match='duplicate_fleet'):
        policy.expected_valid_count([a, a])
    rejected = policy.summarize([draw(0, 90)])
    with pytest.raises(ValueError, match='rejected_or_changed'):
        policy.expected_valid_count([rejected])
    a['smoothed_pass_fraction'] = 1.0
    with pytest.raises(ValueError, match='rejected_or_changed'):
        policy.expected_valid_count([a])
