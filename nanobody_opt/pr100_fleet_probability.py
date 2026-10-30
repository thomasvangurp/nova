"""Fleet admission from complete native draws, preserving failures and uncertainty.

This is selection policy, not scientific evidence authentication. The native
reference route must validate each receipt before consuming this recommendation.
A smoothed observed fraction is a small-sample heuristic, not a calibrated
prediction of an external validator's pass probability or of winning an epoch.
"""
from __future__ import annotations

import math
import re

from nanobody_opt.pr100_validator_reference import measured_gate

POLICY = 'pr100-observed-fleet-v1'


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def summarize(observations, *, minimum_smoothed_fraction=0.4):
    """Recommend admission without requiring every independent draw to pass.

    At least one complete passing observation is mandatory. Default0.4 admits
    1/3,1/2,1/1 and2/3 evidence but rejects0/n and1/4. This is an explicit initial
    fleet policy for measurement, not an empirically optimized cutoff. It lets
    useful uncertain candidates enter binding comparisons instead of losing all
    of them on the first borderline repeat. Infrastructure errors are not draws.

    Observations must retain the complete exact gate metrics and binding IDs.
    Mixed runtime/execution policies and duplicate draws are invalid. This
    function never certifies those IDs or authorizes publication by itself.
    """
    if (type(minimum_smoothed_fraction) not in (int, float)
            or not math.isfinite(minimum_smoothed_fraction)
            or not 0 < minimum_smoothed_fraction <= 1):
        raise ValueError('invalid_minimum_smoothed_fraction')
    if not isinstance(observations, list):
        raise ValueError('observations_must_be_list')
    draws, binding, passed, reasons = set(), None, [], []
    for index, row in enumerate(observations):
        if not isinstance(row, dict) or row.get('complete') is not True:
            raise ValueError('incomplete_observation_is_not_a_biological_draw')
        identity = tuple(row.get(k) for k in
                         ('sequence_hash', 'runtime_identity_sha256', 'execution_policy_sha256'))
        if not all(_hash(v) for v in identity) or not _hash(row.get('request_sha256')):
            raise ValueError('missing_observation_binding')
        if binding is None:
            binding = identity
        elif identity != binding:
            raise ValueError('mixed_candidate_runtime_or_execution_policy')
        request = row['request_sha256']
        if request in draws:
            raise ValueError('duplicate_draw')
        draws.add(request)
        actual = measured_gate(row.get('metrics'))
        if type(row.get('passed')) is not bool or row['passed'] is not actual:
            raise ValueError('reported_verdict_disagrees_with_complete_gate')
        if actual:
            passed.append(index)
        reasons.append('passed' if actual else 'measured_gate_failed')
    n, k = len(observations), len(passed)
    smoothed = (k + 1) / (n + 2) if n else None
    admitted = bool(k and smoothed >= minimum_smoothed_fraction)
    return {
        'policy': POLICY,
        'sequence_hash': binding[0] if binding else None,
        'runtime_identity_sha256': binding[1] if binding else None,
        'execution_policy_sha256': binding[2] if binding else None,
        'admit_to_binding_comparison': admitted,
        'reason': ('no_complete_draws' if not n else 'no_measured_pass' if not k else
                   'observed_support_too_low' if not admitted else 'measured_pass_with_uncertainty'),
        'complete_draws': n, 'passing_draws': k, 'failing_draws': n - k,
        'observed_pass_fraction': k / n if n else None,
        'smoothed_pass_fraction': smoothed,
        'smoothing': {'passing_pseudocount': 1, 'failing_pseudocount': 1},
        'minimum_smoothed_fraction': minimum_smoothed_fraction,
        'calibrated_validator_probability': False,
        'primary_passing_observation_index': passed[0] if passed else None,
        'observation_verdicts': reasons,
        'observations': list(observations),
        'native_authority_validated': False,
        'publication_authorized': False,
    }


def expected_valid_count(candidates):
    """Descriptive fleet expectation proxy; no independence assumption required.

    This sums the selected small-sample estimates. It cannot estimate probability
    of any winner: shared scaffolds, stochastic preparation and binding quality
    create correlations and are not modeled here.
    """
    total, seen = 0.0, set()
    for row in candidates:
        checked = summarize(row.get('observations'),
                            minimum_smoothed_fraction=row.get('minimum_smoothed_fraction'))
        key = checked['sequence_hash']
        if (row.get('policy') != POLICY or not checked['admit_to_binding_comparison']
                or row.get('admit_to_binding_comparison') is not True
                or row.get('sequence_hash') != key
                or row.get('smoothed_pass_fraction') != checked['smoothed_pass_fraction']):
            raise ValueError('fleet_contains_rejected_or_changed_candidate')
        if key in seen:
            raise ValueError('duplicate_fleet_candidate')
        seen.add(key)
        value = checked['smoothed_pass_fraction']
        total += value
    return {'expected_valid_count_proxy': total, 'candidate_count': len(candidates),
            'calibrated': False, 'probability_of_winning_estimated': False}


def from_measurements(measurements, *, minimum_smoothed_fraction=0.4):
    """Authenticate the registered native route before computing any recommendation."""
    from nanobody_opt import pr100_native_reference as native
    observations = []
    registration = None
    for value in measurements:
        identity = native.measurement_identity(value)
        if identity is None:
            raise ValueError('non_authoritative_measurement')
        if registration is None:
            registration = identity
        elif identity != registration:
            raise ValueError('mixed_native_registration')
        receipt = value['native_reference']['receipt']
        observations.append({
            'complete': True, 'sequence_hash': value['sequence_hash'],
            'runtime_identity_sha256': identity['runtime_identity_sha256'],
            'execution_policy_sha256': identity['policy_sha256'],
            'request_sha256': receipt['request_sha256'],
            'passed': receipt['observation']['passed'],
            'metrics': receipt['observation']['metrics'],
        })
    result = summarize(observations, minimum_smoothed_fraction=minimum_smoothed_fraction)
    result['native_authority_validated'] = bool(observations)
    result['registration'] = registration
    return result
