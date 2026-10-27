"""Repeat-and-margin selection for the PR100 native developability stage.

Pure decision logic: no I/O, no profiling, no gate semantics of its own. The
per-measurement verdict always comes from ``rules.developability_reason``, so
validator-matched thresholds, order and provenance strings live in one place.

The upper margin rule is the k-scaled form from the audit
(nova-reports/pr100-audit-20261026/7_developability_algorithm.md, 2.1):
``mean_k <= cap - z*sigma/sqrt(k)`` together with a hard margin that gives
about 80 at k=2 and about 82 at k=3. PSH carries a per-measurement noise SD of
roughly 4.4. Both ends of the validator's accepted PSH interval matter. These
margin heuristics distinguish confidence tiers; they do not estimate the
probability that a fresh validator measurement will pass.
"""
from __future__ import annotations

import math

# Audit-derived defaults. sigma is estimated from 24 repeat triples; these are
# configurable estimates, not constants of nature.
DEFAULT_MARGIN = {
    'cap': 89,
    'z': 1.3,
    'sigma': 4.4,
    'min_repeats': 2,
    'hard_margin': {'2': 80, '3': 82},
}
DEFAULT_REPEATS = 3
PSH_KEY = 'surface_hydrophobic_patches'
PSH_LOWER_BOUND = 73.4
PSH_UPPER_BOUND = 89.0


def margin_settings(supplied=None):
    """Merge supplied margin parameters over the audit defaults."""
    merged = dict(DEFAULT_MARGIN)
    for key, value in dict(supplied or {}).items():
        if key not in DEFAULT_MARGIN:
            raise ValueError('unknown margin parameter: ' + str(key))
        merged[key] = value
    hard = {str(k): v for k, v in dict(merged['hard_margin']).items()}
    if not hard or any(not _finite(v) for v in hard.values()):
        raise ValueError('hard_margin must map repeat counts to finite values')
    merged['hard_margin'] = hard
    for key in ('cap', 'z', 'sigma'):
        if not _finite(merged[key]):
            raise ValueError('margin ' + key + ' must be finite')
    if merged['sigma'] < 0 or merged['z'] < 0:
        raise ValueError('margin z and sigma must be non-negative')
    minimum = merged['min_repeats']
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        # A single measurement is never margin-safe, by construction.
        raise ValueError('min_repeats must be an integer >= 2')
    return merged


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hard_margin_for(settings, k):
    """Hard floor for k repeats; counts above the table reuse its largest key."""
    table = settings['hard_margin']
    if str(k) in table:
        return table[str(k)]
    keys = sorted(int(key) for key in table)
    usable = [key for key in keys if key <= k]
    return table[str(usable[-1] if usable else keys[0])]


def margin_threshold(k, settings=None):
    """The mean-PSH a candidate with k repeats must not exceed."""
    if not (isinstance(settings, dict) and set(DEFAULT_MARGIN) <= set(settings)):
        settings = margin_settings(settings)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('repeat count must be a positive integer')
    statistical = settings['cap'] - settings['z'] * settings['sigma'] / math.sqrt(k)
    return min(statistical, hard_margin_for(settings, k))


def margin_lower_threshold(k, settings=None):
    """Keep the mean away from the ordinary TNP lower rejection boundary."""
    settings = margin_settings(settings)
    # Share count validation with the existing upper-margin calculation.
    margin_threshold(k, settings)
    return PSH_LOWER_BOUND + settings['z'] * settings['sigma'] / math.sqrt(k)


def psh_margin(value, *, lower=PSH_LOWER_BOUND, upper=PSH_UPPER_BOUND):
    """Signed distance to the closest validator boundary; larger is better."""
    if not _finite(value):
        return None
    if not _finite(lower) or not _finite(upper) or lower >= upper:
        raise ValueError('invalid PSH interval')
    return min(value - lower, upper - value)


def psh_values(measurements):
    return [m.get(PSH_KEY) for m in measurements]


def mean_and_sd(values):
    """Population mean and SD of the raw measurements; (None, None) if unusable."""
    usable = [v for v in values if _finite(v)]
    if not usable or len(usable) != len(values):
        return None, None
    mean = sum(usable) / len(usable)
    if len(usable) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in usable) / (len(usable) - 1)
    return mean, math.sqrt(variance)


def decide(measurements, requirements, *, sequence=None, repeats=DEFAULT_REPEATS,
           margin=None, reason_for=None):
    """Classify a candidate from its raw repeat measurements.

    Returns a dict with the tier ('A', 'B' or None), every raw measurement, the
    mean and SD of PSH, the applied threshold and a rejection reason when the
     candidate is not publishable. Tier A means the repeated measurements meet
     both margin heuristics. Tier B means every measured gate passed, but a
     margin or the minimum repeat count is missing.

    ``reason_for`` is the per-measurement gate verdict (default
    ``rules.developability_reason``); this function never re-implements a gate.
    """
    settings = margin_settings(margin)
    if reason_for is None:
        from nanobody_opt import rules
        reason_for = lambda value: rules.developability_reason(
            value, requirements, sequence=sequence)
    measurements = list(measurements)
    verdict = {'measurements': measurements, 'repeats': len(measurements),
               'tier': None, 'margin_safe': False, 'psh_values': [],
               'psh_mean': None, 'psh_sd': None,
               'margin_threshold': None, 'margin_lower_threshold': None,
               'psh_boundary_margin': None, 'reason': ''}
    if not measurements:
        verdict['reason'] = 'not_run'
        return verdict
    # Every gate must pass on every repeat, per existing rule semantics.
    for value in measurements:
        reason = reason_for(value)
        if reason:
            verdict['reason'] = reason
            return verdict
    values = psh_values(measurements)
    mean, sd = mean_and_sd(values)
    verdict.update(psh_values=values, psh_mean=mean, psh_sd=sd)
    if mean is None:
        verdict['reason'] = 'psh_unavailable'
        return verdict
    threshold = margin_threshold(len(measurements), settings)
    lower_threshold = margin_lower_threshold(len(measurements), settings)
    verdict['margin_threshold'] = threshold
    verdict['margin_lower_threshold'] = lower_threshold
    verdict['psh_boundary_margin'] = psh_margin(mean)
    enough = len(measurements) >= settings['min_repeats'] and len(measurements) >= min(repeats, settings['min_repeats'])
    if enough and lower_threshold <= mean <= threshold:
        verdict.update(tier='A', margin_safe=True)
    else:
        # Gates all passed; only the margin (or the repeat count) is missing.
        verdict.update(tier='B', reason='margin_missed' if enough else 'insufficient_repeats')
    return verdict


def order_candidates(candidates, *, allow_marginal=True):
    """Tier A first, then prefer distance from either PSH rejection boundary."""
    def key(row):
        margin = psh_margin(row.get('psh_mean'))
        return (0 if row.get('tier') == 'A' else 1,
                -margin if margin is not None else math.inf,
                str(row.get('sequence_hash') or ''))
    ordered = sorted((r for r in candidates if r.get('tier') in ('A', 'B')), key=key)
    if not allow_marginal:
        ordered = [r for r in ordered if r.get('tier') == 'A']
    return ordered


def shard(candidates, gpus):
    """Deal candidates round-robin across gpu ids, best candidates spread first."""
    if not isinstance(gpus, (list, tuple)) or not gpus:
        raise ValueError('at least one gpu id is required')
    ids = list(gpus)
    if any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in ids):
        raise ValueError('gpu ids must be non-negative integers')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate gpu id')
    shards = {g: [] for g in ids}
    for index, row in enumerate(candidates):
        shards[ids[index % len(ids)]].append(row)
    return shards
