"""Trust each metric in proportion to how reproducible it actually is.

The validator scores a dense rank sum over ten metrics, so that sum is the
objective and nothing here changes it.  What changes is how the sum is
*estimated* from noisy observations.

Measured across 42 sequences observed seven times each, the ten metrics differ
enormously in reproducibility.  Within the narrow band the contest is actually
fought in, ``design_to_target_iptm`` and the two PAE terms are almost perfectly
reproducible, while ``plip_hbonds_refolded`` varies as much between repeats of
one sequence as it does between different sequences -- its observed value
carries no information about the sequence at all.

Taking an unreliable metric at face value imports its noise into the estimate
without importing any signal.  The optimal estimate of an unobservable true
value from a noisy one shrinks it toward the population mean by its
unreliability, so that is what this does: a metric with reliability one is kept
as observed, a metric with reliability zero is replaced by the population value
and contributes an identical constant to every candidate's rank sum.

This helps most exactly where it is needed.  A screened candidate has one
observation, so its unreliable metrics are pure noise; a replicated finalist has
thirteen, which already divides that noise by more than three.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

# Measured on 44 live replicated sequences with the estimator below.  These are
# the reliabilities of ONE observation among near-identical finalists -- the set
# selection actually chooses between -- and they are the reason deep replication
# matters: with n observations reliability becomes n*r / (1 + (n-1)*r), so
# design_to_target_iptm goes 0.18 -> 0.74 at thirteen observations and 0.83 at
# twenty-three.
#
# Reliability is a property of the COHORT, not of the metric: a wider candidate
# set has more true variance and therefore higher reliability at the same noise.
# Do not apply these to a diverse screening pool without remeasuring on it, or
# real signal will be shrunk away.
MEASURED_RELIABILITY: dict[str, float] = {
    "design_iiptm": 0.30,
    "design_ptm": 0.16,
    "design_to_target_iptm": 0.18,
    "min_design_to_target_pae": 0.26,
    "interaction_pae": 0.43,
    "plip_hbonds_refolded": 0.01,
    "plip_saltbridge_refolded": 0.00,
    "delta_sasa_refolded": 0.12,
    "liability_score": 1.00,
    "liability_num_violations": 1.00,
}


def reliability_from_observations(
    repeats: Mapping[str, list[Mapping[str, float]]],
    metrics: Iterable[str],
) -> dict[str, float]:
    """Measure reliability per metric from sequences observed more than once.

    ``repeats`` maps a sequence to its independent observations.  A metric whose
    repeat-to-repeat variance matches its sequence-to-sequence variance scores
    zero: observing it tells you nothing about which sequence you observed.
    """
    reliability: dict[str, float] = {}
    for metric in metrics:
        within: list[float] = []
        means: list[float] = []
        for observations in repeats.values():
            values = [
                float(item[metric]) for item in observations if metric in item
            ]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            means.append(mean)
            within.append(
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            )
        if len(means) < 2 or not within:
            continue
        var_within = sum(within) / len(within)
        grand = sum(means) / len(means)
        spread = sum((mean - grand) ** 2 for mean in means) / (len(means) - 1)
        # Each group mean is itself an average of noisy draws, so its spread
        # already contains var_within / group_size.  Subtract that to recover the
        # true between-sequence variance; adding it instead would overstate the
        # signal and make every metric look more reliable than it is.
        sizes = [
            len([1 for item in observations if metric in item])
            for observations in repeats.values()
        ]
        typical = sum(size for size in sizes if size > 1) / max(
            len([size for size in sizes if size > 1]), 1
        )
        var_true = max(spread - var_within / max(typical, 1.0), 0.0)
        var_observed = var_true + var_within
        if var_observed <= 0:
            continue
        reliability[metric] = max(0.0, min(1.0, var_true / var_observed))
    return reliability


def load_reliability(path: Path | None) -> dict[str, float]:
    """Read measured reliabilities, falling back to the shipped measurement."""
    if path is None:
        return dict(MEASURED_RELIABILITY)
    payload = json.loads(path.read_text())
    values = payload.get("reliability", payload)
    if not isinstance(values, dict):
        raise ValueError(f"reliability file must be an object: {path}")
    merged = dict(MEASURED_RELIABILITY)
    for metric, value in values.items():
        number = float(value)
        if not 0.0 <= number <= 1.0 or not math.isfinite(number):
            raise ValueError(f"reliability for {metric} must be within [0, 1]")
        merged[metric] = number
    return merged


def population_values(
    observations: Iterable[Mapping[str, float]], metrics: Iterable[str]
) -> dict[str, float]:
    """The value an uninformative metric is shrunk toward: the pool median."""
    pooled: dict[str, list[float]] = {metric: [] for metric in metrics}
    for record in observations:
        for metric in pooled:
            if metric in record:
                pooled[metric].append(float(record[metric]))
    medians: dict[str, float] = {}
    for metric, values in pooled.items():
        if not values:
            continue
        ordered = sorted(values)
        middle = len(ordered) // 2
        medians[metric] = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
    return medians


def shrink(
    record: Mapping[str, float],
    *,
    reliability: Mapping[str, float],
    population: Mapping[str, float],
) -> dict[str, Any]:
    """Pull each metric toward the population value by its unreliability."""
    shrunk = dict(record)
    for metric, centre in population.items():
        if metric not in shrunk:
            continue
        weight = float(reliability.get(metric, 1.0))
        value = float(shrunk[metric])
        shrunk[metric] = weight * value + (1.0 - weight) * centre
    return shrunk


def shrink_observations(
    by_sequence: Mapping[str, list[Mapping[str, float]]],
    *,
    reliability: Mapping[str, float] | None = None,
    metrics: Iterable[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Apply reliability shrinkage across a whole cohort of observations."""
    weights = dict(reliability or MEASURED_RELIABILITY)
    names = list(metrics or weights)
    flat = [record for records in by_sequence.values() for record in records]
    centres = population_values(flat, names)
    return {
        sequence: [
            shrink(record, reliability=weights, population=centres)
            for record in records
        ]
        for sequence, records in by_sequence.items()
    }
