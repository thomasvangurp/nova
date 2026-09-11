"""Refuse candidates that are catastrophically bad on any single metric.

The published score is a dense rank sum over ten metrics, so one metric ranked
near the bottom of the field costs as much as being mid-pack on four.  Epoch
25061 is the worked example: our submission ranked 7th of 48 on ``design_iiptm``
and 47th on ``min_design_to_target_pae``, with two hydrogen bonds against a field
median of four.  It scored 190 against a field median of 156.  A candidate with
an outlier like that cannot win no matter how good its best metric is, and only
rank one pays.

The guard is deliberately about the *worst* metric rather than the average.
Averages already enter through the rank sum; what the rank sum does not express
is that a single catastrophic metric is disqualifying rather than merely costly.

It also refuses to empty the shortlist.  If too few candidates clear the bar the
least-bad ones are kept instead, because screening and replicating something
beats publishing nothing.
"""

from __future__ import annotations

import bisect
from typing import Any, Iterable, Mapping, Sequence

from nanobody_opt.simulate_round import METRICS


def field_distribution(
    rows: Iterable[Mapping[str, Any]], metrics: Mapping[str, str] = METRICS
) -> dict[str, list[float]]:
    """Sorted per-metric values from a competitor field, for percentile lookup."""
    collected: dict[str, list[float]] = {name: [] for name in metrics}
    for row in rows:
        for name in collected:
            value = row.get(name)
            if value in (None, ""):
                continue
            try:
                collected[name].append(float(value))
            except (TypeError, ValueError):
                continue
    return {name: sorted(values) for name, values in collected.items() if values}


def worst_percentile(
    candidate: Mapping[str, float],
    field: Mapping[str, list[float]],
    metrics: Mapping[str, str] = METRICS,
) -> tuple[float, str]:
    """Fraction of the field beating this candidate on its weakest metric.

    Zero means best in the field on every metric; one means last on at least one.
    """
    worst = 0.0
    culprit = ""
    for name, direction in metrics.items():
        values = field.get(name)
        if not values or name not in candidate:
            continue
        value = float(candidate[name])
        if direction == "min":
            better = bisect.bisect_left(values, value)
        else:
            better = len(values) - bisect.bisect_right(values, value)
        fraction = better / len(values)
        if fraction > worst:
            worst = fraction
            culprit = name
    return worst, culprit


def without_catastrophic_outliers(
    candidates: Sequence[tuple[str, Mapping[str, float]]],
    field: Mapping[str, list[float]],
    *,
    maximum_percentile: float = 0.90,
    minimum_keep: int = 1,
    metrics: Mapping[str, str] = METRICS,
) -> tuple[list[str], list[tuple[str, float, str]]]:
    """Keep candidates whose weakest metric is not near the bottom of the field.

    Returns the surviving sequences in their input order, plus what was dropped
    and why, so the decision is inspectable rather than silent.
    """
    scored = [
        (sequence, *worst_percentile(record, field, metrics))
        for sequence, record in candidates
    ]
    kept = [
        sequence
        for sequence, fraction, _ in scored
        if fraction <= maximum_percentile
    ]
    dropped = [
        (sequence, fraction, culprit)
        for sequence, fraction, culprit in scored
        if fraction > maximum_percentile
    ]
    if len(kept) < minimum_keep:
        # Keeping the least-bad candidates beats handing back an empty shortlist.
        ordered = sorted(scored, key=lambda item: item[1])
        kept = [sequence for sequence, _, _ in ordered[:minimum_keep]]
        keep_set = set(kept)
        dropped = [item for item in dropped if item[0] not in keep_set]
    return kept, dropped
