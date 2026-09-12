"""Rank candidates by their chance of drawing a winning score, not by their average.

Only rank one pays, and the validator scores a submission with a single
observation.  Measured on epoch 25062, the molecule that won drew 85 rank points
better than its own three-observation average, against a median competitor
fluctuation of 23, and sat 13th of 46 on averaged quality.  Twelve better
molecules lost to it.  Epochs are decided by the largest upward fluctuation.

That makes spread an asset rather than a defect.  Given a winning threshold ``t``,
a candidate with mean 130 and spread 50 beats one with mean 120 and spread 30,
because P(draw < t) is higher even though its average is worse.  Selecting on the
mean actively discards the candidates most likely to win.

The estimator is the normal-fitted probability that one fresh draw lands below
``t``.  Two reasons not to use the empirical fraction instead: with nine
observations it only resolves in steps of 11%, and it assigns exactly zero to
every candidate that has not yet produced a winning draw -- which is most of
them, leaving the ordering undefined where it matters most.
"""

from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Sequence

# A spread floor, in rank points.  Without it a candidate whose few observations
# happen to coincide gets an infinitely confident verdict in whichever direction
# its mean falls.
MINIMUM_SPREAD = 5.0


def pooled_spread(spreads: Sequence[float]) -> float:
    """A stand-in spread for candidates with too few observations to have one."""
    usable = [value for value in spreads if value and math.isfinite(value) and value > 0]
    if not usable:
        return MINIMUM_SPREAD
    ordered = sorted(usable)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return max(median, MINIMUM_SPREAD)


def observation_spread(scores: Sequence[float]) -> float | None:
    """Draw-to-draw spread of one candidate, or None when it cannot be estimated."""
    if len(scores) < 2:
        return None
    value = stdev(scores)
    return value if math.isfinite(value) else None


def win_probability(
    scores: Sequence[float],
    threshold: float,
    *,
    fallback_spread: float = MINIMUM_SPREAD,
) -> float:
    """Probability that a single fresh draw of this candidate beats ``threshold``.

    ``scores`` are the candidate's per-observation scores, lower being better, so
    the event is ``draw < threshold``.
    """
    if not scores:
        return 0.0
    mean = fmean(scores)
    spread = observation_spread(scores)
    if spread is None or spread <= 0:
        spread = fallback_spread
    spread = max(spread, MINIMUM_SPREAD)
    # Normal CDF without pulling in scipy: the publisher and the scorer run in
    # different environments and this module must import in both.
    return 0.5 * (1.0 + math.erf((threshold - mean) / (spread * math.sqrt(2.0))))


def rank_by_win_probability(
    candidates: Sequence[dict],
    threshold: float,
    *,
    scores_key: str = "per_observation_median_rank_sum",
) -> list[dict]:
    """Annotate candidates with win probability and order them best-first.

    The spread of candidates that have one observation is unknowable, so they
    borrow the cohort's median spread rather than being ranked as if certain.
    """
    spreads = [
        observation_spread(item.get(scores_key) or []) for item in candidates
    ]
    fallback = pooled_spread([value for value in spreads if value is not None])
    annotated = []
    for item, spread in zip(candidates, spreads, strict=True):
        scores = item.get(scores_key) or []
        probability = win_probability(
            scores, threshold, fallback_spread=fallback
        )
        enriched = dict(item)
        enriched["win_probability"] = probability
        enriched["win_threshold"] = threshold
        enriched["observation_spread"] = spread if spread is not None else fallback
        enriched["observation_spread_estimated"] = spread is None
        annotated.append(enriched)
    annotated.sort(
        key=lambda item: (
            -item["win_probability"],
            item.get("score_median", 0.0),
            item.get("sequence", ""),
        )
    )
    return annotated
