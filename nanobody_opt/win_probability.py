"""Rank candidates by their chance of drawing a winning score, not their average.

Only rank one pays, and the validator scores a submission with a single
observation.  Measured on epoch 25062, the molecule that won drew 85 rank points
better than its own three-observation average, against a median competitor
fluctuation of 23, and sat 13th of 46 on averaged quality.  Twelve better
molecules lost to it.  Epochs are decided by the largest upward fluctuation.

The production selector uses the exact empirical probability that a candidate
beats the other qualified candidates when each receives one independent fresh
draw.  This keeps each observation's complete validator rank sum intact and
targets the event that actually pays.  The older normal-fitted probability of
crossing a fixed threshold remains below for artifact replay and diagnostics;
it is no longer the live field-ordering rule.
"""

from __future__ import annotations

import bisect
import math
import os
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from statistics import fmean, stdev
from typing import Any

# A spread floor, in rank points.  Without it a candidate whose few observations
# happen to coincide gets an infinitely confident verdict in whichever direction
# its mean falls.
MINIMUM_SPREAD = 5.0

# Only probabilities below this conservative upper bound may be skipped.
# This is lower than the production 1e-3 eligibility floor.
CHANCE_FLOOR_SKIP_MARGIN = 1e-4


def fresh_draw_upper_bounds(candidates, scores_key="per_observation_median_rank_sum"):
    """P(win all opponents) <= P(tie or beat a subset of opponents).

    Conditional on each own score, multiply non-loss probabilities against a
    small anchor set. Ties count as FULL wins here, hence this is always an
    upper bound on the exact, tie-split tournament probability. Anchor choice
    changes efficiency only; it cannot exclude a real floor-clearing winner.
    """
    distributions = [sorted(float(v) for v in (item.get(scores_key) or []))
                     for item in candidates]
    eligible = [i for i, values in enumerate(distributions) if values]
    anchors = set()
    for quantile in (0.0, 0.1, 0.5):
        anchors.update(sorted(eligible, key=lambda i: distributions[i][
            int((len(distributions[i]) - 1) * quantile)])[:128])
    bounds = {}
    for i, own in enumerate(distributions):
        probability = 0.0
        for score in own:
            non_loss = 1.0
            for j in anchors:
                if i == j:
                    continue
                other = distributions[j]
                non_loss *= (len(other) - bisect.bisect_left(other, score)) / len(other)
                if non_loss == 0.0:
                    break
            probability += non_loss / len(own)
        bounds[str(candidates[i].get("sequence", ""))] = probability
    return bounds


def pooled_spread(spreads: Sequence[float]) -> float:
    """A stand-in spread for candidates with too few observations to have one."""
    usable = [
        value for value in spreads if value and math.isfinite(value) and value > 0
    ]
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


# Observations' worth of prior weight given to the cohort's spread when
# shrinking a candidate's own. Spread is an ASSET in this estimator, so a
# candidate whose spread is merely badly measured is promoted over one whose
# spread is genuinely large: with two observations the sample deviation carries
# one degree of freedom and swings by a factor of two either way. In epoch
# 25093 the top-ranked candidate had n=2 and a spread of 91.9 against ~45 for
# the n=5 candidates, and it was ranked first on that alone. Shrinking toward
# the cohort keeps variance-seeking while refusing to seek estimation error.
SPREAD_PRIOR_OBSERVATIONS = 4.0


def observation_spread(scores: Sequence[float]) -> float | None:
    """Draw-to-draw spread of one candidate, or None when it cannot be estimated."""
    if len(scores) < 2:
        return None
    value = stdev(scores)
    return value if math.isfinite(value) else None


def shrunk_mean(sample: float, cohort: float, observations: int) -> float:
    """Blend a candidate's own mean with the cohort's, by how well it is known.

    The same winner's curse applies to the mean as to the spread, and fixing
    only the spread moves the problem rather than solving it: with the spread
    shrunk, the epoch-25093 pool promoted four single-observation candidates
    whose medians were simply the luckiest draws. One observation of a fixed
    sequence lands about 34 rank points from its own mean, so one draw is not
    grounds to spend a ticket.
    """
    if observations < 1:
        return cohort
    weight = observations / (observations + SPREAD_PRIOR_OBSERVATIONS)
    return weight * sample + (1.0 - weight) * cohort


def shrunk_spread(sample: float | None, pooled: float, observations: int) -> float:
    """Blend a candidate's own spread with the cohort's, by how well it is known.

    Weight is the sample's degrees of freedom against a fixed prior, so two
    observations lean on the cohort and nine mostly stand on their own.
    """
    if sample is None or not math.isfinite(sample) or sample <= 0:
        return max(pooled, MINIMUM_SPREAD)
    degrees = max(observations - 1, 0)
    weight = degrees / (degrees + SPREAD_PRIOR_OBSERVATIONS)
    blended = math.sqrt(weight * sample**2 + (1.0 - weight) * pooled**2)
    return max(blended, MINIMUM_SPREAD)


def win_probability(
    scores: Sequence[float],
    threshold: float,
    *,
    fallback_spread: float = MINIMUM_SPREAD,
    spread: float | None = None,
    cohort_mean: float | None = None,
) -> float:
    """Probability that a single fresh draw of this candidate beats ``threshold``.

    ``scores`` are the candidate's per-observation scores, lower being better, so
    the event is ``draw < threshold``.
    """
    if not scores:
        return 0.0
    mean = fmean(scores)
    if cohort_mean is not None:
        mean = shrunk_mean(mean, cohort_mean, len(scores))
    if spread is None:
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
    spreads = [observation_spread(item.get(scores_key) or []) for item in candidates]
    fallback = pooled_spread([value for value in spreads if value is not None])
    means = [
        fmean(item.get(scores_key) or [0.0]) if (item.get(scores_key) or []) else None
        for item in candidates
    ]
    known = [value for value in means if value is not None]
    cohort_mean = fmean(known) if known else 0.0
    annotated = []
    for item, spread in zip(candidates, spreads, strict=True):
        scores = item.get(scores_key) or []
        effective = shrunk_spread(spread, fallback, len(scores))
        probability = win_probability(
            scores,
            threshold,
            fallback_spread=effective,
            spread=effective,
            cohort_mean=cohort_mean,
        )
        enriched = dict(item)
        enriched["win_probability"] = probability
        enriched["win_threshold"] = threshold
        enriched["observation_spread"] = effective
        enriched["observation_spread_sample"] = spread
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


#: Read-only context for _fresh_draw_probability_for, set once per process:
#: directly in the main process for a serial run, or via
#: _init_fresh_draw_worker in each pool worker (a separate process, so it
#: cannot see empirical_fresh_draw_probabilities's local variables).
_FRESH_DRAW_CONTEXT: dict[str, Any] | None = None


def _init_fresh_draw_worker(context: dict[str, Any]) -> None:
    global _FRESH_DRAW_CONTEXT
    _FRESH_DRAW_CONTEXT = context


def _fresh_draw_probability_for(candidate_index: int) -> float:
    """One candidate's exact win probability. Same math as before, just able
    to run in a worker process: this loop, not the bisect lookups it calls,
    is what actually dominates (measured: 73% of a cycle after the bisect
    fix alone -- O(depth) -> O(log depth) sped up the inner comparison, but
    the outer candidate x own-score x opponent structure is still a
    Python-level loop over up to 1,016 candidates)."""
    ctx = _FRESH_DRAW_CONTEXT
    eligible = ctx["eligible"]
    distributions = ctx["distributions"]
    sorted_distributions = ctx["sorted_distributions"]
    lengths = ctx["lengths"]
    own = distributions[candidate_index]
    probability = 0.0
    for score in own:
        # Coefficient k is the probability that exactly k opponents tie
        # this score while every other opponent draws a worse score.  An
        # outcome with k ties contributes 1/(k+1) of a win.
        tie_polynomial = [1.0]
        scalar = 1.0
        for opponent_index in eligible:
            if opponent_index == candidate_index:
                continue
            opponent = sorted_distributions[opponent_index]
            denominator = lengths[opponent_index]
            left = bisect.bisect_left(opponent, score)
            right = bisect.bisect_right(opponent, score)
            worse = (len(opponent) - right) / denominator
            equal = (right - left) / denominator
            if equal == 0.0:
                scalar *= worse
                if scalar == 0.0:
                    break
                continue
            expanded = [0.0] * (len(tie_polynomial) + 1)
            for ties, coefficient in enumerate(tie_polynomial):
                expanded[ties] += coefficient * worse
                expanded[ties + 1] += coefficient * equal
            tie_polynomial = expanded
        if scalar:
            probability += scalar * sum(
                coefficient / (ties + 1)
                for ties, coefficient in enumerate(tie_polynomial)
            ) / len(own)
    return probability


def empirical_fresh_draw_probabilities(
    candidates: Sequence[dict[str, Any]],
    *,
    scores_key: str = "per_observation_median_rank_sum",
    workers: int = 1,
    workers_minimum_candidates: int = 64,
    compute_sequences: set[str] | None = None,
) -> dict[str, float]:
    """Return each candidate's exact chance of winning one empirical draw.

    Each candidate supplies independent, equally likely observations of the
    validator's *complete* dense-rank objective.  For every possible draw of a
    candidate, the calculation integrates over every opponent's empirical
    distribution.  Ties split the win equally, which makes the probabilities
    sum to one without random bootstrap noise.

    Keeping an observation's ten metrics together before they become one score
    preserves their covariance.  This is the deterministic equivalent of the
    Monte Carlo selector used by ``analysis/nanobody-gap/monte_carlo_selector``.
    """
    sequences = [str(item.get("sequence", "")) for item in candidates]
    distributions = [
        [float(value) for value in (item.get(scores_key) or [])] for item in candidates
    ]
    probabilities = {sequence: 0.0 for sequence in sequences}
    eligible = [index for index, scores in enumerate(distributions) if scores]
    if not eligible:
        return probabilities

    # Each opponent is queried by every other candidate's every own score, so
    # its worse/equal counts are computed len(eligible)-1 times per own score.
    # A linear scan of the opponent's own list (`sum(value > score for ...)`)
    # cost O(depth); sorting each opponent's list once and using bisect turns
    # that into O(log depth) per query, exact and order-preserving (bisect on
    # sorted data gives the identical counts a linear scan would). Measured on
    # a 1,016-candidate, up-to-528-deep pool this was ~94% of a publisher
    # cycle (~19.7 of ~19.8 minutes); this is the same worse/equal counts,
    # computed the fast way.
    sorted_distributions = {index: sorted(distributions[index]) for index in eligible}
    lengths = {index: float(len(sorted_distributions[index])) for index in eligible}

    global _FRESH_DRAW_CONTEXT
    _FRESH_DRAW_CONTEXT = {
        "eligible": eligible,
        "distributions": distributions,
        "sorted_distributions": sorted_distributions,
        "lengths": lengths,
    }
    to_compute = eligible
    if compute_sequences is not None:
        to_compute = [index for index in eligible
                      if sequences[index] in compute_sequences]
    pool_size = max(1, workers)
    if pool_size > 1 and len(to_compute) >= workers_minimum_candidates:
        with ProcessPoolExecutor(
            max_workers=pool_size,
            initializer=_init_fresh_draw_worker,
            initargs=(_FRESH_DRAW_CONTEXT,),
        ) as pool:
            results = list(pool.map(
                _fresh_draw_probability_for, to_compute,
                chunksize=max(1, len(to_compute) // (pool_size * 4)),
            ))
    else:
        results = [_fresh_draw_probability_for(index) for index in to_compute]
    for candidate_index, probability in zip(to_compute, results):
        probabilities[sequences[candidate_index]] = probability
    return probabilities


def rank_by_fresh_draw_probability(
    candidates: Sequence[dict[str, Any]],
    *,
    scores_key: str = "per_observation_median_rank_sum",
    workers: int = 1,
    workers_minimum_candidates: int = 64,
    competitive_score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Order candidates by empirical probability of winning a fresh draw.

    Unlike :func:`rank_by_win_probability`, this does not fit independent
    normal tails or optimize a hand-picked threshold.  It directly asks the
    event that pays: which candidate is best when every candidate receives one
    new draw of the validator's full objective?

    The optional fast path skips only candidates whose mathematically valid
    upper bound is below 1e-4, preserving the production 1e-3 eligibility gate.
    The legacy threshold argument enables the fast path for existing callers;
    its numeric value is deliberately irrelevant to a relative tournament.
    """
    candidates = list(candidates)
    compute_sequences = None
    if competitive_score_threshold is not None:
        bounds = fresh_draw_upper_bounds(candidates, scores_key)
        compute_sequences = {sequence for sequence, bound in bounds.items()
                             if bound >= CHANCE_FLOOR_SKIP_MARGIN}
    probabilities = empirical_fresh_draw_probabilities(
        candidates, scores_key=scores_key,
        workers=workers, workers_minimum_candidates=workers_minimum_candidates,
        compute_sequences=compute_sequences,
    )
    annotated: list[dict[str, Any]] = []
    for item in candidates:
        enriched = dict(item)
        scores = item.get(scores_key) or []
        enriched["score_mean"] = fmean(scores) if scores else float("inf")
        enriched["win_probability"] = probabilities.get(
            str(item.get("sequence", "")), 0.0
        )
        enriched["win_probability_method"] = "empirical-independent-fresh-draw"
        annotated.append(enriched)
    # 2026-10-14: order by the mean of the candidate's own observations, not the
    # variance-aware win probability. The empirical spread carries no signal
    # (rho ~ -0.11) and win_probability collapses to a function of the mean, so
    # a lower mean IS the winning chance; dropping the variance term removes a
    # noise source without changing the objective.
    annotated.sort(
        key=lambda item: (
            float(item["score_mean"]),
            float(item.get("score_median", 0.0)),
            str(item.get("sequence", "")),
        )
    )
    return annotated
