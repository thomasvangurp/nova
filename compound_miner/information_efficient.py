"""Value-of-information policies for the compound scoring cascade.

Reduced Boltz is a useful ranking signal, but measured frontier correlation is
not strong enough to make it a hard veto.  Full-score observations are also
expensive enough that passes two and three must be scheduled sequentially:
after every observation, spend the next pass on the molecule most likely to
change the entropy-constrained release portfolio.

This module is deliberately independent of the scorer and database.  Live
tools pass small dictionaries into it, which makes the policy deterministic,
testable, and reusable by both the box producer and offline replay.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


NORMAL_SEED_NOISE = 0.005234
GATE_PASSES = 3


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    name: str
    probability: float
    expected_improvement: float
    standard_error: float
    component_novelty: float
    reason: str


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def posterior_standard_error(
    passes: int,
    observed_sd: float,
    *,
    noise_floor: float = NORMAL_SEED_NOISE,
) -> float:
    """Conservative uncertainty of a molecule's mean full-Boltz score."""
    count = max(1, int(passes))
    empirical = max(0.0, float(observed_sd)) / math.sqrt(count)
    seeded = max(1e-9, float(noise_floor)) / math.sqrt(count)
    return max(empirical, seeded)


def expected_positive_improvement(mean: float, threshold: float, se: float) -> tuple[float, float]:
    """Return ``(P(true > threshold), E[(true-threshold)+])`` for a normal posterior."""
    if se <= 0:
        probability = float(mean > threshold)
        return probability, max(0.0, mean - threshold)
    z = (mean - threshold) / se
    probability = _normal_cdf(z)
    improvement = (mean - threshold) * probability + se * _normal_pdf(z)
    return probability, max(0.0, improvement)


def _component_novelty(components: Sequence[int], incumbent_components: Iterable[Sequence[int]]) -> float:
    values = tuple(int(value) for value in components)
    if not values:
        return 0.0
    used = {int(value) for row in incumbent_components for value in row}
    return sum(value not in used for value in values) / len(values)


def select_confirmation_candidates(
    rows: Iterable[Mapping],
    *,
    threshold: float,
    incumbent_components: Iterable[Sequence[int]] = (),
    cap: int,
    minimum_probability: float = 0.10,
    admission_margin: float = 0.010,
    noise_floor: float = NORMAL_SEED_NOISE,
    information_fraction: float = 0.20,
    diversity_fraction: float = 0.15,
) -> list[ConfirmationDecision]:
    """Choose the *next one* full-score observation for each selected molecule.

    The value arm targets expected portfolio gain.  The information arm keeps
    uncertain near-frontier molecules from being discarded after one unlucky
    draw.  The diversity arm protects chemistry that can replace a redundant
    portfolio member.  The caller must ingest these observations and call this
    function again before scheduling another pass.
    """
    if cap <= 0:
        return []
    incumbents = tuple(tuple(int(value) for value in row) for row in incumbent_components)
    candidates = []
    for row in rows:
        passes = int(row.get("passes", row.get("observation_count", 0)) or 0)
        if passes >= GATE_PASSES:
            continue
        mean = float(row.get("mean", row.get("mean_score", 0.0)) or 0.0)
        sd = float(row.get("sd", row.get("sd_score", 0.0)) or 0.0)
        se = posterior_standard_error(passes, sd, noise_floor=noise_floor)
        probability, improvement = expected_positive_improvement(mean, threshold, se)
        if probability < minimum_probability and mean < threshold - admission_margin:
            continue
        novelty = _component_novelty(row.get("components") or (), incumbents)
        candidates.append({
            "name": str(row["name"]),
            "mean": mean,
            "passes": passes,
            "probability": probability,
            "improvement": improvement,
            "se": se,
            "novelty": novelty,
        })
    if not candidates:
        return []

    target = min(cap, len(candidates))
    information_count = min(target, max(0, round(target * information_fraction)))
    diversity_count = min(
        target - information_count,
        max(0, round(target * diversity_fraction)),
    )
    value_count = target - information_count - diversity_count
    selected: list[ConfirmationDecision] = []
    names: set[str] = set()

    def extend(ordered: Iterable[dict], count: int, reason: str) -> None:
        for item in ordered:
            if len([decision for decision in selected if decision.reason == reason]) >= count:
                break
            if item["name"] in names:
                continue
            names.add(item["name"])
            selected.append(ConfirmationDecision(
                name=item["name"],
                probability=item["probability"],
                expected_improvement=item["improvement"],
                standard_error=item["se"],
                component_novelty=item["novelty"],
                reason=reason,
            ))

    # Reserve the diagnostic seats first.  If value is filled first, a very
    # uncertain candidate can be swallowed by the value arm and the nominal
    # information quota then goes to a less informative row.
    extend(sorted(candidates, key=lambda item: (
        item["se"] * item["probability"], item["se"], item["improvement"]
    ), reverse=True), information_count, "information")
    extend(sorted(candidates, key=lambda item: (
        item["novelty"], item["improvement"], item["probability"]
    ), reverse=True), diversity_count, "diversity")
    extend(sorted(candidates, key=lambda item: (
        item["improvement"], item["probability"], item["mean"]
    ), reverse=True), value_count, "value")
    # If one protected arm had fewer unique candidates than its quota, fill
    # the batch by value rather than wasting an available scoring slot.
    if len(selected) < target:
        extend(sorted(candidates, key=lambda item: (
            item["improvement"], item["se"], item["novelty"]
        ), reverse=True), target - len(selected), "value_fill")
    return selected
