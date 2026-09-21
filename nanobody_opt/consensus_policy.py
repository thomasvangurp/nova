"""Shared, dependency-free final-ranking policies for the consensus trial."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

POLICIES = ("tier", "off")


def checked_policy(value: str) -> str:
    if value not in POLICIES:
        raise ValueError(f"unknown consensus ranking policy: {value!r}")
    return value


def order_key(item: Mapping[str, Any], policy: str = "tier") -> tuple:
    checked_policy(policy)
    probability = float(item.get("win_probability") or 0.0)
    if not math.isfinite(probability):
        raise ValueError("ranking probability must be finite")
    return (
        0 if probability > 0.0 else 1,
        int(item.get("settled_deviations", 0)) if policy == "tier" else 0,
        -probability,
    )


def ordered_candidates(candidates: Sequence[Mapping[str, Any]], policy: str) -> list:
    """Reorder frozen estimates; never refit their cohort or probabilities."""
    checked_policy(policy)
    return sorted(candidates, key=lambda item: order_key(item, policy))


def paired_shortlist(candidates: Sequence[Mapping[str, Any]], limit: int) -> list:
    """One capped pool containing both policies' leading candidates.

    Each policy contributes limit / 2 candidates. Shared candidates count
    once; spare capacity is filled by alternating the two full orders. This
    avoids imposing the tier's truncation on its comparator without doubling
    the number of Layer-2, novelty and CDR checks.
    """
    if limit < 2:
        raise ValueError("a paired shortlist needs capacity for both policies")
    orders = [ordered_candidates(candidates, policy) for policy in POLICIES]
    selected: list = []
    seen: set[str] = set()

    def add(item: Mapping[str, Any]) -> None:
        sequence = str(item["sequence"])
        if sequence not in seen and len(selected) < limit:
            selected.append(item)
            seen.add(sequence)

    # An even limit makes the per-policy quota exactly equal.
    if limit % 2:
        raise ValueError("paired shortlist capacity must be even")
    for order in orders:
        for item in order[:limit // 2]:
            add(item)
    for index in range(max(map(len, orders), default=0)):
        for order in orders:
            if index < len(order):
                add(order[index])
        if len(selected) == limit:
            break
    return selected
