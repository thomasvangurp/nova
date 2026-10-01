#!/usr/bin/env python3
"""Frozen, dependency-free semantics for the two offline proof tests."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TARGET = "P20809"
METRICS = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
    "liability_score": "min",
    "liability_num_violations": "min",
}
NONINFERIORITY_METRICS = {
    "liability_score": "min",
    "liability_num_violations": "min",
    "plip_hbonds_refolded": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "delta_sasa_refolded": "max",
}
PREFERRED_IDENTITY = 0.94
VALIDATOR_DUPLICATE_IDENTITY = 0.95
STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sequence_identity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))


def valid_nanobody_sequence(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    sequence = value.strip().upper()
    return 80 <= len(sequence) <= 180 and set(sequence) <= STANDARD_AMINO_ACIDS


def dedupe_field_entries(entries: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Match rank_exact field semantics: one complete record per sequence."""
    unique: dict[str, dict[str, Any]] = {}
    for raw in entries:
        sequence = str(raw.get("sequence") or "").strip().upper()
        if not valid_nanobody_sequence(sequence):
            raise ValueError("field contains an invalid sequence")
        row = dict(raw)
        row["sequence"] = sequence
        prior = unique.get(sequence)
        if prior is not None:
            if prior.get("metrics") != row.get("metrics"):
                raise ValueError("duplicate sequence carries inconsistent metrics")
            continue
        unique[sequence] = row
    return list(unique.values()), {
        "raw_complete_entry_count": len(entries),
        "unique_sequence_entry_count": len(unique),
        "duplicate_sequence_count": len(entries) - len(unique),
    }


def closest_identity(sequence: str, chosen: Sequence[str]) -> float:
    return max((sequence_identity(sequence, item) for item in chosen), default=0.0)


def live_order_fill(
    ordered: Sequence[str], count: int, *, preselected: Sequence[str] = ()
) -> list[str]:
    """Mirror fleet.distinct_sequences with value_share=0 and a preseed.

    The live selector first prefers pairs below 0.94, then fills farthest-first
    while never crossing the validator's hard duplicate line at 0.95.
    """
    chosen = list(dict.fromkeys(item.strip().upper() for item in preselected))
    eligible = list(dict.fromkeys(item.strip().upper() for item in ordered if item))
    for sequence in eligible:
        if len(chosen) == count:
            break
        if sequence in chosen:
            continue
        if closest_identity(sequence, chosen) < PREFERRED_IDENTITY:
            chosen.append(sequence)
    remaining = [item for item in eligible if item not in chosen]
    while remaining and len(chosen) < count:
        pick = min(remaining, key=lambda item: closest_identity(item, chosen))
        if closest_identity(pick, chosen) >= VALIDATOR_DUPLICATE_IDENTITY:
            break
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def identity_audit(sequences: Sequence[str]) -> dict[str, Any]:
    pairs = [
        (left, right, sequence_identity(sequences[left], sequences[right]))
        for left in range(len(sequences))
        for right in range(left + 1, len(sequences))
    ]
    return {
        "count": len(sequences),
        "max_positional_identity": max((row[2] for row in pairs), default=0.0),
        "pairs_at_or_above_preferred_0_94": sum(
            value >= PREFERRED_IDENTITY for _, _, value in pairs
        ),
        "pairs_at_or_above_validator_0_95": sum(
            value >= VALIDATOR_DUPLICATE_IDENTITY for _, _, value in pairs
        ),
    }


def dense_rank_sums(records: Sequence[Mapping[str, float]]) -> list[int]:
    totals = [0] * len(records)
    for metric, direction in METRICS.items():
        values = sorted(
            {float(record[metric]) for record in records},
            reverse=(direction == "max"),
        )
        ranks = {value: index + 1 for index, value in enumerate(values)}
        for index, record in enumerate(records):
            totals[index] += ranks[float(record[metric])]
    return totals


def joint_field_outcomes(
    fleet_metrics: Sequence[Mapping[str, float]],
    rivals: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Rank the entire arm jointly with rivals, exactly once per field.

    Multiple arm entries may tie/beat the best rival. Own entries compete for
    the global top-ten slots. This is deliberately not candidate insertion.
    """
    if not fleet_metrics or not rivals:
        raise ValueError("joint outcome requires arm entries and rivals")
    records = [*rivals, *fleet_metrics]
    scores = dense_rank_sums(records)
    split = len(rivals)
    rival_scores = scores[:split]
    arm_scores = scores[split:]
    top10_cut = sorted(scores)[min(9, len(scores) - 1)]
    best_rival = min(rival_scores)
    return {
        "arm_rank_sums": arm_scores,
        "best_rival_rank_sum": best_rival,
        "global_top10_cut": top10_cut,
        "wins": [score <= best_rival for score in arm_scores],
        "top10": [score <= top10_cut for score in arm_scores],
    }


def load_draw(path: Path, expected_count: int) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol: {path}")
    if int(payload.get("n", -1)) != expected_count:
        raise RuntimeError(f"wrong candidate count: {path}")
    found: dict[str, dict[str, float]] = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            if not all(metric in metrics for metric in METRICS):
                raise RuntimeError(f"incomplete ten-metric vector for {sequence}")
            found[sequence] = {
                metric: float(metrics[metric]) for metric in METRICS
            }
    if len(found) != expected_count:
        raise RuntimeError(f"draw sequence count mismatch: {path}")
    return found


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def favorable_delta(left: float, right: float, direction: str) -> float:
    return left - right if direction == "max" else right - left


def median_absolute_deviation(values: Sequence[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)
