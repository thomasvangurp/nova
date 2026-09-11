"""Structural-core identities and historical basin evidence.

The validator field shows that N- and C-terminal variants often share a
binding core while retaining materially different measured outcomes.  A basin
is therefore an allocation group, not a claim that its members are equivalent.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from statistics import median
from typing import Any, Iterable


DEFAULT_N_TERMINAL_WIDTH = 10
DEFAULT_C_TERMINAL_WIDTH = 17


def structural_core(
    sequence: str,
    *,
    n_terminal_width: int = DEFAULT_N_TERMINAL_WIDTH,
    c_terminal_width: int = DEFAULT_C_TERMINAL_WIDTH,
) -> str:
    """Mask terminal haplotypes while preserving the putative binding core."""
    normalized = sequence.strip().upper()
    if n_terminal_width < 0 or c_terminal_width < 0:
        raise ValueError("terminal widths cannot be negative")
    if len(normalized) <= n_terminal_width + c_terminal_width:
        return normalized
    end = len(normalized) - c_terminal_width if c_terminal_width else len(normalized)
    return normalized[n_terminal_width:end]


def basin_id(sequence: str) -> str:
    """Return a compact stable identifier for a sequence's structural core."""
    core = structural_core(sequence)
    return hashlib.sha256(core.encode("ascii")).hexdigest()[:16]


def terminal_haplotype(sequence: str) -> str:
    normalized = sequence.strip().upper()
    if len(normalized) <= DEFAULT_N_TERMINAL_WIDTH + DEFAULT_C_TERMINAL_WIDTH:
        return normalized
    return (
        normalized[:DEFAULT_N_TERMINAL_WIDTH]
        + normalized[-DEFAULT_C_TERMINAL_WIDTH:]
    )


def rank_basin_evidence(
    observations: Iterable[dict[str, Any]],
    *,
    recent_epoch_window: int = 16,
    strong_score: float = 70.0,
    very_strong_score: float = 55.0,
) -> dict[str, dict[str, Any]]:
    """Rank cores using only evidence available before the current epoch.

    The ordering is the leakage-free rule validated in the recent replay:
    recent competitive density, all-time elite density, all-time competitive
    density, best score, recency, then median score.
    """
    if recent_epoch_window < 0:
        raise ValueError("recent_epoch_window cannot be negative")
    rows = []
    for observation in observations:
        sequence = str(observation.get("sequence") or "").strip().upper()
        try:
            score = float(observation["score"])
            epoch = int(observation["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if not sequence or not math.isfinite(score):
            continue
        rows.append((sequence, score, epoch))
    recent_epochs = set(
        sorted({epoch for _, _, epoch in rows}, reverse=True)[:recent_epoch_window]
    )
    grouped: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for sequence, score, epoch in rows:
        grouped[basin_id(sequence)].append((sequence, score, epoch))

    evidence: dict[str, dict[str, Any]] = {}
    for identifier, items in grouped.items():
        scores = [score for _, score, _ in items]
        sequences = {sequence for sequence, _, _ in items}
        evidence[identifier] = {
            "basin_core_id": identifier,
            "basin_recent_strong_count": sum(
                epoch in recent_epochs and score <= strong_score
                for _, score, epoch in items
            ),
            "basin_very_strong_count": sum(score <= very_strong_score for score in scores),
            "basin_strong_count": sum(score <= strong_score for score in scores),
            "basin_best_score": min(scores),
            "basin_median_score": float(median(scores)),
            "basin_latest_epoch": max(epoch for _, _, epoch in items),
            "basin_observation_count": len(items),
            "basin_sequence_count": len(sequences),
            "basin_terminal_haplotype_count": len(
                {terminal_haplotype(sequence) for sequence in sequences}
            ),
        }

    ordered = sorted(
        evidence.values(),
        key=lambda item: (
            -int(item["basin_recent_strong_count"]),
            -int(item["basin_very_strong_count"]),
            -int(item["basin_strong_count"]),
            float(item["basin_best_score"]),
            -int(item["basin_latest_epoch"]),
            float(item["basin_median_score"]),
            str(item["basin_core_id"]),
        ),
    )
    for rank, item in enumerate(ordered, start=1):
        item["basin_rank"] = rank
    return evidence
