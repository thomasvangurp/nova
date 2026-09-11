"""Leakage-safe priors for allocating terminal VHH mutation routes.

The policy deliberately predicts a mutation *class*, not an exact BoltzGen
score.  Historical public winners supply position and residue frequencies;
all public competitive observations supply a smoothed residue fallback.  The
caller retains a separate legacy/exploration lane because these priors cannot
identify the exact winning parent/haplotype deterministically.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from nanobody_opt.mutation_network import AA_INDEX


@dataclass(frozen=True)
class TerminalPriors:
    """Rank maps learned solely from completed public validator fields."""

    position_rank: dict[int, int]
    residue_rank: dict[int, dict[str, int]]
    exploration_residues: dict[int, frozenset[str]]
    sequence_latest_epoch: dict[str, int]
    sequence_score: dict[str, float]
    winner_step_count: dict[int, int]
    public_epoch_count: int


def _terminal_positions(length: int, terminal_width: int) -> list[int]:
    return [
        *range(min(10, length)),
        *range(max(10, length - terminal_width), length),
    ]


def learn_terminal_priors(
    records: Iterable[dict[str, Any]],
    *,
    length: int,
    terminal_width: int = 17,
    competitive_score: float = 70.0,
    exploration_residues_per_position: int = 5,
) -> TerminalPriors:
    """Learn terminal route ranks without observing the current epoch.

    Only rows explicitly identified as public validator rank sums are used.
    For each completed epoch, its best public sequence is compared with all
    earlier public sequences.  A one-step predecessor contributes one winner
    mutation-position observation.  Residue ranks use the complete winner
    sequence at that position, backed off to competitive public frequency.
    """
    if length < 28 or terminal_width < 1:
        raise ValueError("invalid terminal prior sequence geometry")
    if exploration_residues_per_position < 1:
        raise ValueError("exploration residue count must be positive")
    observations: dict[int, dict[str, float]] = defaultdict(dict)
    latest_epoch: dict[str, int] = {}
    scores_by_sequence: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.get("score_kind") != "validator_final_rank_sum":
            continue
        sequence = str(record.get("sequence", "")).strip().upper()
        epoch = record.get("epoch")
        try:
            score = float(record["score"])
            epoch = int(epoch)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            len(sequence) != length
            or not math.isfinite(score)
            or any(residue not in AA_INDEX for residue in sequence)
        ):
            continue
        incumbent = observations[epoch].get(sequence)
        observations[epoch][sequence] = (
            score if incumbent is None else min(score, incumbent)
        )
        latest_epoch[sequence] = max(epoch, latest_epoch.get(sequence, epoch))
        scores_by_sequence[sequence].append(score)

    positions = _terminal_positions(length, terminal_width)
    winner_step_count: Counter[int] = Counter()
    winner_residue_count: dict[int, Counter[str]] = {
        position: Counter() for position in positions
    }
    competitive_residue_count: dict[int, Counter[str]] = {
        position: Counter() for position in positions
    }
    prior_sequences: list[str] = []
    prior_vectors: list[np.ndarray] = []
    prior_epochs: list[int] = []
    prior_scores: list[float] = []
    for epoch in sorted(observations):
        field = observations[epoch]
        winner, winner_score = min(
            field.items(), key=lambda pair: (pair[1], pair[0])
        )
        for position in positions:
            winner_residue_count[position][winner[position]] += 1
        if prior_vectors:
            winner_vector = np.fromiter(
                (AA_INDEX[residue] for residue in winner),
                dtype=np.uint8,
                count=length,
            )
            matrix = np.asarray(prior_vectors, dtype=np.uint8)
            distances = np.count_nonzero(matrix != winner_vector, axis=1)
            predecessor_indices = np.flatnonzero(
                (np.asarray(prior_epochs) < epoch) & (distances == 1)
            )
            if predecessor_indices.size:
                predecessor = min(
                    predecessor_indices,
                    key=lambda index: (
                        prior_scores[int(index)],
                        -prior_epochs[int(index)],
                        prior_sequences[int(index)],
                    ),
                )
                changed = np.flatnonzero(
                    matrix[int(predecessor)] != winner_vector
                )
                if changed.size == 1 and int(changed[0]) in positions:
                    winner_step_count[int(changed[0])] += 1
        for sequence, score in field.items():
            if score <= competitive_score:
                for position in positions:
                    competitive_residue_count[position][sequence[position]] += 1
            prior_sequences.append(sequence)
            prior_vectors.append(np.fromiter(
                (AA_INDEX[residue] for residue in sequence),
                dtype=np.uint8,
                count=length,
            ))
            prior_epochs.append(epoch)
            prior_scores.append(score)

    ordered_positions = sorted(
        positions,
        key=lambda position: (-winner_step_count[position], position),
    )
    position_rank = {
        position: rank for rank, position in enumerate(ordered_positions, 1)
    }
    residue_rank: dict[int, dict[str, int]] = {}
    exploration_residues: dict[int, frozenset[str]] = {}
    for position in positions:
        residues = set(winner_residue_count[position]) | set(
            competitive_residue_count[position]
        )
        ordered = sorted(
            residues,
            key=lambda residue: (
                -winner_residue_count[position][residue],
                -competitive_residue_count[position][residue],
                residue,
            ),
        )
        residue_rank[position] = {
            residue: rank for rank, residue in enumerate(ordered, 1)
        }
        exploration_residues[position] = frozenset(
            ordered[:exploration_residues_per_position]
        )
    sequence_score = {
        sequence: 0.5 * min(scores) + 0.5 * float(np.median(scores))
        for sequence, scores in scores_by_sequence.items()
    }
    return TerminalPriors(
        position_rank=position_rank,
        residue_rank=residue_rank,
        exploration_residues=exploration_residues,
        sequence_latest_epoch=latest_epoch,
        sequence_score=sequence_score,
        winner_step_count=dict(winner_step_count),
        public_epoch_count=len(observations),
    )


def basin_parent_recency_ranks(
    sequences: Iterable[str], priors: TerminalPriors
) -> dict[str, int]:
    """Rank validated recent parents first; unobserved archive rows last."""
    unique = set(sequences)
    ordered = sorted(
        unique,
        key=lambda sequence: (
            -priors.sequence_latest_epoch.get(sequence, -1),
            priors.sequence_score.get(sequence, math.inf),
            sequence,
        ),
    )
    return {sequence: rank for rank, sequence in enumerate(ordered, 1)}


def terminal_route_priority(
    candidate: str,
    base: str,
    *,
    priors: TerminalPriors,
    base_rank: dict[str, int],
) -> tuple[int, int, int, str]:
    """Order single routes by winner position, target residue, then parent."""
    changed = [
        position
        for position, (before, after) in enumerate(zip(base, candidate, strict=True))
        if before != after
    ]
    if len(changed) != 1:
        return (10**9, 10**9, base_rank.get(base, 10**9), candidate)
    position = changed[0]
    residue = candidate[position]
    return (
        priors.position_rank.get(position, 10**9),
        priors.residue_rank.get(position, {}).get(residue, 10**9),
        base_rank.get(base, 10**9),
        candidate,
    )
