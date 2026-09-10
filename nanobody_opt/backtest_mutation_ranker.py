"""Leakage-free retrospective for learned nanobody proposal allocation.

The learned model is a budget allocator, not a replacement for Layer-2 or
BoltzGen.  It predicts the validator's ten within-epoch metric percentiles for
candidate/parent pairs, then assigns exactly 160 expensive evaluations while
retaining parent coverage and a small deterministic exploration lane.

The chronological protocol is deliberately strict:

* warm-up epochs supply prior parents but are never training examples;
* model choice is made inside the training period;
* the final holdout is never used for features, model choice, or proposals;
* every holdout proposal uses only fields completed before that epoch;
* held-out submissions are added only to a labelled oracle ranking pool.  This
  measures ranker recall separately from organic proposal discovery.

Every material intermediate is written to the result directory so a later
experiment can change one assumption while reusing the exact same snapshot.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nanobody_opt.analyze_epoch_winners import (
    COMMIT_API,
    fetch_epoch_commits,
)
from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT
from nanobody_opt.regions import find_regions
from nanobody_opt.select_elite_parents import _epoch_candidates, _fetch
from nanobody_opt.simulate_round import METRICS, dense_rank_sums


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
DEFAULT_RANDOM_SEED = 20260910
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "extra_trees_metrics": {
        "family": "extra_trees",
        "target": "metric_percentiles",
        "n_estimators": 192,
        "max_features": 0.35,
        "min_samples_leaf": 2,
    },
    "random_forest_metrics": {
        "family": "random_forest",
        "target": "metric_percentiles",
        "n_estimators": 192,
        "max_features": 0.25,
        "min_samples_leaf": 2,
        "max_samples": 0.80,
    },
    "extra_trees_rank": {
        "family": "extra_trees",
        "target": "final_rank_percentile",
        "n_estimators": 192,
        "max_features": 0.35,
        "min_samples_leaf": 2,
    },
    "random_forest_rank": {
        "family": "random_forest",
        "target": "final_rank_percentile",
        "n_estimators": 192,
        "max_features": 0.25,
        "min_samples_leaf": 2,
        "max_samples": 0.80,
    },
    "ridge_rank": {
        "family": "ridge",
        "target": "final_rank_percentile",
        "alpha": 100.0,
    },
}

LIVE_ACCEPTANCE = {
    "minimum_internal_top5_rate": 0.35,
    "minimum_internal_median_field_spearman": 0.15,
    "maximum_internal_median_winner_percentile": 0.25,
}


@dataclass(frozen=True)
class Parent:
    sequence: str
    epoch: int
    final_score: float
    metrics: tuple[float, ...]


@dataclass(frozen=True)
class Proposal:
    sequence: str
    parent: Parent
    source: str
    edit_distance: int
    mutations: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def fetch_api_fields(
    epochs: Iterable[int], *, target: str, api_root: str, workers: int
) -> dict[int, pd.DataFrame]:
    """Fetch exact public validator fields concurrently."""
    fields: dict[int, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch, api_root, int(epoch)): int(epoch)
            for epoch in epochs
        }
        for future in as_completed(futures):
            epoch, payload = future.result()
            candidates = _epoch_candidates(payload, target)
            if not candidates:
                continue
            rows = []
            for item in candidates:
                rows.append({
                    "epoch": epoch,
                    "sequence": item["sequence"],
                    "final_score": item["validator_final_score"],
                    **item["metrics"],
                })
            # Duplicate submissions receive identical copied metrics.  Keep one
            # training observation per molecule so duplicate miners do not
            # silently up-weight a sequence.
            fields[epoch] = (
                pd.DataFrame(rows)
                .sort_values(["final_score", "sequence"], kind="stable")
                .drop_duplicates("sequence", keep="first")
                .reset_index(drop=True)
            )
    return fields


def percentile_targets(frame: pd.DataFrame) -> np.ndarray:
    """Return ten best-is-zero dense-rank percentiles for one live field."""
    denominator = max(len(frame) - 1, 1)
    columns = []
    for metric, direction in METRICS.items():
        rank = pd.to_numeric(frame[metric], errors="raise").rank(
            method="dense", ascending=(direction == "min")
        )
        columns.append(((rank - 1.0) / denominator).to_numpy(dtype=np.float32))
    return np.column_stack(columns)


def rank_field(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["reconstructed_rank_sum"] = dense_rank_sums(ranked, METRICS)
    ranked["actual_field_rank"] = ranked["final_score"].rank(
        method="min", ascending=True
    ).astype(int)
    return ranked.sort_values(
        ["final_score", "sequence"], kind="stable"
    ).reset_index(drop=True)


def select_parents(
    prior_fields: list[tuple[int, pd.DataFrame]],
    *,
    recent_epochs: int = 16,
    per_epoch: int = 2,
    limit: int = 32,
) -> list[Parent]:
    """Select unique public leaders using only preceding fields."""
    selected: list[Parent] = []
    seen: set[str] = set()
    for epoch, frame in prior_fields[-recent_epochs:]:
        unique = rank_field(frame).drop_duplicates("sequence", keep="first")
        for _, row in unique.head(per_epoch).iterrows():
            sequence = str(row.sequence)
            if sequence in seen:
                continue
            seen.add(sequence)
            selected.append(Parent(
                sequence=sequence,
                epoch=int(epoch),
                final_score=float(row.final_score),
                metrics=tuple(float(row[name]) for name in METRICS),
            ))
    # Preserve two families per round, but allocate parent capacity by actual
    # quality across the complete current-valid bank rather than file recency.
    selected.sort(key=lambda item: (item.final_score, -item.epoch, item.sequence))
    return selected[:limit]


def nearest_parent(sequence: str, parents: list[Parent]) -> Parent:
    compatible = [item for item in parents if len(item.sequence) == len(sequence)]
    if not compatible:
        raise ValueError(f"no same-length parent for sequence length {len(sequence)}")
    return min(
        compatible,
        key=lambda item: (
            sum(left != right for left, right in zip(sequence, item.sequence)),
            item.final_score,
            -item.epoch,
            item.sequence,
        ),
    )


def mutation_description(parent: str, candidate: str) -> tuple[int, str]:
    changes = [
        f"{before}{index + 1}{after}"
        for index, (before, after) in enumerate(zip(parent, candidate))
        if before != after
    ]
    return len(changes), ";".join(changes)


def _region_counts(parent: str, candidate: str) -> list[float]:
    try:
        regions = find_regions(parent)
        sets = [
            set(range(*regions.cdr1)),
            set(range(*regions.cdr2)),
            set(range(*regions.cdr3)),
        ]
    except ValueError:
        sets = [set(), set(), set()]
    changed = {
        index
        for index, (left, right) in enumerate(zip(parent, candidate))
        if left != right
    }
    cdr = [len(changed & positions) for positions in sets]
    terminal = len(changed & (set(range(10)) | set(range(max(len(parent) - 17, 0), len(parent)))))
    framework = len(changed) - len(changed & set().union(*sets))
    return [float(*[value]) for value in [*cdr, framework, terminal]]


def feature_matrix(
    sequences: list[str], parents: list[Parent], *, length: int
) -> np.ndarray:
    """Encode candidate identity plus explicit parent/mutation context."""
    if len(sequences) != len(parents):
        raise ValueError("sequences and parents must be parallel")
    one_hot = length * len(AMINO_ACIDS)
    # candidate one-hot, mutation mask, parent residue code, composition,
    # parent raw metrics, and route summary.
    width = one_hot + length + length + len(AMINO_ACIDS) + len(METRICS) + 8
    matrix = np.zeros((len(sequences), width), dtype=np.float32)
    for row, (sequence, parent) in enumerate(zip(sequences, parents, strict=True)):
        if len(sequence) != length or len(parent.sequence) != length:
            raise ValueError("feature sequence has wrong length")
        offset = 0
        for position, residue in enumerate(sequence):
            matrix[row, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue]] = 1.0
        offset += one_hot
        for position, (before, after) in enumerate(zip(parent.sequence, sequence)):
            if before != after:
                matrix[row, offset + position] = 1.0
        offset += length
        for position, residue in enumerate(parent.sequence):
            matrix[row, offset + position] = AA_INDEX[residue] / (len(AMINO_ACIDS) - 1)
        offset += length
        counts = Counter(sequence)
        for index, residue in enumerate(AMINO_ACIDS):
            matrix[row, offset + index] = counts[residue] / length
        offset += len(AMINO_ACIDS)
        matrix[row, offset : offset + len(METRICS)] = parent.metrics
        offset += len(METRICS)
        edit_distance, _ = mutation_description(parent.sequence, sequence)
        summary = [
            parent.final_score,
            float(edit_distance),
            * _region_counts(parent.sequence, sequence),
            float(parent.epoch),
        ]
        matrix[row, offset : offset + len(summary)] = summary
    return matrix


def training_examples(
    epoch_fields: list[tuple[int, pd.DataFrame]],
    *,
    context_epochs: int,
    train_epochs: int,
    length: int,
) -> tuple[list[str], list[Parent], np.ndarray, np.ndarray, list[int]]:
    sequences: list[str] = []
    parents: list[Parent] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    epochs: list[int] = []
    stop = context_epochs + train_epochs
    for index in range(context_epochs, stop):
        epoch, frame = epoch_fields[index]
        bank = select_parents(epoch_fields[:index])
        if not bank:
            continue
        usable = frame[frame.sequence.str.len() == length].reset_index(drop=True)
        if usable.empty:
            continue
        target = percentile_targets(usable)
        actual = rank_field(usable).set_index("sequence")["actual_field_rank"]
        for row_index, sequence in enumerate(usable.sequence.astype(str)):
            sequences.append(sequence)
            parents.append(nearest_parent(sequence, bank))
            percentile = (float(actual[sequence]) - 1.0) / max(len(usable) - 1, 1)
            targets.append(np.append(target[row_index], percentile))
            weights.append(1.0 + 4.0 * math.exp(-5.0 * percentile))
            epochs.append(epoch)
    return (
        sequences,
        parents,
        np.asarray(targets, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        epochs,
    )


def fit_model(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    seed: int,
    workers: int,
):
    # Imported lazily: pure proposal/split tests do not require scikit-learn.
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    spec = dict(MODEL_SPECS[name])
    family = spec.pop("family")
    target = spec.pop("target")
    fitted_target = y[:, : len(METRICS)] if target == "metric_percentiles" else y[:, -1]
    common = dict(
        random_state=seed,
        n_jobs=workers,
        **spec,
    )
    if family == "extra_trees":
        model = ExtraTreesRegressor(**common)
    elif family == "random_forest":
        model = RandomForestRegressor(bootstrap=True, **common)
    elif family == "ridge":
        common.pop("random_state")
        common.pop("n_jobs")
        model = make_pipeline(
            StandardScaler(with_mean=False),
            Ridge(solver="lsqr", max_iter=4000, **common),
        )
    else:
        raise ValueError(f"unsupported model family {family}")
    if family == "ridge":
        model.fit(x, fitted_target, ridge__sample_weight=weights)
    else:
        model.fit(x, fitted_target, sample_weight=weights)
    return model


def prediction_scores(model: Any, x: np.ndarray) -> np.ndarray:
    predicted = np.asarray(model.predict(x), dtype=np.float64)
    return np.clip(predicted, 0.0, 1.0).sum(axis=1)


def evaluate_fields(
    model: Any,
    epoch_fields: list[tuple[int, pd.DataFrame]],
    *,
    start_index: int,
    count: int,
    length: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    predictions = []
    for index in range(start_index, start_index + count):
        epoch, frame = epoch_fields[index]
        usable = rank_field(
            frame[frame.sequence.str.len() == length].reset_index(drop=True)
        )
        bank = select_parents(epoch_fields[:index])
        sequences = usable.sequence.astype(str).tolist()
        parents = [nearest_parent(sequence, bank) for sequence in sequences]
        scores = prediction_scores(
            model, feature_matrix(sequences, parents, length=length)
        )
        order = pd.Series(scores).rank(method="min", ascending=True).astype(int)
        actual = usable.actual_field_rank.to_numpy(dtype=int)
        actual_winner = int(np.argmin(usable.final_score.to_numpy(dtype=float)))
        actual_rank_series = pd.Series(actual, dtype=float).rank(method="average")
        predicted_rank_series = pd.Series(scores).rank(method="average")
        rho = float(actual_rank_series.corr(predicted_rank_series, method="pearson"))
        summaries.append({
            "phase": phase,
            "epoch": epoch,
            "field_size": len(usable),
            "winner_sequence": sequences[actual_winner],
            "winner_actual_score": float(usable.iloc[actual_winner].final_score),
            "winner_predicted_rank": int(order.iloc[actual_winner]),
            "winner_predicted_percentile": (
                (int(order.iloc[actual_winner]) - 1) / max(len(usable) - 1, 1)
            ),
            "field_spearman": rho,
            "maximum_parent_epoch": max(parent.epoch for parent in parents),
        })
        for row_index, sequence in enumerate(sequences):
            edit_distance, mutations = mutation_description(
                parents[row_index].sequence, sequence
            )
            predictions.append({
                "phase": phase,
                "epoch": epoch,
                "sequence": sequence,
                "actual_score": float(usable.iloc[row_index].final_score),
                "actual_field_rank": int(actual[row_index]),
                "predicted_score": float(scores[row_index]),
                "predicted_field_rank": int(order.iloc[row_index]),
                "parent_sequence": parents[row_index].sequence,
                "parent_epoch": parents[row_index].epoch,
                "parent_score": parents[row_index].final_score,
                "edit_distance": edit_distance,
                "mutations": mutations,
            })
    return pd.DataFrame(summaries), pd.DataFrame(predictions)


def choose_model(summary_by_model: dict[str, pd.DataFrame]) -> str:
    """Choose only on the internal validation tail; holdout is untouched."""
    def key(name: str) -> tuple[float, float, float, str]:
        frame = summary_by_model[name]
        return (
            float(frame.winner_predicted_percentile.median()),
            -float((frame.winner_predicted_rank <= 5).mean()),
            -float(frame.field_spearman.median()),
            name,
        )
    return min(summary_by_model, key=key)


def live_acceptance(summary: pd.DataFrame) -> tuple[bool, dict[str, bool]]:
    """Fail closed using development-period evidence only."""
    checks = {
        "top5_rate": float((summary.winner_predicted_rank <= 5).mean())
        >= LIVE_ACCEPTANCE["minimum_internal_top5_rate"],
        "median_field_spearman": float(summary.field_spearman.median())
        >= LIVE_ACCEPTANCE["minimum_internal_median_field_spearman"],
        "median_winner_percentile": float(
            summary.winner_predicted_percentile.median()
        ) <= LIVE_ACCEPTANCE["maximum_internal_median_winner_percentile"],
    }
    return all(checks.values()), checks


def _proposal(
    parent: Parent, sequence: str, source: str
) -> Proposal | None:
    if sequence == parent.sequence:
        return None
    edit_distance, mutations = mutation_description(parent.sequence, sequence)
    return Proposal(sequence, parent, source, edit_distance, mutations)


def generate_proposals(
    parents: list[Parent],
    donors: list[str],
    *,
    historical: set[str],
    length: int,
    random_candidates_per_parent: int,
    seed: int,
) -> list[Proposal]:
    """Generate a broad, bounded prior-only recombination pool."""
    compatible_donors = [sequence for sequence in donors if len(sequence) == length]
    residue_counts = [Counter(sequence[pos] for sequence in compatible_donors) for pos in range(length)]
    variable = [pos for pos, counts in enumerate(residue_counts) if len(counts) > 1]
    proposals: dict[tuple[str, str], Proposal] = {}

    def add(item: Proposal | None) -> None:
        if item is None or item.sequence in historical:
            return
        if len(item.sequence) != length or any(aa not in AA_INDEX for aa in item.sequence):
            return
        proposals.setdefault((item.parent.sequence, item.sequence), item)

    # Complete one-edit coverage: the model, rather than residue frequency,
    # decides which parent/mutation pairs consume exact slots.
    for parent in parents:
        for position in variable:
            for residue in sorted(residue_counts[position]):
                if residue != parent.sequence[position]:
                    sequence = (
                        parent.sequence[:position]
                        + residue
                        + parent.sequence[position + 1 :]
                    )
                    add(_proposal(parent, sequence, "exhaustive_single"))

    # Coherent donor shells preserve epistatic residue combinations when the
    # transfer is at most six edits from an elite parent.
    for parent in parents:
        for donor in compatible_donors:
            changes = [
                position for position in variable
                if donor[position] != parent.sequence[position]
            ]
            if 2 <= len(changes) <= 6:
                add(_proposal(parent, donor, "coherent_donor_shell"))
            for terminal_name, positions in (
                ("n_terminal_shell", range(0, min(10, length))),
                ("c_terminal_shell", range(max(0, length - 17), length)),
            ):
                selected = [pos for pos in positions if donor[pos] != parent.sequence[pos]]
                if 1 <= len(selected) <= 6:
                    candidate = list(parent.sequence)
                    for position in selected:
                        candidate[position] = donor[position]
                    add(_proposal(parent, "".join(candidate), terminal_name))

    # Deterministic stochastic coverage of the enormous 2..6-edit positional
    # space.  Only earlier-field residue alphabets are visible here.
    rng = random.Random(seed)
    for parent in parents:
        mutable = [
            position for position in variable
            if any(residue != parent.sequence[position] for residue in residue_counts[position])
        ]
        if len(mutable) < 2:
            continue
        for _ in range(random_candidates_per_parent):
            depth = rng.choices([2, 3, 4, 5, 6], weights=[30, 26, 20, 14, 10], k=1)[0]
            depth = min(depth, len(mutable))
            positions = rng.sample(mutable, depth)
            candidate = list(parent.sequence)
            for position in positions:
                options, weights = zip(*[
                    (residue, count)
                    for residue, count in sorted(residue_counts[position].items())
                    if residue != parent.sequence[position]
                ])
                candidate[position] = rng.choices(options, weights=weights, k=1)[0]
            add(_proposal(parent, "".join(candidate), "position_recombination"))
    return list(proposals.values())


def allocate_budget(
    proposals: list[Proposal],
    scores: np.ndarray,
    *,
    budget: int,
    parent_floor: int,
    exploration: int,
    seed: int,
) -> list[tuple[Proposal, float, str]]:
    """Select exactly ``budget`` candidates with parent and exploration floors."""
    if len(proposals) < budget:
        raise ValueError(f"proposal pool {len(proposals)} is smaller than budget {budget}")
    indexed = sorted(
        zip(proposals, scores, strict=True),
        key=lambda item: (float(item[1]), item[0].sequence, item[0].parent.sequence),
    )
    selected: list[tuple[Proposal, float, str]] = []
    used: set[str] = set()

    by_parent: dict[str, list[tuple[Proposal, float]]] = defaultdict(list)
    for proposal, score in indexed:
        by_parent[proposal.parent.sequence].append((proposal, float(score)))
    parent_order = sorted(
        by_parent,
        key=lambda sequence: (
            by_parent[sequence][0][0].parent.final_score,
            sequence,
        ),
    )
    for parent_sequence in parent_order[:parent_floor]:
        for proposal, score in by_parent[parent_sequence]:
            if proposal.sequence not in used:
                selected.append((proposal, score, "parent_floor"))
                used.add(proposal.sequence)
                break

    exploit_limit = budget - exploration
    for proposal, score in indexed:
        if len(selected) >= exploit_limit:
            break
        if proposal.sequence not in used:
            selected.append((proposal, float(score), "model"))
            used.add(proposal.sequence)

    remaining = [item for item in indexed if item[0].sequence not in used]
    rng = random.Random(seed)
    # Stable exploration across edit depths and proposal sources.
    remaining.sort(key=lambda item: (
        hashlib.sha256(
            f"{seed}:{item[0].source}:{item[0].edit_distance}:{item[0].sequence}".encode()
        ).hexdigest(),
        item[0].sequence,
    ))
    for proposal, score in remaining:
        if len(selected) >= budget:
            break
        selected.append((proposal, float(score), "exploration"))
        used.add(proposal.sequence)
    if len(selected) != budget:
        raise AssertionError(f"allocated {len(selected)} of {budget}")
    return selected


def _proposal_support(
    winner: str, parents: list[Parent], donors: list[str], maximum_edits: int = 6
) -> tuple[bool, int | None]:
    compatible = [item for item in parents if len(item.sequence) == len(winner)]
    donor_sequences = [sequence for sequence in donors if len(sequence) == len(winner)]
    if not compatible or not donor_sequences:
        return False, None
    allowed = [set(sequence[pos] for sequence in donor_sequences) for pos in range(len(winner))]
    distances = []
    for parent in compatible:
        changes = [pos for pos, (left, right) in enumerate(zip(parent.sequence, winner)) if left != right]
        if all(winner[pos] in allowed[pos] for pos in changes):
            distances.append(len(changes))
    minimum = min(distances) if distances else None
    return minimum is not None and minimum <= maximum_edits, minimum


def proposal_backtest(
    model: Any,
    epoch_fields: list[tuple[int, pd.DataFrame]],
    *,
    start_index: int,
    count: int,
    length: int,
    random_candidates_per_parent: int,
    budget: int,
    parent_floor: int,
    exploration: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    placements = []
    selected_rows = []
    for index in range(start_index, start_index + count):
        epoch, field = epoch_fields[index]
        preceding = epoch_fields[:index]
        parents = select_parents(preceding)
        donor_fields = preceding[-16:]
        donors = list(dict.fromkeys(
            str(sequence)
            for _, frame in donor_fields
            for sequence in frame.sequence
            if len(str(sequence)) == length
        ))
        historical = {
            str(sequence)
            for _, frame in preceding
            for sequence in frame.sequence
        }
        organic = generate_proposals(
            parents,
            donors,
            historical=historical,
            length=length,
            random_candidates_per_parent=random_candidates_per_parent,
            seed=seed + epoch,
        )
        organic_scores = prediction_scores(
            model,
            feature_matrix(
                [item.sequence for item in organic],
                [item.parent for item in organic],
                length=length,
            ),
        )
        selected = allocate_budget(
            organic,
            organic_scores,
            budget=budget,
            parent_floor=parent_floor,
            exploration=exploration,
            seed=seed + epoch,
        )
        selected_sequences = {item.sequence for item, _, _ in selected}

        actual = rank_field(field[field.sequence.str.len() == length].reset_index(drop=True))
        winner_row = actual.iloc[0]
        winner = str(winner_row.sequence)
        organic_sequences = {item.sequence for item in organic}
        support, minimum_edits = _proposal_support(winner, parents, donors)

        # Oracle retrieval pool: heldout molecules are appended only after the
        # organic proposal set is frozen.  Their placement tests the ranker; it
        # is intentionally not counted as organic discovery.
        augmented = list(organic)
        organic_pairs = {(item.parent.sequence, item.sequence) for item in organic}
        for sequence in actual.sequence.astype(str):
            parent = nearest_parent(sequence, parents)
            pair = (parent.sequence, sequence)
            if pair not in organic_pairs:
                edit_distance, mutations = mutation_description(parent.sequence, sequence)
                augmented.append(Proposal(
                    sequence, parent, "heldout_oracle_only", edit_distance, mutations
                ))
                organic_pairs.add(pair)
        augmented_scores = prediction_scores(
            model,
            feature_matrix(
                [item.sequence for item in augmented],
                [item.parent for item in augmented],
                length=length,
            ),
        )
        augmented_selected = allocate_budget(
            augmented,
            augmented_scores,
            budget=budget,
            parent_floor=parent_floor,
            exploration=exploration,
            seed=seed + epoch,
        )
        augmented_selected_sequences = {
            item.sequence for item, _, _ in augmented_selected
        }
        winner_scores = [
            float(score)
            for item, score in zip(augmented, augmented_scores, strict=True)
            if item.sequence == winner
        ]
        winner_score = min(winner_scores)
        unique_best_score: dict[str, float] = {}
        for item, score in zip(augmented, augmented_scores, strict=True):
            unique_best_score[item.sequence] = min(
                float(score), unique_best_score.get(item.sequence, math.inf)
            )
        oracle_rank = 1 + sum(score < winner_score for score in unique_best_score.values())

        placements.append({
            "epoch": epoch,
            "winner_sequence": winner,
            "winner_actual_score": float(winner_row.final_score),
            "parent_count": len(parents),
            "donor_count": len(donors),
            "organic_pool_size": len({item.sequence for item in organic}),
            "winner_supported_within_six": support,
            "winner_minimum_supported_edits": minimum_edits,
            "winner_organically_generated": winner in organic_sequences,
            "winner_in_organic_top160": winner in selected_sequences,
            "winner_oracle_model_rank": oracle_rank,
            "winner_in_oracle_top160": winner in augmented_selected_sequences,
            "maximum_parent_epoch": max(parent.epoch for parent in parents),
            "leakage_check": max(parent.epoch for parent in parents) < epoch,
        })
        for selection_rank, (item, score, lane) in enumerate(selected, start=1):
            selected_rows.append({
                "epoch": epoch,
                "selection_rank": selection_rank,
                "selection_lane": lane,
                "predicted_score": score,
                "sequence": item.sequence,
                "parent_sequence": item.parent.sequence,
                "parent_epoch": item.parent.epoch,
                "parent_score": item.parent.final_score,
                "proposal_source": item.source,
                "edit_distance": item.edit_distance,
                "mutations": item.mutations,
            })
    return pd.DataFrame(placements), pd.DataFrame(selected_rows)


def cdr1_evidence(
    epoch_fields: list[tuple[int, pd.DataFrame]], *, start_index: int, count: int
) -> pd.DataFrame:
    rows = []
    for index in range(start_index, start_index + count):
        epoch, field = epoch_fields[index]
        parents = select_parents(epoch_fields[:index])
        ranked = rank_field(field)
        for _, row in ranked.iterrows():
            sequence = str(row.sequence)
            matches = []
            for parent in parents:
                if len(sequence) != len(parent.sequence):
                    continue
                try:
                    cdr1 = set(range(*find_regions(parent.sequence).cdr1))
                except ValueError:
                    continue
                changed = [
                    pos for pos, pair in enumerate(zip(parent.sequence, sequence))
                    if pair[0] != pair[1]
                ]
                if len(changed) == 1 and changed[0] in cdr1:
                    matches.append(parent)
            if matches:
                parent = min(matches, key=lambda item: (item.final_score, item.sequence))
                _, mutation = mutation_description(parent.sequence, sequence)
                rows.append({
                    "epoch": epoch,
                    "sequence": sequence,
                    "mutation": mutation,
                    "validator_rank": int(row.actual_field_rank),
                    "validator_final_score": float(row.final_score),
                    "parent_sequence": parent.sequence,
                    "parent_epoch": parent.epoch,
                    "parent_score": parent.final_score,
                })
    return pd.DataFrame(rows)


def _summarize_fields(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "epochs": int(len(frame)),
        "winner_top1_rate": float((frame.winner_predicted_rank <= 1).mean()),
        "winner_top5_rate": float((frame.winner_predicted_rank <= 5).mean()),
        "winner_top10_rate": float((frame.winner_predicted_rank <= 10).mean()),
        "median_winner_predicted_rank": float(frame.winner_predicted_rank.median()),
        "median_winner_predicted_percentile": float(
            frame.winner_predicted_percentile.median()
        ),
        "median_field_spearman": float(frame.field_spearman.median()),
    }


def mutation_context_diagnostics(
    evidence: pd.DataFrame, field_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach unseen model predictions to exact CDR1 substitution outcomes."""
    if evidence.empty:
        return evidence.copy(), {"examples": 0, "recurring_mutations": {}}
    prediction_columns = field_predictions[[
        "epoch",
        "sequence",
        "predicted_score",
        "predicted_field_rank",
        "actual_field_rank",
    ]].copy()
    merged = evidence.merge(
        prediction_columns,
        on=["epoch", "sequence"],
        how="left",
        validate="one_to_one",
        suffixes=("_api", "_model_field"),
    )
    merged["actual_rank_percentile"] = (
        merged["actual_field_rank"] - 1
    ) / merged.groupby("epoch")["actual_field_rank"].transform("max").clip(lower=1)
    recurring = {}
    for mutation, group in merged.groupby("mutation"):
        if len(group) < 2:
            continue
        actual = group.validator_final_score.rank(method="average")
        predicted = group.predicted_score.rank(method="average")
        correlation = actual.corr(predicted, method="pearson")
        recurring[str(mutation)] = {
            "examples": len(group),
            "epochs": sorted(int(value) for value in group.epoch),
            "predicted_score_range": [
                float(group.predicted_score.min()),
                float(group.predicted_score.max()),
            ],
            "actual_score_range": [
                float(group.validator_final_score.min()),
                float(group.validator_final_score.max()),
            ],
            "ordering_spearman": (
                None if pd.isna(correlation) else float(correlation)
            ),
        }
    return merged, {
        "examples": len(merged),
        "recurring_mutations": recurring,
        "interpretation": (
            "positive ordering_spearman means the frozen model ordered lower "
            "validator scores as better within repeated mutation labels"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--filename", default="P20809_nanobodies.csv")
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--holdout-epochs", type=int, default=20)
    parser.add_argument("--internal-validation-epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--random-candidates-per-parent", type=int, default=500)
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--parent-floor", type=int, default=32)
    parser.add_argument("--exploration", type=int, default=8)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--model-workers",
        type=int,
        default=24,
        help="bounded sklearn parallelism so a live miner retains CPU capacity",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--commit-api", default=COMMIT_API)
    parser.add_argument("--leaderboard-api-root", default=LEADERBOARD_API_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    total = args.warmup_epochs + args.train_epochs + args.holdout_epochs
    if args.internal_validation_epochs >= args.train_epochs:
        parser.error("internal validation must be smaller than training period")
    if args.parent_floor + args.exploration > args.budget:
        parser.error("parent floor plus exploration exceeds budget")
    if args.output_dir.exists():
        if not args.overwrite:
            parser.error(f"output directory exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config = {
        **vars(args),
        "archive": str(args.archive.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "metric_directions": METRICS,
        "model_specs": MODEL_SPECS,
        "live_acceptance": LIVE_ACCEPTANCE,
        "protocol": "warmup -> train/internal-selection -> sealed holdout",
        "oracle_pool_warning": (
            "heldout field sequences are appended only after organic generation; "
            "oracle placement measures ranking, never discovery"
        ),
    }
    _atomic_json(args.output_dir / "config.json", config)

    batches = fetch_epoch_commits(args.commit_api, args.filename, total)
    epoch_order = sorted(int(item["epoch"]) for item in batches)
    fields = fetch_api_fields(
        epoch_order,
        target=args.target,
        api_root=args.leaderboard_api_root,
        workers=args.workers,
    )
    missing = [epoch for epoch in epoch_order if epoch not in fields]
    if missing:
        raise SystemExit(f"missing exact API fields for epochs: {missing}")
    epoch_fields = [(epoch, fields[epoch]) for epoch in epoch_order]
    dataset = pd.concat([frame for _, frame in epoch_fields], ignore_index=True)
    dataset.to_csv(args.output_dir / "api_field_snapshot.csv.gz", index=False)
    split = {
        "warmup_epochs": epoch_order[: args.warmup_epochs],
        "training_epochs": epoch_order[
            args.warmup_epochs : args.warmup_epochs + args.train_epochs
        ],
        "internal_validation_epochs": epoch_order[
            args.warmup_epochs + args.train_epochs - args.internal_validation_epochs
            : args.warmup_epochs + args.train_epochs
        ],
        "holdout_epochs": epoch_order[-args.holdout_epochs :],
        "maximum_training_epoch": epoch_order[args.warmup_epochs + args.train_epochs - 1],
        "minimum_holdout_epoch": epoch_order[-args.holdout_epochs],
        "holdout_sealed": True,
        "commit_batches": batches,
    }
    if split["maximum_training_epoch"] >= split["minimum_holdout_epoch"]:
        raise AssertionError("chronological split overlaps")
    _atomic_json(args.output_dir / "split.json", split)

    sequences, parents, y, weights, example_epochs = training_examples(
        epoch_fields,
        context_epochs=args.warmup_epochs,
        train_epochs=args.train_epochs,
        length=args.sequence_length,
    )
    x = feature_matrix(sequences, parents, length=args.sequence_length)
    validation_epoch_set = set(split["internal_validation_epochs"])
    development_mask = np.asarray([
        epoch not in validation_epoch_set for epoch in example_epochs
    ])
    selection_summaries: dict[str, pd.DataFrame] = {}
    selection_payload: dict[str, Any] = {}
    internal_start = args.warmup_epochs + args.train_epochs - args.internal_validation_epochs
    for name in MODEL_SPECS:
        model = fit_model(
            name,
            x[development_mask],
            y[development_mask],
            weights[development_mask],
            args.random_seed,
            args.model_workers,
        )
        summary, _ = evaluate_fields(
            model,
            epoch_fields,
            start_index=internal_start,
            count=args.internal_validation_epochs,
            length=args.sequence_length,
            phase="internal_validation",
        )
        selection_summaries[name] = summary
        selection_payload[name] = _summarize_fields(summary)
    chosen_name = choose_model(selection_summaries)
    accepted_for_live, acceptance_checks = live_acceptance(
        selection_summaries[chosen_name]
    )
    selection_payload["chosen_model"] = chosen_name
    selection_payload["accepted_for_live"] = accepted_for_live
    selection_payload["acceptance_thresholds"] = LIVE_ACCEPTANCE
    selection_payload["acceptance_checks"] = acceptance_checks
    selection_payload["selection_rule"] = (
        "lowest median winner percentile, then highest top-5 rate, then "
        "highest median field Spearman; internal validation only"
    )
    _atomic_json(args.output_dir / "model_selection.json", selection_payload)

    model = fit_model(
        chosen_name, x, y, weights, args.random_seed, args.model_workers
    )
    from joblib import dump
    dump(model, args.output_dir / "model.joblib", compress=3)

    holdout_start = args.warmup_epochs + args.train_epochs
    field_summary, field_predictions = evaluate_fields(
        model,
        epoch_fields,
        start_index=holdout_start,
        count=args.holdout_epochs,
        length=args.sequence_length,
        phase="sealed_holdout",
    )
    field_summary.to_csv(args.output_dir / "holdout_field_summary.csv", index=False)
    field_predictions.to_csv(
        args.output_dir / "holdout_field_predictions.csv.gz", index=False
    )

    placements, selected = proposal_backtest(
        model,
        epoch_fields,
        start_index=holdout_start,
        count=args.holdout_epochs,
        length=args.sequence_length,
        random_candidates_per_parent=args.random_candidates_per_parent,
        budget=args.budget,
        parent_floor=args.parent_floor,
        exploration=args.exploration,
        seed=args.random_seed,
    )
    placements.to_csv(args.output_dir / "holdout_proposal_placements.csv", index=False)
    selected.to_csv(args.output_dir / "holdout_top160.csv.gz", index=False)
    evidence = cdr1_evidence(
        epoch_fields,
        start_index=holdout_start,
        count=args.holdout_epochs,
    )
    evidence.to_csv(args.output_dir / "cdr1_substitution_evidence.csv", index=False)
    context_predictions, context_summary = mutation_context_diagnostics(
        evidence, field_predictions
    )
    context_predictions.to_csv(
        args.output_dir / "cdr1_context_predictions.csv", index=False
    )

    summary = {
        "chosen_model": chosen_name,
        "accepted_for_live": accepted_for_live,
        "training_examples": len(sequences),
        "training_epoch_count": args.train_epochs,
        "holdout_epoch_count": args.holdout_epochs,
        "holdout": _summarize_fields(field_summary),
        "proposal": {
            "budget": args.budget,
            "mean_organic_pool_size": float(placements.organic_pool_size.mean()),
            "winner_supported_within_six_rate": float(
                placements.winner_supported_within_six.mean()
            ),
            "winner_organic_discovery_rate": float(
                placements.winner_organically_generated.mean()
            ),
            "winner_organic_top160_rate": float(
                placements.winner_in_organic_top160.mean()
            ),
            "winner_oracle_top160_rate": float(
                placements.winner_in_oracle_top160.mean()
            ),
            "median_winner_oracle_model_rank": float(
                placements.winner_oracle_model_rank.median()
            ),
            "all_parent_epochs_precede_target": bool(placements.leakage_check.all()),
        },
        "cdr1": {
            "matching_entries": len(evidence),
            "matching_epochs": int(evidence.epoch.nunique()) if len(evidence) else 0,
            "wins": int((evidence.validator_rank == 1).sum()) if len(evidence) else 0,
            "model_context_diagnostic": context_summary,
        },
    }
    _atomic_json(args.output_dir / "summary.json", summary)

    manifest = {
        "archive_sha256": _sha256(args.archive),
        "source_archive": str(args.archive.resolve()),
        "files": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "manifest.json" and path.is_file():
            manifest["files"][path.name] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    print(f"MUTATION_RANKER_BACKTEST_READY output={args.output_dir}")


if __name__ == "__main__":
    main()
