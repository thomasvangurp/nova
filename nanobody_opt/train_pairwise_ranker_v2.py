"""Train a 200-epoch pairwise VHH acquisition model with sealed tests.

The failed v1 model predicted absolute scores.  V2 instead learns whether the
left sequence beats the right sequence inside the same validator epoch, which
cancels epoch difficulty and makes the inference question match acquisition:
does this child improve on its declared elite parent?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nanobody_opt.backtest_mutation_ranker import AMINO_ACIDS, AA_INDEX
from nanobody_opt.train_parent_relative_ranker import (
    AA_PROPERTIES,
    _atomic_json,
    _region_summary,
    _sha256,
    build_dataset,
    current_proposals,
)


@dataclass(frozen=True)
class PairModelSpec:
    name: str
    family: str
    min_samples_leaf: int


MODEL_SPECS = (
    PairModelSpec("extra_trees_leaf2", "extra_trees", 2),
    PairModelSpec("extra_trees_leaf5", "extra_trees", 5),
    PairModelSpec("random_forest_leaf2", "random_forest", 2),
    PairModelSpec("random_forest_leaf5", "random_forest", 5),
)


def sampled_epoch_pairs(
    frame: pd.DataFrame, *, maximum: int, seed: int
) -> list[tuple[int, int]]:
    """Stratify pairs across winners, tails, neighbours, and random field rows."""
    ordered = frame.sort_values(["final_score", "sequence"]).index.tolist()
    pairs: set[tuple[int, int]] = set()

    def add(left: int, right: int) -> None:
        if left != right:
            pairs.add(tuple(sorted((left, right))))

    for other in ordered[1:]:
        add(ordered[0], other)
    for left in ordered[: min(5, len(ordered))]:
        for right in ordered[max(0, len(ordered) - 10) :]:
            add(left, right)
    for left, right in zip(ordered, ordered[1:]):
        add(left, right)
    all_pairs = [
        (left, right)
        for offset, left in enumerate(ordered)
        for right in ordered[offset + 1 :]
    ]
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    for pair in all_pairs:
        if len(pairs) >= maximum:
            break
        pairs.add(tuple(sorted(pair)))
    selected = sorted(pairs)
    if len(selected) > maximum:
        required = set()
        winner = ordered[0]
        for pair in selected:
            if winner in pair:
                required.add(pair)
        remainder = [pair for pair in selected if pair not in required]
        rng.shuffle(remainder)
        selected = sorted([*required, *remainder[: max(0, maximum - len(required))]])
    return selected


def build_pair_rows(
    snapshot: pd.DataFrame,
    epochs: Iterable[int],
    *,
    maximum_per_epoch: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        field = snapshot[snapshot.epoch == epoch].reset_index(drop=True)
        field = field[field.sequence.str.len() == field.sequence.str.len().mode().iloc[0]]
        field = field.reset_index(drop=True)
        for pair_index, (first, second) in enumerate(sampled_epoch_pairs(
            field, maximum=maximum_per_epoch, seed=seed + int(epoch)
        )):
            left, right = first, second
            if (seed + int(epoch) + pair_index) % 2:
                left, right = right, left
            left_row = field.iloc[left]
            right_row = field.iloc[right]
            left_score = float(left_row.final_score)
            right_score = float(right_row.final_score)
            if left_score == right_score:
                continue
            rows.append({
                "epoch": int(epoch),
                "left_sequence": str(left_row.sequence),
                "right_sequence": str(right_row.sequence),
                "left_score": left_score,
                "right_score": right_score,
                "left_wins": int(left_score < right_score),
                "score_gap": abs(left_score - right_score),
                "sample_weight": 1.0 + min(abs(left_score - right_score) / 50.0, 3.0),
            })
    return pd.DataFrame(rows)


def pair_feature_matrix(
    left_sequences: Iterable[str], right_sequences: Iterable[str], *, length: int
) -> csr_matrix:
    """Sparse position-aware sequence-pair representation with VHH context."""
    left_items = list(left_sequences)
    right_items = list(right_sequences)
    if len(left_items) != len(right_items):
        raise ValueError("left and right sequences must be parallel")
    property_count = len(next(iter(AA_PROPERTIES.values())))
    left_one_hot = length * len(AMINO_ACIDS)
    right_one_hot = left_one_hot
    mutation_mask = length
    property_delta = length * property_count
    composition = 2 * len(AMINO_ACIDS)
    summary = 6
    width = (
        left_one_hot + right_one_hot + mutation_mask + property_delta
        + composition + summary
    )
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    def add(row: int, column: int, value: float) -> None:
        if value:
            row_indices.append(row)
            column_indices.append(column)
            values.append(float(value))

    for row, (left, right) in enumerate(zip(left_items, right_items, strict=True)):
        if len(left) != length or len(right) != length:
            raise ValueError("pairwise sequences have inconsistent lengths")
        offset = 0
        for position, residue in enumerate(left):
            add(row, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue], 1)
        offset += left_one_hot
        for position, residue in enumerate(right):
            add(row, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue], 1)
        offset += right_one_hot
        for position, (before, after) in enumerate(zip(right, left, strict=True)):
            add(row, offset + position, float(before != after))
        offset += mutation_mask
        for position, (before, after) in enumerate(zip(right, left, strict=True)):
            for property_index, (right_value, left_value) in enumerate(
                zip(AA_PROPERTIES[before], AA_PROPERTIES[after], strict=True)
            ):
                add(
                    row,
                    offset + position * property_count + property_index,
                    left_value - right_value,
                )
        offset += property_delta
        for residue, index in AA_INDEX.items():
            add(row, offset + index, left.count(residue) / length)
            add(
                row,
                offset + len(AMINO_ACIDS) + index,
                right.count(residue) / length,
            )
        offset += composition
        region = _region_summary(right, left)
        changed_fraction = sum(a != b for a, b in zip(left, right, strict=True)) / length
        for index, value in enumerate([changed_fraction, *region]):
            add(row, offset + index, value)
    return csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, column_indices)),
        shape=(len(left_items), width),
        dtype=np.float32,
    )


def fit_pair_model(
    spec: PairModelSpec,
    pairs: pd.DataFrame,
    *,
    length: int,
    workers: int,
    seed: int,
):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    features = pair_feature_matrix(
        pairs.left_sequence, pairs.right_sequence, length=length
    )
    common = dict(
        n_estimators=256,
        min_samples_leaf=spec.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=workers,
    )
    if spec.family == "extra_trees":
        model = ExtraTreesClassifier(**common)
    elif spec.family == "random_forest":
        model = RandomForestClassifier(
            bootstrap=True, max_samples=0.85, **common
        )
    else:
        raise ValueError(f"unsupported pair model {spec.family}")
    model.fit(
        features,
        pairs.left_wins.to_numpy(dtype=int),
        sample_weight=pairs.sample_weight.to_numpy(dtype=float),
    )
    return model


def symmetric_win_probability(
    model: Any,
    left_sequences: Iterable[str],
    right_sequences: Iterable[str],
    *,
    length: int,
) -> np.ndarray:
    left = list(left_sequences)
    right = list(right_sequences)
    forward = model.predict_proba(
        pair_feature_matrix(left, right, length=length)
    )[:, list(model.classes_).index(1)]
    reverse = model.predict_proba(
        pair_feature_matrix(right, left, length=length)
    )[:, list(model.classes_).index(1)]
    return 0.5 * (forward + (1.0 - reverse))


def _spearman(actual: Iterable[float], predicted: Iterable[float]) -> float:
    left = pd.Series(list(actual), dtype=float).rank(method="average")
    right = pd.Series(list(predicted), dtype=float).rank(method="average")
    result = left.corr(right, method="pearson")
    return float(result) if pd.notna(result) else 0.0


def evaluate_public_fields(
    model: Any,
    snapshot: pd.DataFrame,
    epochs: Iterable[int],
    *,
    length: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for epoch in epochs:
        field = snapshot[
            (snapshot.epoch == epoch) & (snapshot.sequence.str.len() == length)
        ].sort_values(["final_score", "sequence"]).reset_index(drop=True)
        sequences = field.sequence.astype(str).tolist()
        left: list[str] = []
        right: list[str] = []
        owner: list[int] = []
        for index, sequence in enumerate(sequences):
            for other_index, other in enumerate(sequences):
                if index == other_index:
                    continue
                left.append(sequence)
                right.append(other)
                owner.append(index)
        probabilities = symmetric_win_probability(
            model, left, right, length=length
        )
        totals = np.zeros(len(sequences), dtype=float)
        counts = np.zeros(len(sequences), dtype=int)
        for index, probability in zip(owner, probabilities, strict=True):
            totals[index] += probability
            counts[index] += 1
        tournament = totals / np.maximum(counts, 1)
        predicted_rank = pd.Series(-tournament).rank(method="min").astype(int)
        winner_index = int(np.argmin(field.final_score.to_numpy(dtype=float)))
        actual_rank = field.final_score.rank(method="min").astype(int)
        summaries.append({
            "phase": phase,
            "epoch": int(epoch),
            "field_size": len(field),
            "winner_score": float(field.iloc[winner_index].final_score),
            "winner_predicted_rank": int(predicted_rank.iloc[winner_index]),
            "winner_predicted_percentile": (
                (int(predicted_rank.iloc[winner_index]) - 1) / max(len(field) - 1, 1)
            ),
            "field_spearman": _spearman(-field.final_score, tournament),
        })
        for index, sequence in enumerate(sequences):
            predictions.append({
                "phase": phase,
                "epoch": int(epoch),
                "sequence": sequence,
                "validator_score": float(field.iloc[index].final_score),
                "actual_rank": int(actual_rank.iloc[index]),
                "predicted_win_probability": float(tournament[index]),
                "predicted_rank": int(predicted_rank.iloc[index]),
            })
    return pd.DataFrame(summaries), pd.DataFrame(predictions)


def summarize_public(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "epochs": int(len(frame)),
        "median_field_spearman": float(frame.field_spearman.median()),
        "median_winner_rank": float(frame.winner_predicted_rank.median()),
        "median_winner_percentile": float(frame.winner_predicted_percentile.median()),
        "winner_top1_rate": float((frame.winner_predicted_rank <= 1).mean()),
        "winner_top5_rate": float((frame.winner_predicted_rank <= 5).mean()),
        "winner_top10_rate": float((frame.winner_predicted_rank <= 10).mean()),
    }


def evaluate_local_forward(
    model: Any, frame: pd.DataFrame, *, length: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.metrics import roc_auc_score

    output = frame.copy().reset_index(drop=True)
    output["predicted_improvement_probability"] = symmetric_win_probability(
        model, output.sequence, output.parent_sequence, length=length
    )
    output["actually_improves_parent"] = (
        output.target_improvement > 0
    ).astype(int)
    auc = (
        float(roc_auc_score(
            output.actually_improves_parent,
            output.predicted_improvement_probability,
        ))
        if output.actually_improves_parent.nunique() == 2
        else 0.5
    )
    epoch_metrics = []
    for epoch, group in output.groupby("epoch"):
        group = group.reset_index(drop=True)
        actual_top10 = set(group.nsmallest(min(10, len(group)), "child_rank_sum").index)
        model_order = group.sort_values(
            ["predicted_improvement_probability", "proposal_prior_score"],
            ascending=[False, False],
        ).index.tolist()
        heuristic_order = group.sort_values("screen_order").index.tolist()
        row: dict[str, Any] = {"epoch": int(epoch), "examples": len(group)}
        for requested in (20, 40, 80, 160):
            budget = min(requested, len(group))
            model_selected = model_order[:budget]
            heuristic_selected = heuristic_order[:budget]
            optimum = float(group.child_rank_sum.min())
            row[f"model_regret_at_{requested}"] = float(
                group.loc[model_selected, "child_rank_sum"].min() - optimum
            )
            row[f"heuristic_regret_at_{requested}"] = float(
                group.loc[heuristic_selected, "child_rank_sum"].min() - optimum
            )
            row[f"model_top10_recall_at_{requested}"] = float(
                len(set(model_selected) & actual_top10) / max(len(actual_top10), 1)
            )
            row[f"heuristic_top10_recall_at_{requested}"] = float(
                len(set(heuristic_selected) & actual_top10) / max(len(actual_top10), 1)
            )
        epoch_metrics.append(row)
    metrics = pd.DataFrame(epoch_metrics)
    summary = {
        "examples": len(output),
        "epochs": int(output.epoch.nunique()),
        "parent_improvement_auc": auc,
        "improvement_spearman": _spearman(
            output.target_improvement,
            output.predicted_improvement_probability,
        ),
        "median_by_epoch": {
            column: float(metrics[column].median())
            for column in metrics.columns
            if column != "epoch"
        },
    }
    return output, summary


def score_current_proposals(
    model: Any, frame: pd.DataFrame, *, length: int
) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output["predicted_improvement_probability"] = symmetric_win_probability(
        model, output.sequence, output.parent_sequence, length=length
    )
    output = output.sort_values(
        ["predicted_improvement_probability", "proposal_prior_score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    output["model_rank"] = np.arange(1, len(output) + 1)
    return output


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--before-epoch", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--training-epochs", type=int, default=200)
    parser.add_argument("--holdout-epochs", type=int, default=20)
    parser.add_argument("--internal-validation-epochs", type=int, default=20)
    parser.add_argument("--pairs-per-epoch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--random-seed", type=int, default=20260910)
    args = parser.parse_args()

    snapshot = pd.read_csv(args.api_snapshot)
    snapshot.sequence = snapshot.sequence.astype(str).str.strip().str.upper()
    epochs = sorted(int(value) for value in snapshot.epoch.unique())
    required = args.training_epochs + args.holdout_epochs
    if len(epochs) < required:
        raise SystemExit(f"require {required} complete API epochs; found {len(epochs)}")
    selected_epochs = epochs[-required:]
    training_epochs = selected_epochs[: args.training_epochs]
    holdout_epochs = selected_epochs[args.training_epochs :]
    internal_epochs = training_epochs[-args.internal_validation_epochs :]
    internal_train_epochs = training_epochs[: -args.internal_validation_epochs]
    length = int(snapshot.sequence.str.len().mode().iloc[0])
    snapshot = snapshot[snapshot.sequence.str.len() == length].reset_index(drop=True)

    internal_pairs = build_pair_rows(
        snapshot,
        internal_train_epochs,
        maximum_per_epoch=args.pairs_per_epoch,
        seed=args.random_seed,
    )
    model_selection: dict[str, Any] = {}
    best_spec: PairModelSpec | None = None
    best_key: tuple[float, ...] | None = None
    for spec in MODEL_SPECS:
        model = fit_pair_model(
            spec,
            internal_pairs,
            length=length,
            workers=args.workers,
            seed=args.random_seed,
        )
        field_summary, _ = evaluate_public_fields(
            model,
            snapshot,
            internal_epochs,
            length=length,
            phase="internal_validation",
        )
        summary = summarize_public(field_summary)
        model_selection[spec.name] = summary
        key = (
            summary["winner_top5_rate"],
            summary["median_field_spearman"],
            -summary["median_winner_percentile"],
        )
        if best_key is None or key > best_key:
            best_spec, best_key = spec, key
    assert best_spec is not None

    all_training_pairs = build_pair_rows(
        snapshot,
        training_epochs,
        maximum_per_epoch=args.pairs_per_epoch,
        seed=args.random_seed,
    )
    final_model = fit_pair_model(
        best_spec,
        all_training_pairs,
        length=length,
        workers=args.workers,
        seed=args.random_seed,
    )
    holdout_fields, holdout_predictions = evaluate_public_fields(
        final_model,
        snapshot,
        holdout_epochs,
        length=length,
        phase="sealed_holdout",
    )
    holdout_summary = summarize_public(holdout_fields)
    local_dataset, local_inputs = build_dataset(
        args.runtime_root, target=args.target, before_epoch=args.before_epoch
    )
    local_dataset = local_dataset[
        (local_dataset.sequence.str.len() == length)
        & (local_dataset.parent_sequence.str.len() == length)
    ].reset_index(drop=True)
    local_predictions, local_summary = evaluate_local_forward(
        final_model, local_dataset, length=length
    )

    internal_summary = model_selection[best_spec.name]
    local_medians = local_summary["median_by_epoch"]
    gates = {
        "internal_median_spearman_at_least_0_15": (
            internal_summary["median_field_spearman"] >= 0.15
        ),
        "internal_winner_top5_at_least_0_35": (
            internal_summary["winner_top5_rate"] >= 0.35
        ),
        "sealed_median_spearman_at_least_0_15": (
            holdout_summary["median_field_spearman"] >= 0.15
        ),
        "sealed_winner_top5_at_least_0_35": (
            holdout_summary["winner_top5_rate"] >= 0.35
        ),
        "forward_parent_improvement_auc_at_least_0_55": (
            local_summary["parent_improvement_auc"] >= 0.55
        ),
        "forward_recall40_beats_or_ties_live_order": (
            local_medians["model_top10_recall_at_40"]
            >= local_medians["heuristic_top10_recall_at_40"]
        ),
    }
    accepted = all(gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_training_pairs.to_csv(args.output_dir / "training_pairs.csv.gz", index=False)
    pd.DataFrame([
        {"model": name, **values} for name, values in model_selection.items()
    ]).to_csv(args.output_dir / "model_selection.csv", index=False)
    holdout_fields.to_csv(args.output_dir / "sealed_holdout_fields.csv", index=False)
    holdout_predictions.to_csv(
        args.output_dir / "sealed_holdout_predictions.csv.gz", index=False
    )
    local_dataset.to_csv(args.output_dir / "local_forward_dataset.csv.gz", index=False)
    local_predictions.to_csv(
        args.output_dir / "local_forward_predictions.csv.gz", index=False
    )
    current = current_proposals(
        args.runtime_root, epoch=args.before_epoch, target=args.target
    )
    if not current.empty:
        score_current_proposals(final_model, current, length=length).to_csv(
            args.output_dir / "current_epoch_predictions.csv", index=False
        )
    joblib.dump(final_model, args.output_dir / "model.joblib")
    config = {
        "target": args.target,
        "api_snapshot": str(args.api_snapshot),
        "api_snapshot_sha256": _sha256(args.api_snapshot),
        "available_api_epochs": len(epochs),
        "selected_api_epoch_range": [selected_epochs[0], selected_epochs[-1]],
        "training_epoch_range": [training_epochs[0], training_epochs[-1]],
        "training_epochs": len(training_epochs),
        "internal_training_epochs": len(internal_train_epochs),
        "internal_validation_epoch_range": [internal_epochs[0], internal_epochs[-1]],
        "sealed_holdout_epoch_range": [holdout_epochs[0], holdout_epochs[-1]],
        "sealed_holdout_epochs": len(holdout_epochs),
        "pairs_per_epoch": args.pairs_per_epoch,
        "training_pairs": len(all_training_pairs),
        "modal_sequence_length": length,
        "random_seed": args.random_seed,
        "model_specs": [asdict(spec) for spec in MODEL_SPECS],
        "objective": "within-validator-epoch probability that left sequence beats right",
    }
    _atomic_json(args.output_dir / "config.json", config)
    summary = {
        "accepted_for_live_allocation": accepted,
        "acceptance_gates": gates,
        "selected_model": asdict(best_spec),
        "model_selection": model_selection,
        "sealed_holdout": holdout_summary,
        "post_holdout_local_forward_test": local_summary,
        "decision": (
            "eligible for shadow acquisition"
            if accepted
            else "rejected; live heuristic remains authoritative"
        ),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    files = {}
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _atomic_json(args.output_dir / "manifest.json", {
        "api_snapshot": {
            "path": str(args.api_snapshot),
            "sha256": _sha256(args.api_snapshot),
        },
        "local_exact_inputs": local_inputs,
        "files": files,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
