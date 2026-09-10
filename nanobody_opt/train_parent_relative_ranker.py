"""Train and audit a known-ancestry parent-relative nanobody ranker.

V2 learns the change in validator-compatible dense rank sum from a declared
elite parent to its locally exact-scored child.  Model choice uses grouped
parent-family folds strictly before the final completed epoch; that final epoch
is opened once as a chronological holdout.  Current-epoch proposals are scored
only after the evaluation is frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nanobody_opt.backtest_mutation_ranker import AMINO_ACIDS, AA_INDEX
from nanobody_opt.rank_exact import _observations
from nanobody_opt.regions import find_regions
from nanobody_opt.simulate_round import METRICS, rank_sum_in_field


MODEL_SPECS = (
    ("extra_trees_leaf2", "extra_trees", 2, 0.35),
    ("extra_trees_leaf4", "extra_trees", 4, 0.50),
    ("random_forest_leaf2", "random_forest", 2, 0.35),
    ("random_forest_leaf4", "random_forest", 4, 0.50),
)
PROPOSAL_TYPES = (
    "exhaustive_single_substitution",
    "local_substitution",
    "whole_cdr_motif",
)
AA_PROPERTIES = {
    # Kyte-Doolittle hydropathy, approximate charge, volume, polarity flag.
    "A": (1.8, 0.0, 88.6, 0.0), "C": (2.5, 0.0, 108.5, 0.0),
    "D": (-3.5, -1.0, 111.1, 1.0), "E": (-3.5, -1.0, 138.4, 1.0),
    "F": (2.8, 0.0, 189.9, 0.0), "G": (-0.4, 0.0, 60.1, 0.0),
    "H": (-3.2, 0.1, 153.2, 1.0), "I": (4.5, 0.0, 166.7, 0.0),
    "K": (-3.9, 1.0, 168.6, 1.0), "L": (3.8, 0.0, 166.7, 0.0),
    "M": (1.9, 0.0, 162.9, 0.0), "N": (-3.5, 0.0, 114.1, 1.0),
    "P": (-1.6, 0.0, 112.7, 0.0), "Q": (-3.5, 0.0, 143.8, 1.0),
    "R": (-4.5, 1.0, 173.4, 1.0), "S": (-0.8, 0.0, 89.0, 1.0),
    "T": (-0.7, 0.0, 116.1, 1.0), "V": (4.2, 0.0, 140.0, 0.0),
    "W": (-0.9, 0.0, 227.8, 0.0), "Y": (-1.3, 0.0, 193.6, 1.0),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    min_samples_leaf: int
    max_features: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _epoch_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"epoch(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"cannot determine epoch from {path}")


def _median_metrics(observations: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.median([observation[name] for observation in observations]))
        for name in METRICS
    }


def _screen_order(rows: list[dict[str, str]]) -> dict[str, int]:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (
            0
            if item[1].get("proposal_type") == "exhaustive_single_substitution"
            else 1,
            item[0],
        ),
    )
    return {
        str(row["sequence"]).strip().upper(): order
        for order, (_, row) in enumerate(ordered)
    }


def build_dataset(
    runtime_root: Path, *, target: str, before_epoch: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join exact children to their declared parents and historical metrics."""
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    pattern = "live/epoch*/producer/iteration_*/candidates.csv"
    for candidates_path in sorted(runtime_root.glob(pattern)):
        epoch = _epoch_number(candidates_path)
        if epoch >= before_epoch:
            continue
        iteration_root = candidates_path.parent
        exact_paths = sorted((iteration_root / "race").glob("**/exact_all.json"))
        if not exact_paths:
            continue
        archive_path = iteration_root.parent / "archive.csv"
        if not archive_path.is_file():
            continue
        with candidates_path.open(newline="") as handle:
            proposal_rows = list(csv.DictReader(handle))
        by_sequence = {
            str(row["sequence"]).strip().upper(): row for row in proposal_rows
        }
        order = _screen_order(proposal_rows)
        observations = _observations(exact_paths, target)
        archive = pd.read_csv(archive_path)
        archive.sequence = archive.sequence.astype(str).str.strip().str.upper()
        archive = archive.drop_duplicates("sequence", keep="last")
        archive_by_sequence = archive.set_index("sequence", drop=False)
        reference = archive.tail(min(300, len(archive))).reset_index(drop=True)
        for sequence, child_observations in observations.items():
            sequence = sequence.strip().upper()
            proposal = by_sequence.get(sequence)
            if proposal is None:
                continue
            parent = str(proposal["parent_sequence"]).strip().upper()
            if parent not in archive_by_sequence.index:
                continue
            parent_row = archive_by_sequence.loc[parent]
            if isinstance(parent_row, pd.DataFrame):
                parent_row = parent_row.iloc[-1]
            try:
                parent_metrics = {
                    name: float(parent_row[name]) for name in METRICS
                }
                child_metrics = _median_metrics(child_observations)
            except (KeyError, TypeError, ValueError):
                continue
            if not all(
                math.isfinite(value)
                for value in (*parent_metrics.values(), *child_metrics.values())
            ):
                continue
            parent_rank_sum = float(rank_sum_in_field(parent_metrics, reference))
            child_rank_sum = float(rank_sum_in_field(child_metrics, reference))
            record: dict[str, Any] = {
                "epoch": epoch,
                "iteration": iteration_root.name,
                "sequence": sequence,
                "parent_sequence": parent,
                "proposal_type": str(proposal.get("proposal_type") or "unknown"),
                "edit_distance": int(float(proposal.get("edit_distance") or 0)),
                "mutations": str(proposal.get("mutations") or ""),
                "proposal_prior_score": float(
                    proposal.get("proposal_prior_score") or 0.0
                ),
                "parent_selection_score": float(
                    proposal.get("parent_selection_score") or parent_rank_sum
                ),
                "screen_order": int(order[sequence]),
                "observation_count": len(child_observations),
                "parent_rank_sum": parent_rank_sum,
                "child_rank_sum": child_rank_sum,
                "target_improvement": parent_rank_sum - child_rank_sum,
            }
            record.update({f"parent_{name}": value for name, value in parent_metrics.items()})
            record.update({f"child_{name}": value for name, value in child_metrics.items()})
            records.append(record)
        inputs.append({
            "epoch": epoch,
            "candidates": str(candidates_path),
            "candidates_sha256": _sha256(candidates_path),
            "archive": str(archive_path),
            "archive_sha256": _sha256(archive_path),
            "exact": [
                {"path": str(path), "sha256": _sha256(path)} for path in exact_paths
            ],
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("no known-ancestry exact examples were discovered")
    frame = frame.sort_values(
        ["epoch", "iteration", "screen_order", "sequence"]
    ).reset_index(drop=True)
    return frame, {"inputs": inputs}


def _region_summary(parent: str, sequence: str) -> list[float]:
    try:
        regions = find_regions(parent)
        region_sets = [
            set(range(*regions.cdr1)),
            set(range(*regions.cdr2)),
            set(range(*regions.cdr3)),
        ]
    except ValueError:
        region_sets = [set(), set(), set()]
    changed = {
        position
        for position, (left, right) in enumerate(zip(parent, sequence, strict=True))
        if left != right
    }
    terminal = set(range(min(10, len(parent)))) | set(
        range(max(len(parent) - 17, 0), len(parent))
    )
    cdr_counts = [len(changed & positions) for positions in region_sets]
    framework = len(changed - set().union(*region_sets))
    return [*map(float, cdr_counts), float(framework), float(len(changed & terminal))]


def feature_matrix(frame: pd.DataFrame, *, length: int) -> np.ndarray:
    """Encode parent context, exact mutation identity, route, and parent metrics."""
    property_count = len(next(iter(AA_PROPERTIES.values())))
    width = (
        2 * length * len(AMINO_ACIDS)
        + length
        + length * property_count
        + len(METRICS)
        + len(PROPOSAL_TYPES)
        + 9
    )
    matrix = np.zeros((len(frame), width), dtype=np.float32)
    for row_index, row in enumerate(frame.itertuples(index=False)):
        sequence = str(row.sequence)
        parent = str(row.parent_sequence)
        if len(sequence) != length or len(parent) != length:
            raise ValueError("all v2 sequences must have the modal length")
        offset = 0
        for position, residue in enumerate(sequence):
            matrix[row_index, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue]] = 1
        offset += length * len(AMINO_ACIDS)
        for position, residue in enumerate(parent):
            matrix[row_index, offset + position * len(AMINO_ACIDS) + AA_INDEX[residue]] = 1
        offset += length * len(AMINO_ACIDS)
        for position, (before, after) in enumerate(zip(parent, sequence, strict=True)):
            if before != after:
                matrix[row_index, offset + position] = 1
        offset += length
        for position, (before, after) in enumerate(zip(parent, sequence, strict=True)):
            before_properties = AA_PROPERTIES[before]
            after_properties = AA_PROPERTIES[after]
            for property_index, (left, right) in enumerate(
                zip(before_properties, after_properties, strict=True)
            ):
                matrix[
                    row_index, offset + position * property_count + property_index
                ] = right - left
        offset += length * property_count
        for metric_index, name in enumerate(METRICS):
            matrix[row_index, offset + metric_index] = float(
                getattr(row, f"parent_{name}")
            )
        offset += len(METRICS)
        proposal_type = str(row.proposal_type)
        if proposal_type in PROPOSAL_TYPES:
            matrix[row_index, offset + PROPOSAL_TYPES.index(proposal_type)] = 1
        offset += len(PROPOSAL_TYPES)
        summary = [
            float(row.edit_distance),
            float(row.proposal_prior_score),
            float(row.parent_selection_score),
            float(row.observation_count),
            *_region_summary(parent, sequence),
        ]
        matrix[row_index, offset : offset + len(summary)] = summary
    return matrix


def _fit(spec: ModelSpec, x: np.ndarray, y: np.ndarray, workers: int, seed: int):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    common = dict(
        n_estimators=384,
        min_samples_leaf=spec.min_samples_leaf,
        max_features=spec.max_features,
        random_state=seed,
        n_jobs=workers,
    )
    if spec.family == "extra_trees":
        model = ExtraTreesRegressor(**common)
    elif spec.family == "random_forest":
        model = RandomForestRegressor(bootstrap=True, max_samples=0.85, **common)
    else:
        raise ValueError(f"unknown model family {spec.family}")
    model.fit(x, y)
    return model


def prediction_distribution(model: Any, x: np.ndarray) -> tuple[np.ndarray, ...]:
    tree_predictions = np.vstack([tree.predict(x) for tree in model.estimators_])
    mean = tree_predictions.mean(axis=0)
    standard_deviation = tree_predictions.std(axis=0)
    probability_improves = (tree_predictions > 0).mean(axis=0)
    return mean, standard_deviation, probability_improves


def _spearman(actual: Iterable[float], predicted: Iterable[float]) -> float:
    left = pd.Series(list(actual), dtype=float).rank(method="average")
    right = pd.Series(list(predicted), dtype=float).rank(method="average")
    value = left.corr(right, method="pearson")
    return float(value) if pd.notna(value) else 0.0


def allocation_metrics(
    frame: pd.DataFrame,
    predicted_mean: np.ndarray,
    probability_improves: np.ndarray,
    *,
    budgets: tuple[int, ...] = (20, 40, 80, 160),
) -> dict[str, float]:
    """Compare learned acquisition with the exact live ordering on labelled rows."""
    work = frame.copy().reset_index(drop=True)
    work["predicted_improvement"] = predicted_mean
    work["probability_improves"] = probability_improves
    model_order = work.sort_values(
        ["probability_improves", "predicted_improvement", "proposal_prior_score"],
        ascending=[False, False, False],
    ).index.tolist()
    heuristic_order = work.sort_values(
        ["screen_order", "proposal_prior_score"], ascending=[True, False]
    ).index.tolist()
    actual_order = work.sort_values(
        ["child_rank_sum", "target_improvement"], ascending=[True, False]
    ).index.tolist()
    best = float(work.child_rank_sum.min())
    actual_top10 = set(actual_order[: min(10, len(work))])
    metrics: dict[str, float] = {
        "examples": float(len(work)),
        "spearman_improvement": _spearman(
            work.target_improvement, work.predicted_improvement
        ),
    }
    for requested in budgets:
        budget = min(requested, len(work))
        model_selected = model_order[:budget]
        heuristic_selected = heuristic_order[:budget]
        metrics[f"model_best_rank_sum_at_{requested}"] = float(
            work.loc[model_selected, "child_rank_sum"].min()
        )
        metrics[f"heuristic_best_rank_sum_at_{requested}"] = float(
            work.loc[heuristic_selected, "child_rank_sum"].min()
        )
        metrics[f"model_regret_at_{requested}"] = (
            metrics[f"model_best_rank_sum_at_{requested}"] - best
        )
        metrics[f"heuristic_regret_at_{requested}"] = (
            metrics[f"heuristic_best_rank_sum_at_{requested}"] - best
        )
        metrics[f"model_top10_recall_at_{requested}"] = float(
            len(set(model_selected) & actual_top10) / max(len(actual_top10), 1)
        )
        metrics[f"heuristic_top10_recall_at_{requested}"] = float(
            len(set(heuristic_selected) & actual_top10) / max(len(actual_top10), 1)
        )
    return metrics


def grouped_model_selection(
    frame: pd.DataFrame,
    *,
    length: int,
    workers: int,
    seed: int,
) -> tuple[ModelSpec, pd.DataFrame, dict[str, dict[str, float]]]:
    from sklearn.model_selection import GroupKFold

    groups = frame.parent_sequence.astype(str).to_numpy()
    unique_groups = len(set(groups))
    folds = min(4, unique_groups)
    if folds < 2:
        raise ValueError("v2 requires at least two independent parent families")
    x = feature_matrix(frame, length=length)
    y = frame.target_improvement.to_numpy(dtype=float)
    results: list[dict[str, Any]] = []
    specs = [ModelSpec(*values) for values in MODEL_SPECS]
    for spec in specs:
        splitter = GroupKFold(n_splits=folds)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(x, y, groups), start=1
        ):
            model = _fit(spec, x[train_index], y[train_index], workers, seed + fold)
            mean, deviation, probability = prediction_distribution(
                model, x[test_index]
            )
            metrics = allocation_metrics(frame.iloc[test_index], mean, probability)
            results.append({
                "model": spec.name,
                "fold": fold,
                "train_examples": len(train_index),
                "validation_examples": len(test_index),
                "validation_parents": len(set(groups[test_index])),
                "mean_predicted_standard_deviation": float(np.mean(deviation)),
                **metrics,
            })
    result_frame = pd.DataFrame(results)
    summary: dict[str, dict[str, float]] = {}
    for spec in specs:
        subset = result_frame[result_frame.model == spec.name]
        summary[spec.name] = {
            column: float(subset[column].median())
            for column in subset.columns
            if column not in {"model", "fold"}
        }
    selected = max(
        specs,
        key=lambda spec: (
            summary[spec.name]["model_top10_recall_at_40"],
            summary[spec.name]["spearman_improvement"],
            -summary[spec.name]["model_regret_at_20"],
            spec.name,
        ),
    )
    return selected, result_frame, summary


def score_frame(model: Any, frame: pd.DataFrame, *, length: int) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    mean, deviation, probability = prediction_distribution(
        model, feature_matrix(output, length=length)
    )
    output["predicted_improvement"] = mean
    output["predicted_standard_deviation"] = deviation
    output["probability_improves"] = probability
    output["model_rank"] = output.sort_values(
        ["probability_improves", "predicted_improvement", "proposal_prior_score"],
        ascending=[False, False, False],
    ).reset_index().reset_index().set_index("index")["level_0"].add(1)
    return output


def current_proposals(
    runtime_root: Path, *, epoch: int, target: str
) -> pd.DataFrame:
    candidates_paths = sorted(
        runtime_root.glob(f"live/epoch{epoch}/producer/iteration_*/candidates.csv")
    )
    rows: list[dict[str, Any]] = []
    for candidates_path in candidates_paths:
        iteration_root = candidates_path.parent
        archive_path = iteration_root.parent / "archive.csv"
        if not archive_path.is_file():
            continue
        archive = pd.read_csv(archive_path)
        archive.sequence = archive.sequence.astype(str).str.strip().str.upper()
        archive = archive.drop_duplicates("sequence", keep="last").set_index("sequence")
        with candidates_path.open(newline="") as handle:
            proposals = list(csv.DictReader(handle))
        order = _screen_order(proposals)
        for proposal in proposals:
            parent = str(proposal["parent_sequence"]).strip().upper()
            sequence = str(proposal["sequence"]).strip().upper()
            if parent not in archive.index:
                continue
            parent_row = archive.loc[parent]
            if isinstance(parent_row, pd.DataFrame):
                parent_row = parent_row.iloc[-1]
            row: dict[str, Any] = {
                "epoch": epoch,
                "iteration": iteration_root.name,
                "sequence": sequence,
                "parent_sequence": parent,
                "proposal_type": str(proposal.get("proposal_type") or "unknown"),
                "edit_distance": int(float(proposal.get("edit_distance") or 0)),
                "mutations": str(proposal.get("mutations") or ""),
                "proposal_prior_score": float(proposal.get("proposal_prior_score") or 0),
                "parent_selection_score": float(proposal.get("parent_selection_score") or 0),
                "screen_order": int(order[sequence]),
                "observation_count": 0,
            }
            row.update({f"parent_{name}": float(parent_row[name]) for name in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def _write_manifest(output_dir: Path, source: dict[str, Any]) -> None:
    files = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    _atomic_json(output_dir / "manifest.json", {**source, "files": files})


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--before-epoch", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260910)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, input_manifest = build_dataset(
        args.runtime_root, target=args.target, before_epoch=args.before_epoch
    )
    length = int(dataset.sequence.str.len().mode().iloc[0])
    dataset = dataset[
        (dataset.sequence.str.len() == length)
        & (dataset.parent_sequence.str.len() == length)
    ].reset_index(drop=True)
    completed_epochs = sorted(dataset.epoch.unique())
    if len(completed_epochs) < 3:
        raise SystemExit("v2 requires exact data from at least three completed epochs")
    holdout_epoch = int(completed_epochs[-1])
    training = dataset[dataset.epoch < holdout_epoch].reset_index(drop=True)
    holdout = dataset[dataset.epoch == holdout_epoch].reset_index(drop=True)
    seen_sequences = set(training.sequence)
    novel_holdout = holdout[~holdout.sequence.isin(seen_sequences)].reset_index(drop=True)
    if len(training) < 100 or len(novel_holdout) < 20:
        raise SystemExit(
            f"insufficient v2 split: train={len(training)} novel_holdout={len(novel_holdout)}"
        )

    selected, cv_results, cv_summary = grouped_model_selection(
        training, length=length, workers=args.workers, seed=args.random_seed
    )
    final_model = _fit(
        selected,
        feature_matrix(training, length=length),
        training.target_improvement.to_numpy(dtype=float),
        args.workers,
        args.random_seed,
    )
    holdout_predictions = score_frame(final_model, novel_holdout, length=length)
    holdout_metrics = allocation_metrics(
        holdout_predictions,
        holdout_predictions.predicted_improvement.to_numpy(),
        holdout_predictions.probability_improves.to_numpy(),
    )
    internal = cv_summary[selected.name]
    gates = {
        "internal_parent_group_spearman_at_least_0_15": (
            internal["spearman_improvement"] >= 0.15
        ),
        "internal_recall40_beats_or_ties_heuristic": (
            internal["model_top10_recall_at_40"]
            >= internal["heuristic_top10_recall_at_40"]
        ),
        "holdout_spearman_at_least_0_10": (
            holdout_metrics["spearman_improvement"] >= 0.10
        ),
        "holdout_regret20_beats_or_ties_heuristic": (
            holdout_metrics["model_regret_at_20"]
            <= holdout_metrics["heuristic_regret_at_20"]
        ),
        "holdout_recall40_beats_or_ties_heuristic": (
            holdout_metrics["model_top10_recall_at_40"]
            >= holdout_metrics["heuristic_top10_recall_at_40"]
        ),
    }
    accepted = all(gates.values())

    dataset.to_csv(args.output_dir / "known_parent_exact_dataset.csv.gz", index=False)
    cv_results.to_csv(args.output_dir / "parent_group_cv.csv", index=False)
    holdout_predictions.to_csv(
        args.output_dir / "chronological_holdout_predictions.csv", index=False
    )
    current = current_proposals(
        args.runtime_root, epoch=args.before_epoch, target=args.target
    )
    if not current.empty:
        current_scored = score_frame(final_model, current, length=length)
        current_scored.sort_values("model_rank").to_csv(
            args.output_dir / "current_epoch_predictions.csv", index=False
        )
    joblib.dump(final_model, args.output_dir / "model.joblib")
    config = {
        "protocol": "known ancestry -> parent-group CV -> sealed chronological holdout",
        "target": args.target,
        "before_epoch": args.before_epoch,
        "holdout_epoch": holdout_epoch,
        "training_epochs": [int(value) for value in sorted(training.epoch.unique())],
        "training_examples": len(training),
        "holdout_examples": len(holdout),
        "novel_holdout_examples": len(novel_holdout),
        "modal_sequence_length": length,
        "workers": args.workers,
        "random_seed": args.random_seed,
        "model_specs": [asdict(ModelSpec(*values)) for values in MODEL_SPECS],
        "target_definition": (
            "parent validator-compatible dense rank sum minus median exact child "
            "dense rank sum in the same trailing-300 reference field; higher is better"
        ),
    }
    _atomic_json(args.output_dir / "config.json", config)
    summary = {
        "accepted_for_live_allocation": accepted,
        "acceptance_gates": gates,
        "selected_model": asdict(selected),
        "internal_parent_group_cv": cv_summary,
        "chronological_holdout": holdout_metrics,
        "data": {
            "completed_epochs": [int(value) for value in completed_epochs],
            "training_examples": len(training),
            "holdout_examples": len(holdout),
            "novel_holdout_examples": len(novel_holdout),
            "repeat_observation_examples": int((dataset.observation_count > 1).sum()),
        },
        "decision": (
            "eligible for shadow/live integration"
            if accepted
            else "remain shadow-only; heuristic retains live control"
        ),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    _write_manifest(args.output_dir, input_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
