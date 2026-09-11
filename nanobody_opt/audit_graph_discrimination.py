"""Audit whether exact outcomes are smooth enough to prune mutation-graph arms.

This is deliberately an outcome audit, not an acquisition model.  It hides one
candidate at a time, exposes the outcomes of related candidates, and asks how
much those graph relationships say about the hidden exact outcome.  A graph is
not safe for adaptive pruning merely because nearby sequences look similar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nanobody_opt.simulate_round import (
    METRICS,
    METRIC_CATEGORIES,
    dense_rank_sums,
    rank_sum_in_field,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_exact(path: Path, *, target: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    if str(payload.get("target")) != target:
        raise ValueError(f"{path} targets {payload.get('target')!r}, not {target}")
    rows: list[dict[str, Any]] = []
    for by_sequence in payload.get("per_nanobody_components", {}).values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target.get(target)
            if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
                continue
            values = {metric: float(metrics[metric]) for metric in METRICS}
            if all(math.isfinite(value) for value in values.values()):
                rows.append({"sequence": str(sequence).strip().upper(), **values})
    frame = pd.DataFrame(rows).drop_duplicates("sequence", keep="last")
    if frame.empty:
        raise ValueError(f"{path} has no complete {target} exact observations")
    return frame.reset_index(drop=True)


def bootstrap_validator_scores(
    frame: pd.DataFrame,
    archive: pd.DataFrame,
    *,
    recent: int,
    field_size: int,
    bootstrap_fields: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    reference = archive.tail(recent).reset_index(drop=True)
    if len(reference) < field_size:
        raise ValueError("archive is smaller than the requested competitor field")
    missing = set(METRICS) - set(reference.columns)
    if missing:
        raise ValueError(f"archive lacks validator metrics: {sorted(missing)}")
    rng = np.random.default_rng(random_seed)
    fields = [
        reference.iloc[rng.choice(len(reference), field_size, replace=False)]
        for _ in range(bootstrap_fields)
    ]
    observations = np.asarray([
        [
            rank_sum_in_field(
                {metric: float(row[metric]) for metric in METRICS}, field
            )
            for field in fields
        ]
        for _, row in frame.iterrows()
    ])
    return np.median(observations, axis=1), np.quantile(observations, 0.9, axis=1)


def distance_matrix(sequences: list[str]) -> np.ndarray:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("candidate graph requires one sequence length")
    encoded = np.asarray([list(sequence.encode("ascii")) for sequence in sequences])
    return np.count_nonzero(encoded[:, None, :] != encoded[None, :, :], axis=2)


def _mutation_sets(frame: pd.DataFrame) -> list[set[str]]:
    if "mutations" not in frame:
        return [set() for _ in range(len(frame))]
    return [
        {item for item in str(value).split(";") if item and item != "nan"}
        if pd.notna(value)
        else set()
        for value in frame.mutations
    ]


def relationship_masks(
    frame: pd.DataFrame, distances: np.ndarray
) -> dict[str, np.ndarray]:
    size = len(frame)
    diagonal = np.eye(size, dtype=bool)
    parents = frame.get(
        "parent_sequence", pd.Series([""] * size, index=frame.index)
    ).fillna("").astype(str).to_numpy()
    same_parent = (parents[:, None] == parents[None, :]) & (parents[:, None] != "")
    mutation_sets = _mutation_sets(frame)
    shared_mutation = np.asarray([
        [bool(mutation_sets[left] & mutation_sets[right]) for right in range(size)]
        for left in range(size)
    ])
    return {
        "distance_1": (distances == 1) & ~diagonal,
        "distance_at_most_2": (distances > 0) & (distances <= 2),
        "same_parent": same_parent & ~diagonal,
        "shared_mutation": shared_mutation & ~diagonal,
        "same_parent_and_shared_mutation": (
            same_parent & shared_mutation & ~diagonal
        ),
    }


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _conditional_rate(values: np.ndarray, condition: np.ndarray) -> float | None:
    return float(values[condition].mean()) if condition.any() else None


def relationship_discrimination(
    scores: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    good_fraction: float = 0.1,
    bad_fraction: float = 0.5,
) -> dict[str, Any]:
    ranks = pd.Series(scores).rank(method="min").to_numpy()
    good = ranks <= max(1, math.ceil(len(scores) * good_fraction))
    bad = ranks > math.floor(len(scores) * (1.0 - bad_fraction))
    baseline_bad = float(bad.mean())
    result: dict[str, Any] = {
        "good_definition": f"best {good_fraction:.0%}",
        "bad_definition": f"worst {bad_fraction:.0%}",
        "baseline_good_rate": float(good.mean()),
        "baseline_bad_rate": baseline_bad,
        "relationships": {},
    }
    for name, mask in masks.items():
        left, right = np.where(mask)
        if not len(left):
            continue
        source_good = good[left]
        source_bad = bad[left]
        bad_given_bad = _conditional_rate(bad[right], source_bad)
        result["relationships"][name] = {
            "directed_pairs": int(len(left)),
            "score_spearman": _finite_spearman(scores[left], scores[right]),
            "p_good_neighbor_given_good_source": _conditional_rate(
                good[right], source_good
            ),
            "p_good_neighbor_given_bad_source": _conditional_rate(
                good[right], source_bad
            ),
            "p_bad_neighbor_given_bad_source": bad_given_bad,
            "bad_elimination_precision_lift": (
                bad_given_bad / baseline_bad if bad_given_bad is not None else None
            ),
        }
    return result


def component_smoothness(
    frame: pd.DataFrame, masks: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for relation, mask in masks.items():
        left, right = np.where(np.triu(mask, 1))
        output[relation] = {
            metric: _finite_spearman(
                frame[metric].to_numpy(dtype=float)[left],
                frame[metric].to_numpy(dtype=float)[right],
            )
            for metric in METRICS
        }
    return output


def category_discrimination(
    frame: pd.DataFrame, masks: dict[str, np.ndarray]
) -> dict[str, Any]:
    output = {}
    for category, metrics in METRIC_CATEGORIES.items():
        directions = {metric: METRICS[metric] for metric in metrics}
        scores = dense_rank_sums(frame, directions)
        output[category] = relationship_discrimination(scores, masks)
    return output


def graph_leave_one_out_upper_bound(
    frame: pd.DataFrame,
    scores: np.ndarray,
    distances: np.ndarray,
    *,
    maximum_distance: int,
    neighbours: int,
    good_fraction: float = 0.1,
) -> dict[str, Any]:
    """Use every other hidden outcome; this is an optimistic graph-only bound."""
    sequences = frame.sequence.astype(str).to_numpy()
    predictions = []
    fallback = float(np.median(scores))
    for index in range(len(frame)):
        candidates = np.flatnonzero(
            (distances[index] > 0) & (distances[index] <= maximum_distance)
        )
        ordered = sorted(
            candidates,
            key=lambda other: (distances[index, other], sequences[other]),
        )[:neighbours]
        if not ordered:
            predictions.append(fallback)
            continue
        selected = np.asarray(ordered, dtype=int)
        weights = 1.0 / distances[index, selected]
        predictions.append(float(np.average(scores[selected], weights=weights)))
    predicted = np.asarray(predictions)
    good_count = max(1, math.ceil(len(frame) * good_fraction))
    actual_good = set(np.argsort(scores, kind="stable")[:good_count])
    order = sorted(
        range(len(frame)), key=lambda index: (predicted[index], sequences[index])
    )
    result: dict[str, Any] = {
        "description": "all other exact outcomes visible; not achievable live",
        "maximum_distance": maximum_distance,
        "neighbours": neighbours,
        "score_spearman": _finite_spearman(scores, predicted),
    }
    for budget in (40, 80, 160):
        selected = set(order[: min(budget, len(order))])
        result[f"top_10pct_recall_at_{budget}"] = (
            len(selected & actual_good) / len(actual_good)
        )
    return result


def audit(
    frame: pd.DataFrame,
    *,
    score_column: str,
) -> dict[str, Any]:
    scores = frame[score_column].to_numpy(dtype=float)
    distances = distance_matrix(frame.sequence.astype(str).tolist())
    masks = relationship_masks(frame, distances)
    return {
        "candidate_count": int(len(frame)),
        "parent_count": int(frame.parent_sequence.nunique()),
        "score_column": score_column,
        "relationship_discrimination": relationship_discrimination(scores, masks),
        "component_smoothness": component_smoothness(frame, masks),
        "category_discrimination": category_discrimination(frame, masks),
        "graph_leave_one_out_upper_bounds": [
            graph_leave_one_out_upper_bound(
                frame,
                scores,
                distances,
                maximum_distance=maximum_distance,
                neighbours=neighbours,
            )
            for maximum_distance in (1, 2, 3)
            for neighbours in (3, 5, 10)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--exact", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scored-output", type=Path)
    parser.add_argument("--recent", type=int, default=300)
    parser.add_argument("--field-size", type=int, default=49)
    parser.add_argument("--bootstrap-fields", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    exact = load_exact(args.exact, target=args.target)
    proposals = pd.read_csv(args.proposals)
    proposals.sequence = proposals.sequence.astype(str).str.strip().str.upper()
    frame = exact.merge(proposals, on="sequence", how="left", validate="one_to_one")
    if frame.parent_sequence.isna().any():
        raise SystemExit("some exact candidates lack proposal graph metadata")
    archive = pd.read_csv(args.archive)
    median, p90 = bootstrap_validator_scores(
        frame,
        archive,
        recent=args.recent,
        field_size=args.field_size,
        bootstrap_fields=args.bootstrap_fields,
        random_seed=args.random_seed,
    )
    frame["bootstrap_median_rank_sum"] = median
    frame["bootstrap_p90_rank_sum"] = p90
    frame["bootstrap_rank"] = frame.bootstrap_median_rank_sum.rank(
        method="min"
    ).astype(int)

    summary = {
        "protocol": {
            "outcome": "median validator rank sum over historical bootstrap fields",
            "interpretation": (
                "conditional neighbour rates test whether a revealed bad arm can "
                "safely eliminate correlated candidates"
            ),
            "recent": args.recent,
            "field_size": args.field_size,
            "bootstrap_fields": args.bootstrap_fields,
            "random_seed": args.random_seed,
        },
        "inputs": {
            "proposals": {
                "path": str(args.proposals),
                "sha256": _sha256(args.proposals),
            },
            "exact": {"path": str(args.exact), "sha256": _sha256(args.exact)},
            "archive": {"path": str(args.archive), "sha256": _sha256(args.archive)},
        },
        **audit(frame, score_column="bootstrap_median_rank_sum"),
        "best_candidates": frame.sort_values(
            ["bootstrap_median_rank_sum", "bootstrap_p90_rank_sum", "sequence"],
        ).head(min(10, len(frame)))[
            [
                "sequence",
                "bootstrap_rank",
                "bootstrap_median_rank_sum",
                "bootstrap_p90_rank_sum",
                "parent_sequence",
                "proposal_type",
                "edit_distance",
                "mutations",
            ]
        ].to_dict(orient="records"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.scored_output:
        args.scored_output.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values(
            ["bootstrap_median_rank_sum", "bootstrap_p90_rank_sum", "sequence"]
        ).to_csv(args.scored_output, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
