#!/usr/bin/env python3
"""Analyse the existing five-pass, same-epoch rival-versus-us exact replay.

The replay was run before this report.  This script is intentionally local and
read-only: it turns the retained metric vectors into reproducible aggregate and
per-metric comparisons without starting another GPU job.
"""

from __future__ import annotations

import bisect
import collections
import csv
import glob
import json
import math
import sqlite3
import statistics
from pathlib import Path


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
CONFIDENCE_DRIVER_STRENGTH = {
    "interaction_pae": 0.812,
    "min_design_to_target_pae": 0.729,
    "design_to_target_iptm": 0.725,
    "design_ptm": 0.585,
    "design_iiptm": 0.463,
}
MEASUREMENT_BIAS = {
    "design_ptm": 0.00078,
    "design_iiptm": 0.00867,
    "design_to_target_iptm": 0.00071,
    "min_design_to_target_pae": -0.01154,
    "interaction_pae": -0.03304,
}


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def observations(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    result = {}
    for by_sequence in (payload.get("per_nanobody_components") or {}).values():
        for sequence, by_target in by_sequence.items():
            if TARGET in by_target:
                result[sequence] = by_target[TARGET]
    return result


def selection_scores(metrics_by_sequence: dict[str, dict[str, float]]) -> dict[str, float]:
    """Reproduce rank_exact's historical-drivers score without pandas/numpy."""
    compiled = {
        metric: sorted({float(values[metric]) for values in metrics_by_sequence.values()})
        for metric in METRICS
    }
    confidence_total = sum(CONFIDENCE_DRIVER_STRENGTH.values())
    weights = {
        metric: (
            CONFIDENCE_DRIVER_STRENGTH[metric] / confidence_total * 5
            if metric in CONFIDENCE_DRIVER_STRENGTH
            else 1.0
        )
        for metric in METRICS
    }
    scores = {}
    for sequence, values in metrics_by_sequence.items():
        score = 0.0
        for metric, direction in METRICS.items():
            value = float(values[metric]) - MEASUREMENT_BIAS.get(metric, 0.0)
            field = compiled[metric]
            if direction == "min":
                rank = bisect.bisect_left(field, value) + 1
            else:
                rank = len(field) - bisect.bisect_right(field, value) + 1
            score += weights[metric] * rank
        scores[sequence] = score
    return scores


def average_ranks(values: list[float]) -> list[float]:
    """One-based average ranks, including ties."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(average_ranks(left), average_ranks(right))


def main() -> int:
    root = Path(__file__).resolve().parent
    replay_dir = root / "remeasure"
    output_dir = root / "results"
    manifest = json.loads((replay_dir / "remeasure_manifest.json").read_text())
    meta_by_sequence = {item["sequence"]: item for item in manifest}
    passes = [observations(Path(path)) for path in sorted(glob.glob(str(replay_dir / "pass_*.json")))]
    folds: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for one_pass in passes:
        for sequence, metrics in one_pass.items():
            folds[sequence].append(metrics)

    averaged = {
        sequence: {
            metric: statistics.mean(float(fold[metric]) for fold in sequence_folds)
            for metric in METRICS
        }
        for sequence, sequence_folds in folds.items()
    }
    averaged_scores = selection_scores(averaged)

    graph = sqlite3.connect(f"file:{root / 'mutations.sqlite'}?mode=ro", uri=True)
    graph.row_factory = sqlite3.Row
    public_metrics = {
        (int(row["epoch"]), row["sequence"]): json.loads(row["metrics_json"])
        for row in graph.execute(
            """
            SELECT o.epoch,n.sequence,o.metrics_json
            FROM observations o JOIN nodes n ON n.id=o.node_id
            WHERE o.source='public_validator_api' AND n.target=?
            """,
            (TARGET,),
        )
    }
    leaderboard = sqlite3.connect(
        f"file:{root.parent.parent / 'data' / 'nanobody_dash.sqlite'}?mode=ro", uri=True
    )
    public_scores = {
        (int(epoch), sequence): float(score)
        for epoch, sequence, score in leaderboard.execute(
            "SELECT epoch,sequence,score FROM entries WHERE sequence IS NOT NULL"
        )
    }

    pairs: dict[int, dict[str, str]] = collections.defaultdict(dict)
    for sequence, meta in meta_by_sequence.items():
        pairs[int(meta["epoch"])][meta["side"]] = sequence

    aggregate_pairs = []
    for epoch, sides in sorted(pairs.items()):
        if set(sides) < {"ours", "rival"}:
            continue
        ours = averaged_scores[sides["ours"]]
        rival = averaged_scores[sides["rival"]]
        aggregate_pairs.append(
            {
                "epoch": epoch,
                "ours_score": ours,
                "rival_score": rival,
                "ours_minus_rival": ours - rival,
                "depth": min(len(folds[sides["ours"]]), len(folds[sides["rival"]])),
            }
        )

    metric_rows = []
    for metric, direction in METRICS.items():
        advantages = []
        paired_deltas = []
        within_sd = []
        for epoch, sides in sorted(pairs.items()):
            if set(sides) < {"ours", "rival"}:
                continue
            ours = averaged[sides["ours"]][metric]
            rival = averaged[sides["rival"]][metric]
            raw_delta = rival - ours
            advantages.append(raw_delta if direction == "max" else -raw_delta)
            paired_deltas.append(raw_delta)
        for sequence_folds in folds.values():
            values = [float(fold[metric]) for fold in sequence_folds]
            within_sd.append(statistics.pstdev(values))
        metric_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "pairs": len(advantages),
                "mean_rival_advantage": statistics.mean(advantages),
                "median_rival_advantage": statistics.median(advantages),
                "rival_better_pairs": sum(value > 0 for value in advantages),
                "rival_better_rate": sum(value > 0 for value in advantages) / len(advantages),
                "median_within_sequence_sd": statistics.median(within_sd),
            }
        )

    calibration_rows = []
    for metric in METRICS:
        local_values, validator_values = [], []
        for sequence, meta in meta_by_sequence.items():
            validator = public_metrics.get((int(meta["epoch"]), sequence))
            if validator and sequence in averaged:
                local_values.append(float(averaged[sequence][metric]))
                validator_values.append(float(validator[metric]))
        errors = [local - validator for local, validator in zip(local_values, validator_values)]
        calibration_rows.append(
            {
                "metric": metric,
                "sequences": len(local_values),
                "pearson_local_vs_public": pearson(local_values, validator_values),
                "spearman_local_vs_public": spearman(local_values, validator_values),
                "mean_local_minus_public": statistics.mean(errors),
                "median_absolute_error": statistics.median(abs(value) for value in errors),
            }
        )

    local_aggregate, validator_aggregate = [], []
    for sequence, meta in meta_by_sequence.items():
        public_score = public_scores.get((int(meta["epoch"]), sequence))
        if public_score is not None and sequence in averaged_scores:
            local_aggregate.append(averaged_scores[sequence])
            validator_aggregate.append(public_score)

    per_pass_pairs = []
    for pass_number, one_pass in enumerate(passes, 1):
        scores = selection_scores(one_pass)
        for epoch, sides in sorted(pairs.items()):
            if sides.get("ours") not in scores or sides.get("rival") not in scores:
                continue
            delta = scores[sides["ours"]] - scores[sides["rival"]]
            per_pass_pairs.append(
                {
                    "pass": pass_number,
                    "epoch": epoch,
                    "ours_score": scores[sides["ours"]],
                    "rival_score": scores[sides["rival"]],
                    "ours_minus_rival": delta,
                    "rival_better": delta > 0,
                }
            )

    deltas = [item["ours_minus_rival"] for item in aggregate_pairs]
    per_pass_deltas = [item["ours_minus_rival"] for item in per_pass_pairs]
    standard_error = statistics.pstdev(deltas) / math.sqrt(len(deltas))
    report = {
        "passes": len(passes),
        "sequences": len(folds),
        "paired_epochs": len(deltas),
        "depth_per_sequence": min(len(items) for items in folds.values()),
        "averaged_historical_driver_score": {
            "mean_ours_minus_rival": statistics.mean(deltas),
            "median_ours_minus_rival": statistics.median(deltas),
            "standard_error": standard_error,
            "z": statistics.mean(deltas) / standard_error if standard_error else None,
            "epochs_rival_better": sum(value > 0 for value in deltas),
            "note": "negative means our sequence scored better; rival was the realised public winner",
        },
        "individual_pass_score": {
            "comparisons": len(per_pass_deltas),
            "median_ours_minus_rival": statistics.median(per_pass_deltas),
            "rival_better_rate": sum(value > 0 for value in per_pass_deltas)
            / len(per_pass_deltas),
        },
        "metric_summary": metric_rows,
        "calibration": {
            "sequences": len(local_aggregate),
            "aggregate_spearman_local_vs_public": spearman(
                local_aggregate, validator_aggregate
            ),
            "metric_summary": calibration_rows,
        },
        "selection_caveat": "rival sequences were selected because they won publicly, so this is retrospective and subject to winner's curse",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "remeasure_aggregate_pairs.csv", aggregate_pairs)
    write_csv(output_dir / "remeasure_metric_summary.csv", metric_rows)
    write_csv(output_dir / "remeasure_calibration.csv", calibration_rows)
    write_csv(output_dir / "remeasure_pass_pairs.csv", per_pass_pairs)
    (output_dir / "remeasure_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["averaged_historical_driver_score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
