#!/usr/bin/env python3
"""Backtest selector policies against our realised public submissions.

The unit of evaluation is an epoch: among sequences we actually submitted in
that epoch, did the local exact measurements order them the same way as the
public validator?  Local metrics use only observations available no later than
the submission epoch.  The public epoch field is used only as the dense-rank
reference for this diagnostic; it is not available to the live selector.
"""

from __future__ import annotations

import bisect
import collections
import itertools
import json
import math
import sqlite3
import statistics
from pathlib import Path

from analyze_remeasure import average_ranks, pearson, spearman, write_csv

from nanobody_dash import DEFAULT_COLDKEY


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
CONFIDENCE = {
    "design_iiptm",
    "design_ptm",
    "design_to_target_iptm",
    "min_design_to_target_pae",
    "interaction_pae",
}
CONFIDENCE_DRIVER_STRENGTH = {
    "interaction_pae": 0.812,
    "min_design_to_target_pae": 0.729,
    "design_to_target_iptm": 0.725,
    "design_ptm": 0.585,
    "design_iiptm": 0.463,
}
MEASURED_SIGNAL_STRENGTH = {
    "design_to_target_iptm": 0.263,
    "design_ptm": 0.217,
    "interaction_pae": 0.129,
    "design_iiptm": 0.075,
    "min_design_to_target_pae": 0.068,
}
MEASUREMENT_BIAS = {
    "design_ptm": 0.00078,
    "design_iiptm": 0.00867,
    "design_to_target_iptm": 0.00071,
    "min_design_to_target_pae": -0.01154,
    "interaction_pae": -0.03304,
}


def normalised(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {
        metric: (weights.get(metric, 0.0) / total * 5 if metric in CONFIDENCE else 1.0)
        for metric in METRICS
    }


POLICIES = {
    "plain": ({metric: 1.0 for metric in METRICS}, False),
    "historical-drivers": (normalised(CONFIDENCE_DRIVER_STRENGTH), True),
    "measured-signal": (normalised(MEASURED_SIGNAL_STRENGTH), True),
    "confidence-equal": (
        {metric: (1.0 if metric in CONFIDENCE else 0.0) for metric in METRICS},
        True,
    ),
    "design-target-iptm": (
        {metric: (1.0 if metric == "design_to_target_iptm" else 0.0) for metric in METRICS},
        True,
    ),
    "design-iiptm": (
        {metric: (1.0 if metric == "design_iiptm" else 0.0) for metric in METRICS},
        True,
    ),
}


def dense_rank(value: float, field: list[float], direction: str) -> int:
    if direction == "min":
        return bisect.bisect_left(field, value) + 1
    return len(field) - bisect.bisect_right(field, value) + 1


def policy_score(
    metrics: dict[str, float],
    field: dict[str, list[float]],
    weights: dict[str, float],
    bias_correct: bool,
) -> float:
    total = 0.0
    for metric, direction in METRICS.items():
        value = float(metrics[metric])
        if bias_correct:
            value -= MEASUREMENT_BIAS.get(metric, 0.0)
        total += weights[metric] * dense_rank(value, field[metric], direction)
    return total


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    graph = sqlite3.connect(
        f"file:{root / 'analysis' / 'nanobody-gap' / 'mutations.sqlite'}?mode=ro",
        uri=True,
    )
    graph.row_factory = sqlite3.Row
    leaderboard = sqlite3.connect(
        f"file:{root / 'data' / 'nanobody_dash.sqlite'}?mode=ro", uri=True
    )
    leaderboard.row_factory = sqlite3.Row

    nodes = {
        row["sequence"]: int(row["id"])
        for row in graph.execute("SELECT id,sequence FROM nodes WHERE target=?", (TARGET,))
    }
    local_by_node: dict[int, list[tuple[int, dict[str, float]]]] = collections.defaultdict(list)
    for row in graph.execute(
        """
        SELECT node_id,epoch,metrics_json FROM observations
        WHERE source IN ('live_exact_screen','live_exact_repeat','local_exact_raw')
          AND epoch IS NOT NULL
        ORDER BY node_id,epoch
        """
    ):
        local_by_node[int(row["node_id"])].append(
            (int(row["epoch"]), json.loads(row["metrics_json"]))
        )

    public_fields: dict[int, dict[str, list[float]]] = {}
    public_metrics: dict[tuple[int, str], dict[str, float]] = {}
    raw_public: dict[int, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for row in graph.execute(
        """
        SELECT o.epoch,n.sequence,o.metrics_json
        FROM observations o JOIN nodes n ON n.id=o.node_id
        WHERE o.source='public_validator_api' AND n.target=?
        """,
        (TARGET,),
    ):
        epoch = int(row["epoch"])
        metrics = json.loads(row["metrics_json"])
        public_metrics[(epoch, row["sequence"])] = metrics
        for metric in METRICS:
            raw_public[epoch][metric].append(float(metrics[metric]))
    for epoch, metrics in raw_public.items():
        public_fields[epoch] = {
            metric: sorted(set(values)) for metric, values in metrics.items()
        }

    examples = []
    for entry in leaderboard.execute(
        """
        SELECT epoch,uid,sequence,rank,score FROM entries
        WHERE coldkey=? AND sequence IS NOT NULL ORDER BY epoch,rank
        """,
        (DEFAULT_COLDKEY,),
    ):
        epoch, sequence = int(entry["epoch"]), entry["sequence"]
        node_id = nodes.get(sequence)
        validator = public_metrics.get((epoch, sequence))
        field = public_fields.get(epoch)
        if node_id is None or validator is None or field is None:
            continue
        observations = [metrics for seen_epoch, metrics in local_by_node[node_id] if seen_epoch <= epoch]
        if not observations:
            continue
        local = {
            metric: statistics.mean(float(item[metric]) for item in observations)
            for metric in METRICS
        }
        record = {
            "epoch": epoch,
            "uid": int(entry["uid"]),
            "sequence": sequence,
            "public_rank": int(entry["rank"]),
            "public_score": float(entry["score"]),
            "observation_count": len(observations),
            "local": local,
            "public": validator,
        }
        for policy, (weights, correct) in POLICIES.items():
            record[policy] = policy_score(local, field, weights, correct)
        examples.append(record)

    by_epoch: dict[int, list[dict]] = collections.defaultdict(list)
    for example in examples:
        by_epoch[example["epoch"]].append(example)

    epoch_rows = []
    for epoch, group in sorted(by_epoch.items()):
        if len(group) < 2:
            continue
        for policy in POLICIES:
            predicted = [float(item[policy]) for item in group]
            realised = [float(item["public_score"]) for item in group]
            chosen = min(group, key=lambda item: (item[policy], item["sequence"]))
            oracle = min(group, key=lambda item: (item["public_score"], item["sequence"]))
            agreements = []
            for left, right in itertools.combinations(group, 2):
                predicted_delta = float(left[policy]) - float(right[policy])
                realised_delta = float(left["public_score"]) - float(right["public_score"])
                if predicted_delta and realised_delta:
                    agreements.append((predicted_delta > 0) == (realised_delta > 0))
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "policy": policy,
                    "candidates": len(group),
                    "spearman": spearman(predicted, realised),
                    "pair_accuracy": statistics.mean(agreements) if agreements else None,
                    "chosen_public_score": chosen["public_score"],
                    "oracle_public_score": oracle["public_score"],
                    "chosen_is_oracle": chosen["sequence"] == oracle["sequence"],
                    "chosen_public_rank": chosen["public_rank"],
                }
            )

    epochs = sorted({row["epoch"] for row in epoch_rows})
    holdout_start = epochs[max(0, int(len(epochs) * 0.7))]
    summaries = []
    for window, selected_epochs in (
        ("all", set(epochs)),
        ("recent_30pct_holdout", {epoch for epoch in epochs if epoch >= holdout_start}),
    ):
        for policy in POLICIES:
            rows = [
                row for row in epoch_rows
                if row["policy"] == policy and row["epoch"] in selected_epochs
            ]
            correlations = [row["spearman"] for row in rows if row["spearman"] is not None]
            accuracies = [row["pair_accuracy"] for row in rows if row["pair_accuracy"] is not None]
            summaries.append(
                {
                    "window": window,
                    "policy": policy,
                    "epochs": len(rows),
                    "mean_epoch_spearman": statistics.mean(correlations) if correlations else None,
                    "median_epoch_spearman": statistics.median(correlations) if correlations else None,
                    "mean_pair_accuracy": statistics.mean(accuracies) if accuracies else None,
                    "oracle_pick_rate": statistics.mean(row["chosen_is_oracle"] for row in rows),
                    "median_chosen_public_rank": statistics.median(row["chosen_public_rank"] for row in rows),
                    "mean_score_regret": statistics.mean(
                        row["chosen_public_score"] - row["oracle_public_score"] for row in rows
                    ),
                }
            )

    metric_rows = []
    for metric in METRICS:
        local = [float(item["local"][metric]) for item in examples]
        public = [float(item["public"][metric]) for item in examples]
        metric_rows.append(
            {
                "metric": metric,
                "sequences": len(local),
                "spearman_local_vs_public": spearman(local, public),
                "pearson_local_vs_public": pearson(local, public),
                "mean_local_minus_public": statistics.mean(
                    left - right for left, right in zip(local, public)
                ),
            }
        )

    output = root / "analysis" / "nanobody-gap" / "results"
    write_csv(output / "selector_policy_epochs.csv", epoch_rows)
    write_csv(output / "selector_policy_summary.csv", summaries)
    write_csv(output / "selector_metric_calibration.csv", metric_rows)
    report = {
        "matched_submissions": len(examples),
        "epochs": len(epochs),
        "holdout_start_epoch": holdout_start,
        "summaries": summaries,
        "metric_calibration": metric_rows,
        "caveat": "diagnostic uses the realised public epoch field as rank reference; live ranking must use historical fields",
    }
    (output / "selector_calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "matched_submissions": len(examples),
        "epochs": len(epochs),
        "holdout_start_epoch": holdout_start,
        "holdout": [row for row in summaries if row["window"] == "recent_30pct_holdout"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
