#!/usr/bin/env python3
"""Evaluate preregistered A/B/C fleets on untouched draws 9-16."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selector import (
    FLEET_NONINFERIORITY_METRICS,
    METRICS,
    TARGET,
    candidate_field_outcome,
    dense_rank_sums,
)


HERE = Path(__file__).resolve().parent
FRESH_DRAWS = tuple(range(9, 17))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_draw(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol: {path}")
    if int(payload.get("n", -1)) != 888:
        raise RuntimeError(f"wrong candidate count: {path}")
    found: dict[str, dict[str, float]] = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            if not all(metric in metrics for metric in METRICS):
                raise RuntimeError(f"incomplete metrics for {sequence}")
            found[sequence] = {
                metric: float(metrics[metric]) for metric in METRICS
            }
    if len(found) != 888:
        raise RuntimeError(f"draw sequence mismatch: {path}")
    return found


def fleet_field_win(
    fleet: list[str], draw: dict[str, dict[str, float]], field: dict[str, Any]
) -> bool:
    records = field["records"] + [draw[sequence] for sequence in fleet]
    scores = dense_rank_sums(records)
    split = len(field["records"])
    return min(scores[split:]) <= min(scores[:split])


def evaluate_arm(
    fleet: list[str],
    draws: list[dict[str, dict[str, float]]],
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    blocks = []
    for draw_number, draw in zip(FRESH_DRAWS, draws, strict=True):
        individual_wins = 0
        top10_entries = 0
        fleet_wins = 0
        for field in fields:
            for sequence in fleet:
                outcome = candidate_field_outcome(draw[sequence], field["records"])
                individual_wins += int(outcome["win"])
                top10_entries += int(outcome["top10"])
            fleet_wins += int(fleet_field_win(fleet, draw, field))
        hard_metrics = {
            metric: statistics.median(draw[sequence][metric] for sequence in fleet)
            for metric in FLEET_NONINFERIORITY_METRICS
        }
        blocks.append({
            "draw": draw_number,
            "fleet_size": len(fleet),
            "field_count": len(fields),
            "individual_wins": individual_wins,
            "individual_entries": len(fleet) * len(fields),
            "individual_win_per_entry": individual_wins / (len(fleet) * len(fields)),
            "top10_entries": top10_entries,
            "top10_entry_opportunities": len(fleet) * len(fields),
            "top10_per_entry": top10_entries / (len(fleet) * len(fields)),
            "fleet_wins": fleet_wins,
            "fleet_fields": len(fields),
            "fleet_win_probability": fleet_wins / len(fields),
            "fleet_median_hard_metrics": hard_metrics,
        })
    return {
        "fleet": fleet,
        "fleet_size": len(fleet),
        "draw_blocks": blocks,
        "individual_wins": sum(block["individual_wins"] for block in blocks),
        "individual_entries": sum(block["individual_entries"] for block in blocks),
        "top10_entries": sum(block["top10_entries"] for block in blocks),
        "top10_entry_opportunities": sum(
            block["top10_entry_opportunities"] for block in blocks
        ),
        "fleet_wins": sum(block["fleet_wins"] for block in blocks),
        "fleet_fields": sum(block["fleet_fields"] for block in blocks),
    }


def favorable_delta(left: float, right: float, direction: str) -> float:
    return left - right if direction == "max" else right - left


def compare(
    left_name: str,
    right_name: str,
    left: dict[str, Any],
    right: dict[str, Any],
    margins: dict[str, float],
) -> dict[str, Any]:
    block_rows = []
    for left_block, right_block in zip(
        left["draw_blocks"], right["draw_blocks"], strict=True
    ):
        hard = {
            metric: favorable_delta(
                left_block["fleet_median_hard_metrics"][metric],
                right_block["fleet_median_hard_metrics"][metric],
                direction,
            )
            for metric, direction in FLEET_NONINFERIORITY_METRICS.items()
        }
        block_rows.append({
            "draw": left_block["draw"],
            "individual_win_per_entry_delta": (
                left_block["individual_win_per_entry"]
                - right_block["individual_win_per_entry"]
            ),
            "top10_per_entry_delta": (
                left_block["top10_per_entry"] - right_block["top10_per_entry"]
            ),
            "fleet_win_probability_delta": (
                left_block["fleet_win_probability"]
                - right_block["fleet_win_probability"]
            ),
            "fleet_win_count_delta": (
                left_block["fleet_wins"] - right_block["fleet_wins"]
            ),
            "hard_metric_favorable_deltas": hard,
        })
    hard_noninferiority = {}
    for metric in FLEET_NONINFERIORITY_METRICS:
        values = [row["hard_metric_favorable_deltas"][metric] for row in block_rows]
        hard_noninferiority[metric] = {
            "median_favorable_delta": statistics.median(values),
            "worst_block_favorable_delta": min(values),
            "worst_block_margin": margins[metric],
            "median_nonregression": statistics.median(values) >= 0.0,
            "worst_block_noninferiority": min(values) >= -margins[metric],
        }
        hard_noninferiority[metric]["passed"] = (
            hard_noninferiority[metric]["median_nonregression"]
            and hard_noninferiority[metric]["worst_block_noninferiority"]
        )
    positive_fleet = [
        max(row["fleet_win_count_delta"], 0) for row in block_rows
    ]
    total_positive = sum(positive_fleet)
    individual_lodo = []
    top10_lodo = []
    for omitted in range(len(block_rows)):
        remaining = [row for index, row in enumerate(block_rows) if index != omitted]
        individual_lodo.append(
            sum(row["individual_win_per_entry_delta"] for row in remaining) > 0.0
        )
        top10_lodo.append(
            sum(row["top10_per_entry_delta"] for row in remaining) > 0.0
        )
    return {
        "left": left_name,
        "right": right_name,
        "draw_blocks": block_rows,
        "individual_favorable_draws": sum(
            row["individual_win_per_entry_delta"] > 0.0 for row in block_rows
        ),
        "top10_favorable_draws": sum(
            row["top10_per_entry_delta"] > 0.0 for row in block_rows
        ),
        "individual_leave_one_draw_out_favorable": individual_lodo,
        "top10_leave_one_draw_out_favorable": top10_lodo,
        "fleet_total_delta": left["fleet_wins"] - right["fleet_wins"],
        "fleet_positive_draws": sum(
            row["fleet_win_count_delta"] > 0 for row in block_rows
        ),
        "fleet_largest_positive_share": (
            max(positive_fleet) / total_positive if total_positive else 1.0
        ),
        "hard_metric_noninferiority": hard_noninferiority,
    }


def main() -> None:
    prereg = json.loads((HERE / "preregistration.json").read_text())
    for relative, expected in prereg["frozen_code_hashes"].items():
        if sha256(HERE / relative) != expected:
            raise RuntimeError(f"frozen code changed after preregistration: {relative}")
    fields = [
        json.loads((HERE / "reserved_fields" / f"epoch_{epoch}.json").read_text())
        for epoch in prereg["reserved_evaluation_fields"]
    ]
    for field in fields:
        expected = prereg["reserved_field_hashes"][str(field["epoch"])]
        path = HERE / "reserved_fields" / f"epoch_{field['epoch']}.json"
        if sha256(path) != expected:
            raise RuntimeError(f"reserved field hash changed: {field['epoch']}")
    draws = [
        load_draw(HERE / f"fresh_{draw:02d}" / "exact_all.json")
        for draw in FRESH_DRAWS
    ]
    arms = {
        name: evaluate_arm(record["fleet"], draws, fields)
        for name, record in prereg["arms"].items()
    }
    comparisons = {
        "proposal_A_to_B": compare(
            "B_ours_h2_current_selector", "A_current_pool_current_selector",
            arms["B"], arms["A"], prereg["fleet_metric_noninferiority_margins"],
        ),
        "selector_B_to_C": compare(
            "C_ours_h2_repaired_selector", "B_ours_h2_current_selector",
            arms["C"], arms["B"], prereg["fleet_metric_noninferiority_margins"],
        ),
        "combined_A_to_C": compare(
            "C_ours_h2_repaired_selector", "A_current_pool_current_selector",
            arms["C"], arms["A"], prereg["fleet_metric_noninferiority_margins"],
        ),
    }
    combined = comparisons["combined_A_to_C"]
    timing = json.loads((HERE / "run_metadata.json").read_text())
    projected = max(
        row["duration_seconds"] for row in timing["draw_timings"]
    ) * 2.0 * prereg["runtime_gate"]["projection_guard_multiplier"]
    gates = {
        "individual_wins_per_entry": combined["individual_favorable_draws"] >= 6,
        "top10_per_entry": combined["top10_favorable_draws"] >= 6,
        "leave_one_draw_out": all(
            combined["individual_leave_one_draw_out_favorable"]
        ) and all(combined["top10_leave_one_draw_out_favorable"]),
        "whole_fleet_probability": (
            combined["fleet_total_delta"] > 0
            and combined["fleet_positive_draws"] >= 2
            and combined["fleet_largest_positive_share"] <= 0.5
        ),
        "hard_metrics": all(
            item["passed"] for item in combined["hard_metric_noninferiority"].values()
        ),
        "runtime": projected <= prereg["runtime_gate"]["deadline_seconds"],
    }
    passed = all(gates.values())
    result = {
        "decision": "offline-proof-passed" if passed else "no-deploy",
        "passed": passed,
        "effective_independent_blocks": 8,
        "reserved_field_epochs": prereg["reserved_evaluation_fields"],
        "arms": arms,
        "comparisons": comparisons,
        "proof_gates": gates,
        "runtime": {
            "measured_draw_seconds": [
                row["duration_seconds"] for row in timing["draw_timings"]
            ],
            "projected_four_gpu_seconds_with_guard": projected,
            "deadline_seconds": prereg["runtime_gate"]["deadline_seconds"],
        },
        "run_metadata": timing,
    }
    (HERE / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Offline H2 + repaired-selector proof",
        "",
        "## Decision",
        "",
        "**PASS**" if passed else "**NO-DEPLOY**",
        "",
        "The eight scorer draws are the independent blocks. The 64 draw×field cells per arm are repeated measurements, not 64 independent observations.",
        "",
        "## Arm outcomes",
        "",
        "| arm | individual wins/entries | top-10/entries | whole-fleet wins/fields |",
        "|---|---:|---:|---:|",
    ]
    for name in ("A", "B", "C"):
        arm = arms[name]
        lines.append(
            f"| {name} | {arm['individual_wins']}/{arm['individual_entries']} | "
            f"{arm['top10_entries']}/{arm['top10_entry_opportunities']} | "
            f"{arm['fleet_wins']}/{arm['fleet_fields']} |"
        )
    lines.extend(["", "## Preregistered combined-treatment gates", ""])
    for name, value in gates.items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines.extend([
        "",
        "## Effective-n=8 block results",
        "",
        "| draw | A individual | C individual | A top-10 | C top-10 | A fleet | C fleet |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for a, c in zip(arms["A"]["draw_blocks"], arms["C"]["draw_blocks"], strict=True):
        lines.append(
            f"| {a['draw']} | {a['individual_win_per_entry']:.1%} | "
            f"{c['individual_win_per_entry']:.1%} | {a['top10_per_entry']:.1%} | "
            f"{c['top10_per_entry']:.1%} | {a['fleet_win_probability']:.1%} | "
            f"{c['fleet_win_probability']:.1%} |"
        )
    lines.extend([
        "",
        "## Runtime",
        "",
        f"Four-GPU guarded projection: {projected:.1f}s versus {prereg['runtime_gate']['deadline_seconds']:.1f}s deadline.",
    ])
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"decision": result["decision"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
