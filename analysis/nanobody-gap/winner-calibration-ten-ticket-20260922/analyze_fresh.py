#!/usr/bin/env python3
"""Evaluate the two frozen tests after all new draws 17-32 exist."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment import (
    METRICS,
    NONINFERIORITY_METRICS,
    favorable_delta,
    joint_field_outcomes,
    load_draw,
    quantile,
    sha256,
    write_json,
)


HERE = Path(__file__).resolve().parent
FRESH_DRAWS = tuple(range(17, 33))


def verify_freeze(prereg: dict[str, Any]) -> None:
    for name, expected in prereg["frozen_hashes"]["code"].items():
        if sha256(HERE / name) != expected:
            raise RuntimeError(f"frozen code changed: {name}")
    artifact_keys = {
        "sequence_manifest": "sequences.json",
        "calibration_cohort": "calibration_cohort.json",
        "historical_sampling": "historical_sampling.json",
        "rival_panels": "rival_panels.json",
        "fleets": "fleets.json",
    }
    for key, name in artifact_keys.items():
        if sha256(HERE / name) != prereg["frozen_hashes"][key]:
            raise RuntimeError(f"frozen artifact changed: {name}")


def weighted_mean(rows: list[dict[str, Any]], value_key: str) -> float:
    denominator = sum(float(row["weight"]) for row in rows)
    return sum(float(row["weight"]) * float(row[value_key]) for row in rows) / denominator


def weighted_auc(rows: list[dict[str, Any]]) -> float:
    positives = [row for row in rows if row["actual"]]
    negatives = [row for row in rows if not row["actual"]]
    numerator = 0.0
    denominator = 0.0
    for positive in positives:
        for negative in negatives:
            pair_weight = float(positive["weight"]) * float(negative["weight"])
            denominator += pair_weight
            if positive["predicted_win_rate"] > negative["predicted_win_rate"]:
                numerator += pair_weight
            elif positive["predicted_win_rate"] == negative["predicted_win_rate"]:
                numerator += 0.5 * pair_weight
    return numerator / denominator


def bootstrap_calibration(
    epoch_rows: list[dict[str, Any]], replicates: int
) -> dict[str, Any]:
    seed = int.from_bytes(
        hashlib.sha256(b"winner-calibration-bootstrap-v1").digest()[:8], "big"
    )
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for epoch in epoch_rows:
        groups[(epoch["stratum"], epoch["cohort_role"])].append(epoch)
    rate_samples = []
    separation_samples = []
    for _ in range(replicates):
        draw_sample = [rng.randrange(len(FRESH_DRAWS)) for _ in FRESH_DRAWS]
        sampled_epochs = []
        for group in groups.values():
            sampled_epochs.extend(rng.choice(group) for _ in range(len(group)))
        numerator = 0.0
        denominator = 0.0
        winner_numerator = 0.0
        winner_denominator = 0.0
        control_numerator = 0.0
        control_denominator = 0.0
        for epoch in sampled_epochs:
            weight = float(epoch["weight"])
            for entry in epoch["entries"]:
                predicted = sum(entry["draw_wins"][index] for index in draw_sample) / len(draw_sample)
                numerator += weight * predicted
                denominator += weight
                if entry["actual"]:
                    winner_numerator += weight * predicted
                    winner_denominator += weight
                else:
                    control_numerator += weight * predicted
                    control_denominator += weight
        rate_samples.append(numerator / denominator)
        separation_samples.append(
            winner_numerator / winner_denominator
            - control_numerator / control_denominator
        )
    return {
        "replicates": replicates,
        "seed": seed,
        "predicted_rate_95_interval": [
            quantile(rate_samples, 0.025),
            quantile(rate_samples, 0.975),
        ],
        "winner_minus_control_95_interval": [
            quantile(separation_samples, 0.025),
            quantile(separation_samples, 0.975),
        ],
    }


def evaluate_calibration(
    cohort: dict[str, Any],
    draws: list[dict[str, dict[str, float]]],
    target_rate: float,
    replicates: int,
) -> dict[str, Any]:
    epochs = []
    flat = []
    recovered = []
    for epoch in cohort["epochs"]:
        sequences = [entry["sequence"] for entry in epoch["entries"]]
        rivals = [entry["metrics"] for entry in epoch["rivals"]]
        per_entry = [{"wins": [], "top10": [], "rank_sums": []} for _ in sequences]
        for draw in draws:
            outcome = joint_field_outcomes([draw[sequence] for sequence in sequences], rivals)
            for index in range(len(sequences)):
                per_entry[index]["wins"].append(int(outcome["wins"][index]))
                per_entry[index]["top10"].append(int(outcome["top10"][index]))
                per_entry[index]["rank_sums"].append(outcome["arm_rank_sums"][index])
        entries = []
        for original, observations in zip(epoch["entries"], per_entry, strict=True):
            row = {
                "entry_id": original["entry_id"],
                "sequence": original["sequence"],
                "actual": bool(original["actual_field_beater"]),
                "predicted_win_rate": statistics.mean(observations["wins"]),
                "predicted_top10_rate": statistics.mean(observations["top10"]),
                "mean_rank_sum": statistics.mean(observations["rank_sums"]),
                "draw_wins": observations["wins"],
                "draw_top10": observations["top10"],
                "weight": float(epoch["epoch_weight"]),
                "epoch": epoch["epoch"],
                "stratum": epoch["stratum"],
                "cohort_role": epoch["cohort_role"],
            }
            entries.append(row)
            flat.append(row)
        ordered = sorted(
            entries,
            key=lambda row: (
                -row["predicted_win_rate"], row["mean_rank_sum"], row["entry_id"]
            ),
        )
        if any(row["actual"] for row in entries):
            success = bool(ordered[0]["actual"])
            recovered.append(success)
        epochs.append({
            "epoch": epoch["epoch"],
            "stratum": epoch["stratum"],
            "cohort_role": epoch["cohort_role"],
            "weight": float(epoch["epoch_weight"]),
            "winner_recovered_first": success if any(row["actual"] for row in entries) else None,
            "entries": entries,
        })
    predicted_rate = weighted_mean(flat, "predicted_win_rate")
    actual_rate = weighted_mean([
        {**row, "actual_float": float(row["actual"])} for row in flat
    ], "actual_float")
    winners = [row for row in flat if row["actual"]]
    controls = [row for row in flat if not row["actual"]]
    separation = weighted_mean(winners, "predicted_win_rate") - weighted_mean(
        controls, "predicted_win_rate"
    )
    brier_rows = [
        {**row, "brier": (row["predicted_win_rate"] - float(row["actual"])) ** 2}
        for row in flat
    ]
    bootstrap = bootstrap_calibration(epochs, replicates)
    interval = bootstrap["predicted_rate_95_interval"]
    separation_interval = bootstrap["winner_minus_control_95_interval"]
    gates = {
        "winner_recovery_20_of_27": sum(recovered) >= 20 and len(recovered) == 27,
        "target_rate_inside_bootstrap_interval": interval[0] <= target_rate <= interval[1],
        "winner_control_separation_lower_bound_positive": separation_interval[0] > 0,
    }
    return {
        "entry_records": len(flat),
        "winning_epoch_groups": len(recovered),
        "winner_recovered_groups": sum(recovered),
        "weighted_actual_rate_audit": actual_rate,
        "weighted_predicted_win_rate": predicted_rate,
        "target_full_history_rate": target_rate,
        "weighted_actual_winner_mean_prediction": weighted_mean(winners, "predicted_win_rate"),
        "weighted_control_mean_prediction": weighted_mean(controls, "predicted_win_rate"),
        "winner_minus_control": separation,
        "weighted_brier": weighted_mean(brier_rows, "brier"),
        "weighted_auc": weighted_auc(flat),
        "weighted_predicted_top10_rate": weighted_mean(flat, "predicted_top10_rate"),
        "bootstrap": bootstrap,
        "gates": gates,
        "pass": all(gates.values()),
        "epoch_results": epochs,
    }


def evaluate_arm(
    fleet: list[str],
    draws: list[dict[str, dict[str, float]]],
    panels: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    blocks = []
    panel_totals = {
        name: {"wins": 0, "top10": 0, "opportunities": 0, "fleet_wins": 0, "fields": 0}
        for name in panels
    }
    for draw_number, draw in zip(FRESH_DRAWS, draws, strict=True):
        block = {
            "draw": draw_number,
            "wins": 0,
            "top10": 0,
            "opportunities": 0,
            "fleet_wins": 0,
            "fields": 0,
            "panels": {},
            "fleet_median_metrics": {
                metric: statistics.median(draw[sequence][metric] for sequence in fleet)
                for metric in NONINFERIORITY_METRICS
            },
        }
        for panel_name, fields in panels.items():
            result = {"wins": 0, "top10": 0, "opportunities": 0, "fleet_wins": 0, "fields": 0}
            for field in fields:
                rivals = [entry["metrics"] for entry in field["rivals"]]
                outcome = joint_field_outcomes([draw[sequence] for sequence in fleet], rivals)
                result["wins"] += sum(outcome["wins"])
                result["top10"] += sum(outcome["top10"])
                result["opportunities"] += len(fleet)
                result["fleet_wins"] += int(any(outcome["wins"]))
                result["fields"] += 1
            block["panels"][panel_name] = result
            for key in ("wins", "top10", "opportunities", "fleet_wins", "fields"):
                block[key] += result[key]
                panel_totals[panel_name][key] += result[key]
        block["individual_win_per_entry"] = block["wins"] / block["opportunities"]
        block["top10_per_entry"] = block["top10"] / block["opportunities"]
        block["fleet_win_probability"] = block["fleet_wins"] / block["fields"]
        blocks.append(block)
    totals = {
        key: sum(block[key] for block in blocks)
        for key in ("wins", "top10", "opportunities", "fleet_wins", "fields")
    }
    totals["individual_win_per_entry"] = totals["wins"] / totals["opportunities"]
    totals["top10_per_entry"] = totals["top10"] / totals["opportunities"]
    totals["fleet_win_probability"] = totals["fleet_wins"] / totals["fields"]
    return {"fleet": fleet, "draw_blocks": blocks, "panel_totals": panel_totals, "totals": totals}


def domination_gate(deltas: list[int]) -> dict[str, Any]:
    positive = [value for value in deltas if value > 0]
    top_two_share = sum(sorted(positive, reverse=True)[:2]) / sum(positive) if positive else 1.0
    return {
        "positive_draw_blocks": len(positive),
        "largest_two_positive_share": top_two_share,
        "pass": len(positive) >= 3 and top_two_share <= 0.50,
    }


def runtime_gate(prereg: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    durations = [
        float(row["duration_seconds"])
        for row in metadata["draw_timings"] if row.get("complete")
    ]
    if len(durations) != len(FRESH_DRAWS):
        raise RuntimeError("runtime gate requires all 16 draw timings")
    frozen = prereg["runtime"]
    fixed = float(frozen["fixed_load_seconds"])
    implied = (max(durations) - fixed) * 8 / prereg["candidate_manifest_count"]
    effective = max(float(frozen["frozen_gpu_seconds_per_observation"]), implied)
    projections = {}
    for name in ("production_treatment", "production_current"):
        workload = frozen[name]
        unguarded = (
            workload["model_loads"] * fixed
            + effective * workload["candidate_observations"] / frozen["production_gpu_count"]
            + workload["selector_cpu_seconds"]
        )
        projections[name] = {
            "unguarded_seconds": unguarded,
            "guarded_seconds": unguarded * frozen["projection_guard_multiplier"],
        }
    treatment = projections["production_treatment"]["guarded_seconds"]
    current = projections["production_current"]["guarded_seconds"]
    pass_gate = treatment <= frozen["production_deadline_seconds"] and treatment <= current
    return {
        "draw_durations_seconds": durations,
        "slowest_draw_seconds": max(durations),
        "effective_gpu_seconds_per_observation": effective,
        "projections": projections,
        "deadline_seconds": frozen["production_deadline_seconds"],
        "research_scoring_seconds_sum": sum(durations),
        "research_scoring_cost": sum(durations) / 3600 * frozen["hourly_cost"],
        "pass": pass_gate,
    }


def compare_ab(
    a: dict[str, Any],
    b: dict[str, Any],
    margins: dict[str, float],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    blocks = []
    for a_block, b_block in zip(a["draw_blocks"], b["draw_blocks"], strict=True):
        blocks.append({
            "draw": a_block["draw"],
            "win_delta": b_block["wins"] - a_block["wins"],
            "top10_delta": b_block["top10"] - a_block["top10"],
            "fleet_win_delta": b_block["fleet_wins"] - a_block["fleet_wins"],
            "hard_metric_favorable_deltas": {
                metric: favorable_delta(
                    b_block["fleet_median_metrics"][metric],
                    a_block["fleet_median_metrics"][metric],
                    direction,
                )
                for metric, direction in NONINFERIORITY_METRICS.items()
            },
        })
    win_deltas = [row["win_delta"] for row in blocks]
    top10_deltas = [row["top10_delta"] for row in blocks]
    lodo_win = [sum(win_deltas) - value for value in win_deltas]
    lodo_top10 = [sum(top10_deltas) - value for value in top10_deltas]
    panel_deltas = {}
    for panel in a["panel_totals"]:
        panel_deltas[panel] = {
            "win_delta": b["panel_totals"][panel]["wins"] - a["panel_totals"][panel]["wins"],
            "top10_delta": b["panel_totals"][panel]["top10"] - a["panel_totals"][panel]["top10"],
        }
    hard = {}
    for metric in NONINFERIORITY_METRICS:
        values = [row["hard_metric_favorable_deltas"][metric] for row in blocks]
        hard[metric] = {
            "median_favorable_delta": statistics.median(values),
            "worst_favorable_delta": min(values),
            "development_mad_margin": float(margins[metric]),
            "pass": statistics.median(values) >= 0 and min(values) >= -float(margins[metric]),
        }
    win_domination = domination_gate(win_deltas)
    top10_domination = domination_gate(top10_deltas)
    gates = {
        "wins_favorable_in_at_least_12_of_16_draws": sum(value > 0 for value in win_deltas) >= 12,
        "top10_favorable_in_at_least_12_of_16_draws": sum(value > 0 for value in top10_deltas) >= 12,
        "every_lodo_win_aggregate_favorable": all(value > 0 for value in lodo_win),
        "every_lodo_top10_aggregate_favorable": all(value > 0 for value in lodo_top10),
        "both_panels_win_positive": all(row["win_delta"] > 0 for row in panel_deltas.values()),
        "both_panels_top10_positive": all(row["top10_delta"] > 0 for row in panel_deltas.values()),
        "wins_not_two_draw_dominated": win_domination["pass"],
        "top10_not_two_draw_dominated": top10_domination["pass"],
        "all_interface_liability_metrics_noninferior": all(row["pass"] for row in hard.values()),
        "runtime": runtime["pass"],
    }
    return {
        "draw_blocks": blocks,
        "win_favorable_draws": sum(value > 0 for value in win_deltas),
        "top10_favorable_draws": sum(value > 0 for value in top10_deltas),
        "lodo_win_deltas": lodo_win,
        "lodo_top10_deltas": lodo_top10,
        "panel_deltas": panel_deltas,
        "win_domination": win_domination,
        "top10_domination": top10_domination,
        "hard_metric_noninferiority": hard,
        "gates": gates,
        "pass": all(gates.values()),
        "decision": "PASS" if all(gates.values()) else "NO-DEPLOY",
    }


def main() -> None:
    prereg_path = HERE / "preregistration.json"
    expected_prereg = (HERE / "PREREGISTRATION.sha256").read_text().split()[0]
    if sha256(prereg_path) != expected_prereg:
        raise RuntimeError("preregistration hash mismatch")
    prereg = json.loads(prereg_path.read_text())
    verify_freeze(prereg)
    expected_count = int(prereg["candidate_manifest_count"])
    draws = [
        load_draw(HERE / f"fresh_{draw:02d}" / "exact_all.json", expected_count)
        for draw in FRESH_DRAWS
    ]
    sequences = json.loads((HERE / "sequences.json").read_text())["sequences"]
    if any(set(draw) != set(sequences) for draw in draws):
        raise RuntimeError("fresh draw sequence manifest mismatch")
    cohort = json.loads((HERE / "calibration_cohort.json").read_text())
    panels = json.loads((HERE / "rival_panels.json").read_text())
    fleets = json.loads((HERE / "fleets.json").read_text())
    calibration = evaluate_calibration(
        cohort,
        draws,
        prereg["test1_historical_calibration"]["full_history_rate_value"],
        prereg["test1_historical_calibration"]["bootstrap"]["replicates"],
    )
    a = evaluate_arm(fleets["A"]["fleet"], draws, panels)
    b = evaluate_arm(fleets["B"]["fleet"], draws, panels)
    runtime = runtime_gate(prereg, json.loads((HERE / "run_metadata.json").read_text()))
    comparison = compare_ab(
        a, b, fleets["development_noninferiority_mad_margins"], runtime
    )
    result = {
        "experiment": prereg["experiment"],
        "preregistration_sha256": expected_prereg,
        "test1_historical_calibration": calibration,
        "test2_literal_ten_ticket": {
            "A": a,
            "B": b,
            "comparison": comparison,
            "decision": comparison["decision"],
        },
        "runtime": runtime,
        "overall_decision": (
            "PASS" if calibration["pass"] and comparison["pass"] else "NO-DEPLOY"
        ),
    }
    write_json(HERE / "analysis.json", result)
    print(json.dumps({
        "overall_decision": result["overall_decision"],
        "calibration_pass": calibration["pass"],
        "winner_recovered": calibration["winner_recovered_groups"],
        "ab_decision": comparison["decision"],
        "ab_gates": comparison["gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
