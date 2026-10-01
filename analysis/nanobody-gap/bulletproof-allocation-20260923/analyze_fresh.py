#!/usr/bin/env python3
"""Analyze the untouched 32-draw prospective multi-cohort proof."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from select_fleets import METRICS, TARGET, load_draw, outcome_table  # noqa: E402


ARMS = (
    "A_live_selector_live_allocation",
    "B_field_selector_live_allocation",
    "C_field_selector_delayed_6_4",
    "D_field_selector_uniform_8",
)
PRIMARY_LEFT = "C_field_selector_delayed_6_4"
PRIMARY_RIGHT = "A_live_selector_live_allocation"
PROTECTED = {
    "design_iiptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_two_way(delta: np.ndarray, *, trials: int = 100_000) -> dict[str, Any]:
    rng = np.random.default_rng(20260923)
    pieces = []
    for _ in range(10):
        cohorts = rng.integers(0, delta.shape[0], size=(trials // 10, delta.shape[0]))
        draws = rng.integers(0, delta.shape[1], size=(trials // 10, delta.shape[1]))
        sampled = delta[cohorts[:, :, None], draws[:, None, :]].mean(axis=(1, 2))
        pieces.append(sampled)
    values = np.concatenate(pieces)
    low, median, high = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "trials": trials, "seed": 20260923,
        "ci95": [float(low), float(high)], "median": float(median),
        "probability_nonpositive": float(np.mean(values <= 0.0)),
    }


def concentration(delta: np.ndarray, axis: int) -> float:
    values = delta.mean(axis=axis)
    positive = np.maximum(values, 0.0)
    return float(positive.max() / positive.sum()) if positive.sum() else 1.0


def leave_one_out_positive(delta: np.ndarray, axis: int) -> list[bool]:
    return [
        bool(np.delete(delta, index, axis=axis).mean() > 0.0)
        for index in range(delta.shape[axis])
    ]


def main() -> None:
    for line in (HERE / "FREEZE_MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        if sha256(HERE / name) != expected:
            raise RuntimeError(f"frozen input changed: {name}")
    fleets = json.loads((HERE / "selected_fleets.json").read_text())
    fresh_sequences = json.loads((HERE / "fresh_sequences.json").read_text())["sequences"]
    draws = [load_draw(HERE / f"fresh_{index:02d}/exact_all.json") for index in range(1, 33)]
    if any(set(draw) != set(fresh_sequences) for draw in draws):
        raise RuntimeError("fresh draw sequence mismatch")
    fields = json.loads((HERE / "evaluation_fields.json").read_text())
    table = outcome_table(fresh_sequences, draws, fields)
    sequence_index = {sequence: index for index, sequence in enumerate(fresh_sequences)}
    epochs = sorted(fleets["cohorts"], key=int)
    rates: dict[str, dict[str, np.ndarray]] = {
        arm: {
            "wins": np.zeros((len(epochs), len(draws))),
            "top10": np.zeros((len(epochs), len(draws))),
            "fleet_wins": np.zeros((len(epochs), len(draws))),
        }
        for arm in ARMS
    }
    metric_values = {
        arm: {metric: np.zeros((len(epochs), len(draws))) for metric in PROTECTED}
        for arm in ARMS
    }
    for cohort_index, epoch in enumerate(epochs):
        for arm in ARMS:
            fleet = fleets["cohorts"][epoch][arm]["fleet"]
            rows = [sequence_index[sequence] for sequence in fleet]
            for draw_index, draw in enumerate(draws):
                won = table["wins"][draw_index, rows, :]
                placed = table["top10"][draw_index, rows, :]
                rates[arm]["wins"][cohort_index, draw_index] = won.mean()
                rates[arm]["top10"][cohort_index, draw_index] = placed.mean()
                rates[arm]["fleet_wins"][cohort_index, draw_index] = won.any(axis=0).mean()
                for metric in PROTECTED:
                    metric_values[arm][metric][cohort_index, draw_index] = float(
                        np.median([draw[sequence][metric] for sequence in fleet])
                    )

    summaries = {}
    for arm in ARMS:
        summaries[arm] = {
            name: {
                "rate": float(values.mean()),
                "by_cohort": values.mean(axis=1).tolist(),
                "by_draw": values.mean(axis=0).tolist(),
            }
            for name, values in rates[arm].items()
        }

    comparisons = {}
    pairs = {
        "combined_C_vs_A": ("C_field_selector_delayed_6_4", "A_live_selector_live_allocation"),
        "selector_B_vs_A": ("B_field_selector_live_allocation", "A_live_selector_live_allocation"),
        "allocation_C_vs_B": ("C_field_selector_delayed_6_4", "B_field_selector_live_allocation"),
        "uniform_D_vs_C": ("D_field_selector_uniform_8", "C_field_selector_delayed_6_4"),
    }
    for label, (left, right) in pairs.items():
        comparison: dict[str, Any] = {"left": left, "right": right}
        for outcome_name in ("wins", "top10", "fleet_wins"):
            delta = rates[left][outcome_name] - rates[right][outcome_name]
            comparison[outcome_name] = {
                "delta": float(delta.mean()),
                "bootstrap": bootstrap_two_way(delta),
                "favorable_cohorts": int((delta.mean(axis=1) > 0).sum()),
                "favorable_draws": int((delta.mean(axis=0) > 0).sum()),
                "tied_draws": int((delta.mean(axis=0) == 0).sum()),
                "leave_one_cohort_out_positive": leave_one_out_positive(delta, 0),
                "leave_one_draw_out_positive": leave_one_out_positive(delta, 1),
                "largest_positive_cohort_share": concentration(delta, 1),
                "largest_positive_draw_share": concentration(delta, 0),
            }
        protected = {}
        for metric, direction in PROTECTED.items():
            raw = metric_values[left][metric] - metric_values[right][metric]
            favorable = raw if direction == "max" else -raw
            protected[metric] = {
                "median_favorable_delta": float(np.median(favorable)),
                "mean_favorable_delta": float(np.mean(favorable)),
                "nonregression": bool(np.median(favorable) >= 0.0),
            }
        comparison["protected_metrics"] = protected
        comparisons[label] = comparison

    primary = comparisons["combined_C_vs_A"]
    control = summaries[PRIMARY_RIGHT]
    gates = {
        "win_delta_positive": primary["wins"]["delta"] > 0,
        "top10_delta_positive": primary["top10"]["delta"] > 0,
        "win_bootstrap_lower_positive": primary["wins"]["bootstrap"]["ci95"][0] > 0,
        "top10_bootstrap_lower_positive": primary["top10"]["bootstrap"]["ci95"][0] > 0,
        "win_favorable_at_least_5_of_6_cohorts": primary["wins"]["favorable_cohorts"] >= 5,
        "top10_favorable_at_least_5_of_6_cohorts": primary["top10"]["favorable_cohorts"] >= 5,
        "all_leave_one_cohort_out_positive": all(primary["wins"]["leave_one_cohort_out_positive"]) and all(primary["top10"]["leave_one_cohort_out_positive"]),
        "all_leave_one_draw_out_positive": all(primary["wins"]["leave_one_draw_out_positive"]) and all(primary["top10"]["leave_one_draw_out_positive"]),
        "win_not_concentrated_by_draw": primary["wins"]["largest_positive_draw_share"] <= 0.25,
        "win_not_concentrated_by_cohort": primary["wins"]["largest_positive_cohort_share"] <= 0.35,
        "all_protected_metrics_nonregressing": all(item["nonregression"] for item in primary["protected_metrics"].values()),
        "control_top10_calibrated_to_live": abs(control["top10"]["rate"] - 0.141) <= 0.03,
        "control_win_rate_plausible": 0.012 <= control["wins"]["rate"] <= 0.04,
    }
    passed = all(gates.values())
    result = {
        "experiment": "prospective-multicohort-selector-allocation-proof-v1",
        "effective_scorer_draws": 32,
        "cohorts": [int(epoch) for epoch in epochs],
        "evaluation_fields": [field["epoch"] for field in fields],
        "nominal_entries_per_arm": len(epochs) * len(draws) * len(fields) * 10,
        "summaries": summaries,
        "comparisons": comparisons,
        "decision_gates": gates,
        "passed": passed,
        "decision": "eligible_for_shadow_deployment" if passed else "NO_DEPLOY",
        "fleet_freeze_sha256": sha256(HERE / "selected_fleets.json"),
    }
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    c = summaries[PRIMARY_LEFT]
    a = summaries[PRIMARY_RIGHT]
    lines = [
        "# Prospective multi-cohort selector/allocation proof", "",
        f"Decision: **{result['decision']}**", "",
        "Production was not changed.", "",
        "| arm | wins / entry | top-10 / entry | fleet-field win rate |", "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = summaries[arm]
        lines.append(f"| {arm} | {row['wins']['rate']:.3%} | {row['top10']['rate']:.3%} | {row['fleet_wins']['rate']:.3%} |")
    lines += ["", "## Primary C versus A", "",
        f"- wins/entry delta: {primary['wins']['delta']:+.3%}; 95% two-way cluster bootstrap {primary['wins']['bootstrap']['ci95'][0]:+.3%} to {primary['wins']['bootstrap']['ci95'][1]:+.3%}",
        f"- top-10/entry delta: {primary['top10']['delta']:+.3%}; 95% two-way cluster bootstrap {primary['top10']['bootstrap']['ci95'][0]:+.3%} to {primary['top10']['bootstrap']['ci95'][1]:+.3%}",
        f"- favorable cohorts: wins {primary['wins']['favorable_cohorts']}/6; top-10 {primary['top10']['favorable_cohorts']}/6", "",
        "## Gates", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in gates.items())
    lines += ["", "The 30,720 nominal entry/field observations are correlated. The inferential unit is the 6-by-32 cohort/draw grid, and the bootstrap resamples both axes.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print(f"DECISION={result['decision']} win_delta={primary['wins']['delta']:+.4f} top10_delta={primary['top10']['delta']:+.4f}")


if __name__ == "__main__":
    main()
