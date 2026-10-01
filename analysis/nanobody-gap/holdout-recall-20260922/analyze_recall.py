#!/usr/bin/env python3
"""Measure full-cohort shortlist recall on four untouched scorer draws."""

from __future__ import annotations

import json
import sqlite3
import statistics
import itertools
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "gpu-capacity-validation-20260922"
DB = HERE.parent / "mutations.sqlite"
TARGET = "P20809"
DEPTHS = (1, 2, 4)
# 30 and 95 reproduce 96/2800 and 300/2800 on the full 888-case scored cohort.
SIZES = (24, 30, 48, 95, 96, 192, 300)
PRODUCTION_POOL = 2800
CURRENT_FINALISTS = 300
PROPOSED_FINALISTS = 96
FLEET_SIZE = 10
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


def mean(values) -> float:
    return float(statistics.fmean(values))


def components(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol in {path}")
    result = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            if not all(metric in metrics for metric in METRICS):
                raise RuntimeError(f"incomplete metric vector for {sequence} in {path}")
            result[sequence] = {
                metric: float(metrics[metric]) for metric in METRICS
            }
    if len(result) != int(payload["n"]):
        raise RuntimeError(f"result count mismatch in {path}")
    return result


def dense_rank_sums(records: list[dict[str, float]]) -> list[int]:
    totals = [0] * len(records)
    for metric, mode in METRICS.items():
        values = sorted(
            {record[metric] for record in records}, reverse=(mode == "max")
        )
        ranks = {value: index + 1 for index, value in enumerate(values)}
        for index, record in enumerate(records):
            totals[index] += ranks[record[metric]]
    return totals


def rank_map(
    observed: dict[str, dict[str, float]], sequences: list[str]
) -> dict[str, int]:
    return dict(zip(
        sequences,
        dense_rank_sums([observed[sequence] for sequence in sequences]),
    ))


def public_fields() -> list[dict]:
    connection = sqlite3.connect(DB)
    rows = connection.execute(
        """
        SELECT o.epoch, o.metrics_json
        FROM observations o JOIN nodes n ON n.id=o.node_id
        WHERE o.source='public_validator_api' AND n.target=?
        ORDER BY o.epoch, n.sequence
        """,
        (TARGET,),
    ).fetchall()
    connection.close()
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for epoch, raw in rows:
        metrics = json.loads(raw)
        if all(metric in metrics for metric in METRICS):
            grouped[int(epoch)].append({
                metric: float(metrics[metric]) for metric in METRICS
            })
    epochs = sorted(
        epoch for epoch, records in grouped.items() if len(records) >= 30
    )[-30:]
    if len(epochs) != 30:
        raise RuntimeError(f"expected 30 public fields, found {len(epochs)}")
    return [{"epoch": epoch, "records": grouped[epoch]} for epoch in epochs]


def fleet_outcome(
    fleet: list[str], observed: dict[str, dict[str, float]], field: dict
) -> bool:
    rows = field["records"] + [observed[sequence] for sequence in fleet]
    scores = dense_rank_sums(rows)
    split = len(field["records"])
    return min(scores[split:]) <= min(scores[:split])


def scenario_wins(
    candidates: list[str],
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
) -> dict[str, set[int]]:
    result = {sequence: set() for sequence in candidates}
    scenario = 0
    for observed in passes:
        for field in fields:
            for sequence in candidates:
                if fleet_outcome([sequence], observed, field):
                    result[sequence].add(scenario)
            scenario += 1
    return result


def correlation_aware_fleet(
    candidates: list[str], wins: dict[str, set[int]], scenarios: int
) -> tuple[list[str], list[dict]]:
    selected: list[str] = []
    covered: set[int] = set()
    trace = []
    for _ in range(min(FLEET_SIZE, len(candidates))):
        choice = max(
            (sequence for sequence in candidates if sequence not in selected),
            key=lambda sequence: (
                len(covered | wins[sequence]), len(wins[sequence]), sequence
            ),
        )
        selected.append(choice)
        covered |= wins[choice]
        trace.append({
            "slot": len(selected),
            "sequence": choice,
            "covered_training_scenarios": len(covered),
            "training_scenarios": scenarios,
        })
    return selected, trace


def evaluate_fleet(
    fleet: list[str],
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
) -> dict:
    outcomes_by_draw = []
    top10_entries_by_draw = []
    individual_wins_by_draw = []
    for observed in passes:
        draw_outcomes = []
        draw_top10_entries = []
        draw_individual_wins = 0
        for field in fields:
            rows = field["records"] + [observed[sequence] for sequence in fleet]
            scores = dense_rank_sums(rows)
            split = len(field["records"])
            fleet_scores = scores[split:]
            draw_outcomes.append(min(fleet_scores) <= min(scores[:split]))
            top10_cutoff = sorted(scores)[min(9, len(scores) - 1)]
            draw_top10_entries.append(
                sum(score <= top10_cutoff for score in fleet_scores)
            )
            draw_individual_wins += sum(
                fleet_outcome([sequence], observed, field) for sequence in fleet
            )
        outcomes_by_draw.append(draw_outcomes)
        top10_entries_by_draw.append(draw_top10_entries)
        individual_wins_by_draw.append(draw_individual_wins)
    by_draw = [sum(outcomes) for outcomes in outcomes_by_draw]
    outcomes = [outcome for draw in outcomes_by_draw for outcome in draw]
    wins = sum(by_draw)
    scenarios = len(passes) * len(fields)
    top10_entries = sum(sum(draw) for draw in top10_entries_by_draw)
    top10_any = sum(
        hits > 0 for draw in top10_entries_by_draw for hits in draw
    )
    individual_wins = sum(individual_wins_by_draw)
    individual_entries = scenarios * len(fleet)
    return {
        "wins": wins,
        "scenarios": scenarios,
        "win_yield": wins / scenarios,
        "wins_by_draw": by_draw,
        "win_yield_by_draw": [wins / len(fields) for wins in by_draw],
        "scenario_outcomes": outcomes,
        "fields_per_draw": len(fields),
        "draw_yield_min": min(by_draw) / len(fields),
        "draw_yield_max": max(by_draw) / len(fields),
        "top10_any_scenarios": top10_any,
        "top10_any_probability": top10_any / scenarios,
        "top10_entries": top10_entries,
        "fleet_entries": scenarios * len(fleet),
        "top10_per_entry": top10_entries / (scenarios * len(fleet)),
        "individual_wins": individual_wins,
        "individual_entries": individual_entries,
        "individual_win_per_entry": individual_wins / individual_entries,
        "individual_wins_by_draw": individual_wins_by_draw,
        "individual_win_per_entry_by_draw": [
            wins / (len(fields) * len(fleet))
            for wins in individual_wins_by_draw
        ],
    }


def paired_fleet_comparison(left: dict, right: dict) -> dict[str, int]:
    pairs = list(zip(left["scenario_outcomes"], right["scenario_outcomes"]))
    return {
        "left_only_wins": sum(a and not b for a, b in pairs),
        "right_only_wins": sum(b and not a for a, b in pairs),
        "both_win": sum(a and b for a, b in pairs),
        "neither_wins": sum(not a and not b for a, b in pairs),
    }


def arm_counts(sequences: list[str], by_sequence: dict[str, dict]) -> dict[str, int]:
    counts = Counter()
    for sequence in sequences:
        arms = by_sequence[sequence]["arms"]
        counts.update(arms or ["control"])
    return dict(sorted(counts.items()))


def cohort_outcomes(
    memberships: dict[str, set[str]],
    heldout_mean: dict[str, float],
    oracle_top10: set[str],
) -> dict[str, dict]:
    outcomes = {}
    for label, members in sorted(memberships.items()):
        eligible_members = sorted(set(members) & set(heldout_mean))
        if not eligible_members:
            continue
        best = min(
            eligible_members,
            key=lambda sequence: (heldout_mean[sequence], sequence),
        )
        outcomes[label] = {
            "eligible_candidates": len(eligible_members),
            "best_sequence": best,
            "best_heldout_mean_rank_sum": heldout_mean[best],
            "consensus_top10_members": len(set(eligible_members) & oracle_top10),
        }
    return outcomes


def main() -> None:
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    by_sequence = {case["sequence"]: case for case in manifest["cases"]}
    all_sequences = [case["sequence"] for case in manifest["cases"]]
    eligible = [
        sequence for sequence in all_sequences
        if not by_sequence[sequence].get("known_public", False)
    ]
    if len(all_sequences) != 888 or len(eligible) != 858:
        raise RuntimeError(
            f"expected 888 total / 858 novel candidates, found "
            f"{len(all_sequences)} / {len(eligible)}"
        )

    training_paths = [
        SOURCE / f"discovery_{index:02d}" / "exact_all.json"
        for index in range(1, 5)
    ]
    holdout_paths = [
        HERE / f"heldout_{index:02d}" / "exact_all.json"
        for index in range(5, 9)
    ]
    missing = [str(path) for path in training_paths + holdout_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing exact observations: {missing}")
    training = [components(path) for path in training_paths]
    holdout = [components(path) for path in holdout_paths]
    if any(set(observed) != set(all_sequences) for observed in training + holdout):
        raise RuntimeError("full-cohort mismatch")

    heldout_ranks = [rank_map(observed, eligible) for observed in holdout]
    heldout_mean = {
        sequence: mean([ranks[sequence] for ranks in heldout_ranks])
        for sequence in eligible
    }
    oracle_order = sorted(eligible, key=lambda sequence: (heldout_mean[sequence], sequence))
    oracle_best = oracle_order[0]
    oracle_top1 = {
        sequence for sequence in eligible
        if heldout_mean[sequence] == heldout_mean[oracle_best]
    }
    oracle_top10 = set(oracle_order[:10])
    arm_memberships: dict[str, set[str]] = defaultdict(set)
    basin_memberships: dict[str, set[str]] = defaultdict(set)
    for sequence in eligible:
        case = by_sequence[sequence]
        for arm in case.get("arms", []) or ["control"]:
            arm_memberships[arm].add(sequence)
        for groups in case.get("arm_groups", {}).values():
            for group in groups:
                basin_memberships[group].add(sequence)
    arm_outcomes = cohort_outcomes(arm_memberships, heldout_mean, oracle_top10)
    basin_outcomes = cohort_outcomes(basin_memberships, heldout_mean, oracle_top10)
    per_draw_top1 = []
    per_draw_top10 = []
    for ranks in heldout_ranks:
        ordered = sorted(eligible, key=lambda sequence: (ranks[sequence], sequence))
        best_value = ranks[ordered[0]]
        per_draw_top1.append({sequence for sequence in ordered if ranks[sequence] == best_value})
        per_draw_top10.append(set(ordered[:10]))
    holdout_stability = []
    for left, right in itertools.combinations(range(len(heldout_ranks)), 2):
        holdout_stability.append({
            "draw_pair": [left + 5, right + 5],
            "candidate_rank_correlation": statistics.correlation(
                [heldout_ranks[left][sequence] for sequence in eligible],
                [heldout_ranks[right][sequence] for sequence in eligible],
            ),
            "top10_overlap": len(per_draw_top10[left] & per_draw_top10[right]),
        })

    fields = public_fields()
    selection_fields = fields[:-8]
    evaluation_fields = fields[-8:]
    rows = []
    depth_details = {}
    for depth in DEPTHS:
        train_ranks = [rank_map(observed, eligible) for observed in training[:depth]]
        train_mean = {
            sequence: mean([ranks[sequence] for ranks in train_ranks])
            for sequence in eligible
        }
        screen_order = sorted(
            eligible, key=lambda sequence: (train_mean[sequence], sequence)
        )
        individual_wins = scenario_wins(
            eligible, training[:depth], selection_fields
        )
        depth_rows = []
        for size in SIZES:
            retained = screen_order[:size]
            retained_set = set(retained)
            best_retained = min(
                retained, key=lambda sequence: (heldout_mean[sequence], sequence)
            )
            screen_champion = retained[0]
            fleet, trace = correlation_aware_fleet(
                retained,
                individual_wins,
                depth * len(selection_fields),
            )
            fleet_yield = evaluate_fleet(
                fleet, holdout, evaluation_fields
            )
            row = {
                "screening_depth": depth,
                "shortlist_size": size,
                "cohort_retention_fraction": size / len(eligible),
                "full_scored_cohort_retention_fraction": size / len(all_sequences),
                "production_pool_equivalent_fraction": size / PRODUCTION_POOL,
                "consensus_top1_recalled": bool(oracle_top1 & retained_set),
                "consensus_top10_recalled": len(oracle_top10 & retained_set),
                "consensus_top10_recall": len(oracle_top10 & retained_set) / 10,
                "per_draw_top1_recalled_draws": sum(
                    bool(winners & retained_set) for winners in per_draw_top1
                ),
                "per_draw_top1_recall": sum(
                    bool(winners & retained_set) for winners in per_draw_top1
                ) / len(holdout),
                "per_draw_top10_recalled": [
                    len(top10 & retained_set) for top10 in per_draw_top10
                ],
                "per_draw_top10_recall_mean": mean([
                    len(top10 & retained_set) / 10 for top10 in per_draw_top10
                ]),
                "best_retained_sequence": best_retained,
                "coverage_oracle_regret": (
                    heldout_mean[best_retained] - heldout_mean[oracle_best]
                ),
                "screen_champion": screen_champion,
                "screen_champion_oracle_regret": (
                    heldout_mean[screen_champion] - heldout_mean[oracle_best]
                ),
                "retained_arm_counts": arm_counts(retained, by_sequence),
                "correlation_aware_fleet": fleet,
                "fleet_selection_trace": trace,
                "heldout_fleet": fleet_yield,
            }
            rows.append(row)
            depth_rows.append(row)
        depth_details[str(depth)] = {
            "screen_order": screen_order,
            "rows": depth_rows,
        }

    by_key = {
        (row["screening_depth"], row["shortlist_size"]): row for row in rows
    }
    comparisons = {}
    for depth in DEPTHS:
        current = by_key[(depth, CURRENT_FINALISTS)]
        proposed = by_key[(depth, PROPOSED_FINALISTS)]
        comparisons[str(depth)] = {
            "current_300": current,
            "proposed_96": proposed,
            "proposed_minus_current": {
                "consensus_top10_recalled": (
                    proposed["consensus_top10_recalled"]
                    - current["consensus_top10_recalled"]
                ),
                "per_draw_top1_recall": (
                    proposed["per_draw_top1_recall"]
                    - current["per_draw_top1_recall"]
                ),
                "coverage_oracle_regret": (
                    proposed["coverage_oracle_regret"]
                    - current["coverage_oracle_regret"]
                ),
                "fleet_win_yield": (
                    proposed["heldout_fleet"]["win_yield"]
                    - current["heldout_fleet"]["win_yield"]
                ),
            },
            "fraction_normalized_current_95": by_key[(depth, 95)],
            "fraction_normalized_proposed_30": by_key[(depth, 30)],
            "paired_absolute_fleet_outcomes": paired_fleet_comparison(
                current["heldout_fleet"], proposed["heldout_fleet"]
            ),
            "paired_fraction_normalized_fleet_outcomes": paired_fleet_comparison(
                by_key[(depth, 95)]["heldout_fleet"],
                by_key[(depth, 30)]["heldout_fleet"],
            ),
        }

    metadata = json.loads((HERE / "run_metadata.json").read_text())
    result = {
        "protocol": "validator-dual-sample-v1",
        "metrics": METRICS,
        "total_scored_candidates": len(all_sequences),
        "eligible_novel_candidates": len(eligible),
        "excluded_known_public_controls": len(all_sequences) - len(eligible),
        "screening_passes": [1, 2, 3, 4],
        "heldout_passes": [5, 6, 7, 8],
        "screening_depths": list(DEPTHS),
        "shortlist_sizes": list(SIZES),
        "public_selection_epochs": [field["epoch"] for field in selection_fields],
        "public_evaluation_epochs": [field["epoch"] for field in evaluation_fields],
        "oracle": {
            "best_sequence": oracle_best,
            "top1_sequences": sorted(oracle_top1),
            "best_heldout_mean_rank_sum": heldout_mean[oracle_best],
            "top10_sequences": oracle_order[:10],
            "top10_arm_counts": arm_counts(oracle_order[:10], by_sequence),
        },
        "holdout_stability": holdout_stability,
        "arm_outcomes": arm_outcomes,
        "basin_outcomes": basin_outcomes,
        "rows": rows,
        "depth_details": depth_details,
        "production_comparison": {
            "production_pool": PRODUCTION_POOL,
            "current_shortlist": CURRENT_FINALISTS,
            "current_retention_fraction": CURRENT_FINALISTS / PRODUCTION_POOL,
            "proposed_shortlist": PROPOSED_FINALISTS,
            "proposed_retention_fraction": PROPOSED_FINALISTS / PRODUCTION_POOL,
            "fraction_normalized_denominator": len(all_sequences),
            "fraction_normalized_current_test_size": 95,
            "fraction_normalized_proposed_test_size": 30,
            "by_depth": comparisons,
        },
        "run_metadata": metadata,
    }
    (HERE / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    required_sizes = SIZES
    lines = [
        "# Full-cohort held-out shortlist recall",
        "",
        "## Verdict",
        "",
        "VERDICT_PLACEHOLDER",
        "",
        "## Experiment",
        "",
        "- Four existing screening draws and four newly generated untouched draws, all over the same 888-sequence cohort.",
        "- Selection excludes 30 known-public controls, leaving 858 eligible novel candidates; controls were still scored in every draw.",
        "- Every comparison uses the complete ten-metric dense-rank sum. Fleet yield preserves intact sequence/pass/public-field scenarios.",
        f"- Fresh-draw candidate-rank correlations span {min(row['candidate_rank_correlation'] for row in holdout_stability):.3f}–{max(row['candidate_rank_correlation'] for row in holdout_stability):.3f}; pairwise top-10 overlap spans {min(row['top10_overlap'] for row in holdout_stability)}–{max(row['top10_overlap'] for row in holdout_stability)}/10.",
        f"- Absolute-K comparisons use 300 and 96 retained. Fraction-normalized comparisons use 95/888 for current 300/2800 ({CURRENT_FINALISTS / PRODUCTION_POOL:.2%}) and 30/888 for proposed 96/2800 ({PROPOSED_FINALISTS / PRODUCTION_POOL:.2%}).",
        "",
        "## Recall and yield on untouched draws 5–8",
        "",
        "| screen draws | kept | top-1 consensus | top-10 consensus | draw-wise top-1 recall | draw-wise top-10 recall | coverage regret | champion regret | fleet wins | individual wins/entry | fleet top-10/entry |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        for size in required_sizes:
            row = by_key[(depth, size)]
            lines.append(
                f"| {depth} | {size} | {'yes' if row['consensus_top1_recalled'] else 'no'} | "
                f"{row['consensus_top10_recalled']}/10 | "
                f"{row['per_draw_top1_recalled_draws']}/4 | "
                f"{row['per_draw_top10_recall_mean']:.1%} | "
                f"{row['coverage_oracle_regret']:.1f} | "
                f"{row['screen_champion_oracle_regret']:.1f} | "
                f"{row['heldout_fleet']['wins']}/{row['heldout_fleet']['scenarios']} | "
                f"{row['heldout_fleet']['individual_wins']}/{row['heldout_fleet']['individual_entries']} | "
                f"{row['heldout_fleet']['top10_entries']}/{row['heldout_fleet']['fleet_entries']} |"
            )
    depth4_current = by_key[(4, 300)]
    depth4_proposed = by_key[(4, 96)]
    depth4_current_fraction = by_key[(4, 95)]
    depth4_proposed_fraction = by_key[(4, 30)]
    depth4_paired = comparisons["4"]["paired_absolute_fleet_outcomes"]
    lines.extend([
        "",
        "## Production decision at four screening draws",
        "",
        f"- Literal 300 retained: top-1 {'recalled' if depth4_current['consensus_top1_recalled'] else 'missed'}, top-10 {depth4_current['consensus_top10_recalled']}/10, coverage regret {depth4_current['coverage_oracle_regret']:.1f}, fleet yield {depth4_current['heldout_fleet']['wins']}/{depth4_current['heldout_fleet']['scenarios']} (per-draw range {depth4_current['heldout_fleet']['draw_yield_min']:.0%}–{depth4_current['heldout_fleet']['draw_yield_max']:.0%}).",
        f"- Literal 96 retained: top-1 {'recalled' if depth4_proposed['consensus_top1_recalled'] else 'missed'}, top-10 {depth4_proposed['consensus_top10_recalled']}/10, coverage regret {depth4_proposed['coverage_oracle_regret']:.1f}, fleet yield {depth4_proposed['heldout_fleet']['wins']}/{depth4_proposed['heldout_fleet']['scenarios']} (per-draw range {depth4_proposed['heldout_fleet']['draw_yield_min']:.0%}–{depth4_proposed['heldout_fleet']['draw_yield_max']:.0%}).",
        f"- Fraction-normalized current (95/888, matching 300/2800): top-10 {depth4_current_fraction['consensus_top10_recalled']}/10, coverage regret {depth4_current_fraction['coverage_oracle_regret']:.1f}, fleet yield {depth4_current_fraction['heldout_fleet']['wins']}/{depth4_current_fraction['heldout_fleet']['scenarios']}.",
        f"- Fraction-normalized proposed (30/888, matching 96/2800): top-10 {depth4_proposed_fraction['consensus_top10_recalled']}/10, coverage regret {depth4_proposed_fraction['coverage_oracle_regret']:.1f}, fleet yield {depth4_proposed_fraction['heldout_fleet']['wins']}/{depth4_proposed_fraction['heldout_fleet']['scenarios']}.",
        f"- Paired fleet scenarios, absolute K: current-only wins {depth4_paired['left_only_wins']}, proposed-only wins {depth4_paired['right_only_wins']}, both {depth4_paired['both_win']}, neither {depth4_paired['neither_wins']}.",
        "- Variation is grouped by the four scorer draws (effective n=4); the 32 draw×field scenarios are repeated measures, not 32 independent observations.",
        f"- Literal 300 per-draw fleet win rates: {', '.join(f'{value:.1%}' for value in depth4_current['heldout_fleet']['win_yield_by_draw'])}; individual win/entry: {', '.join(f'{value:.1%}' for value in depth4_current['heldout_fleet']['individual_win_per_entry_by_draw'])}.",
        f"- Literal 96 per-draw fleet win rates: {', '.join(f'{value:.1%}' for value in depth4_proposed['heldout_fleet']['win_yield_by_draw'])}; individual win/entry: {', '.join(f'{value:.1%}' for value in depth4_proposed['heldout_fleet']['individual_win_per_entry_by_draw'])}.",
    ])
    lines.extend([
        "",
        "### Effective-n=4 scorer-draw variation at four screening draws",
        "",
        "| kept | fleet win rate by scorer draw 5/6/7/8 | individual win/entry by scorer draw 5/6/7/8 |",
        "|---:|---|---|",
    ])
    for size in (30, 95, 96, 192, 300):
        fleet = by_key[(4, size)]["heldout_fleet"]
        lines.append(
            f"| {size} | "
            f"{' / '.join(f'{value:.1%}' for value in fleet['win_yield_by_draw'])} | "
            f"{' / '.join(f'{value:.1%}' for value in fleet['individual_win_per_entry_by_draw'])} |"
        )
    lines.extend([
        "",
        "## Arm outcomes on untouched draws",
        "",
        "| arm | eligible candidates | best mean rank sum | consensus top-10 members |",
        "|---|---:|---:|---:|",
    ])
    for arm, outcome in sorted(arm_outcomes.items()):
        lines.append(
            f"| {arm} | {outcome['eligible_candidates']} | "
            f"{outcome['best_heldout_mean_rank_sum']:.1f} | "
            f"{outcome['consensus_top10_members']} |"
        )
    lines.extend([
        "",
        "## Basin outcomes on untouched draws",
        "",
        "| basin | eligible candidates | best mean rank sum | consensus top-10 members |",
        "|---|---:|---:|---:|",
    ])
    for basin, outcome in sorted(
        basin_outcomes.items(),
        key=lambda item: (item[1]["best_heldout_mean_rank_sum"], item[0]),
    ):
        lines.append(
            f"| {basin} | {outcome['eligible_candidates']} | "
            f"{outcome['best_heldout_mean_rank_sum']:.1f} | "
            f"{outcome['consensus_top10_members']} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `analysis.json`: every depth/size result, retained order, selected fleets, exact scenario counts, and oracle identities.",
        "- `run_metadata.json`: scorer/input hashes and exact timing metadata.",
        "- `heldout_05`–`heldout_08`: four complete fresh full-cohort observations.",
    ])
    verdict = (
        f"The added Hamming-2 proposal capacity is useful: closure candidates supply "
        f"{arm_outcomes['ours_hamming2']['consensus_top10_members'] + arm_outcomes['rival_hamming2']['consensus_top10_members']}/10 "
        f"held-out consensus top-10 sequences, versus "
        f"{arm_outcomes['current_production']['consensus_top10_members']}/10 from the current arm. "
        f"The proposed 96/2800 retention fraction is not validated: its K=30 analogue "
        f"recalls {depth4_proposed_fraction['consensus_top10_recalled']}/10, versus "
        f"{depth4_current_fraction['consensus_top10_recalled']}/10 for current-fraction K=95. "
        f"Although K=30 has {depth4_proposed_fraction['heldout_fleet']['wins']}/32 fleet wins "
        f"versus {depth4_current_fraction['heldout_fleet']['wins']}/32, that reversal is confined "
        f"to scorer-draw variation (effective n=4) and is not stable evidence for pruning. "
        f"Keep the wider retention fraction and spend extra GPU on repeated complete-metric "
        f"confirmation: K=95 top-10 recall rises from "
        f"{by_key[(1, 95)]['consensus_top10_recalled']}/10 at one draw to "
        f"{by_key[(4, 95)]['consensus_top10_recalled']}/10 at four."
    )
    lines[4] = verdict
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "depth4_300": {
            "top1": depth4_current["consensus_top1_recalled"],
            "top10": depth4_current["consensus_top10_recalled"],
            "coverage_regret": depth4_current["coverage_oracle_regret"],
            "fleet": depth4_current["heldout_fleet"],
        },
        "depth4_96": {
            "top1": depth4_proposed["consensus_top1_recalled"],
            "top10": depth4_proposed["consensus_top10_recalled"],
            "coverage_regret": depth4_proposed["coverage_oracle_regret"],
            "fleet": depth4_proposed["heldout_fleet"],
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
