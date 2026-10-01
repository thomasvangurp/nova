#!/usr/bin/env python3
"""Analyze repeated exact observations and select a correlation-aware fleet."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DB = HERE.parent / "mutations.sqlite"
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


def components(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol in {path}")
    result = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            if set(metrics) != set(METRICS):
                raise RuntimeError(f"wrong metric set for {sequence} in {path}")
            result[sequence] = {key: float(metrics[key]) for key in METRICS}
    if len(result) != int(payload["n"]):
        raise RuntimeError(f"result count mismatch in {path}")
    return result


def dense_rank_sums(records: list[dict[str, float]]) -> list[int]:
    totals = [0] * len(records)
    for metric, mode in METRICS.items():
        ordered = sorted({row[metric] for row in records}, reverse=(mode == "max"))
        ranks = {value: index + 1 for index, value in enumerate(ordered)}
        for index, row in enumerate(records):
            totals[index] += ranks[row[metric]]
    return totals


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def identity(left: str, right: str) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right)) / len(left)


def public_fields() -> tuple[list[dict], set[str]]:
    connection = sqlite3.connect(DB)
    rows = connection.execute(
        """
        SELECT o.epoch, n.sequence, o.metrics_json
        FROM observations o JOIN nodes n ON n.id=o.node_id
        WHERE o.source='public_validator_api' AND n.target=?
        ORDER BY o.epoch, n.sequence
        """,
        (TARGET,),
    ).fetchall()
    connection.close()
    all_public = {sequence for _, sequence, _ in rows}
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for epoch, _sequence, raw in rows:
        data = json.loads(raw)
        if all(key in data for key in METRICS):
            grouped[int(epoch)].append({key: float(data[key]) for key in METRICS})
    epochs = sorted(epoch for epoch, field in grouped.items() if len(field) >= 30)[-30:]
    return [{"epoch": epoch, "records": grouped[epoch]} for epoch in epochs], all_public


def individual_calibration(
    passes: list[dict[str, dict[str, float]]], fields: list[dict], candidates: list[str]
) -> dict[str, dict]:
    result = {}
    denominator = len(passes) * len(fields)
    for sequence in candidates:
        wins = 0
        score_values = []
        margins = []
        for pass_metrics in passes:
            candidate = pass_metrics[sequence]
            for field in fields:
                rows = field["records"] + [candidate]
                rank_sums = dense_rank_sums(rows)
                candidate_score = rank_sums[-1]
                public_best = min(rank_sums[:-1])
                wins += candidate_score <= public_best
                score_values.append(candidate_score)
                margins.append(public_best - candidate_score)
        result[sequence] = {
            "win_probability": wins / denominator,
            "wins": wins,
            "scenarios": denominator,
            "mean_rank_sum": mean(score_values),
            "mean_winning_margin": mean(margins),
        }
    return result


def fleet_outcome(
    selected: list[str], pass_metrics: dict[str, dict[str, float]], field: dict
) -> bool:
    rows = field["records"] + [pass_metrics[sequence] for sequence in selected]
    rank_sums = dense_rank_sums(rows)
    split = len(field["records"])
    return min(rank_sums[split:]) <= min(rank_sums[:split])


def evaluate_fleet(
    selected: list[str],
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
) -> dict:
    """Evaluate a fixed fleet without reusing its selection observations."""
    denominator = len(passes) * len(fields)
    wins = sum(
        fleet_outcome(selected, pass_metrics, field)
        for pass_metrics in passes
        for field in fields
    )
    return {
        "fleet_wins": wins,
        "scenarios": denominator,
        "fleet_win_probability": wins / denominator,
    }


def greedy_fleet(
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
    calibration: dict[str, dict],
    candidates: list[str],
    size: int = 10,
) -> list[dict]:
    shortlist = sorted(
        candidates,
        key=lambda seq: (
            calibration[seq]["win_probability"],
            calibration[seq]["mean_winning_margin"],
        ),
        reverse=True,
    )[:128]
    selected: list[str] = []
    trace = []
    denominator = len(passes) * len(fields)
    for _ in range(size):
        best = None
        for sequence in shortlist:
            if sequence in selected:
                continue
            if any(identity(sequence, previous) >= 0.95 for previous in selected):
                continue
            proposal = selected + [sequence]
            wins = sum(
                fleet_outcome(proposal, pass_metrics, field)
                for pass_metrics in passes
                for field in fields
            )
            key = (wins, calibration[sequence]["win_probability"])
            if best is None or key > best[0]:
                best = (key, sequence, wins)
        if best is None:
            break
        selected.append(best[1])
        trace.append({
            "slot": len(selected),
            "sequence": best[1],
            "individual_win_probability": calibration[best[1]]["win_probability"],
            "fleet_win_probability": best[2] / denominator,
            "fleet_wins": best[2],
            "scenarios": denominator,
        })
    return trace


def easynano_analysis(
    main_passes: list[dict[str, dict[str, float]]],
    extra_passes: list[dict[str, dict[str, float]]],
    cases: list[dict],
) -> dict:
    by_id = {case["id"]: case["sequence"] for case in cases}
    passes = main_passes + extra_passes
    aggregate_wins = defaultdict(int)
    rank_sums = defaultdict(list)
    paired = {}
    for pass_metrics in passes:
        order = [by_id[key] for key in ("ours_anchor", "easynano_h1", "rival_anchor")]
        totals = dense_rank_sums([pass_metrics[sequence] for sequence in order])
        best = min(totals)
        for key, total in zip(("ours_anchor", "easynano_h1", "rival_anchor"), totals):
            aggregate_wins[key] += total == best
            rank_sums[key].append(total)
    proposal = by_id["easynano_h1"]
    for comparator_id in ("ours_anchor", "rival_anchor"):
        comparator = by_id[comparator_id]
        metric_rows = {}
        for metric, mode in METRICS.items():
            raw = [p[proposal][metric] - p[comparator][metric] for p in passes]
            favorable = [delta > 0 if mode == "max" else delta < 0 for delta in raw]
            metric_rows[metric] = {
                "mean_raw_delta": mean(raw),
                "favorable_passes": sum(favorable),
                "passes": len(raw),
            }
        paired[comparator_id] = metric_rows
    return {
        "passes": len(passes),
        "aggregate_wins": dict(aggregate_wins),
        "mean_three_way_rank_sum": {key: mean(value) for key, value in rank_sums.items()},
        "paired_metrics": paired,
        "sequences": by_id,
    }


def basin_analysis(
    passes: list[dict[str, dict[str, float]]], manifest: dict
) -> dict:
    candidate_rank_sums: dict[str, list[int]] = defaultdict(list)
    for pass_metrics in passes:
        sequences = list(pass_metrics)
        totals = dense_rank_sums([pass_metrics[sequence] for sequence in sequences])
        for sequence, total in zip(sequences, totals):
            candidate_rank_sums[sequence].append(total)
    candidate_mean = {seq: mean(values) for seq, values in candidate_rank_sums.items()}
    levels: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for case in manifest["cases"]:
        for source in case["provenance"]:
            if source["group"] == "easynano-three-way":
                continue
            levels[(source["group"], source["anchor_side"])][int(source["edit_count"])].add(case["sequence"])
    rows = []
    for (group, side), depth in sorted(levels.items()):
        best_by_depth = {}
        for edit_count, sequences in sorted(depth.items()):
            winner = min(sequences, key=lambda seq: candidate_mean[seq])
            best_by_depth[str(edit_count)] = {
                "sequence": winner,
                "mean_rank_sum": candidate_mean[winner],
                "candidate_count": len(sequences),
            }
        best_depth = min(best_by_depth, key=lambda key: best_by_depth[key]["mean_rank_sum"])
        rows.append({
            "group": group, "side": side, "best_depth": int(best_depth),
            "best_by_depth": best_by_depth,
        })
    return {
        "basins": len(rows),
        "best_depth_counts": {
            str(depth): sum(row["best_depth"] == depth for row in rows)
            for depth in (0, 1, 2)
        },
        "rows": rows,
    }


def basin_prospective_holdout(
    selection_passes: list[dict[str, dict[str, float]]],
    evaluation_passes: list[dict[str, dict[str, float]]],
    manifest: dict,
) -> dict:
    """Select basin members on early draws and score them on unseen draws.

    The anchor remains eligible.  This makes a tie an honest decision to keep
    the known molecule instead of forcing every basin to take a noisy edit.
    """
    if not selection_passes or not evaluation_passes:
        raise ValueError("prospective basin analysis needs selection and evaluation passes")

    sequences = list(selection_passes[0])
    selection_scores: dict[str, list[int]] = defaultdict(list)
    evaluation_scores: list[dict[str, int]] = []
    for pass_metrics in selection_passes:
        totals = dense_rank_sums([pass_metrics[sequence] for sequence in sequences])
        for sequence, total in zip(sequences, totals):
            selection_scores[sequence].append(total)
    for pass_metrics in evaluation_passes:
        totals = dense_rank_sums([pass_metrics[sequence] for sequence in sequences])
        evaluation_scores.append(dict(zip(sequences, totals)))

    levels: dict[tuple[str, str], dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for case in manifest["cases"]:
        for source in case["provenance"]:
            if source["group"] == "easynano-three-way":
                continue
            levels[(source["group"], source["anchor_side"])][
                int(source["edit_count"])
            ].add(case["sequence"])

    comparisons = {"better": 0, "tied": 0, "worse": 0}
    selected_depths = defaultdict(int)
    deltas = []
    rows = []
    for (group, side), depth in sorted(levels.items()):
        anchor = next(iter(depth[0]))
        candidates = sorted(set().union(*depth.values()))
        selected = min(
            candidates,
            key=lambda sequence: (
                mean(selection_scores[sequence]),
                sequence,
            ),
        )
        selected_depth = min(
            edit_count
            for edit_count, depth_sequences in depth.items()
            if selected in depth_sequences
        )
        selected_depths[str(selected_depth)] += 1
        row_deltas = []
        for scores in evaluation_scores:
            delta = scores[selected] - scores[anchor]
            row_deltas.append(delta)
            deltas.append(delta)
            if delta < 0:
                comparisons["better"] += 1
            elif delta > 0:
                comparisons["worse"] += 1
            else:
                comparisons["tied"] += 1
        rows.append({
            "group": group,
            "side": side,
            "selected_sequence": selected,
            "selected_depth": selected_depth,
            "evaluation_rank_sum_deltas": row_deltas,
        })
    return {
        "selection_passes": len(selection_passes),
        "evaluation_passes": len(evaluation_passes),
        "basins": len(rows),
        "selected_depth_counts": dict(selected_depths),
        "comparisons": comparisons,
        "mean_evaluation_rank_sum_delta": mean(deltas),
        "rows": rows,
    }


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    main_paths = sorted(HERE.glob("pass_*/exact_all.json"))
    extra_paths = sorted(HERE.glob("easynano_extra_*/exact_all.json"))
    if len(main_paths) != 12 or len(extra_paths) != 8:
        raise RuntimeError(f"expected 12+8 passes, found {len(main_paths)}+{len(extra_paths)}")
    main_passes = [components(path) for path in main_paths]
    extra_passes = [components(path) for path in extra_paths]
    expected = {case["sequence"] for case in manifest["cases"]}
    if any(set(p) != expected for p in main_passes):
        raise RuntimeError("main-pass sequence cohort mismatch")

    fields, public_sequences = public_fields()
    eligible = sorted(expected - public_sequences)
    calibration = individual_calibration(main_passes, fields, eligible)
    fleet = greedy_fleet(main_passes, fields, calibration, eligible)
    easynano = easynano_analysis(main_passes, extra_passes, manifest["easynano_cases"])
    basins = basin_analysis(main_passes, manifest)
    # Reserve the last third of independently generated observations as a
    # prospective test.  Selecting and evaluating on all twelve draws looked
    # impressive in early analysis even though the winning sequences changed
    # almost completely from draw to draw.
    holdout_count = max(1, len(main_passes) // 3)
    selection_passes = main_passes[:-holdout_count]
    evaluation_passes = main_passes[-holdout_count:]
    selection_calibration = individual_calibration(
        selection_passes, fields, eligible
    )
    selection_fleet = greedy_fleet(
        selection_passes, fields, selection_calibration, eligible
    )
    selected_sequences = [row["sequence"] for row in selection_fleet]
    prospective_fleet = {
        "selection_passes": len(selection_passes),
        "evaluation_passes": len(evaluation_passes),
        "selected_sequences": selected_sequences,
        "selection_trace": selection_fleet,
        "selection_result": evaluate_fleet(
            selected_sequences, selection_passes, fields
        ),
        "evaluation_result": evaluate_fleet(
            selected_sequences, evaluation_passes, fields
        ),
    }
    prospective_basins = basin_prospective_holdout(
        selection_passes, evaluation_passes, manifest
    )
    leaders = sorted(
        eligible,
        key=lambda seq: (
            calibration[seq]["win_probability"],
            calibration[seq]["mean_winning_margin"],
        ),
        reverse=True,
    )[:20]
    result = {
        "protocol": "validator-dual-sample-v1",
        "metrics": METRICS,
        "main_passes": len(main_passes),
        "easynano_passes": len(main_passes) + len(extra_passes),
        "public_calibration_epochs": [field["epoch"] for field in fields],
        "public_field_sizes": [len(field["records"]) for field in fields],
        "scenario_count": len(main_passes) * len(fields),
        "candidate_count": len(expected),
        "eligible_novel_candidate_count": len(eligible),
        "basin_closure": basins,
        "basin_prospective_holdout": prospective_basins,
        "easynano": easynano,
        "individual_leaders": [
            {"sequence": seq, **calibration[seq]} for seq in leaders
        ],
        "correlation_aware_fleet": fleet,
        "prospective_fleet_holdout": prospective_fleet,
    }
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    e = result["easynano"]
    p_anchor = e["paired_metrics"]["ours_anchor"]
    p_rival = e["paired_metrics"]["rival_anchor"]
    final_fleet = fleet[-1] if fleet else None
    heldout = prospective_fleet["evaluation_result"]
    prospective_closure = result["basin_prospective_holdout"]
    lines = [
        "# Fast-BoltzGen Hamming-2 closure and EasyNano replay",
        "",
        "## Verdict",
        "",
        "VERDICT_PLACEHOLDER",
        "",
        "## Experiment",
        "",
        f"- Exact protocol: `{result['protocol']}`; all ten validator metrics.",
        f"- Closure: {manifest['closure_unique_sequences']} unique sequences from 24 basin anchors and every compatible Hamming-0/1/2 combination of the top 20 public-win-associated substitutions.",
        f"- Repeats: {result['main_passes']} complete observations per closure sequence; {result['easynano_passes']} per EasyNano three-way sequence.",
        f"- Calibration context: {len(fields)} public epoch fields ({fields[0]['epoch']}–{fields[-1]['epoch']}). Candidate selection uses the first {len(selection_passes)} scorer passes and every headline performance number uses the final {len(evaluation_passes)} previously unseen passes.",
        "",
        "## EasyNano Hamming-1 proposal",
        "",
        f"- Three-way aggregate wins: proposal {e['aggregate_wins'].get('easynano_h1', 0)}/{e['passes']}, our anchor {e['aggregate_wins'].get('ours_anchor', 0)}/{e['passes']}, rival anchor {e['aggregate_wins'].get('rival_anchor', 0)}/{e['passes']}.",
        f"- Proposal design_iiptm versus our anchor: mean raw delta {p_anchor['design_iiptm']['mean_raw_delta']:+.5f}; favorable in {p_anchor['design_iiptm']['favorable_passes']}/{p_anchor['design_iiptm']['passes']} passes.",
        f"- Proposal design_iiptm versus rival anchor: mean raw delta {p_rival['design_iiptm']['mean_raw_delta']:+.5f}; favorable in {p_rival['design_iiptm']['favorable_passes']}/{p_rival['design_iiptm']['passes']} passes.",
        "",
        "## Deterministic basin closure",
        "",
        f"Exploratory best depth using all observations across {basins['basins']} side-specific basins: anchor={basins['best_depth_counts']['0']}, Hamming-1={basins['best_depth_counts']['1']}, Hamming-2={basins['best_depth_counts']['2']}.",
        f"Prospective selection retained anchor/Hamming-1/Hamming-2 in {prospective_closure['selected_depth_counts'].get('0', 0)}/{prospective_closure['selected_depth_counts'].get('1', 0)}/{prospective_closure['selected_depth_counts'].get('2', 0)} basins. On unseen draws those choices were better/tied/worse than their anchors in {prospective_closure['comparisons']['better']}/{prospective_closure['comparisons']['tied']}/{prospective_closure['comparisons']['worse']} basin-draw comparisons.",
        "",
        "## Correlation-aware fleet",
        "",
    ]
    if final_fleet:
        lines.append(
            f"The fleet chosen without the held-out draws wins or ties the best public entry in {heldout['fleet_wins']}/{heldout['scenarios']} prospective replay scenarios ({heldout['fleet_win_probability']:.1%}). The all-pass fleet below is the final post-evaluation fit; its in-sample probability is descriptive, not the performance estimate."
        )
    lines.extend(["", "| slot | individual P(win) | fleet P(any win) | sequence |", "|---:|---:|---:|---|"])
    for row in fleet:
        lines.append(
            f"| {row['slot']} | {row['individual_win_probability']:.1%} | {row['fleet_win_probability']:.1%} | `{row['sequence']}` |"
        )
    lines.extend([
        "",
        "The selector preserves the complete ten-metric vector from each scorer pass and uses one intact public epoch field per scenario. It therefore does not multiply independent per-metric or per-sequence probabilities. The prospective estimate also prevents a lucky scorer draw from serving as both selection evidence and proof.",
        "",
        "## Files",
        "",
        "- `analysis.json`: complete paired metric, basin, calibration, and fleet results.",
        "- `manifest.json`: exact closure definition and provenance for every sequence.",
    ])
    # The numeric evidence above is the report; choose the short verdict from it.
    proposal_wins = e["aggregate_wins"].get("easynano_h1", 0)
    ours_wins = e["aggregate_wins"].get("ours_anchor", 0)
    h2_count = prospective_closure["selected_depth_counts"].get("2", 0)
    closure_cmp = prospective_closure["comparisons"]
    verdict = (
        f"EasyNano's Hamming-1 is {'supported' if proposal_wins > ours_wins else 'not supported'} "
        f"by the exact selector ({proposal_wins} versus {ours_wins} aggregate wins). "
        f"The prospective closure selector chose Hamming-2 in {h2_count}/{basins['basins']} basin sides and was better/tied/worse than the anchor on unseen draws in "
        f"{closure_cmp['better']}/{closure_cmp['tied']}/{closure_cmp['worse']} comparisons. "
        f"The held-out fleet win rate is {heldout['fleet_win_probability']:.1%}. "
        "Those unseen-draw results, not the more favorable all-pass fit, decide whether the strategy is supported."
    )
    lines[4] = verdict
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "easynano_proposal_wins": proposal_wins,
        "ours_anchor_wins": ours_wins,
        "h2_selected_basins": h2_count,
        "heldout_fleet_probability": heldout["fleet_win_probability"],
        "all_pass_fleet_probability": final_fleet["fleet_win_probability"] if final_fleet else None,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
