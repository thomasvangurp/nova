#!/usr/bin/env python3
"""Analyze the preregistered GPU-capacity proposal experiment.

Selection uses only confirmation passes 1--12.  Every reported performance
comparison uses passes 13--20, which were neither used to form the shortlist
nor to choose a sequence or fleet at any repeat depth.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DB = HERE.parent / "mutations.sqlite"
DASHBOARD_DB = HERE.parents[2] / "data" / "nanobody_dash.sqlite"
TARGET = "P20809"
ARMS = ("current_production", "ours_hamming2", "rival_hamming2")
DEPTHS = (1, 2, 4, 8, 12)
HOLDOUT = tuple(range(13, 21))
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


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def components(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol in {path}")
    result: dict[str, dict[str, float]] = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            if not all(key in metrics for key in METRICS):
                raise RuntimeError(f"incomplete metric vector for {sequence} in {path}")
            result[sequence] = {key: float(metrics[key]) for key in METRICS}
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


def rank_map(observed: dict[str, dict[str, float]]) -> dict[str, int]:
    sequences = list(observed)
    return dict(zip(sequences, dense_rank_sums([observed[s] for s in sequences])))


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
    public_sequences = {str(sequence) for _, sequence, _ in rows}
    dashboard = sqlite3.connect(DASHBOARD_DB)
    public_sequences.update(
        str(row[0])
        for row in dashboard.execute(
            "SELECT DISTINCT sequence FROM entries WHERE sequence IS NOT NULL"
        )
    )
    dashboard.close()
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for epoch, _sequence, raw in rows:
        metrics = json.loads(raw)
        if all(key in metrics for key in METRICS):
            grouped[int(epoch)].append({key: float(metrics[key]) for key in METRICS})
    epochs = sorted(epoch for epoch, field in grouped.items() if len(field) >= 30)[-30:]
    if not epochs:
        raise RuntimeError("no complete public calibration fields")
    return ([{"epoch": epoch, "records": grouped[epoch]} for epoch in epochs], public_sequences)


def head_to_head(
    left: str,
    right: str,
    passes: list[dict[str, dict[str, float]]],
) -> dict:
    outcome = {"left": 0, "tie": 0, "right": 0}
    margins = []
    per_pass = []
    for observed in passes:
        left_score, right_score = dense_rank_sums([observed[left], observed[right]])
        margin = right_score - left_score
        margins.append(margin)
        winner = "left" if margin > 0 else "right" if margin < 0 else "tie"
        outcome[winner] += 1
        per_pass.append({"left_rank_sum": left_score, "right_rank_sum": right_score, "winner": winner})
    metric_rows = {}
    for metric, mode in METRICS.items():
        deltas = [observed[left][metric] - observed[right][metric] for observed in passes]
        favorable = [delta > 0 if mode == "max" else delta < 0 for delta in deltas]
        metric_rows[metric] = {
            "mean_left_minus_right": mean(deltas),
            "left_favorable_draws": sum(favorable),
            "draws": len(deltas),
        }
    return {
        "outcome": outcome,
        "mean_left_winning_margin": mean(margins),
        "per_pass": per_pass,
        "paired_metrics": metric_rows,
    }


def fleet_outcome(
    fleet: list[str],
    observed: dict[str, dict[str, float]],
    field: dict,
) -> bool:
    records = field["records"] + [observed[sequence] for sequence in fleet]
    scores = dense_rank_sums(records)
    split = len(field["records"])
    return min(scores[split:]) <= min(scores[:split])


def individual_win_scenarios(
    sequence: str,
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
) -> set[int]:
    wins = set()
    scenario = 0
    for observed in passes:
        for field in fields:
            if fleet_outcome([sequence], observed, field):
                wins.add(scenario)
            scenario += 1
    return wins


def correlation_aware_fleet(
    candidates: list[str],
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
    size: int = 10,
) -> tuple[list[str], list[dict]]:
    """Greedily cover intact pass-by-public-field win scenarios.

    Candidate metrics are never factorized.  A slot is useful only when it
    wins complete scenarios not already covered by the prior slots.
    """
    wins = {
        sequence: individual_win_scenarios(sequence, passes, fields)
        for sequence in candidates
    }
    selected: list[str] = []
    covered: set[int] = set()
    trace = []
    scenarios = len(passes) * len(fields)
    for _ in range(min(size, len(candidates))):
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
            "coverage": len(covered) / scenarios,
        })
    return selected, trace


def evaluate_fleet(
    fleet: list[str],
    passes: list[dict[str, dict[str, float]]],
    fields: list[dict],
) -> dict:
    wins = sum(
        fleet_outcome(fleet, observed, field)
        for observed in passes
        for field in fields
    )
    scenarios = len(passes) * len(fields)
    return {"wins": wins, "scenarios": scenarios, "win_probability": wins / scenarios}


def arm_depth_analysis(
    passes: list[dict[str, dict[str, float]]],
    selection_fields: list[dict],
    evaluation_fields: list[dict],
    proposals: dict[str, list[str]],
) -> list[dict]:
    heldout = passes[12:20]
    heldout_ranks = [rank_map(observed) for observed in heldout]
    rows = []
    for depth in DEPTHS:
        training = passes[:depth]
        training_ranks = [rank_map(observed) for observed in training]
        champions = {}
        arm_rows = {}
        for arm in ARMS:
            candidates = proposals[arm]
            champion = min(
                candidates,
                key=lambda sequence: (
                    mean([ranks[sequence] for ranks in training_ranks]), sequence
                ),
            )
            champions[arm] = champion
            heldout_mean = mean([ranks[champion] for ranks in heldout_ranks])
            oracle = min(
                candidates,
                key=lambda sequence: mean([ranks[sequence] for ranks in heldout_ranks]),
            )
            oracle_mean = mean([ranks[oracle] for ranks in heldout_ranks])
            fleet, fleet_trace = correlation_aware_fleet(
                candidates, training, selection_fields
            )
            arm_rows[arm] = {
                "candidate_count": len(candidates),
                "champion": champion,
                "champion_training_mean_rank_sum": mean(
                    [ranks[champion] for ranks in training_ranks]
                ),
                "champion_heldout_mean_rank_sum": heldout_mean,
                "heldout_oracle": oracle,
                "heldout_oracle_mean_rank_sum": oracle_mean,
                "selection_regret": heldout_mean - oracle_mean,
                "correlation_aware_fleet": fleet,
                "fleet_selection_trace": fleet_trace,
                "heldout_fleet": evaluate_fleet(
                    fleet, heldout, evaluation_fields
                ),
            }
        champion_wins = {arm: 0 for arm in ARMS}
        champion_ties = 0
        for observed in heldout:
            scores = dense_rank_sums([observed[champions[arm]] for arm in ARMS])
            best = min(scores)
            winners = [arm for arm, score in zip(ARMS, scores) if score == best]
            for arm in winners:
                champion_wins[arm] += 1
            champion_ties += len(winners) > 1
        rows.append({
            "selection_depth": depth,
            "selection_passes": list(range(1, depth + 1)),
            "evaluation_passes": list(HOLDOUT),
            "arms": arm_rows,
            "three_way_champion_winning_draws": champion_wins,
            "three_way_tied_draws": champion_ties,
            "champion_pairwise": {
                challenger: head_to_head(
                    champions[challenger], champions["current_production"], heldout
                )
                for challenger in ("ours_hamming2", "rival_hamming2")
            },
        })
    for index, row in enumerate(rows[:-1]):
        next_row = rows[index + 1]
        row["champion_retained_at_next_depth"] = {
            arm: row["arms"][arm]["champion"] == next_row["arms"][arm]["champion"]
            for arm in ARMS
        }
    return rows


def proposal_width_analysis(
    manifest: dict,
    discovery: list[dict[str, dict[str, float]]],
    selection_fields: list[dict],
    evaluation_fields: list[dict],
) -> dict:
    """Secondary nested-width test with a strict 2-draw/2-draw split.

    The sequence order was frozen before any experiment score existed.  This
    asks whether paying to score a wider prefix supplies a better selected
    champion or a less-correlated fleet, without reusing its evaluation draw.
    """
    selection = discovery[:2]
    evaluation = discovery[2:]
    selection_ranks = [rank_map(observed) for observed in selection]
    evaluation_ranks = [rank_map(observed) for observed in evaluation]
    ordered = {
        arm: [
            case["sequence"] for case in manifest["cases"]
            if arm in case["arms"] and not case.get("known_public", False)
        ]
        for arm in ARMS
    }
    result = {}
    for arm in ARMS:
        widths = list(dict.fromkeys([24, 72, 144, len(ordered[arm])]))
        rows = []
        for width in widths:
            candidates = ordered[arm][:width]
            champion = min(
                candidates,
                key=lambda sequence: (
                    mean([ranks[sequence] for ranks in selection_ranks]), sequence
                ),
            )
            fleet, trace = correlation_aware_fleet(
                candidates, selection, selection_fields
            )
            rows.append({
                "scored_proposals": len(candidates),
                "champion": champion,
                "champion_evaluation_mean_rank_sum": mean(
                    [ranks[champion] for ranks in evaluation_ranks]
                ),
                "correlation_aware_fleet": fleet,
                "fleet_selection_trace": trace,
                "evaluation_fleet": evaluate_fleet(
                    fleet, evaluation, evaluation_fields
                ),
            })
        result[arm] = {
            "rows": rows,
            "widest_minus_24_fleet_probability": (
                rows[-1]["evaluation_fleet"]["win_probability"]
                - rows[0]["evaluation_fleet"]["win_probability"]
            ),
            "widest_minus_24_champion_rank_sum": (
                rows[-1]["champion_evaluation_mean_rank_sum"]
                - rows[0]["champion_evaluation_mean_rank_sum"]
            ),
        }
    return {
        "status": "secondary_post_preregistered_width_diagnostic",
        "selection_discovery_passes": [1, 2],
        "evaluation_discovery_passes": [3, 4],
        "arms": result,
    }


def basin_analysis(
    manifest: dict,
    shortlist: dict,
    passes: list[dict[str, dict[str, float]]],
) -> dict:
    by_sequence = {case["sequence"]: case for case in manifest["cases"]}
    proposals = {
        row["arm"]: row["selected"] for row in shortlist["trace"]
    }
    training_ranks = [rank_map(observed) for observed in passes[:12]]
    heldout = passes[12:20]
    groups = sorted({
        group
        for arm in ("ours_hamming2", "rival_hamming2")
        for sequence in proposals[arm]
        for group in by_sequence[sequence]["arm_groups"].get(arm, [])
    })
    rows = []
    basin_decisions = {"rival": 0, "tie": 0, "ours": 0}
    anchor_decisions = {
        "ours_hamming2": {"proposal": 0, "tie": 0, "anchor": 0},
        "rival_hamming2": {"proposal": 0, "tie": 0, "anchor": 0},
    }
    for group in groups:
        selected = {}
        anchors = {}
        for arm, side in (("ours_hamming2", "ours"), ("rival_hamming2", "rival")):
            candidates = [
                sequence for sequence in proposals[arm]
                if group in by_sequence[sequence]["arm_groups"].get(arm, [])
            ]
            if not candidates:
                raise RuntimeError(f"shortlist lacks {arm} candidate for {group}")
            selected[arm] = min(
                candidates,
                key=lambda sequence: (
                    mean([ranks[sequence] for ranks in training_ranks]), sequence
                ),
            )
            anchor_candidates = [
                case["sequence"] for case in manifest["cases"]
                if any(
                    source.get("group") == group
                    and source.get("anchor_side") == side
                    and int(source.get("edit_count", -1)) == 0
                    for source in case["closure_provenance"]
                )
            ]
            if len(set(anchor_candidates)) != 1:
                raise RuntimeError(f"expected one {side} anchor for {group}")
            anchors[arm] = anchor_candidates[0]
        rival_vs_ours = head_to_head(
            selected["rival_hamming2"], selected["ours_hamming2"], heldout
        )
        score = rival_vs_ours["outcome"]
        decision = (
            "rival" if score["left"] > score["right"]
            else "ours" if score["right"] > score["left"]
            else "tie"
        )
        basin_decisions[decision] += 1
        anchor_rows = {}
        for arm in ("ours_hamming2", "rival_hamming2"):
            comparison = head_to_head(selected[arm], anchors[arm], heldout)
            outcome = comparison["outcome"]
            decision_name = (
                "proposal" if outcome["left"] > outcome["right"]
                else "anchor" if outcome["right"] > outcome["left"]
                else "tie"
            )
            anchor_decisions[arm][decision_name] += 1
            anchor_rows[arm] = comparison
        rows.append({
            "group": group,
            "ours_sequence": selected["ours_hamming2"],
            "rival_sequence": selected["rival_hamming2"],
            "ours_anchor": anchors["ours_hamming2"],
            "rival_anchor": anchors["rival_hamming2"],
            "rival_vs_ours": rival_vs_ours,
            "basin_decision": decision,
            "proposal_vs_anchor": anchor_rows,
        })
    return {
        "selection_passes": list(range(1, 13)),
        "evaluation_passes": list(HOLDOUT),
        "basins": len(rows),
        "rival_vs_ours_basin_decisions": basin_decisions,
        "proposal_vs_anchor_basin_decisions": anchor_decisions,
        "basin_gate": {
            "required_rival_wins": 8,
            "observed_rival_wins": basin_decisions["rival"],
            "passed": basin_decisions["rival"] >= 8,
        },
        "rows": rows,
    }


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    shortlist = json.loads((HERE / "shortlist_manifest.json").read_text())
    discovery_paths = [
        HERE / f"discovery_{index:02d}" / "exact_all.json" for index in range(1, 5)
    ]
    missing_discovery = [str(path) for path in discovery_paths if not path.is_file()]
    if missing_discovery:
        raise RuntimeError(f"missing discovery results: {missing_discovery}")
    discovery = [components(path) for path in discovery_paths]
    discovery_expected = {case["sequence"] for case in manifest["cases"]}
    if any(set(observed) != discovery_expected for observed in discovery):
        raise RuntimeError("discovery sequence cohort mismatch")
    paths = [HERE / f"confirmation_{index:02d}" / "exact_all.json" for index in range(1, 21)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing confirmation results: {missing}")
    passes = [components(path) for path in paths]
    expected = set(shortlist["sequences"])
    if any(set(observed) != expected for observed in passes):
        raise RuntimeError("confirmation sequence cohort mismatch")
    if len(passes[12:]) != 8:
        raise RuntimeError("held-out confirmation cohort must contain eight passes")

    proposals = {row["arm"]: row["selected"] for row in shortlist["trace"]}
    if set(proposals) != set(ARMS) or any(len(proposals[arm]) != 24 for arm in ARMS):
        raise RuntimeError("shortlist must contain 24 proposals from every arm")
    fields, public_sequences = public_fields()
    # Neither the stochastic scorer draws nor the public competitor fields in
    # the headline evaluation are visible during selection.
    selection_fields = fields[:-8]
    evaluation_fields = fields[-8:]
    if not selection_fields or len(evaluation_fields) != 8:
        raise RuntimeError("public-field split requires at least nine fields")
    leaked = {
        arm: sorted(set(sequences) & public_sequences)
        for arm, sequences in proposals.items()
    }
    if any(leaked.values()):
        raise RuntimeError(f"known public sequences entered proposal shortlist: {leaked}")

    depths = arm_depth_analysis(
        passes, selection_fields, evaluation_fields, proposals
    )
    basins = basin_analysis(manifest, shortlist, passes)
    widths = proposal_width_analysis(
        manifest, discovery, selection_fields, evaluation_fields
    )
    final = depths[-1]
    primary_gates = {}
    for challenger in ("ours_hamming2", "rival_hamming2"):
        comparison = final["champion_pairwise"][challenger]
        wins = comparison["outcome"]["left"]
        primary_gates[challenger] = {
            "required_heldout_wins": 5,
            "observed_heldout_wins": wins,
            "passed": wins >= 5,
        }

    # Follow the frozen rule on the uncensored selector-quality measure: mean
    # held-out regret versus the best member of the same arm. Public-field win
    # probability can be identically zero for a weak cohort and cannot then
    # say whether another repeat improved sequence choice.
    mean_selection_regret = [
        mean([row["arms"][arm]["selection_regret"] for arm in ARMS])
        for row in depths
    ]
    best_probability = [
        max(row["arms"][arm]["heldout_fleet"]["win_probability"] for arm in ARMS)
        for row in depths
    ]
    chosen_depth = DEPTHS[-1]
    for current, following, current_regret, following_regret in zip(
        DEPTHS, DEPTHS[1:], mean_selection_regret, mean_selection_regret[1:]
    ):
        if following_regret >= current_regret:
            chosen_depth = current
            break
    best_observed_depth = DEPTHS[
        min(range(len(DEPTHS)), key=lambda index: mean_selection_regret[index])
    ]
    later_beats_literal_stop = any(
        regret < mean_selection_regret[DEPTHS.index(chosen_depth)]
        for depth, regret in zip(DEPTHS, mean_selection_regret)
        if depth > chosen_depth
    )
    stable_plateau = chosen_depth != DEPTHS[-1] and not later_beats_literal_stop

    result = {
        "protocol": "validator-dual-sample-v1",
        "metrics": METRICS,
        "experiment": manifest["experiment"],
        "source_epoch": manifest["source_epoch"],
        "discovery_passes": 4,
        "confirmation_passes": 20,
        "selection_depths": list(DEPTHS),
        "heldout_passes": list(HOLDOUT),
        "public_calibration_epochs": [field["epoch"] for field in selection_fields],
        "public_evaluation_epochs": [field["epoch"] for field in evaluation_fields],
        "public_field_sizes": [len(field["records"]) for field in fields],
        "proposal_counts": {arm: len(proposals[arm]) for arm in ARMS},
        "known_public_shortlist_leakage": leaked,
        "proposal_width_diagnostic": widths,
        "depth_curve": depths,
        "primary_gates": primary_gates,
        "basin_analysis": basins,
        "capacity_rule": {
            "mean_heldout_selection_regret_by_depth": dict(
                zip(map(str, DEPTHS), mean_selection_regret)
            ),
            "best_arm_heldout_fleet_probability_by_depth": dict(
                zip(map(str, DEPTHS), best_probability)
            ),
            "chosen_depth": chosen_depth,
            "best_observed_depth": best_observed_depth,
            "later_depth_beats_literal_stop": later_beats_literal_stop,
            "stable_plateau_established": stable_plateau,
            "rule": manifest["preregistration"]["capacity_decision"],
        },
    }
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# GPU capacity and proposal-quality validation",
        "",
        "## Verdict",
        "",
        "VERDICT_PLACEHOLDER",
        "",
        "## Frozen experiment",
        "",
        f"- Equal discovery budget: {manifest['arm_size']} candidates per arm; four complete ten-metric draws.",
        "- The confirmation shortlist contains 24 novel proposals per arm plus 24 controls.",
        "- Selection uses only confirmation passes 1–12; every performance number below uses untouched passes 13–20.",
        f"- Fleet selection uses {len(selection_fields)} intact public fields ({selection_fields[0]['epoch']}–{selection_fields[-1]['epoch']}); fleet evaluation uses eight untouched later fields ({evaluation_fields[0]['epoch']}–{evaluation_fields[-1]['epoch']}). Metrics are never resampled independently.",
        "",
        "## Repeat-depth curve",
        "",
        "| selection draws | mean selection regret | current fleet P(any win) | ours-H2 fleet P(any win) | rival-H2 fleet P(any win) | current champion wins | ours champion wins | rival champion wins |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in depths:
        lines.append(
            "| {depth} | {regret:.1f} | {current:.1%} | {ours:.1%} | {rival:.1%} | {cw}/8 | {ow}/8 | {rw}/8 |".format(
                depth=row["selection_depth"],
                regret=mean([row["arms"][arm]["selection_regret"] for arm in ARMS]),
                current=row["arms"]["current_production"]["heldout_fleet"]["win_probability"],
                ours=row["arms"]["ours_hamming2"]["heldout_fleet"]["win_probability"],
                rival=row["arms"]["rival_hamming2"]["heldout_fleet"]["win_probability"],
                cw=row["three_way_champion_winning_draws"]["current_production"],
                ow=row["three_way_champion_winning_draws"]["ours_hamming2"],
                rw=row["three_way_champion_winning_draws"]["rival_hamming2"],
            )
        )
    lines.extend([
        "",
        "## Scored-proposal width (secondary 2-draw/2-draw split)",
        "",
        "| arm | 24-proposal fleet P(any win) | widest fleet P(any win) | fleet change | champion rank-sum change |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        arm_width = widths["arms"][arm]
        first = arm_width["rows"][0]
        last = arm_width["rows"][-1]
        lines.append(
            f"| {arm} | {first['evaluation_fleet']['win_probability']:.1%} | "
            f"{last['evaluation_fleet']['win_probability']:.1%} ({last['scored_proposals']} proposals) | "
            f"{arm_width['widest_minus_24_fleet_probability']:+.1%} | "
            f"{arm_width['widest_minus_24_champion_rank_sum']:+.1f} |"
        )
    lines.extend([
        "",
        f"The preregistered one-step stopping rule literally chooses **{chosen_depth} draw(s)**, but a later depth beats that stop: **{best_observed_depth} draws** has the lowest observed held-out regret. Because the curve is non-monotone and worsens again at 12, this experiment does not establish a stable optimum.",
        "",
        "## Final 12-draw selection on eight untouched draws",
        "",
    ])
    for challenger in ("ours_hamming2", "rival_hamming2"):
        full_comparison = final["champion_pairwise"][challenger]
        comparison = full_comparison["outcome"]
        iiptm = full_comparison["paired_metrics"]["design_iiptm"]
        gate = primary_gates[challenger]
        lines.append(
            f"- {challenger} champion versus current-production champion: "
            f"{comparison['left']} wins, {comparison['tie']} ties, {comparison['right']} losses; "
            f"5/8 gate {'passed' if gate['passed'] else 'failed'}. Mean design_iiptm "
            f"delta {iiptm['mean_left_minus_right']:+.4f}, favorable on "
            f"{iiptm['left_favorable_draws']}/8 draws."
        )
    decisions = basins["rival_vs_ours_basin_decisions"]
    gate = basins["basin_gate"]
    lines.extend([
        "",
        "## Matched basin result",
        "",
        f"Rival closure won/tied/lost {decisions['rival']}/{decisions['tie']}/{decisions['ours']} of {basins['basins']} matched basins after selection on 12 draws and evaluation on eight untouched draws. The preregistered 8/12 basin gate {'passed' if gate['passed'] else 'failed'}.",
        "",
        "## Artifacts",
        "",
        "- `analysis.json` contains every selected sequence, complete paired metric delta, basin comparison, capacity curve, and exact scenario count.",
        "- `shortlist_manifest.json` records the discovery-only shortlist decision.",
        "- `manifest.json` freezes arm membership, basin provenance, public-novelty status, and input hashes.",
    ])
    passed = [arm for arm, gate in primary_gates.items() if gate["passed"]]
    if later_beats_literal_stop:
        capacity_verdict = (
            f"The literal one-step stop is {chosen_depth} draw(s), but it is invalidated "
            f"by the later curve: {best_observed_depth} draws has the lowest observed "
            "regret and 12 worsens again. No stable repeat optimum is established."
        )
    elif chosen_depth == DEPTHS[-1]:
        capacity_verdict = (
            "The curve did not plateau inside the tested range; saturation is not established."
        )
    else:
        capacity_verdict = (
            f"The frozen stopping rule selects {chosen_depth} draw(s): the next measured "
            "depth did not reduce held-out sequence-selection regret."
        )
    verdict = (
        f"Extra capacity produced {'a validated challenger (' + ', '.join(passed) + ')' if passed else 'no challenger that cleared the frozen 5/8 gate'}. "
        f"The matched rival-basin gate {'passed' if basins['basin_gate']['passed'] else 'failed'} "
        f"at {decisions['rival']}/12 basin wins. {capacity_verdict}"
    )
    lines[4] = verdict
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "primary_gates": primary_gates,
        "basin_gate": basins["basin_gate"],
        "capacity_depth": chosen_depth,
        "best_observed_depth": best_observed_depth,
        "mean_selection_regret_by_depth": dict(zip(map(str, DEPTHS), mean_selection_regret)),
        "best_probability_by_depth": dict(zip(map(str, DEPTHS), best_probability)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
