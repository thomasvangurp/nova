#!/usr/bin/env python3
"""Rank the retained our-versus-rival replay with the current selector.

The input cohort contains one public top-miner winner and our best submitted
sequence from the same epoch.  Every sequence was remeasured in every scorer
pass.  Each pass is converted to the validator's unweighted ten-metric dense
rank sum, then the intact per-pass rank sums are passed to the production
fresh-draw probability selector.

This is deliberately a ranking of existing molecules only.  It neither
generates neighbours nor uses public ownership, public score, or side labels in
the ranking calculation.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanobody_opt.win_probability import rank_by_fresh_draw_probability


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


def load_observation(path: Path, target: str) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    found: dict[str, dict[str, float]] = {}
    for by_sequence in (payload.get("per_nanobody_components") or {}).values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target.get(target)
            if metrics and set(METRICS).issubset(metrics):
                found[sequence] = {
                    metric: float(metrics[metric]) for metric in METRICS
                }
    return found


def validator_rank_sums(
    observation: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Apply the validator's unweighted dense-rank sum to one complete pass."""
    fields = {
        metric: sorted({values[metric] for values in observation.values()})
        for metric in METRICS
    }
    scores: dict[str, float] = {}
    for sequence, values in observation.items():
        total = 0.0
        for metric, direction in METRICS.items():
            field = fields[metric]
            value = values[metric]
            total += (
                bisect.bisect_left(field, value) + 1
                if direction == "min"
                else len(field) - bisect.bisect_right(field, value) + 1
            )
        scores[sequence] = total
    return scores


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_cohort(
    manifest: list[dict[str, Any]],
    observations: list[dict[str, dict[str, float]]],
) -> dict[str, Any]:
    meta_by_sequence = {str(item["sequence"]): item for item in manifest}
    sequences = sorted(meta_by_sequence)
    missing = [
        (pass_number, sequence)
        for pass_number, observation in enumerate(observations, 1)
        for sequence in sequences
        if sequence not in observation
    ]
    if missing:
        number, sequence = missing[0]
        raise ValueError(
            f"pass {number} is missing cohort sequence {sequence[:12]}..."
        )

    per_pass_scores = [validator_rank_sums(item) for item in observations]
    score_draws = {
        sequence: [scores[sequence] for scores in per_pass_scores]
        for sequence in sequences
    }
    candidates = [
        {
            "sequence": sequence,
            "per_observation_median_rank_sum": score_draws[sequence],
            "score_median": statistics.median(score_draws[sequence]),
        }
        for sequence in sequences
    ]
    global_ranking = rank_by_fresh_draw_probability(candidates)

    ranked_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(global_ranking, 1):
        sequence = str(item["sequence"])
        meta = meta_by_sequence[sequence]
        ranked_rows.append(
            {
                "rank": rank,
                "side": meta["side"],
                "epoch": int(meta["epoch"]),
                "fresh_draw_win_probability": float(item["win_probability"]),
                "mean_validator_rank_sum": statistics.mean(score_draws[sequence]),
                "median_validator_rank_sum": statistics.median(score_draws[sequence]),
                "best_validator_rank_sum": min(score_draws[sequence]),
                "worst_validator_rank_sum": max(score_draws[sequence]),
                "public_realised_score": meta.get("realised"),
                "sequence": sequence,
            }
        )

    by_epoch: dict[int, dict[str, str]] = collections.defaultdict(dict)
    for sequence, meta in meta_by_sequence.items():
        by_epoch[int(meta["epoch"])][str(meta["side"])] = sequence

    pair_rows: list[dict[str, Any]] = []
    for epoch, sides in sorted(by_epoch.items()):
        if set(sides) != {"ours", "rival"}:
            raise ValueError(f"epoch {epoch} does not contain one sequence per side")
        pair_candidates = [
            {
                "sequence": sequence,
                "per_observation_median_rank_sum": score_draws[sequence],
                "score_median": statistics.median(score_draws[sequence]),
            }
            for sequence in (sides["ours"], sides["rival"])
        ]
        pair_ranked = rank_by_fresh_draw_probability(pair_candidates)
        probabilities = {
            str(item["sequence"]): float(item["win_probability"])
            for item in pair_ranked
        }
        ours = sides["ours"]
        rival = sides["rival"]
        mutations = [
            f"{left}{position + 1}{right}"
            for position, (left, right) in enumerate(zip(ours, rival, strict=True))
            if left != right
        ]
        pair_rows.append(
            {
                "epoch": epoch,
                "selected_side": meta_by_sequence[str(pair_ranked[0]["sequence"])][
                    "side"
                ],
                "ours_pair_win_probability": probabilities[ours],
                "rival_pair_win_probability": probabilities[rival],
                "ours_mean_validator_rank_sum": statistics.mean(score_draws[ours]),
                "rival_mean_validator_rank_sum": statistics.mean(score_draws[rival]),
                "ours_minus_rival_mean_rank_sum": statistics.mean(score_draws[ours])
                - statistics.mean(score_draws[rival]),
                "hamming_distance": len(mutations),
                "ours_to_rival_mutations": ";".join(mutations),
                "ours_sequence": ours,
                "rival_sequence": rival,
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for metric, direction in METRICS.items():
        advantages: list[float] = []
        for sides in by_epoch.values():
            ours_values = [item[sides["ours"]][metric] for item in observations]
            rival_values = [item[sides["rival"]][metric] for item in observations]
            raw = statistics.mean(rival_values) - statistics.mean(ours_values)
            advantages.append(raw if direction == "max" else -raw)
        metric_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "mean_rival_advantage": statistics.mean(advantages),
                "median_rival_advantage": statistics.median(advantages),
                "rival_better_pairs": sum(value > 0 for value in advantages),
                "ours_better_pairs": sum(value < 0 for value in advantages),
                "ties": sum(value == 0 for value in advantages),
            }
        )

    position_counts: dict[int, collections.Counter[tuple[str, str, str]]] = (
        collections.defaultdict(collections.Counter)
    )
    for row in pair_rows:
        selected = str(row["selected_side"])
        winner = str(row[f"{selected}_sequence"])
        loser_side = "rival" if selected == "ours" else "ours"
        loser = str(row[f"{loser_side}_sequence"])
        for position, (winner_residue, loser_residue) in enumerate(
            zip(winner, loser, strict=True)
        ):
            if winner_residue != loser_residue:
                position_counts[position][
                    (winner_residue, loser_residue, selected)
                ] += 1
    mutation_rows = []
    for position, transitions in sorted(
        position_counts.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    ):
        mutation_rows.append(
            {
                "position_zero": position,
                "position_one": position + 1,
                "differing_pairs": sum(transitions.values()),
                "ours_selected_when_different": sum(
                    count
                    for (_, _, side), count in transitions.items()
                    if side == "ours"
                ),
                "rival_selected_when_different": sum(
                    count
                    for (_, _, side), count in transitions.items()
                    if side == "rival"
                ),
                "winner_over_loser_transitions": json.dumps(
                    [
                        {
                            "winner": winner,
                            "loser": loser,
                            "selected_side": side,
                            "count": count,
                        }
                        for (winner, loser, side), count in sorted(
                            transitions.items(), key=lambda item: (-item[1], item[0])
                        )
                    ],
                    separators=(",", ":"),
                ),
            }
        )

    global_by_side = collections.Counter(
        row["side"] for row in ranked_rows[: max(1, len(ranked_rows) // 4)]
    )
    pair_selected = collections.Counter(row["selected_side"] for row in pair_rows)
    return {
        "target": TARGET,
        "passes": len(observations),
        "sequences": len(sequences),
        "pairs": len(pair_rows),
        "ranking_method": "unweighted-validator-dense-rank-sum + exact-empirical-independent-fresh-draw probability",
        "global_top_quartile": dict(global_by_side),
        "pairwise_selected": dict(pair_selected),
        "ranking": ranked_rows,
        "pairs_detail": pair_rows,
        "metric_differences": metric_rows,
        "sequence_difference_positions": mutation_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if not isinstance(manifest, list):
        raise SystemExit("manifest must be a JSON list")
    paths = [Path(path) for path in sorted(glob.glob(args.observations))]
    if len(paths) < 2:
        raise SystemExit("at least two repeated observations are required")
    observations = [load_observation(path, args.target) for path in paths]
    result = rank_cohort(manifest, observations)
    result["observation_files"] = [str(path) for path in paths]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "existing_cohort_rank.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_csv(args.output / "existing_cohort_rank.csv", result["ranking"])
    write_csv(args.output / "existing_cohort_pairs.csv", result["pairs_detail"])
    write_csv(
        args.output / "existing_cohort_metric_differences.csv",
        result["metric_differences"],
    )
    write_csv(
        args.output / "existing_cohort_sequence_differences.csv",
        result["sequence_difference_positions"],
    )
    print(
        json.dumps(
            {
                "sequences": result["sequences"],
                "passes": result["passes"],
                "pairwise_selected": result["pairwise_selected"],
                "global_top_quartile": result["global_top_quartile"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
