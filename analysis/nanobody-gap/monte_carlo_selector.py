"""Compare selector strategies on independent fast-scorer observations.

For every held-out observation, each strategy selects a molecule using all
remaining observations.  The untouched observation is then scored with the
validator's unweighted ten-metric dense-rank objective.  Inputs may contain any
sequences; only the manifest's ``group`` and ``sequence`` fields are required.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nanobody_opt.win_probability import rank_by_fresh_draw_probability  # noqa: E402

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
CONFIDENCE = set(list(METRICS)[:5])
POLICIES = {
    "validator-plain": {metric: 1.0 for metric in METRICS},
    "confidence-only": {metric: float(metric in CONFIDENCE) for metric in METRICS},
    "design-iiptm-only": {
        metric: float(metric == "design_iiptm") for metric in METRICS
    },
    "target-iptm-only": {
        metric: float(metric == "design_to_target_iptm") for metric in METRICS
    },
}


def load_observation(path: Path, target: str) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    found = {}
    for by_sequence in (payload.get("per_nanobody_components") or {}).values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target.get(target)
            if metrics and set(METRICS).issubset(metrics):
                found[sequence] = {metric: float(metrics[metric]) for metric in METRICS}
    return found


def dense_scores(
    values: dict[str, dict[str, float]], weights: dict[str, float]
) -> dict[str, float]:
    fields = {
        metric: sorted({row[metric] for row in values.values()}) for metric in METRICS
    }
    scores = {}
    for sequence, metrics in values.items():
        total = 0.0
        for metric, direction in METRICS.items():
            field, value = fields[metric], metrics[metric]
            rank = (
                bisect.bisect_left(field, value) + 1
                if direction == "min"
                else len(field) - bisect.bisect_right(field, value) + 1
            )
            total += weights[metric] * rank
        scores[sequence] = total
    return scores


def observation_scores(
    observations: list[dict[str, dict[str, float]]], sequences: list[str]
) -> dict[str, list[float]]:
    """Score each intact metric draw against a fixed mean-metric field.

    Keeping all ten metrics from one observation together preserves their
    covariance.  The reference field is fixed at the training means so a draw
    cannot improve merely because a different synthetic field was sampled.
    """
    means = {
        sequence: {
            metric: statistics.mean(item[sequence][metric] for item in observations)
            for metric in METRICS
        }
        for sequence in sequences
    }
    fields = {
        metric: sorted({row[metric] for row in means.values()}) for metric in METRICS
    }
    scored: dict[str, list[float]] = {sequence: [] for sequence in sequences}
    for observation in observations:
        for sequence in sequences:
            total = 0.0
            for metric, direction in METRICS.items():
                field = fields[metric]
                value = observation[sequence][metric]
                total += (
                    bisect.bisect_left(field, value) + 1
                    if direction == "min"
                    else len(field) - bisect.bisect_right(field, value) + 1
                )
            scored[sequence].append(total)
    return scored


def fresh_draw_winners(
    draw_scores: dict[str, list[float]],
    groups: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, float]]:
    """Exactly calculate who most often wins one untouched validator draw."""
    chosen: dict[str, str] = {}
    selected_probability: dict[str, float] = {}
    for group, members in groups.items():
        ranked = rank_by_fresh_draw_probability(
            [
                {
                    "sequence": sequence,
                    "per_observation_median_rank_sum": draw_scores[sequence],
                    "score_median": statistics.median(draw_scores[sequence]),
                }
                for sequence in members
            ]
        )
        winner = str(ranked[0]["sequence"])
        chosen[group] = winner
        selected_probability[group] = float(ranked[0]["win_probability"])
    return chosen, selected_probability


def evaluate(
    manifest: list[dict],
    observations: list[dict[str, dict[str, float]]],
    *,
    training_depth: int | None = None,
) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for item in manifest:
        groups[str(item["group"])].append(str(item["sequence"]))
    sequences = sorted(
        {sequence for members in groups.values() for sequence in members}
    )
    if len(observations) < 2:
        raise ValueError("Monte Carlo comparison needs at least two observations")
    if any(
        sequence not in observation
        for observation in observations
        for sequence in sequences
    ):
        raise ValueError("every observation must contain every manifest sequence")

    decisions = []
    for held_out, evaluation in enumerate(observations):
        available_training = [
            observation
            for index, observation in enumerate(observations)
            if index != held_out
        ]
        training_observations = (
            available_training[:training_depth]
            if training_depth is not None
            else available_training
        )
        training = {
            sequence: {
                metric: statistics.mean(
                    observation[sequence][metric]
                    for observation in training_observations
                )
                for metric in METRICS
            }
            for sequence in sequences
        }
        evaluation_scores = dense_scores(evaluation, POLICIES["validator-plain"])
        for policy, weights in POLICIES.items():
            selection_scores = dense_scores(training, weights)
            for group, members in groups.items():
                chosen = min(
                    members, key=lambda sequence: (selection_scores[sequence], sequence)
                )
                oracle = min(
                    members,
                    key=lambda sequence: (evaluation_scores[sequence], sequence),
                )
                decisions.append(
                    {
                        "held_out_observation": held_out + 1,
                        "group": group,
                        "policy": policy,
                        "chosen_sequence": chosen,
                        "oracle_sequence": oracle,
                        "chosen_is_oracle": chosen == oracle,
                        "chosen_validator_score": evaluation_scores[chosen],
                        "oracle_validator_score": evaluation_scores[oracle],
                        "regret": evaluation_scores[chosen] - evaluation_scores[oracle],
                    }
                )
        draws = observation_scores(training_observations, sequences)
        fresh_choices, fresh_probabilities = fresh_draw_winners(
            draws,
            groups,
        )
        for group, members in groups.items():
            chosen = fresh_choices[group]
            oracle = min(
                members, key=lambda sequence: (evaluation_scores[sequence], sequence)
            )
            decisions.append(
                {
                    "held_out_observation": held_out + 1,
                    "group": group,
                    "policy": "fresh-draw-win-probability",
                    "chosen_sequence": chosen,
                    "oracle_sequence": oracle,
                    "chosen_is_oracle": chosen == oracle,
                    "chosen_validator_score": evaluation_scores[chosen],
                    "oracle_validator_score": evaluation_scores[oracle],
                    "regret": evaluation_scores[chosen] - evaluation_scores[oracle],
                    "estimated_win_probability": fresh_probabilities[group],
                }
            )
    summaries = []
    for policy in (*POLICIES, "fresh-draw-win-probability"):
        rows = [row for row in decisions if row["policy"] == policy]
        summaries.append(
            {
                "policy": policy,
                "held_out_decisions": len(rows),
                "oracle_pick_rate": statistics.mean(
                    row["chosen_is_oracle"] for row in rows
                ),
                "mean_regret": statistics.mean(row["regret"] for row in rows),
                "median_regret": statistics.median(row["regret"] for row in rows),
            }
        )
    return decisions, sorted(
        summaries, key=lambda row: (row["mean_regret"], -row["oracle_pick_rate"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--observations", required=True, help="glob for scorer JSON files"
    )
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--training-depth",
        type=int,
        help="maximum non-held-out observations available to each selector",
    )
    args = parser.parse_args()
    if args.training_depth is not None and args.training_depth < 1:
        parser.error("--training-depth must be positive")
    raw_manifest = json.loads(args.manifest.read_text())
    manifest = (
        raw_manifest.get("cases", raw_manifest)
        if isinstance(raw_manifest, dict)
        else raw_manifest
    )
    # The retained replay calls the grouping key ``epoch``; new arbitrary
    # cohorts can name it ``group`` directly.
    manifest = [
        {**item, "group": item.get("group", item.get("epoch"))} for item in manifest
    ]
    paths = [Path(path) for path in sorted(glob.glob(args.observations))]
    observations = [load_observation(path, args.target) for path in paths]
    decisions, summaries = evaluate(
        manifest,
        observations,
        training_depth=args.training_depth,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "target": args.target,
                "observation_files": [str(path) for path in paths],
                "observations": len(observations),
                "groups": len({item["group"] for item in manifest}),
                "training_depth": args.training_depth,
                "fresh_draw_probability_method": "exact-empirical",
                "summary": summaries,
                "decisions": decisions,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
