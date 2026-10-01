#!/usr/bin/env python3
"""Frozen selectors; writes fleets before any fresh observation exists."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TARGET = "P20809"
METRICS = {
    "design_iiptm": "max", "design_ptm": "max",
    "design_to_target_iptm": "max", "min_design_to_target_pae": "min",
    "interaction_pae": "min", "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max", "delta_sasa_refolded": "max",
    "liability_score": "min", "liability_num_violations": "min",
}
POLICIES = {
    "A_live_selector_live_allocation": {
        "ranker": "internal", "stages": ((4, 180), (4, 108), (3, None)),
    },
    "B_field_selector_live_allocation": {
        "ranker": "field", "stages": ((4, 180), (4, 108), (3, None)),
    },
    "C_field_selector_delayed_6_4": {
        "ranker": "field", "stages": ((6, 111), (4, None)),
    },
    "D_field_selector_uniform_8": {
        "ranker": "field", "stages": ((8, None),),
    },
}


def load_draw(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol: {path}")
    found = {}
    for wave in payload["per_nanobody_components"].values():
        for sequence, by_target in wave.items():
            metrics = by_target[TARGET]
            if set(METRICS).issubset(metrics):
                found[sequence] = {name: float(metrics[name]) for name in METRICS}
    return found


def outcome_table(
    sequences: Sequence[str], draws: Sequence[Mapping[str, Mapping[str, float]]],
    fields: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    shape = (len(draws), len(sequences), len(fields))
    scores = np.zeros(shape, dtype=np.int16)
    wins = np.zeros(shape, dtype=np.bool_)
    top10 = np.zeros(shape, dtype=np.bool_)
    for draw_index, draw in enumerate(draws):
        for field_index, field_record in enumerate(fields):
            field = field_record["records"]
            candidate_total = np.zeros(len(sequences), dtype=np.int16)
            competitor_total = np.zeros((len(sequences), len(field)), dtype=np.int16)
            for metric, direction in METRICS.items():
                candidate = np.array([draw[sequence][metric] for sequence in sequences])
                competitor = np.array([float(row[metric]) for row in field])
                unique = np.unique(competitor)
                if direction == "max":
                    candidate_rank = 1 + (unique[None, :] > candidate[:, None]).sum(1)
                    competitor_rank = 1 + (unique[None, :] > competitor[:, None]).sum(1)
                    better = candidate[:, None] > competitor[None, :]
                else:
                    candidate_rank = 1 + (unique[None, :] < candidate[:, None]).sum(1)
                    competitor_rank = 1 + (unique[None, :] < competitor[:, None]).sum(1)
                    better = candidate[:, None] < competitor[None, :]
                is_new = ~np.isin(candidate, unique)
                candidate_total += candidate_rank
                competitor_total += competitor_rank[None, :] + (better & is_new[:, None])
            scores[draw_index, :, field_index] = candidate_total
            wins[draw_index, :, field_index] = candidate_total <= competitor_total.min(1)
            top10[draw_index, :, field_index] = (
                competitor_total < candidate_total[:, None]
            ).sum(1) <= 9
    return {"scores": scores, "wins": wins, "top10": top10}


def empirical_internal_probability(distributions: list[list[float]]) -> list[float]:
    probabilities = [0.0] * len(distributions)
    for candidate_index, own in enumerate(distributions):
        for score in own:
            polynomial = [1.0]
            scalar = 1.0
            for opponent_index, opponent in enumerate(distributions):
                if opponent_index == candidate_index:
                    continue
                denominator = float(len(opponent))
                worse = sum(value > score for value in opponent) / denominator
                equal = sum(value == score for value in opponent) / denominator
                if equal == 0.0:
                    scalar *= worse
                    if scalar == 0.0:
                        break
                    continue
                expanded = [0.0] * (len(polynomial) + 1)
                for ties, coefficient in enumerate(polynomial):
                    expanded[ties] += coefficient * worse
                    expanded[ties + 1] += coefficient * equal
                polynomial = expanded
            if scalar:
                probabilities[candidate_index] += scalar * sum(
                    coefficient / (ties + 1)
                    for ties, coefficient in enumerate(polynomial)
                ) / len(own)
    return probabilities


def identity(left: str, right: str) -> float:
    return sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))


def deviations(sequence: str, consensus: Mapping[str, Any]) -> int:
    return sum(
        position < len(sequence) and share >= 0.8 and sequence[position] != residue
        for raw_position, (residue, share) in consensus.items()
        for position in [int(raw_position)]
    )


def rank_candidates(
    alive: Sequence[str], observed: Mapping[str, Sequence[int]],
    table: Mapping[str, np.ndarray], sequence_index: Mapping[str, int],
    ranker: str, consensus: Mapping[str, Any],
) -> list[str]:
    rows = []
    distributions = []
    for sequence in alive:
        row = sequence_index[sequence]
        indices = list(observed[sequence])
        scores = np.median(table["scores"][indices, row, :], axis=1).astype(float).tolist()
        distributions.append(scores)
        if ranker == "field":
            wins = int(table["wins"][indices, row, :].sum())
            top10 = int(table["top10"][indices, row, :].sum())
            n = len(indices) * table["scores"].shape[2]
            probability = (wins + 0.5) / (n + 1.0)
            auxiliary = (top10 + 0.5) / (n + 1.0)
        else:
            probability = 0.0
            auxiliary = 0.0
        rows.append({
            "sequence": sequence, "probability": probability,
            "auxiliary": auxiliary, "median": statistics.median(scores),
            "deviations": deviations(sequence, consensus),
        })
    if ranker == "internal":
        probabilities = empirical_internal_probability(distributions)
        for row, probability in zip(rows, probabilities, strict=True):
            row["probability"] = probability
    rows.sort(key=lambda item: (
        0 if item["probability"] > 0 else 1,
        item["deviations"], -item["probability"], -item["auxiliary"],
        item["median"], item["sequence"],
    ))
    return [item["sequence"] for item in rows]


def family_slate(ranked: Sequence[str], keep: int) -> list[str]:
    slate = []
    for sequence in ranked:
        if len(slate) >= keep:
            break
        if all(identity(sequence, chosen) < 0.95 for chosen in slate):
            slate.append(sequence)
    taken = set(slate)
    slate.extend(sequence for sequence in ranked if sequence not in taken)
    return slate[:keep]


def fleet(ranked: Sequence[str], count: int = 10) -> list[str]:
    chosen = []
    for sequence in ranked:
        if all(identity(sequence, prior) < 0.94 for prior in chosen):
            chosen.append(sequence)
            if len(chosen) == count:
                return chosen
    remaining = [sequence for sequence in ranked if sequence not in chosen]
    while remaining and len(chosen) < count:
        pick = min(remaining, key=lambda sequence: max(
            (identity(sequence, prior) for prior in chosen), default=0.0
        ))
        if any(identity(pick, prior) >= 0.95 for prior in chosen):
            break
        chosen.append(pick)
        remaining.remove(pick)
    if len(chosen) != count:
        raise RuntimeError(f"fleet has only {len(chosen)} nonduplicate entries")
    return chosen


def select(
    cohort: Sequence[str], stages: Sequence[tuple[int, int | None]], ranker: str,
    table: Mapping[str, np.ndarray], sequence_index: Mapping[str, int],
    consensus: Mapping[str, Any],
) -> dict[str, Any]:
    alive = list(cohort)
    observed = {sequence: [0] for sequence in cohort}
    cursor = 1
    repeats = 0
    trace = []
    for rounds, retain in stages:
        indices = list(range(cursor, cursor + rounds))
        cursor += rounds
        for sequence in alive:
            observed[sequence].extend(indices)
        repeats += rounds * len(alive)
        ranked = rank_candidates(alive, observed, table, sequence_index, ranker, consensus)
        trace.append({"rounds": rounds, "alive": len(alive), "retain": retain})
        if retain is not None:
            alive = family_slate(ranked, retain)
    ranked = rank_candidates(alive, observed, table, sequence_index, ranker, consensus)
    chosen = fleet(ranked)
    return {
        "fleet": chosen, "repeat_observations": repeats,
        "model_loads": cursor - 1,
        "minimum_current_epoch_observations": min(len(observed[x]) for x in chosen),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    if list(root.glob("fresh_*/exact_all.json")):
        raise RuntimeError("fresh results exist before fleet freeze")
    cohort_payload = json.loads((root / "cohorts.json").read_text())
    cohorts = cohort_payload["cohorts"]
    sequences = json.loads((root / "development_sequences.json").read_text())["sequences"]
    sequence_index = {sequence: index for index, sequence in enumerate(sequences)}
    fields = json.loads((root / "development_fields.json").read_text())
    consensus = json.loads((root / "development_consensus.json").read_text())["positions"]
    draws = [load_draw(root / f"development_{index:02d}/exact_all.json") for index in range(1, 13)]
    if any(set(draw) != set(sequences) for draw in draws):
        raise RuntimeError("development draw sequence mismatch")
    table = outcome_table(sequences, draws, fields)
    result = {"target": TARGET, "cohorts": {}, "policies": POLICIES}
    selected_union = set()
    for epoch, cohort in cohorts.items():
        result["cohorts"][epoch] = {}
        for name, policy in POLICIES.items():
            selected = select(
                cohort, policy["stages"], policy["ranker"], table,
                sequence_index, consensus,
            )
            result["cohorts"][epoch][name] = selected
            selected_union.update(selected["fleet"])
    (root / "selected_fleets.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "fresh_sequences.json").write_text(json.dumps({
        "target": TARGET, "sequences": sorted(selected_union),
    }, indent=2, sort_keys=True) + "\n")
    print(f"FLEETS_FROZEN cohorts={len(cohorts)} union={len(selected_union)}")


if __name__ == "__main__":
    main()
