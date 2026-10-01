#!/usr/bin/env python3
"""Freeze the confirmation shortlist using discovery passes only."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = "P20809"
METRICS = {
    "design_iiptm": "max", "design_ptm": "max",
    "design_to_target_iptm": "max", "min_design_to_target_pae": "min",
    "interaction_pae": "min", "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max", "delta_sasa_refolded": "max",
    "liability_score": "min", "liability_num_violations": "min",
}


def components(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong scorer protocol in {path}")
    result = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            result[sequence] = {key: float(metrics[key]) for key in METRICS}
    if len(result) != int(payload["n"]):
        raise RuntimeError(f"result count mismatch in {path}")
    return result


def dense_rank_sums(rows: list[dict[str, float]]) -> list[int]:
    totals = [0] * len(rows)
    for metric, mode in METRICS.items():
        values = sorted({row[metric] for row in rows}, reverse=mode == "max")
        ranks = {value: index + 1 for index, value in enumerate(values)}
        for index, row in enumerate(rows):
            totals[index] += ranks[row[metric]]
    return totals


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text())
    expected = [case["sequence"] for case in manifest["cases"]]
    scores: dict[str, list[int]] = defaultdict(list)
    for index in range(1, 5):
        observed = components(HERE / f"discovery_{index:02d}" / "exact_all.json")
        if set(observed) != set(expected):
            raise RuntimeError(f"discovery {index} cohort mismatch")
        totals = dense_rank_sums([observed[sequence] for sequence in expected])
        for sequence, total in zip(expected, totals):
            scores[sequence].append(total)
    mean_score = {sequence: statistics.fmean(values) for sequence, values in scores.items()}
    by_sequence = {case["sequence"]: case for case in manifest["cases"]}
    anchors = [sequence for sequence in expected if by_sequence[sequence]["control_anchor"]]
    selected = list(anchors)
    trace = []
    for arm in ("current_production", "ours_hamming2", "rival_hamming2"):
        candidates = [
            sequence for sequence in expected
            if arm in by_sequence[sequence]["arms"]
            and sequence not in anchors
            and not by_sequence[sequence].get("known_public", False)
        ]
        ranked = sorted(candidates, key=lambda sequence: (mean_score[sequence], sequence))
        arm_selected: list[str] = []
        if arm != "current_production":
            groups = sorted({
                group for sequence in ranked
                for group in by_sequence[sequence]["arm_groups"].get(arm, [])
            })
            for group in groups:
                choice = next(
                    (sequence for sequence in ranked
                     if group in by_sequence[sequence]["arm_groups"].get(arm, [])
                     and sequence not in arm_selected),
                    None,
                )
                if choice is not None:
                    arm_selected.append(choice)
        for sequence in ranked:
            if len(arm_selected) == 24:
                break
            if sequence not in arm_selected:
                arm_selected.append(sequence)
        arm_selected = arm_selected[:24]
        if len(arm_selected) != 24:
            raise RuntimeError(f"{arm} supplied only {len(arm_selected)} finalists")
        for sequence in arm_selected:
            if sequence not in selected:
                selected.append(sequence)
        trace.append({
            "arm": arm,
            "selected": arm_selected,
            "discovery_mean_rank_sums": {
                sequence: mean_score[sequence] for sequence in arm_selected
            },
        })
    payload = {
        "target": TARGET,
        "selection_passes": [1, 2, 3, 4],
        "selection_rule": (
            "all anchors plus 24 proposals per arm by mean complete ten-metric "
            "dense-rank sum; known public sequences are ineligible as proposals; "
            "closure arms reserve the best member of every basin"
        ),
        "sequence_count": len(selected),
        "sequences": selected,
        "trace": trace,
    }
    (HERE / "shortlist_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "shortlist_sequences.json").write_text(json.dumps({
        "target": TARGET, "sequences": selected,
    }, indent=2) + "\n")
    print(json.dumps({"shortlist": len(selected), "anchors": len(anchors)}))


if __name__ == "__main__":
    main()
