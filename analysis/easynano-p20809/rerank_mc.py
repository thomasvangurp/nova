#!/usr/bin/env python3
"""Rerank unique EasyNano candidates on shared full-model diffusion seeds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch


ROOT = Path(os.environ.get("EASYNANO_RUN_ROOT", "/root/easynano-run"))
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
EXPERIMENT = INPUT / "experiment.json"
TARGET_PDB = INPUT / "P20809.pdb"
DESIGN_RUNS = OUTPUT / "design_runs.json"
EVALUATION = OUTPUT / "monte_carlo_evaluation.json"
SUMMARY = OUTPUT / "monte_carlo_summary.json"
PROGRESS = OUTPUT / "progress.json"
EVAL_SEEDS = (0, 1, 2)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def progress(stage: str, **fields: object) -> None:
    write_json(PROGRESS, {"stage": stage, "updated_unix": time.time(), **fields})


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("sequence length mismatch")
    return sum(a != b for a, b in zip(left, right))


def main() -> int:
    from easynano.evaluate import eval_one, load_model
    from easynano.setup import setup_target_design

    experiment = json.loads(EXPERIMENT.read_text())
    design_runs = json.loads(DESIGN_RUNS.read_text())
    anchors = {item["id"]: item for item in experiment["anchors"]}
    candidates: dict[str, dict[str, dict]] = {anchor_id: {} for anchor_id in anchors}
    for run in design_runs:
        anchor = anchors[run["anchor_id"]]
        for item in run["design_candidates"]:
            sequence = item.get("full_seq")
            if not sequence or sequence == anchor["sequence"]:
                continue
            record = candidates[anchor["id"]].setdefault(sequence, {
                "sequence": sequence,
                "source_runs": [],
                "source_steps": [],
            })
            record["source_runs"].append(run["run_id"])
            record["source_steps"].append(item.get("step"))

    chain = experiment["target"]["chain"]
    epitope = experiment["target"]["easynano_epitope_token_indices"]
    setups = {}
    for anchor in anchors.values():
        setup = setup_target_design(
            pdb_path=str(TARGET_PDB), target_chain=chain,
            epitope_indices=epitope, binder_sequence=anchor["sequence"],
        )
        cdr = set(setup["cdr_indices"])
        for record in candidates[anchor["id"]].values():
            changed = [
                index for index, (left, right) in enumerate(zip(anchor["sequence"], record["sequence"]))
                if left != right
            ]
            if not changed or any(index not in cdr for index in changed):
                raise ValueError(f"invalid candidate mutation set for {anchor['id']}")
            record["changed_zero_based_positions"] = changed
        setups[anchor["id"]] = setup

    items = []
    for anchor in anchors.values():
        items.append({
            "kind": "anchor", "anchor_id": anchor["id"],
            "cohort": anchor["cohort"], "sequence": anchor["sequence"],
            "hamming_from_anchor": 0,
        })
        for record in candidates[anchor["id"]].values():
            items.append({
                "kind": "candidate", "anchor_id": anchor["id"],
                "cohort": anchor["cohort"], "sequence": record["sequence"],
                "hamming_from_anchor": hamming(anchor["sequence"], record["sequence"]),
                "changed_zero_based_positions": record["changed_zero_based_positions"],
                "source_runs": sorted(set(record["source_runs"])),
                "source_steps": sorted(set(record["source_steps"])),
            })

    model = load_model()
    expected = len(items) * len(EVAL_SEEDS)
    evaluations = []
    progress("mc-evaluating", completed=0, expected=expected)
    for item in items:
        setup = setups[item["anchor_id"]]
        for seed in EVAL_SEEDS:
            torch.manual_seed(10_000 + seed)
            np.random.seed(10_000 + seed)
            iptm, ptm, cdr_distance = eval_one(
                model, item["sequence"], setup["target_sequence"],
                len(item["sequence"]), len(setup["target_sequence"]),
                setup["cdr_indices"], epitope, setup["prior_bins"], setup["prior_mask"],
            )
            evaluations.append({
                **item, "eval_seed": seed, "iptm": iptm, "ptm": ptm,
                "cdr_to_epitope": cdr_distance,
            })
            write_json(EVALUATION, evaluations)
            progress(
                "mc-evaluating", completed=len(evaluations), expected=expected,
                last_anchor=item["anchor_id"], last_seed=seed,
            )

    grouped = {}
    for item in evaluations:
        grouped.setdefault((item["anchor_id"], item["sequence"]), []).append(item)
    ranked = []
    for (anchor_id, sequence), records in grouped.items():
        ranked.append({
            "anchor_id": anchor_id,
            "cohort": records[0]["cohort"],
            "kind": records[0]["kind"],
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "hamming_from_anchor": records[0]["hamming_from_anchor"],
            "mean_iptm": statistics.mean(item["iptm"] for item in records),
            "median_iptm": statistics.median(item["iptm"] for item in records),
            "min_iptm": min(item["iptm"] for item in records),
            "max_iptm": max(item["iptm"] for item in records),
            "mean_ptm": statistics.mean(item["ptm"] for item in records),
            "mean_cdr_to_epitope": statistics.mean(item["cdr_to_epitope"] for item in records),
            "scores_by_seed": {str(item["eval_seed"]): item["iptm"] for item in records},
            "source_runs": records[0].get("source_runs", []),
            "source_steps": records[0].get("source_steps", []),
        })

    comparisons = []
    for anchor in anchors.values():
        anchor_rank = next(
            item for item in ranked
            if item["anchor_id"] == anchor["id"] and item["kind"] == "anchor"
        )
        generated = [
            item for item in ranked
            if item["anchor_id"] == anchor["id"] and item["kind"] == "candidate"
        ]
        best = max(generated, key=lambda item: (item["mean_iptm"], item["mean_ptm"]))
        paired = {
            seed: best["scores_by_seed"][seed] - anchor_rank["scores_by_seed"][seed]
            for seed in anchor_rank["scores_by_seed"]
        }
        comparisons.append({
            "anchor_id": anchor["id"], "cohort": anchor["cohort"],
            "anchor_mean_iptm": anchor_rank["mean_iptm"],
            "best_mutated_mean_iptm": best["mean_iptm"],
            "mean_iptm_change": best["mean_iptm"] - anchor_rank["mean_iptm"],
            "paired_iptm_change_by_seed": paired,
            "best_mutated_sequence": best["sequence"],
            "best_mutated_hamming": best["hamming_from_anchor"],
            "best_mutated_mean_cdr_to_epitope": best["mean_cdr_to_epitope"],
            "unique_mutated_candidates": len(generated),
        })
    result = {
        "stage": "complete", "experiment": experiment["experiment"],
        "eval_seeds": list(EVAL_SEEDS), "unique_sequences": len(items),
        "full_evaluations": len(evaluations), "comparisons": comparisons,
        "ranking": sorted(ranked, key=lambda item: (item["cohort"], -item["mean_iptm"])),
    }
    write_json(SUMMARY, result)
    progress("complete", runs=len(design_runs), full_evaluations=len(evaluations))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        progress("failed", error_class=type(error).__name__)
        raise
