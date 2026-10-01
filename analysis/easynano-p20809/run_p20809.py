#!/usr/bin/env python3
"""Prepare and run the isolated EasyNano P20809 anchor experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(os.environ.get("EASYNANO_RUN_ROOT", "/root/easynano-run"))
INPUT = ROOT / "input"
OUTPUT = ROOT / "output"
EXPERIMENT = INPUT / "experiment.json"
SOURCE_CIF = INPUT / "P20809_input.cif"
TARGET_PDB = INPUT / "P20809.pdb"
PROGRESS = OUTPUT / "progress.json"
SUMMARY = OUTPUT / "summary.json"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _progress(stage: str, **fields: object) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(PROGRESS, {"stage": stage, "updated_unix": time.time(), **fields})


def _load_experiment() -> dict:
    data = json.loads(EXPERIMENT.read_text())
    target = data["target"]
    residue_ids = target["configured_epitope_residue_ids"]
    token_indices = target["easynano_epitope_token_indices"]
    start = target["canonical_residue_range"][0]
    expected = [residue_id - start for residue_id in residue_ids]
    if token_indices != expected:
        raise ValueError(f"epitope mapping mismatch: {token_indices} != {expected}")
    if len(data["anchors"]) != 2 or {a["cohort"] for a in data["anchors"]} != {"ours", "competitor"}:
        raise ValueError("experiment must contain one anchor from each cohort")
    return data


def prepare() -> int:
    import gemmi

    data = _load_experiment()
    if not SOURCE_CIF.is_file():
        raise FileNotFoundError(SOURCE_CIF)
    structure = gemmi.read_structure(str(SOURCE_CIF))
    structure.write_pdb(str(TARGET_PDB))
    model = structure[0]
    chain = model[data["target"]["chain"]]
    residues = [
        residue for residue in chain
        if any(atom.name == "CA" for atom in residue)
    ]
    residue_ids = [residue.seqid.num for residue in residues]
    expected_range = list(range(2, 170))
    if residue_ids != expected_range:
        raise ValueError(f"target residue IDs differ: {residue_ids[:3]}..{residue_ids[-3:]}")

    from easynano.setup import load_target_chain, setup_target_design

    sequence, _ = load_target_chain(str(TARGET_PDB), data["target"]["chain"])
    if len(sequence) != 168:
        raise ValueError(f"unexpected target length {len(sequence)}")
    setups = []
    for anchor in data["anchors"]:
        setup = setup_target_design(
            pdb_path=str(TARGET_PDB),
            target_chain=data["target"]["chain"],
            epitope_indices=data["target"]["easynano_epitope_token_indices"],
            binder_sequence=anchor["sequence"],
        )
        if not setup["cdr_indices"]:
            raise RuntimeError(f"no CDR positions detected for {anchor['id']}")
        setups.append({
            "anchor": anchor["id"],
            "binder_length": len(setup["binder_full_sequence"]),
            "cdr_indices": setup["cdr_indices"],
            "cdr_count": len(setup["cdr_indices"]),
        })
    record = {
        "stage": "prepared",
        "target_length": len(sequence),
        "target_chain": data["target"]["chain"],
        "canonical_residue_range": [residue_ids[0], residue_ids[-1]],
        "epitope_residue_ids": data["target"]["configured_epitope_residue_ids"],
        "epitope_token_indices": data["target"]["easynano_epitope_token_indices"],
        "source_cif_sha256": _sha256(SOURCE_CIF),
        "target_pdb_sha256": _sha256(TARGET_PDB),
        "setups": setups,
    }
    _write_json(OUTPUT / "prepared.json", record)
    _progress("prepared", target_length=len(sequence), anchors=len(setups))
    print(json.dumps(record, sort_keys=True))
    return 0


def _hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("sequence length mismatch")
    return sum(a != b for a, b in zip(left, right))


def run() -> int:
    import numpy as np
    import torch
    from easynano.design import run_design
    from easynano.evaluate import eval_one, load_model
    from easynano.setup import setup_target_design

    data = _load_experiment()
    if not TARGET_PDB.is_file():
        prepare()
    epi = data["target"]["easynano_epitope_token_indices"]
    chain = data["target"]["chain"]
    steps = data["easynano"]["steps"]
    seeds = data["easynano"]["seeds"]
    top_n = data["easynano"]["full_evaluation_candidates_per_seed"]
    runs = []
    _progress("designing", completed=0, expected=len(data["anchors"]) * len(seeds))
    for anchor in data["anchors"]:
        for seed in seeds:
            run_id = f"{anchor['id']}-seed{seed}"
            out_dir = OUTPUT / "design" / run_id
            history, snapshots, best = run_design(
                pdb_path=str(TARGET_PDB),
                target_chain=chain,
                epitope_indices=epi,
                init_seq=anchor["sequence"],
                seed=seed,
                steps=steps,
                out_dir=str(out_dir),
                snapshot_path=str(out_dir / "snapshots.json"),
            )
            candidates = sorted(snapshots, key=lambda item: item.get("inter", float("inf")))[:top_n]
            if best["best_seq"] and all(item.get("full_seq") != best["best_seq"] for item in candidates):
                matching = next((item for item in snapshots if item.get("full_seq") == best["best_seq"]), None)
                candidates.append(matching or {"step": best["best_step"], "full_seq": best["best_seq"]})
            runs.append({
                "run_id": run_id,
                "anchor_id": anchor["id"],
                "cohort": anchor["cohort"],
                "seed": seed,
                "best_info": best,
                "design_candidates": candidates,
                "history_records": len(history),
            })
            _write_json(OUTPUT / "design_runs.json", runs)
            _progress("designing", completed=len(runs), expected=len(data["anchors"]) * len(seeds), last_run=run_id)

    _progress("evaluating", completed=0)
    model = load_model()
    evaluated = []
    total_eval = len(data["anchors"]) + sum(len(run["design_candidates"]) for run in runs)
    completed = 0
    anchor_cache = {}
    for anchor in data["anchors"]:
        setup = setup_target_design(
            pdb_path=str(TARGET_PDB), target_chain=chain,
            epitope_indices=epi, binder_sequence=anchor["sequence"],
        )
        anchor_cache[anchor["id"]] = setup
        torch.manual_seed(10_000)
        np.random.seed(10_000)
        iptm, ptm, cdr_epi = eval_one(
            model, anchor["sequence"], setup["target_sequence"],
            len(anchor["sequence"]), len(setup["target_sequence"]),
            setup["cdr_indices"], epi, setup["prior_bins"], setup["prior_mask"],
        )
        evaluated.append({
            "kind": "anchor", "anchor_id": anchor["id"], "cohort": anchor["cohort"],
            "sequence": anchor["sequence"], "iptm": iptm, "ptm": ptm,
            "cdr_to_epitope": cdr_epi, "hamming_from_anchor": 0,
        })
        completed += 1
        _progress("evaluating", completed=completed, expected=total_eval, last_run=anchor["id"])

    anchors = {anchor["id"]: anchor for anchor in data["anchors"]}
    seen = set()
    for run_record in runs:
        anchor = anchors[run_record["anchor_id"]]
        setup = anchor_cache[anchor["id"]]
        cdr_set = set(setup["cdr_indices"])
        for candidate in run_record["design_candidates"]:
            sequence = candidate.get("full_seq")
            key = (run_record["run_id"], sequence)
            if not sequence or key in seen:
                continue
            seen.add(key)
            changed = [index for index, (a, b) in enumerate(zip(anchor["sequence"], sequence)) if a != b]
            if any(index not in cdr_set for index in changed):
                raise ValueError(f"{run_record['run_id']} mutated framework positions")
            torch.manual_seed(10_000 + run_record["seed"])
            np.random.seed(10_000 + run_record["seed"])
            iptm, ptm, cdr_epi = eval_one(
                model, sequence, setup["target_sequence"],
                len(sequence), len(setup["target_sequence"]),
                setup["cdr_indices"], epi, setup["prior_bins"], setup["prior_mask"],
            )
            evaluated.append({
                "kind": "candidate", "run_id": run_record["run_id"],
                "anchor_id": anchor["id"], "cohort": anchor["cohort"],
                "seed": run_record["seed"], "step": candidate.get("step"),
                "sequence": sequence, "iptm": iptm, "ptm": ptm,
                "cdr_to_epitope": cdr_epi,
                "hamming_from_anchor": _hamming(anchor["sequence"], sequence),
                "changed_zero_based_positions": changed,
            })
            completed += 1
            _write_json(OUTPUT / "full_evaluation.json", evaluated)
            _progress("evaluating", completed=completed, expected=total_eval, last_run=run_record["run_id"])

    comparisons = []
    for anchor in data["anchors"]:
        baseline = next(item for item in evaluated if item["kind"] == "anchor" and item["anchor_id"] == anchor["id"])
        generated = [item for item in evaluated if item["kind"] == "candidate" and item["anchor_id"] == anchor["id"]]
        best = max(generated, key=lambda item: item["iptm"])
        comparisons.append({
            "anchor_id": anchor["id"], "cohort": anchor["cohort"],
            "anchor_iptm": baseline["iptm"], "best_generated_iptm": best["iptm"],
            "iptm_change": best["iptm"] - baseline["iptm"],
            "best_generated_sequence": best["sequence"],
            "best_seed": best["seed"], "best_step": best["step"],
            "best_hamming_from_anchor": best["hamming_from_anchor"],
            "evaluated_candidates": len(generated),
        })
    summary = {
        "stage": "complete", "experiment": data["experiment"],
        "target": data["target"], "anchors": data["anchors"],
        "runs": len(runs), "full_evaluations": len(evaluated),
        "comparisons": comparisons,
    }
    _write_json(SUMMARY, summary)
    _progress("complete", runs=len(runs), full_evaluations=len(evaluated))
    print(json.dumps(summary, sort_keys=True))
    return 0


def status() -> int:
    if PROGRESS.is_file():
        print(PROGRESS.read_text().strip())
    else:
        print(json.dumps({"stage": "not_started"}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "run", "status"))
    operation = parser.parse_args().operation
    return {"prepare": prepare, "run": run, "status": status}[operation]()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        _progress("failed", error_class=type(error).__name__)
        raise
