"""Select strong recent public nanobodies as interface-geometry teachers.

These sequences are never submission candidates: they are already public and
therefore fail novelty.  Their purpose is to reproduce known strong complexes
with retained exact-scoring work directories, then measure which configured
epitope residues competitive VHHs actually contact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nanobody_opt.simulate_round import dense_rank_sums


STRUCTURAL_METRICS = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
}


def structural_rank_sum(frame: pd.DataFrame) -> np.ndarray:
    missing = set(STRUCTURAL_METRICS) - set(frame.columns)
    if missing:
        raise ValueError(f"archive lacks structural metrics: {sorted(missing)}")
    return dense_rank_sums(frame, STRUCTURAL_METRICS, na_option="bottom")


def select_teachers(frame: pd.DataFrame, recent: int, count: int) -> pd.DataFrame:
    if recent <= 0 or count <= 0:
        raise ValueError("recent and count must be positive")
    missing = ({"sequence"} | set(STRUCTURAL_METRICS)) - set(frame.columns)
    if missing:
        raise ValueError(f"archive lacks required columns: {sorted(missing)}")

    # Select the era first: deduplicating the entire archive with keep=first can
    # silently discard a current score for a sequence that appeared previously.
    selected_era = frame.tail(min(recent, len(frame))).copy()
    selected_era["sequence"] = (
        selected_era["sequence"].astype(str).str.strip().str.upper()
    )
    selected_era = selected_era[
        selected_era.sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
    ].drop_duplicates("sequence", keep="last").reset_index(drop=True)
    if selected_era.empty:
        raise ValueError("archive has no valid protein sequences in the recent era")

    selected_era["structural_rank_sum"] = structural_rank_sum(selected_era)
    return selected_era.sort_values(
        ["structural_rank_sum", "sequence"], kind="stable"
    ).head(min(count, len(selected_era)))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recent", type=int, default=300)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    nanobody = config["protein_selection"]["nanobody"]
    target = str(nanobody["target"])
    archive = pd.read_csv(args.archive)
    teachers = select_teachers(archive, recent=args.recent, count=args.count)
    cases = []
    for row in teachers.to_dict("records"):
        cases.append({
            "sequence": row["sequence"],
            "structural_rank_sum": float(row["structural_rank_sum"]),
            "expected": {
                metric: float(row[metric]) for metric in STRUCTURAL_METRICS
            },
        })
    payload = {
        "purpose": "public structural teachers; never submit",
        "target": target,
        "structure": str(nanobody["structure"]),
        "structure_binding_site": str(nanobody["structure_binding_site"]),
        "recent_archive_rows": min(args.recent, len(archive)),
        "metrics": STRUCTURAL_METRICS,
        "cases": cases,
    }
    _atomic_json(args.output, payload)
    print(f"selected {len(cases)} structural teachers for {target}")
    for index, case in enumerate(cases, start=1):
        print(f"{index:2d}  {case['structural_rank_sum']:7.1f}  "
              f"{case['sequence'][:48]}...")


if __name__ == "__main__":
    main()
