#!/usr/bin/env python3
"""Multi-pass score the rival's submissions and compare rank-sums to ours.

The validator's leaderboard rank is a single stochastic draw (BoltzGen); the
stable read is the mean of N replay rank-sums. This scores the rival's 25533
sequences N times with the fast scorer, computes the validator's dense-rank-sum
each pass (vs the archive field), and prints the per-sequence + fleet means.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEQUENCES = Path("/tmp/rival_25533.txt")
ARCHIVE = Path("/root/live/epoch25533/producer/archive.csv")
CONFIG = Path("/root/live/epoch25533/producer/config.yaml")
SCORER = Path("/root/nanobody_opt/score_boltzgen_fast_adapter.py")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

METRIC_DIR = {
    "design_iiptm": "max", "design_ptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
}


def dense_rank_sum(metrics_frame: pd.DataFrame) -> pd.Series:
    """Validator's final_rank_sum: sum of per-metric dense ranks (1 = best)."""
    ranks = pd.DataFrame(index=metrics_frame.index)
    for metric, direction in METRIC_DIR.items():
        if metric not in metrics_frame.columns:
            continue
        ascending = direction == "min"
        ranks[metric] = metrics_frame[metric].rank(
            ascending=ascending, method="dense"
        )
    return ranks.sum(axis=1)


def main() -> int:
    seqs = [l.strip().upper() for l in SEQUENCES.read_text().splitlines() if l.strip()]
    input_json = json.dumps({"sequences": [{"sequence": s} for s in seqs]})
    inp = Path("/tmp/rival_input.json")
    inp.write_text(input_json)

    archive = pd.read_csv(ARCHIVE)
    archive = archive.drop_duplicates("sequence", keep="last")
    print(f"archive field: {len(archive)} sequences", flush=True)

    per_run: list[pd.Series] = []
    for run in range(1, N + 1):
        out_dir = Path(f"/tmp/rival_score{run}")
        subprocess.run(
            [sys.executable, "-m", "nanobody_opt.score_boltzgen_waves",
             "--sequences", str(inp), "--config", str(CONFIG),
             "--target", "P20809", "--output-dir", str(out_dir),
             "--gpus", "0,1,2,3,4,5,6,7", "--wave-size", str(len(seqs)),
             "--predict-batch-size", "2", "--limit", str(len(seqs)),
             "--scorer", str(SCORER), "--capture-structures"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        payload = json.loads((out_dir / "exact_all.json").read_text())
        wave = payload["per_nanobody_components"]["0"]
        rows = {}
        for seq, comp in wave.items():
            m = comp.get("P20809", {})
            rows[seq] = {k: float(v) for k, v in m.items()
                         if k in METRIC_DIR and isinstance(v, (int, float))}
        rival = pd.DataFrame(rows).T
        field = pd.concat([archive[list(METRIC_DIR)], rival], ignore_index=True)
        ranks = dense_rank_sum(field)
        rival_ranks = ranks.iloc[len(archive):]
        per_run.append(rival_ranks)
        print(f"run {run}/{N}: rival rank-sums "
              f"{[int(x) for x in rival_ranks]}", flush=True)

    all_runs = pd.DataFrame(per_run)
    means = all_runs.mean(axis=0)
    print("\n=== rival multi-pass mean rank-sum (lower = better) ===")
    for seq, mean in zip(seqs, means):
        print(f"  {seq[:16]}... mean={mean:.1f}")
    print(f"  FLEET MEAN = {means.mean():.1f}  (n={N} passes, {len(seqs)} seqs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
