#!/usr/bin/env python3
"""Work-stealing tail for the children scoring (additive, idempotent).

A worker that has finished its statically-assigned shard calls this instead of
exiting. It repeatedly:

  1. Reads the shared "remaining work" ledger /root/children_remaining.json
     (a list of {sequence, hash} the merge loop refreshes every 5 min),
  2. Drops children it has already fully scored locally (its results JSON),
  3. Pulls one batch (default 256) and 8x-scores it via the same fast_score
     path as score_worker.py,
  4. Loops until the ledger is empty or no new work appears for a quiet window.

Idempotent by sequence hash: re-scoring a child another box finished in the
meantime is a no-op at the DB layer (observation_id upsert in
ingest_backfill.py), and the local results merge keeps the first-complete copy.

The ledger is refreshed and pushed to every box by the 5-min merge loop on the
orchestrator (the boxes cannot SSH to production, so the orchestrator is the
distributor). See merge_children.sh -> children_remaining.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

LEDGER = Path("/root/children_remaining.json")
RESULTS = Path("/root/results_children9")
WORK_ROOT = Path("/root/bgen_shards_children9")
SCORER = "/root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py"
LAUNCHER = "/root/fastscore/pinned-530f5dcbc54d/release_python.sh"
CONFIG = "/root/config/config.yaml"
SYNTH_REF = "/root/fastscore/pinned-530f5dcbc54d/refdesign/0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif"
TARGET = "P20809"
PASSES = 8


def seq_hash(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def run_pass(batch: list[str], gpus: str, batch_dir: Path) -> dict[str, dict]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "in.json").write_text(json.dumps([{"sequence": s} for s in batch]))
    out = batch_dir / "out.json"
    import os
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    cmd = [LAUNCHER, SCORER, "--sequences", str(batch_dir / "in.json"),
           "--config", CONFIG, "--target", TARGET, "--gpus", gpus,
           "--shards-per-gpu", "1", "--batch-size", "24",
           "--shard-root", str(batch_dir), "--out", str(out),
           "--synth-reference", SYNTH_REF]
    r = subprocess.run(cmd, cwd="/root", env=env)
    if r.returncode != 0:
        raise RuntimeError(f"fast_score rc={r.returncode}")
    payload = json.loads(out.read_text())
    metrics = {}
    for by_seq in payload.get("per_nanobody_components", {}).values():
        for seq, by_target in by_seq.items():
            metrics[seq.strip().upper()] = by_target[TARGET]
    return metrics


def steal(gpus: str, batch_size: int, quiet_polls: int) -> int:
    done = 0
    quiet = 0
    while True:
        if not LEDGER.exists():
            quiet += 1
            if quiet >= quiet_polls:
                break
            time.sleep(30)
            continue
        remaining = json.loads(LEDGER.read_text())
        if not isinstance(remaining, list) or not remaining:
            break  # ledger empty -> nothing left to steal
        # drop children already fully scored by this box
        local = {}
        for p in RESULTS.glob("*.json"):
            try:
                local.update(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        # A stealer scores ONLY the passes a child is still missing, so a
        # partially-scored child (e.g. 6 passes) is not re-8x-scored.
        candidates = []          # (sequence, missing_pass_indices)
        for item in remaining:
            seq = item["sequence"] if isinstance(item, dict) else item
            h = seq_hash(seq)
            missing = item.get("missing", list(range(1, PASSES + 1))) if isinstance(item, dict) else list(range(1, PASSES + 1))
            entry = local.get(h)
            if entry:
                missing = [i for i in missing if f"p{i}" not in entry]
            if not missing:
                continue
            candidates.append((seq, missing))
        if not candidates:
            break
        batch = candidates[:batch_size]
        batch_dir = WORK_ROOT / f"steal_{int(time.time())}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS / "steal.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        # Group the batch by missing pass so one scorer run covers many
        # sequences. The folding-model load dominates (~25s), so a single-seq
        # run wastes the other 7 GPUs and stalls the whole steal tail.
        by_pass: dict[int, list[str]] = {}
        for seq, missing in batch:
            for pass_i in missing:
                by_pass.setdefault(pass_i, []).append(seq)
        for pass_i, seqs in by_pass.items():
            metrics = run_pass(seqs, gpus, batch_dir / f"p{pass_i}")
            for seq in seqs:
                h = seq_hash(seq)
                entry = results.setdefault(h, {"sequence": seq, "epoch": 25423})
                entry[f"p{pass_i}"] = metrics[seq.strip().upper()]
        tmp = results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(results_path)
        done += len(batch)
        quiet = 0
        print(f"STEAL scored {len(batch)} children (missing passes only) ({done} total)", flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--quiet-polls", type=int, default=10,
                    help="stop after this many empty/no-new-work polls")
    args = ap.parse_args()
    n = steal(args.gpus, args.batch_size, args.quiet_polls)
    print(f"WORK_STEAL_DONE stolen={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
