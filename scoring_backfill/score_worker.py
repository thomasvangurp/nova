#!/usr/bin/env python3
"""8x multi-pass scoring worker. Runs on a scoring box. Resumable by sequence hash."""
import argparse, json, os, subprocess, sys, hashlib, time
from pathlib import Path

SCORER = "/root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py"
LAUNCHER = "/root/fastscore/pinned-530f5dcbc54d/release_python.sh"
CONFIG = "/root/config/config.yaml"
SYNTH_REF = "/root/fastscore/pinned-530f5dcbc54d/refdesign/0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif"
TARGET = "P20809"
PASSES = 8

def seq_hash(s):
    return hashlib.sha256(s.encode("ascii")).hexdigest()

def main():
    global PASSES
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--work-root", default="/root/bgen_shards")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--passes", type=int, default=PASSES,
                    help="total passes to reach; existing passes are kept and only "
                         "the missing ones are scored (deepen an 8x bank to 20x)")
    a = ap.parse_args()

    PASSES = a.passes
    shard = json.load(open(a.shard))
    results_path = Path(a.results)
    results = json.load(open(results_path)) if results_path.exists() else {}
    by_hash = {seq_hash(s["sequence"]): s for s in shard}
    work_root = Path(a.work_root); work_root.mkdir(parents=True, exist_ok=True)

    for pass_idx in range(1, PASSES + 1):
        pending = [s for s in shard
                   if seq_hash(s["sequence"]) not in results
                   or f"p{pass_idx}" not in results[seq_hash(s["sequence"])]]
        if not pending:
            print(f"pass {pass_idx}: all done", flush=True); continue
        print(f"pass {pass_idx}: {len(pending)} pending", flush=True)
        for start in range(0, len(pending), a.batch_size):
            batch = pending[start:start + a.batch_size]
            # Content-address the batch dir: the pending set shrinks as passes
            # complete, so a fixed `passN_batchK` name would collide with a
            # stale out.json from an earlier run (different children) and skip
            # scoring entirely -- leaving passes silently unfinished.
            batch_key = hashlib.sha256(json.dumps([s["sequence"] for s in batch]).encode()).hexdigest()[:12]
            bdir = work_root / f"pass{pass_idx}_batch{start // a.batch_size}_{batch_key}"
            bdir.mkdir(parents=True, exist_ok=True)
            out = bdir / "out.json"
            if not out.exists():
                (bdir / "in.json").write_text(json.dumps([{"sequence": s["sequence"]} for s in batch]))
                cmd = [LAUNCHER, SCORER, "--sequences", str(bdir / "in.json"),
                       "--config", CONFIG, "--target", TARGET, "--gpus", a.gpus,
                       "--shards-per-gpu", "1", "--batch-size", "24",
                       "--shard-root", str(bdir), "--out", str(out),
                       "--synth-reference", SYNTH_REF]
                t0 = time.time()
                import os as _os
                _env = dict(_os.environ); _env["HF_HUB_OFFLINE"] = "1"; _env["HF_HUB_DISABLE_TELEMETRY"] = "1"
                r = subprocess.run(cmd, cwd="/root", env=_env)
                dt = time.time() - t0
                if r.returncode != 0:
                    print(f"  pass {pass_idx} batch {start // a.batch_size} FAILED rc={r.returncode} {dt:.0f}s", flush=True)
                    continue
                print(f"  pass {pass_idx} batch {start // a.batch_size}: {len(batch)} seq {dt:.0f}s", flush=True)
            if not out.exists():
                continue
            payload = json.load(open(out))
            for by_seq in payload.get("per_nanobody_components", {}).values():
                for seq, by_target in by_seq.items():
                    seq = seq.strip().upper(); h = seq_hash(seq)
                    if h not in by_hash:
                        continue
                    entry = results.setdefault(h, {"sequence": seq, "epoch": by_hash[h]["epoch"], "node_id": by_hash[h].get("node_id")})
                    entry[f"p{pass_idx}"] = by_target[TARGET]
            tmp = results_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(results)); tmp.replace(results_path)

    complete = sum(1 for e in results.values() if all(f"p{i}" in e for i in range(1, PASSES + 1)))
    print(f"RESULT complete={complete}/{len(shard)}", flush=True)

if __name__ == "__main__":
    main()
