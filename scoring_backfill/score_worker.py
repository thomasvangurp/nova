#!/usr/bin/env python3
"""8x multi-pass scoring worker. Runs on a scoring box. Resumable by sequence hash."""
import argparse, json, os, subprocess, sys, hashlib, time
from collections import defaultdict
from pathlib import Path

SCORER = "/root/fastscore/pinned-530f5dcbc54d/fast_score_batch.py"
LAUNCHER = "/root/fastscore/pinned-530f5dcbc54d/release_python.sh"
CONFIG = "/root/config/config.yaml"
SYNTH_REF_DIR = "/root/fastscore/pinned-530f5dcbc54d/refdesign"
# Exact bundled assets, indexed by target AND designed nanobody-chain length.
# Filename design IDs are not file hashes; verify the bytes separately.
SYNTH_REFERENCES = {
    ("P20809", 127): (
        "0000e1b3814df2814ccd415e06b26f32204cc45753777f284216e0243098d916_P20809_input.cif",
        "30c9d30229543761aed127e4211036a8aa891b83001541b4e1fe2d33612a4405"),
    ("Q9NZQ7", 114): (
        "4b1044579d74a85c8adc7e8bd28ee8c740a1f0b3d17d9ebb27e69c0b6e5b1f98_Q9NZQ7_input.cif",
        "e146f7564ce97374b10d8ba110634a89841e588c73a589ea31f621ec0a27305e"),
    ("Q9NZQ7", 120): (
        "748276653c297786e8e1d83240604f36c34dd50ca4f5ced55c0311308948a61d_Q9NZQ7_input.cif",
        "7caec01a18a0ee5daa5eb43d6fd9e05d1a5d35f5fcce8deb527cbc2294bb52db"),
}
PASSES = 8


def synth_reference_for(target: str, length: int, *, reference_dir=None, override=None) -> str | None:
    registered = SYNTH_REFERENCES.get((target, length))
    if registered is None:
        if override:
            raise ValueError("unqualified synth reference for target/length")
        return None  # Ordinary design handles lengths without a qualified asset.
    name, digest = registered
    path = Path(override) if override else Path(reference_dir or SYNTH_REF_DIR) / name
    if not path.is_file() and not override:
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("synth reference does not match qualified target/length asset")
    return str(path)


def batches_by_length(rows, batch_size):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[len(row['sequence'])].append(row)
    for length in sorted(groups):
        for start in range(0, len(groups[length]), batch_size):
            yield length, groups[length][start:start + batch_size]


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
    ap.add_argument("--target", required=True, help="nanobody target to score against")
    ap.add_argument("--synth-reference", default=None,
                    help="fast-scorer synth reference (default: derived from --target)")
    ap.add_argument("--passes", type=int, default=PASSES,
                    help="total passes to reach; existing passes are kept and only "
                         "the missing ones are scored (deepen an 8x bank to 20x)")
    a = ap.parse_args()

    target = a.target
    PASSES = a.passes
    shard = json.load(open(a.shard))
    results_path = Path(a.results)
    results = json.load(open(results_path)) if results_path.exists() else {}
    if any(entry.get('target') != target for entry in results.values()):
        raise ValueError("existing backfill results lack matching target provenance")
    by_hash = {seq_hash(s["sequence"]): s for s in shard}
    work_root = Path(a.work_root); work_root.mkdir(parents=True, exist_ok=True)

    for pass_idx in range(1, PASSES + 1):
        pending = [s for s in shard
                   if seq_hash(s["sequence"]) not in results
                   or f"p{pass_idx}" not in results[seq_hash(s["sequence"])]]
        if not pending:
            print(f"pass {pass_idx}: all done", flush=True); continue
        print(f"pass {pass_idx}: {len(pending)} pending", flush=True)
        for batch_index, (length, batch) in enumerate(batches_by_length(pending, a.batch_size)):
            synth_ref = synth_reference_for(target, length, override=a.synth_reference)
            # Content-address the batch dir: the pending set shrinks as passes
            # complete, so a fixed `passN_batchK` name would collide with a
            # stale out.json from an earlier run (different children) and skip
            # scoring entirely -- leaving passes silently unfinished.
            batch_key = hashlib.sha256(json.dumps([target, length,
                [s["sequence"] for s in batch]]).encode()).hexdigest()[:12]
            bdir = work_root / f"pass{pass_idx}_batch{batch_index}_{batch_key}"
            bdir.mkdir(parents=True, exist_ok=True)
            out = bdir / "out.json"
            if not out.exists():
                (bdir / "in.json").write_text(json.dumps([{"sequence": s["sequence"]} for s in batch]))
                cmd = [LAUNCHER, SCORER, "--sequences", str(bdir / "in.json"),
                       "--config", CONFIG, "--target", target, "--gpus", a.gpus,
                       "--shards-per-gpu", "1", "--batch-size", "24",
                       "--shard-root", str(bdir), "--out", str(out)]
                if synth_ref:
                    cmd.extend(["--synth-reference", synth_ref])
                t0 = time.time()
                import os as _os
                _env = dict(_os.environ); _env["HF_HUB_OFFLINE"] = "1"; _env["HF_HUB_DISABLE_TELEMETRY"] = "1"
                r = subprocess.run(cmd, cwd="/root", env=_env)
                dt = time.time() - t0
                if r.returncode != 0:
                    print(f"  pass {pass_idx} batch {batch_index} FAILED rc={r.returncode} {dt:.0f}s", flush=True)
                    continue
                print(f"  pass {pass_idx} batch {batch_index}: {len(batch)} seq {dt:.0f}s", flush=True)
            if not out.exists():
                continue
            payload = json.load(open(out))
            if payload.get('target', target) != target:
                raise ValueError("backfill scorer payload target mismatch")
            for by_seq in payload.get("per_nanobody_components", {}).values():
                for seq, by_target in by_seq.items():
                    seq = seq.strip().upper(); h = seq_hash(seq)
                    if h not in by_hash:
                        continue
                    entry = results.setdefault(h, {"sequence": seq, "target": target,
                        "epoch": by_hash[h]["epoch"], "node_id": by_hash[h].get("node_id")})
                    entry[f"p{pass_idx}"] = by_target[target]
            tmp = results_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(results)); tmp.replace(results_path)

    complete = sum(1 for e in results.values() if all(f"p{i}" in e for i in range(1, PASSES + 1)))
    print(f"RESULT complete={complete}/{len(shard)}", flush=True)

if __name__ == "__main__":
    main()
