#!/usr/bin/env python3
"""Resume a deepen instead of re-scoring from scratch.

Merges every recoverable per-pass result from a per-pass directory into a
score_worker results file for a target shard, then re-emits the shard.  Run
score_worker against the outputs and it scores ONLY the missing passes
(8x -> 14x costs 6 passes, not 14).

The per-pass directory is the durable store of results JSONs (see
/root/results_perpass). Keeping per-pass results on the producer box, keyed by
sequence hash, is what makes deepening incremental.

Invocation:
  python deepen_resume.py <target_shard.json> <perpass_dir> <results_out.json> <shard_out.json>
"""
import glob
import hashlib
import json
import sys


def seq_hash(s: str) -> str:
    return hashlib.sha256(s.encode("ascii")).hexdigest()


def main() -> int:
    if len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    shard_path, perpass_dir, results_out, shard_out = sys.argv[1:5]

    shard = json.load(open(shard_path))

    # merge all per-pass results files into one hash->entry map
    merged: dict[str, dict] = {}
    for f in sorted(glob.glob(f"{perpass_dir.rstrip('/')}/*.json")):
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        for k, e in d.items():
            if isinstance(e, dict) and e.get("sequence"):
                slot = merged.setdefault(k, {})
                # update: keep sequence/epoch and union passes; a later file
                # with a full pass set wins per-pass keys
                slot.update(e)

    results: dict[str, dict] = {}
    for c in shard:
        h = seq_hash(c["sequence"])
        if h in merged:
            entry = dict(merged[h])
            entry.setdefault("sequence", c["sequence"])
            entry.setdefault("epoch", c.get("epoch"))
            results[h] = entry

    json.dump(results, open(results_out, "w"))
    json.dump(shard, open(shard_out, "w"))

    depths: dict[int, int] = {}
    for e in results.values():
        d = sum(1 for i in range(1, 41) if f"p{i}" in e)
        depths[d] = depths.get(d, 0) + 1
    print(f"targets={len(shard)} recoverable={len(results)} "
          f"depth_hist={dict(sorted(depths.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
