#!/usr/bin/env python3
"""Refresh the work-stealing ledger with per-child missing-pass indices.

A child is in the ledger iff it has fewer than 8 passes across the merged
per-box results. Each entry carries `missing` = the 1-based pass indices that
still need scoring, so a stealer scores ONLY those passes — a child at 6 passes
is never re-8x-scored from scratch.
"""
import hashlib, json, glob
from pathlib import Path

REPO = Path(__file__).resolve().parent
CHILDREN = REPO / "children_queue.json"
RESULTS_GLOB = str(REPO / "results_children9" / "*.json")
OUT = REPO / "children_remaining.json"
PASSES = 8


def seq_hash(s): return hashlib.sha256(s.encode()).hexdigest()


def main():
    children = json.loads(CHILDREN.read_text())
    merged = {}
    for p in glob.glob(RESULTS_GLOB):
        try:
            merged.update(json.load(open(p)))
        except (OSError, json.JSONDecodeError):
            continue
    remaining = []
    for it in children:
        seq = it["sequence"]
        h = seq_hash(seq)
        entry = merged.get(h)
        if not entry:
            missing = list(range(1, PASSES + 1))
        else:
            missing = [i for i in range(1, PASSES + 1) if f"p{i}" not in entry]
        if missing:
            remaining.append({"sequence": seq, "hash": h, "missing": missing})
    OUT.write_text(json.dumps(remaining))
    # sanity: report how many are fully unscored vs partially scored
    unscored = sum(1 for r in remaining if len(r["missing"]) == PASSES)
    partial = len(remaining) - unscored
    print(f"remaining={len(remaining)}/{len(children)} (unscored={unscored}, partial={partial})")


if __name__ == "__main__":
    main()
