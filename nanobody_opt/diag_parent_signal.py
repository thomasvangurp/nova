#!/usr/bin/env python3
"""Reverse-engineer the competitor's parent selection from recent winners.

For each recent winner, find its nearest public predecessor (the sequence it is
a small number of edits from), then report: the parent's own final_score, the
edit distance, and the positions/residues that changed.  This exposes the
signal in how winning miners pick parents.
"""
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127


def fetch(epoch):
    p = json.load(urllib.request.urlopen(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies", timeout=25))
    out = []
    for sub in p.get("data") or []:
        sc = sub.get("final_score")
        if sc is None:
            continue
        for nb in sub.get("nanobodies") or []:
            if nb.get("protein_name") == TARGET:
                out.append((str(nb.get("sequence") or "").strip().upper(), sc))
    return out


def main():
    import sys
    end_epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 25455
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    # build public pool (seq -> best score) from BEFORE the winners
    pool = {}
    for ep in range(end_epoch - 1, end_epoch - 1 - n_epochs, -1):
        try:
            for s, sc in fetch(ep):
                if len(s) == LEN:
                    pool[s] = min(pool.get(s, 1e9), sc)
        except Exception:
            continue
    seqs = list(pool)
    A = np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)

    # winners to dissect: the top-3 of the last few epochs
    for ep in [end_epoch, end_epoch - 1, end_epoch - 2]:
        try:
            rows = sorted([(s, sc) for s, sc in fetch(ep) if len(s) == LEN],
                          key=lambda x: x[1])
        except Exception:
            continue
        print(f"\n=== epoch {ep} top-3 winners ===")
        for s, sc in rows[:3]:
            a = np.frombuffer(s.encode(), dtype=np.uint8)
            d = (A != a).sum(axis=1)
            j = int(d.argmin())
            parent = seqs[j]
            # positions that differ from the parent
            diffs = [i for i in range(LEN) if parent[i] != s[i]]
            # parent's own rank within the pool (how mid-rank was it?)
            ranked = sorted(pool.items(), key=lambda x: x[1])
            prank = next(i for i, (x, _) in enumerate(ranked) if x == parent)
            print(f"  winner score={sc:.0f}  parent_score={pool[parent]:.0f} "
                  f"parent_rank={prank}/{len(pool)}  edits={len(diffs)} "
                  f"diffs={[(p, parent[p], '->', s[p]) for p in diffs][:6]}")


if __name__ == "__main__":
    main()
