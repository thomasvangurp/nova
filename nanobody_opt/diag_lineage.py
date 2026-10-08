#!/usr/bin/env python3
"""Is the competitor's signal the LINEAGE?  For each winner, find its parent
(nearest prior public sequence) then the parent's own parent ("grandparent"),
and check whether the grandparent is good (low single-pass score).  If winners
descend 1-2 edits from GOOD grandparents but pass through a bad-scoring parent,
the signal is: pick parents near good sequences (parent's low score is noise)."""
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
                out.append((str(nb.get("sequence") or "").strip().upper(),
                            sc, sub.get("coldkey", "")))
    return out


def main():
    # public pool (60 epochs) with epoch order
    pool = {}
    for ep in range(25454, 25454 - 60, -1):
        try:
            for s, sc, _ in fetch(ep):
                if len(s) == LEN:
                    pool.setdefault(s, sc)
        except Exception:
            continue
    seqs = list(pool)
    A = np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)

    def nearest(s, exclude):
        a = np.frombuffer(s.encode(), dtype=np.uint8)
        d = (A != a).sum(axis=1)
        best = None
        for j in np.argsort(d):
            if seqs[j] in exclude:
                continue
            best = (int(d[j]), seqs[j])
            break
        return best

    # 5GhREcsK winners in 25455
    winners = [(s, sc) for s, sc, ck in fetch(25455) if "5GhREcsK" in ck and len(s) == LEN]
    winners.sort(key=lambda x: x[1])
    print("winner -> parent -> grandparent (lower = better):")
    for s, wsc in winners[:8]:
        d1, parent = nearest(s, {s})
        d2, grand = nearest(parent, {s, parent})
        print(f"  winner={wsc:5.0f} --{d1}edit--> parent={pool[parent]:5.0f} "
              f"--{d2}edit--> grandparent={pool[grand]:5.0f}")


if __name__ == "__main__":
    main()
