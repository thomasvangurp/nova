#!/usr/bin/env python3
"""Are there forgotten basins in the historical submissions?

The field converged to one basin only recently.  Fetch a wide epoch window and
look for sequences >=7 edits from the CURRENT top-50 that nonetheless scored
competitively in their own epoch -- i.e. other basins the field abandoned.
"""
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127


def dmin(seq, T):
    a = np.frombuffer(seq.encode(), dtype=np.uint8)
    return int((T != a).sum(axis=1).min())


def main():
    import sys
    idx = json.load(open(sys.argv[1]))
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    top = np.array([[ord(c) for c in s] for s in idx["sequences"] if len(s) == LEN], dtype=np.uint8)

    pub = {}  # seq -> best (min) final_score seen
    for ep in range(25455, 25455 - epochs, -1):
        try:
            p = json.load(urllib.request.urlopen(
                f"{API}/api/competitions/leaderboard/{ep}/nanobodies", timeout=25))
        except Exception:
            continue
        for sub in p.get("data") or []:
            sc = sub.get("final_score")
            if sc is None:
                continue
            for nb in sub.get("nanobodies") or []:
                if nb.get("protein_name") != TARGET:
                    continue
                s = str(nb.get("sequence") or "").strip().upper()
                if len(s) == LEN:
                    pub[s] = min(pub.get(s, 1e9), sc)

    novel = [(sc, s) for s, sc in pub.items() if dmin(s, top) >= 7]
    novel.sort()
    print(f"scanned {epochs} epochs: {len(pub)} unique seqs, {len(novel)} are >=7 edits from current top-50")
    print("\nbest novel-basin sequences (min final_score, lower=better):")
    for sc, s in novel[:25]:
        print(f"  {sc:6.1f}  edits={dmin(s, top)}  {s[:40]}")
    # also: histogram of edit distance
    dists = [dmin(s, top) for s in pub]
    from collections import Counter
    c = Counter(dists)
    print("\nedit-distance histogram:", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
