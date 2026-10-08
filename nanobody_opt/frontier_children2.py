#!/usr/bin/env python3
"""Frontier mining, spec: 1-2 framework edits of 5-6-edit frontier parents,
using the exhaustive sweep's best small residues (Ala/Ser/Val) at up to 8 flat
framework positions.  Single + double edits; keep children >=7 edits from the
current top-50 (valid)."""
import itertools
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127
POSITIONS = [0, 2, 72, 120, 121, 122, 123, 126]  # 8 flat framework positions
RESIDUES = ["A", "S", "V"]


def main():
    import sys
    idx_path, out_path = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    idx = json.load(open(idx_path))
    top = np.array([[ord(c) for c in s] for s in idx["sequences"] if len(s) == LEN], dtype=np.uint8)

    def dmin(seq):
        a = np.frombuffer(seq.encode(), dtype=np.uint8)
        return int((top != a).sum(axis=1).min())

    pub = {}
    for ep in range(25455, 25455 - epochs, -1):
        try:
            p = json.load(urllib.request.urlopen(
                f"{API}/api/competitions/leaderboard/{ep}/nanobodies", timeout=25))
        except Exception:
            continue
        for sub in p.get("data") or []:
            if sub.get("final_score") is None:
                continue
            for nb in sub.get("nanobodies") or []:
                if nb.get("protein_name") != TARGET:
                    continue
                s = str(nb.get("sequence") or "").strip().upper()
                if len(s) == LEN:
                    pub.setdefault(s, sub["final_score"])

    parents = [s for s in pub if dmin(s) in (5, 6)]
    print(f"frontier parents (dmin 5-6): {len(parents)}", flush=True)

    children, seen = [], set(parents)
    for p in parents:
        # single edits
        for pos in POSITIONS:
            for to in RESIDUES:
                if p[pos] == to:
                    continue
                child = p[:pos] + to + p[pos + 1:]
                if child in seen:
                    continue
                seen.add(child)
                if dmin(child) >= 7:
                    children.append({"sequence": child, "epoch": 25456,
                                     "parent": p, "n_edits": 1})
        # double edits
        for pos1, pos2 in itertools.combinations(POSITIONS, 2):
            for r1 in RESIDUES:
                for r2 in RESIDUES:
                    if p[pos1] == r1 or p[pos2] == r2:
                        continue
                    s = list(p)
                    s[pos1], s[pos2] = r1, r2
                    child = "".join(s)
                    if child in seen:
                        continue
                    seen.add(child)
                    if dmin(child) >= 7:
                        children.append({"sequence": child, "epoch": 25456,
                                         "parent": p, "n_edits": 2})
    json.dump(children, open(out_path, "w"))
    n1 = sum(1 for c in children if c["n_edits"] == 1)
    n2 = sum(1 for c in children if c["n_edits"] == 2)
    print(f"emitted {len(children)} valid (single={n1} double={n2})", flush=True)


if __name__ == "__main__":
    main()
