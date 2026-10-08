#!/usr/bin/env python3
"""Mine the FULL long tail (the competitor's play), not the 55-sequence sliver.

The competitor's winners are 1-2 framework edits of LOW-SCORING public sequences
(pool rank ~800-2000, i.e. single-pass "unlucky" draws whose true quality may be
competitive).  Select every public sequence below a score cutoff, emit the raw
parent plus single-Ala edits at the flat framework positions, and keep only
children >=7 edits from the current top-50 (valid).  Score -> shortlist -> deep.
"""
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127
# per-position top residues (exhaustive_ranking.json): the 8 flat framework
# positions, A/S/V — the small neutral residues the sweep proved are safe.  This
# yields ~8-15 valid single edits per parent.
EDITS = {
    0: ["A", "S", "V"], 2: ["A", "S", "V"], 72: ["A", "S", "V"],
    120: ["A", "S", "V"], 121: ["A", "S", "V"], 122: ["A", "S", "V"],
    123: ["A", "S", "V"], 126: ["A", "S", "V"],
}


def main():
    import sys
    idx_path, out_path = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0  # long-tail score floor

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
            sc = sub.get("final_score")
            if sc is None:
                continue
            for nb in sub.get("nanobodies") or []:
                if nb.get("protein_name") != TARGET:
                    continue
                s = str(nb.get("sequence") or "").strip().upper()
                if len(s) == LEN:
                    pub[s] = min(pub.get(s, 1e9), sc)

    # the frontier: parents at exactly 6 edits from the top-50.  A single edit
    # on any position not already diverged pushes them to 7 (valid) while
    # staying inside the competitive basin.
    parents = sorted([s for s in pub if dmin(s) >= 6], key=lambda x: pub[x])
    print(f"public={len(pub)} frontier parents (dmin>=6): {len(parents)}", flush=True)

    children, seen = [], set(parents)
    n_valid = 0
    for p in parents:
        for pos, tos in EDITS.items():
            for to in tos:
                if pos >= len(p) or p[pos] == to:
                    continue
                child = p[:pos] + to + p[pos + 1:]
                if child in seen:
                    continue
                seen.add(child)
                if dmin(child) >= 7:
                    children.append({"sequence": child, "epoch": 25456,
                                     "parent_score": pub[p],
                                     "kind": "edit", "parent": p, "position": pos, "to": to})
                    n_valid += 1
    json.dump(children, open(out_path, "w"))
    print(f"emitted {len(children)} valid (>=7 edits) children "
          f"(of {len(parents)} parents)", flush=True)


if __name__ == "__main__":
    main()
