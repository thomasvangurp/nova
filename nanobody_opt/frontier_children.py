#!/usr/bin/env python3
"""Frontier strategy: proven mutations on DIVERSE parents.

Apply the exhaustive-sweep's measured-best framework residues (Ala at the flat
surface positions) to every diverse parent (>=7 edits from the current top-50,
i.e. other lineages), plus the raw parent as a baseline.  Framework positions
are lineage-independent, so the sweep's "Ala is neutral/best" finding should
transfer -- this is the hypothesis the run tests against the legacy ~174 floor.

Emits a score shard: raw parents (baseline) + one-Ala-edit variants at the flat
framework positions, all >=7 edits from top-50 (CDR3 stays the parent's, which is
already novel because the parent is >=7 edits away).
"""
import json
import urllib.request

import numpy as np

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
LEN = 127
# flat surface framework positions -> measured-best residue (exhaustive sweep)
FLAT = {0: "A", 2: "A", 120: "A", 121: "A", 123: "A", 126: "A", 122: "V", 72: "A"}


def main():
    import sys
    idx_path, out_path = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    idx = json.load(open(idx_path))
    top = np.array([[ord(c) for c in s] for s in idx["sequences"] if len(s) == LEN], dtype=np.uint8)

    def dmin(seq):
        a = np.frombuffer(seq.encode(), dtype=np.uint8)
        return int((top != a).sum(axis=1).min())

    # collect diverse parents: >=7 edits from current top-50, best score
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
                if len(s) == LEN and dmin(s) >= 7:
                    pub[s] = min(pub.get(s, 1e9), sc)

    parents = [s for s, _ in sorted(pub.items(), key=lambda x: x[1])]
    print(f"diverse parents (>=7 edits): {len(parents)}", flush=True)

    children, seen = [], set(parents)
    for p in parents:
        # baseline raw parent
        children.append({"sequence": p, "epoch": 25456, "kind": "parent"})
        # single-Ala framework edits (proven pattern)
        for pos, to in FLAT.items():
            if pos >= len(p) or p[pos] == to:
                continue
            child = p[:pos] + to + p[pos + 1:]
            if child in seen:
                continue
            seen.add(child)
            if dmin(child) >= 7:
                children.append({"sequence": child, "epoch": 25456,
                                 "kind": "ala", "parent": p, "position": pos})
    json.dump(children, open(out_path, "w"))
    n_ala = sum(1 for c in children if c["kind"] == "ala")
    print(f"emitted {len(children)} (parents={len(parents)} ala-edits={n_ala})", flush=True)


if __name__ == "__main__":
    main()
