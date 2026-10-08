#!/usr/bin/env python3
"""Test additivity: does stacking ~7 near-optimal (neutral) mutations preserve
quality, or do they epistatically collapse?

For each parent, pick the best-scoring residue that DIFFERS from the parent at
each of 6 flat framework positions, plus one CDR3 edit (novel clonotype), for
>=7 actual edits.  Deep-score the result: ~54-100 => additive (we can walk out
of the basin); ~174 => epistatic (neutral network broken).

Emits a deep-score shard for score_worker.py.
"""
import json
import sqlite3
import sys

import numpy as np

DB = "/root/artifacts/nanobody_mutations.sqlite"
LEN = 127

# per-position residue preference, best-first (from exhaustive_ranking.json,
# lower rank-sum = better). Framework-flat positions + CDR3 positions.
PREFS = {
    0:   ["A", "D", "Q", "G", "N", "K"],
    2:   ["A", "M", "V", "E", "S", "K", "L", "T", "I", "R", "Q"],
    120: ["A", "T", "S", "V", "P", "C"],
    121: ["A", "P", "S", "V", "G", "T"],
    123: ["A", "S", "P", "G", "V", "Q"],
    126: ["A", "P", "G", "E", "S", "T"],
    122: ["V", "A", "T", "I", "L", "S"],
    72:  ["A", "T", "D", "V", "Q", "G"],
    113: ["H", "M", "V", "Q", "S", "Y"],   # CDR3
    115: ["A", "S", "T", "E", "R", "H"],   # CDR3
    110: ["E", "D", "Q", "K", "P", "S"],   # CDR3
}
FRAMEWORK = [0, 2, 120, 121, 123, 126, 122, 72]
CDR3 = [113, 115, 110]


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    idx_path, out_path = sys.argv[1], sys.argv[2]
    n_parents = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    idx = json.load(open(idx_path))
    top50 = [s for s in idx.get("sequences", []) if len(s) == LEN]
    T = np.array([[ord(c) for c in s] for s in top50], dtype=np.uint8)

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    acc = {}
    for seq, sc in conn.execute(
        """SELECT n.sequence, o.score FROM nodes n JOIN observations o
           ON o.node_id=n.id AND o.score_kind='local_validator_replay_rank_sum'
           WHERE n.length=127"""):
        acc.setdefault(seq, []).append(sc)
    conn.close()
    parents = [s for s, _ in sorted(
        acc.items(), key=lambda x: (sum(x[1]) / len(x[1]), x[0]))[:n_parents]]

    def dmin(seq):
        a = np.frombuffer(seq.encode(), dtype=np.uint8)
        return int((T != a).sum(axis=1).min())

    children, seen = [], set(parents)
    for p in parents:
        for cdr3_pos in CDR3:
            for cdr3_res in PREFS[cdr3_pos][:3]:
                # 6 framework edits: best residue that DIFFERS from parent
                fw = {}
                for pos in FRAMEWORK:
                    for r in PREFS[pos]:
                        if p[pos] != r:
                            fw[pos] = r
                            break
                if len(fw) < 6:
                    continue
                edits = dict(list(fw.items())[:6])
                if p[cdr3_pos] == cdr3_res:
                    continue
                edits[cdr3_pos] = cdr3_res
                seq = list(p)
                for pos, to in edits.items():
                    seq[pos] = to
                child = "".join(seq)
                if child in seen:
                    continue
                seen.add(child)
                children.append({"sequence": child, "epoch": 25456,
                                 "parent": p, "n_edits": len(edits)})
    novel = [c for c in children if dmin(c["sequence"]) >= 7]
    json.dump(novel, open(out_path, "w"))
    print(f"parents={len(parents)} generated={len(children)} novel_ge7={len(novel)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
