#!/usr/bin/env python3
"""After exhaustive_8x scoring: rank residues per position by mean rank-sum.

Groups children by (position, to), computes mean local_validator_replay_rank_sum
(lower = better), prints ascending per position, marking the current top-2
residues (from children_topres.json winners) with '*'.
"""
import json, sqlite3, statistics, sys

DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"
POSITIONS = [0, 2, 113, 121, 120, 123, 115, 122, 125, 110, 72, 114, 126]
TOP2 = {0: "QR", 2: "TH", 113: "HF", 115: "AS", 120: "TS", 121: "ST", 123: "TP"}


def main(out):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    spec = json.load(open("exhaustive_children.json"))
    meta = {c["sequence"]: (c["position"], c["to"]) for c in spec["children"]}
    scores = {}
    for seq, sc in c.execute(
        "SELECT n.sequence, o.score FROM nodes n JOIN observations o "
        "ON o.node_id=n.id AND o.score_kind='local_validator_replay_rank_sum' "
        "AND o.source='exhaustive_8x' WHERE n.length=127"):
        scores.setdefault(seq, []).append(sc)
    c.close()

    ranking = {}
    for seq, (pos, to) in meta.items():
        v = scores.get(seq)
        if v:
            ranking.setdefault((pos, to), []).append(statistics.mean(v))

    print("=== per-position residue ranking (mean rank-sum, ascending) ===")
    for pos in POSITIONS:
        rows = [(to, statistics.mean(v)) for (p, to), v in ranking.items() if p == pos]
        if not rows:
            continue
        rows.sort(key=lambda x: x[1])
        cur = TOP2.get(pos, "")
        line = ", ".join(f"{to}:{sc:.1f}{'*' if to in cur else ''}" for to, sc in rows)
        print(f"pos {pos:3d} (top2={cur}): {line}")
    json.dump({f"{p}{to}": round(statistics.mean(v), 2)
               for (p, to), v in ranking.items()}, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
