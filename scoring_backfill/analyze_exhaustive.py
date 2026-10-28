#!/usr/bin/env python3
"""After exhaustive_8x scoring: rank residues per position by mean rank-sum.

Reads the sweep spec (``exhaustive_children.json``) for the target, modal
length and swept positions instead of a P20809/127 hardcode.  Groups children
by (position, to), computes mean local_validator_replay_rank_sum (lower =
better), prints ascending per position, marking the current top-2 residues
(from ``children_topres.json`` winners, when present) with '*'.
"""
import json
import sqlite3
import statistics
import sys

DB = "/root/artifacts/nanobody_mutations.sqlite"


def load_top2(path: str = "children_topres.json") -> dict[int, str]:
    """Load {0-indexed position: top-2 residue letters} from the winners file.

    The original sweep kept this as a hardcoded P20809 map.  It is only used to
    decorate the printout, so a missing file (e.g. a fresh Q9NZQ7 sweep with no
    winners yet) simply yields no '*' marks.
    """
    try:
        winners = json.load(open(path))
    except (OSError, json.JSONDecodeError):
        return {}
    top2: dict[int, str] = {}
    for key, value in winners.items():
        try:
            position = int(key)
        except (TypeError, ValueError):
            continue
        top2[position] = "".join(str(v) for v in (value if isinstance(value, (list, tuple)) else [value])[:2])
    return top2


def main(out):
    spec = json.load(open("exhaustive_children.json"))
    target = spec["target"]
    length = spec["length"]
    positions = spec["positions"]
    meta = {c["sequence"]: (c["position"], c["to"]) for c in spec["children"]}

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    scores: dict[str, list[float]] = {}
    for seq, sc in conn.execute(
        "SELECT n.sequence, o.score FROM nodes n JOIN observations o "
        "ON o.node_id=n.id AND o.score_kind='local_validator_replay_rank_sum' "
        "AND o.source='exhaustive_8x' WHERE n.target=? AND n.length=?",
        (target, length)):
        scores.setdefault(seq, []).append(sc)
    conn.close()

    ranking: dict[tuple[int, str], list[float]] = {}
    for seq, (pos, to) in meta.items():
        v = scores.get(seq)
        if v:
            ranking.setdefault((pos, to), []).append(statistics.mean(v))

    top2 = load_top2()

    print("=== per-position residue ranking (mean rank-sum, ascending) ===")
    for pos in positions:
        rows = [(to, statistics.mean(v)) for (p, to), v in ranking.items() if p == pos]
        if not rows:
            continue
        rows.sort(key=lambda x: x[1])
        cur = top2.get(pos, "")
        line = ", ".join(f"{to}:{sc:.1f}{'*' if to in cur else ''}" for to, sc in rows)
        print(f"pos {pos:3d} (top2={cur}): {line}")
    json.dump({f"{p}{to}": round(statistics.mean(v), 2)
               for (p, to), v in ranking.items()}, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1])
