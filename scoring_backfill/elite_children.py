#!/usr/bin/env python3
"""After backfill: rank public multi-pass sequences by mean replay score, cut
top-12% elite set, generate single-substitution children at 7 hot positions."""
import sqlite3, json, sys, statistics

TARGET = "P20809"
# Top-2 residues per framework position, MEASURED by the exhaustive sweep
# (2026-10-01: 13 positions x 19 residues x 20 parents, each scored 8x; see
# scoring_backfill/exhaustive_ranking.json). The previous hand-picked set was
# wrong at 6 of the 7 positions it covered -- notably R at pos 0 (ranked 12/20,
# 37 rank-sum worse than A) and H at pos 2 (13/20). Alanine is best or tied-best
# at 7 of 13 positions, which the old set almost entirely missed.
#
# Position 114 is deliberately ABSENT: its best residue (D, 221.5) scores worse
# than most positions' worst, so it only wastes candidates.
HOT = {
    0:   "ADQGN",
    2:   "AMVEK",
    72:  "ATDVQ",
    110: "EDQKA",
    113: "HMFVY",
    115: "ASTEH",
    120: "ATSVP",
    121: "APSVG",
    122: "VATIL",
    123: "ASPGV",
    125: "GASPT",
    126: "APGES",
}

def main():
    db = sys.argv[1]
    out = sys.argv[2]
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    # public sequences (have validator_final_rank_sum), length 127
    rows = conn.execute("""
        SELECT n.id, n.sequence, MIN(o.epoch)
        FROM nodes n JOIN observations o ON o.node_id=n.id AND o.score_kind='validator_final_rank_sum'
        WHERE n.length=127
        GROUP BY n.id
    """).fetchall()
    pub = {nid: {"id": nid, "seq": seq, "first_epoch": e} for nid, seq, e in rows}
    # replay scores
    rep = conn.execute("""
        SELECT node_id, score FROM observations WHERE score_kind='local_validator_replay_rank_sum'
        AND node_id IN (SELECT node_id FROM observations WHERE score_kind='validator_final_rank_sum')
    """).fetchall()
    by_node = {}
    for nid, s in rep:
        by_node.setdefault(nid, []).append(s)
    scored = []
    for nid, info in pub.items():
        if nid in by_node:
            info["mean_replay"] = statistics.mean(by_node[nid])
            scored.append(info)
    scored.sort(key=lambda x: (x["mean_replay"], x["seq"]))
    k = max(1, int(len(scored) * 0.12))
    elite = scored[:k]
    threshold = elite[-1]["mean_replay"]
    print(f"public={len(pub)} multi_pass={len(scored)} elite={len(elite)} threshold={threshold:.2f}", flush=True)

    # existing sequences for dedup
    existing = set(r[0] for r in conn.execute("SELECT sequence FROM nodes WHERE target=?", (TARGET,)))
    conn.close()

    children = {}
    for p in elite:
        seq = p["seq"]
        for pos, tos in HOT.items():
            for to in tos:
                if pos >= len(seq) or seq[pos] == to:
                    continue
                child = seq[:pos] + to + seq[pos+1:]
                if child in existing:
                    continue
                children.setdefault(child, {"sequence": child, "parent": seq, "epoch": p["first_epoch"], "position": pos, "to": to})
    print(f"unique_children={len(children)}", flush=True)
    json.dump({"elite": elite, "threshold": threshold, "children": list(children.values())}, open(out, "w"))
    print(f"wrote {out}", flush=True)

if __name__ == "__main__":
    main()
