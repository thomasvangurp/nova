import json, sqlite3, statistics, hashlib, sys
DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"
AA = "ACDEFGHIKLMNPQRSTVWY"
# 13 framework positions (top-15 minus CDR1 pos30 and CDR3 pos99)
POSITIONS = [0, 2, 113, 121, 120, 123, 115, 122, 125, 110, 72, 114, 126]
def main(parents_n):
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    acc = {}
    for seq, sc in c.execute(
        "SELECT n.sequence, o.score FROM nodes n JOIN observations o "
        "ON o.node_id=n.id AND o.score_kind='local_validator_replay_rank_sum' "
        "WHERE n.length=127"):
        acc.setdefault(seq, []).append(sc)
    parents = [s for s, v in sorted(acc.items(), key=lambda x: (statistics.mean(x[1]), x[0]))[:parents_n]]
    existing = set(r[0] for r in c.execute("SELECT sequence FROM nodes WHERE target=?", (TARGET,)))
    c.close()
    children = {}
    for p in parents:
        for pos in POSITIONS:
            for to in AA:
                if pos >= len(p) or p[pos] == to:
                    continue
                child = p[:pos] + to + p[pos+1:]
                if child in existing:
                    continue
                children.setdefault(child, {"sequence": child, "parent": p, "position": pos, "to": to})
    work = [{"sequence": c["sequence"], "epoch": 25423, "parent": c["parent"],
             "position": c["position"], "to": c["to"]} for c in children.values()]
    json.dump({"parents": parents, "positions": POSITIONS, "children": work}, open("exhaustive_children.json", "w"))
    from collections import Counter
    per = Counter(w["position"] for w in work)
    print(f"parents={len(parents)} children={len(work)}")
    print("per-position:", dict(sorted(per.items())))
main(20)
