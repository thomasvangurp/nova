#!/usr/bin/env python3
"""Remote half of the epoch loop. Runs on production. Emits JSON on stdout."""
import json, sys, sqlite3, statistics, urllib.request, hashlib

DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"
API = "https://compound-api.metanova-labs.ai"

def op(op, **kw):
    if op == "new_public":
        epochs = int(kw["epochs"])
        known = set()
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        for (seq,) in c.execute(
            "SELECT DISTINCT n.sequence FROM nodes n JOIN observations o "
            "ON o.node_id=n.id AND o.score_kind='validator_final_rank_sum' "
            "WHERE n.length=127"):
            known.add(hashlib.sha256(seq.encode()).hexdigest())
        c.close()
        new = {}
        for epoch in range(25433, 25433 - epochs, -1):
            try:
                req = urllib.request.Request(
                    f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
                    headers={"User-Agent": "nova-epoch-loop/1"})
                payload = json.load(urllib.request.urlopen(req, timeout=30))
            except Exception:
                continue
            for sub in (payload.get("data") or []):
                for nb in (sub.get("nanobodies") or []):
                    if nb.get("protein_name") != TARGET:
                        continue
                    s = str(nb.get("sequence") or "").strip().upper()
                    h = hashlib.sha256(s.encode()).hexdigest()
                    if len(s) == 127 and h not in known and s not in new:
                        new[s] = epoch
        print(json.dumps({"new_public": [{"sequence": s, "epoch": e}
                                         for s, e in sorted(new.items(), key=lambda x: x[1])]}))
    elif op == "replay_scores":
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        acc = {}
        for seq, sc in c.execute(
            "SELECT n.sequence, o.score FROM nodes n JOIN observations o "
            "ON o.node_id=n.id AND o.score_kind='local_validator_replay_rank_sum' "
            "WHERE n.length=127"):
            acc.setdefault(seq, []).append(sc)
        c.close()
        out = [{"sequence": s, "mean_replay": statistics.mean(v)}
               for s, v in acc.items()]
        print(json.dumps({"replay_scores": out}))
    elif op == "existing":
        c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        seqs = [r[0] for r in c.execute("SELECT sequence FROM nodes WHERE target=?", (TARGET,))]
        c.close()
        print(json.dumps({"existing": seqs}))

if __name__ == "__main__":
    op(sys.argv[1], **dict(kv.split("=", 1) for kv in sys.argv[2:]))
