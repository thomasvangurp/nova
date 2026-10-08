#!/usr/bin/env python3
"""Compare our epoch-25456 submissions' offline replay (8x vs 14x) against the
validator's actual final_score."""
import json
import sqlite3
import urllib.request

API = "https://compound-api.metanova-labs.ai"
DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"


def main():
    p = json.load(urllib.request.urlopen(
        f"{API}/api/competitions/leaderboard/25456/nanobodies", timeout=25))
    entries = []
    for sub in p.get("data") or []:
        if not (sub.get("coldkey") or "").startswith("5EkCwcN"):
            continue
        for nb in sub.get("nanobodies") or []:
            if nb.get("protein_name") == TARGET:
                entries.append((str(nb.get("sequence") or "").strip().upper(),
                                sub.get("final_score")))
    entries.sort(key=lambda x: x[1])

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    print("our 25456 entries: validator final_score vs offline replay (8x / 14x), lower=better")
    for seq, vsc in entries:
        row8 = conn.execute(
            """SELECT AVG(o.score) FROM observations o JOIN nodes n ON n.id=o.node_id
               WHERE n.sequence=? AND o.score_kind='local_validator_replay_rank_sum'
               AND o.source='children_8x'""", (seq,)).fetchone()
        row14 = conn.execute(
            """SELECT AVG(o.score) FROM observations o JOIN nodes n ON n.id=o.node_id
               WHERE n.sequence=? AND o.score_kind='local_validator_replay_rank_sum'
               AND o.source='children_20x'""", (seq,)).fetchone()
        r8 = row8[0] if row8 and row8[0] is not None else None
        r14 = row14[0] if row14 and row14[0] is not None else None
        print(f"  validator={vsc:6.1f}  8x={r8 if r8 is None else round(r8,1):>7}  "
              f"14x={r14 if r14 is None else round(r14,1):>7}  {seq[:24]}")
    conn.close()


if __name__ == "__main__":
    main()
