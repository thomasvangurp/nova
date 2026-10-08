#!/usr/bin/env python3
"""Current-archive replay rank-sum for the admissible (>=7-edit + CDR3-novel)
children, using their stored aggregate metrics.  Lower = better.  Compare to
parents (47-106) and winners (54-86)."""
import json
import sqlite3
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("ingest", "/root/ingest_backfill.py")
ib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ib)

archive = sorted(__import__("glob").glob("/root/live/*/archive.csv"))[-1]
fields = ib.load_fields(archive)

shard = json.load(open(sys.argv[1]))
conn = sqlite3.connect("file:/root/artifacts/nanobody_mutations.sqlite?mode=ro", uri=True)

rows = []
for c in shard:
    seq = c["sequence"]
    r = conn.execute(
        """SELECT o.metrics_json FROM observations o JOIN nodes n ON n.id=o.node_id
           WHERE n.sequence=? AND o.score_kind='local_validator_replay_rank_sum'
           ORDER BY o.observation_id LIMIT 1""", (seq,)).fetchone()
    if not r:
        continue
    metrics = json.loads(r[0])
    try:
        rs = ib.replay_distribution(metrics, fields)
    except Exception:
        continue
    rows.append((sum(rs) / len(rs), seq))
conn.close()

rows.sort()
print(f"admissible children scored on current archive: {len(rows)}")
print(f"  min={rows[0][0]:.1f}  p25={rows[len(rows)//4][0]:.1f}  median={rows[len(rows)//2][0]:.1f}  max={rows[-1][0]:.1f}")
print("\ntop 15 (lower=better):")
for sc, seq in rows[:15]:
    print(f"  {sc:6.1f}  {seq[:40]}")
