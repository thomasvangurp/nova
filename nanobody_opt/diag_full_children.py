#!/usr/bin/env python3
"""Diagnostic: across EVERY children source in the DB, how many are valid
(>=7 edits from the top-50)?  Also report each source's parent count."""
import json
import sqlite3

import numpy as np

DB = "/root/artifacts/nanobody_mutations.sqlite"
idx = json.load(open("/root/artifacts/top50_cdr3_index.json"))
T = np.array([[ord(c) for c in s] for s in idx["sequences"] if len(s) == 127], dtype=np.uint8)


def dmin(seq: str) -> int:
    a = np.frombuffer(seq.encode(), dtype=np.uint8)
    return int((T != a).sum(axis=1).min())


conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

# 1) all children sources + sizes
print("=== children sources (score_kind=local_validator_replay_rank_sum) ===")
rows = conn.execute(
    """SELECT o.source, COUNT(DISTINCT n.id), SUM(o.observation_count)
       FROM observations o JOIN nodes n ON n.id=o.node_id
       WHERE o.score_kind='local_validator_replay_rank_sum'
       GROUP BY o.source ORDER BY 2 DESC""").fetchall()
for src, n, obs in rows:
    print(f"  {src:20s} nodes={n:8d} obs={obs}")

# 2) total distinct child sequences across ALL replay sources, and validity
print("\n=== validity of ALL replay-scored children vs top-50 ===")
all_rows = conn.execute(
    """SELECT DISTINCT n.sequence FROM observations o JOIN nodes n ON n.id=o.node_id
       WHERE o.score_kind='local_validator_replay_rank_sum' AND n.length=127""").fetchall()
seqs = [r[0] for r in all_rows]
valid = [s for s in seqs if dmin(s) >= 7]
print(f"  total replay-scored children: {len(seqs)}")
print(f"  valid (>=7 edits from top-50): {len(valid)}")
print(f"  invalid (<=6 edits): {len(seqs) - len(valid)}")

# 3) also check the FULL nodes table (everything we've ever stored, len 127)
print("\n=== FULL nodes table (all sequences ever stored, len 127) ===")
alln = conn.execute("SELECT sequence FROM nodes WHERE length=127").fetchall()
allseqs = [r[0] for r in alln]
v2 = [s for s in allseqs if dmin(s) >= 7]
print(f"  total nodes: {len(allseqs)}")
print(f"  valid (>=7 edits): {len(v2)}")

# 4) distribution of min-edit-distance for the replay-scored children
print("\n=== replay-scored children min-edit-dist histogram ===")
dists = [dmin(s) for s in seqs]
from collections import Counter
c = Counter(dists)
for d in sorted(c):
    print(f"  {d} edits: {c[d]}")
conn.close()
