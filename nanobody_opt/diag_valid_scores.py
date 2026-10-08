#!/usr/bin/env python3
"""Are the VALID (>=7-edit) children competitive? Compare their 8x scores to
the near-dup (<=6-edit) children, and count how many families they span."""
import json
import sqlite3
import statistics as st

import numpy as np

DB = "/root/artifacts/nanobody_mutations.sqlite"
idx = json.load(open("/root/artifacts/top50_cdr3_index.json"))
T = np.array([[ord(c) for c in s] for s in idx["sequences"] if len(s) == 127], dtype=np.uint8)


def dmin(seq: str) -> int:
    a = np.frombuffer(seq.encode(), dtype=np.uint8)
    return int((T != a).sum(axis=1).min())


conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = conn.execute(
    """SELECT n.sequence, AVG(o.score), SUM(o.observation_count)
       FROM observations o JOIN nodes n ON n.id=o.node_id
       WHERE o.score_kind='local_validator_replay_rank_sum' AND n.length=127
       GROUP BY n.id""").fetchall()

valid, invalid = [], []
for seq, mean, obs in rows:
    if mean is None:
        continue
    if dmin(seq) >= 7:
        valid.append((mean, seq, obs))
    else:
        invalid.append((mean, seq, obs))

valid.sort()
invalid.sort()
print(f"invalid (<=6 edits): {len(invalid)}  best-8x mean={invalid[0][0]:.1f}  median={invalid[len(invalid)//2][0]:.1f}")
print(f"valid   (>=7 edits): {len(valid)}  best-8x mean={valid[0][0]:.1f}  median={valid[len(valid)//2][0]:.1f}")
print("\ntop-15 VALID children by 8x mean score (lower better):")
for mean, seq, obs in valid[:15]:
    print(f"  {mean:6.1f}  edits={dmin(seq)}  obs={obs}  {seq[:40]}")
conn.close()
