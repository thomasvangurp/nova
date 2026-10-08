#!/usr/bin/env python3
"""Compute the frontier test's current partial replay scores (4-pass mean)
from the in-progress results JSON."""
import json
import sys
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location("ingest", "/root/ingest_backfill.py")
ib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ib)

archive = sorted(__import__("glob").glob("/root/live/*/archive.csv"))[-1]
fields = ib.load_fields(archive)

results = json.load(open(sys.argv[1]))
rows = []
for h, e in results.items():
    passes = [e[f"p{i}"] for i in range(1, 9) if f"p{i}" in e]
    if not passes:
        continue
    rankm = np.array([ib.replay_distribution(p, fields) for p in passes])
    rows.append((float(rankm.reshape(-1).mean()), e.get("sequence", "")))

rows.sort()
print(f"frontier partial ({len(rows)} scored, {len(passes)} passes): "
      f"min={rows[0][0]:.1f} median={rows[len(rows)//2][0]:.1f} max={rows[-1][0]:.1f}")
print("best 12 (lower=better):")
for sc, seq in rows[:12]:
    print(f"  {sc:6.1f}  {seq[:34]}")
