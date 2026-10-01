#!/usr/bin/env python3
"""Emit a compact backfill summary from the production DB to a known Mac path."""
import sqlite3, json, subprocess, sys
DB = "/root/artifacts/nanobody_mutations.sqlite"
PROD = "root@ssh3.vast.ai"; PROD_PORT = 11806
OUT = "/Users/tvgdeena/nova-nanobody/data/backfill_summary.json"

q = """
SELECT COUNT(*), COALESCE(SUM(observation_count),0)
FROM observations WHERE source='backfill_8x' AND score_kind='local_validator_replay_rank_sum'
"""
py = (f"import sqlite3,json;c=sqlite3.connect('{DB}');"
      f"r=c.execute('''{q}''').fetchone();"
      f"print(json.dumps({{'rows':r[0],'passes':r[1]}}))")
r = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    "-p", str(PROD_PORT), PROD, f"python3 -c \"{py}\""],
                   capture_output=True, text=True)
try:
    summary = json.loads(r.stdout.strip())
except Exception:
    summary = {"rows": -1, "passes": -1}
open(OUT, "w").write(json.dumps(summary, indent=2))
print(json.dumps(summary))
