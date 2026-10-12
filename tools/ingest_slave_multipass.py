#!/usr/bin/env python3
"""Insert slave-computed multipass observation rows into MASTER's live local
SQLite primary (/root/artifacts/nanobody_mutations.sqlite).

This is the ONLY piece of the slave-writeback pipeline that ever opens the
live DB file. It runs exclusively ON MASTER, reads NDJSON rows shipped over
from the slave box (see /root/slave_multipass_writeback.py and
tools/run_slave_multipass_epoch.sh), and writes them through the exact same
``nanobody_opt.mutation_network.update_observation_store`` the live
producer/race uses for its own writes -- just pointed at the local-sqlite
backend (``NOVA_LOCAL_SQLITE=1``) instead of MySQL. That function already:

  * resolves/creates nodes by (target, sequence) -- no node_id plumbing here;
  * runs the observation_labels registry check before writing anything (a
    row with the wrong writer/source/formula for its score_kind is REFUSED,
    and nothing in the batch is written);
  * upserts on ``observation_id`` (the table's PRIMARY KEY), so re-ingesting
    the same NDJSON file twice updates the same rows in place -- it never
    duplicates them.

Never edits, deletes, or locks out the live producer/race/publisher/scorer
daemon: this opens the same WAL-mode sqlite file with sqlite3's normal
``timeout=30`` busy wait (via LocalConnection), the same way every other
writer already shares it, and writes in a handful of short, chunked
transactions.

Usage (on master)::

    python3 /root/tools/ingest_slave_multipass.py \\
        --ndjson /root/artifacts/slave_writeback_incoming/epoch25641.ndjson
    python3 /root/tools/ingest_slave_multipass.py --ndjson FILE --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DB_PATH = Path("/root/artifacts/nanobody_mutations.sqlite")
EXPECTED_SOURCE = "public_multipass_rivals"
EXPECTED_SCORE_KIND = "multipass_sum10_pass"
EXPECTED_WRITERS = ("slave_multipass_writeback", "replay_queue_slave")
# NDJSON written before the 2026-10-11 label unification (files already in
# slave_writeback_incoming/, or a slave run in flight at the cutover) still
# carries the retired slave-only label; same measurement, so rewrite it.
RETIRED_LABEL = ("slave_public_multipass_rivals", "slave_multipass_sum10_pass")


def load_rows(path: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (row.get("source"), row.get("score_kind")) == RETIRED_LABEL:
                row["source"], row["score_kind"] = EXPECTED_SOURCE, EXPECTED_SCORE_KIND
            # De-dup by observation_id client-side (last write in the file
            # wins); the DB-side upsert makes this belt-and-suspenders, not
            # load-bearing.
            rows[str(row.get("observation_id"))] = row
    return list(rows.values())


def validate(rows: list[dict]) -> list[str]:
    """Defense in depth ahead of the registry check: fail loudly and early on
    anything that isn't this pipeline's own shape, rather than let a stray
    row from a different source/writer slip into update_observation_store's
    batch (which would refuse the WHOLE batch anyway -- this just gives a
    clearer error before any file I/O against the live DB happens)."""
    problems = []
    for row in rows:
        oid = row.get("observation_id", "<missing>")
        if row.get("source") != EXPECTED_SOURCE:
            problems.append(f"{oid}: source={row.get('source')!r} != {EXPECTED_SOURCE!r}")
        if row.get("score_kind") != EXPECTED_SCORE_KIND:
            problems.append(f"{oid}: score_kind={row.get('score_kind')!r} != {EXPECTED_SCORE_KIND!r}")
        if row.get("writer") not in EXPECTED_WRITERS:
            problems.append(f"{oid}: writer={row.get('writer')!r} not in {EXPECTED_WRITERS!r}")
        if not row.get("sequence") or not row.get("target"):
            problems.append(f"{oid}: missing sequence/target")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ndjson", required=True, type=Path)
    ap.add_argument("--db", default=DB_PATH, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.ndjson)
    print(f"INGEST_READ rows={len(rows)} file={args.ndjson}", flush=True)
    if not rows:
        print("INGEST_EMPTY", flush=True)
        return 0

    problems = validate(rows)
    if problems:
        print(f"INGEST_VALIDATION_FAILED n={len(problems)}", flush=True)
        for p in problems[:20]:
            print(f"  {p}", flush=True)
        return 2

    if args.dry_run:
        print(f"INGEST_DRY_RUN rows={len(rows)} (not written)", flush=True)
        for row in rows[:3]:
            print(f"  sample: {json.dumps(row, sort_keys=True)[:300]}", flush=True)
        return 0

    sys.path.insert(0, "/root")
    from nanobody_opt.mutation_network import update_observation_store

    written = update_observation_store(args.db, rows, commit_every=1000, commit_interval=5.0)
    print(f"INGEST_DONE written={written} db={args.db}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
