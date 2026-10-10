#!/usr/bin/env python3
"""Replicate master's local SQLite store to Hetzner MySQL (ARCHITECTURE_REDESIGN
new finding 8). Durability and cross-box visibility for the local-sqlite-as-
primary design -- this is the piece that was missing when per-machine sqlite
was dropped before: a box's local-only data had no copy anywhere else, so
losing the box lost the data (exactly what happened to the "slave" box's
20x-refresh data, recovered from disk by hand earlier today).

Crash-safety: every replicable table has a `replicated_at` column, NULL until
a row is confirmed written to Hetzner. A row is never deleted or marked
replicated until the Hetzner write actually commits, so a crash mid-run just
means the next run re-finds the same NULL rows -- no replication queue to
lose, no separate journal to go out of sync with the data. Re-sending an
already-replicated row is a no-op (the MySQL side upserts on primary key).

Usage::

    python3 -m nanobody_opt.replicate_local --local /root/artifacts/nanobody_local.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from nanobody_opt import db

# (table, replicable_columns, upsert_sql) -- columns exclude replicated_at.
TABLES: list[tuple[str, list[str], str]] = [
    (
        "nodes",
        ["id", "target", "sequence", "length", "residue_vector"],
        """INSERT INTO nodes (id, target, sequence, length, residue_vector)
           VALUES (%s,%s,%s,%s,%s)
           ON DUPLICATE KEY UPDATE length=VALUES(length)""",
    ),
    (
        "observations",
        ["observation_id", "node_id", "epoch", "source", "score_kind", "score",
         "observation_count", "metrics_json", "pose_json", "provenance_json"],
        """INSERT INTO observations (observation_id, node_id, epoch, source,
               score_kind, score, observation_count, metrics_json, pose_json,
               provenance_json)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON DUPLICATE KEY UPDATE score=VALUES(score),
               observation_count=VALUES(observation_count),
               metrics_json=VALUES(metrics_json)""",
    ),
    (
        "observation_artifacts",
        ["observation_id", "confidence_path", "writer_path", "structure_json",
         "sample_selection_json"],
        """INSERT INTO observation_artifacts (observation_id, confidence_path,
               writer_path, structure_json, sample_selection_json)
           VALUES (%s,%s,%s,%s,%s)
           ON DUPLICATE KEY UPDATE structure_json=VALUES(structure_json)""",
    ),
    (
        "lineage_edges",
        ["parent_node_id", "child_node_id", "epoch", "source", "proposal_type",
         "mutations_json", "metadata_json"],
        """INSERT INTO lineage_edges (parent_node_id, child_node_id, epoch,
               source, proposal_type, mutations_json, metadata_json)
           VALUES (%s,%s,%s,%s,%s,%s,%s)
           ON DUPLICATE KEY UPDATE proposal_type=VALUES(proposal_type)""",
    ),
    (
        "exact_runs",
        ["target", "run_id", "epoch", "source_path", "file_sha256",
         "scorer_protocol", "expected_rows", "ingested_rows", "complete",
         "ingested_at"],
        """INSERT INTO exact_runs (target, run_id, epoch, source_path,
               file_sha256, scorer_protocol, expected_rows, ingested_rows,
               complete, ingested_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON DUPLICATE KEY UPDATE complete=VALUES(complete),
               ingested_rows=VALUES(ingested_rows)""",
    ),
    (
        "exact_run_observations",
        ["target", "run_id", "observation_id"],
        """INSERT INTO exact_run_observations (target, run_id, observation_id)
           VALUES (%s,%s,%s)
           ON DUPLICATE KEY UPDATE observation_id=VALUES(observation_id)""",
    ),
]

REPLICATION_BATCH = 1000


def _open_local(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def replicate_table(
    local: sqlite3.Connection,
    remote: "db.Connection",
    table: str,
    columns: list[str],
    upsert_sql: str,
    *,
    batch_size: int = REPLICATION_BATCH,
    dry_run: bool = False,
) -> int:
    """Push every row with replicated_at IS NULL; mark it replicated on success."""
    col_list = ", ".join(columns)
    cur = local.execute(
        f"SELECT rowid, {col_list} FROM {table} WHERE replicated_at IS NULL "
        f"LIMIT {batch_size}"
    )
    rows = cur.fetchall()
    if not rows:
        return 0
    payload = [tuple(row[col] for col in columns) for row in rows]
    rowids = [row["rowid"] for row in rows]
    if dry_run:
        print(f"  [dry-run] would replicate {len(rows)} rows from {table}")
        return len(rows)
    remote.executemany(upsert_sql, payload)
    remote.commit()
    # Only mark replicated AFTER the remote commit succeeds above -- if the
    # process dies between the two, these rows are still NULL and get
    # re-sent next run (a harmless re-upsert), never silently dropped.
    now = time.time()
    local.execute(
        f"UPDATE {table} SET replicated_at=? "
        f"WHERE rowid IN ({','.join('?' * len(rowids))})",
        (now, *rowids),
    )
    local.commit()
    return len(rows)


def replicate_once(local_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    local = _open_local(local_path)
    remote = db.connect()
    totals: dict[str, int] = {}
    try:
        for table, columns, upsert_sql in TABLES:
            total = 0
            while True:
                n = replicate_table(
                    local, remote, table, columns, upsert_sql, dry_run=dry_run
                )
                total += n
                if n < REPLICATION_BATCH:
                    break
            totals[table] = total
    finally:
        remote.close()
        local.close()
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop-interval", type=float, default=0.0,
                         help="if set, replicate repeatedly every N seconds "
                              "instead of running once")
    args = parser.parse_args()

    if not args.local.is_file():
        print(f"REPLICATE_LOCAL_MISSING_DB {args.local}", file=sys.stderr)
        return 1

    while True:
        try:
            totals = replicate_once(args.local, dry_run=args.dry_run)
        except Exception as error:  # noqa: BLE001
            print(f"REPLICATE_LOCAL_FAILED {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
            if not args.loop_interval:
                return 1
            time.sleep(args.loop_interval)
            continue
        replicated_any = sum(totals.values())
        print(f"REPLICATE_LOCAL_OK {totals}", flush=True)
        if not args.loop_interval:
            return 0
        time.sleep(args.loop_interval if replicated_any == 0 else 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
