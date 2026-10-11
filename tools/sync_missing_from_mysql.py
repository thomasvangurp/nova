#!/usr/bin/env python3
"""Export observations from central MySQL into master's local sqlite primary
for whatever (source, score_kind) this box's own writes are missing.

Local sqlite is the primary store (ARCHITECTURE_REDESIGN new finding 8); the
replicator (replicate_local.py) only ever flows local -> MySQL, one way. Any
box without master's local sqlite file at the expected path falls through to
MySQL automatically (mutation_network._connect checks file existence, not an
opt-in flag -- see its 2026-10-11 docstring); writing there silently orphans
the data from local's point of view, no error, the write just succeeds into
the wrong store.

Confirmed 2026-10-11 for two sources:
  - public_multipass_rivals / multipass_sum10_pass: the dashboard box
    (95.3.33.46) is a genuinely separate machine with no copy of master's
    sqlite file, so it always writes MySQL. There is no way to point a
    remote box at a file that only exists on master, so going forward this
    script (run on a schedule, see --loop) is the permanent bridge, not a
    one-time patch.
  - local_exact_raw / local_exact_validator_replay_rank_sum: the scorer
    daemon's own supervisor script never exported NOVA_LOCAL_SQLITE=1 (the
    env var _connect used to require), so every observation it wrote went to
    MySQL regardless of the producer's own correctly-configured environment
    -- the exact fragility _connect no longer has. This script only needed
    to backfill the gap that already accumulated; now that locality is
    file-existence-based, the same silent divergence cannot recur here.

Idempotent: observation_id is the primary key, update_observation_store
upserts on it, so re-running (or overlapping windows) never duplicates rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LOCAL_DB = Path("/root/artifacts/nanobody_mutations.sqlite")


def _local_max_epoch(source: str, score_kind: str) -> int:
    import sqlite3

    conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True, timeout=30)
    try:
        row = conn.execute(
            "SELECT MAX(epoch) FROM observations WHERE source = ? AND score_kind = ?",
            (source, score_kind),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def _mysql_rows(source: str, score_kind: str, since_epoch: int):
    from nanobody_opt import db

    conn = db.connect()
    cursor = conn.execute(
        """
        SELECT o.observation_id, o.epoch, n.target, n.sequence, o.score,
               o.observation_count, o.metrics_json, o.provenance_json
        FROM observations o JOIN nodes n ON n.id = o.node_id
        WHERE o.source = %s AND o.score_kind = %s AND o.epoch > %s
          AND o.metrics_json IS NOT NULL AND o.metrics_json != ''
        """,
        (source, score_kind, since_epoch),
    )
    for (observation_id, epoch, target, sequence, score, observation_count,
         metrics_json, provenance_json) in cursor:
        try:
            metrics = json.loads(metrics_json)
        except (TypeError, ValueError):
            metrics = {}
        try:
            provenance = json.loads(provenance_json) if provenance_json else {}
        except (TypeError, ValueError):
            provenance = {}
        if not isinstance(provenance, dict):
            provenance = {}
        # provenance carries writer/formula/field -- required by the strict
        # label registry (observation_labels.check_observations) for labels
        # like multipass_sum10_pass. The explicit keys below always win over
        # anything stale in provenance (defensive; they should never collide).
        row = dict(provenance)
        row.update({
            "observation_id": observation_id,
            "sequence": sequence,
            "target": target,
            "score": score,
            "epoch": epoch,
            "source": source,
            "score_kind": score_kind,
            "observation_count": observation_count,
            "metrics": metrics,
        })
        yield row


def sync_one(source: str, score_kind: str, *, dry_run: bool = False) -> int:
    from nanobody_opt.mutation_network import update_observation_store

    since = _local_max_epoch(source, score_kind)
    rows = list(_mysql_rows(source, score_kind, since))
    print(f"SYNC_READ source={source} score_kind={score_kind} "
          f"local_max_epoch={since} mysql_rows_beyond={len(rows)}", flush=True)
    if not rows:
        return 0
    if dry_run:
        epochs = sorted({r["epoch"] for r in rows})
        print(f"SYNC_DRY_RUN would_write={len(rows)} "
              f"epochs={epochs[0]}..{epochs[-1]}", flush=True)
        return 0
    written = update_observation_store(LOCAL_DB, rows)
    print(f"SYNC_DONE source={source} score_kind={score_kind} written={written}", flush=True)
    return written


PAIRS = [
    ("public_multipass_rivals", "multipass_sum10_pass"),
    ("local_exact_raw", "local_exact_validator_replay_rank_sum"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--score-kind", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--loop", action="store_true",
                        help="repeat forever, --interval seconds apart")
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()

    if args.source or args.score_kind:
        if len(args.source) != len(args.score_kind):
            parser.error("--source and --score-kind must be given in matching pairs")
        pairs = list(zip(args.source, args.score_kind))
    else:
        pairs = PAIRS

    while True:
        for source, score_kind in pairs:
            try:
                sync_one(source, score_kind, dry_run=args.dry_run)
            except Exception as exc:  # noqa: BLE001
                print(f"SYNC_FAILED source={source} score_kind={score_kind} "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if not args.loop:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
