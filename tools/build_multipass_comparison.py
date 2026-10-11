#!/usr/bin/env python3
"""Build the dashboard's multipass_comparison.json directly from master's
local sqlite -- the one source of truth -- instead of the old multi-hop
chain (dashboard box writes per-epoch JSON -> sync_multipass_rivals.py pulls
every file over SSH with its own pin-tracking state -> Mac merges).

User, 2026-10-11: "Dashboard needs to read from SQLite!" / "One source of
truth please." The old chain is exactly what produced today's mess: the
dashboard box's writes never reached master's sqlite (NOVA_LOCAL_SQLITE
gap, now fixed), its per-epoch JSON files were stale snapshots of whatever
field was live when each one was written, and syncing 114 separate files
with out-of-band-edit pin tracking was its own source of drift.

This reuses rescale_multipass_history.rescale_epoch() -- already reads
master's local sqlite (observations.metrics_json, the formula-independent
raw per-pass data) via a three-tier fallback (multipass replay, our own
race history, the real validator's published metrics) and ranks against the
real last-10-epochs field, never the static archive. One run covers every
epoch from the earliest available data through the current chain epoch, in
one consistent regime, and writes ONE file. Meant to run on a schedule (see
--loop) so the Mac always has something to pull, not just a one-time patch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tools.rescale_multipass_history import rescale_epoch

LOCAL_DB = Path("/root/artifacts/nanobody_mutations.sqlite")
DEFAULT_OUT = Path("/root/artifacts/multipass_comparison.json")


def _current_epoch() -> int | None:
    try:
        return int(json.loads(Path("/root/artifacts/chain_state.json").read_text())["epoch"])
    except Exception:
        return None


def _earliest_epoch() -> int | None:
    import sqlite3

    conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True, timeout=30)
    try:
        row = conn.execute(
            "SELECT MIN(epoch) FROM observations WHERE source='public_multipass_rivals' "
            "AND score_kind='multipass_sum10_pass'"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()


def build(out_path: Path, *, lo: int | None, hi: int | None) -> dict:
    lo = lo if lo is not None else (_earliest_epoch() or 0)
    hi = hi if hi is not None else (_current_epoch() or lo)
    existing: dict = {}
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
    result: dict = {}
    ok, skipped = 0, 0
    for epoch in range(lo, hi + 1):
        entry = rescale_epoch(epoch)
        if entry is None:
            if str(epoch) in existing:
                result[str(epoch)] = existing[str(epoch)]
            skipped += 1
            continue
        result[str(epoch)] = entry
        ok += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"BUILD_DONE epochs={lo}-{hi} ok={ok} skipped_kept_prior={skipped} "
          f"out={out_path}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lo", type=int, default=None)
    parser.add_argument("--hi", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()

    while True:
        try:
            build(args.out, lo=args.lo, hi=args.hi)
        except Exception as exc:  # noqa: BLE001
            print(f"BUILD_FAILED {type(exc).__name__}: {exc}", flush=True)
        if not args.loop:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
