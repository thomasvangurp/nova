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
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path

from tools.rescale_multipass_history import rescale_epoch
from nanobody_opt.rules import ACTIVATION_EPOCH, target_for_epoch

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


def _save(out_path: Path, result: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=out_path.parent, delete=False) as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary = handle.name
    os.replace(temporary, out_path)


def build(out_path: Path, *, lo: int | None, hi: int | None, recent: int = 0) -> dict:
    hi = hi if hi is not None else (_current_epoch() or lo or 0)
    lo = lo if lo is not None else (max(0, hi - recent + 1) if recent else (_earliest_epoch() or 0))
    existing: dict = {}
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
    # An incremental refresh must never discard older measured epochs.
    result: dict = dict(existing)
    for key, prior in list(result.items()):
        if (str(key).isdigit() and int(key) >= ACTIVATION_EPOCH
                and prior.get("method", {}).get("target") != target_for_epoch(int(key))):
            result.pop(key)
    ok, skipped = 0, 0
    for epoch in range(hi, lo - 1, -1):
        entry = rescale_epoch(epoch)
        if entry is None:
            prior = existing.get(str(epoch), {})
            if (prior.get("method", {}).get("coverage_policy") == "complete_fleet_minimum_10_passes"
                    and (epoch < ACTIVATION_EPOCH or prior.get("method", {}).get("target") == target_for_epoch(epoch))):
                result[str(epoch)] = prior
            skipped += 1
            continue
        result[str(epoch)] = entry
        ok += 1
        # Publish progress before the next epoch's network/database work.
        # A killed or timed-out older epoch cannot lose today's completed work.
        _save(out_path, result)
    _save(out_path, result)
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
    parser.add_argument("--recent", type=int, default=0,
                        help="refresh only the newest N epochs, retaining all history")
    args = parser.parse_args()

    while True:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.with_suffix(".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                build(args.out, lo=args.lo, hi=args.hi, recent=args.recent)
        except Exception as exc:  # noqa: BLE001
            print(f"BUILD_FAILED {type(exc).__name__}", flush=True)
            if not args.loop:
                return 1
        if not args.loop:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
