"""Fetch leaderboard epochs and slot decisions into SQLite.

The public leaderboard is the only source that says who actually SCORED in an
epoch.  The slot log is the only source that says how many uids we were paying
for at the time.  The gap between the two is the number worth watching, so both
land in the same database.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import DEFAULT_COLDKEY

ROOT = "https://compound-api.metanova-labs.ai/api"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    epoch INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    uid INTEGER,
    hotkey TEXT NOT NULL,
    coldkey TEXT,
    score REAL,
    molecule_score REAL,
    sequence TEXT,
    PRIMARY KEY (epoch, hotkey)
);
CREATE INDEX IF NOT EXISTS entries_epoch ON entries(epoch);
CREATE INDEX IF NOT EXISTS entries_coldkey ON entries(coldkey);
CREATE TABLE IF NOT EXISTS epochs (
    epoch INTEGER PRIMARY KEY,
    field_size INTEGER,
    winner REAL,
    median REAL,
    top10_cut REAL,
    fetched_at REAL
);
CREATE TABLE IF NOT EXISTS slots (
    time REAL PRIMARY KEY,
    epoch INTEGER,
    block INTEGER,
    registered INTEGER,
    target INTEGER,
    recycle_tao REAL,
    decision_recycle_tao REAL,
    executed INTEGER,
    reason TEXT,
    hotkey TEXT
);
CREATE INDEX IF NOT EXISTS slots_epoch ON slots(epoch);
"""


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    return connection


def _rows(body):
    """The API has returned both a bare list and three different envelopes."""
    if isinstance(body, list):
        return body
    for key in ("results", "entries", "data", "leaderboard"):
        if isinstance(body.get(key), list):
            return body[key]
    return []


def fetch_epoch(epoch: int, timeout: int = 90):
    url = f"{ROOT}/competitions/leaderboard/{epoch}/nanobodies?limit=10000"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode()
    return _rows(json.loads(body))          # parse before anything is stored


def _sequence(row):
    nanobodies = row.get("nanobodies") or []
    if not nanobodies:
        return None
    first = nanobodies[0]
    return first.get("sequence") if isinstance(first, dict) else first


def store_epoch(connection, epoch: int, rows) -> int:
    scored = [r for r in rows if r.get("final_score") is not None]
    if not scored:
        return 0
    scored.sort(key=lambda r: r["final_score"])
    values = [(epoch, i + 1, r.get("uid"), r.get("hotkey"), r.get("coldkey"),
               r.get("final_score"), r.get("molecule_final_score"), _sequence(r))
              for i, r in enumerate(scored)]
    connection.executemany(
        "INSERT OR REPLACE INTO entries "
        "(epoch, rank, uid, hotkey, coldkey, score, molecule_score, sequence) "
        "VALUES (?,?,?,?,?,?,?,?)", values)
    ladder = [r["final_score"] for r in scored]
    connection.execute(
        "INSERT OR REPLACE INTO epochs "
        "(epoch, field_size, winner, median, top10_cut, fetched_at) VALUES (?,?,?,?,?,?)",
        (epoch, len(ladder), ladder[0], ladder[len(ladder) // 2],
         ladder[9] if len(ladder) >= 10 else None, time.time()))
    connection.commit()
    return len(ladder)


def have(connection) -> set[int]:
    return {e for (e,) in connection.execute("SELECT epoch FROM epochs")}


def collect_leaderboard(connection, first: int, last: int, workers: int = 6,
                        refetch_last: int = 2) -> dict:
    """Fetch every epoch in range that is not already stored.

    The newest couple of epochs are re-fetched even when present: an epoch can
    be read while the validator is still writing it, and a partial ladder that
    never gets corrected would silently understate the field for good.
    """
    known = have(connection)
    # ``last`` intentionally probes ahead of the newest stored epoch.  Using
    # it as the refetch boundary therefore re-requested only future 404s and
    # left a partially written newest leaderboard frozen forever.  Anchor the
    # retry window to the newest epoch we actually hold instead.
    newest_known = max(known) if known else None
    refetch_from = (
        newest_known - refetch_last + 1
        if newest_known is not None and refetch_last > 0
        else last + 1
    )
    wanted = [e for e in range(first, last + 1)
              if e not in known or e >= refetch_from]
    done = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_epoch, e): e for e in wanted}
        for future in concurrent.futures.as_completed(futures):
            epoch = futures[future]
            try:
                rows = future.result()
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    json.JSONDecodeError, OSError):
                failed += 1               # an unscored future epoch 404s; normal
                continue
            if store_epoch(connection, epoch, rows):
                done += 1
    return dict(requested=len(wanted), stored=done, unavailable=failed)


def collect_slots(connection, log: Path) -> dict:
    """Ingest the slot-manager decision log, which is append-only JSONL."""
    if not log.is_file():
        return dict(slot_rows=0, note=f"no slot log at {log}")
    kept = 0
    for line in log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("time") is None:
            continue                      # early rows predate the timestamp
        connection.execute(
            "INSERT OR REPLACE INTO slots (time, epoch, block, registered, target,"
            " recycle_tao, decision_recycle_tao, executed, reason, hotkey)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (record["time"], record.get("epoch"), record.get("block"),
             record.get("registered"), record.get("target"),
             record.get("recycle_tao"), record.get("decision_recycle_tao"),
             int(bool(record.get("executed"))), record.get("reason"),
             record.get("hotkey")))
        kept += 1
    connection.commit()
    return dict(slot_rows=kept)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--first", type=int, default=None,
                        help="first epoch to collect (default: 200 before --last)")
    parser.add_argument("--last", type=int, default=None,
                        help="last epoch (default: derived from the chain clock)")
    parser.add_argument("--history", type=int, default=200)
    parser.add_argument("--slot-log", type=Path,
                        default=Path.home() / ".nova" / "subnet68_slots.jsonl")
    parser.add_argument("--coldkey", default=DEFAULT_COLDKEY)
    arguments = parser.parse_args()

    connection = connect(arguments.database)
    last = arguments.last
    if last is None:
        known = have(connection)
        # Probe forward from the newest stored epoch rather than trusting a
        # local clock: the dashboard must not invent epochs that do not exist.
        # Probe only just beyond the newest real epoch.  The old +12 window
        # generated twelve predictable 404s on every daemon tick and delayed
        # the useful newest-epoch refetch.
        last = (max(known) if known else 25256) + 2
    first = arguments.first if arguments.first is not None else last - arguments.history
    report = collect_leaderboard(connection, first, last)
    report.update(collect_slots(connection, arguments.slot_log))
    known = have(connection)
    report.update(epochs_held=len(known),
                  newest=max(known) if known else None,
                  oldest=min(known) if known else None)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
