"""Proxy model retraining accuracy, per reaction, over time.

Separate from compound.py's gap_history: that tracks submission readiness
(is the SELECTED portfolio good enough to clear the floor); this tracks
whether the proxy model driving candidate search is actually getting more
accurate as it retrains. No accuracy metric existed anywhere in the
retraining pipeline before 2026-10-02 -- fit_proxy_on_box.py (run every 4h
by proxy_refit_loop.sh on all 5 reaction boxes) only logged train_rows and
elapsed_seconds. held_out_proxy_eval.py already computed the right thing
(Spearman rho + MAE on a held-out split) but was a manual-only diagnostic,
never wired into the automatic loop -- this module collects its now-
automatic output.

Read-only: every SSH command is a plain ``cat`` of a log file. Nothing here
writes to, restarts, or competes with any live mining process.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from .compound import REACTION_BOXES, _ssh_port

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "proxy_accuracy_status.json"
DATABASE_PATH = ROOT / "data" / "nanobody_dash.sqlite"

PROXY_REFIT_LOG = "search/proxy_refit.log"

# proxy_accuracy_history is this module's own table in the shared dashboard
# database -- same "each module owns its own tables in the shared file"
# convention as economics.py and compound.py's gap_history. Composite
# (reaction, at) key: one retrain cycle per reaction every ~4h, "at" is the
# real timestamp of that cycle's accuracy-eval result.
SCHEMA = """
CREATE TABLE IF NOT EXISTS proxy_accuracy_history (
    reaction INTEGER NOT NULL,
    at REAL NOT NULL,
    rho REAL,
    p REAL,
    mae REAL,
    n INTEGER,
    train_rows INTEGER,
    PRIMARY KEY (reaction, at)
);
CREATE INDEX IF NOT EXISTS proxy_accuracy_reaction ON proxy_accuracy_history(reaction, at);
"""


def connect(database: Path = DATABASE_PATH) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    return connection


def proxy_accuracy_rows(connection: sqlite3.Connection) -> list[dict]:
    """Read proxy_accuracy_history with an already-open connection (build.py's
    own -- never open a second one). Ensures the table exists first since
    build.py may run before this module's collect_all() ever has."""
    connection.executescript(SCHEMA)
    cursor = connection.execute(
        "SELECT reaction, at, rho, p, mae, n, train_rows "
        "FROM proxy_accuracy_history ORDER BY reaction, at")
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# Matches the ISO timestamp markers proxy_refit_loop.sh writes around each
# step, e.g. "[2026-10-02T20:15:03Z] refit done, running accuracy eval".
_AT_MARKER_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]')
_RESULT_RE = re.compile(r'\{"event": "result".*\}')
_FIT_COMPLETE_RE = re.compile(
    r'\{"event": "fit_complete", "reaction": (\d+), "train_rows": (\d+)')


def _parse_refit_log(text: str) -> list[dict]:
    """Pull accuracy-eval results out of proxy_refit.log, paired with the
    nearest preceding ISO-timestamp marker and the most recent fit_complete's
    train_rows -- same "nearest preceding event in document order" technique
    compound.py's _parse_loop_log uses to attach epoch/reaction to a
    release_below_baseline line that doesn't carry them itself.

    The "result" JSON line is emitted by held_out_proxy_eval.py DURING the
    eval, so it always falls between the "refit done, running accuracy eval"
    marker (eval start) and "accuracy eval done" marker (eval finish) --
    nearest-preceding therefore pairs each result with its eval's START
    time, not finish. That's deliberate: start-of-eval is the more faithful
    "when was this retrain cycle's model tested" timestamp.
    """
    results: list[dict] = []
    current_at: str | None = None
    current_train_rows: int | None = None
    for line in text.splitlines():
        marker = _AT_MARKER_RE.match(line)
        if marker:
            current_at = marker.group(1)
            continue
        fit_complete = _FIT_COMPLETE_RE.search(line)
        if fit_complete:
            current_train_rows = int(fit_complete.group(2))
            continue
        result = _RESULT_RE.search(line)
        if not result:
            continue
        try:
            payload = json.loads(result.group(0))
        except json.JSONDecodeError:
            continue
        if current_at is None:
            continue
        try:
            # calendar.timegm treats the struct_time as already-UTC and
            # converts straight to a Unix timestamp -- unlike time.mktime,
            # which assumes local time and would need a DST-aware correction
            # (time.timezone alone is wrong half the year). This text is
            # always UTC (date -u in proxy_refit_loop.sh), so this is exact.
            at_ts = calendar.timegm(time.strptime(current_at, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            continue
        results.append(dict(
            at=at_ts, rho=payload.get("rho"), p=payload.get("p"),
            mae=payload.get("mae"), n=payload.get("n"),
            train_rows=current_train_rows,
        ))
    return results


def collect_reaction(reaction: int, timeout: int = 30) -> list[dict] | None:
    """SSH one reaction box's proxy_refit.log and parse it. Returns None on
    an SSH/connection failure (box unreachable) and [] when the box is
    reachable but its log simply has no accuracy-eval result yet (e.g.
    right after a restart, before the first retrain cycle completes) --
    these are different conditions and callers must not conflate them."""
    user_host, port, _floor = REACTION_BOXES[reaction]
    text = _ssh_port(user_host, port, "cat", PROXY_REFIT_LOG, timeout=timeout)
    if text is None:
        return None
    return _parse_refit_log(text)


def collect_all() -> dict:
    """One full snapshot across all 5 reaction boxes, and persist new rows
    into proxy_accuracy_history (INSERT OR REPLACE, so re-running is
    idempotent -- a result already seen just gets rewritten with the same
    values)."""
    by_reaction: dict[int, list[dict]] = {}
    reachable = 0
    db = connect()
    try:
        for reaction in REACTION_BOXES:
            rows = collect_reaction(reaction)
            if rows is None:
                continue
            reachable += 1
            by_reaction[reaction] = rows
            for row in rows:
                db.execute(
                    "INSERT OR REPLACE INTO proxy_accuracy_history "
                    "(reaction, at, rho, p, mae, n, train_rows) VALUES (?,?,?,?,?,?,?)",
                    (reaction, row["at"], row["rho"], row["p"], row["mae"],
                     row["n"], row["train_rows"]))
        db.commit()
    finally:
        db.close()
    return dict(
        collected_at=time.time(),
        reachable_boxes=reachable,
        latest_by_reaction={
            str(r): rows[-1] for r, rows in by_reaction.items() if rows
        },
    )


def save(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    staged = STATUS_PATH.with_suffix(".json.next")
    staged.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    os.chmod(staged, 0o600)
    os.replace(staged, STATUS_PATH)


def load() -> dict:
    if not STATUS_PATH.is_file():
        return {}
    try:
        return json.loads(STATUS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    """Run as `python -m nanobody_dash.proxy_accuracy`, matching
    compound.py's own collector entry point -- refresh.py invokes this as a
    bounded-timeout subprocess, not in-process, so a hung SSH call can never
    block the rest of the dashboard refresh indefinitely."""
    status = collect_all()
    save(status)
    print(json.dumps(dict(
        reachable_boxes=status.get("reachable_boxes"),
        latest_by_reaction={
            k: {kk: vv for kk, vv in v.items() if kk in ("rho", "mae", "n")}
            for k, v in (status.get("latest_by_reaction") or {}).items()
        },
    )))
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
