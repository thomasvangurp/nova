"""SN68 compound-mining fleet operational status: are we ready to submit, how
often do we actually clear, how often do we actually win.

Deliberately separate from compound's infra-cost classification (see
infra.py's ``_COMPOUND_LABEL``/``cost_scope`` -- that part is cheap, live
Vast.ai API data, refreshed every cycle). This module is expensive: it SSHes
into the Pi's submission log and, for the live per-reaction gap check, into
each of the 5 dedicated mining boxes to run the real submission gate's
own entropy-enforced selection. Run on its own long cadence (see refresh.py's
``_should_refresh_compound``/``COMPOUND_REFRESH_SECONDS``), not every 60s --
collect() writes one JSON snapshot that build() just reads, so a slow or
failed SSH round never blocks the rest of the dashboard refresh.

Read-only throughout: every SSH command is either a plain ``cat``/``grep`` of
a log file or a ``SELECT`` against a sqlite3 URI opened ``mode=ro``. Nothing
here writes to, restarts, or competes meaningfully with any live mining
process.
"""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import sqlite3
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "data" / "compound_status.json"
DATABASE_PATH = ROOT / "data" / "nanobody_dash.sqlite"

# gap_history is compound.py's own table in the shared dashboard database --
# same pattern as economics.py owning delegations/transfers/etc. in the same
# file. Composite (checked_at, reaction) key, not a bare time PK like slots'
# table: one collect() cycle writes 5 rows (one per reaction) at slightly
# different real timestamps, and correctness shouldn't depend on those being
# distinct by accident.
SCHEMA = """
CREATE TABLE IF NOT EXISTS gap_history (
    checked_at REAL NOT NULL,
    reaction INTEGER NOT NULL,
    selected_score REAL,
    floor REAL,
    gap REAL,
    clears_now INTEGER,
    pool_size INTEGER,
    PRIMARY KEY (checked_at, reaction)
);
CREATE INDEX IF NOT EXISTS gap_history_reaction ON gap_history(reaction, checked_at);
"""


def connect(database: Path = DATABASE_PATH) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript(SCHEMA)
    return connection


def gap_history_rows(connection: sqlite3.Connection) -> list[dict]:
    """Read gap_history with an already-open connection (build.py's own --
    never open a second one). Ensures the table exists first since build.py
    may run before compound.py's collect() ever has."""
    connection.executescript(SCHEMA)
    cursor = connection.execute(
        "SELECT checked_at, reaction, selected_score, floor, gap, clears_now, pool_size "
        "FROM gap_history ORDER BY reaction, checked_at")
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

PI_HOST = os.environ.get("NOVA_PI_SSH", "thomasg@100.64.243.65")
PI_LOOP_LOG = "/opt/nova-submit/loop.log"

# host:port and real submission floor per reaction, confirmed live 2026-10-02
# against the Pi's actual running pi_epoch_loop.py process args
# (--minimum-compound-score-map).
REACTION_BOXES = {
    1: ("root@ssh6.vast.ai", 38762, 1.61),  # 54154421 (replaced dead 53168856, 2026-10-04)
    2: ("root@ssh1.vast.ai", 13786, 1.93),
    3: ("root@ssh3.vast.ai", 34868, 1.56),  # 54556994 (replaced broken 53172464, 2026-10-07)
    4: ("root@ssh7.vast.ai", 23482, 1.87),  # 54163340 (replaced unreachable 53174363, 2026-10-04)
    5: ("root@ssh5.vast.ai", 30358, 1.95),
}

LEADERBOARD_API = (
    "https://compound-api.metanova-labs.ai/api/competitions/leaderboard/"
    "{epoch}/molecules?limit=10000"
)

# Measured 2026-10-02: offline (de-noised, 3-pass) selected_score vs the real
# on-chain score, across 9 epochs with clean (non -999.99) on-chain data.
# Context for the dashboard reader: clearing the baseline offline is not the
# same as being likely to actually win -- see the note this feeds into the
# rendered tab.
WINNERS_CURSE_GAP = dict(mean=0.1285, median=0.0945, stdev=0.0879, n=9)


def _ssh(host: str, *args: str, timeout: int = 30) -> str | None:
    """Run one read-only remote command. Returns stdout, or None on any
    failure -- a Pi/box outage must not take down the whole collector."""
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _ssh_port(user_host: str, port: int, *args: str, timeout: int = 60) -> str | None:
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             # rebuilt producer boxes come with new proxy ports; accept-new adds
             # an unseen host key but still refuses a CHANGED one
             "-o", "StrictHostKeyChecking=accept-new",
             "-p", str(port), user_host, *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _local_hhmm(timestamp: float) -> str:
    import datetime as _dt
    moment = _dt.datetime.fromtimestamp(timestamp).astimezone()
    name = moment.tzname() or ""
    name = {"-04": "EDT", "-05": "EST"}.get(name, name or "local")  # pixi's python lacks tz names
    return moment.strftime("%H:%M ") + name


def _parse_loop_log(text: str) -> dict:
    """Pull window_open (deduped by epoch -- retries log it more than once),
    allocated (a real clear), and release_below_baseline (a blocked attempt)
    events out of the Pi's loop.log. The error text on a blocked attempt is
    not clean JSON (it is a truncated/concatenated string ending in
    "selection X does not beat baseline Y"), so this is regex extraction,
    not json.loads, for that one event type.
    """
    window_epoch_reaction: dict[int, int] = {}
    window_epoch_at: dict[int, str] = {}
    for match in re.finditer(
        r'"at": "(\d{2}:\d{2}:\d{2})", "event": "window_open", '
        r'"epoch": (\d+), "reaction": (\d+)', text,
    ):
        epoch = int(match.group(2))
        window_epoch_reaction[epoch] = int(match.group(3))
        window_epoch_at[epoch] = match.group(1)

    allocated: list[dict] = []
    for match in re.finditer(
        r'"event": "allocated", "uids": (\d+), "epoch": (\d+), "reaction": (\d+), '
        r'"best": ([\d.]+), "worst": ([\d.]+), "entropy_min": ([\d.]+), "distinct": (\d+)',
        text,
    ):
        allocated.append(dict(
            uids=int(match.group(1)), epoch=int(match.group(2)), reaction=int(match.group(3)),
            best=float(match.group(4)), worst=float(match.group(5)),
            entropy_min=float(match.group(6)), distinct=int(match.group(7)),
        ))
    # de-dupe by epoch (a retried commit can log the same allocation twice)
    allocated = list({row["epoch"]: row for row in allocated}.values())

    # Since 2026-10-06, ``allocated`` means only that candidate portfolios
    # were constructed. A second, exact post-allocation gate decides whether
    # those portfolios are safe to submit. Associate that decision with the
    # current window and do not count a rejected allocation as a clear.
    final_gates: dict[int, dict] = {}
    current_window = None
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if event.get("event") == "window_open":
            current_window = {
                "epoch": int(event["epoch"]), "reaction": int(event["reaction"]),
            }
            continue
        if event.get("event") not in ("fleet_gate_passed", "fleet_below_baseline") \
                or current_window is None:
            continue
        final_gates[current_window["epoch"]] = {
            **current_window,
            "passed": event["event"] == "fleet_gate_passed",
            **{key: event.get(key) for key in (
                "raw_floor", "worst_raw", "prediction_offset", "realized_floor",
                "worst_predicted_realized",
            )},
        }
    rejected_epochs = {epoch for epoch, gate in final_gates.items() if not gate["passed"]}
    allocated = [row for row in allocated if row["epoch"] not in rejected_epochs]

    blocked: list[dict] = []
    for match in re.finditer(
        r'"event": "release_below_baseline", "err": "(.*?)"\}', text,
    ):
        err = match.group(0)
        score_match = re.search(r'selection ([\d.]+) does not beat baseline ([\d.]+)', err)
        if not score_match:
            continue
        entry = dict(selected_score=float(score_match.group(1)), baseline=float(score_match.group(2)))
        rich = re.search(
            r'"search_pool": (\d+).*?"top_count_score": ([\d.]+).*?'
            r'"top_count_entropy": ([\d.]+).*?"swapped_in": (\d+)', err,
        )
        if rich:
            entry.update(
                search_pool=int(rich.group(1)), top_count_score=float(rich.group(2)),
                top_count_entropy=float(rich.group(3)), swapped_in=int(rich.group(4)),
            )
        blocked.append(entry)

    # Attach the blocked-entry list to the nearest preceding window_open
    # epoch/reaction in document order (blocked entries don't carry epoch
    # themselves) by re-walking the raw text in order.
    ordered_events: list[tuple[int, dict]] = []  # (position, {"type":, epoch, reaction, ...})
    for match in re.finditer(r'"event": "window_open", "epoch": (\d+), "reaction": (\d+)', text):
        ordered_events.append((match.start(), dict(
            type="window", epoch=int(match.group(1)), reaction=int(match.group(2)),
        )))
    for match in re.finditer(r'"event": "release_below_baseline", "err": "(.*?)"\}', text):
        ordered_events.append((match.start(), dict(type="blocked", raw=match.group(0))))
    ordered_events.sort(key=lambda pair: pair[0])

    blocked_with_epoch: list[dict] = []
    current = None
    for _, event in ordered_events:
        if event["type"] == "window":
            current = event
            continue
        if current is None:
            continue
        score_match = re.search(
            r'selection ([\d.]+) does not beat baseline ([\d.]+)', event["raw"],
        )
        if not score_match:
            continue
        entry = dict(
            epoch=current["epoch"], reaction=current["reaction"],
            selected_score=float(score_match.group(1)), baseline=float(score_match.group(2)),
        )
        rich = re.search(
            r'"top_count_entropy": ([\d.]+).*?"swapped_in": (\d+)', event["raw"],
        )
        if rich:
            entry.update(
                top_count_entropy=float(rich.group(1)), swapped_in=int(rich.group(2)),
            )
        blocked_with_epoch.append(entry)
    blocked_with_epoch = list({
        (row["epoch"], row["reaction"]): row for row in blocked_with_epoch
    }.values())
    for epoch in sorted(rejected_epochs):
        gate = final_gates[epoch]
        blocked_with_epoch.append({
            "epoch": epoch,
            "reaction": gate["reaction"],
            "selected_score": gate.get("worst_raw"),
            "baseline": gate.get("raw_floor"),
            "prediction_offset": gate.get("prediction_offset"),
            "predicted_realized": gate.get("worst_predicted_realized"),
            "realized_floor": gate.get("realized_floor"),
            "final_gate": True,
        })
    blocked_with_epoch = list({
        (row["epoch"], row["reaction"]): row for row in blocked_with_epoch
    }.values())

    # (at, epoch, reaction) in log order -- epochs only ever increase, so
    # sorting by epoch reproduces file/chronological order without needing
    # to track match positions again.
    ordered_window_events = sorted(
        (
            (window_epoch_at[epoch], epoch, reaction)
            for epoch, reaction in window_epoch_reaction.items()
            if epoch in window_epoch_at
        ),
        key=lambda row: row[1],
    )

    return dict(
        window_epoch_reaction=window_epoch_reaction,
        allocated=allocated,
        blocked=blocked_with_epoch,
        final_gates=final_gates,
        ordered_window_events=ordered_window_events,
    )


def _attach_wallclock(
    ordered_window_events: list[tuple[str, int, int]], collected_at: float,
) -> dict[tuple[int, int], float]:
    """HH:MM:SS-only log lines -> real unix timestamps, for the (epoch,
    reaction) pairs in ordered_window_events (file/chronological order).

    The Pi logs only time-of-day (no date), in a fixed -04:00 offset
    (America/La_Paz, confirmed same offset as US Eastern in October). Two
    phases: first place every line RELATIVE to each other on an arbitrary
    placeholder date, incrementing a running day-offset whenever time-of-day
    drops by more than 12 hours (the only way that can happen in an
    append-only, oldest-first log is a real midnight rollover). Then anchor
    the WHOLE batch to real time in one shot: shift every placeholder
    timestamp by whatever whole number of days makes the newest line line up
    with `collected_at` (time.time() at the moment this collect() call ran,
    always at or shortly after the real time of the newest line). Only
    resolves one day-ambiguity for the entire batch, not per line -- adequate
    for loop.log's actual retention window (a few days), not hardened against
    multi-day gaps with zero window_open events spanning a rollover.
    """
    if not ordered_window_events:
        return {}

    pi_tz = timezone(timedelta(hours=-4))
    day_offset = 0
    previous_seconds_of_day = None
    provisional: list[tuple[int, int, float]] = []  # (epoch, reaction, ts)
    for at, epoch, reaction in ordered_window_events:
        hh, mm, ss = (int(part) for part in at.split(":"))
        seconds_of_day = hh * 3600 + mm * 60 + ss
        if previous_seconds_of_day is not None and \
                seconds_of_day < previous_seconds_of_day - 12 * 3600:
            day_offset += 1
        previous_seconds_of_day = seconds_of_day
        local_dt = datetime(2000, 1, 1, hh, mm, ss, tzinfo=pi_tz) \
            + timedelta(days=day_offset)
        provisional.append((epoch, reaction, local_dt.timestamp()))

    # provisional timestamps are anchored to an arbitrary placeholder date
    # (2000-01-01), so this correction is ALWAYS a large number of days in
    # real use -- that's expected, not a sign anything is wrong. What
    # actually needs sanity-checking is the ROUNDING: it picks the correct
    # day only if the newest event's time-of-day is within 12h of
    # collected_at's time-of-day, which normally holds (collect() runs on a
    # ~20 min cadence, so the newest window_open is almost always recent).
    newest_provisional_ts = provisional[-1][2]
    delta_days = round((collected_at - newest_provisional_ts) / 86400)
    shift = delta_days * 86400
    corrected_newest = newest_provisional_ts + shift

    # Sanity guard: the newest real event should land shortly before (never
    # meaningfully after) collected_at. A few minutes of slack covers normal
    # SSH/collection latency and Pi/Mac clock skew; anything further off
    # means the rollover resolution above likely picked the wrong day --
    # degrade to "no timestamps" rather than plot nonsense (same "never
    # block or corrupt the tab" ethos as collect()'s pi_reachable fallback).
    if not (-6 * 3600 <= corrected_newest - collected_at <= 300):
        return {}

    return {
        (epoch, reaction): ts + shift
        for epoch, reaction, ts in provisional
    }


def _committed_uids_by_epoch(text: str, epochs: list[int]) -> dict[int, list[int]]:
    """For each target epoch, walk forward from its "plan" event and collect
    the uids whose commit carried real molecules, stopping at "done"."""
    lines = text.splitlines()
    plan_at: dict[int, int] = {}
    for index, line in enumerate(lines):
        match = re.search(r'"event": "plan", "epoch": (\d+)', line)
        if match and int(match.group(1)) in epochs:
            plan_at[int(match.group(1))] = index
    out: dict[int, list[int]] = {}
    for epoch, start in plan_at.items():
        uids = []
        for line in lines[start:start + 20]:
            commit_match = re.search(
                r'"event": "commit", "uid": (\d+), "rc": 0, "molecules": true', line,
            )
            if commit_match:
                uids.append(int(commit_match.group(1)))
            if '"event": "done"' in line:
                break
        out[epoch] = uids
    return out


def _leaderboard(epoch: int) -> list[dict] | None:
    try:
        request = urllib.request.Request(
            LEADERBOARD_API.format(epoch=epoch), headers={"User-Agent": "nova-compound-dash/1"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    return payload.get("data", [])


def _real_outcome(epoch: int, uids: list[int]) -> dict | None:
    """Our best rank/score for one epoch's real on-chain field, among the
    uids we actually committed molecules for that epoch."""
    submissions = _leaderboard(epoch)
    if submissions is None:
        return None
    ranked = sorted(submissions, key=lambda item: -float(item.get("final_score") or -999999))
    rank_by_uid = {
        int(item["uid"]): (position + 1, float(item.get("final_score") or -999999))
        for position, item in enumerate(ranked)
    }
    best_rank, best_score = min(
        (rank_by_uid.get(uid, (10**9, -999999.0)) for uid in uids),
        key=lambda pair: pair[0],
        default=(None, None),
    )
    winner_score = float(ranked[0].get("final_score") or 0) if ranked else None
    if best_rank is None or best_rank >= 10**9:
        return dict(field_size=len(ranked), rank=None, score=None, won=False,
                    winner_score=winner_score)
    # -999.99 is the validator's "not scored" sentinel: the submission was
    # rejected (e.g. epoch 25464), not merely outscored.
    rejected = best_score <= -999
    return dict(field_size=len(ranked), rank=None if rejected else best_rank,
                score=None if rejected else best_score, won=best_rank == 1 and not rejected,
                rejected=rejected, winner_score=winner_score)


# The live check runs the Pi's own release gate (a copy of
# /opt/nova-submit/tools/bank_release.py deployed to each box) instead of a
# hand-written replica. The replica it replaced skipped the gate's two hard
# filters -- >=3 rescore passes and novelty against the growing public archive
# -- and so showed rxn 3 "clearing" at 1.566 for hours while the real gate
# scored 1.467: 887 of 1261 eligible rxn 3 molecules were refused as non-novel
# (2026-10-02). Never re-replicate the gate here; redeploy the copy instead.
GATE_REMOTE = "tools/dash_release_gate.py"
ARCHIVE_REMOTE = "artifacts/rxn3-extratrees-20260924/P40261_molecules.csv"
SEARCH_ARTIFACTS_REMOTE = "artifacts/rxn3-extratrees-20260924"
SEARCH_BANK_REMOTE = f"{SEARCH_ARTIFACTS_REMOTE}/rxn3-bank-live.sqlite"
AB_PERFORMANCE_REMOTE = "tools/competitor_motif_ab_performance.py"
LANE_PERFORMANCE_REMOTE = "tools/lane_performance.py"
STRATEGY_POLICY_VERSION = "uplift-v1"


def _gate_events(text: str) -> dict:
    events = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in row:
                events[row["event"]] = row
    return events


SUBMITTED_LOCAL = Path(__file__).resolve().parents[1] / "data" / "submitted_archive"
SUBMITTED_REMOTE = "artifacts/submitted_archive"


def _pull_submitted_archive() -> bool:
    """Copy the Pi's per-epoch committed compound_fleet.json snapshots locally,
    so every box's gate run excludes molecules we already submitted -- the
    leaderboard does not list an epoch until it is scored, and the boxes have
    no copy of the Pi's archive. Without this a reaction we just submitted
    keeps reading its pre-submission (already spent) score."""
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", PI_HOST,
             "cd /opt/nova-submit/archive && tar cf - epoch*/compound_fleet.json 2>/dev/null"],
            capture_output=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if not completed.stdout:
        return False
    SUBMITTED_LOCAL.mkdir(parents=True, exist_ok=True)
    extracted = subprocess.run(["tar", "xf", "-", "-C", str(SUBMITTED_LOCAL)],
                               input=completed.stdout, capture_output=True, check=False)
    return extracted.returncode == 0


def _push_submitted_archive(user_host: str, port: int) -> bool:
    if not SUBMITTED_LOCAL.is_dir():
        return False
    try:
        packed = subprocess.run(["tar", "cf", "-", "-C", str(SUBMITTED_LOCAL), "."],
                                capture_output=True, check=False).stdout
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-p", str(port), user_host,
             f"mkdir -p ~/search/{SUBMITTED_REMOTE} && tar xf - -C ~/search/{SUBMITTED_REMOTE}"],
            input=packed, capture_output=True, timeout=60, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


FLOOR_WINDOW = 5
FLOOR_MIN_SAMPLES = 3
FLOOR_OPTIMISM_MARGIN = 0.10  # must match pi_epoch_loop.py
FLOOR_PINNED: dict = {}  # must match pi_epoch_loop.py
FLOOR_BURST_LOOKBACK = 5  # must match pi_epoch_loop.py
FLOOR_BURST_MIN = 3  # must match pi_epoch_loop.py
FLOOR_LOOKBACK_EPOCHS = 150  # must match pi_epoch_loop.py


def _calibrated_optimism(evidence: list[dict], prior: float,
                         prior_epochs: int = 2) -> tuple[float, dict]:
    """Mirror the Pi's empirical-Bayes raw-minus-realised calibration."""
    by_epoch: dict[int, list[float]] = {}
    for row in evidence:
        try:
            by_epoch.setdefault(int(row["epoch"]), []).append(float(row["optimism"]))
        except (KeyError, TypeError, ValueError):
            continue
    epoch_values = [statistics.median(by_epoch[epoch]) for epoch in sorted(by_epoch)][-8:]
    if not epoch_values:
        return prior, {"own_evidence_epochs": 0, "own_evidence_portfolios": 0}
    observed = statistics.median(epoch_values)
    n = len(epoch_values)
    estimate = min(0.30, max(0.0, (n * observed + prior_epochs * prior) /
                                 (n + prior_epochs)))
    return estimate, {
        "own_evidence_epochs": n,
        "own_evidence_portfolios": len(evidence),
        "observed_optimism": round(observed, 4),
    }


def _dynamic_floors_from_cache(cache: dict, next_epoch: int) -> tuple[dict, dict]:
    """Pure mirror of the Pi's calibrated external-rival floor calculation."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import rival_margin

    floors = {r: REACTION_BOXES[r][2] for r in REACTION_BOXES}
    details: dict[int, dict] = {}
    calibration = cache.get("_self_calibration") or {}
    owned = {str(value) for value in calibration.get("coldkeys") or []}
    evidence = calibration.get("evidence") or []
    rows = []
    for epoch, entries in cache.items():
        if not str(epoch).isdigit() or not isinstance(entries, list):
            continue
        epoch_number = int(epoch)
        if not max(0, next_epoch - FLOOR_LOOKBACK_EPOCHS) <= epoch_number < next_epoch:
            continue
        for row in entries:
            try:
                rows.append({
                    "epoch": epoch_number, "coldkey": str(row["coldkey"]),
                    "reaction": int(row["reaction"]), "score": float(row["score"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    external_rows = [row for row in rows if row["coldkey"] not in owned]
    for reaction in REACTION_BOXES:
        if reaction in FLOOR_PINNED:
            floors[reaction] = FLOOR_PINNED[reaction]
            details[reaction] = {"source": "user_pinned", "margin": floors[reaction]}
            continue
        reaction_evidence = [row for row in evidence
                             if str(row.get("reaction")) == str(reaction)]
        optimism, own = _calibrated_optimism(reaction_evidence, rival_margin.CURSE)
        optimism_by_epoch: dict[int, list[float]] = {}
        for row in reaction_evidence:
            try:
                optimism_by_epoch.setdefault(int(row["epoch"]), []).append(
                    float(row["optimism"]))
            except (KeyError, TypeError, ValueError):
                continue
        optimism_samples = [
            statistics.median(optimism_by_epoch[epoch])
            for epoch in sorted(optimism_by_epoch)
        ][-8:]
        margin, info = rival_margin.required_margin(
            reaction, external_rows, next_epoch, optimism=optimism,
            optimism_samples=optimism_samples)
        if margin is not None:
            floors[reaction] = margin
        details[reaction] = {
            **info, **own, "source": "rival_projection" if margin is not None else "static_fallback",
            "owned_coldkeys_excluded": len(owned), "external_rows": len(external_rows),
            "floor": floors[reaction],
        }
    return floors, details


def _dynamic_floors(next_epoch: int) -> tuple[dict, dict]:
    """Required margin per reaction from the rival-supply model, the same rule
    the Pi's dynamic_floor enforces: external-rival projected winner, corrected
    rival bias, plus our calibrated raw-minus-realised optimism. Reads the Pi's
    rival_cache.json and excludes our own coldkey. Falls back to static floors."""
    floors = {r: REACTION_BOXES[r][2] for r in REACTION_BOXES}
    text = _ssh(PI_HOST, "cat", "/opt/nova-submit/archive/rival_cache.json", timeout=60)
    try:
        cache = json.loads(text) if text else {}
    except ValueError:
        cache = {}
    return _dynamic_floors_from_cache(cache, next_epoch) if cache else (floors, {})


DYNAMIC_FLOORS: dict = {}
DYNAMIC_FLOOR_DETAILS: dict = {}


def _live_gap_check(reaction: int, epoch: int = 0) -> dict | None:
    user_host, port, floor = REACTION_BOXES[reaction]
    floor = DYNAMIC_FLOORS.get(reaction, floor)
    _push_submitted_archive(user_host, port)
    command = (
        "cd ~/search && uv run --no-project --with rdkit --with pandas --with numpy "
        f"python {GATE_REMOTE} --bank legacy_bank.sqlite --reaction {reaction} --epoch {epoch} "
        f"--archive {ARCHIVE_REMOTE} --no-refresh-archive --baseline {floor} --fleet 1 "
        f"--submitted-archive {SUBMITTED_REMOTE} "
        f"--output /tmp/dash_gate_rxn{reaction}.json 2>&1"
    )
    try:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-n", "-p", str(port),
             user_host, command],
            capture_output=True, text=True, timeout=300, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    events = _gate_events(completed.stdout)
    observed = events.get("observation_filter") or {}
    novelty = events.get("history_filter") or {}
    recent = events.get("recent_submission_filter") or {}
    selection = events.get("entropy_selection")
    detail = dict(
        floor=floor,
        bank_rows=observed.get("considered"),
        undenoised_dropped=observed.get("dropped_undenoised"),
        non_novel_refused=novelty.get("refused"),
        recently_submitted_refused=recent.get("refused"),
        pool_size=recent.get("kept", novelty.get("kept", observed.get("kept"))),
    )
    if selection is None:
        tail = [line for line in completed.stdout.splitlines() if line.strip()]
        return dict(detail, error=(tail[-1] if tail else "no output")[:200])
    score = float(selection["selected_score"])
    return dict(detail, selected_score=score, gap=score - floor, clears_now=score >= floor,
                entropy=selection.get("selected_entropy"))


def _refresh_boxes_from_pi() -> None:
    """Override REACTION_BOXES host:port with the producer watchdog's live addresses
    (/opt/nova-submit/producer_boxes.json on the Pi, vast API + identity check, every
    5 min), so a rebuilt box is picked up without editing this file. Floors and any
    reaction missing from a fresh (<30 min) file keep their static entries."""
    text = _ssh(PI_HOST, "cat", "/opt/nova-submit/producer_boxes.json", timeout=30)
    try:
        data = json.loads(text or "")
        if time.time() - float(data.get("updated_unix", 0)) > 1800:
            return
        for key, box in (data.get("boxes") or {}).items():
            reaction = int(key)
            if reaction in REACTION_BOXES and box.get("host") and box.get("port"):
                REACTION_BOXES[reaction] = (f"root@{box['host']}", int(box["port"]),
                                            REACTION_BOXES[reaction][2])
    except (ValueError, TypeError, AttributeError):
        return


def _last_json_object(text: str | None) -> dict | None:
    for line in reversed((text or "").splitlines()):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _strategy_verdict(ab: dict, lanes: dict) -> dict:
    """Separate relative selection signal from absolute chemistry progress."""
    lane_metrics = (lanes.get("lanes") or {}) if isinstance(lanes, dict) else {}
    active_strategy = str(lanes.get("active_strategy") or "")
    if active_strategy.startswith("field-recombination-"):
        control = lane_metrics.get("block_bandit/field_recombination/random") or {}
        treatment = lane_metrics.get("block_bandit/field_recombination/proxy") or {}
        control_n = int(control.get("scored") or 0)
        treatment_n = int(treatment.get("scored") or 0)
        relative_lift = (
            float(treatment["mean"]) - float(control["mean"])
            if isinstance(treatment.get("mean"), (int, float)) and
            isinstance(control.get("mean"), (int, float)) else None
        )
        if min(control_n, treatment_n) < 20:
            status, label = "collecting", "collecting field A/B"
        elif int(treatment.get("ge_entry") or 0) > 0:
            status, label = "breakthrough", "working: bank-entry hit"
        elif relative_lift is not None and relative_lift > 0:
            status, label = "relative_only", "field ranker beats random"
        elif relative_lift is not None:
            status, label = "worse", "field ranker trails random"
        else:
            status, label = "inconclusive", "inconclusive"
        base_lanes = [(name, values) for name, values in lane_metrics.items()
                      if "/" not in name and isinstance(values, dict)
                      and isinstance(values.get("max"), (int, float))]
        best_name, best = max(base_lanes, key=lambda item: item[1]["max"],
                              default=(None, {}))
        return {
            "mode": "field_recombination", "status": status, "label": label,
            "relative_lift": relative_lift, "control_n": control_n,
            "treatment_n": treatment_n, "entry_hits": treatment.get("ge_entry"),
            "treatment_max": treatment.get("max"), "entry_score": lanes.get("entry_score"),
            "geometry": lanes.get("active_geometry"), "block_share": lanes.get("block_share"),
            "motif_share": lanes.get("motif_share"), "best_lane": best_name,
            "best_lane_max": best.get("max"),
        }

    arms = ab.get("arms") or {}
    control = arms.get("control") or {}
    treatment = arms.get("treatment") or {}
    comparison = arms.get("comparison") or {}
    minimum = int((ab.get("verdict") or {}).get("minimum_per_arm", 20))
    if min(int(control.get("n") or 0), int(treatment.get("n") or 0)) < minimum:
        status = "collecting"
        label = "collecting exact A/B"
    elif int(treatment.get("ge_entry") or 0) > 0:
        status = "breakthrough"
        label = "working: bank-entry hit"
    elif (comparison.get("mean_source_delta_difference") or 0) > 0:
        status = "relative_only"
        label = "relative signal; no entry hit"
    elif (ab.get("verdict") or {}).get("status") == "worse":
        status = "worse"
        label = "worse than control"
    else:
        status = "inconclusive"
        label = "inconclusive"

    base_lanes = [(name, values) for name, values in lane_metrics.items()
                  if "/" not in name and isinstance(values, dict)
                  and isinstance(values.get("max"), (int, float))]
    best_name, best = max(base_lanes, key=lambda item: item[1]["max"],
                          default=(None, {}))
    return {
        "mode": "legacy_motif", "status": status,
        "label": label,
        "relative_lift": comparison.get("mean_source_delta_difference"),
        "treatment_source_delta": treatment.get("mean_source_delta"),
        "entry_hits": treatment.get("ge_entry"),
        "control_n": int(control.get("n") or 0),
        "treatment_n": int(treatment.get("n") or 0),
        "treatment_max": treatment.get("max"), "entry_score": ab.get("entry_score"),
        "best_lane": best_name,
        "best_lane_max": best.get("max"),
    }


def _search_strategy_check(reaction: int) -> dict | None:
    user_host, port, _ = REACTION_BOXES[reaction]
    common = (f"--art {SEARCH_ARTIFACTS_REMOTE} --bank {SEARCH_BANK_REMOTE} "
              f"--reaction {reaction}")
    ab_command = (
        f"cd ~/search && uv run --no-project python {AB_PERFORMANCE_REMOTE} {common} "
        f"--last-files 20 --policy-version {STRATEGY_POLICY_VERSION} --minimum-per-arm 20"
    )
    lane_command = (
        f"cd ~/search && uv run --no-project python {LANE_PERFORMANCE_REMOTE} {common} "
        "--rounds 10"
    )
    lanes = _last_json_object(_ssh_port(user_host, port, lane_command, timeout=180))
    # The old motif experiment had its own treatment/control tags.  The field
    # lane uses the producer's uniform random arm as its true online control,
    # which lane_performance reports directly.
    ab = None
    if not lanes or not str(lanes.get("active_strategy") or "").startswith(
            "field-recombination-"):
        ab = _last_json_object(_ssh_port(user_host, port, ab_command, timeout=180))
    if ab is None and lanes is None:
        return None
    ab = ab or {}
    lanes = lanes or {}
    return {
        "checked_at": time.time(), "reaction": reaction,
        "ab": ab, "lanes": lanes, "verdict": _strategy_verdict(ab, lanes),
    }


def refresh_strategy_only() -> dict:
    """Atomically refresh the lightweight strategy panel without rerunning gates."""
    status = load() or {}
    _refresh_boxes_from_pi()
    checks = dict(status.get("search_strategy") or {})
    with ThreadPoolExecutor(max_workers=len(REACTION_BOXES)) as pool:
        current = dict(zip(REACTION_BOXES, pool.map(
            _search_strategy_check, REACTION_BOXES)))
    for reaction, result in current.items():
        if result is not None:
            checks[str(reaction)] = result
    status["search_strategy"] = checks
    temporary = STATUS_PATH.with_suffix(".json.tmp-strategy")
    temporary.write_text(json.dumps(status))
    temporary.replace(STATUS_PATH)
    return checks


def collect(live_gap_check: bool = True) -> dict:
    """One full snapshot. live_gap_check=False skips the expensive per-box
    SELECT (useful for a quick/forced refresh); the rendered page falls back
    to whatever the previous snapshot had for that field, so a skipped or
    failed box never blanks out the whole tab."""
    previous = {}
    if STATUS_PATH.is_file():
        try:
            previous = json.loads(STATUS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            previous = {}

    _refresh_boxes_from_pi()
    log_text = _ssh(PI_HOST, "cat", PI_LOOP_LOG, timeout=30)
    if log_text is None:
        status = dict(previous)
        status["collected_at"] = time.time()
        status["pi_reachable"] = False
        return status

    parsed = _parse_loop_log(log_text)
    window = parsed["window_epoch_reaction"]
    allocated = parsed["allocated"]
    blocked = parsed["blocked"]
    final_gates = parsed.get("final_gates") or {}

    draws_by_reaction: dict[int, int] = {}
    for reaction in window.values():
        draws_by_reaction[reaction] = draws_by_reaction.get(reaction, 0) + 1
    clears_by_reaction: dict[int, int] = {}
    for row in allocated:
        clears_by_reaction[row["reaction"]] = clears_by_reaction.get(row["reaction"], 0) + 1

    # Real on-chain outcome for every clear -- this is the expensive-ish part
    # (one leaderboard fetch per clear, cheap individually, ~13 today).
    clear_epochs = [row["epoch"] for row in allocated]
    committed = _committed_uids_by_epoch(log_text, clear_epochs) if clear_epochs else {}
    outcomes = []
    wins_by_reaction: dict[int, int] = {}
    for row in allocated:
        uids = committed.get(row["epoch"], [])
        outcome = _real_outcome(row["epoch"], uids) if uids else None
        entry = dict(row, uids=uids, outcome=outcome)
        outcomes.append(entry)
        if outcome and outcome.get("won"):
            wins_by_reaction[row["reaction"]] = wins_by_reaction.get(row["reaction"], 0) + 1

    # Recent history: every unique epoch, clear or blocked, most recent first.
    history: dict[int, dict] = {}
    for epoch, reaction in window.items():
        history[epoch] = dict(epoch=epoch, reaction=reaction, outcome="unknown")
    for row in allocated:
        entry = history.setdefault(row["epoch"], dict(epoch=row["epoch"], reaction=row["reaction"]))
        entry["outcome"] = "cleared"
        entry["selected_score"] = row["best"]
        gate = final_gates.get(row["epoch"]) or {}
        baseline = gate.get("raw_floor")
        if baseline is None:
            baseline = REACTION_BOXES[row["reaction"]][2]
        entry["baseline"] = baseline
        entry["gap"] = row["best"] - baseline
        if gate:
            entry["prediction_offset"] = gate.get("prediction_offset")
            entry["predicted_realized"] = gate.get("worst_predicted_realized")
            entry["realized_floor"] = gate.get("realized_floor")
            entry["final_gate"] = True
    for row in blocked:
        entry = history.setdefault(row["epoch"], dict(epoch=row["epoch"], reaction=row["reaction"]))
        if entry.get("outcome") != "cleared":
            entry["outcome"] = "blocked"
            entry["selected_score"] = row.get("selected_score")
            entry["baseline"] = row.get("baseline")
            for key in ("prediction_offset", "predicted_realized", "realized_floor", "final_gate"):
                if key in row:
                    entry[key] = row[key]
            entry["gap"] = (
                row["selected_score"] - row["baseline"]
                if "selected_score" in row and "baseline" in row else None
            )
    for row in outcomes:
        entry = history.get(row["epoch"])
        if entry is not None and row.get("outcome"):
            entry["real_rank"] = row["outcome"].get("rank")
            entry["real_won"] = row["outcome"].get("won")
            entry["real_score"] = row["outcome"].get("score")
            entry["winner_score"] = row["outcome"].get("winner_score")
            entry["field_size"] = row["outcome"].get("field_size")
            entry["real_rejected"] = bool(row["outcome"].get("rejected"))

    wallclock = _attach_wallclock(parsed["ordered_window_events"], collected_at=time.time())
    for entry in history.values():
        entry["at"] = wallclock.get((entry["epoch"], entry["reaction"]))

    recent = sorted(history.values(), key=lambda row: -row["epoch"])[:30]

    gap_checks = dict(previous.get("gap_checks") or {})
    strategy_checks = dict(previous.get("search_strategy") or {})
    floor_details = dict(previous.get("dynamic_floor_details") or {})
    if live_gap_check:
        _pull_submitted_archive()
        floors, details = _dynamic_floors((max(window) + 1) if window else 0)
        DYNAMIC_FLOORS.update(floors)
        DYNAMIC_FLOOR_DETAILS.update(details)
        floor_details = {str(key): value for key, value in details.items()}
        with ThreadPoolExecutor(max_workers=len(REACTION_BOXES)) as pool:
            # The next epoch to be drawn, so the gate also excludes molecules
            # submitted in epochs not yet in the local archive copy.
            next_epoch = (max(window) + 1) if window else 0
            checked = dict(zip(REACTION_BOXES, pool.map(
                lambda reaction: _live_gap_check(reaction, next_epoch), REACTION_BOXES)))
        with ThreadPoolExecutor(max_workers=len(REACTION_BOXES)) as pool:
            strategy = dict(zip(REACTION_BOXES, pool.map(
                _search_strategy_check, REACTION_BOXES)))
        for reaction, result in strategy.items():
            if result is not None:
                strategy_checks[str(reaction)] = result
        gap_db = connect()
        try:
            for reaction in REACTION_BOXES:
                result = checked[reaction]
                if result is not None:
                    result["checked_at"] = time.time()
                    gap_checks[str(reaction)] = result
                    # Skip the error-only branch of _live_gap_check's result
                    # (dict(pool_size=..., error=...)) -- nothing plottable.
                    if "selected_score" in result:
                        gap_db.execute(
                            "INSERT OR REPLACE INTO gap_history "
                            "(checked_at, reaction, selected_score, floor, gap, "
                            "clears_now, pool_size) VALUES (?,?,?,?,?,?,?)",
                            (result["checked_at"], reaction, result["selected_score"],
                             result["floor"], result["gap"],
                             int(bool(result["clears_now"])), result.get("pool_size")))
                        gap_db.commit()
        finally:
            gap_db.close()

    status = dict(
        collected_at=time.time(),
        pi_reachable=True,
        total_draws=len(window),
        total_clears=len(allocated),
        draws_by_reaction=draws_by_reaction,
        clears_by_reaction=clears_by_reaction,
        wins_by_reaction=wins_by_reaction,
        recent_history=recent,
        gap_checks=gap_checks,
        dynamic_floor_details=floor_details,
        search_strategy=strategy_checks,
        winners_curse_gap=WINNERS_CURSE_GAP,
    )
    return status


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




# ---------------------------------------------------------------------------
# Rendering -- plain HTML tables, following build.py's selector_replay_table
# / easynano_status_table pattern (a <table class="grid"> string, no Plotly
# dependency for tabular content).


def _safe(value) -> str:
    return html_lib.escape(str(value), quote=True)


def compound_status_table(status: dict | None = None) -> str:
    """Per-reaction: real historical clear rate, real win count, and the
    live real-gate check for "would this clear right now"."""
    status = status if status is not None else load()
    if not status or not status.get("pi_reachable", True) and not status.get("draws_by_reaction"):
        return '<p class="note">Compound status is not available yet -- the Pi was unreachable on the last collection attempt and no prior snapshot exists.</p>'

    draws = status.get("draws_by_reaction") or {}
    clears = status.get("clears_by_reaction") or {}
    wins = status.get("wins_by_reaction") or {}
    gaps = status.get("gap_checks") or {}
    floor_details = status.get("dynamic_floor_details") or {}
    floors = {str(r): (gaps.get(str(r)) or {}).get("floor", REACTION_BOXES[r][2]) for r in REACTION_BOXES}

    rows = []
    for reaction in range(1, 6):
        key = str(reaction)
        d = draws.get(key, draws.get(reaction, 0))
        c = clears.get(key, clears.get(reaction, 0))
        w = wins.get(key, wins.get(reaction, 0))
        rate = f"{100 * c / d:.1f}%" if d else "—"
        gap = gaps.get(key)
        if gap and "selected_score" in gap:
            clears_now = "READY" if gap.get("clears_now") else "not ready"
            gap_cell = f"{gap['selected_score']:.3f} vs {gap['floor']:.2f} ({gap['gap']:+.3f}) — {clears_now}"
            if gap.get("pool_size") is not None:
                gap_cell += (f"; {gap['pool_size']} eligible "
                             f"({gap.get('non_novel_refused') or 0} non-novel, "
                             f"{gap.get('recently_submitted_refused') or 0} recently submitted, "
                             f"{gap.get('undenoised_dropped') or 0} single/partial-pass records "
                             "outside the gate—not a queued backlog)")
            # Absolute time, never "Nm ago": the page is static, so a relative
            # age is frozen at build time and reads fresh forever.
            if gap.get("checked_at"):
                gap_cell += ", checked " + _local_hhmm(gap["checked_at"])
        elif gap and "error" in gap:
            gap_cell = f"blocked before scoring: {_safe(gap['error'])[:100]}"
        else:
            gap_cell = "not yet checked"
        # Green ONLY when the live real-gate check clears the floor right now --
        # never for past wins (that coloured 3 not-ready reactions green).
        ready = bool(gap and gap.get("clears_now"))
        css = ' class="us"' if ready else ""
        floor_cell = f"{floors[key]:.4f}"
        detail = floor_details.get(key) or floor_details.get(reaction) or {}
        if isinstance(detail.get("winner"), (int, float)):
            floor_cell += (
                f"<br><small>rival {detail['winner']:.3f}; own gap "
                f"{float(detail.get('optimism') or 0):.3f}; "
                f"{int(detail.get('own_evidence_epochs') or 0)} epochs</small>"
            )
        rows.append(
            f"<tr{css}><td>rxn {reaction}</td><td>{floor_cell}</td>"
            f"<td>{c}/{d}</td><td>{rate}</td><td>{w}</td><td>{_safe(gap_cell)}</td></tr>"
        )
    return (
        '<table class="grid"><thead><tr><th>reaction</th><th>floor</th>'
        '<th>cleared/drawn</th><th>clear rate</th><th>real wins</th>'
        '<th>live gate check (right now, not tied to any epoch below)</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def compound_strategy_table(status: dict | None = None) -> str:
    """Exact-score verdict for the active public-field recombination lane."""
    status = status if status is not None else load()
    checks = (status or {}).get("search_strategy") or {}
    if not checks:
        return '<p class="note">Search-strategy A/B evidence has not been collected yet.</p>'

    def number(value, signed=False):
        if not isinstance(value, (int, float)):
            return "—"
        return f"{value:+.4f}" if signed else f"{value:.4f}"

    rows = []
    relative_positive = entry_hits = measured = active_field = 0
    newest = None
    for reaction in range(1, 6):
        check = checks.get(str(reaction), checks.get(reaction)) or {}
        ab = check.get("ab") or {}
        lane_report = check.get("lanes") or {}
        verdict = check.get("verdict") or _strategy_verdict(ab, lane_report)
        control_n = int(verdict.get("control_n") or 0)
        treatment_n = int(verdict.get("treatment_n") or 0)
        active_field += int(verdict.get("mode") == "field_recombination")
        if control_n and treatment_n:
            measured += 1
            relative_positive += int((verdict.get("relative_lift") or 0) > 0)
            entry_hits += int(verdict.get("entry_hits") or 0)
        checked_at = check.get("checked_at")
        if isinstance(checked_at, (int, float)):
            newest = max(newest or checked_at, checked_at)
        best_lane = verdict.get("best_lane") or "—"
        best_lane = best_lane.replace("competitor_motif", "motif")
        best_lane_cell = best_lane
        if isinstance(verdict.get("best_lane_max"), (int, float)):
            best_lane_cell += f" {verdict['best_lane_max']:.4f}"
        max_entry = f"{number(verdict.get('treatment_max'))} / {number(verdict.get('entry_score'))}"
        policy = verdict.get("geometry") or verdict.get("mode") or "—"
        if verdict.get("block_share") is not None:
            policy += f"; {int(verdict['block_share'])}/500"
        rows.append(
            f"<tr><td>rxn {reaction}</td><td>{_safe(verdict.get('label') or 'unavailable')}</td>"
            f"<td>{_safe(policy)}</td><td>{control_n} / {treatment_n}</td>"
            f"<td>{number(verdict.get('relative_lift'), True)}</td>"
            f"<td>{int(verdict.get('entry_hits') or 0)}</td><td>{max_entry}</td>"
            f"<td>{_safe(best_lane_cell)}</td></tr>"
        )
    if measured:
        verdict_text = (
            f'Field selection beats its uniform-random control on {relative_positive}/{measured} '
            f'measured reactions and has {entry_hits} bank-entry hits.'
        )
    else:
        verdict_text = (
            'The cutover is active; the first verdict waits for 20 exact random-arm and 20 exact '
            'field-selected scores per reaction.'
        )
    headline = (
        f'<p class="note"><b>Public-field recombination active on {active_field}/5 reactions.</b> '
        f'{verdict_text} The comparison uses exact scores; the random arm is sampled before proxy '
        'selection, so it measures the new block/affinity ranker rather than its own shortlist.'
    )
    if newest:
        headline += f' Checked {_safe(_local_hhmm(newest))}.'
    headline += '</p>'
    return (
        headline +
        '<table class="grid"><thead><tr><th>reaction</th><th>verdict</th>'
        '<th>active geometry; pool share</th><th>exact n random/proxy</th>'
        '<th>proxy − random mean</th><th>entry hits</th>'
        '<th>proxy max / bank entry</th><th>best recent lane</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def compound_history_table(status: dict | None = None) -> str:
    status = status if status is not None else load()
    recent = (status or {}).get("recent_history") or []
    if not recent:
        return '<p class="note">No recent compound submission history collected yet.</p>'
    rows = []
    for row in recent:
        outcome = row.get("outcome", "unknown")
        if outcome == "cleared":
            won = row.get("real_won")
            label = ("REJECTED by validator" if row.get("real_rejected") else
                     "WON" if won else ("cleared, lost" if won is False else "cleared, outcome unknown"))
            css = ' class="us"' if won else ""
        elif outcome == "blocked":
            label = "blocked"
            css = ""
        else:
            label = "unresolved"
            css = ""
        def number(value, signed=False):
            if not isinstance(value, (int, float)):
                return "—"
            return f"{value:+.3f}" if signed else f"{value:.3f}"
        predicted = row.get("selected_score")
        realized = row.get("real_score")
        winner = row.get("winner_score")
        shrink = (realized - predicted
                  if isinstance(realized, (int, float)) and isinstance(predicted, (int, float))
                  else None)
        to_winner = (realized - winner
                     if isinstance(realized, (int, float)) and isinstance(winner, (int, float))
                     else None)
        rank = row.get("real_rank")
        field = row.get("field_size")
        rank_cell = (f"#{rank} of {field}" if rank and field else f"#{rank}") if rank else "—"
        rows.append(
            f"<tr{css}><td>{_safe(row.get('epoch'))}</td><td>rxn {_safe(row.get('reaction'))}</td>"
            f"<td>{_safe(label)}</td><td>{number(predicted)}</td><td>{number(realized)}</td>"
            f"<td>{number(shrink, True)}</td><td>{number(winner)}</td>"
            f"<td>{number(to_winner, True)}</td><td>{rank_cell}</td></tr>"
        )
    realized_rows = [r for r in recent
                     if isinstance(r.get("real_score"), (int, float))
                     and isinstance(r.get("selected_score"), (int, float))]
    summary = ""
    if realized_rows:
        shrinks = [r["real_score"] - r["selected_score"] for r in realized_rows]
        summary = (
            f'<p class="note">Across {len(shrinks)} submitted epochs above, the validator\'s '
            f'realized score averaged {sum(shrinks) / len(shrinks):+.3f} versus our predicted '
            f'selection (median {sorted(shrinks)[len(shrinks) // 2]:+.3f}). A reaction needs to '
            'clear its floor by more than that shrink to be a likely win.</p>'
        )
    return (
        summary +
        '<table class="grid"><thead><tr><th>epoch</th><th>reaction</th><th>outcome</th>'
        '<th>predicted</th><th>realized (validator)</th><th>realized − predicted</th>'
        '<th>winner</th><th>realized − winner</th><th>real rank</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def main() -> int:
    """Run as `python -m nanobody_dash.compound`, matching economics.py/
    infra.py's own collector entry points -- refresh.py invokes this as a
    bounded-timeout subprocess, not in-process, so a hung SSH call can never
    block the rest of the dashboard refresh indefinitely."""
    status = collect(live_gap_check=True)
    save(status)
    print(json.dumps(dict(
        pi_reachable=status.get("pi_reachable"),
        total_draws=status.get("total_draws"),
        total_clears=status.get("total_clears"),
        wins_by_reaction=status.get("wins_by_reaction"),
        gap_checks_collected=len(status.get("gap_checks") or {}),
        strategy_checks_collected=len(status.get("search_strategy") or {}),
    )))
    return 0


def compound_caveat_note(status: dict | None = None) -> str:
    status = status if status is not None else load()
    gap = (status or {}).get("winners_curse_gap") or WINNERS_CURSE_GAP
    collected = (status or {}).get("collected_at")
    age = "at " + _local_hhmm(collected) if collected else "unknown"
    return (
        '<p class="note"><b>The "live gate check" runs the real release gate</b> (a copy of the '
        'Pi\'s bank_release.py on each box): only molecules with at least 3 rescore passes AND '
        'novel against the public archive (similarity at most 0.5) are eligible, then the best '
        '20 under the entropy constraint are summed against the floor. Before 2026-10-02 this '
        'column used a hand-written replica that skipped both filters, so it overstated readiness '
        '(rxn 3 showed 1.566 "clearing" while the real gate scored 1.467, with 887 of 1261 eligible '
        'molecules refused as non-novel) -- an earlier note here calling that gap "not a bug" was '
        'wrong. Gap-history points before that date come from the old replica and read high. '
        'Remaining difference from the Pi: the boxes lack the validator tree, so the Pi also '
        'applies the validator\'s non-similarity validity gates, and its bank is synced every 5 '
        'minutes.</p>'
        '<p class="note"><b>The single/partial-pass count is a discovery pool, not a rescore '
        'queue.</b> The live policy now purchases one confirmation observation at a time and '
        're-evaluates after each result. A molecule reaches the 3-pass release gate only when '
        'its posterior chance of improving the constrained top 20—or its information value—'
        'still justifies another full Boltz call. This avoids spending two more passes on every '
        'initially noisy high score.</p>'
        '<p class="note">Separately: clearing a reaction\'s baseline offline is NOT the same as '
        'being likely to win live. The live floor now excludes our own coldkey from the rival '
        'forecast, collapses each rival to its best UID per epoch, corrects the forecast-bias '
        'sign, and adds a per-reaction empirical-Bayes estimate of our raw-to-realized optimism. '
        'The final decision is rerun on the exact allocated portfolio; an intermediate '
        '<code>allocated</code> event is not counted as a clear if that final gate rejects it. '
        'The starting calibration was measured 2026-10-02 across '
        f'{gap["n"]} epochs with clean on-chain data: offline (de-noised) selected_score runs a '
        f'median {gap["median"]:.3f} above what actually lands on-chain (mean {gap["mean"]:.3f}, '
        f'stdev {gap["stdev"]:.3f}). The displayed floor uses reaction-specific live evidence '
        'rather than treating that historical median as an immutable offset. Reactions 1 and 3 '
        'use an aggressive opportunity floor calibrated to a 25% estimated win probability from '
        'no-lookahead rival forecast errors and our own realized optimism; reactions 2, 4, and 5 '
        'retain the approximately 50% point-estimate floor. '
        'Compound is winner-take-all: only rank 1 earns anything; every other submission just '
        'burns a slot and permanently burns that chemical space\'s novelty for nothing. '
        f'Snapshot collected {age}.</p>'
        '<p class="note">About the "Submission readiness growth" chart below: its gap-check line '
        'only starts accumulating from when this chart was first deployed (2026-10-02) -- a '
        'reaction showing one point, or a short line, simply has not had many refresh cycles yet, '
        'not a data problem. The epoch-boundary markers on the SAME chart are NOT subject to that '
        'limit: they are re-derived from the Pi\'s own submission log on every refresh and can '
        'already show up to the ~30 most recent real epochs on first load. Also note: a cleared '
        'epoch\'s marker uses the exact floor logged by the final gate from 2026-10-06 onward. '
        'Older <code>allocated</code> events did not record a baseline and retain the static-floor '
        'approximation; blocked epochs use the real floor logged at that moment.</p>'
    )


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
