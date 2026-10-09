#!/usr/bin/env python3
"""Liveness checker for the SN68 nanobody mining pipeline.

A tmux session existing says nothing about whether the process inside it is
doing useful work: today live-producer crash-looped for ~19h and a frozen
selection replayed a 190-epoch-old bank, both behind healthy-looking sessions.
Every check here reads the process's own *output* and asks whether it is fresh
and successful on the cadence that process is supposed to keep:

* producer   (master)  -- per-iteration outcome markers the producer writes on
  disk: ``iteration_NN/race/complete.json`` (success) vs ``failed.json``
  (failure), plus the newest mtime anywhere in the live epoch trees.
* publisher  (master)  -- ``/root/artifacts/nanobody_result.json`` mtime and
  its ``epoch`` field compared with the chain epoch in ``chain_state.json``.
* dashboard  (ssh2)    -- ``cycle start`` stamps in the refresh-loop log, and
  separately whether ``multipass_<E>.json`` exists for every epoch that closed
  more than a grace window ago (the loop can tick forever while scoring fails
  or is skipped behind a held flock).
* pi-submit  (Pi)      -- ``"committed"`` events in ``/opt/nova-submit/loop.log``
  compared with the chain epoch, plus loop.log freshness.

Unhealthy results go to ``data/liveness_alerts.log`` and a macOS notification
(on transition, then every ``REALERT_SECONDS`` while still unhealthy, plus a
recovery notice). Per-check state lives in ``data/liveness_state.json``; the
latest full report in ``data/liveness_status.json``.

Run: ``/usr/bin/python3 tools/check_mining_liveness.py`` (stdlib only).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
STATE_PATH = DATA / "liveness_state.json"
STATUS_PATH = DATA / "liveness_status.json"
ALERT_LOG = DATA / "liveness_alerts.log"

SSH_COMMON = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
              "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
              "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=10"]
HOSTS = {
    "master": ["-p", "10616", "root@ssh9.vast.ai"],
    "dashboard": ["-p", "12752", "root@ssh2.vast.ai"],
    "pi": ["thomasg@100.64.243.65"],
}

BLOCK_SECONDS = 12.0

# --- thresholds -------------------------------------------------------------
# Producer polls every 30s and writes into iteration_NN/ continuously while an
# iteration runs; the only intentional quiet span is the ~55-block (~11 min)
# hand-off reserve at the end of each epoch.
PRODUCER_ACTIVITY_MAX_AGE = 20 * 60
# Iterations take ~3 min when failing fast and longer when racing; within one
# epoch (~72 min) a healthy producer finishes at least one race.
PRODUCER_SUCCESS_WINDOW = 90 * 60
PRODUCER_MIN_ATTEMPTS = 2
# Publisher rewrites the artifact once per ~72 min epoch.
PUBLISHER_MAX_AGE = 100 * 60
# Dashboard loop: one cycle every ~5 min plus the cycle's own work. A cycle that
# is actually scoring an epoch can run 15-60 min, so a quiet log is tolerated
# while the scoring tree is being written; never beyond the hard cap.
DASHBOARD_CYCLE_MAX_AGE = 12 * 60
DASHBOARD_BUSY_MAX_AGE = 15 * 60
DASHBOARD_CYCLE_HARD_CAP = 120 * 60
# Observed close->multipass lag on 2026-10-09: 49-103 min.
MULTIPASS_GRACE = 150 * 60
# Pi commits epoch E late in epoch E; one epoch is ~72 min.
PI_COMMIT_MAX_AGE = 100 * 60
PI_LOG_MAX_AGE = 15 * 60

REALERT_SECONDS = 3 * 3600
UNREACHABLE_ALERT_AFTER = 2  # consecutive failed probes before alerting

# --- remote probes (stdlib python3, sent on stdin; read-only) ---------------
MASTER_PROBE = r'''
import json, os, re, subprocess, time
out = {"now": time.time()}
try:
    out["chain"] = json.load(open("/root/artifacts/chain_state.json"))
    out["chain"] = {k: out["chain"].get(k) for k in ("epoch", "block", "epoch_length", "updated_at_unix")}
except Exception as e:
    out["chain_error"] = repr(e)
live = "/root/live"
epochs = []
try:
    for name in os.listdir(live):
        m = re.fullmatch(r"epoch(\d+)", name)
        if m and os.path.isdir(os.path.join(live, name, "producer")):
            epochs.append(int(m.group(1)))
except Exception as e:
    out["live_error"] = repr(e)
epochs.sort()
iterations = []
for e in epochs[-8:]:
    root = os.path.join(live, "epoch%d" % e, "producer")
    for name in sorted(os.listdir(root)):
        if not re.fullmatch(r"iteration_\d+", name):
            continue
        d = os.path.join(root, name)
        rec = {"epoch": e, "iteration": int(name.split("_")[1]), "mtime": os.path.getmtime(d)}
        f = os.path.join(d, "failed.json")
        c = os.path.join(d, "race", "complete.json")
        if os.path.exists(f):
            rec["failed"] = os.path.getmtime(f)
        if os.path.exists(c):
            rec["complete"] = os.path.getmtime(c)
        iterations.append(rec)
out["iterations"] = iterations
newest, newest_path = 0.0, None
for e in epochs[-2:]:
    for dirpath, dirnames, filenames in os.walk(os.path.join(live, "epoch%d" % e, "producer")):
        for n in filenames + dirnames:
            p = os.path.join(dirpath, n)
            try:
                t = os.path.getmtime(p)
            except OSError:
                continue
            if t > newest:
                newest, newest_path = t, p
out["producer_newest"] = newest
out["producer_newest_path"] = newest_path
def procs(pattern):
    r = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True)
    hits = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3 and pattern in parts[2] and "python" in parts[2]:
            hits.append({"pid": int(parts[0]), "etimes": int(parts[1])})
    return hits
out["producer_procs"] = procs("nanobody_opt.run_live_producer")
out["publisher_procs"] = procs("publish_current_nanobody.py")
r = subprocess.run(["tmux", "capture-pane", "-p", "-J", "-t", "nova-nanobody-producer", "-S", "-600"],
                   capture_output=True, text=True)
lines = [l.rstrip() for l in r.stdout.splitlines() if l.strip()]
out["producer_last_marker"] = next((l[:240] for l in reversed(lines) if l.startswith("PRODUCER_")), None)
out["producer_last_error"] = next((l[:240] for l in reversed(lines)
                                   if re.search(r"(Error|Exception)\b.*:", l) and not l.startswith("PRODUCER_ITERATION_FAILED")), None)
art = "/root/artifacts/nanobody_result.json"
try:
    out["artifact_mtime"] = os.path.getmtime(art)
    d = json.load(open(art))
    out["artifact"] = {k: d.get(k) for k in ("epoch", "status", "publication_stage", "submission_count")}
except Exception as e:
    out["artifact_error"] = repr(e)
# The submitter switch (audit §0.2): which program the watchdog relaunches as
# the publisher. NUL-separated argv; only the program identity is reported.
try:
    argv = open("/root/artifacts/publisher_argv.bin", "rb").read().split(b"\0")
    words = [a.decode("utf-8", "replace") for a in argv if a]
    out["publisher_argv"] = {
        "mtime": os.path.getmtime("/root/artifacts/publisher_argv.bin"),
        "is_real_publisher": any(("publish-current-nanobody" in w or "publish_current_nanobody" in w)
                                 for w in words),
        "mentions_children": any("children" in w for w in words),
        "head": [w[:60] for w in words[:6]],
    }
except Exception as e:
    out["publisher_argv_error"] = repr(e)
fl = "/root/artifacts/pipeline_failures.jsonl"
try:
    with open(fl, "rb") as h:
        h.seek(max(0, os.path.getsize(fl) - 65536))
        recs = []
        for l in h.read().decode("utf-8", "replace").splitlines():
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    out["failures"] = [{k: r.get(k) for k in ("at", "component", "marker", "error", "context")}
                       for r in recs[-30:]]
except FileNotFoundError:
    out["failures"] = []
except Exception as e:
    out["failures_error"] = repr(e)
print(json.dumps(out))
'''

DASHBOARD_PROBE = r'''
import json, os, re, time
out = {"now": time.time()}
log = "/root/dashboard_refresh_loop.log"
try:
    out["loop_log_mtime"] = os.path.getmtime(log)
    with open(log, "rb") as h:
        h.seek(max(0, os.path.getsize(log) - 65536))
        tail = h.read().decode("utf-8", "replace").splitlines()
    stamps = [l.split()[0] for l in tail if l.endswith("cycle start")]
    out["last_cycle_start"] = stamps[-1] if stamps else None
    out["cycle_starts_in_tail"] = len(stamps)
except Exception as e:
    out["loop_log_error"] = repr(e)
root = "/root/multipass_all_rivals"
files, newest = {}, 0.0
try:
    for name in os.listdir(root):
        if name.isdigit():
            p = os.path.join(root, name, "multipass_%s.json" % name)
            if os.path.exists(p):
                files[name] = os.path.getmtime(p)
    for name in sorted((n for n in os.listdir(root) if n.isdigit()), key=int)[-3:]:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, name)):
            for n in filenames:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, n)))
                except OSError:
                    pass
    sl = os.path.join(root, "score_latest.log")
    out["score_log_mtime"] = os.path.getmtime(sl)
    newest = max(newest, out["score_log_mtime"])
    with open(sl, "rb") as h:
        h.seek(max(0, os.path.getsize(sl) - 32768))
        tail = h.read().decode("utf-8", "replace").splitlines()
    out["last_scoring_line"] = next((l[:200] for l in reversed(tail) if "scoring epoch" in l), None)
    out["last_score_error"] = next((l[:240] for l in reversed(tail) if re.search(r"Error\b", l)), None)
except Exception as e:
    out["multipass_error"] = repr(e)
out["multipass_files"] = files
out["multipass_tree_newest"] = newest
# Swallowed-failure markers in the scorer log (P0b): newest occurrence of each.
try:
    sl = os.path.join(root, "score_latest.log")
    with open(sl, "rb") as h:
        h.seek(max(0, os.path.getsize(sl) - 262144))
        tail = h.read().decode("utf-8", "replace").splitlines()
    out["db_write_failed_lines"] = [l[:200] for l in tail if "DB_WRITE_FAILED" in l or "MULTIPASS_FAILED" in l][-5:]
    out["db_write_ok_lines"] = [l[:120] for l in tail if l.startswith("DB_WRITE ")][-2:]
except Exception as e:
    out["score_log_tail_error"] = repr(e)
fl = "/root/artifacts/pipeline_failures.jsonl"
try:
    with open(fl, "rb") as h:
        h.seek(max(0, os.path.getsize(fl) - 65536))
        recs = []
        for l in h.read().decode("utf-8", "replace").splitlines():
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    out["failures"] = [{k: r.get(k) for k in ("at", "component", "marker", "error", "context")}
                       for r in recs[-30:]]
except FileNotFoundError:
    out["failures"] = []
except Exception as e:
    out["failures_error"] = repr(e)
print(json.dumps(out))
'''

PI_PROBE = r'''
import datetime as dt, json, os, time
out = {"now": time.time()}
log = "/opt/nova-submit/loop.log"
try:
    out["log_mtime"] = os.path.getmtime(log)
    with open(log, "rb") as h:
        h.seek(max(0, os.path.getsize(log) - 262144))
        tail = h.read().decode("utf-8", "replace").splitlines()
    now_local = dt.datetime.now()
    commits, last_event, identity = [], None, []
    for line in tail:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        last_event = ev
        if ev.get("event") == "committed":
            commits.append(ev)
        if ev.get("event") in ("UNEXPECTED_PRODUCER", "producer_identity_ok",
                               "unexpected_producer_refused"):
            identity.append(ev)
    def absolute(at):
        # "at" is HH:MM:SS local time without a date; anchor to today, or
        # yesterday if that would put it in the future.
        t = dt.datetime.combine(now_local.date(), dt.time.fromisoformat(at))
        if t > now_local + dt.timedelta(minutes=5):
            t -= dt.timedelta(days=1)
        return t.timestamp()
    out["commits"] = [{"epoch": c.get("epoch"), "rc": c.get("rc"), "at": c.get("at"),
                       "at_unix": absolute(c["at"]) if c.get("at") else None} for c in commits[-6:]]
    out["last_event"] = {k: last_event.get(k) for k in ("event", "at", "epoch", "block")} if last_event else None
    out["identity_events"] = [{k: ev.get(k) for k in ("event", "at", "epoch", "publication_stage", "stage",
                                                      "problems", "refuse_enabled", "using")}
                              for ev in identity[-6:]]
except Exception as e:
    out["log_error"] = repr(e)
# What was actually committed: the archived artifact of each recent committed
# epoch (independent of the loop's own identity check, so it also covers a Pi
# still running code without that check).
committed = []
# Last two committed epochs: enough to catch a swap within one epoch without
# re-alerting on an already-handled incident for hours.
for c in out.get("commits", [])[-2:]:
    try:
        d = json.load(open("/opt/nova-submit/archive/epoch%s/nanobody_result.json" % c["epoch"]))
        committed.append({"epoch": c["epoch"], "kind": d.get("artifact_kind"),
                          "stage": d.get("publication_stage"), "status": d.get("status"),
                          "pool": d.get("sequence_pool_size"),
                          "producer": (d.get("producer") or {}).get("name") if isinstance(d.get("producer"), dict) else None})
    except Exception as e:
        committed.append({"epoch": c.get("epoch"), "error": repr(e)[:120]})
out["committed_artifacts"] = committed
out["refuse_toggle"] = os.path.exists("/opt/nova-submit/REFUSE_UNEXPECTED_PRODUCER")
print(json.dumps(out))
'''


def run_probe(host: str, script: str) -> tuple[dict | None, str | None]:
    cmd = ["ssh", "-T", *SSH_COMMON, *HOSTS[host], "python3", "-"]
    try:
        r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "ssh probe timed out after 120s"
    if r.returncode != 0:
        return None, f"ssh rc={r.returncode}: {r.stderr.strip()[-300:]}"
    try:
        return json.loads(r.stdout.strip().splitlines()[-1]), None
    except Exception as e:  # noqa: BLE001
        return None, f"unparseable probe output ({e}): {r.stdout[-200:]!r}"


def fmt_ts(ts: float | None) -> str:
    if not ts:
        return "never"
    return dt.datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z")


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


class Chain:
    """Chain epoch arithmetic from master's chain_state.json, normalised to our clock."""

    def __init__(self, chain: dict, remote_now: float, local_now: float):
        self.epoch = int(chain["epoch"])
        self.length = int(chain["epoch_length"])
        self.block = int(chain["block"])
        # chain_state.updated_at is in the box's clock; shift into ours.
        self.block_time = float(chain["updated_at_unix"]) + (local_now - remote_now)

    def epoch_close(self, epoch: int) -> float:
        return self.block_time - (self.block - (epoch + 1) * self.length) * BLOCK_SECONDS


def result(name: str, host: str, healthy: bool | None, summary: str,
           since: float | None = None, evidence: dict | None = None) -> dict:
    return {"check": name, "host": host,
            "status": "OK" if healthy else ("UNKNOWN" if healthy is None else "UNHEALTHY"),
            "summary": summary, "since": since, "since_local": fmt_ts(since) if since else None,
            "evidence": evidence or {}}


def check_producer(m: dict, now: float, skew: float) -> dict:
    its = [dict(i, mtime=i["mtime"] + skew,
                failed=i.get("failed") and i["failed"] + skew,
                complete=i.get("complete") and i["complete"] + skew) for i in m.get("iterations", [])]
    newest = (m.get("producer_newest") or 0) + skew
    activity_age = now - newest if newest > skew else None
    window = [i for i in its if i["mtime"] >= now - PRODUCER_SUCCESS_WINDOW]
    finished = [i for i in window if i.get("failed") or i.get("complete")]
    successes = [i for i in its if i.get("complete")]
    last_success = max((i["complete"] for i in successes), default=None)
    recent_success = [i for i in window if i.get("complete")]
    failures_after = [i for i in its if i.get("failed") and (last_success is None or i["failed"] > last_success)]
    first_fail = min((i["failed"] for i in failures_after), default=None)
    ev = {
        "processes": len(m.get("producer_procs", [])),
        "newest_output": fmt_ts(newest) if newest > skew else None,
        "newest_output_age": fmt_age(activity_age),
        "newest_output_path": m.get("producer_newest_path"),
        "iterations_in_window": len(window),
        "finished_in_window": len(finished),
        "succeeded_in_window": len(recent_success),
        "failed_in_window": sum(1 for i in window if i.get("failed")),
        "last_success": fmt_ts(last_success) if last_success else "none in retained epoch trees",
        "retained_iterations": len(its),
        "retained_failed": sum(1 for i in its if i.get("failed")),
        "last_marker": m.get("producer_last_marker"),
        "last_error": m.get("producer_last_error"),
    }
    problems, since = [], None
    if activity_age is None or activity_age > PRODUCER_ACTIVITY_MAX_AGE:
        problems.append(f"STALLED: no producer output written for {fmt_age(activity_age)} "
                        f"(limit {fmt_age(PRODUCER_ACTIVITY_MAX_AGE)})")
        since = newest if newest > skew else None
    if len(finished) >= PRODUCER_MIN_ATTEMPTS and not recent_success:
        problems.append(f"FAILING: {ev['failed_in_window']}/{len(finished)} iterations finished in the last "
                        f"{fmt_age(PRODUCER_SUCCESS_WINDOW)} failed, 0 completed a race "
                        f"(last success: {ev['last_success']}; "
                        f"{ev['retained_failed']}/{len(its)} retained iterations failed)")
        if first_fail:
            since = min(since, first_fail) if since else first_fail
    elif not finished and not recent_success and activity_age is not None and \
            activity_age <= PRODUCER_ACTIVITY_MAX_AGE and len(window) == 0:
        # Writing files but never reaching an iteration boundary for a whole window.
        problems.append(f"NO ITERATIONS: output is changing but no iteration started in "
                        f"{fmt_age(PRODUCER_SUCCESS_WINDOW)}")
    if not ev["processes"]:
        problems.append("no run_live_producer python process present")
    if problems:
        if first_fail and last_success is None:
            ev["since_note"] = "failing since at least the oldest retained iteration"
        return result("producer", "master", False, "; ".join(problems), since, ev)
    return result("producer", "master", True,
                  f"output {fmt_age(activity_age)} old; {len(recent_success)} race(s) completed in last "
                  f"{fmt_age(PRODUCER_SUCCESS_WINDOW)}", None, ev)


def check_publisher(m: dict, now: float, skew: float, chain: Chain | None, state: dict) -> dict:
    if "artifact_mtime" not in m:
        return result("publisher", "master", False, f"artifact unreadable: {m.get('artifact_error')}",
                      None, {})
    mtime = m["artifact_mtime"] + skew
    art = m.get("artifact", {})
    epoch = art.get("epoch")
    hist = state.setdefault("publisher_epoch_history", [])
    if not hist or hist[-1]["epoch"] != epoch:
        hist.append({"epoch": epoch, "first_seen": now})
        del hist[:-20]
    ev = {"artifact_epoch": epoch, "status": art.get("status"), "stage": art.get("publication_stage"),
          "written": fmt_ts(mtime), "age": fmt_age(now - mtime),
          "chain_epoch": chain.epoch if chain else None,
          "epoch_first_seen_by_checker": fmt_ts(hist[-1]["first_seen"]),
          "processes": len(m.get("publisher_procs", []))}
    problems, since = [], None
    if now - mtime > PUBLISHER_MAX_AGE:
        problems.append(f"STALE: nanobody_result.json last rewritten {fmt_age(now - mtime)} ago "
                        f"(limit {fmt_age(PUBLISHER_MAX_AGE)})")
        since = mtime
    if chain and isinstance(epoch, int) and epoch < chain.epoch - 1:
        problems.append(f"FROZEN EPOCH: artifact epoch {epoch} but chain is at {chain.epoch} "
                        f"(expected {chain.epoch - 1} or {chain.epoch})")
        frozen_since = chain.epoch_close(epoch + 1)
        since = min(since, frozen_since) if since else frozen_since
    elif not chain and now - hist[-1]["first_seen"] > PUBLISHER_MAX_AGE * 1.5:
        problems.append(f"FROZEN EPOCH: epoch field stuck at {epoch} across checks since "
                        f"{fmt_ts(hist[-1]['first_seen'])}")
        since = hist[-1]["first_seen"]
    if not ev["processes"]:
        problems.append("no publish_current_nanobody.py process present")
    if problems:
        return result("publisher", "master", False, "; ".join(problems), since, ev)
    return result("publisher", "master", True,
                  f"artifact epoch {epoch} ({art.get('status')}) written {fmt_age(now - mtime)} ago; "
                  f"chain epoch {chain.epoch if chain else '?'}", None, ev)


def check_dashboard(d: dict, now: float, skew: float, chain: Chain | None) -> list[dict]:
    out = []
    # (a) the loop is ticking
    last = d.get("last_cycle_start")
    last_ts = dt.datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc).timestamp() + skew if last else None
    busy = (d.get("multipass_tree_newest") or 0) + skew
    ev = {"last_cycle_start": fmt_ts(last_ts), "cycle_age": fmt_age(now - last_ts if last_ts else None),
          "multipass_tree_last_write": fmt_ts(busy) if busy > skew else None,
          "last_scoring_line": d.get("last_scoring_line"), "last_score_error": d.get("last_score_error")}
    if last_ts is None:
        out.append(result("dashboard-loop", "dashboard", False,
                          f"no 'cycle start' found in loop log ({d.get('loop_log_error')})", None, ev))
    elif now - last_ts <= DASHBOARD_CYCLE_MAX_AGE:
        out.append(result("dashboard-loop", "dashboard", True,
                          f"last cycle start {fmt_age(now - last_ts)} ago", None, ev))
    elif now - last_ts <= DASHBOARD_CYCLE_HARD_CAP and now - busy <= DASHBOARD_BUSY_MAX_AGE:
        out.append(result("dashboard-loop", "dashboard", True,
                          f"last cycle start {fmt_age(now - last_ts)} ago but cycle is busy scoring "
                          f"(multipass tree written {fmt_age(now - busy)} ago)", None, ev))
    else:
        out.append(result("dashboard-loop", "dashboard", False,
                          f"STALLED: no 'cycle start' for {fmt_age(now - last_ts)} "
                          f"(limit {fmt_age(DASHBOARD_CYCLE_MAX_AGE)}, or {fmt_age(DASHBOARD_CYCLE_HARD_CAP)} "
                          f"while scoring) and multipass tree idle {fmt_age(now - busy)}",
                          last_ts + DASHBOARD_CYCLE_MAX_AGE, ev))
    # (b) the loop is actually producing a multipass file for each newly finalized epoch
    files = {int(k): v + skew for k, v in (d.get("multipass_files") or {}).items()}
    newest_epoch = max(files, default=None)
    ev2 = {"newest_multipass_epoch": newest_epoch,
           "newest_multipass_written": fmt_ts(files.get(newest_epoch)) if newest_epoch else None,
           "chain_epoch": chain.epoch if chain else None,
           "last_scoring_line": d.get("last_scoring_line"), "last_score_error": d.get("last_score_error")}
    if chain is None:
        out.append(result("dashboard-multipass", "dashboard", None,
                          "chain epoch unknown (master unreachable); cannot judge multipass freshness", None, ev2))
        return out
    due = [e for e in range(chain.epoch - 12, chain.epoch) if chain.epoch_close(e) <= now - MULTIPASS_GRACE]
    required = max(due) if due else None
    ev2["required_epoch"] = required
    if required is not None:
        ev2["required_epoch_closed"] = fmt_ts(chain.epoch_close(required))
    if required is not None and (newest_epoch is None or newest_epoch < required):
        first_missing = (newest_epoch + 1) if newest_epoch is not None else required
        since = chain.epoch_close(first_missing) + MULTIPASS_GRACE
        out.append(result("dashboard-multipass", "dashboard", False,
                          f"NOT SCORING: newest multipass is epoch {newest_epoch}, but epoch {required} closed "
                          f"{fmt_age(now - chain.epoch_close(required))} ago (grace {fmt_age(MULTIPASS_GRACE)})",
                          since, ev2))
    else:
        out.append(result("dashboard-multipass", "dashboard", True,
                          f"newest multipass epoch {newest_epoch} (written {fmt_age(now - files[newest_epoch]) if newest_epoch else '?'} ago); "
                          f"required >= {required}", None, ev2))
    return out


def check_pi(p: dict, now: float, skew: float, chain: Chain | None) -> dict:
    if "log_mtime" not in p:
        return result("pi-submit", "pi", False, f"loop.log unreadable: {p.get('log_error')}", None, {})
    log_mtime = p["log_mtime"] + skew
    commits = [c for c in p.get("commits", []) if c.get("rc") == 0]
    last = commits[-1] if commits else None
    last_ts = last["at_unix"] + skew if last and last.get("at_unix") else None
    ev = {"last_committed_epoch": last and last.get("epoch"), "last_commit": fmt_ts(last_ts),
          "last_commit_age": fmt_age(now - last_ts if last_ts else None),
          "chain_epoch": chain.epoch if chain else None,
          "log_written": fmt_ts(log_mtime), "log_age": fmt_age(now - log_mtime),
          "last_event": p.get("last_event")}
    problems, since = [], None
    if last_ts is None or now - last_ts > PI_COMMIT_MAX_AGE:
        problems.append(f"NO COMMIT: last successful 'committed' event {fmt_age(now - last_ts if last_ts else None)} ago "
                        f"(limit {fmt_age(PI_COMMIT_MAX_AGE)})")
        since = last_ts
    if chain and last and isinstance(last.get("epoch"), int) and last["epoch"] < chain.epoch - 1:
        problems.append(f"BEHIND: last committed epoch {last['epoch']}, chain at {chain.epoch}")
        behind_since = chain.epoch_close(last["epoch"] + 1)
        since = min(since, behind_since) if since else behind_since
    if now - log_mtime > PI_LOG_MAX_AGE:
        problems.append(f"LOOP SILENT: loop.log not written for {fmt_age(now - log_mtime)}")
        since = min(since, log_mtime) if since else log_mtime
    if problems:
        return result("pi-submit", "pi", False, "; ".join(problems), since, ev)
    return result("pi-submit", "pi", True,
                  f"committed epoch {last['epoch']} {fmt_age(now - last_ts)} ago; chain epoch "
                  f"{chain.epoch if chain else '?'}; log {fmt_age(now - log_mtime)} old", None, ev)


# --- semantic checks (ARCHITECTURE_REDESIGN P0b + P1) -----------------------
# Must match tools/pi_epoch_loop.py EXPECTED_PUBLICATION_STAGES.
EXPECTED_PUBLICATION_STAGES = {"provisional", "qualified"}
# A swallowed failure recorded in pipeline_failures.jsonl stays alertable this long.
FAILURE_WINDOW = 3 * 3600


def check_submitter_identity(m: dict | None, p: dict | None, now: float) -> dict:
    """Who wrote what the Pi committed, and who the watchdog will relaunch.

    Epochs 25613-25617 committed the children-lane artifact on every uid while
    every freshness check stayed green (audit §0.2). This reads the semantic
    invariant instead: the committed artifact's stage, the master artifact's
    stage, the publisher argv, and the Pi loop's own UNEXPECTED_PRODUCER events.
    """
    problems, ev = [], {}
    if p and "committed_artifacts" in p:
        ev["committed"] = p["committed_artifacts"]
        ev["refuse_toggle_on_pi"] = p.get("refuse_toggle")
        bad = [c for c in p["committed_artifacts"]
               if "error" not in c and (c.get("stage") not in EXPECTED_PUBLICATION_STAGES
                                        or c.get("kind") != "nanobody-fleet"
                                        or (c.get("producer") not in (None, "publish_current_nanobody")))]
        if bad:
            problems.append("COMMITTED NON-PUBLISHER ARTIFACT: " + ", ".join(
                f"epoch {c['epoch']} stage={c.get('stage')} pool={c.get('pool')}" for c in bad))
        unexpected = [e for e in (p.get("identity_events") or []) if e.get("event") == "UNEXPECTED_PRODUCER"]
        latest = (p.get("identity_events") or [None])[-1]
        if unexpected and latest and latest.get("event") != "producer_identity_ok":
            problems.append(f"Pi loop logged UNEXPECTED_PRODUCER at {unexpected[-1].get('at')} "
                            f"epoch {unexpected[-1].get('epoch')}: {unexpected[-1].get('problems')}")
        ev["pi_identity_events"] = p.get("identity_events")
    if m:
        art = m.get("artifact") or {}
        ev["master_artifact_stage"] = art.get("publication_stage")
        if art and art.get("publication_stage") not in EXPECTED_PUBLICATION_STAGES:
            problems.append(f"master nanobody_result.json stage={art.get('publication_stage')!r} "
                            "(not the real publisher)")
        pa = m.get("publisher_argv")
        if pa is not None:
            ev["publisher_argv"] = pa
            if not pa.get("is_real_publisher"):
                problems.append("SUBMITTER_SWAPPED: publisher_argv.bin does not launch "
                                f"publish-current-nanobody (head={pa.get('head')})")
        elif m.get("publisher_argv_error"):
            ev["publisher_argv_error"] = m["publisher_argv_error"]
    if not m and not p:
        return result("submitter-identity", "pi+master", None, "UNREACHABLE: no probe data")
    if problems:
        return result("submitter-identity", "pi+master", False, "; ".join(problems), None, ev)
    stages = sorted({c.get("stage") for c in ev.get("committed", []) if c.get("stage")})
    return result("submitter-identity", "pi+master", True,
                  f"committed stages {stages}; master stage {ev.get('master_artifact_stage')}; "
                  "publisher argv is publish-current-nanobody", None, ev)


def check_pipeline_failures(host: str, probe: dict, now: float, skew: float) -> dict:
    """Swallowed-but-recorded failures (nanobody_opt/failure_log.py and the
    dashboard scorer's MULTIPASS_FAILED/DB_WRITE_FAILED markers)."""
    name = f"pipeline-failures-{host}"
    if "failures_error" in probe:
        return result(name, host, None, f"failure log unreadable: {probe['failures_error']}")
    recent = [f for f in probe.get("failures", [])
              if isinstance(f.get("at"), (int, float)) and now - (f["at"] + skew) <= FAILURE_WINDOW]
    ev = {"recent": recent[-8:], "total_in_tail": len(probe.get("failures", []))}
    if host == "dashboard":
        ev["score_log_failure_lines"] = probe.get("db_write_failed_lines")
        ev["score_log_last_db_write"] = probe.get("db_write_ok_lines")
    if recent:
        by = {}
        for f in recent:
            key = f"{f.get('component')}:{f.get('marker')}"
            by[key] = by.get(key, 0) + 1
        first = min(f["at"] + skew for f in recent)
        return result(name, host, False,
                      f"{len(recent)} recorded failure(s) in the last {fmt_age(FAILURE_WINDOW)}: "
                      + ", ".join(f"{k} x{v}" for k, v in sorted(by.items()))
                      + f"; latest: {recent[-1].get('error') or ''}"[:200], first, ev)
    return result(name, host, True, f"no recorded failures in the last {fmt_age(FAILURE_WINDOW)}",
                  None, ev)


def notify(title: str, message: str) -> None:
    script = f'display notification {json.dumps(message[:220])} with title {json.dumps(title)} sound name "Basso"'
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001
        pass


def append_alert(line: str) -> None:
    with ALERT_LOG.open("a") as h:
        h.write(line.rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-alert", action="store_true", help="report only; do not alert or update state")
    parser.add_argument("--json", action="store_true", help="print full JSON report")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when any check is not OK (default exits 0 so launchd's "
                             "last-exit column only flags a crashed checker)")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    try:
        state = json.loads(STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        state = {}
    checks_state = state.setdefault("checks", {})

    now = time.time()
    results: list[dict] = []
    master, merr = run_probe("master", MASTER_PROBE)
    dash, derr = run_probe("dashboard", DASHBOARD_PROBE)
    pi, perr = run_probe("pi", PI_PROBE)

    chain = None
    if master:
        mskew = now - master["now"]
        if master.get("chain"):
            try:
                chain = Chain(master["chain"], master["now"], now)
            except Exception:  # noqa: BLE001
                chain = None
        results.append(check_producer(master, now, mskew))
        results.append(check_publisher(master, now, mskew, chain, state))
        results.append(check_pipeline_failures("master", master, now, mskew))
    else:
        for name in ("producer", "publisher"):
            results.append(result(name, "master", None, f"UNREACHABLE: {merr}"))
    if dash:
        results.extend(check_dashboard(dash, now, now - dash["now"], chain))
        results.append(check_pipeline_failures("dashboard", dash, now, now - dash["now"]))
    else:
        for name in ("dashboard-loop", "dashboard-multipass"):
            results.append(result(name, "dashboard", None, f"UNREACHABLE: {derr}"))
    if pi:
        results.append(check_pi(pi, now, now - pi["now"], chain))
    else:
        results.append(result("pi-submit", "pi", None, f"UNREACHABLE: {perr}"))
    results.append(check_submitter_identity(master, pi, now))

    stamp = dt.datetime.fromtimestamp(now).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"[{stamp}] mining liveness (chain epoch {chain.epoch if chain else '?'})"]
    for r in results:
        cs = checks_state.setdefault(r["check"], {})
        prev = cs.get("status")
        if r["status"] == "OK":
            cs["last_ok"] = now
            cs.pop("unhealthy_since", None)
            cs["unknown_streak"] = 0
        elif r["status"] == "UNKNOWN":
            cs["unknown_streak"] = cs.get("unknown_streak", 0) + 1
        else:
            cs["unknown_streak"] = 0
            cs.setdefault("unhealthy_since", r["since"] or now)
            if r["since"]:
                cs["unhealthy_since"] = min(cs["unhealthy_since"], r["since"])
        r["checker_last_ok"] = fmt_ts(cs.get("last_ok")) if cs.get("last_ok") else "never (by this checker)"
        since_txt = ""
        if r["status"] != "OK":
            since_txt = f" | since {fmt_ts(cs.get('unhealthy_since') or r['since'])}" if r["status"] == "UNHEALTHY" else ""
            since_txt += f" | last OK {r['checker_last_ok']}"
        lines.append(f"  {r['status']:<9} {r['check']:<20} [{r['host']}] {r['summary']}{since_txt}")

        alertable = r["status"] == "UNHEALTHY" or (
            r["status"] == "UNKNOWN" and cs["unknown_streak"] >= UNREACHABLE_ALERT_AFTER)
        if not args.no_alert:
            if alertable:
                due = (prev != r["status"]) or cs.get("alerted_status") != r["status"] or \
                      now - cs.get("last_alert", 0) >= REALERT_SECONDS
                if due:
                    msg = f"{r['check']} on {r['host']}: {r['summary']}{since_txt}"
                    append_alert(f"[{stamp}] ALERT {r['status']} {msg}")
                    for k, v in r["evidence"].items():
                        append_alert(f"      {k}: {v}")
                    notify(f"Nova mining: {r['check']} {r['status']}", r["summary"])
                    cs["last_alert"] = now
                    cs["alerted_status"] = r["status"]
            elif r["status"] == "OK" and cs.get("alerted_status"):
                append_alert(f"[{stamp}] RECOVERED {r['check']} on {r['host']}: {r['summary']}")
                notify(f"Nova mining: {r['check']} recovered", r["summary"])
                cs.pop("alerted_status", None)
                cs.pop("last_alert", None)
        cs["status"] = r["status"]

    report = {"checked_at": now, "checked_at_local": stamp, "chain_epoch": chain.epoch if chain else None,
              "results": results}
    print("\n".join(lines))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    if not args.no_alert:
        STATUS_PATH.write_text(json.dumps(report, indent=2, default=str))
        STATE_PATH.write_text(json.dumps(state, indent=2))
    if args.strict:
        return 0 if all(r["status"] == "OK" for r in results) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
