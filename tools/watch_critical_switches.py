#!/usr/bin/env python3
"""Trip-wire for the master box's "critical switch" files.

A critical switch is a file (or live fact) that decides WHICH process runs as
our on-chain nanobody submitter, or how it is (re)launched.  On 2026-10-09
``/root/artifacts/publisher_argv.bin`` was silently rewritten to launch the
frozen-bank ``children_lane_loop.sh`` instead of the real publisher, and there
was no record of what it held before, when it changed, or who changed it.

Each run (launchd, every 5 min) this script:

1. Collects, in ONE ssh round-trip, sha256 / size / mtime / ctime / inode /
   owner of every tracked path on master, plus the content of small
   non-secret files and two virtual switches (the tmux sessions' start
   commands, and whether ``session_watchdog.sh`` is alive).
2. Compares each with the LAST line of its append-only history
   ``data/critical_switch_history/<slug>.jsonl``.  The first observation is
   written as a ``baseline`` entry; every later content change is a
   ``change`` entry holding the previous and current state.  Content goes to
   a content-addressed ``blobs/<sha256>`` store (written once, never
   overwritten), so the before/after bytes of every change are recoverable.
3. On a change, gathers co-occurring facts from the box (other files under
   /root modified within +-180 s of the new ctime, still-alive processes
   started within +-15 min, tmux sessions, recent logins).  These are facts
   recorded next to the change, NOT an attribution: a short-lived ``cp`` or
   ``python -c`` that did the write leaves no process behind.
4. Alerts: appends to ``alerts.log`` and posts a macOS notification.

Nothing here writes to the box.  Secret-bearing files are tracked hash-only.
Usage: watch_critical_switches.py [--host root@H] [--port P] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORY = Path(os.environ.get("NOVA_SWITCH_HISTORY", ROOT / "data" / "critical_switch_history"))
BLOBS = HISTORY / "blobs"
ALERTS = HISTORY / "alerts.log"
RUNS = HISTORY / "runs.log"

DEFAULT_HOST = os.environ.get("NOVA_MASTER_HOST", "root@ssh9.vast.ai")
DEFAULT_PORT = os.environ.get("NOVA_MASTER_PORT", "10616")
SSHO = [
    "-T",
    "-o", "ConnectTimeout=15",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "BatchMode=yes",
    "-o", "LogLevel=ERROR",
]
UNREACHABLE_ALERT_RUNS = 6  # ~30 min at a 5 min interval

# path -> (role, mode).  mode "content" stores bytes; "hash_only" never reads
# content off the box beyond hashing it there (secret-bearing files).
TRACKED: dict[str, tuple[str, str]] = {
    "/root/artifacts/publisher_argv.bin": (
        "THE switch: NUL-separated argv that session_watchdog.sh and "
        "deploy/slot_host/ensure_vast_running.py exec to (re)launch the "
        "nova-nanobody-publisher tmux session (our on-chain submitter).",
        "content"),
    "/root/artifacts/publisher_argv.PRODUCER-BACKUP.bin": (
        "Rollback source: rollback_to_producer.sh copies it over "
        "publisher_argv.bin. If this changes, the rollback is poisoned.",
        "content"),
    "/root/artifacts/publisher_argv.pre-rollback-20261009.bin": (
        "Forensic copy of the degraded argv (children_lane_loop.sh) found "
        "2026-10-09; should never change.",
        "content"),
    "/root/artifacts/producer_argv.txt": (
        "Producer launch argv edited in place by nanobody_opt/fix_basin_args.py "
        "(absent at baseline; appearance or edits are switch events).",
        "content"),
    "/root/session_watchdog.sh": (
        "The launcher: relaunches miner/producer/scorerd/rosterwatch/publisher "
        "sessions and reads publisher_argv.bin.",
        "content"),
    "/root/miner_relaunch.sh": ("Watchdog relaunch target for nova-nanobody-miner.", "content"),
    "/root/producer_relaunch.sh": ("Watchdog relaunch target for nova-nanobody-producer.", "content"),
    "/root/scorer_daemon_supervisor.sh": ("Watchdog relaunch target for nova-nanobody-scorerd.", "content"),
    "/root/roster_watch.sh": ("Watchdog relaunch target for nova-rosterwatch.", "content"),
    "/root/nanobody_opt/children_lane_loop.sh": (
        "The degraded fallback publisher loop that publisher_argv.bin pointed "
        "at on 2026-10-09.", "content"),
    "/root/nanobody_opt/rollback_to_producer.sh": (
        "Script that rewrites publisher_argv.bin and kills the publisher.", "content"),
    "/root/nanobody_opt/frontier_loop.sh": ("Alternative long-running lane loop.", "content"),
    "/root/nanobody_opt/fix_basin_args.py": ("Script that rewrites producer_argv.txt.", "content"),
    "/root/pixi.toml": (
        "Defines the pixi tasks the argv names (publish-current-nanobody, "
        "live-producer) and therefore which script each actually runs.",
        "content"),
    "/root/tools/publish_current_nanobody.py": (
        "Entrypoint of the publish-current-nanobody task.", "content"),
    "/root/nanobody_opt/run_live_producer.py": ("Entrypoint of the live-producer task.", "content"),
    "/root/artifacts/nanobody_fast_scorer.json": ("Pinned fast-scorer selection.", "content"),
    "/root/artifacts/nanobody_deployed_commit.txt": ("Deployed code commit marker.", "content"),
    "/root/artifacts/nanobody_chain_deployed_commit.txt": ("Deployed chain code commit marker.", "content"),
    "/root/artifacts/nanobody_publisher_running_commit.txt": ("Commit the running publisher reports.", "content"),
    "/root/onstart.sh": ("Container boot hook.", "content"),
    "/root/nanobody.env": (
        "Miner environment file (secret-bearing): tracked by hash/size/mtime "
        "only, content never leaves the box.", "hash_only"),
}
VIRTUAL = {
    "virtual:tmux_session_commands": (
        "What ACTUALLY runs: tmux session name -> pane start command. A swap of "
        "publisher_argv.bin takes effect here when the session restarts.",
        "content"),
    "virtual:session_watchdog_running": (
        "Whether /root/session_watchdog.sh (the consumer of publisher_argv.bin) "
        "is alive.", "content"),
}
CONTENT_LIMIT = 512 * 1024
INLINE_TEXT_LIMIT = 8 * 1024
SECRET_FLAG = re.compile(r"(passw|secret|token|api[-_]?key|private)", re.I)

REMOTE = r'''
import base64, hashlib, json, os, re, subprocess, sys, time
SPEC = json.loads(sys.argv[1]); LIMIT = int(sys.argv[2])
SECRET = re.compile(r"(passw|secret|token|api[-_]?key|private)", re.I)
def redact(cmd):
    parts = cmd.split(" ")
    for i in range(1, len(parts)):
        if SECRET.search(parts[i - 1]) and parts[i - 1].startswith("-"):
            parts[i] = "<redacted>"
    return " ".join(parts)
out = {"remote_time": time.time(), "files": {}, "virtual": {}}
try:
    out["box_label"] = open("/root/.vast_containerlabel").read().strip()
except OSError:
    out["box_label"] = None
for path, mode in SPEC.items():
    rec = {"exists": False}
    try:
        st = os.stat(path)
        data = open(path, "rb").read()
        rec = {"exists": True, "sha256": hashlib.sha256(data).hexdigest(),
               "size": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
               "inode": st.st_ino, "mode": oct(st.st_mode), "uid": st.st_uid, "gid": st.st_gid}
        if mode == "content" and len(data) <= LIMIT:
            rec["content_b64"] = base64.b64encode(data).decode()
        del data
    except FileNotFoundError:
        pass
    except OSError as e:
        rec = {"exists": None, "error": e.strerror}
    out["files"][path] = rec
try:
    panes = subprocess.run(["tmux", "list-panes", "-a", "-F",
        "#{session_name}\t#{session_created}\t#{pane_pid}\t#{pane_start_command}"],
        capture_output=True, text=True, timeout=10).stdout.splitlines()
except Exception:
    panes = []
cmds, created = {}, {}
for line in panes:
    name, ctime, pid, cmd = (line.split("\t", 3) + ["", "", ""])[:4]
    cmds[name] = redact(cmd)
    created[name] = {"created": int(ctime or 0), "pane_pid": int(pid or 0)}
out["virtual"]["virtual:tmux_session_commands"] = {"value": cmds, "detail": created}
alive = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except OSError:
        continue
    if any(a.endswith(b"session_watchdog.sh") for a in argv[:3]):
        alive.append(int(pid))
out["virtual"]["virtual:session_watchdog_running"] = {"value": bool(alive), "detail": {"pids": sorted(alive)}}
print(json.dumps(out))
'''

CONTEXT = r'''
import json, os, re, subprocess, sys, time
T = float(sys.argv[1]); WIN = 180; PWIN = 900
SECRET = re.compile(r"(passw|secret|token|api[-_]?key|private)", re.I)
SKIP = {".pixi", ".cache", "fastscore", "layer2_cache", "live", "nanobody_history",
        "multipass_all_rivals", "bgen_shim", ".bittensor", "proc", "sys"}
near = []
for top, dirs, files in os.walk("/root"):
    depth = top.count("/") - 1
    dirs[:] = [d for d in dirs if d not in SKIP and depth < 3]
    for f in files:
        p = os.path.join(top, f)
        try:
            m = os.lstat(p).st_ctime
        except OSError:
            continue
        if abs(m - T) <= WIN:
            near.append({"path": p, "ctime": m, "delta_s": round(m - T, 3)})
near.sort(key=lambda r: r["delta_s"])
hz = os.sysconf("SC_CLK_TCK")
boot = None
for line in open("/proc/stat"):
    if line.startswith("btime"):
        boot = int(line.split()[1])
procs = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        start = boot + int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
        argv = [a.decode(errors="replace") for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
    except Exception:
        continue
    if abs(start - T) <= PWIN and argv:
        for i in range(1, len(argv)):
            if argv[i - 1].startswith("-") and SECRET.search(argv[i - 1]):
                argv[i] = "<redacted>"
        procs.append({"pid": int(pid), "start": start, "delta_s": round(start - T, 1),
                      "cmd": " ".join(argv)[:400]})
procs.sort(key=lambda r: r["start"])
try:
    logins = subprocess.run(["last", "-F", "-n", "15"], capture_output=True, text=True, timeout=10).stdout
except Exception:
    logins = ""
print(json.dumps({"files_changed_within_180s": near[:60],
                  "alive_processes_started_within_15min": procs[:60],
                  "last_logins": logins.splitlines()[:15]}))
'''


def now_local() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def iso_local(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.strip("/"))


def ssh_python(host: str, port: str, script: str, args: list[str], timeout: int = 60) -> str:
    cmd = ["ssh", *SSHO, "-p", port, host, "python3", "-", *[_shq(a) for a in args]]
    done = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"ssh rc={done.returncode}: {done.stderr.strip()[-300:]}")
    return done.stdout


def _shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def store_blob(data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    BLOBS.mkdir(parents=True, exist_ok=True)
    target = BLOBS / sha
    if not target.exists():  # content-addressed: never overwrite
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.rename(target)
    return sha


def render(data: bytes, path: str) -> object:
    """Human-readable view of small content: argv files decode to a list."""
    if path.endswith(".bin") and b"\0" in data:
        return [a.decode(errors="replace") for a in data.split(b"\0") if a]
    if len(data) > INLINE_TEXT_LIMIT:
        return None
    try:
        return data.decode()
    except UnicodeDecodeError:
        return None


def state_from_file(path: str, rec: dict, mode: str) -> dict:
    if rec.get("exists") is None:
        return {"exists": None, "sha256": "ERROR", "error": rec.get("error")}
    if not rec.get("exists"):
        return {"exists": False, "sha256": "ABSENT"}
    st = {k: rec[k] for k in ("sha256", "size", "mtime_ns", "ctime_ns", "inode", "mode", "uid", "gid")}
    st["exists"] = True
    st["mtime_local"] = iso_local(rec["mtime_ns"] / 1e9)
    st["ctime_local"] = iso_local(rec["ctime_ns"] / 1e9)
    if mode == "content" and "content_b64" in rec:
        data = base64.b64decode(rec["content_b64"])
        st["blob"] = store_blob(data)
        view = render(data, path)
        if view is not None:
            st["content"] = view
    elif mode == "hash_only":
        st["content_policy"] = "hash_only (secret-bearing; never read)"
    return st


def state_from_virtual(rec: dict) -> dict:
    canon = json.dumps(rec["value"], sort_keys=True).encode()
    return {"exists": True, "sha256": hashlib.sha256(canon).hexdigest(),
            "blob": store_blob(canon), "content": rec["value"], "detail": rec.get("detail")}


def last_entry(history: Path) -> dict | None:
    if not history.exists():
        return None
    last = None
    with history.open("rb") as fh:
        for line in fh:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def append(history: Path, entry: dict) -> None:
    with history.open("a") as fh:  # append-only
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def text_diff(prev: dict, cur: dict, path: str) -> str:
    def lines(st: dict) -> list[str]:
        if not st.get("blob"):
            return [f"<{st.get('sha256')}>"]
        data = (BLOBS / st["blob"]).read_bytes()
        if b"\0" in data:
            return [a.decode(errors="replace") for a in data.split(b"\0") if a]
        try:
            return data.decode().splitlines()
        except UnicodeDecodeError:
            return [f"<binary {st['blob']}>"]
    diff = difflib.unified_diff(lines(prev), lines(cur), "before", "after", lineterm="", n=2)
    out = "\n".join(diff)
    return out if len(out) < 6000 else out[:6000] + "\n... (truncated; see blobs)"


def notify(title: str, message: str) -> None:
    script = f'display notification {json.dumps(message[:220])} with title {json.dumps(title)} sound name "Basso"'
    try:
        subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, timeout=10)
    except Exception as exc:  # notification is best effort; alerts.log is the record
        print(f"notification failed: {exc}", file=sys.stderr)


def alert(lines: list[str]) -> None:
    with ALERTS.open("a") as fh:
        fh.write("\n".join(lines) + "\n\n")


def recent_unreachable_streak() -> int:
    if not RUNS.exists():
        return 0
    streak = 0
    for line in reversed(RUNS.read_text().splitlines()[-50:]):
        if " status=unreachable" in line:
            streak += 1
        else:
            break
    return streak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--dry-run", action="store_true", help="collect and print, write nothing")
    args = ap.parse_args()

    HISTORY.mkdir(parents=True, exist_ok=True)
    spec = {p: mode for p, (_, mode) in TRACKED.items()}
    observed = now_local()
    try:
        snap = json.loads(ssh_python(args.host, args.port, REMOTE, [json.dumps(spec), str(CONTENT_LIMIT)]))
    except Exception as exc:
        streak = recent_unreachable_streak() + 1
        msg = f"{observed} status=unreachable host={args.host}:{args.port} err={str(exc)[:200]!r}"
        with RUNS.open("a") as fh:
            fh.write(msg + "\n")
        print(msg, file=sys.stderr)
        if streak == UNREACHABLE_ALERT_RUNS:
            alert([f"=== {observed} WATCHER BLIND: master unreachable for {streak} consecutive runs",
                   f"host {args.host}:{args.port}; last error: {exc}",
                   "Switch changes during this window will be seen (and diffed) on the next "
                   "successful run, but not timed precisely. If master moved, set "
                   "NOVA_MASTER_HOST/NOVA_MASTER_PORT in the LaunchAgent."])
            notify("Nova switch watcher BLIND", f"master unreachable for {streak} runs")
        return 1

    if args.dry_run:
        print(json.dumps({p: {k: v for k, v in r.items() if k != "content_b64"}
                          for p, r in snap["files"].items()}, indent=2))
        print(json.dumps(snap["virtual"], indent=2))
        return 0

    box = snap.get("box_label")
    remote_time = iso_local(snap.get("remote_time"))
    roles = {**TRACKED, **VIRTUAL}
    current: dict[str, dict] = {}
    for path, rec in snap["files"].items():
        current[path] = state_from_file(path, rec, TRACKED[path][1])
    for name, rec in snap["virtual"].items():
        current[name] = state_from_virtual(rec)

    changes, rewrites, baselines = [], [], 0
    for path, cur in current.items():
        history = HISTORY / f"{slug(path)}.jsonl"
        prev_entry = last_entry(history)
        base = {"observed_at": observed, "box_remote_time": remote_time, "box_label": box,
                "host": f"{args.host}:{args.port}", "path": path, "role": roles[path][0]}
        if prev_entry is None:
            append(history, {**base, "kind": "baseline", "current": cur,
                             "note": "first observation by the watcher; prior history unknown"})
            baselines += 1
            continue
        prev = prev_entry["current"]
        if prev.get("sha256") != cur.get("sha256"):
            changes.append((path, prev, cur, prev_entry, history, base))
        elif not path.startswith("virtual:") and cur.get("exists") and (
                prev.get("ctime_ns") != cur.get("ctime_ns") or prev.get("inode") != cur.get("inode")):
            append(history, {**base, "kind": "rewritten_same_content", "previous": prev, "current": cur})
            rewrites.append(path)
        elif path.startswith("virtual:") and prev.get("detail") != cur.get("detail"):
            # same effective command, but e.g. a session restarted: record, don't alert
            append(history, {**base, "kind": "detail_change", "previous": prev, "current": cur})

    for path, prev, cur, prev_entry, history, base in changes:
        context = None
        # Anchor on ctime: unlike mtime it cannot be preserved/forged by
        # `cp -p`, `touch -d` or `mv`, so it is when the inode was last written.
        anchor = cur.get("ctime_ns")
        if anchor:
            try:
                context = json.loads(ssh_python(args.host, args.port, CONTEXT, [str(anchor / 1e9)]))
                context["anchor"] = f"ctime {cur.get('ctime_local')}"
                context["caveat"] = ("Co-occurring facts only. A short-lived writer (cp, python -c, "
                                     "an ssh one-liner) leaves no process behind; do not read "
                                     "these as attribution.")
            except Exception as exc:
                context = {"error": f"context collection failed: {exc}"}
        box_changed = prev_entry.get("box_label") not in (None, box)
        entry = {**base, "kind": "change", "previous": prev, "current": cur,
                 "previous_observed_at": prev_entry.get("observed_at"),
                 "change_window": (f"between {prev_entry.get('observed_at')} and {observed} "
                                   f"(file ctime {cur.get('ctime_local')}, mtime {cur.get('mtime_local')})"),
                 "box_changed": box_changed, "context": context}
        append(history, entry)
        lines = [f"=== {observed} CRITICAL SWITCH CHANGED: {path}",
                 f"role: {roles[path][0]}",
                 f"box: {box} ({args.host}:{args.port})" + ("  ** BOX LABEL CHANGED **" if box_changed else ""),
                 f"window: {entry['change_window']}",
                 f"before: sha={prev.get('sha256')} size={prev.get('size')} mtime={prev.get('mtime_local')} ctime={prev.get('ctime_local')}",
                 f"after:  sha={cur.get('sha256')} size={cur.get('size')} mtime={cur.get('mtime_local')} "
                 f"ctime={cur.get('ctime_local')} inode={cur.get('inode')}"]
        if TRACKED.get(path, ("", "content"))[1] != "hash_only":
            lines += ["diff:", text_diff(prev, cur, path)]
        if context and "files_changed_within_180s" in context:
            lines.append("files under /root whose ctime is within 180s of the new ctime:")
            lines += [f"  {r['delta_s']:+8.1f}s {r['path']}" for r in context["files_changed_within_180s"][:15]]
            lines.append("alive processes started within 15 min (facts, not attribution):")
            lines += [f"  {r['delta_s']:+8.1f}s pid={r['pid']} {r['cmd'][:160]}"
                      for r in context["alive_processes_started_within_15min"][:10]]
        lines.append(f"history: {history}")
        alert(lines)
        notify("Nova CRITICAL SWITCH changed", f"{path} changed on master (ctime {cur.get('ctime_local')}). See alerts.log")

    for path in rewrites:
        alert([f"=== {observed} rewritten with identical content (no switch change): {path}"])

    status = (f"{observed} status=ok box={box} tracked={len(current)} baselines={baselines} "
              f"changes={len(changes)} same_content_rewrites={len(rewrites)}")
    with RUNS.open("a") as fh:
        fh.write(status + "\n")
    print(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
