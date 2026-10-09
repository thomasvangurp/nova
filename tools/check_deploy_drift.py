#!/usr/bin/env python3
"""Detect deploy drift: a repo file changed, but the box never got the update.

Origin (2026-10-09): ``nanobody_opt/hf_field_refresh.py`` was fixed and
committed, but master's daemon runs ``/root/hf_field_refresh.py`` -- a
root-level copy nobody updated -- so the old scoring formula kept running for
hours. Nothing compared what a box actually runs against what is committed.

What this does
--------------
For every *tracked* box file (see ``ENTRYPOINTS`` below) it computes the git
blob hash of the box's live copy (one SSH round-trip per box, script sent on
stdin, nothing written on the box) and compares it against the blob of the
corresponding file at the repo's committed ``HEAD``. Local uncommitted edits
are ignored on purpose: the question is "is the box on what is committed".

Each mismatch is classified using the repo history of that path:

* ``STALE``    -- the box copy equals an OLDER committed version. This is the
                  bug class: the repo moved on, the box did not. Drift age =
                  time of the first commit after the box's version.
* ``LOCALMOD`` -- the box copy matches no committed version: a hand edit on
                  the box that never round-tripped into the repo.
* ``MISSING``  -- a tracked live path no longer exists on the box.
* ``VARIANT_REPO_MOVED`` -- a file with a known, intentional local variant
                  (``ACCEPTED_VARIANTS``) whose repo copy changed since the
                  variant was accepted: the change may need porting.
* ``VARIANT_BOX_MOVED`` -- the accepted variant itself was changed on the box.

Box backup files (``f.bak.<unix-ts>``, ``f.pre-<desc>-<YYYYMMDD>``, ``f.bak``)
and the box file's mtime are reported as "last touched on box" evidence.

Which files are tracked (and why)
---------------------------------
Only files a box *actually executes* and that are supposed to equal a repo
file. The list is entrypoints (taken from the live process table, tmux
sessions, daemon/loop scripts and pixi tasks on each box) plus, for Python
entrypoints, the transitive closure of ``nanobody_opt`` modules they import or
launch (``nanobody_opt.X`` / ``nanobody_opt/X`` / ``from nanobody_opt import X``),
computed from HEAD each run so new dependencies are picked up automatically.

Deliberately NOT tracked (documented, not forgotten):

* Pi ``/opt/nova-submit/nanobody_opt/*``, ``pi_submit.py``, Pi ``tools/*``
  other than ``pi_epoch_loop.py``: the Pi's code is owned by
  ``~/nova-sandbox-repro/pi-submit`` (``pi_drift.py``, launchd
  ``ai.nova.pi-code-drift``). Several Pi modules are intentional local patches
  (chain_state/commit_submission/fleet/make_fleet/bank_release match no repo
  version) and the rest are a Sept snapshot; "syncing the repo version" would
  undo Pi fixes. ``pi_epoch_loop.py`` IS tracked: it is the Pi's live loop
  and is developed in this repo.
* master ``/root/nanobody_opt/hf_field_refresh.py`` (+ loop/daemon copies in
  ``nanobody_opt/``): not executed -- the daemon runs the ``/root/`` copies,
  which ARE tracked. (That decoy copy is what hid the original bug.)
* master ``/root/multipass_all_rivals.py`` / ``rescore_master.sh``: one-off
  backfill on master; the multipass loop lives on the dashboard box.
* master ``/root/tools/sync_multipass_rivals.py``: runs on the Mac (launchd),
  the box copy is an rsync leftover.
* master ``children_lane*.sh``, ``frontier_loop.sh``, ``children_lane.py``,
  ``children_submit.py``: no live process or loop invokes them right now.
* dashboard ``hf_field_refresh*`` / ``import_hf_field_20x.py`` and
  ``nanobody_opt/children_submit.py`` etc.: present but dormant there.
* box-owned scripts with no repo counterpart (``dashboard_refresh_loop.sh``,
  ``fetch_hf_archive_daemon.sh``, ``mysql_tunnel_watchdog.sh``, ``sync_loop.sh``,
  ad-hoc ``check_*.py``/``*_backfill.py``): nothing to compare against.
* dashboard ``score_latest_rival_epoch.sh`` is an intentional variant (no
  producer there: it reads the current epoch from master and passes explicit
  archive/config args). It is pinned in ``ACCEPTED_VARIANTS`` so that it only
  alerts if either side changes.

To accept a newly found intentional difference, add the (box, path) to
``ACCEPTED_VARIANTS`` with both blob hashes from the log line and a reason.

Alerting: ``data/deploy-drift.log`` (every run) and a macOS notification
(osascript) when the set of drift findings changes, plus a reminder every
``REMIND_HOURS`` while drift persists. Claude Code's PushNotification tool
is not callable from a launchd job (and was not available in the session
that built this), so the macOS notification is the push channel.

Usage: ``python3 tools/check_deploy_drift.py [--box master] [--no-notify] [--json]``
Exit status: 0 clean, 1 drift found, 2 a box could not be checked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "data" / "deploy-drift.log"
STATE = REPO / "data" / "deploy-drift.state.json"
REMIND_HOURS = 12
UNREACHABLE_ALERT_RUNS = 3

SSH_COMMON = ["-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR",
              "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]

BOXES: dict[str, dict] = {
    "master": {"ssh": ["-p", "10616", "root@ssh9.vast.ai"], "nanobody_opt": "/root/nanobody_opt"},
    "dashboard": {"ssh": ["-p", "12752", "root@ssh2.vast.ai"], "nanobody_opt": "/root/nanobody_opt"},
    "pi": {"ssh": ["thomasg@100.64.243.65"], "nanobody_opt": None},  # Pi nanobody_opt is pi-submit owned
}

# (box path, repo path, follow nanobody_opt imports?, why it is live)
ENTRYPOINTS: dict[str, list[tuple[str, str, bool, str]]] = {
    "master": [
        ("/root/nanobody_opt/run_live_producer.py", "nanobody_opt/run_live_producer.py", True,
         "tmux nova-nanobody-producer -> pixi task live-producer"),
        ("/root/tools/publish_current_nanobody.py", "tools/publish_current_nanobody.py", True,
         "tmux nova-nanobody-publisher -> pixi task publish-current-nanobody"),
        ("/root/nanobody_opt/scorer_daemon.py", "nanobody_opt/scorer_daemon.py", True,
         "resident scorer daemon + 8 shard workers"),
        ("/root/nanobody_opt/score_boltzgen_fast_adapter.py", "nanobody_opt/score_boltzgen_fast_adapter.py", True,
         "producer --exact-scorer; hf_field_refresh scorer"),
        ("/root/nanobody_opt/fetch_hf_archive.py", "nanobody_opt/fetch_hf_archive.py", True,
         "fetch_hf_archive_daemon.sh (hourly)"),
        ("/root/hf_field_refresh.py", "nanobody_opt/hf_field_refresh.py", True,
         "hf_field_refresh_loop.sh runs THIS root-level copy every 30 min"),
        ("/root/import_hf_field_20x.py", "nanobody_opt/import_hf_field_20x.py", True,
         "hf_field_refresh_loop.sh"),
        ("/root/hf_field_refresh_loop.sh", "nanobody_opt/hf_field_refresh_loop.sh", False,
         "hf_field_refresh_daemon.sh"),
        ("/root/hf_field_refresh_daemon.sh", "nanobody_opt/hf_field_refresh_daemon.sh", False,
         "running daemon"),
        ("/root/session_watchdog.sh", "deploy/producer/session_watchdog.sh", False,
         "restarts the tmux sessions"),
        ("/root/bgen_python.sh", "ansible/files/root/bgen_python.sh", False, "bgen env wrapper"),
        ("/root/pixi.toml", "pixi.toml", False, "defines live-producer/publish tasks + envs"),
    ],
    "dashboard": [
        ("/root/multipass_all_rivals.py", "tools/multipass_all_rivals.py", True,
         "dashboard_refresh_loop.sh -> score_latest_rival_epoch.sh"),
        ("/root/nanobody_opt/fetch_hf_archive.py", "nanobody_opt/fetch_hf_archive.py", True,
         "dashboard_refresh_loop.sh"),
        ("/root/score_latest_rival_epoch.sh", "tools/score_latest_rival_epoch.sh", False,
         "dashboard_refresh_loop.sh (accepted variant)"),
        ("/root/bgen_python.sh", "ansible/files/root/bgen_python.sh", False, "bgen env wrapper"),
    ],
    "pi": [
        ("/opt/nova-submit/tools/pi_epoch_loop.py", "tools/pi_epoch_loop.py", False,
         "nova-submit-loop.service"),
    ],
}

# Intentional local differences: (box, box path) -> pinned blobs + reason.
ACCEPTED_VARIANTS: dict[tuple[str, str], dict[str, str]] = {
    ("dashboard", "/root/score_latest_rival_epoch.sh"): {
        "box_blob": "b07f6cdc3e4936ce452eed1c035b8d69bddad558",   # audited 2026-10-09
        "repo_blob": "ee374d6aa7dd60c5718e26940ccb9b968abbe812",  # HEAD 629ec0e
        "reason": "dashboard has no producer: reads current epoch from master's chain_state "
                  "over ssh and passes explicit hf_archive/config args",
    },
}

REMOTE_SCRIPT = r'''
import glob, hashlib, json, os, sys
out = {}
for p in json.loads(sys.stdin.readline()):
    rec = {"exists": os.path.isfile(p)}
    if rec["exists"]:
        d = open(p, "rb").read()
        rec["blob"] = hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
        rec["mtime"] = int(os.path.getmtime(p))
        rec["backups"] = sorted(
            [[os.path.basename(b), int(os.path.getmtime(b))]
             for b in glob.glob(glob.escape(p) + ".bak*") + glob.glob(glob.escape(p) + ".pre-*")],
            key=lambda x: x[1])
    out[p] = rec
print("@@DRIFT@@" + json.dumps(out))
'''

IMPORT_PATTERNS = [
    re.compile(r"nanobody_opt[./]([A-Za-z_][A-Za-z0-9_]*)"),
]
FROM_PKG = re.compile(r"from\s+nanobody_opt\s+import\s+\(?([A-Za-z0-9_,\s]+)\)?")
FROM_REL = re.compile(r"from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import")
FROM_DOT = re.compile(r"from\s+\.\s+import\s+\(?([A-Za-z0-9_,\s]+)\)?")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout


def head_tree() -> dict[str, str]:
    tree = {}
    for line in git("ls-tree", "-r", "HEAD").splitlines():
        meta, path = line.split("\t", 1)
        tree[path] = meta.split()[2]
    return tree


def closure(repo_path: str, tree: dict[str, str]) -> set[str]:
    """nanobody_opt modules a Python file imports or launches, transitively (from HEAD)."""
    seen: set[str] = set()
    todo = [repo_path]
    while todo:
        path = todo.pop()
        if path in seen or not path.endswith(".py") or path not in tree:
            continue
        seen.add(path)
        text = git("cat-file", "-p", tree[path])
        names: set[str] = set()
        for pat in IMPORT_PATTERNS:
            names.update(pat.findall(text))
        for group in FROM_PKG.findall(text) + FROM_DOT.findall(text):
            names.update(n.split()[0] for n in group.split(",") if n.strip())
        if path.startswith("nanobody_opt/"):
            names.update(FROM_REL.findall(text))
        for name in names:
            dep = f"nanobody_opt/{name}.py"
            if dep in tree:
                todo.append(dep)
    seen.discard(repo_path)
    return seen


def tracked_files(box: str, tree: dict[str, str]) -> dict[str, tuple[str, str]]:
    """box path -> (repo path, reason)."""
    out: dict[str, tuple[str, str]] = {}
    nb_root = BOXES[box]["nanobody_opt"]
    for box_path, repo_path, follow, why in ENTRYPOINTS[box]:
        out[box_path] = (repo_path, why)
        if follow and nb_root:
            for dep in sorted(closure(repo_path, tree)):
                dep_box = f"{nb_root}/{Path(dep).name}"
                out.setdefault(dep_box, (dep, f"imported/launched by {Path(repo_path).name}"))
    return out


def fetch_box(box: str, paths: list[str]) -> dict | str:
    try:
        r = subprocess.run(["ssh", *SSH_COMMON, *BOXES[box]["ssh"], "python3 -c " + shlex.quote(REMOTE_SCRIPT)],
                           input=json.dumps(paths) + "\n", capture_output=True, text=True,
                           timeout=90)
    except subprocess.TimeoutExpired:
        return "ssh timeout"
    if "@@DRIFT@@" not in r.stdout:
        return f"unreachable rc={r.returncode}: {r.stderr.strip()[-160:]}"
    return json.loads(r.stdout.split("@@DRIFT@@", 1)[1].strip())


def path_history(repo_path: str) -> list[tuple[str, int, str]]:
    """[(commit, commit unix time, blob after commit)] newest first, current branch only."""
    out = git("log", "--format=@%H %ct", "--raw", "--no-abbrev", "--", repo_path)
    hist, cur = [], None
    for line in out.splitlines():
        if line.startswith("@"):
            sha, ts = line[1:].split()
            cur = (sha, int(ts))
        elif line.startswith(":") and cur:
            parts = line.split()
            hist.append((cur[0], cur[1], parts[3]))
    return hist


def any_branch_has_blob(repo_path: str, blob: str) -> str | None:
    out = git("log", "--all", "--format=@%h %cs", "--raw", "--no-abbrev", "--", repo_path)
    cur = None
    for line in out.splitlines():
        if line.startswith("@"):
            cur = line[1:]
        elif line.startswith(":") and line.split()[3] == blob:
            return cur
    return None


def fmt_ts(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d %H:%M %Z")


def fmt_age(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:.1f}h" if h < 48 else f"{h / 24:.1f}d"


def check_box(box: str, tree: dict[str, str]) -> tuple[list[dict], int, str | None]:
    tracked = tracked_files(box, tree)
    data = fetch_box(box, sorted(tracked))
    if isinstance(data, str):
        return [], len(tracked), data
    now = time.time()
    findings = []
    for box_path, (repo_path, why) in sorted(tracked.items()):
        rec = data.get(box_path, {"exists": False})
        repo_blob = tree.get(repo_path)
        base = {"box": box, "path": box_path, "repo_path": repo_path, "why": why,
                "repo_blob": repo_blob}
        if not rec.get("exists"):
            # Closure deps that were never deployed are reported too: a live
            # entrypoint would fail to import them.
            findings.append({**base, "kind": "MISSING", "box_blob": None,
                             "detail": "tracked live path absent on box"})
            continue
        box_blob = rec["blob"]
        evidence = f"box mtime {fmt_ts(rec['mtime'])}"
        if rec.get("backups"):
            name, ts = rec["backups"][-1]
            evidence += f"; newest backup {name} ({fmt_ts(ts)}); {len(rec['backups'])} backup(s)"
        variant = ACCEPTED_VARIANTS.get((box, box_path))
        if variant:
            if box_blob == variant["box_blob"] and repo_blob == variant["repo_blob"]:
                continue
            kind = "VARIANT_BOX_MOVED" if box_blob != variant["box_blob"] else "VARIANT_REPO_MOVED"
            findings.append({**base, "kind": kind, "box_blob": box_blob,
                             "detail": f"accepted variant ({variant['reason']}) changed; {evidence}"})
            continue
        if box_blob == repo_blob:
            continue
        hist = path_history(repo_path)
        idx = next((i for i, h in enumerate(hist) if h[2] == box_blob), None)
        if idx is not None:
            # hist is newest-first; the commit right after the box's version
            # (hist[idx-1]) is when the box started falling behind.
            behind = hist[:idx]
            since = behind[-1][1] if behind else hist[idx][1]
            findings.append({**base, "kind": "STALE", "box_blob": box_blob,
                             "detail": (f"box == {hist[idx][0][:8]} ({fmt_ts(hist[idx][1])}), "
                                        f"{len(behind)} commit(s) behind HEAD; drifted since "
                                        f"{fmt_ts(since)} (~{fmt_age(now - since)}); {evidence}")})
        else:
            other = any_branch_has_blob(repo_path, box_blob)
            where = f"matches commit {other} on another branch" if other else "matches no committed version"
            findings.append({**base, "kind": "LOCALMOD", "box_blob": box_blob,
                             "detail": f"box copy {where} (hand edit on box?); "
                                       f"last touched ~{fmt_age(now - rec['mtime'])} ago; {evidence}"})
    return findings, len(tracked), None


def log(line: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with LOG.open("a") as fh:
        fh.write(f"{stamp} {line}\n")


def notify(title: str, msg: str) -> None:
    subprocess.run(["osascript", "-e",
                    f"display notification {json.dumps(msg[:220])} with title {json.dumps(title)}"],
                   capture_output=True, timeout=15)


def signature(f: dict) -> str:
    return f"{f['box']}:{f['path']}:{f['kind']}:{f['box_blob']}:{f['repo_blob']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--box", action="append", choices=sorted(BOXES))
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--json", action="store_true", help="print findings as JSON")
    args = ap.parse_args()

    head = git("rev-parse", "--short", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    tree = head_tree()
    try:
        state = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}

    findings, unreachable, counts = [], [], {}
    for box in args.box or list(BOXES):
        f, n, err = check_box(box, tree)
        counts[box] = n
        if err:
            unreachable.append(box)
            state.setdefault("unreachable", {})[box] = state.get("unreachable", {}).get(box, 0) + 1
            log(f"WARN {box}: {err} ({state['unreachable'][box]} consecutive run(s))")
        else:
            state.setdefault("unreachable", {})[box] = 0
        findings += f

    log(f"RUN head={head} ({branch}) checked "
        + ", ".join(f"{b}={n}" for b, n in counts.items())
        + f" drift={len(findings)} unreachable={unreachable or 'none'}")
    for f in findings:
        log(f"DRIFT {f['kind']} {f['box']} {f['path']} (repo {f['repo_path']}; live via {f['why']}): {f['detail']}")

    sigs = sorted(signature(f) for f in findings)
    new = [f for f in findings if signature(f) not in set(state.get("signatures", []))]
    remind_due = findings and time.time() - state.get("last_notify", 0) > REMIND_HOURS * 3600
    if not args.no_notify:
        if new or remind_due:
            shown = new or findings
            msg = "; ".join(f"{f['box']}:{Path(f['path']).name} {f['kind']}" for f in shown[:4])
            more = f" (+{len(shown) - 4} more)" if len(shown) > 4 else ""
            notify(f"Deploy drift: {len(findings)} file(s)" + ("" if new else " (still unresolved)"),
                   msg + more + " -- see data/deploy-drift.log")
            state["last_notify"] = time.time()
        stuck = [b for b in unreachable if state["unreachable"][b] == UNREACHABLE_ALERT_RUNS]
        if stuck:
            notify("Deploy drift check blind", f"cannot reach {', '.join(stuck)} for "
                   f"{UNREACHABLE_ALERT_RUNS} runs")
    if not args.no_notify:
        # Only a run that could notify advances the "already reported" set.
        if not findings and state.get("signatures"):
            log("RESOLVED all previously reported drift")
        state["signatures"] = sigs
    STATE.write_text(json.dumps(state, indent=2) + "\n")

    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        print(f"head={head} ({branch}) checked {counts} unreachable={unreachable or 'none'}")
        for f in findings:
            print(f"{f['kind']:18s} {f['box']:9s} {f['path']}\n    repo: {f['repo_path']} | live via: {f['why']}\n    {f['detail']}")
        if not findings:
            print("no drift")
    return 2 if unreachable else (1 if findings else 0)


if __name__ == "__main__":
    sys.exit(main())
