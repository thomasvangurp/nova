#!/usr/bin/env python3
"""Pull a consistent copy of master's local SQLite primary to THIS machine
(ARCHITECTURE_REDESIGN new finding 8, "and here" leg).

Second, independent off-box copy, next to the Hetzner MySQL replica that
nanobody_opt/replicate_local.py maintains. It also holds what the replicator
does not send (mutation_effects / metadata / edges, the replication
bookkeeping), and it survives a replication bug or a Hetzner loss.

Mechanism: SQLite's official ``sqlite3_rsync`` (https://www.sqlite.org/rsync.html,
>= 3.50 on both ends). It runs over ssh, keeps both databases live, sends only
the pages that differ (hash-negotiated), and leaves the replica as a
consistent snapshot of the origin as of the moment the sync started. A plain
rsync/scp of a WAL-mode file being written can copy a torn, corrupt database;
this cannot. The origin side only takes a read transaction, so the producer's
writes are never blocked.

Per run:
  1. make sure master has the sqlite3_rsync binary (/root/bin/sqlite3_rsync;
     copied from ~/.local/share/nanobody-backup/bin/ if a fresh box lacks it --
     the sqlite.org linux build needs glibc 2.38, master has 2.35, so the
     cached binary is one built on master from the 3.54.0 sources);
  2. sqlite3_rsync master:<db> -> <dest>/latest.sqlite;
  3. PRAGMA quick_check + row counts on the replica (read-only);
  4. once a day, a point-in-time copy daily/<YYYY-MM-DD>.sqlite via the SQLite
     backup API (protects against a logical error that the next sync would
     faithfully copy); dailies older than --keep-days are pruned;
  5. <dest>/status.json records the outcome (ok, sizes, counts, durations).

A missing remote database is not an error (exit 0, status "remote_absent"):
the live write path has not been switched to local SQLite yet.

Scheduled through deploy/macos/backup_all.sh (the existing
ai.nova.nanobody-backup LaunchAgent, every 30 min), next to
deploy/slot_host/backup_artifacts.sh, which keeps its 24 full gzip snapshots
of the same file (12 h of 30-min restore points); this job adds the cheap,
always-fresh delta copy plus 14 daily restore points. Run by hand::

    python3 tools/pull_local_sqlite_backup.py                 # discover box by label
    python3 tools/pull_local_sqlite_backup.py --host root@ssh9.vast.ai --port 10616

Binaries (not in git), in ~/.local/share/nanobody-backup/bin/:
  sqlite3_rsync-darwin-arm64        sqlite.org sqlite-tools-osx-arm64-3540000.zip
                                    (SHA3-256 272ee0d6...a2c3, as published)
  sqlite3_rsync-linux-x64-glibc2.35 built on master from sqlite-src-3540000
                                    tool/sqlite3_rsync.c + the 3.54.0
                                    amalgamation: gcc -O2 -DSQLITE_ENABLE_DBPAGE_VTAB
                                    -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION
                                    -DSQLITE_OMIT_DEPRECATED sqlite3_rsync.c sqlite3.c -lm
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LABEL = "nova-nanobody-pixi"
DEFAULT_REMOTE_DB = "/root/artifacts/nanobody_mutations.sqlite"
DEFAULT_DEST = Path(os.environ.get("NOVA_BACKUP_DIR", str(Path.home() / "nova-backups"))) / "local-sqlite"
BIN_DIR = Path.home() / ".local" / "share" / "nanobody-backup" / "bin"
LOCAL_RSYNC = BIN_DIR / "sqlite3_rsync-darwin-arm64"
REMOTE_RSYNC_SOURCE = BIN_DIR / "sqlite3_rsync-linux-x64-glibc2.35"
REMOTE_RSYNC = "/root/bin/sqlite3_rsync"
COUNT_TABLES = ("nodes", "observations", "observation_artifacts", "lineage_edges",
                "exact_runs", "exact_run_observations", "mutation_effects", "edges")


def ssh_base(host: str, port: str) -> list[str]:
    return ["ssh", "-n", "-p", port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=30", host]


def run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def discover_box(label: str, vastai: str):
    """(host, port) of the running instance with this label, found the same
    way deploy/slot_host/backup_artifacts.sh does, so a rotated box or ssh
    port is followed automatically. The provider response is parsed here in
    process and never printed."""
    try:
        raw = subprocess.run([vastai, "show", "instances", "--raw"], capture_output=True,
                             text=True, timeout=120).stdout
        for item in json.loads(raw):
            if item.get("label") == label and item.get("actual_status") == "running":
                return f"root@{item['ssh_host']}", str(item["ssh_port"])
    except Exception:  # noqa: BLE001
        return None
    return None


def write_status(dest: Path, status: dict) -> None:
    tmp = dest / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=1, sort_keys=True))
    os.replace(tmp, dest / "status.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("NANOBODY_BACKUP_HOST"),
                        help="user@host; default: discover the running box by vast.ai label")
    parser.add_argument("--port", default=os.environ.get("NANOBODY_BACKUP_PORT"))
    parser.add_argument("--label", default=os.environ.get("NOVA_VAST_LABEL", DEFAULT_LABEL))
    parser.add_argument("--vastai", default=os.environ.get("NOVA_VASTAI", "vastai"))
    parser.add_argument("--remote-db", default=os.environ.get("NOVA_MUTATION_DATABASE",
                                                              DEFAULT_REMOTE_DB))
    parser.add_argument("--dest", type=Path,
                        default=Path(os.environ.get("NANOBODY_BACKUP_DEST", DEFAULT_DEST)))
    parser.add_argument("--keep-days", type=int, default=14)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "daily").mkdir(exist_ok=True)
    lock = open(args.dest / ".pull.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("BACKUP_ALREADY_RUNNING", file=sys.stderr)
        return 0
    started = time.time()
    if not args.host or not args.port:
        found = discover_box(args.label, args.vastai)
        if found is None:
            # No box rented is a normal state between rentals, not an error.
            write_status(args.dest, {"started_at": started, "ok": True, "state": "no_box"})
            print(f"BACKUP_NO_BOX no running instance labelled {args.label}")
            return 0
        args.host, args.port = found
    for binary in (LOCAL_RSYNC, REMOTE_RSYNC_SOURCE):
        if not binary.is_file():
            print(f"BACKUP_MISSING_BINARY {binary} (see the module docstring)", file=sys.stderr)
            return 1
    status: dict = {"started_at": started, "host": args.host, "remote_db": args.remote_db}
    ssh = ssh_base(args.host, args.port)

    # 1. remote presence of the database and of the binary.
    probe = run(ssh + [f"test -s {args.remote_db} && echo DB; "
                       f"test -x {REMOTE_RSYNC} && {REMOTE_RSYNC} --version"], 60)
    if probe.returncode == 255:
        status.update(ok=False, state="ssh_failed", error=probe.stderr.strip()[-500:])
        write_status(args.dest, status)
        print(f"BACKUP_SSH_FAILED {status['error']}", file=sys.stderr)
        return 1
    if "DB" not in probe.stdout.split():
        status.update(ok=True, state="remote_absent", finished_at=time.time())
        write_status(args.dest, status)
        print(f"BACKUP_REMOTE_ABSENT {args.remote_db}")
        return 0
    if "DB" == probe.stdout.strip() or len(probe.stdout.split()) < 2:
        copy = run(["scp", "-q", "-P", args.port, "-o", "BatchMode=yes",
                    str(REMOTE_RSYNC_SOURCE), f"{args.host}:{REMOTE_RSYNC}.tmp"], 300)
        install = run(ssh + [f"mkdir -p {Path(REMOTE_RSYNC).parent} && chmod 755 "
                             f"{REMOTE_RSYNC}.tmp && mv {REMOTE_RSYNC}.tmp {REMOTE_RSYNC}"], 60)
        if copy.returncode or install.returncode:
            status.update(ok=False, state="remote_binary_install_failed",
                          error=(copy.stderr + install.stderr)[-500:])
            write_status(args.dest, status)
            return 1
        status["installed_remote_binary"] = True

    # 2. sync (consistent snapshot of origin at sync start; only diffs travel).
    latest = args.dest / "latest.sqlite"
    sync_start = time.time()
    sync = run([str(LOCAL_RSYNC), f"{args.host}:{args.remote_db}", str(latest),
                "-p", args.port, "--exe", REMOTE_RSYNC, "-v"], args.timeout)
    status["sync_seconds"] = round(time.time() - sync_start, 1)
    status["sync_output"] = (sync.stdout + sync.stderr).strip()[-800:]
    if sync.returncode:
        status.update(ok=False, state="sync_failed", finished_at=time.time())
        write_status(args.dest, status)
        print(f"BACKUP_SYNC_FAILED {status['sync_output']}", file=sys.stderr)
        return 1

    # 3. verify the replica (read-only).
    # Not mode=ro: a WAL-mode replica needs its -shm; this process only reads.
    conn = sqlite3.connect(str(latest))
    check = conn.execute("PRAGMA quick_check").fetchone()[0]
    counts = {}
    for table in COUNT_TABLES:
        try:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            counts[table] = None
    status.update(quick_check=check, counts=counts, bytes=latest.stat().st_size)
    if check != "ok":
        conn.close()
        status.update(ok=False, state="replica_corrupt", finished_at=time.time())
        write_status(args.dest, status)
        print(f"BACKUP_REPLICA_CHECK_FAILED {check}", file=sys.stderr)
        return 1

    # 4. daily point-in-time copy (backup API: consistent even with a WAL).
    today = args.dest / "daily" / f"{dt.date.today().isoformat()}.sqlite"
    if not today.exists():
        tmp = today.with_suffix(".sqlite.tmp")
        tmp.unlink(missing_ok=True)
        out = sqlite3.connect(tmp)
        conn.backup(out)
        out.close()
        os.replace(tmp, today)
        status["daily_written"] = today.name
    conn.close()
    cutoff = dt.date.today() - dt.timedelta(days=args.keep_days)
    for old in (args.dest / "daily").glob("*.sqlite"):
        try:
            if dt.date.fromisoformat(old.stem) < cutoff:
                old.unlink()
        except ValueError:
            pass
    status["free_bytes"] = shutil.disk_usage(args.dest).free
    status.update(ok=True, state="ok", finished_at=time.time())
    write_status(args.dest, status)
    print(f"BACKUP_OK bytes={status['bytes']} sync_s={status['sync_seconds']} "
          f"counts={json.dumps(counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
