#!/usr/bin/env python3
"""Pull the box's multi-pass rival scoring into the dashboard JSON.

The box cron loop runs ``multipass_all_rivals.py`` per completed epoch and
writes ``/root/multipass_all_rivals/<epoch>/multipass_<epoch>.json``. This
script reads every such JSON from the box and merges it into
``nanobody_dash/multipass_comparison.json`` (overwriting any null placeholder
with the real per-rival ``mean``/``stdev``/``fleet_size``/``scores``).

The dashboard refresh loop (launchd, every 60s) then rebuilds + publishes the
changed JSON with no further action here.

Out-of-band edit protection
---------------------------
2026-10-09: a manual fix to two epochs of the local JSON (patched from a direct
DB query) was silently reverted ten minutes later, because the box's per-epoch
output files are cached ("skip if output already exists") and still held the
broken values, and this script overwrote every epoch unconditionally.

This script now records, in ``data/multipass_sync_state.json``, the sha256 of
the whole file it last wrote and the sha256 of every epoch value it last wrote.
On each run it compares the current local file against that record. Any epoch
whose current value differs from what this script last wrote (changed, added
or deleted by someone else) is *pinned*: the box can no longer overwrite it,
other epochs keep syncing normally, and an alert is logged and raised as a
macOS notification. Pins persist across runs (so the protection does not lapse
after one cycle) and are released either automatically, when the box starts
producing exactly the pinned value, or explicitly with ``--release EPOCH`` /
``--release-all`` once the box has been regenerated.

Per-epoch pinning rather than skipping the whole run: an out-of-band fix to
two epochs must not freeze every new epoch on the dashboard until someone
notices. Per-epoch hashing rather than mtime/git: the file is gitignored, mtime
is touched by editors without content changes, and a hand edit inside one key
of a 780 KB JSON must still be caught.

With no state file yet (first run), there is no record of what this script
wrote, so any epoch where the local value differs from the box value is
treated as out-of-band and pinned (conservative), and everything else becomes
the baseline.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOX = "root@95.3.33.46"
PORT = "13920"
REMOTE = "/root/multipass_all_rivals"
LOCAL = Path(os.environ.get(
    "NOVA_MULTIPASS_LOCAL", ROOT / "nanobody_dash" / "multipass_comparison.json"))
STATE = Path(os.environ.get(
    "NOVA_MULTIPASS_SYNC_STATE", ROOT / "data" / "multipass_sync_state.json"))
LOCK = STATE.with_suffix(".lock")
# Test hook: read box output from a local directory of multipass_*.json files
# instead of over SSH.
REMOTE_DIR_OVERRIDE = os.environ.get("NOVA_MULTIPASS_REMOTE_DIR")
NOTIFY = os.environ.get("NOVA_MULTIPASS_NOTIFY", "1") != "0"
SSHO = [
    "-o", "ConnectTimeout=20",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "BatchMode=yes",
]


def _ssh(args: list[str]) -> str:
    completed = subprocess.run(
        ["ssh", *SSHO, "-p", PORT, BOX, *args],
        capture_output=True, text=True, timeout=45,
    )
    return completed.stdout if completed.returncode == 0 else ""


def _remote_files() -> list[str]:
    if REMOTE_DIR_OVERRIDE:
        return sorted(str(p) for p in Path(REMOTE_DIR_OVERRIDE).rglob("multipass_*.json"))
    listing = _ssh(["find", REMOTE, "-name", "multipass_*.json"])
    return sorted(line.strip() for line in listing.splitlines()
                  if line.strip().endswith(".json"))


def _read_remote(path: str) -> str:
    if REMOTE_DIR_OVERRIDE:
        try:
            return Path(path).read_text()
        except OSError:
            return ""
    return _ssh(["cat", path])


def _digest(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _serialise(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _load_state() -> dict | None:
    try:
        state = json.loads(STATE.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        # A corrupt state file must not be mistaken for "no edits": treat it
        # like a first run (conservative bootstrap), never as a clean baseline.
        return None
    return state if isinstance(state, dict) else None


def _save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    staged = STATE.with_suffix(".json.next")
    staged.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(staged, STATE)


def _alert(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] ALERT multipass-sync: {message}", file=sys.stderr, flush=True)
    if not NOTIFY:
        return
    safe = message.replace("\\", "\\\\").replace('"', "'")[:240]
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{safe}" with title "multipass sync: out-of-band edit"'],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _fetch_box() -> tuple[dict, int]:
    box: dict = {}
    files = _remote_files()
    for remote in files:
        raw = _read_remote(remote)
        if not raw:
            continue
        try:
            epoch_data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(epoch_data, dict):
            continue
        for epoch, value in epoch_data.items():
            if not isinstance(value, dict):
                continue
            # only commit epochs that carry a real ours-entry or >=1 rival
            if value.get("ours") or value.get("rivals"):
                box[epoch] = value
    return box, len(files)


def run(release: list[str], release_all: bool) -> int:
    raw_local = None
    try:
        raw_local = LOCAL.read_text()
        data = json.loads(raw_local)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except FileNotFoundError:
        data = {}
    except (OSError, json.JSONDecodeError, ValueError):
        # Previously this fell back to {} and rewrote the file from box data
        # alone, silently discarding whatever was there. Refuse instead.
        _alert(f"{LOCAL.name} exists but is unreadable/invalid JSON; not touching it")
        return 1

    state = _load_state()
    bootstrap = state is None
    if bootstrap:
        state = {"file_sha256": None, "epochs": {}, "pinned": {}}
    written: dict = state.setdefault("epochs", {})
    pinned: dict = state.setdefault("pinned", {})

    # Explicit release of pins (operator says: the box is now authoritative).
    if release_all:
        release = list(pinned)
    for epoch in release:
        if pinned.pop(epoch, None) is not None:
            # Accept the current local value as the baseline so it is not
            # re-detected as out-of-band below; the box merge then replaces it.
            if epoch in data:
                written[epoch] = _digest(data[epoch])
            else:
                written.pop(epoch, None)
            print(f"released pin on epoch {epoch}; box value may overwrite it")

    # 1. Detect out-of-band edits since this script's last write.
    if not bootstrap and raw_local is not None \
            and hashlib.sha256(raw_local.encode()).hexdigest() != state.get("file_sha256"):
        newly = []
        for epoch in sorted(set(data) | set(written)):
            if epoch in pinned:
                continue
            current = _digest(data[epoch]) if epoch in data else None
            if current != written.get(epoch):
                pinned[epoch] = {
                    "detected_at": int(time.time()),
                    "local_sha256": current,  # None = deleted out-of-band
                    "sync_sha256": written.get(epoch),
                }
                newly.append(epoch)
        if newly:
            _alert(f"{LOCAL.name} was edited outside the sync for epoch(s) "
                   f"{', '.join(newly)}; pinned, box will NOT overwrite them. "
                   f"Release with: tools/sync_multipass_rivals.py --release EPOCH")

    box, n_files = _fetch_box()
    if not n_files:
        # Box unreachable: persist any newly detected pins, change nothing else.
        if not bootstrap:
            _save_state(state)
        return 0

    if bootstrap:
        newly = []
        for epoch, value in box.items():
            if epoch in data and _digest(data[epoch]) != _digest(value):
                pinned[epoch] = {
                    "detected_at": int(time.time()),
                    "local_sha256": _digest(data[epoch]),
                    "sync_sha256": None,
                    "reason": "first run without sync history; local differs from box",
                }
                newly.append(epoch)
        if newly:
            _alert(f"first protected run: local {LOCAL.name} differs from box for "
                   f"epoch(s) {', '.join(sorted(newly))}; pinned local values. "
                   f"Release with --release EPOCH if the box is right.")

    # 2. Merge box values, never into pinned epochs.
    merged = dict(data)
    blocked = []
    for epoch, value in box.items():
        if epoch in pinned:
            local_val = merged.get(epoch)
            if local_val is not None and _digest(local_val) == _digest(value):
                # The box caught up with the manual fix: no conflict any more.
                pinned.pop(epoch)
                print(f"box now matches pinned epoch {epoch}; pin released")
            else:
                blocked.append(epoch)
            continue
        merged[epoch] = value
    if blocked:
        print(f"kept out-of-band local value for pinned epoch(s): {', '.join(sorted(blocked))}",
              file=sys.stderr)

    # 3. Write only if content changes, re-checking the file was not edited
    #    while we were fetching from the box.
    text = _serialise(merged)
    if text != raw_local:
        try:
            latest = LOCAL.read_text()
        except FileNotFoundError:
            latest = None
        if latest != raw_local:
            _alert(f"{LOCAL.name} changed during the sync run; skipped this write")
            _save_state(state)
            return 0
        staged = LOCAL.with_suffix(".json.sync-next")
        staged.write_text(text)
        os.replace(staged, LOCAL)
        print(f"merged {n_files} remote files into {LOCAL}")

    # 4. Record exactly what is now on disk as this script's own write.
    #    Pinned epochs keep the sync's previous hash so they stay pinned.
    state["file_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    state["epochs"] = {
        epoch: (written.get(epoch) if epoch in pinned else _digest(value))
        for epoch, value in merged.items()
    }
    state["updated_at"] = int(time.time())
    _save_state(state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release", action="append", default=[], metavar="EPOCH",
                        help="unpin an epoch so the box value may overwrite it")
    parser.add_argument("--release-all", action="store_true")
    parser.add_argument("--status", action="store_true", help="print pinned epochs and exit")
    args = parser.parse_args()
    if args.status:
        state = _load_state() or {}
        print(json.dumps(state.get("pinned", {}), indent=1, sort_keys=True))
        return 0
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("multipass sync already running")
            return 0
        return run(args.release, args.release_all)


if __name__ == "__main__":
    raise SystemExit(main())
