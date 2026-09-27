"""Atomically refresh the local dashboard for the macOS LaunchAgent."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "nanobody_dash.refresh.lock"
NEXT = ROOT / "public" / "index.html.next"
LIVE = ROOT / "public" / "index.html"
SLOT_SNAPSHOT = ROOT / "data" / "subnet68_slots.jsonl"
SLOT_SNAPSHOT_NEXT = ROOT / "data" / "subnet68_slots.jsonl.next"
# The new dashboard Mac is on the same LAN as the Pi.  Use the direct address;
# mDNS/Tailscale name resolution has failed independently of Pi health.
SLOT_HOST = os.environ.get("NOVA_SLOT_ADDRESS", "192.168.1.68")
SLOT_USER = os.environ.get("NOVA_SLOT_USER", "thomasg")
SLOT_REMOTE_PATH = "/home/thomasg/.nova/subnet68_slots.jsonl"
FUNDING_TARGET = ROOT / "data" / "infra_funding_target.json"
FUNDING_REMOTE_PATH = "/home/thomasg/.nova/infra_funding_target.json"
SAFE_SLOT_FIELDS = {
    "time", "epoch", "block", "registered", "registered_uids", "target",
    "recycle_tao", "decision_recycle_tao", "executed", "reason", "hotkey",
}
# Money data (balances, flows, TAO price, AWS/Vast spend) is daily-granular.
# Collecting it every 60s exhausts the Taostats API quota (HTTP 429) and buys
# nothing.  Run the money collectors at most hourly.
MONEY_REFRESH_SECONDS = 3600
MONEY_MARKER = ROOT / "data" / "money_refresh.marker"


def sync_slot_snapshot() -> int:
    """Fetch the Pi's public operational log and persist only allowed fields.

    SSH resolves its key through the user's normal agent/config.  No credential
    value is placed in argv, and neither remote stderr nor the unfiltered log is
    emitted by this process.
    """
    endpoint = f"{SLOT_USER}@{SLOT_HOST}"
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", endpoint):
        raise RuntimeError("unsafe purchase-host endpoint")
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         endpoint, "cat", SLOT_REMOTE_PATH],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        # Keep the last good snapshot.  A transient Pi outage must not prevent
        # leaderboard collection or replace current data with an empty file.
        return 0
    clean = []
    for raw in completed.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("time") is None:
            continue
        clean.append(json.dumps(
            {key: record.get(key) for key in SAFE_SLOT_FIELDS if key in record},
            separators=(",", ":"), sort_keys=True,
        ))
    if not clean:
        return 0
    SLOT_SNAPSHOT_NEXT.write_text("\n".join(clean) + "\n")
    os.chmod(SLOT_SNAPSHOT_NEXT, 0o600)
    os.replace(SLOT_SNAPSHOT_NEXT, SLOT_SNAPSHOT)
    return len(clean)


def _should_refresh_money() -> bool:
    try:
        return time.time() - MONEY_MARKER.stat().st_mtime >= MONEY_REFRESH_SECONDS
    except OSError:
        return True


def _touch_money_marker() -> None:
    MONEY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    MONEY_MARKER.write_text(str(int(time.time())) + "\n")


def refresh_money_sources() -> dict[str, bool]:
    """Refresh daily balance/flow and infrastructure history without leaking.

    Both collectors resolve their own Keychain locators inside their process.
    Their stdout and stderr are discarded so an upstream client change cannot
    put an authorization-bearing response into the LaunchAgent log.  A money
    API outage must not block leaderboard or slot-history publication.
    """
    status = {}
    for name, module in (
        ("economics", "nanobody_dash.economics"),
        ("infrastructure", "nanobody_dash.infra"),
    ):
        completed = subprocess.run(
            [sys.executable, "-m", module],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=180,
        )
        status[name] = completed.returncode == 0
    return status


def publish_infrastructure_funding_target() -> bool:
    """Send a bounded, non-secret cost target to the Pi signing agent."""
    from . import economics, funding

    connection = economics.connect(ROOT / "data" / "nanobody_dash.sqlite")
    try:
        tao_usd = economics.resolve_tao_usd(connection)
        cloud_state = json.loads(
            (ROOT / "data" / "cloud_costs.json").read_text(encoding="utf-8")
        )
        target = funding.build_target(connection, cloud_state, tao_usd)
        funding.write_target(FUNDING_TARGET, target)
        return funding.publish_target(
            target, host=SLOT_HOST, user=SLOT_USER,
            remote_path=FUNDING_REMOTE_PATH,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError):
        # Funding fails closed.  Dashboard refresh and mining do not depend on
        # a payment instruction reaching the Pi.
        return False
    finally:
        connection.close()


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("dashboard refresh already running")
            return 0
        slot_rows = sync_slot_snapshot()
        subprocess.run(
            [sys.executable, "-m", "nanobody_dash.collect",
             "--slot-log", str(SLOT_SNAPSHOT)],
            cwd=ROOT,
            check=True,
        )
        if _should_refresh_money():
            money_status = refresh_money_sources()
            funding_target = publish_infrastructure_funding_target()
            _touch_money_marker()
        else:
            money_status = {"economics": "skipped", "infrastructure": "skipped"}
            funding_target = "skipped"
        subprocess.run(
            [sys.executable, "-m", "nanobody_dash.build", "--out", str(NEXT)],
            cwd=ROOT,
            check=True,
        )
        os.replace(NEXT, LIVE)
        # Best-effort: the private Vercel mirror is a convenience for remote
        # viewing, throttled internally to avoid redeploying every 60s. Its
        # own failures must never take down the local dashboard that already
        # works today.
        try:
            from . import publish_vercel
            published = publish_vercel.publish()
        except Exception:
            published = False
        print(
            f"dashboard operational snapshot: {slot_rows} Pi rows; "
            f"money={money_status}; funding_target={funding_target}; "
            f"vercel_publish_attempted={published}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
