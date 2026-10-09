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
SLOT_HOST = os.environ.get("NOVA_SLOT_ADDRESS", "raspberrypi.local")
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

# Compound status requires one SSH round to the Pi plus the real release-gate
# SELECT and two lightweight read-only strategy summaries on each of the 5
# dedicated mining boxes -- measured ~140s total. Running that every 60s would mean
# the collector is effectively always running and would keep those 5 live
# mining boxes under constant (if cheap) extra SSH load. 20 minutes keeps the
# "ready to submit" numbers reasonably fresh without that.
COMPOUND_REFRESH_SECONDS = 600
# How soon to retry after a failed/unreachable attempt, instead of waiting
# out the full cadence above. Confirmed live 2026-10-02: the Pi went down
# for a reboot right as a refresh attempt landed; compound.py's main()
# always exits 0 even when pi_reachable is False (it falls back to the
# previous snapshot rather than erroring), so the marker was getting touched
# on failure too -- the dashboard then sat on a stale snapshot for a FULL
# extra 20 minutes after the Pi was already back up, rather than catching it
# on the very next cycle.
COMPOUND_RETRY_SECONDS = 180
COMPOUND_MARKER = ROOT / "data" / "compound_refresh.marker"
COMPOUND_STATUS_PATH = ROOT / "data" / "compound_status.json"

# Proxy accuracy only changes once per ~4h retrain cycle per reaction box
# (proxy_refit_loop.sh), so there is no point checking every 20 minutes like
# compound -- hourly is already far more often than the data can change.
PROXY_ACCURACY_REFRESH_SECONDS = 3600
PROXY_ACCURACY_RETRY_SECONDS = 300
PROXY_ACCURACY_MARKER = ROOT / "data" / "proxy_accuracy_refresh.marker"
PROXY_ACCURACY_STATUS_PATH = ROOT / "data" / "proxy_accuracy_status.json"


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


def _should_refresh_compound() -> bool:
    try:
        return time.time() - COMPOUND_MARKER.stat().st_mtime >= COMPOUND_REFRESH_SECONDS
    except OSError:
        return True


def _touch_compound_marker(success: bool = True) -> None:
    COMPOUND_MARKER.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    COMPOUND_MARKER.write_text(str(int(now)) + "\n")
    if not success:
        # Backdate the mtime so _should_refresh_compound() goes true again
        # after COMPOUND_RETRY_SECONDS rather than the full cadence.
        backdated = now - (COMPOUND_REFRESH_SECONDS - COMPOUND_RETRY_SECONDS)
        os.utime(COMPOUND_MARKER, (backdated, backdated))


def refresh_compound_status() -> bool:
    """Collect and save a fresh compound-fleet status snapshot, as a bounded-
    timeout subprocess (matching refresh_money_sources()'s isolation) so a
    hung SSH call to the Pi or a mining box can never block the rest of the
    dashboard refresh. Worst case is ~5 boxes x 90s + the Pi call, so the
    timeout here is generous; on any failure or timeout the previous
    snapshot is left in place (compound.py's collect() reads it as a
    fallback base), so the tab degrades to "stale" rather than empty.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "nanobody_dash.compound"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=600,
    )
    return completed.returncode == 0


def _should_refresh_proxy_accuracy() -> bool:
    try:
        return time.time() - PROXY_ACCURACY_MARKER.stat().st_mtime \
            >= PROXY_ACCURACY_REFRESH_SECONDS
    except OSError:
        return True


def _touch_proxy_accuracy_marker(success: bool = True) -> None:
    PROXY_ACCURACY_MARKER.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    PROXY_ACCURACY_MARKER.write_text(str(int(now)) + "\n")
    if not success:
        backdated = now - (PROXY_ACCURACY_REFRESH_SECONDS - PROXY_ACCURACY_RETRY_SECONDS)
        os.utime(PROXY_ACCURACY_MARKER, (backdated, backdated))


def refresh_proxy_accuracy_status() -> bool:
    """Collect and save a fresh proxy-accuracy snapshot, same bounded-
    timeout subprocess isolation as refresh_compound_status() -- a hung SSH
    call to one reaction box must never block the rest of the dashboard
    refresh. Only 5 small `cat` reads, so the timeout is generous but much
    shorter than compound's (no per-box SELECT to run)."""
    completed = subprocess.run(
        [sys.executable, "-m", "nanobody_dash.proxy_accuracy"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=180,
    )
    return completed.returncode == 0


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
    except Exception as exc:
        # Funding fails closed.  Dashboard refresh and mining do not depend on
        # a payment instruction reaching the Pi.  Catch everything: a narrow
        # tuple let a TypeError (money.daily_money_history signature change)
        # abort every refresh before the build, freezing the dashboard.  Only
        # the exception class is logged, never its message.
        print(f"funding target skipped: {type(exc).__name__}", file=sys.stderr)
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
        if _should_refresh_compound():
            try:
                compound_ok = refresh_compound_status()
            except subprocess.TimeoutExpired:
                compound_ok = False
            except Exception as exc:
                print(f"compound refresh failed: {type(exc).__name__}", file=sys.stderr)
                compound_ok = False
            # main() always exits 0 even when the Pi was unreachable, so the
            # subprocess returncode alone can't tell a real refresh from a
            # fallback -- check the saved snapshot's own pi_reachable field.
            pi_reachable = True
            if compound_ok:
                try:
                    pi_reachable = bool(
                        json.loads(COMPOUND_STATUS_PATH.read_text())
                        .get("pi_reachable", True))
                except Exception:
                    pi_reachable = True
            _touch_compound_marker(success=bool(compound_ok) and pi_reachable)
        else:
            compound_ok = "skipped"
        if _should_refresh_proxy_accuracy():
            try:
                proxy_accuracy_ok = refresh_proxy_accuracy_status()
            except subprocess.TimeoutExpired:
                proxy_accuracy_ok = False
            except Exception as exc:
                print(f"proxy accuracy refresh failed: {type(exc).__name__}", file=sys.stderr)
                proxy_accuracy_ok = False
            reachable_boxes = 0
            if proxy_accuracy_ok:
                try:
                    reachable_boxes = int(
                        json.loads(PROXY_ACCURACY_STATUS_PATH.read_text())
                        .get("reachable_boxes", 0))
                except Exception:
                    reachable_boxes = 0
            _touch_proxy_accuracy_marker(
                success=bool(proxy_accuracy_ok) and reachable_boxes > 0)
        else:
            proxy_accuracy_ok = "skipped"
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
            f"compound={compound_ok}; vercel_publish_attempted={published}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
