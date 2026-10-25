"""Independent, bounded Pi dashboard jobs. No laptop or interactive session."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COMPARISON = ROOT / "nanobody_dash" / "multipass_comparison.json"
VAST = "/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai"
IDENTITY = Path.home() / ".ssh" / "id_ed25519"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".next")
    pending.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(pending, path)


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    # Even authentication/timeout errors remain private. Callers emit fixed
    # reason codes, never CLI output, HTTP bodies or exception messages.
    return subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout,
                          check=False)


def module(name: str, timeout: int, *args: str) -> bool:
    return run([sys.executable, "-m", name, *args], timeout).returncode == 0


def latest_leaderboard() -> int | None:
    try:
        with sqlite3.connect(f"file:{DATA / 'nanobody_dash.sqlite'}?mode=ro", uri=True) as db:
            return db.execute("SELECT MAX(epoch) FROM epochs").fetchone()[0]
    except sqlite3.Error:
        return None


def latest_multipass(payload: dict) -> int | None:
    return max((int(k) for k in payload if k.isdecimal()), default=None)


def refresh() -> dict:
    # Collection failure still permits rendering last good data, with an
    # explicit failed stage in health. Auxiliary collectors have other locks.
    try:
        collected = module("nanobody_dash.collect", 150, "--history", "20",
                           "--slot-log", str(Path.home() / ".nova/subnet68_slots.jsonl"))
    except subprocess.TimeoutExpired:
        collected = False
    out = ROOT / "public" / "index.html.next"
    built = module("nanobody_dash.build", 120, "--out", str(out))
    if built:
        os.replace(out, out.with_name("index.html"))
        os.replace(out.with_name("version.json.next"), out.with_name("version.json"))
    return {"ok": collected and built, "collected": collected, "built": built,
            "leaderboard_epoch": latest_leaderboard(),
            "multipass_epoch": latest_multipass(read_json(COMPARISON))}


def publish() -> dict:
    from . import publish_vercel
    attempted = publish_vercel.publish()
    state = publish_vercel._state()
    return {"ok": bool(state.get("ok")), "attempted": attempted,
            "published_at": state.get("published_at"),
            "reason": "published_or_throttled" if state.get("ok") else "vercel_deploy_failed"}


def master_endpoint() -> tuple[str, int]:
    result = run([VAST, "show", "instances", "--raw"], 45)
    if result.returncode:
        raise RuntimeError("endpoint_lookup_failed")
    body = json.loads(result.stdout)
    instances = body.get("instances", []) if isinstance(body, dict) else body
    matches = [v for v in instances if v.get("label") == "nova-nanobody-pixi"
               and v.get("actual_status") == "running"]
    if len(matches) != 1:
        raise RuntimeError("master_not_unique")
    host, port = matches[0].get("ssh_host"), int(matches[0].get("ssh_port", 0))
    if not isinstance(host, str) or not re.fullmatch(r"[a-zA-Z0-9.-]+", host) or not 0 < port < 65536:
        raise ValueError("invalid_endpoint")
    return host, port


def accept_comparison(pending: Path) -> int:
    body = json.loads(pending.read_text())
    if not isinstance(body, dict) or not body or any(
            not k.isdecimal() or not isinstance(v, dict) or "method" not in v
            for k, v in body.items()):
        raise ValueError("invalid_comparison")
    newest = latest_multipass(body)
    previous = latest_multipass(read_json(COMPARISON))
    if previous is not None and newest < previous:
        raise ValueError("comparison_regressed")
    os.replace(pending, COMPARISON)
    return newest


def multipass() -> dict:
    host, port = master_endpoint()
    options = ["-i", str(IDENTITY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=12", "-o", "StrictHostKeyChecking=accept-new"]
    outcomes_ok = submission_outcomes(host, port, options)
    # timeout runs remotely too: losing SSH must not leave an unbounded export.
    try:
        exported = run(["ssh", *options, "-p", str(port), "root@" + host,
                        "cd /root && timeout --kill-after=10 240 "
                        "/root/.pixi/envs/bgen/bin/python -m tools.build_multipass_comparison --recent 6"], 270)
        export_code = exported.returncode
    except subprocess.TimeoutExpired:
        export_code = 124
    pending = COMPARISON.with_suffix(".json.download")
    try:
        pulled = run(["scp", *options, "-P", str(port),
                      f"root@{host}:/root/artifacts/multipass_comparison.json", str(pending)], 60)
        if pulled.returncode:
            return {"ok": False, "reason": "multipass_transfer_failed"}
        newest = accept_comparison(pending)
    finally:
        pending.unlink(missing_ok=True)
    latest = latest_leaderboard()
    caught_up = latest is None or newest >= latest
    return {"ok": export_code == 0 and caught_up and outcomes_ok,
            "export_returncode": export_code, "multipass_epoch": newest,
            "submission_outcomes_synced": outcomes_ok,
            "leaderboard_epoch": latest,
            "reason": "current" if caught_up else "awaiting_multipass_export"}


def submission_outcomes(host, port, options) -> bool:
    """Push Pi's finalized input/outcome proof through the existing scheduled job."""
    from .submission_outcomes import export
    path = DATA / "pi_submission_outcomes.json"
    try:
        export(DATA / "nanobody_dash.sqlite", Path("/opt/nova-submit/archive"), path)
        pending = "/root/artifacts/pi_submission_outcomes.json.next"
        copied = run(["scp", "-O", *options, "-P", str(port), str(path),
                      f"root@{host}:{pending}"], 60)
        if copied.returncode:
            return False
        promoted = run(["ssh", *options, "-p", str(port), "root@" + host,
                        "mv /root/artifacts/pi_submission_outcomes.json.next "
                        "/root/artifacts/pi_submission_outcomes.json"], 30)
        return promoted.returncode == 0
    except (OSError, ValueError, sqlite3.Error, subprocess.TimeoutExpired):
        return False


def status_text() -> str:
    if os.environ.get("NOVA_DASH_LOCAL_PI") != "1":
        return ""
    payload = read_json(COMPARISON)
    newest = latest_multipass(payload)
    measured = max((int(k) for k, v in payload.items() if k.isdecimal()
                    and isinstance(v, dict) and (v.get("ours") or {}).get("mean") is not None), default=None)
    health = read_json(DATA / "dashboard_multipass_health.json")
    age = int(max(0, time.time() - float(health.get("last_success_at") or 0)) / 60)
    sync = "pending" if not health else ("failed; retrying" if not health.get("ok") else f"{age}m ago")
    return (f" · updates from Pi · multipass export {newest or 'pending'}"
            f" · measured fleet {measured or 'pending'} · multipass refresh {sync}")


def auxiliary(job: str) -> dict:
    if job == "compound":
        ok = module("nanobody_dash.compound", 1200)
        state = read_json(DATA / "compound_status.json")
        return {"ok": ok and bool(state.get("pi_reachable"))
                and bool(state.get("gap_refresh_complete"))}
    if job == "proxy":
        return {"ok": module("nanobody_dash.proxy_accuracy", 180)}
    # Independent attempts: an unavailable API never prevents chain balances.
    status = {}
    for name in ("economics", "infra"):
        try:
            status[name] = module("nanobody_dash." + name, 240)
        except (OSError, subprocess.TimeoutExpired):
            status[name] = False
    return {"ok": all(status.values()), **status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("refresh", "publish", "multipass", "compound", "money", "proxy"))
    job = parser.parse_args().job
    if os.environ.get("NOVA_DASH_LOCAL_PI") != "1":
        print(json.dumps({"ok": False, "reason": "pi_mode_required"}))
        return 1
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / f"dashboard_{job}.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        path = DATA / f"dashboard_{job}_health.json"
        previous = read_json(path)
        started = time.time()
        try:
            result = {"refresh": refresh, "publish": publish, "multipass": multipass}.get(
                job, lambda: auxiliary(job))()
        except Exception as error:
            result = {"ok": False, "reason": type(error).__name__}
        result.update(job=job, host="pi", attempted_at=started, finished_at=time.time(),
                      last_success_at=time.time() if result["ok"] else previous.get("last_success_at"))
        save(path, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
