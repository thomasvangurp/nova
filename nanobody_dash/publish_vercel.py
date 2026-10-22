"""Publish the static dashboard using the Pi's own Vercel CLI authentication.

Production collection, rendering and deployment run as independent Pi systemd
jobs. A deployment failure is persisted separately from an attempted publish;
last successful content remains online. The legacy SSH transport is available
for manual maintenance, but no scheduled Mac job is required.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "public" / "index.html"
STATE = ROOT / "data" / "nanobody_dash.vercel_publish_state.json"
COMPOUND_STATUS = ROOT / "data" / "compound_status.json"
MIN_INTERVAL_SECONDS = 300
FAILED_RETRY_SECONDS = 60

#: Deploy host. Overridable for local testing; production always goes
#: through the Pi (see module docstring).
PI_HOST = os.environ.get("NOVA_DASH_DEPLOY_HOST", "thomasg@100.64.243.65")
PI_PUBLISH_DIR = os.environ.get("NOVA_DASH_DEPLOY_DIR", "nova-dash-publish/public")


def _deploy_local() -> bool:
    """Use the Pi's existing CLI authentication without reading its store."""
    destination = Path(PI_PUBLISH_DIR).expanduser()
    if not destination.is_absolute():
        destination = Path.home() / destination
    destination.mkdir(parents=True, exist_ok=True)
    # Only public artifacts: never copy/delete the CLI's project/auth state.
    for name in ("index.html", "version.json"):
        source = ROOT / "public" / name
        if source.is_file():
            pending = destination / (name + ".next")
            shutil.copyfile(source, pending)
            os.replace(pending, destination / name)
    completed = subprocess.run(
        ["npx", "--no-install", "vercel@65.1.0", "deploy", "--prod", "--yes", "--non-interactive"],
        cwd=destination, capture_output=True, text=True, timeout=150,
    )
    if completed.returncode:
        print(f"VERCEL_DEPLOY_FAILED rc={completed.returncode}", flush=True)
    return completed.returncode == 0


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deploy_via_pi() -> bool:
    """rsync public/ to the Pi, then run ``vercel deploy`` there over SSH.

    Returns whether the deploy succeeded. Never raises -- an unreachable Pi
    or a Vercel failure both just mean this cycle's deploy didn't happen; the
    next refresh cycle retries. No secret ever appears in an argv here: SSH
    key auth only, and the Vercel token lives solely in a file on the Pi
    this process never reads.
    """
    rsync = subprocess.run(
        [
            "rsync", "-az", "--delete",
            f"{ROOT / 'public'}/",
            f"{PI_HOST}:{PI_PUBLISH_DIR}/",
        ],
        capture_output=True, text=True, timeout=60,
    )
    if rsync.returncode != 0:
        print(f"PI_DEPLOY_RSYNC_FAILED rc={rsync.returncode}", flush=True)
        return False
    deploy = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", PI_HOST,
            f"cd {PI_PUBLISH_DIR} && npx --yes vercel@latest deploy "
            "--prod --yes --non-interactive",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if deploy.returncode != 0:
        print(f"PI_DEPLOY_VERCEL_FAILED rc={deploy.returncode}", flush=True)
        return False
    return True


def publish(*, force: bool = False) -> bool:
    """Deploy to production if enough time passed and content changed.

    Returns whether a deploy was attempted (not whether it succeeded) so the
    caller can log something meaningful without this raising.
    """
    if not LIVE.is_file():
        return False
    state = _state()
    now = time.time()
    content_hash = _sha256(LIVE)
    published_at = float(state.get("published_at", 0))
    last_attempted_at = float(state.get("last_attempted_at", 0))
    compound_changed = (
        COMPOUND_STATUS.is_file()
        and COMPOUND_STATUS.stat().st_mtime > published_at
    )
    if not force and (
        state.get("content_hash") == content_hash
        and not compound_changed
    ):
        return False
    if (not force and not state.get("ok", True)
            and now - last_attempted_at < FAILED_RETRY_SECONDS):
        return False
    # A new compound-gate snapshot is operational data, not cosmetic page
    # churn: publish it immediately. Other changes retain the deployment
    # throttle.
    if (not force and not compound_changed
            and now - published_at < MIN_INTERVAL_SECONDS):
        return False

    try:
        ok = _deploy_local() if os.environ.get("NOVA_DASH_LOCAL_PI") == "1" else _deploy_via_pi()
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    STATE.parent.mkdir(parents=True, exist_ok=True)
    next_state = dict(state)
    next_state.update({"last_attempted_at": now, "ok": ok})
    if ok:
        next_state.update({"content_hash": content_hash, "published_at": now})
    pending = STATE.with_suffix(".json.next")
    pending.write_text(json.dumps(next_state))
    os.replace(pending, STATE)
    return True


if __name__ == "__main__":
    publish()
