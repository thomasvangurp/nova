"""Publish the built dashboard to the private Vercel deployment.

refresh.py rebuilds public/index.html every 60 seconds; redeploying on every
build would mean ~1,440 deploys/day for content that mostly hasn't moved
between data-collection cycles. This throttles on BOTH a minimum interval and
a content-hash change, and is fully optional -- a Vercel outage or missing
CLI must not block the local dashboard refresh that already works today.

Access control lives on Vercel's side (Vercel Authentication / SSO), not
here: this script only ever pushes the already-built static file.

User, 2026-10-11: the actual ``vercel deploy`` must run from the Pi, not
this Mac -- the Mac's launchd jobs are session-gated (LimitLoadToSessionType
Aqua) and silently stop firing whenever the lid closes or the screen locks,
which is exactly the kind of fragile dependency the Pi's always-on
submission loop was already moved off a laptop to avoid (see
nova-submit-loop.service's own docstring). refresh.py keeps building
public/index.html here (it reads from hosts the Pi does not have the same
access path to), but the deploy step now rsyncs the built ``public/``
directory to the Pi and triggers ``vercel deploy`` over SSH there, using the
Vercel CLI auth transferred to the Pi's own
~/.local/share/com.vercel.cli/auth.json (never re-read or displayed from
this process -- SSH key auth only, no secret ever appears in an argv or in
this script's own output).
"""
from __future__ import annotations

import hashlib
import json
import os
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
        print(f"PI_DEPLOY_RSYNC_FAILED rc={rsync.returncode} "
              f"stderr={rsync.stderr[-500:]}", flush=True)
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
        print(f"PI_DEPLOY_VERCEL_FAILED rc={deploy.returncode} "
              f"stderr={deploy.stderr[-500:]}", flush=True)
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

    ok = _deploy_via_pi()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    next_state = dict(state)
    next_state.update({"last_attempted_at": now, "ok": ok})
    if ok:
        next_state.update({"content_hash": content_hash, "published_at": now})
    STATE.write_text(json.dumps(next_state))
    return True


if __name__ == "__main__":
    publish()
