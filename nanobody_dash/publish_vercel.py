"""Publish the built dashboard to the private Vercel deployment.

refresh.py rebuilds public/index.html every 60 seconds; redeploying on every
build would mean ~1,440 deploys/day for content that mostly hasn't moved
between data-collection cycles. This throttles on BOTH a minimum interval and
a content-hash change, and is fully optional -- a Vercel outage or missing
CLI must not block the local dashboard refresh that already works today.

Access control lives on Vercel's side (Vercel Authentication / SSO), not
here: this script only ever pushes the already-built static file.
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
MIN_INTERVAL_SECONDS = 300


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


def publish() -> bool:
    """Deploy to production if enough time passed and content changed.

    Returns whether a deploy was attempted (not whether it succeeded) so the
    caller can log something meaningful without this raising.
    """
    if not LIVE.is_file():
        return False
    state = _state()
    now = time.time()
    content_hash = _sha256(LIVE)
    if (
        state.get("content_hash") == content_hash
        and now - float(state.get("published_at", 0)) < MIN_INTERVAL_SECONDS
    ):
        return False
    if now - float(state.get("published_at", 0)) < MIN_INTERVAL_SECONDS:
        return False

    npx = os.environ.get("NOVA_NPX_PATH", "/opt/homebrew/bin/npx")
    result = subprocess.run(
        [npx, "vercel", "deploy", "--prod", "--yes", "--non-interactive"],
        cwd=str(ROOT / "public"),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
    )
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({
        "content_hash": content_hash,
        "published_at": now,
        "ok": result.returncode == 0,
    }))
    return True


if __name__ == "__main__":
    publish()
