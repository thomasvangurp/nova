#!/usr/bin/env python3
"""Pull the box's multi-pass rival scoring into the dashboard JSON.

The box cron loop runs ``multipass_all_rivals.py`` per completed epoch and
writes ``/root/multipass_all_rivals/<epoch>/multipass_<epoch>.json``. This
script reads every such JSON from the box and merges it into
``nanobody_dash/multipass_comparison.json`` (overwriting any null placeholder
with the real per-rival ``mean``/``stdev``/``fleet_size``/``scores``).

The dashboard refresh loop (launchd, every 60s) then rebuilds + publishes the
changed JSON with no further action here.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

BOX = "root@ssh3.vast.ai"
PORT = "11806"
REMOTE = "/root/multipass_all_rivals"
LOCAL = Path(__file__).resolve().parents[1] / "nanobody_dash" / "multipass_comparison.json"
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


def main() -> int:
    listing = _ssh(["find", REMOTE, "-name", "multipass_*.json"])
    remote_files = [line.strip() for line in listing.splitlines()
                    if line.strip().endswith(".json")]
    if not remote_files:
        return 0

    try:
        data = json.loads(LOCAL.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}

    changed = False
    for remote in sorted(remote_files):
        raw = _ssh(["cat", remote])
        if not raw:
            continue
        try:
            epoch_data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for epoch, value in epoch_data.items():
            if not isinstance(value, dict):
                continue
            # only commit epochs that carry a real ours-entry or >=1 rival
            if value.get("ours") or value.get("rivals"):
                data[epoch] = value
                changed = True

    if changed:
        LOCAL.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"merged {len(remote_files)} remote files into {LOCAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
