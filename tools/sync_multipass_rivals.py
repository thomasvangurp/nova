#!/usr/bin/env python3
"""Pull multipass_comparison.json directly from master's local sqlite build.

User, 2026-10-11: "Dashboard needs to read from SQLite!" / "One source of
truth please." This used to pull 100+ separate per-epoch JSON files from the
dashboard box (a snapshot of whatever field was live when each was written,
independently cached, with its own pin-tracking state to survive the box
silently reverting out-of-band fixes -- see git history on this file for
that whole saga). master's local sqlite is the actual primary store now
(nanobody_opt/mutation_network.py routes there by file existence, no env
var); tools/build_multipass_comparison.py runs continuously there and
already produces the complete, consistently-scaled file this dashboard
wants. Pulling that one file is now a straight overwrite -- there is no
second, independently-drifting source left to protect against, so the old
pin/merge machinery is gone, not hardened further.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = "root@ssh9.vast.ai"
PORT = "10616"
REMOTE_PATH = "/root/artifacts/multipass_comparison.json"
LOCAL = ROOT / "nanobody_dash" / "multipass_comparison.json"


def main() -> int:
    tmp = LOCAL.with_suffix(".json.tmp")
    result = subprocess.run(
        [
            "scp", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-P", PORT, f"{MASTER}:{REMOTE_PATH}", str(tmp),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"MULTIPASS_SYNC_FAILED rc={result.returncode} "
              f"stderr={result.stderr.strip()[-500:]}", file=sys.stderr, flush=True)
        tmp.unlink(missing_ok=True)
        return 1
    if not tmp.is_file() or tmp.stat().st_size == 0:
        print("MULTIPASS_SYNC_FAILED empty or missing file after scp",
              file=sys.stderr, flush=True)
        tmp.unlink(missing_ok=True)
        return 1
    tmp.replace(LOCAL)
    print(f"MULTIPASS_SYNC_OK bytes={LOCAL.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
