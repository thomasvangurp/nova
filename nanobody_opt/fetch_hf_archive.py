#!/usr/bin/env python3
"""Fetch the validator's exact HF archive (P20809_nanobodies.csv).

The validator builds its dedup top-50 from Metanova/Submission-Archive's
P20809_nanobodies.csv (all history, refreshed every epoch), NOT the live
leaderboard. Our builder was ranking only the last 15 API epochs, so it indexed
a different (current-basin) fifty sequences and over-filtered. This fetches the
real archive with the read-only token at /root/.hftoken, retries, falls back to
the last cached copy on failure, and refuses an empty result.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/P20809_nanobodies.csv"
TOKEN_PATH = Path("/root/.hftoken")


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/P20809_nanobodies.csv")
    # The archive is a public dataset; the token is optional and only sent when
    # a /root/.hftoken exists. A gated dataset would need a real read token, but
    # Submission-Archive resolves anonymously (user_id=public).
    token = None
    if TOKEN_PATH.is_file():
        token = TOKEN_PATH.read_text().strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    data: bytes | None = None
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(URL, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            break
        except Exception as exc:  # noqa: BLE001
            print(f"attempt {attempt} failed: {type(exc).__name__}", flush=True)

    if data is None or not data.strip():
        if out.exists() and out.stat().st_size > 0:
            print(f"using cached {out} ({out.stat().st_size} bytes)", flush=True)
            return 0
        raise SystemExit("HF archive fetch failed and no cached copy")

    out.write_bytes(data)
    print(f"HF_ARCHIVE_OK bytes={len(data)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
