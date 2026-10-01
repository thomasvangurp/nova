#!/usr/bin/env python3
"""Persist exact wave-to-artifact timings while fresh scoring runs."""

from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path("/root/h2-selector-proof-20260922")


def update() -> None:
    path = ROOT / "run_metadata.json"
    metadata = json.loads(path.read_text())
    timings = []
    for draw in range(9, 17):
        directory = ROOT / f"fresh_{draw:02d}"
        wave = directory / "wave_000_input.json"
        exact = directory / "exact_all.json"
        row = {
            "draw": draw,
            "started": wave.is_file(),
            "complete": exact.is_file(),
        }
        if wave.is_file():
            row["started_unix"] = wave.stat().st_mtime
        if exact.is_file():
            row["completed_unix"] = exact.stat().st_mtime
            if wave.is_file():
                row["duration_seconds"] = exact.stat().st_mtime - wave.stat().st_mtime
        timings.append(row)
    metadata["draw_timings"] = timings
    complete = [row for row in timings if row["complete"]]
    if complete:
        metadata["first_wave_started_unix"] = timings[0].get("started_unix")
        metadata["last_draw_completed_unix"] = complete[-1]["completed_unix"]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


while True:
    update()
    if (ROOT / "COMPLETE").is_file():
        break
    time.sleep(2)
