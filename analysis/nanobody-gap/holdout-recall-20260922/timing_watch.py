#!/usr/bin/env python3
"""Persist exact per-draw file timings while the remote run is active."""

from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path("/root/holdout-recall-20260922")


def update() -> None:
    metadata_path = ROOT / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    timings = []
    for draw in range(5, 9):
        directory = ROOT / f"heldout_{draw:02d}"
        wave = directory / "wave_000_input.json"
        exact = directory / "exact_all.json"
        row: dict[str, float | int | bool] = {
            "draw": draw,
            "started": wave.is_file(),
            "complete": exact.is_file(),
        }
        if wave.is_file():
            row["started_unix"] = wave.stat().st_mtime
        if exact.is_file():
            row["completed_unix"] = exact.stat().st_mtime
            if wave.is_file():
                row["duration_seconds"] = (
                    exact.stat().st_mtime - wave.stat().st_mtime
                )
        timings.append(row)
    metadata["draw_timings"] = timings
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_path)


while True:
    update()
    if (ROOT / "COMPLETE").is_file():
        break
    time.sleep(5)
