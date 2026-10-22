"""Leaderboards the public API never published, rebuilt from the validator's own log.

Since 2026-10-13 the validator's upload to the Dashboard API is rejected (HTTP 422), so
``/competitions/leaderboard/<epoch>/nanobodies`` answers 404 for every epoch from 25693 on, although the epoch was
ranked and paid.  ``tools/leaderboard_from_validator_log.py`` rebuilds API-shaped rows from the validator's score dump;
the multipass replay and the dashboard's comparison builder fall back to them when the API has nothing.  The API
always wins: these files are only read when it returns no rows.
"""
from __future__ import annotations

import json
from pathlib import Path

DIRECTORY = Path("/root/artifacts/leaderboard_from_validator_log")


def load_rows(epoch: int, directory: Path | str | None = None) -> list[dict] | None:
    """The rebuilt submission rows for ``epoch`` (API shape), or None when there is no usable file."""
    try:
        body = json.loads((Path(directory or DIRECTORY) / f"{int(epoch)}.json").read_text())
    except (OSError, ValueError):
        return None
    rows = body.get("data") if isinstance(body, dict) else body
    return rows if isinstance(rows, list) and rows else None
