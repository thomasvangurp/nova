"""Entry- and fleet-level leaderboard outcomes.

An epoch can have only one winning coldkey, but a coldkey can submit several
entries that all beat the best entry from every rival.  Those are distinct
entry-level successes and must not be collapsed to one epoch win when the
denominator is the number of submitted entries.
"""
from __future__ import annotations

from collections.abc import Iterable


def annotate_field_beating_entries(entries: Iterable[dict]) -> None:
    """Add ``entry_win`` to every entry in place.

    An entry wins when its score is no worse than the best score from every
    *other* coldkey in that epoch.  Consequently, if one coldkey owns overall
    positions one and two, both entries win.  Ties with the best rival also
    count, matching the validator replay's candidate-versus-field semantics.

    The rank-one fallback is only for incomplete fixtures or epochs containing
    one coldkey; real leaderboard epochs contain rival scores.
    """
    by_epoch: dict[int, list[dict]] = {}
    for entry in entries:
        by_epoch.setdefault(int(entry["epoch"]), []).append(entry)

    for rows in by_epoch.values():
        best_by_coldkey: dict[str | None, float] = {}
        for row in rows:
            coldkey = row.get("coldkey")
            score = row.get("score")
            if score is None:
                continue
            value = float(score)
            best_by_coldkey[coldkey] = min(
                value, best_by_coldkey.get(coldkey, value)
            )
        for row in rows:
            coldkey = row.get("coldkey")
            score = row.get("score")
            rival_scores = [
                value for rival, value in best_by_coldkey.items()
                if rival != coldkey
            ]
            if score is not None and rival_scores:
                row["entry_win"] = float(score) <= min(rival_scores)
            else:
                row["entry_win"] = int(row.get("rank") or 0) == 1


def is_entry_win(entry: dict) -> bool:
    """Return the annotated entry result, with a rank-one fixture fallback."""
    if "entry_win" in entry:
        return bool(entry["entry_win"])
    return int(entry.get("rank") or 0) == 1
