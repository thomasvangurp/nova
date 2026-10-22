"""Emulate a specific rival's own single-edit strategy against the full public pool.

Built from a concrete finding, not a general model: one coldkey's public
submissions (201 of them, across 91 epochs) are 196/201 exactly distance-1
from something already public, and 63% of those single edits land at just two
framework positions (the extreme N-terminus). A walk-forward split (position
list learned only from earlier epochs, checked against later, held-out
submissions) recovered 84% of held-out edits at the top-10 positions and 94%
at the top-15 -- the concentration is stable over time, not hindsight.

This is deliberately NOT our own elite-parent bank. `select_elite_parents.py`
narrows to rank/basin-qualified sequences; backtesting against this rival's
actual moves showed 5/5 of their real parents were mediocre in their own
epoch (ranks as low as 28/51) and were never in our bank. The parent source
here is every sequence anyone has published, unfiltered by rank.

A general historical mutation-effect proxy (mutation_network.load_mutation_effects,
averaged across every background it was ever tried in) was checked first and
rejected: it pointed the wrong direction for 4 of 6 known winning edits. Rival-
specific position frequency, not a marginal score-delta model, is the signal
that held up.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from nanobody_opt import gates


def learn_hot_positions(
    submissions: list[tuple[int, int, list[int]]],
    *,
    before_epoch: int,
    top_n: int,
) -> list[int]:
    """Rank 1-indexed positions by how often this rival's OWN distance-1
    submissions, strictly before ``before_epoch``, land there.

    ``submissions`` is ``(epoch, edit_distance_to_nearest_prior_public, diff_positions)``
    per submission -- the same shape the walk-forward backtest and any live
    caller both produce. Only distance-1 rows carry a single, unambiguous
    position; distance-0 (resubmission) and distance>=2 rows are excluded on
    purpose so one multi-position edit cannot dilute the frequency count.
    """
    counts = Counter()
    for epoch, distance, positions in submissions:
        if epoch >= before_epoch or distance != 1 or len(positions) != 1:
            continue
        counts[positions[0]] += 1
    return [position for position, _ in counts.most_common(top_n)]


def position_restricted_candidates(
    parent: str,
    *,
    allowed: list[list[str]],
    positions: list[int],
) -> dict[str, int]:
    """Single-substitution candidates at a fixed, externally-supplied position set.

    ``positions`` is 1-indexed (matching ``learn_hot_positions`` output and
    every other position reported in this session's analysis). Returns
    ``{candidate_sequence: position_used(1-indexed)}``; a position at or past
    the sequence length, or with no supported alternate residue, contributes
    nothing rather than raising.
    """
    if len(allowed) != len(parent):
        raise ValueError("allowed residue table has the wrong length")
    candidates: dict[str, int] = {}
    for position in positions:
        index = position - 1
        if not 0 <= index < len(parent):
            continue
        for residue in allowed[index]:
            if residue == parent[index] or residue not in gates.ALLOWED_AAS:
                continue
            candidate = parent[:index] + residue + parent[index + 1 :]
            candidates.setdefault(candidate, position)
    return candidates


def _parent_cdr3_in_top50_basin(parent: str, top50_cdr3s: list[str] | None) -> bool:
    """True when a parent's CDR3 clonotype sits at >= 0.85 positional identity
    to an archived top-50 clonotype. Fail-closed: a missing index or an
    unannotatable parent is treated as in-basin (excluded)."""
    if top50_cdr3s is None:
        # Gate not requested (pure-generation caller): do not exclude. The
        # producer always supplies the index, and its upstream competitor_lane
        # filter fails closed when the index is unavailable.
        return False
    if not top50_cdr3s:
        # Index requested but empty/unparseable: fail-closed (treat as in-basin).
        return True
    from nanobody_opt.regions import find_regions
    try:
        regions = find_regions(parent)
    except Exception:
        return True
    cdr3 = parent[regions.cdr3[0]:regions.cdr3[1]]
    if not cdr3:
        return True
    length = len(cdr3)
    for other in top50_cdr3s:
        if not other:
            continue
        shared = sum(1 for a, b in zip(cdr3, other) if a == b)
        if shared / max(length, len(other)) >= 0.85:
            return True
    return False


def competitor_position_candidates(
    parents: list[str],
    *,
    allowed: list[list[str]],
    positions: list[int],
    top50_cdr3s: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Apply ``position_restricted_candidates`` across every supplied parent.

    ``parents`` is meant to be the WIDE public pool (any submitter, not our
    rank/basin-filtered elite bank -- see module docstring for why). Returns
    ``{candidate_sequence: {"parent": ..., "position": ...}}``, first-parent-wins
    on a collision so the result stays small and deterministic.

    ``top50_cdr3s`` (optional) gates parents whose CDR3 clonotype matches an
    archived top-50 clonotype: a framework single-edit preserves the parent CDR3,
    so such a parent can only seed doomed children. ``None`` leaves the gate
    inactive for pure-generation callers; the producer always supplies the real
    index and fails closed upstream (competitor_lane) when it is unavailable.
    """
    results: dict[str, dict[str, Any]] = {}
    for parent in parents:
        if _parent_cdr3_in_top50_basin(parent, top50_cdr3s):
            continue
        for candidate, position in position_restricted_candidates(
            parent, allowed=allowed, positions=positions
        ).items():
            results.setdefault(candidate, {"parent": parent, "position": position})
    return results


def double_position_candidates(
    parent: str,
    *,
    allowed: list[list[str]],
    positions: list[int],
    pair_limit: int,
) -> dict[str, tuple[int, int]]:
    """Two-substitution candidates at pairs of the given 1-indexed positions.

    Two edits still clear the validator's dedup (the dedup reference is the
    full-history HF top-50, ~30 edits from the live field), and landing both at
    the rival's measured framework positions keeps the edit set concentrated.
    ``pair_limit`` caps the number of (position, position) pairs so the
    residues^2 fan-out stays bounded.
    """
    if len(allowed) != len(parent):
        raise ValueError("allowed residue table has the wrong length")
    candidates: dict[str, tuple[int, int]] = {}
    pairs = [(a, b) for i, a in enumerate(positions) for b in positions[i + 1:]]
    for pos1, pos2 in pairs[:pair_limit]:
        i1, i2 = pos1 - 1, pos2 - 1
        if not (0 <= i1 < len(parent) and 0 <= i2 < len(parent)):
            continue
        for r1 in allowed[i1]:
            if r1 == parent[i1] or r1 not in gates.ALLOWED_AAS:
                continue
            for r2 in allowed[i2]:
                if r2 == parent[i2] or r2 not in gates.ALLOWED_AAS:
                    continue
                seq = parent[:i1] + r1 + parent[i1 + 1:i2] + r2 + parent[i2 + 1:]
                candidates.setdefault(seq, (pos1, pos2))
    return candidates
