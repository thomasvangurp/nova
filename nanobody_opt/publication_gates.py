"""The single source of truth for what may be published.

There are two programs that decide whether a candidate is publishable: the race,
which runs strict ranking at the end of an epoch, and the live publisher, which
watches for a qualified ranking and submits it.  They used to hold separate
hardcoded copies of the same thresholds, in separate repositories, and only one
of those repositories is version controlled.

That cost real epochs.  Relaxing the quality bar in this repository and
deploying it changed nothing, because the program that actually publishes read a
different file -- so for hours the miner kept refusing to publish while the
config said otherwise.  Both now import from here.

This module deliberately has NO third-party imports.  The publisher runs in a
different Pixi environment from the scorer, so anything requiring pandas or
numpy could not be imported there, and a shared source of truth that only one
side can load is not shared at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# Only rank one pays.  An epoch that publishes nothing and an epoch that
# publishes a mediocre candidate both earn nothing -- but the published one still
# holds a ticket on the validator's own single draw, and costs nothing to hold.
# The quality entries below are therefore a sanity floor, not a selection
# criterion: selection happens in ranking, where it can use evidence.
#
# They were previously absolute thresholds on a drifting scale.  Rank sums are
# computed against a replayed competitor field, and that field fills with
# stronger competitors every epoch, so a fixed ceiling of 90 became unreachable
# without any change to our molecules: re-scoring the sequence the validator paid
# 79 for put it at median 180.
QUALITY_GATES: dict[str, str] = {
    "--competitive-score-threshold": "60",
    "--minimum-competitive-rate": "0.0",
    "--minimum-observation-competitive-rate": "0.0",
    "--maximum-p90-rank-sum": "400",
    "--maximum-p90-confidence-rank-sum": "300",
    "--maximum-p90-physical-interaction-rank-sum": "100",
    "--maximum-p90-developability-rank-sum": "30",
}

# Evidence requirements.  One observation of a fixed sequence lands about 34
# leaderboard rank points from its own mean, so more observations buy a better
# estimate.  Refusing to publish for want of the eighth observation would trade a
# possible win for a certain zero, so this is not a hard floor: ranking relaxes
# it one observation at a time rather than publishing nothing.
#
# It is not one either.  A single pass is where our screen stops agreeing with
# the validator at all -- measured Spearman by observation count was -0.184 at
# one, +0.230 at two to three and +0.240 at four or more -- so selecting on a
# one-observation estimate is selecting on noise.  Replaying 27 real epoch
# fields with screen noise calibrated to the measured single-pass correlation,
# the best true final_score of a twelve-ticket basket improved monotonically
# with passes (72.4 -> 70.9 -> 68.9 -> 66.3 for k = 1, 2, 4, 8), and by three
# times as much for a single-ticket epoch.  Four is where the measured
# correlation plateaus; the race already targets eight and narrows the
# shortlist rather than the observation count, so this floor is reachable.
EVIDENCE_GATES: dict[str, str] = {
    "--minimum-observations": "4",
    "--minimum-structure-observations": "1",
}

# Integrity requirements.  These are NOT about how good a candidate looks; they
# are about whether the thing we would publish is the thing we measured.  They
# are not relaxed for score.
INTEGRITY_GATES: dict[str, str] = {
    "--minimum-median-contact-jaccard": "0.35",
    "--maximum-cdr3-rmsd": "5",
    "--minimum-epitope-occupancy": "0",
}


def gate_defaults() -> dict[str, str]:
    """Every gate, by flag name."""
    return {**EVIDENCE_GATES, **QUALITY_GATES, **INTEGRITY_GATES}


def _canonical(flag: str) -> str:
    return flag if flag.startswith("--") else f"--{flag.replace('_', '-')}"


def gate_arguments(overrides: Mapping[str, Any] | None = None) -> list[str]:
    """Flatten the gates into an argument list, applying overrides by name.

    Overrides accept either ``--maximum-p90-rank-sum`` or
    ``maximum_p90_rank_sum``; an unknown name is an error rather than a silently
    ignored typo, because a typo here would quietly restore a gate nobody
    intended.
    """
    gates = gate_defaults()
    for flag, value in (overrides or {}).items():
        name = _canonical(str(flag))
        if name not in gates:
            raise ValueError(f"unknown publication gate: {flag}")
        gates[name] = str(value)
    return [item for flag, value in gates.items() for item in (flag, value)]


def load_overrides(path: Path | str | None) -> dict[str, Any]:
    """Read re-derived thresholds, accepting a bare object or a ``gates`` key."""
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    gates = payload.get("gates", payload) if isinstance(payload, dict) else None
    if not isinstance(gates, dict):
        raise ValueError(f"publication gate overrides must be an object: {path}")
    return gates
