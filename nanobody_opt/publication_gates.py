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

# Evidence requirements.  The floor is ONE, and raising it is a trap.
#
# It looks like it should be higher.  Our screen's agreement with the validator
# rises sharply with the number of observations a sequence has (measured
# Spearman -0.184 at one, +0.230 at two to three, +0.240 at four or more), and
# replaying past epochs with independent repeat screens says averaging them is
# worth several rank points.  Both of those are real.  Neither licenses a floor
# above one, because of what an "observation" actually is here.
#
# Measured on the live score bank: a sequence's observation count equals the
# number of distinct epochs it appears in, exactly, at every depth -- 1115
# sequences at depth 1 spanning a mean of 1.00 epochs, 202 at depth 5 spanning
# 5.00, and so on.  The producer screens each candidate once per epoch and never
# repeats one within an epoch, so depth is not replication.  It is age.
#
# A floor above one is therefore a survivorship filter, not an evidence filter.
# Every one of the 200 candidates our generator produced in epoch 25134 had
# depth 1; a floor of four would have excluded all of them and restricted
# selection to molecules carried over from four or more previous epochs.  That
# also explains the correlation above: depth is confounded with having survived,
# so the rise in Spearman cannot be read as replication buying accuracy.
#
# Averaging whatever observations a sequence does have is still right, and
# ranking already does it -- those are genuine repeat draws of a fixed sequence.
# The change that would actually buy the replay's benefit is in the PRODUCER:
# screen each candidate k times within one epoch.  Until that exists, this stays
# at one.
#
# 2026-09-28: it exists and the floor is now eight.  The producer's confirmation
# stage repeat-scores its 96 finalists up to 12 times within the epoch, and it
# now finishes (--reserve-blocks 120) before the Pi commits, which the Pi holds
# until just before its safe deadline.  A fresh candidate therefore reaches eight
# observations in its first epoch, so the floor is an evidence filter again, not
# a survivorship filter.  Measured on real validator outcomes, epochs
# 25335-25394: our committed tickets with 1-5 observations made the top 10 at
# 12.3% (399 tickets, 53% of the fleet) against ~20% at 6-40 observations.
# Eight matches the producer's own --minimum-publishable-observations.
EVIDENCE_GATES: dict[str, str] = {
    "--minimum-observations": "8",
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
