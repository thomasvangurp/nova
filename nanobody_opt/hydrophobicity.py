"""Cheap sequence-level hydrophobicity pre-screen for PR100 nanobody candidates.

The native developability gate computes ``surface_hydrophobic_patches`` (PSH)
from a 3D fold (NanoBodyBuilder2 + OpenMM refine), ~200 ms per candidate, and
rejects 83% of fresh Q9NZQ7 candidates at PSH > 89. This module is the
sequence-only safety net: it estimates CDR3 hydrophobicity from the amino-acid
string in microseconds and drops candidates that are clearly hydrophobic
before the fold. It is the same shape as the Boltz-2 "skip-pass-1" approximate
screen: cheap, sequence-only, and only allowed to reject the obvious tail.

Metric
------
The score is the hydrophobic fraction of the CDR3 loop over the residue set
F, L, I, V, M, W, Y, A, located with ``nanobody_opt.regions.find_regions``.
CDR3 is the paratope and the most discriminating region. ``find_regions`` CDR2
is a fixed 17-residue framework window that is ~0.41 for every VHH in this
family, so it does not separate top miners from rejects and is deliberately
not used for the pass/fail decision.

Calibration (top-50 public Q9NZQ7 miners vs our PSH-rejected candidates):

* CDR3 hydrophobic fraction: top miners median ~0.31 (observed ceiling ~0.41),
  our rejects median ~0.46.
* Overall hydrophobic fraction: top ~0.35, our rejects ~0.40.

The default ceiling sits above the observed top-miner ceiling and just below
the reject median, so it only drops the clearly hydrophobic tail. It is
configurable via ``SN68_HYDROPHOBIC_CDR3_MAX`` (and the ``cdr3_max`` keyword
argument) so it can be tuned without a code change.
"""
from __future__ import annotations

import os

from nanobody_opt.regions import find_regions

#: F, L, I, V, M, W, Y, A -- the empirical hydrophobic set used for the
#: calibration numbers above (A is borderline, but it is included to match the
#: observed top-miner/reject separation).
HYDROPHOBIC = frozenset("FLIVMWYA")

#: Default ceiling on the CDR3 hydrophobic fraction. See the module docstring
#: for the calibration. Env-overridable so the threshold is tunable in the
#: deployed box without editing this file.
CDR3_HYDROPHOBIC_FRACTION_MAX = float(
    os.environ.get("SN68_HYDROPHOBIC_CDR3_MAX", "0.44"))

#: Distinct gate reason recorded by sequence_phase when this screen rejects.
REJECT_REASON = "surface_hydrophobicity_screen"


def _fraction(seq: str, positions) -> float:
    if not positions:
        return 0.0
    return sum(1 for i in positions if seq[i] in HYDROPHOBIC) / len(positions)


def score(seq: str) -> dict | None:
    """Return a breakdown of hydrophobic fractions, or None if not VHH-shaped.

    The returned mapping is diagnostics-friendly (``overall``, ``cdr1``,
    ``cdr2``, ``cdr3``). The pass/fail decision in :func:`check` uses only the
    CDR3 fraction.
    """
    s = seq.strip().upper()
    try:
        regions = find_regions(s)
    except ValueError:
        return None
    return {
        "overall": _fraction(s, range(len(s))),
        "cdr1": _fraction(s, range(regions.cdr1[0], regions.cdr1[1])),
        "cdr2": _fraction(s, range(regions.cdr2[0], regions.cdr2[1])),
        "cdr3": _fraction(s, range(regions.cdr3[0], regions.cdr3[1])),
    }


def check(seq: str, *, cdr3_max: float | None = None) -> tuple[float | None, bool]:
    """Return ``(cdr3_fraction_or_None, passed)`` against the configured ceiling.

    A sequence that ``find_regions`` cannot parse is returned as passed: the
    screen is only a cheap safety net, and the annotation/native gates remain
    the authority for candidates it cannot cheaply assess.
    """
    breakdown = score(seq)
    if breakdown is None:
        return None, True
    ceiling = (CDR3_HYDROPHOBIC_FRACTION_MAX if cdr3_max is None
               else float(cdr3_max))
    return breakdown["cdr3"], breakdown["cdr3"] <= ceiling


def reason(seq: str, *, cdr3_max: float | None = None) -> str:
    """Return ``REJECT_REASON`` when the screen rejects, else ``""`` (pass)."""
    _, passed = check(seq, cdr3_max=cdr3_max)
    return "" if passed else REJECT_REASON
