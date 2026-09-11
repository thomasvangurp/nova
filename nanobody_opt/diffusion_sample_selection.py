"""Validator-exact diffusion-sample selection metadata.

BoltzGen deliberately uses two argmax rules for one refolding run.  The
prediction writer selects the CIF with whole-complex confidence, while the
analysis task selects the confidence metrics (and its in-memory pose) with
design-to-target confidence.  Keeping both indices explicit prevents local
pose checks from silently inspecting the wrong diffusion sample.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


SCORER_PROTOCOL = "validator-dual-sample-v1"
EXPECTED_DIFFUSION_SAMPLES = 5
WRITER_SELECTION_RULE = "argmax(0.8*iptm+0.2*ptm)"
ANALYZE_SELECTION_RULE = (
    "argmax(0.8*design_to_target_iptm+0.2*design_ptm)"
)
ANALYZE_CONFIDENCE_METRICS = (
    "design_iiptm",
    "design_ptm",
    "design_to_target_iptm",
    "min_design_to_target_pae",
    "interaction_pae",
)


@dataclass(frozen=True)
class DiffusionSampleSelection:
    sample_count: int
    writer_sample_index: int
    analyze_sample_index: int
    sample_selection_disagrees: bool
    writer_selection_rule: str
    analyze_selection_rule: str
    writer_scores: tuple[float, ...]
    analyze_scores: tuple[float, ...]

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def require_expected_sample_count(
    record: Mapping[str, Any],
    *,
    expected: int = EXPECTED_DIFFUSION_SAMPLES,
    context: str = "observation",
) -> int:
    """Refuse an observation folded with fewer diffusion samples than upstream.

    The reported confidence metrics are a maximum over the diffusion samples, so
    a scorer that draws fewer samples does not merely add noise: it biases every
    confidence metric downward and every PAE upward.  A faster scorer can make
    that trade invisibly, so the sample count is validated rather than trusted.
    """
    if expected < 1:
        raise ValueError("expected diffusion-sample count must be positive")
    try:
        count = int(record["sample_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{context} lacks a diffusion sample count"
        ) from error
    if count != expected:
        raise ValueError(
            f"{context} used {count} diffusion samples, expected {expected}; "
            "the reported metrics are a maximum over samples, so this biases "
            "the observation rather than only adding noise"
        )
    return count


def _samples(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        raise ValueError(f"{name} has no diffusion samples")
    return array


def select_diffusion_samples(
    folded: Mapping[str, Any],
) -> DiffusionSampleSelection:
    """Apply the two rules used by the live validator, without averaging."""
    required = ("iptm", "ptm", "design_to_target_iptm", "design_ptm")
    missing = [name for name in required if name not in folded]
    if missing:
        raise ValueError(f"fold output lacks sample metrics: {missing}")
    values = {name: _samples(folded[name], name) for name in required}
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("fold sample metric arrays have different lengths")
    writer_scores = 0.8 * values["iptm"] + 0.2 * values["ptm"]
    analyze_scores = (
        0.8 * values["design_to_target_iptm"]
        + 0.2 * values["design_ptm"]
    )
    writer_index = int(np.argmax(writer_scores))
    analyze_index = int(np.argmax(analyze_scores))
    return DiffusionSampleSelection(
        sample_count=lengths.pop(),
        writer_sample_index=writer_index,
        analyze_sample_index=analyze_index,
        sample_selection_disagrees=writer_index != analyze_index,
        writer_selection_rule=WRITER_SELECTION_RULE,
        analyze_selection_rule=ANALYZE_SELECTION_RULE,
        writer_scores=tuple(float(value) for value in writer_scores),
        analyze_scores=tuple(float(value) for value in analyze_scores),
    )
