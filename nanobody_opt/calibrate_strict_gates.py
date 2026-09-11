"""Re-derive strict qualification thresholds for a replacement scorer.

The strict gates are absolute thresholds on field-relative rank sums, chosen
against observations from the production scorer.  A replacement scorer with even
a small systematic offset shifts the whole distribution those thresholds sit in,
so inheriting the numbers silently changes how selective qualification is --
tighter or looser, in a direction nobody chose.

The fix is to preserve the *selectivity* rather than the number.  For each
scale-dependent gate this module measures the quantile the production threshold
occupies in the production observation distribution, then reads the value at
that same quantile of the replacement scorer's distribution.  A scorer that is
genuinely equivalent reproduces the original thresholds; one that is offset
gets thresholds shifted by exactly its offset.

Gates that do not depend on the scorer's numeric scale -- observation counts --
are left alone.  Pose-geometry gates (contact jaccard, CDR3 RMSD) are reported
but not re-derived: they need a pose comparison, not a distribution shift, and
guessing at them would be worse than leaving them explicit.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nanobody_opt.backtest_budget_allocation import (
    METRIC_NAMES,
    collect_observations,
    load_fields,
)
from nanobody_opt.simulate_round import category_rank_sums, rank_sum_in_field

# Gates whose value is a rank sum on the scorer's own numeric scale.
SCALE_DEPENDENT_GATES = {
    "maximum_p90_rank_sum": ("total", 90.0),
    "maximum_p90_confidence_rank_sum": ("confidence", 70.0),
    "maximum_p90_physical_interaction_rank_sum": ("physical_interaction", 40.0),
    "maximum_p90_developability_rank_sum": ("developability", 15.0),
    "competitive_score_threshold": ("total", 75.0),
}
POSE_GATES = ("minimum_median_contact_jaccard", "maximum_cdr3_rmsd")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def score_distributions(
    observations: pd.DataFrame, fields: dict[int, pd.DataFrame]
) -> dict[str, np.ndarray]:
    """Rank sums of every observation, by category, against its epoch's field."""
    totals: list[float] = []
    categories: dict[str, list[float]] = {
        "confidence": [], "physical_interaction": [], "developability": []
    }
    for epoch, group in observations.groupby("epoch"):
        field = fields.get(int(epoch))
        if field is None:
            continue
        for _, row in group.iterrows():
            metrics = {name: float(row[name]) for name in METRIC_NAMES}
            totals.append(rank_sum_in_field(metrics, field))
            for category, value in category_rank_sums(metrics, field).items():
                categories[category].append(value)
    result = {"total": np.array(totals)}
    result.update({name: np.array(values) for name, values in categories.items()})
    return result


def quantile_of(values: np.ndarray, threshold: float) -> float:
    """Fraction of observations at or below ``threshold``; the gate's selectivity."""
    if not len(values):
        raise ValueError("no observations to calibrate against")
    return float((values <= threshold).mean())


def calibrate(
    production: dict[str, np.ndarray],
    replacement: dict[str, np.ndarray],
    *,
    gates: dict[str, tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Map each scale-dependent gate onto the replacement scorer's distribution."""
    derived: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    for gate, (category, current) in (gates or SCALE_DEPENDENT_GATES).items():
        source = production.get(category)
        target = replacement.get(category)
        if source is None or target is None or not len(source) or not len(target):
            continue
        selectivity = quantile_of(source, current)
        value = float(np.quantile(target, selectivity)) if selectivity > 0 else current
        derived[gate] = round(value, 3)
        provenance[gate] = {
            "production_threshold": current,
            "production_pass_rate": round(selectivity, 4),
            "replacement_threshold": round(value, 3),
            "shift": round(value - current, 3),
            "replacement_pass_rate": round(quantile_of(target, value), 4),
        }
    return {"gates": derived, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        type=Path,
        required=True,
        help="live tree of observations from the current production scorer",
    )
    parser.add_argument(
        "--replacement-root",
        type=Path,
        required=True,
        help="tree of observations from the replacement scorer",
    )
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--exclude-uid", type=int, default=8)
    parser.add_argument(
        "--minimum-observations",
        type=int,
        default=200,
        help="refuse to calibrate from a cohort too small to place a quantile",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fields = load_fields(args.field_dir, args.target, exclude_uid=args.exclude_uid)
    if not fields:
        parser.error("no cached competitor fields found")
    production = collect_observations(args.production_root)
    replacement = collect_observations(args.replacement_root)
    for name, frame in (("production", production), ("replacement", replacement)):
        if len(frame) < args.minimum_observations:
            parser.error(
                f"{name} cohort has {len(frame)} observations, "
                f"need at least {args.minimum_observations}"
            )

    payload = calibrate(
        score_distributions(production, fields),
        score_distributions(replacement, fields),
    )
    payload["observation_counts"] = {
        "production": int(len(production)),
        "replacement": int(len(replacement)),
    }
    payload["not_recalibrated"] = {
        gate: "needs a pose comparison, not a distribution shift"
        for gate in POSE_GATES
    }
    _atomic_json(args.output, payload)
    for gate, record in sorted(payload["provenance"].items()):
        print(
            f"{gate}: {record['production_threshold']} -> "
            f"{record['replacement_threshold']} "
            f"(shift {record['shift']:+}, pass rate "
            f"{record['production_pass_rate']} -> {record['replacement_pass_rate']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
