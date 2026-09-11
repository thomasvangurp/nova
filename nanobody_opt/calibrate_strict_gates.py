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
from nanobody_opt.simulate_round import METRIC_CATEGORIES, METRICS

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


def _compiled_field(field: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: np.sort(field[name].to_numpy(float)) for name in METRIC_NAMES}


def _dense_rank(sorted_values: np.ndarray, value: float, direction: str) -> int:
    """Dense rank of one value against a field, matching the validator."""
    unique = np.unique(np.append(sorted_values, value))
    if direction == "min":
        return int(np.searchsorted(unique, value)) + 1
    return int(len(unique) - np.searchsorted(unique, value))


def score_distributions(
    observations: pd.DataFrame,
    fields: dict[int, pd.DataFrame],
    *,
    reference_epochs: int = 5,
) -> dict[str, np.ndarray]:
    """Rank sums of every observation, by category, against common fields.

    Both cohorts are scored against the *same* reference fields.  Scoring each
    against the field of its own epoch would fold differences between epochs'
    competitors into what is supposed to be a comparison of two scorers, and a
    replacement cohort re-scored offline has no epoch of its own at all.  Each
    observation's score is the median across the reference fields, so one
    unusually soft or hard epoch cannot set the thresholds.
    """
    chosen = [fields[epoch] for epoch in sorted(fields)[-reference_epochs:]]
    if not chosen:
        raise ValueError("no competitor fields available to score against")
    compiled = [_compiled_field(field) for field in chosen]
    buckets: dict[str, list[float]] = {
        "total": [], "confidence": [], "physical_interaction": [], "developability": []
    }
    for _, row in observations.iterrows():
        values = {name: float(row[name]) for name in METRIC_NAMES}
        per_field: dict[str, list[float]] = {key: [] for key in buckets}
        for field in compiled:
            ranks = {
                name: _dense_rank(field[name], values[name], direction)
                for name, direction in METRICS.items()
            }
            per_field["total"].append(float(sum(ranks.values())))
            for category, names in METRIC_CATEGORIES.items():
                per_field[category].append(float(sum(ranks[name] for name in names)))
        for key, scores in per_field.items():
            buckets[key].append(float(np.median(scores)))
    return {key: np.array(values) for key, values in buckets.items()}


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
    minimum_tail_observations: int = 20,
) -> dict[str, Any]:
    """Map each scale-dependent gate onto the replacement scorer's distribution.

    A gate that passes only a small tail of observations cannot be placed from a
    small cohort: at a 1% pass rate, 200 observations put the threshold on the
    second order statistic, where it moves tens of rank points on noise alone.
    Such a gate keeps its production value, which errs toward the selectivity we
    already understand instead of inventing one.  ``minimum_tail_observations``
    is how many replacement observations must fall inside the gate for its
    quantile to be trusted.
    """
    derived: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    for gate, (category, current) in (gates or SCALE_DEPENDENT_GATES).items():
        source = production.get(category)
        target = replacement.get(category)
        if source is None or target is None or not len(source) or not len(target):
            continue
        selectivity = quantile_of(source, current)
        supporting = selectivity * len(target)
        record = {
            "production_threshold": current,
            "production_pass_rate": round(selectivity, 4),
            "supporting_observations": round(supporting, 1),
        }
        if selectivity <= 0 or supporting < minimum_tail_observations:
            record.update({
                "applied": False,
                "reason": (
                    f"only {supporting:.1f} replacement observations fall inside "
                    f"this gate; need {minimum_tail_observations} to place it"
                ),
            })
            provenance[gate] = record
            continue
        value = float(np.quantile(target, selectivity))
        derived[gate] = round(value, 3)
        record.update({
            "applied": True,
            "replacement_threshold": round(value, 3),
            "shift": round(value - current, 3),
            "replacement_pass_rate": round(quantile_of(target, value), 4),
        })
        provenance[gate] = record
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
        "--reference-epochs",
        type=int,
        default=5,
        help="how many recent competitor fields both cohorts are scored against",
    )
    parser.add_argument(
        "--minimum-observations",
        type=int,
        default=200,
        help="refuse to calibrate from a cohort too small to place a quantile",
    )
    parser.add_argument(
        "--minimum-tail-observations",
        type=int,
        default=20,
        help="replacement observations that must fall inside a gate to move it",
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
        score_distributions(
            production, fields, reference_epochs=args.reference_epochs
        ),
        score_distributions(
            replacement, fields, reference_epochs=args.reference_epochs
        ),
        minimum_tail_observations=args.minimum_tail_observations,
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
        if not record.get("applied"):
            print(f"{gate}: kept at {record['production_threshold']} -- "
                  f"{record['reason']}", flush=True)
            continue
        print(
            f"{gate}: {record['production_threshold']} -> "
            f"{record['replacement_threshold']} "
            f"(shift {record['shift']:+}, pass rate "
            f"{record['production_pass_rate']} -> {record['replacement_pass_rate']}, "
            f"{record['supporting_observations']} observations)",
            flush=True,
        )


if __name__ == "__main__":
    main()
