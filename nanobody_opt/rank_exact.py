"""Rank exact nanobody observations against recent validator-era fields.

BoltzGen inference is stochastic, so a live finalist must survive more than one
exact observation.  This utility ranks each observation against deterministic
bootstrap fields and uses the candidate's worst observation median as its
selection score.  Lower is better.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt.simulate_round import METRICS, rank_sum_in_field


def _observations(paths: list[Path], target: str) -> dict[str, list[dict[str, float]]]:
    found: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        if str(payload.get("target")) != target:
            raise SystemExit(f"{path} targets {payload.get('target')!r}, not {target}")
        for by_sequence in payload.get("per_nanobody_components", {}).values():
            for sequence, by_target in by_sequence.items():
                metrics = by_target.get(target)
                if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
                    continue
                values = {name: float(metrics[name]) for name in METRICS}
                if all(math.isfinite(value) for value in values.values()):
                    found[str(sequence)].append(values)
    return found


def _excluded_sequences(paths: list[Path]) -> set[str]:
    """Read newline-delimited sequence ledgers, ignoring comments and blanks."""
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        excluded.update(
            line.strip().upper()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="append", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--minimum-observations", type=int, default=1)
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        type=Path,
        help="Newline-delimited sequences that must never be selected again",
    )
    parser.add_argument("--recent", type=int, default=300)
    parser.add_argument("--field-size", type=int, default=49)
    parser.add_argument("--bootstrap-fields", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    full_archive = pd.read_csv(args.archive)
    historical_sequences = set(
        full_archive.sequence.dropna().astype(str).str.strip().str.upper()
    )
    excluded_sequences = _excluded_sequences(args.exclude_file)
    archive = full_archive.tail(args.recent).reset_index(drop=True)
    if len(archive) < args.field_size:
        raise SystemExit("archive is smaller than the requested competitor field")
    missing = set(METRICS) - set(archive.columns)
    if missing:
        raise SystemExit(f"archive lacks metrics: {sorted(missing)}")

    rng = np.random.default_rng(args.random_seed)
    fields = [
        archive.iloc[rng.choice(len(archive), args.field_size, replace=False)]
        for _ in range(args.bootstrap_fields)
    ]
    ranked = []
    for sequence, observations in _observations(args.exact, args.target).items():
        normalized_sequence = sequence.strip().upper()
        if (
            normalized_sequence in historical_sequences
            or normalized_sequence in excluded_sequences
        ):
            continue
        if len(observations) < args.minimum_observations:
            continue
        per_observation = [
            float(np.median([rank_sum_in_field(metrics, field) for field in fields]))
            for metrics in observations
        ]
        aggregate = {
            name: float(np.median([item[name] for item in observations]))
            for name in METRICS
        }
        aggregate_field_scores = [
            rank_sum_in_field(aggregate, field) for field in fields
        ]
        ranked.append({
            "sequence": normalized_sequence,
            "observation_count": len(observations),
            "selection_score": max(per_observation),
            "per_observation_median_rank_sum": per_observation,
            "aggregate_median_rank_sum": float(np.median(aggregate_field_scores)),
            "aggregate_p90_rank_sum": float(np.quantile(aggregate_field_scores, 0.9)),
            "metrics": aggregate,
            "observations": observations,
        })

    ranked.sort(key=lambda item: (
        item["selection_score"], item["aggregate_p90_rank_sum"], item["sequence"]
    ))
    selected = ranked[: args.limit]
    if not selected:
        raise SystemExit("no complete exact candidates met the observation requirement")
    payload = {
        "target": args.target,
        "exact_files": [str(path) for path in args.exact],
        "minimum_observations": args.minimum_observations,
        "excluded_sequence_count": len(excluded_sequences),
        "candidate_count": len(ranked),
        "sequences": [item["sequence"] for item in selected],
        "candidates": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"ranked {len(ranked)} candidates; wrote top {len(selected)} to {args.output}")
    for item in selected[:10]:
        print(
            f"  score={item['selection_score']:.1f} "
            f"p90={item['aggregate_p90_rank_sum']:.1f} "
            f"n={item['observation_count']} {item['sequence']}"
        )


if __name__ == "__main__":
    main()
