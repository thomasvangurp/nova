#!/usr/bin/env python3
"""Compare fixed Boltz batches and separate startup from steady-state cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import pearsonr, spearmanr


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    small = load(args.small)
    large = load(args.large)
    delta_count = int(large["count"]) - int(small["count"])
    delta_time = float(large["wall_seconds"]) - float(small["wall_seconds"])
    if delta_count <= 0 or delta_time <= 0:
        raise SystemExit("large run must contain more molecules and take longer")
    steady_rate = delta_count / delta_time
    startup = float(small["wall_seconds"]) - int(small["count"]) / steady_rate
    result = {
        "small_count": int(small["count"]),
        "large_count": int(large["count"]),
        "fitted_startup_seconds": startup,
        "steady_state_molecules_per_second": steady_rate,
        "steady_state_seconds_per_molecule": 1.0 / steady_rate,
    }

    if args.reference:
        reference = {item["name"]: float(item["score"]) for item in load(args.reference)["scores"]}
        observed = {item["name"]: float(item["score"]) for item in large["scores"]}
        names = sorted(reference.keys() & observed.keys())
        expected_values = [reference[name] for name in names]
        observed_values = [observed[name] for name in names]
        expected_top = set(sorted(names, key=reference.get, reverse=True)[: args.top_k])
        observed_top = set(sorted(names, key=observed.get, reverse=True)[: args.top_k])
        result.update(
            {
                "paired_count": len(names),
                "spearman": float(spearmanr(expected_values, observed_values).statistic),
                "pearson": float(pearsonr(expected_values, observed_values).statistic),
                "mean_absolute_error": sum(
                    abs(left - right) for left, right in zip(expected_values, observed_values)
                )
                / len(names),
                "top_k": args.top_k,
                "top_k_overlap": len(expected_top & observed_top),
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
