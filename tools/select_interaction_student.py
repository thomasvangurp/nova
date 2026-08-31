#!/usr/bin/env python3
"""Select promising and uncertain PSICHIC-fingerprint candidates for Boltz."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compound_miner.interaction_student import RidgeInteractionEnsemble, load_interaction_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--records-json", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--maximum", type=int, default=2048)
    parser.add_argument("--uncertainty-fraction", type=float, default=0.25)
    args = parser.parse_args()
    dataset = load_interaction_features(args.features, args.records_json)
    means, uncertainties = RidgeInteractionEnsemble.load(args.model).predict(dataset.features)
    budget = min(len(dataset.names), args.maximum, max(args.minimum, math.ceil(len(dataset.names) * args.fraction)))
    uncertainty_count = min(budget, round(budget * args.uncertainty_fraction))
    mean_count = budget - uncertainty_count
    selected = list(np.argsort(means)[-mean_count:][::-1]) if mean_count else []
    used = set(selected)
    for index in np.argsort(uncertainties)[::-1]:
        if int(index) not in used:
            selected.append(int(index))
            used.add(int(index))
        if len(selected) == budget:
            break
    payload = {
        "count": len(selected),
        "pool_count": len(dataset.names),
        "fraction": len(selected) / max(1, len(dataset.names)),
        "uncertainty_quota": uncertainty_count,
        "candidates": [
            {
                "name": str(dataset.names[index]),
                "smiles": str(dataset.smiles[index]),
                "predicted_mean": float(means[index]),
                "predicted_std": float(uncertainties[index]),
            }
            for index in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
