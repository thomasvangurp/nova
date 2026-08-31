#!/usr/bin/env python3
"""Fit and benchmark a Boltz adapter over frozen PSICHIC interaction fingerprints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compound_miner.interaction_student import benchmark_interaction_student, load_interaction_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--records-json", type=Path, action="append", default=[])
    parser.add_argument("--validation-records-json", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    dataset = load_interaction_features(args.features, args.records_json)
    validation_names = set()
    for path in args.validation_records_json:
        validation_names.update(row["name"] for row in json.loads(path.read_text())["scores"])
    model, metrics = benchmark_interaction_student(
        dataset,
        args.validation_fraction,
        args.members,
        args.ridge,
        args.top_k,
        args.budget,
        args.device,
        validation_names or None,
    )
    model.save(args.output)
    metrics["model"] = str(args.output.resolve())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
