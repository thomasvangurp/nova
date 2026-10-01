"""Persist the trained parent-relative ranker to out/train/model.joblib.

train_children_ranker.py fits ``final_model`` in-memory and writes only
results.json / holdout_predictions.csv / parent_group_cv.csv -- the estimator
itself is never dumped.  This script reproduces the exact final fit: same
holdout-parent split (seed 20260930, 20% of parents sealed), same selected spec
(read back from results.json), same feature_matrix / _fit / workers / seed, then
joblib.dump to out/train/model.joblib.  Deterministic given the frozen seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, "/Users/tvgdeena/nova-nanobody")
from nanobody_opt import train_parent_relative_ranker as tpr  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "children_parent_exact_dataset.csv.gz")
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "train" / "results.json")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "train" / "model.joblib")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--holdout-parent-fraction", type=float, default=0.20)
    args = ap.parse_args()

    import joblib

    selected = json.loads(args.results.read_text())["selected_model"]
    spec = tpr.ModelSpec(
        name=str(selected["name"]),
        family=str(selected["family"]),
        min_samples_leaf=int(selected["min_samples_leaf"]),
        max_features=float(selected["max_features"]),
    )

    frame = pd.read_csv(args.dataset)
    frame["sequence"] = frame.sequence.astype(str).str.strip().str.upper()
    frame["parent_sequence"] = frame.parent_sequence.astype(str).str.strip().str.upper()
    length = int(frame.sequence.str.len().mode().iloc[0])
    frame = frame[
        (frame.sequence.str.len() == length)
        & (frame.parent_sequence.str.len() == length)
    ].reset_index(drop=True)

    # Same holdout-parent split as train_children_ranker.py: shuffle parents with
    # the frozen seed, seal 20% as holdout, train on the rest.
    rng = np.random.default_rng(args.seed)
    parents = np.array(sorted(frame.parent_sequence.unique()))
    rng.shuffle(parents)
    holdout_parents = set(parents[: max(1, int(len(parents) * args.holdout_parent_fraction))])
    training = frame[~frame.parent_sequence.isin(holdout_parents)].reset_index(drop=True)

    final_model = tpr._fit(
        spec,
        tpr.feature_matrix(training, length=length),
        training.target_improvement.to_numpy(dtype=float),
        args.workers,
        args.seed,
    )
    joblib.dump(final_model, args.output)
    print(f"persisted {args.output} spec={spec.name} "
          f"estimators={len(final_model.estimators_)} "
          f"training_examples={len(training)} length={length}", flush=True)


if __name__ == "__main__":
    main()
