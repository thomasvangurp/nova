"""Compare repeat allocations on anonymized stored folds, without GPU work.

The production planner and second-chance selector are used directly. Twenty
one folds are available to the tournament; all remaining folds evaluate its
selected thirteen candidates. Fold subsets do not overlap within a trial.
This is a conditional replay check, not proof about unseen production pools.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nanobody_opt_v2 import confirmation_policy as policy
from nanobody_opt_v2.compute_budget import TimingModel

CONTROL = dict(candidates=64, rounds=20, rounds_per_stage=5,
               keep_fraction=.65, minimum_keep=20)


def benchmark(folds, *, trials=1000, seed=993731):
    matrix = np.asarray(folds, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] <= 21 or not np.isfinite(matrix).all():
        raise ValueError("independent evaluation folds required")
    plan = policy.plan(**CONTROL, gpus=8,
                       model=TimingModel(9.43, 5.52, 0))
    rng = np.random.default_rng(seed)
    results = []
    for trial in range(trials):
        shuffled = np.take_along_axis(matrix, np.argsort(rng.random(matrix.shape), axis=1), axis=1)
        cumulative = shuffled[:, :21].cumsum(axis=1)
        heldout = shuffled[:, 21:].mean(axis=1)
        order = np.argsort(shuffled[:, 0])
        values = []
        for schedule, broadened in ((plan["control_schedule"], False), (plan["schedule"], True)):
            selected, _ = policy.shortlist(
                [str(i) for i in order], width=schedule[0][0], metadata={},
                observations={}, epoch=trial, broadened=broadened)
            alive = np.asarray(selected, dtype=int)
            depth = 1
            for index, (width, rounds) in enumerate(schedule):
                if index:
                    alive = alive[np.argsort(cumulative[alive, depth - 1])[:width]]
                depth += rounds
            selected = alive[np.argsort(cumulative[alive, depth - 1])[:13]]
            values.append(float(heldout[selected].mean()))
        results.append(values)
    scores = np.asarray(results)
    delta = scores[:, 1] - scores[:, 0]
    return {"policy": plan["policy"], "candidates": len(matrix), "trials": trials,
            "training_folds": 21, "heldout_folds": matrix.shape[1] - 21,
            "control_mean": float(scores[:, 0].mean()), "broad_mean": float(scores[:, 1].mean()),
            "paired_delta": float(delta.mean()),
            "monte_carlo_standard_error": float(delta.std(ddof=1) / np.sqrt(trials)),
            "control_repeat_evaluations": plan["evaluation_cap"],
            "broad_repeat_evaluations": plan["planned_evaluations"],
            "within_time_cap": plan["planned_seconds"] <= plan["seconds_cap"] + 1e-7,
            "passed": bool(delta.mean() < 0 and plan["planned_evaluations"] <= plan["evaluation_cap"]
                           and plan["planned_seconds"] <= plan["seconds_cap"] + 1e-7)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/confirmation_replay"
    results = {p.stem: benchmark(json.loads(p.read_text())["folds"])
               for p in sorted(fixtures.glob("*.json"))}
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    return 0 if len(results) == 2 and all(r["passed"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
