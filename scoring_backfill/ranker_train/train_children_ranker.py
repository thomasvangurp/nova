"""Train the parent-relative ranker on partial children data.

Reuses train_parent_relative_ranker's feature_matrix / model specs / allocation
protocol.  Chronological holdout is impossible (all children share epoch 25423),
so we substitute a novel-parent holdout: children of 20% of parents are sealed
before model selection and used exactly once as holdout.  Grouped parent-family
CV runs on the training 80%.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/tvgdeena/nova-nanobody")
from nanobody_opt import train_parent_relative_ranker as tpr  # noqa: E402
from nanobody_opt.simulate_round import METRICS  # noqa: E402

METRIC_ORDER = list(METRICS.keys())


def _spearman(a, b):
    a = pd.Series(np.asarray(a, dtype=float)).rank(method="average")
    b = pd.Series(np.asarray(b, dtype=float)).rank(method="average")
    v = a.corr(b, method="pearson")
    return float(v) if pd.notna(v) else 0.0


def topk_best_rank_sum(frame: pd.DataFrame, order: list[int], budgets) -> dict:
    best = float(frame.child_rank_sum.min())
    out = {}
    for b in budgets:
        k = min(b, len(frame))
        sel = order[:k]
        rs = float(frame.iloc[sel].child_rank_sum.min())
        out[f"best_rank_sum_at_{b}"] = rs
        out[f"regret_at_{b}"] = rs - best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--holdout-parent-fraction", type=float, default=0.20)
    args = ap.parse_args()

    frame = pd.read_csv(args.dataset)
    frame["sequence"] = frame.sequence.astype(str).str.strip().str.upper()
    frame["parent_sequence"] = frame.parent_sequence.astype(str).str.strip().str.upper()
    length = int(frame.sequence.str.len().mode().iloc[0])
    frame = frame[(frame.sequence.str.len() == length) & (frame.parent_sequence.str.len() == length)].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    parents = np.array(sorted(frame.parent_sequence.unique()))
    rng.shuffle(parents)
    n_holdout = max(1, int(len(parents) * args.holdout_parent_fraction))
    holdout_parents = set(parents[:n_holdout])
    train_mask = ~frame.parent_sequence.isin(holdout_parents)
    training = frame[train_mask].reset_index(drop=True)
    holdout = frame[~train_mask].reset_index(drop=True)
    print(f"length={length} training={len(training)} holdout={len(holdout)} "
          f"train_parents={training.parent_sequence.nunique()} holdout_parents={holdout.parent_sequence.nunique()}", flush=True)

    selected, cv_results, cv_summary = tpr.grouped_model_selection(
        training, length=length, workers=args.workers, seed=args.seed
    )
    final_model = tpr._fit(
        selected,
        tpr.feature_matrix(training, length=length),
        training.target_improvement.to_numpy(dtype=float),
        args.workers,
        args.seed,
    )
    holdout_scored = tpr.score_frame(final_model, holdout, length=length)
    alloc = tpr.allocation_metrics(
        holdout_scored,
        holdout_scored.predicted_improvement.to_numpy(),
        holdout_scored.probability_improves.to_numpy(),
    )

    # ---- explicit baselines on holdout ----
    # A: parent-score (rank children by parent_rank_sum ascending; within-family constant)
    parent_order = holdout_scored.sort_values(["parent_rank_sum", "child_rank_sum"]).index.tolist()
    parent_alloc = topk_best_rank_sum(holdout_scored, parent_order, (20, 40, 80, 160))

    # B: top-2 residue group-mean (train (position,residue) mean improvement -> holdout)
    training2 = training.copy()
    training2["edit_key"] = training2.edit_position.astype(str) + ":" + training2.edit_to
    group_mean = training2.groupby("edit_key").target_improvement.mean().to_dict()
    holdout_scored["edit_key"] = holdout_scored.edit_position.astype(str) + ":" + holdout_scored.edit_to
    holdout_scored["residue_prior"] = holdout_scored.edit_key.map(group_mean)
    residue_pred = holdout_scored.residue_prior.fillna(0.0)
    residue_order = holdout_scored.sort_values(["residue_prior", "parent_rank_sum"], ascending=[False, True]).index.tolist()
    residue_alloc = topk_best_rank_sum(holdout_scored, residue_order, (20, 40, 80, 160))

    results = {
        "selected_model": tpr.asdict(selected) if hasattr(tpr, "asdict") else {"name": selected.name, "family": selected.family, "min_samples_leaf": selected.min_samples_leaf, "max_features": selected.max_features},
        "data": {
            "length": length,
            "training_examples": len(training),
            "holdout_examples": len(holdout),
            "train_parents": int(training.parent_sequence.nunique()),
            "holdout_parents": int(holdout.parent_sequence.nunique()),
        },
        "internal_cv_summary": {k: (v if not isinstance(v, (dict, list)) else v) for k, v in cv_summary.items()},
        "holdout_allocation": alloc,
        "baseline_parent_score": {
            "spearman_parent_rank_sum_vs_child_rank_sum": _spearman(
                holdout_scored.parent_rank_sum, holdout_scored.child_rank_sum),
            **parent_alloc,
        },
        "baseline_top2_residue": {
            "spearman_residue_prior_vs_improvement": _spearman(
                residue_pred, holdout_scored.target_improvement),
            **residue_alloc,
        },
        "model_holdout": {
            "spearman_predicted_vs_improvement": _spearman(
                holdout_scored.predicted_improvement, holdout_scored.target_improvement),
            "spearman_predicted_vs_child_rank_sum": _spearman(
                holdout_scored.predicted_improvement, holdout_scored.child_rank_sum),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(args.output_dir / "parent_group_cv.csv", index=False)
    holdout_scored.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
