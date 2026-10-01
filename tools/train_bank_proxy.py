#!/usr/bin/env python3
"""Train and honestly evaluate the label-bank acquisition proxy.

Evaluation deliberately uses two protocols, because a naive one is misleading:

* Random K-fold over molecules from a family-clustering search leaks near
  duplicates between train and test and inflated Spearman from 0.50 to 0.69 on
  our data. This tool therefore groups folds by role-A reagent.
* Rank correlation measured on a pool that was itself filtered by a ranker
  collapses (range restriction) and can go negative while the model is fine.
  Held-out-run figures are reported per source and must be read with the
  source's selection process in mind.

No credentials, no network. Reads candidate archives only.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from compound_miner.bank_proxy import BankProxy, chemistry_matrix, load_labels


def spearman(a, b) -> float:
    if len(a) < 15:
        return float("nan")
    a = pd.Series(a).rank().to_numpy(dtype=float)
    b = pd.Series(b).rank().to_numpy(dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    return round(float((a * b).sum() / denominator), 4) if denominator else 0.0


def top_decile_precision(truth, prediction) -> float:
    k = max(1, len(truth) // 10)
    best = set(np.argsort(-np.asarray(truth))[:k].tolist())
    picked = set(np.argsort(-np.asarray(prediction))[:k].tolist())
    return round(len(best & picked) / k, 4)


def novelty_mask(frame: pd.DataFrame, history: Path | None) -> np.ndarray:
    if history is None or not Path(history).is_file():
        return np.zeros(len(frame), dtype=bool)
    rows = pd.read_csv(history)
    rows = rows[rows["reaction"] == 3]
    known = [set(rows["component_a"]), set(rows["component_b"]), set(rows["component_c"])]
    return np.array([any(frame.iloc[i][f"component_{r}"] not in known[j]
                         for j, r in enumerate("abc")) for i in range(len(frame))])


def evaluate(frame, matrix, novel, folds, trees, jobs) -> dict:
    report = {}
    groups = frame["component_a"].to_numpy()
    target = frame["score"].to_numpy(dtype=float)
    splits = min(folds, len(np.unique(groups)))
    if splits >= 2:
        prediction = np.zeros(len(target))
        for train, test in GroupKFold(n_splits=splits).split(matrix, target, groups):
            model = BankProxy(trees=trees, jobs=jobs)
            model.fit(frame.iloc[train], chemistry=matrix[train])
            prediction[test] = model.model.predict(matrix[test])
        report["grouped_cv"] = dict(folds=int(splits), spearman=spearman(prediction, target),
            top_decile_precision=top_decile_precision(target, prediction),
            random_precision=round(max(1, len(target) // 10) / len(target), 4))
        if novel.any() and (~novel).any():
            report["grouped_cv"]["spearman_novel"] = spearman(prediction[novel], target[novel])
            report["grouped_cv"]["spearman_known"] = spearman(prediction[~novel], target[~novel])
    sources = sorted(set(frame["source"]))
    if len(sources) > 1:
        held = {}
        for source in sources:
            test = (frame["source"] == source).to_numpy()
            if test.sum() < 50 or (~test).sum() < 200:
                continue
            model = BankProxy(trees=trees, jobs=jobs)
            model.fit(frame.iloc[~test], chemistry=matrix[~test])
            prediction = model.model.predict(matrix[test])
            entry = dict(n_test=int(test.sum()), n_train=int((~test).sum()),
                spearman=spearman(prediction, target[test]),
                top_decile_precision=top_decile_precision(target[test], prediction),
                random_precision=round(max(1, int(test.sum()) // 10) / int(test.sum()), 4))
            if novel[test].any():
                entry["spearman_novel"] = spearman(prediction[novel[test]], target[test][novel[test]])
            held[source] = entry
        report["held_out_run"] = held
    return report


def learning_curve(frame, matrix, novel, trees, jobs, seed=68) -> list:
    groups = frame["component_a"].to_numpy()
    target = frame["score"].to_numpy(dtype=float)
    unique = np.unique(groups)
    if len(unique) < 4:
        return []
    rng = np.random.RandomState(seed)
    held_groups = set(rng.choice(unique, size=max(1, len(unique) // 4), replace=False).tolist())
    test = np.array([g in held_groups for g in groups])
    pool = np.where(~test)[0]
    rng.shuffle(pool)
    curve = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        take = pool[:max(50, int(len(pool) * fraction))]
        model = BankProxy(trees=trees, jobs=jobs)
        model.fit(frame.iloc[take], chemistry=matrix[take])
        prediction = model.model.predict(matrix[test])
        entry = dict(train_n=int(len(take)), spearman=spearman(prediction, target[test]),
                     top_decile_precision=top_decile_precision(target[test], prediction))
        if novel[test].any():
            entry["spearman_novel"] = spearman(prediction[novel[test]], target[test][novel[test]])
        curve.append(entry)
    return curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True,
                        help="candidates.sqlite to pool labels from (repeatable)")
    parser.add_argument("--history", default=None, help="training.csv, for novelty split")
    parser.add_argument("--output", default=None, help="write the fitted BankProxy here")
    parser.add_argument("--report", default=None, help="write the evaluation JSON here")
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--scorer", default=None, help="require this scorer label")
    parser.add_argument("--skip-eval", action="store_true")
    arguments = parser.parse_args()

    frame = load_labels(arguments.archive, scorer=arguments.scorer)
    if frame.empty:
        print(json.dumps(dict(error="no_labels")))
        return 1
    matrix = chemistry_matrix(frame["smiles"])
    novel = novelty_mask(frame, arguments.history)
    report = dict(labels=int(len(frame)),
                  sources={k: int(v) for k, v in frame["source"].value_counts().items()},
                  repeats=int((frame["repeats"] > 1).sum()),
                  novel_fraction=round(float(novel.mean()), 4),
                  feature_dim=int(matrix.shape[1]))
    if not arguments.skip_eval:
        report.update(evaluate(frame, matrix, novel, arguments.folds, arguments.trees, arguments.jobs))
        report["learning_curve"] = learning_curve(frame, matrix, novel, arguments.trees, arguments.jobs)
    model = BankProxy(trees=arguments.trees, jobs=arguments.jobs)
    model.fit(frame, scorer=arguments.scorer, chemistry=matrix)
    report["model"] = model.metadata()
    if arguments.output:
        model.save(arguments.output)
        report["saved"] = str(arguments.output)
    if arguments.report:
        Path(arguments.report).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
