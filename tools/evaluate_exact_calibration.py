#!/usr/bin/env python3
"""Evaluate small-data exact Boltz calibration without scaffold leakage.

This deliberately distinguishes a reduced-Boltz upgrader (which receives the
matched reduced score at inference) from a diffusion-free student (which does
not). Every reported exact prediction is out of fold by Murcko scaffold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold

from compound_miner.features import ComponentEncoder, molecule_features


def _scores(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("scores", payload)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"no score rows in {path}")
    return rows


def _scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    return value or Chem.MolToSmiles(molecule, canonical=True)


def _linear_features(score: np.ndarray, reaction: np.ndarray) -> np.ndarray:
    reaction_bits = np.eye(5, dtype=np.float32)[np.clip(reaction - 1, 0, 4)]
    return np.column_stack((score, reaction_bits, reaction_bits * score[:, None]))


def _tree(seed: int, trees: int, minimum_leaf: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=minimum_leaf,
        max_features=0.55,
        n_jobs=-1,
        random_state=seed,
    )


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    value = spearmanr(truth, prediction).statistic
    return float(value) if np.isfinite(value) else 0.0


def _metrics(truth: np.ndarray, prediction: np.ndarray, top_k: int, budget: int) -> dict[str, float]:
    actual = set(np.argsort(truth)[-min(top_k, len(truth)) :])
    selected = set(np.argsort(prediction)[-min(budget, len(prediction)) :])
    predicted_top = np.asarray(sorted(selected), dtype=np.int64)
    return {
        "mae": float(np.mean(np.abs(truth - prediction))),
        "spearman": _spearman(truth, prediction),
        "top_k": float(min(top_k, len(truth))),
        "budget": float(min(budget, len(truth))),
        "top_k_recall": float(len(actual & selected) / max(1, len(actual))),
        "selected_oracle_mean": float(np.mean(truth[predicted_top])),
        "oracle_top_mean": float(np.mean(np.sort(truth)[-min(budget, len(truth)) :])),
    }


def evaluate(
    exact_path: Path,
    reduced_path: Path,
    repeats: int,
    trees: int,
    top_k: int,
    budget: int,
) -> dict[str, Any]:
    exact = _scores(exact_path)
    reduced = _scores(reduced_path)
    reduced_by_key = {str(row.get("inchikey") or row["smiles"]): row for row in reduced}
    exact_keys = [str(row.get("inchikey") or row["smiles"]) for row in exact]
    missing = [key for key in exact_keys if key not in reduced_by_key]
    if missing:
        raise ValueError(f"{len(missing)} exact rows have no paired reduced score")

    truth = np.asarray([row["score"] for row in exact], dtype=np.float64)
    reduced_truth = np.asarray([reduced_by_key[key]["score"] for key in exact_keys], dtype=np.float64)
    reactions = np.asarray([row["reaction"] for row in exact], dtype=np.int64)
    components = np.full((len(exact), 3), -1, dtype=np.int64)
    for index, row in enumerate(exact):
        values = row.get("components", [])
        components[index, : min(3, len(values))] = values[:3]
    scaffolds = np.asarray([_scaffold(row["smiles"]) for row in exact])
    chemistry = np.vstack([molecule_features(row["smiles"], fingerprint_size=2048) for row in exact])
    fingerprint = chemistry[:, :2048]
    intersections = fingerprint @ fingerprint.T
    bit_counts = fingerprint.sum(axis=1)
    tanimoto = intersections / (
        bit_counts[:, None] + bit_counts[None, :] - intersections + 1e-9
    )
    reaction_bits = np.eye(5, dtype=np.float32)[np.clip(reactions - 1, 0, 4)]
    chemistry = np.column_stack((chemistry, reaction_bits))

    # The diffusion-free base is denied all 65 paired exact molecules. This is
    # stricter than merely hiding their exact labels and prevents identity
    # leakage through their reduced labels.
    exact_key_set = set(exact_keys)
    public_reduced = [
        row for row in reduced if str(row.get("inchikey") or row["smiles"]) not in exact_key_set
    ]
    predicted_reduced: np.ndarray | None = None
    if public_reduced:
        base_x = np.vstack(
            [
                np.concatenate(
                    (
                        molecule_features(row["smiles"], fingerprint_size=2048),
                        np.eye(5, dtype=np.float32)[int(row["reaction"]) - 1],
                    )
                )
                for row in public_reduced
            ]
        )
        base_y = np.asarray([row["score"] for row in public_reduced], dtype=np.float64)
        base = _tree(seed=771, trees=max(512, trees), minimum_leaf=2).fit(base_x, base_y)
        predicted_reduced = base.predict(chemistry)

    names = [
        "raw_reduced",
        "affine_reduced",
        "exact_morgan",
        "exact_component_morgan",
        "exact_morgan_by_reaction",
        "exact_tanimoto_kernel",
        "exact_tanimoto_kernel_by_reaction",
        "reduced_residual_morgan",
        "reduced_residual_component_morgan",
        "reduced_residual_morgan_by_reaction",
        "reduced_residual_tanimoto_kernel",
        "reduced_residual_tanimoto_kernel_by_reaction",
    ]
    if predicted_reduced is not None:
        names.extend(("diffusion_free_reduced", "diffusion_free_residual_morgan"))
    per_repeat: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    prediction_sums = {name: np.zeros(len(exact), dtype=np.float64) for name in names}

    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=91_003 + repeat)
        fold_predictions = {name: np.empty(len(exact), dtype=np.float64) for name in names}
        for fold, (train, validation) in enumerate(splitter.split(chemistry, reactions, scaffolds)):
            seed = 17_003 + repeat * 101 + fold
            fold_predictions["raw_reduced"][validation] = reduced_truth[validation]

            calibrator = Ridge(alpha=1e-3).fit(
                _linear_features(reduced_truth[train], reactions[train]), truth[train]
            )
            fold_predictions["affine_reduced"][validation] = calibrator.predict(
                _linear_features(reduced_truth[validation], reactions[validation])
            )

            exact_model = _tree(seed, trees, minimum_leaf=3).fit(chemistry[train], truth[train])
            fold_predictions["exact_morgan"][validation] = exact_model.predict(chemistry[validation])

            def encoded_matrix(indices: np.ndarray, encoder: ComponentEncoder) -> np.ndarray:
                return np.column_stack(
                    (
                        chemistry[indices],
                        encoder.transform_rows(reactions[indices], components[indices]),
                    )
                )

            component_frame = pd.DataFrame(
                {
                    "reaction": reactions[train],
                    "component_a": components[train, 0],
                    "component_b": components[train, 1],
                    "component_c": components[train, 2],
                    "score": truth[train],
                }
            )
            component_encoder = ComponentEncoder().fit(component_frame)
            component_model = _tree(seed + 30, trees, minimum_leaf=3).fit(
                encoded_matrix(train, component_encoder), truth[train]
            )
            fold_predictions["exact_component_morgan"][validation] = component_model.predict(
                encoded_matrix(validation, component_encoder)
            )

            # Fixed before expanding the exact set. The small reaction term is
            # additive rather than a hard partition, so sparse reactions can
            # still borrow chemical-neighbour evidence from one another.
            exact_kernel = tanimoto + 0.25 * (reactions[:, None] == reactions[None, :])
            exact_mean = float(np.mean(truth[train]))
            kernel_model = KernelRidge(alpha=1.0, kernel="precomputed").fit(
                exact_kernel[np.ix_(train, train)], truth[train] - exact_mean
            )
            fold_predictions["exact_tanimoto_kernel"][validation] = (
                kernel_model.predict(exact_kernel[np.ix_(validation, train)]) + exact_mean
            )

            correction = _tree(seed + 1, trees, minimum_leaf=4).fit(
                chemistry[train], truth[train] - reduced_truth[train]
            )
            fold_predictions["reduced_residual_morgan"][validation] = (
                reduced_truth[validation] + correction.predict(chemistry[validation])
            )
            residual_frame = component_frame.copy()
            residual_frame["score"] = truth[train] - reduced_truth[train]
            residual_encoder = ComponentEncoder().fit(residual_frame)
            residual_component_model = _tree(seed + 31, trees, minimum_leaf=3).fit(
                encoded_matrix(train, residual_encoder), residual_frame["score"].to_numpy()
            )
            fold_predictions["reduced_residual_component_morgan"][validation] = (
                reduced_truth[validation]
                + residual_component_model.predict(encoded_matrix(validation, residual_encoder))
            )

            residual = truth[train] - reduced_truth[train]
            residual_mean = float(np.mean(residual))
            residual_model = KernelRidge(alpha=1.0, kernel="precomputed").fit(
                tanimoto[np.ix_(train, train)], residual - residual_mean
            )
            fold_predictions["reduced_residual_tanimoto_kernel"][validation] = (
                reduced_truth[validation]
                + residual_model.predict(tanimoto[np.ix_(validation, train)])
                + residual_mean
            )

            # Live rounds are reaction-specific. Freeze separate low-data
            # estimators now so the larger prospective set can test whether
            # 64 exact labels per reaction are enough to learn local ranking.
            for reaction in range(1, 6):
                reaction_train = train[reactions[train] == reaction]
                reaction_validation = validation[reactions[validation] == reaction]
                if not len(reaction_validation):
                    continue
                reaction_tree = _tree(seed + 10 + reaction, trees, minimum_leaf=2).fit(
                    chemistry[reaction_train], truth[reaction_train]
                )
                fold_predictions["exact_morgan_by_reaction"][reaction_validation] = (
                    reaction_tree.predict(chemistry[reaction_validation])
                )
                reaction_mean = float(np.mean(truth[reaction_train]))
                reaction_kernel = KernelRidge(alpha=1.0, kernel="precomputed").fit(
                    tanimoto[np.ix_(reaction_train, reaction_train)],
                    truth[reaction_train] - reaction_mean,
                )
                fold_predictions["exact_tanimoto_kernel_by_reaction"][reaction_validation] = (
                    reaction_kernel.predict(tanimoto[np.ix_(reaction_validation, reaction_train)])
                    + reaction_mean
                )

                reaction_residual = truth[reaction_train] - reduced_truth[reaction_train]
                reaction_residual_tree = _tree(
                    seed + 20 + reaction, trees, minimum_leaf=2
                ).fit(chemistry[reaction_train], reaction_residual)
                fold_predictions["reduced_residual_morgan_by_reaction"][reaction_validation] = (
                    reduced_truth[reaction_validation]
                    + reaction_residual_tree.predict(chemistry[reaction_validation])
                )
                reaction_residual_mean = float(np.mean(reaction_residual))
                reaction_residual_kernel = KernelRidge(alpha=1.0, kernel="precomputed").fit(
                    tanimoto[np.ix_(reaction_train, reaction_train)],
                    reaction_residual - reaction_residual_mean,
                )
                fold_predictions[
                    "reduced_residual_tanimoto_kernel_by_reaction"
                ][reaction_validation] = (
                    reduced_truth[reaction_validation]
                    + reaction_residual_kernel.predict(
                        tanimoto[np.ix_(reaction_validation, reaction_train)]
                    )
                    + reaction_residual_mean
                )

            if predicted_reduced is not None:
                student_calibrator = Ridge(alpha=1e-3).fit(
                    _linear_features(predicted_reduced[train], reactions[train]), truth[train]
                )
                fold_predictions["diffusion_free_reduced"][validation] = student_calibrator.predict(
                    _linear_features(predicted_reduced[validation], reactions[validation])
                )

                student_correction = _tree(seed + 2, trees, minimum_leaf=4).fit(
                    chemistry[train], truth[train] - predicted_reduced[train]
                )
                fold_predictions["diffusion_free_residual_morgan"][validation] = (
                    predicted_reduced[validation] + student_correction.predict(chemistry[validation])
                )

        for name in names:
            metrics = _metrics(truth, fold_predictions[name], top_k, budget)
            per_repeat[name].append(metrics)
            prediction_sums[name] += fold_predictions[name]

    models: dict[str, Any] = {}
    for name in names:
        values = per_repeat[name]
        averaged = prediction_sums[name] / repeats
        by_reaction = {
            str(reaction): _metrics(
                truth[reactions == reaction],
                averaged[reactions == reaction],
                top_k,
                budget,
            )
            for reaction in range(1, 6)
        }
        models[name] = {
            "mean": {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            },
            "standard_deviation": {
                key: float(np.std([row[key] for row in values], ddof=1)) if repeats > 1 else 0.0
                for key in values[0]
            },
            "averaged_oof": _metrics(truth, averaged, top_k, budget),
            "metrics_by_reaction": by_reaction,
            "macro_within_reaction_spearman": float(
                np.mean([metrics["spearman"] for metrics in by_reaction.values()])
            ),
            "macro_within_reaction_top_k_recall": float(
                np.mean([metrics["top_k_recall"] for metrics in by_reaction.values()])
            ),
        }

    return {
        "version": 1,
        "exact_rows": len(exact),
        "paired_reduced_rows": len(exact),
        "diffusion_free_reduced_train_rows": len(public_reduced),
        "unique_scaffolds": len(set(scaffolds)),
        "split": f"{repeats}x repeated stratified five-fold Murcko-scaffold",
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--reduced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--budget", type=int, default=20)
    args = parser.parse_args()
    result = evaluate(args.exact, args.reduced, args.repeats, args.trees, args.top_k, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
