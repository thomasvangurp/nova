from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr


@dataclasses.dataclass(frozen=True, slots=True)
class InteractionFeatureSet:
    names: np.ndarray
    smiles: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    base_scores: np.ndarray
    reactions: np.ndarray
    weights: np.ndarray


def load_interaction_features(
    path: str | Path,
    record_paths: Sequence[str | Path] = (),
) -> InteractionFeatureSet:
    """Load target-conditioned fingerprints and recover SMILES from source JSON if needed."""
    with np.load(path) as values:
        names = values["names"].astype(str)
        features = values["fingerprints"].astype(np.float32)
        labels = values["objective"].astype(np.float32)
        base_scores = values["psichic_score"].astype(np.float32)
        reactions = values["reactions"].astype(np.int16) if "reactions" in values else np.zeros(len(names), dtype=np.int16)
        weights = values["weights"].astype(np.float32) if "weights" in values else np.ones(len(names), dtype=np.float32)
        smiles = values["smiles"].astype(str) if "smiles" in values else None
    if smiles is None:
        lookup: dict[str, str] = {}
        for source in record_paths:
            for row in json.loads(Path(source).read_text())["scores"]:
                lookup[row["name"]] = row["smiles"]
        missing = [name for name in names if name not in lookup]
        if missing:
            raise ValueError(f"SMILES unavailable for {len(missing)} interaction rows")
        smiles = np.asarray([lookup[name] for name in names])
    if features.ndim != 2 or features.shape[0] != len(names):
        raise ValueError("interaction fingerprints must have shape [molecules, features]")
    return InteractionFeatureSet(names, smiles, features, labels, base_scores, reactions, weights)


def _scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return f"invalid:{smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    return scaffold or f"acyclic:{Chem.MolToSmiles(molecule)}"


def scaffold_indices(
    smiles: Sequence[str], validation_fraction: float = 0.2, seed: int = 68
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic whole-scaffold holdout; no analogue scaffold crosses the split."""
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(smiles):
        groups.setdefault(_scaffold(str(value)), []).append(index)
    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    wanted = max(1, round(len(smiles) * validation_fraction))
    validation: list[int] = []
    for key in keys:
        if len(validation) >= wanted:
            break
        validation.extend(groups[key])
    validation_array = np.asarray(sorted(validation), dtype=np.int64)
    mask = np.ones(len(smiles), dtype=bool)
    mask[validation_array] = False
    return np.flatnonzero(mask), validation_array


@dataclasses.dataclass(slots=True)
class RidgeInteractionEnsemble:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercepts: np.ndarray
    posterior_basis: np.ndarray
    posterior_factors: np.ndarray
    posterior_noise: float
    ridge: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        members: int = 16,
        ridge: float = 10.0,
        seed: int = 68,
        sample_weights: np.ndarray | None = None,
    ) -> "RidgeInteractionEnsemble":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        mean = features.mean(0)
        scale = features.std(0)
        scale[scale < 1e-6] = 1.0
        normalized = (features - mean) / scale
        rng = np.random.default_rng(seed)
        probabilities = None
        if sample_weights is not None:
            probabilities = np.asarray(sample_weights, dtype=np.float64).clip(min=0)
            probabilities /= probabilities.sum()
        weights = []
        intercepts = []
        for _ in range(members):
            selected = rng.choice(len(labels), len(labels), replace=True, p=probabilities)
            x = normalized[selected]
            y = labels[selected]
            intercept = float(y.mean())
            centered = y - intercept
            # The dual system is much cheaper when the labelled teacher set is
            # smaller than the 400-dimensional interaction fingerprint.
            dual = np.linalg.solve(x @ x.T + ridge * np.eye(len(x)), centered)
            weights.append(x.T @ dual)
            intercepts.append(intercept)
        # Bayesian-ridge leverage supplies an uncertainty term that grows in
        # feature directions unsupported by the teacher set. Bootstrap spread
        # alone is overconfident in those directions.
        full_intercept = float(labels.mean())
        full_dual = np.linalg.solve(
            normalized @ normalized.T + ridge * np.eye(len(normalized)),
            labels - full_intercept,
        )
        full_weights = normalized.T @ full_dual
        residual = labels - (normalized @ full_weights + full_intercept)
        _, singular_values, right = np.linalg.svd(normalized, full_matrices=False)
        singular_square = singular_values**2
        factors = singular_square / (ridge * (singular_square + ridge))
        return cls(
            mean.astype(np.float32),
            scale.astype(np.float32),
            np.asarray(weights, dtype=np.float32),
            np.asarray(intercepts, dtype=np.float32),
            right.astype(np.float32),
            factors.astype(np.float32),
            float(max(np.mean(residual**2), 1e-8)),
            float(ridge),
        )

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = (np.asarray(features, dtype=np.float32) - self.feature_mean) / self.feature_scale
        values = normalized @ self.weights.T + self.intercepts
        projection = normalized @ self.posterior_basis.T
        leverage = (
            np.sum(normalized**2, axis=1) / self.ridge
            - np.sum(projection**2 * self.posterior_factors, axis=1)
        ).clip(min=0)
        variance = values.var(1) + self.posterior_noise * leverage
        return values.mean(1), np.sqrt(variance)

    def predict_torch(
        self, features: torch.Tensor, device: str | torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = torch.device(device or features.device)
        values = features.to(selected, non_blocking=True)
        mean = torch.as_tensor(self.feature_mean, device=selected)
        scale = torch.as_tensor(self.feature_scale, device=selected)
        weights = torch.as_tensor(self.weights, device=selected)
        intercepts = torch.as_tensor(self.intercepts, device=selected)
        predictions = ((values - mean) / scale) @ weights.T + intercepts
        normalized = (values - mean) / scale
        basis = torch.as_tensor(self.posterior_basis, device=selected)
        factors = torch.as_tensor(self.posterior_factors, device=selected)
        projection = normalized @ basis.T
        leverage = (
            normalized.square().sum(1) / self.ridge
            - (projection.square() * factors).sum(1)
        ).clamp_min(0)
        variance = predictions.var(1, correction=0) + self.posterior_noise * leverage
        return predictions.mean(1), variance.sqrt()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            weights=self.weights,
            intercepts=self.intercepts,
            posterior_basis=self.posterior_basis,
            posterior_factors=self.posterior_factors,
            posterior_noise=np.asarray(self.posterior_noise),
            ridge=np.asarray(self.ridge),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RidgeInteractionEnsemble":
        with np.load(path) as values:
            return cls(
                *(values[name] for name in (
                    "feature_mean", "feature_scale", "weights", "intercepts",
                    "posterior_basis", "posterior_factors",
                )),
                float(values["posterior_noise"]),
                float(values["ridge"]),
            )


def _top_recall(labels: np.ndarray, predictions: np.ndarray, top_k: int, budget: int) -> float:
    top_k = min(top_k, len(labels))
    budget = min(max(top_k, budget), len(labels))
    truth = set(np.argsort(labels)[-top_k:])
    selected = set(np.argsort(predictions)[-budget:])
    return len(truth & selected) / top_k


def interaction_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    top_k: int = 20,
    budget: int | None = None,
) -> dict[str, float | int]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    uncertainty = np.asarray(uncertainty)
    budget = budget or min(len(labels), max(top_k, top_k * 2))
    correlation = spearmanr(labels, predictions).statistic if len(labels) > 1 else float("nan")
    error_correlation = (
        spearmanr(np.abs(labels - predictions), uncertainty).statistic
        if len(labels) > 2 and np.std(uncertainty) > 0
        else float("nan")
    )
    return {
        "count": len(labels),
        "mae": float(np.mean(np.abs(labels - predictions))),
        "spearman": float(correlation),
        "top_k": min(top_k, len(labels)),
        "boltz_budget": min(budget, len(labels)),
        "top_k_recall": _top_recall(labels, predictions, top_k, budget),
        "uncertainty_error_spearman": float(error_correlation),
    }


def benchmark_interaction_student(
    dataset: InteractionFeatureSet,
    validation_fraction: float = 0.2,
    members: int = 16,
    ridge: float = 10.0,
    top_k: int = 20,
    budget: int | None = None,
    device: str | None = None,
    validation_names: set[str] | None = None,
) -> tuple[RidgeInteractionEnsemble, dict[str, object]]:
    labelled = np.isfinite(dataset.labels)
    if labelled.sum() < 8:
        raise ValueError("at least eight labelled interaction fingerprints are required")
    if not labelled.all():
        dataset = InteractionFeatureSet(
            dataset.names[labelled], dataset.smiles[labelled], dataset.features[labelled],
            dataset.labels[labelled], dataset.base_scores[labelled], dataset.reactions[labelled],
            dataset.weights[labelled],
        )
    split_kind = "murcko_scaffold"
    if validation_names:
        validation = np.asarray(
            [index for index, name in enumerate(dataset.names) if str(name) in validation_names],
            dtype=np.int64,
        )
        if not len(validation):
            raise ValueError("fixed validation records do not overlap the feature set")
        validation_scaffolds = {_scaffold(str(dataset.smiles[index])) for index in validation}
        validation_set = set(int(value) for value in validation)
        train = np.asarray(
            [
                index for index, smiles in enumerate(dataset.smiles)
                if index not in validation_set and _scaffold(str(smiles)) not in validation_scaffolds
            ],
            dtype=np.int64,
        )
        split_kind = "fixed_exact_records_with_scaffold_embargo"
    else:
        train, validation = scaffold_indices(dataset.smiles, validation_fraction)
    if len(train) < 8:
        raise ValueError("fewer than eight training rows remain after the validation embargo")
    model = RidgeInteractionEnsemble.fit(
        dataset.features[train], dataset.labels[train], members, ridge,
        sample_weights=dataset.weights[train],
    )
    predictions, uncertainty = model.predict(dataset.features[validation])
    base = interaction_metrics(
        dataset.labels[validation], dataset.base_scores[validation], np.zeros(len(validation)), top_k, budget
    )
    adapted = interaction_metrics(dataset.labels[validation], predictions, uncertainty, top_k, budget)

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.from_numpy(dataset.features)
    # Measure warmed adapter throughput separately from PSICHIC feature extraction.
    model.predict_torch(tensor, selected_device)
    if selected_device.startswith("cuda"):
        torch.cuda.synchronize()
    repeats = max(1, min(1000, 100_000 // max(1, len(dataset.names))))
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict_torch(tensor, selected_device)
    if selected_device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics: dict[str, object] = {
        "split": {"train": len(train), "validation": len(validation), "kind": split_kind},
        "raw_psichic": base,
        "adapted": adapted,
        "adapter_device": selected_device,
        "adapter_molecules_per_second": len(dataset.names) * repeats / elapsed,
    }
    per_reaction = {}
    validation_reactions = dataset.reactions[validation]
    for reaction in sorted(set(int(value) for value in validation_reactions)):
        mask = validation_reactions == reaction
        if mask.sum() >= 4:
            per_reaction[str(reaction)] = interaction_metrics(
                dataset.labels[validation][mask], predictions[mask], uncertainty[mask],
                min(top_k, int(mask.sum())), min(budget or top_k * 2, int(mask.sum())),
            )
    if per_reaction:
        metrics["adapted_by_reaction"] = per_reaction
    return model, metrics
