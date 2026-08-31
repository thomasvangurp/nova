from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

from .features import ComponentEncoder, frame_components, molecule_features
from .types import Candidate


def _matrix(
    frame: pd.DataFrame,
    encoder: ComponentEncoder,
    chemistry: np.ndarray | None = None,
) -> np.ndarray:
    if chemistry is None:
        chemistry = np.vstack([molecule_features(value) for value in frame["smiles"]])
    elif len(chemistry) != len(frame):
        raise ValueError("chemistry feature rows must match the candidate frame")
    reactions, components = frame_components(frame)
    reaction_bits = np.eye(5, dtype=np.float32)[np.clip(reactions - 1, 0, 4)]
    return np.hstack((chemistry, reaction_bits, encoder.transform_rows(reactions, components)))


class TargetProxy:
    def __init__(self, trees: int = 256, jobs: int = -1, random_state: int = 68) -> None:
        self.encoder = ComponentEncoder()
        self.model = ExtraTreesRegressor(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features=0.55,
            n_jobs=jobs,
            random_state=random_state,
        )

    def fit(self, frame: pd.DataFrame) -> "TargetProxy":
        self.encoder.fit(frame)
        self.model.fit(_matrix(frame, self.encoder), frame["score"].to_numpy(dtype=np.float32))
        return self

    def predict(
        self,
        candidates: Iterable[Candidate],
        chemistry: np.ndarray | None = None,
    ) -> list[Candidate]:
        values = list(candidates)
        if not values:
            return []
        frame = pd.DataFrame(
            {
                "smiles": [item.smiles for item in values],
                "reaction": [item.reaction for item in values],
                "component_a": [item.components[0] for item in values],
                "component_b": [item.components[1] for item in values],
                "component_c": [item.components[2] if len(item.components) > 2 else -1 for item in values],
            }
        )
        matrix = _matrix(frame, self.encoder, chemistry=chemistry)
        tree_predictions = np.vstack(
            Parallel(n_jobs=self.model.n_jobs, prefer="threads")(
                delayed(tree.predict)(matrix) for tree in self.model.estimators_
            )
        )
        means = np.mean(tree_predictions, axis=0)
        standard_deviations = np.std(tree_predictions, axis=0)
        return [dataclasses.replace(item, proxy_mean=float(mean), proxy_std=float(std)) for item, mean, std in zip(values, means, standard_deviations)]

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary, compress=3)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "TargetProxy":
        return joblib.load(path)


def chronological_evaluation(frame: pd.DataFrame, holdout_epochs: int = 8, top_k: int = 100, trees: int = 96) -> dict[str, float]:
    epochs = sorted(frame["epoch"].unique())
    held_out = set(epochs[-holdout_epochs:])
    train = frame[~frame["epoch"].isin(held_out)]
    test = frame[frame["epoch"].isin(held_out)]
    model = TargetProxy(trees=trees).fit(train)
    encoder = model.encoder
    predictions = model.model.predict(_matrix(test, encoder))
    truth = test["score"].to_numpy()
    predicted_top = set(np.argsort(predictions)[-top_k:])
    actual_top = set(np.argsort(truth)[-top_k:])
    return {
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        "mae": float(mean_absolute_error(truth, predictions)),
        "top_k_recall": float(len(predicted_top & actual_top) / max(1, len(actual_top))),
        "predicted_top_mean": float(np.mean(truth[list(predicted_top)])),
        "oracle_top_mean": float(np.mean(truth[list(actual_top)])),
    }
