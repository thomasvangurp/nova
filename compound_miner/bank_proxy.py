"""Standalone chemistry ranker trained on our own exact-scorer labels.

Why this exists, measured on epoch 25176 data:

`OnlineResidualProxy` cannot escape the frozen historical proxy's bias no matter
how many labels it is given. Its `predict` computes
`baseline = frozen.predict(...)`, builds its design matrix with
`_matrix(..., self.frozen.encoder, ...)`, and returns `proxy_mean + weight*delta`.
The `ComponentEncoder` contributes six target-encoded features that fall back to
the global mean for any reagent absent from the public history -- exactly the
chemistry we need to rank -- and the correction is damped by
`weight = n/(n+prior_strength)`. So it is a residual on a base that is really
detecting "contains known-good reagents".

Measured consequences on our own NIM labels:

    frozen historical proxy   +0.1085 Spearman overall, ~0 within every
                              novelty stratum (it separates groups, not molecules)
    this model, broad pool     0.5527 Spearman, 0.4848 on novel-reagent molecules,
                              top-decile precision 0.3733 against 0.0995 random

The design difference is deliberate and is the whole point: **chemistry features
only**, no component identities and no frozen anchor. The model then generalises
on chemistry rather than on reagent identity, which is why it transfers to novel
reagents in a disjoint run.

What it does NOT fix: discriminating among already-elite candidates. On a pool
that has itself been filtered into a narrow quality band, rank correlation
collapses to ~0 or negative (range restriction). Use this to choose *which few
hundred molecules to score*, never to choose the final portfolio -- that needs
exact scores.

Intended lifecycle: train offline on an accumulating label bank, ship the fitted
model, and optionally refit with the current challenge's labels appended as they
arrive. Pre-training is the point -- round one is otherwise spent selecting
molecules with a ranker that has no labels at all.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .features import molecule_features
from .types import Candidate

BANK_FORMAT = 1


def chemistry_matrix(smiles: Iterable[str]) -> np.ndarray:
    """Morgan fingerprint + physicochemical descriptors, nothing else."""
    values = list(smiles)
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([molecule_features(value) for value in values])


def load_labels(archives: Iterable[str | Path], reaction: int = 3,
                scorer: str | None = None) -> pd.DataFrame:
    """Pool exact-scored molecules from one or more candidate archives.

    Repeats of the same chemical identity are averaged. Rows are tagged with the
    archive they came from so a caller can hold an entire run out.
    """
    merged: dict[str, dict] = {}
    for archive in archives:
        path = Path(archive)
        if not path.is_file():
            continue
        query = ("SELECT name,smiles,components,score,scorer FROM candidates "
                 "WHERE reaction=? AND score IS NOT NULL")
        with sqlite3.connect(path) as connection:
            try:
                rows = connection.execute(query, (reaction,)).fetchall()
            except sqlite3.OperationalError:
                rows = [(n, s, c, v, None) for n, s, c, v in connection.execute(
                    "SELECT name,smiles,components,score FROM candidates "
                    "WHERE reaction=? AND score IS NOT NULL", (reaction,))]
        for name, smiles, components, score, row_scorer in rows:
            if scorer is not None and row_scorer is not None and row_scorer != scorer:
                continue
            if score is None or not math.isfinite(float(score)):
                continue
            entry = merged.setdefault(name, dict(name=name, smiles=smiles,
                components=tuple(json.loads(components)), source=path.parent.name,
                scores=[]))
            entry["scores"].append(float(score))
    if not merged:
        return pd.DataFrame(columns=["name", "smiles", "component_a", "component_b",
                                     "component_c", "score", "repeats", "source"])
    records = []
    for entry in merged.values():
        components = entry["components"]
        if len(components) != 3:
            continue
        records.append(dict(name=entry["name"], smiles=entry["smiles"],
            component_a=components[0], component_b=components[1],
            component_c=components[2], score=float(np.mean(entry["scores"])),
            repeats=len(entry["scores"]), source=entry["source"]))
    return pd.DataFrame.from_records(records).sort_values("name").reset_index(drop=True)


class BankProxy:
    """Drop-in acquisition proxy: same predict() contract as TargetProxy."""

    # The archive refresh hook re-scores unscored rows after the model changes.
    refresh_acquisition_archive = True

    def __init__(self, trees: int = 256, jobs: int = -1, random_state: int = 68,
                 min_samples_leaf: int = 2, max_features: float = 0.55) -> None:
        self.model = ExtraTreesRegressor(n_estimators=trees,
                                         min_samples_leaf=min_samples_leaf,
                                         max_features=max_features, n_jobs=jobs,
                                         random_state=random_state)
        self.format = BANK_FORMAT
        self.label_count = 0
        self.sources: list[str] = []
        self.fitted = False
        self.trained_at = None
        self.scorer = None

    def fit(self, frame: pd.DataFrame, *, scorer: str | None = None,
            chemistry: np.ndarray | None = None) -> "BankProxy":
        if frame.empty:
            raise ValueError("empty_label_bank")
        if "smiles" not in frame or "score" not in frame:
            raise ValueError("label_bank_columns")
        target = frame["score"].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(target)):
            raise ValueError("nonfinite_labels")
        matrix = chemistry_matrix(frame["smiles"]) if chemistry is None else chemistry
        if len(matrix) != len(frame):
            raise ValueError("chemistry_rows_mismatch")
        self.model.fit(matrix, target)
        self.fitted = True
        self.label_count = int(len(frame))
        self.sources = sorted(set(frame["source"])) if "source" in frame else []
        self.trained_at = time.time()
        self.scorer = scorer
        return self

    def predict(self, candidates: Iterable[Candidate],
                chemistry: np.ndarray | None = None) -> list[Candidate]:
        values = list(candidates)
        if not values:
            return []
        if not self.fitted:
            raise RuntimeError("bank_proxy_not_fitted")
        matrix = (chemistry_matrix(item.smiles for item in values)
                  if chemistry is None else chemistry)
        # Stream tree moments rather than materialising trees x candidates; the
        # acquisition pass calls this with hundreds of thousands of rows.
        total = np.zeros(len(values))
        squares = np.zeros(len(values))
        for tree in self.model.estimators_:
            prediction = tree.predict(matrix)
            total += prediction
            squares += prediction * prediction
        count = len(self.model.estimators_)
        mean = total / count
        variance = np.maximum(0.0, squares / count - mean * mean)
        return [dataclasses.replace(item, proxy_mean=float(m), proxy_std=float(np.sqrt(v)))
                for item, m, v in zip(values, mean, variance)]

    def metadata(self) -> dict:
        return dict(format=self.format, fitted=self.fitted, label_count=self.label_count,
                    sources=list(self.sources), scorer=self.scorer,
                    trees=int(self.model.n_estimators),
                    features="morgan1024+descriptors, no component encodings",
                    anchored_on_frozen_proxy=False)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        joblib.dump(self, temporary, compress=3)
        temporary.replace(destination)

    @staticmethod
    def load(path: str | Path) -> "BankProxy":
        model = joblib.load(Path(path))
        if not isinstance(model, BankProxy):
            raise ValueError("not_a_bank_proxy")
        if getattr(model, "format", None) != BANK_FORMAT:
            raise ValueError("bank_proxy_format")
        if not model.fitted:
            raise ValueError("bank_proxy_not_fitted")
        return model
