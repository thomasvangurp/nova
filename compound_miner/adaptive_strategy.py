"""Opt-in, offline-only feedback models. NIM labels stay search-only.

The historical proxy remains frozen. A separate residual head learns from the
current challenge's NIM observations, without relabelling historical validator
scores or incorporating a public leader. No release or publishing interface.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from .contextual import ContextualDeltaPolicy
from .features import molecule_features
from .proxy import _matrix
from .search import ProposalPolicy
from .types import Candidate, ScoredCandidate


def candidate_frame(candidates: list[Candidate]) -> pd.DataFrame:
    return pd.DataFrame({
        "smiles": [c.smiles for c in candidates],
        "reaction": [c.reaction for c in candidates],
        "component_a": [c.components[0] for c in candidates],
        "component_b": [c.components[1] for c in candidates],
        "component_c": [c.components[2] for c in candidates],
    })


class OnlineResidualProxy:
    """Current-challenge NIM correction on a frozen historical proxy.

    Tree disagreement is a heuristic, not calibrated posterior uncertainty.
    A fresh object is required for every challenge/recipe; expected_scorer must
    encode the recipe and runtime provenance. Unseen labels are never fit until
    observe() is called, enabling prediction-before-update replay.
    """

    refresh_acquisition_archive = True

    def __init__(self, frozen, *, expected_scorer: str, reaction: int = 3,
                 trees: int = 64, jobs: int = 4, seed: int = 68,
                 prior_strength: float = 128.0, minimum_labels: int = 32):
        if reaction != 3 or not expected_scorer or trees < 1 or jobs == 0:
            raise ValueError("invalid_adaptive_contract")
        if prior_strength < 0 or minimum_labels < 2:
            raise ValueError("invalid_adaptive_contract")
        self.frozen = frozen
        self.expected_scorer = expected_scorer
        self.reaction = reaction
        self.prior_strength = prior_strength
        self.minimum_labels = minimum_labels
        self.model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=4,
                                        max_features=0.35, n_jobs=jobs,
                                        random_state=seed)
        self.observations: dict[str, list[ScoredCandidate]] = defaultdict(list)
        self.features: dict[str, np.ndarray] = {}
        self.baselines: dict[str, float] = {}
        self.fitted = False
        self.version = 0

    def _validate_candidates(self, candidates):
        for c in candidates:
            if c.reaction != self.reaction or len(c.components) != 3 or not c.inchikey:
                raise ValueError("challenge_identity_mismatch")

    def _design(self, candidates, chemistry):
        """Residual features, matched to whatever the base model exposes.

        A frozen TargetProxy carries a fitted ComponentEncoder, so the residual
        is learned in the same space as its base. A chemistry-only base (see
        BankProxy) deliberately has no encoder -- its whole point is that the
        six component target-encodings return the global mean for any reagent
        absent from history. Reintroducing them here would put the residual back
        in the space the base was built to escape, so fall back to chemistry.
        """
        encoder = getattr(self.frozen, "encoder", None)
        if encoder is None:
            return chemistry
        return _matrix(candidate_frame(candidates), encoder, chemistry)

    def feedback_frame(self) -> pd.DataFrame:
        rows = []
        for values in self.observations.values():
            c = values[0].candidate
            rows.append(dict(reaction=c.reaction, component_a=c.components[0],
                             component_b=c.components[1], component_c=c.components[2],
                             score=float(np.mean([v.score for v in values])),
                             observations=len(values)))
        return pd.DataFrame(rows, columns=["reaction", "component_a", "component_b",
                                           "component_c", "score", "observations"])

    def observe(self, scores: Iterable[ScoredCandidate]) -> dict:
        started = time.monotonic()
        values = list(scores)
        self._validate_candidates([v.candidate for v in values])
        # Validate the whole batch before mutating state. No mixed-recipe labels.
        pending = {}
        for v in values:
            if v.scorer != self.expected_scorer:
                raise ValueError("search_label_provenance_mismatch")
            if not all(math.isfinite(x) for x in (v.score, v.affinity_probability_binary,
                                                 v.affinity_pred_value)):
                raise ValueError("nonfinite_feedback")
            c = v.candidate
            if c.heavy_atoms <= 0 or not math.isclose(
                    v.score, (v.affinity_probability_binary-v.affinity_pred_value)/c.heavy_atoms,
                    rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("feedback_objective_mismatch")
            previous = self.observations.get(c.inchikey)
            known = previous[0].candidate if previous else pending.get(c.inchikey)
            if known and (known.smiles, known.reaction, known.components) != (
                    c.smiles, c.reaction, c.components):
                raise ValueError("feedback_identity_mismatch")
            pending[c.inchikey] = c
        fresh = [c for key, c in pending.items() if key not in self.features]
        if fresh:
            chemistry = np.vstack([molecule_features(c.smiles) for c in fresh])
            baseline = self.frozen.predict(fresh, chemistry=chemistry)
            matrix = self._design(fresh, chemistry)
            for c, b, feature in zip(fresh, baseline, matrix):
                self.features[c.inchikey] = feature
                self.baselines[c.inchikey] = b.proxy_mean
        for v in values:
            self.observations[v.candidate.inchikey].append(v)
        if len(self.observations) >= self.minimum_labels:
            keys = sorted(self.observations)
            y = np.asarray([np.mean([v.score for v in self.observations[k]])
                            - self.baselines[k] for k in keys])
            weights = np.sqrt([len(self.observations[k]) for k in keys])
            self.model.fit(np.vstack([self.features[k] for k in keys]), y,
                           sample_weight=weights)
            self.fitted = True
            self.version += 1
        return dict(unique_labels=len(self.observations), model_version=self.version,
                    fitted=self.fitted, update_seconds=time.monotonic()-started,
                    scorer=self.expected_scorer, canonical_labels_added=0)

    def predict(self, candidates: Iterable[Candidate], chemistry=None) -> list[Candidate]:
        values = list(candidates)
        self._validate_candidates(values)
        if not values:
            return []
        if chemistry is None:
            chemistry = np.vstack([molecule_features(c.smiles) for c in values])
        baseline = self.frozen.predict(values, chemistry=chemistry)
        if not self.fitted:
            return baseline
        matrix = self._design(values, chemistry)
        # Stream moments instead of retaining trees x 250,000 predictions.
        mean = np.zeros(len(values))
        square = np.zeros(len(values))
        for tree in self.model.estimators_:
            prediction = tree.predict(matrix)
            mean += prediction
            square += prediction * prediction
        mean /= len(self.model.estimators_)
        variance = np.maximum(0, square/len(self.model.estimators_)-mean*mean)
        weight = len(self.observations)/(len(self.observations)+self.prior_strength)
        return [dataclasses.replace(c, proxy_mean=c.proxy_mean+weight*delta,
                                    proxy_std=float(np.sqrt((1-weight)**2*c.proxy_std**2
                                                            + weight**2*var)))
                for c, delta, var in zip(baseline, mean, variance)]


class FeedbackProposalPolicy(ProposalPolicy):
    """Reaction-3 contextual moves with separate historical and NIM estimates.

    Preserve the baseline's parent/move mixture. Replacement sources are an
    explicit experimental 65% current-feedback / 20% historical-context /
    15% unrestricted valid-role mix. Missing current feedback falls back to
    historical context. Unrestricted draws keep unseen reagents reachable.
    Product construction and the strict novelty/validator gates stay external.
    """

    def __init__(self, history, reaction, database, rng, exact_parents=None,
                 *, feedback: pd.DataFrame):
        if reaction != 3:
            raise ValueError("adaptive_policy_reaction_mismatch")
        super().__init__(history, reaction, database, rng, exact_parents)
        self.contextual = (ContextualDeltaPolicy(self.frame, len(self.roles))
                           if not self.frame.empty else None)
        if not feedback.empty and set(feedback["reaction"]) != {reaction}:
            raise ValueError("feedback_reaction_mismatch")
        self.current_context = (ContextualDeltaPolicy(feedback, len(self.roles))
                                if not feedback.empty else None)

    def _replacement(self, components, index):
        draw = self.rng.random()
        if draw < 0.65 and self.current_context is not None:
            policy = self.current_context
        elif draw < 0.85 and self.contextual is not None:
            policy = self.contextual
        else:
            choices = self.database.role_ids[self.roles[index]]
            if not choices:
                raise ValueError("empty_reagent_role")
            return self.rng.choice(choices)
        return policy.propose_replacement(tuple(components), index, self.rng)


def refresh_unscored_archive(archive, reaction, proxy, *, batch_size=2048) -> dict:
    """Refresh acquisition signals without altering any exact/NIM score fields.

    INSERT OR IGNORE leaves old proxy signals in a persistent archive. Without
    this refresh, a newly learned proxy would only affect newly generated rows.
    """
    import json
    import sqlite3
    if batch_size < 1:
        raise ValueError("invalid_refresh_batch_size")
    started = time.monotonic()
    count = 0
    with sqlite3.connect(archive.path) as connection:
        rows = connection.execute(
            "SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds "
            "FROM candidates WHERE reaction=? AND score IS NULL ORDER BY name",
            (reaction,)).fetchall()
        for offset in range(0, len(rows), batch_size):
            candidates = [Candidate(r[0], r[1], r[2], tuple(json.loads(r[3])), r[4], r[5], r[6])
                          for r in rows[offset:offset+batch_size]]
            predictions = proxy.predict(candidates)
            if len(predictions) != len(candidates) or [c.name for c in predictions] != [c.name for c in candidates]:
                raise ValueError("proxy_refresh_identity_mismatch")
            if not all(math.isfinite(c.proxy_mean) and math.isfinite(c.proxy_std)
                       and c.proxy_std >= 0 for c in predictions):
                raise ValueError("invalid_acquisition_signal")
            connection.executemany(
                "UPDATE candidates SET proxy_mean=?,proxy_std=? WHERE name=? AND score IS NULL",
                [(c.proxy_mean, c.proxy_std, c.name) for c in predictions])
            count += len(predictions)
    return dict(refreshed_unscored=count, refresh_seconds=time.monotonic()-started,
                scored_rows_changed=0)
