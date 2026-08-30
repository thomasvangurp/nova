from __future__ import annotations

import dataclasses
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .archive import CandidateArchive
from .database import ReactionDatabase
from .proxy import TargetProxy
from .scorer import NIMScorer
from .selection import select_submission
from .types import Candidate
from .validity import HistoricalIndex, ValidityGate


class ProposalPolicy:
    """Mix target-conditioned component priors, local substitutions, and exploration."""

    def __init__(self, history: pd.DataFrame, reaction: int, database: ReactionDatabase, rng: random.Random) -> None:
        self.frame = history[history["reaction"] == reaction].sort_values("score", ascending=False)
        self.reaction = reaction
        self.database = database
        self.rng = rng
        self.roles = database.reactions[reaction].roles
        self.weighted_ids: list[tuple[np.ndarray, np.ndarray]] = []
        for column in ("component_a", "component_b", "component_c")[: len(self.roles)]:
            grouped = self.frame.groupby(column)["score"].agg(["mean", "count"])
            if grouped.empty:
                self.weighted_ids.append((np.asarray([], dtype=np.int64), np.asarray([], dtype=float)))
                continue
            baseline = float(grouped["mean"].quantile(0.5))
            weights = np.exp(np.clip((grouped["mean"].to_numpy() - baseline) / 0.015, -10, 10)) * np.sqrt(grouped["count"].to_numpy())
            self.weighted_ids.append((grouped.index.to_numpy(dtype=np.int64), weights / weights.sum()))

    def _weighted_component(self, index: int) -> int:
        identifiers, weights = self.weighted_ids[index]
        if len(identifiers) == 0:
            return self.rng.choice(self.database.role_ids[self.roles[index]])
        return int(self.rng.choices(identifiers.tolist(), weights=weights.tolist(), k=1)[0])

    def propose(self) -> tuple[int, ...]:
        draw = self.rng.random()
        if draw < 0.55 and not self.frame.empty:
            # Structural neighborhood: retain a successful context and substitute
            # one role using its target-conditioned prior.
            parent = self.frame.iloc[self.rng.randrange(min(len(self.frame), 3000))]
            components = [int(parent["component_a"]), int(parent["component_b"])]
            if len(self.roles) == 3:
                components.append(int(parent["component_c"]))
            index = self.rng.randrange(len(components))
            components[index] = self._weighted_component(index)
            return tuple(components)
        if draw < 0.9:
            return tuple(self._weighted_component(index) for index in range(len(self.roles)))
        return self.database.random_components(self.reaction, self.rng)


class SearchEngine:
    def __init__(
        self,
        database: ReactionDatabase,
        history: pd.DataFrame,
        proxy: TargetProxy,
        archive: CandidateArchive,
        historical_index: HistoricalIndex,
        scorer: NIMScorer | None = None,
        seed: int = 68,
    ) -> None:
        self.database = database
        self.history = history
        self.proxy = proxy
        self.archive = archive
        self.gate = ValidityGate(historical_index)
        self.scorer = scorer
        self.rng = random.Random(seed)

    def run_round(
        self,
        reaction: int,
        proposals: int = 10000,
        exact_shortlist: int = 64,
        similarity_shortlist: int = 800,
        result_path: str | Path | None = None,
    ) -> dict[str, float]:
        started = time.monotonic()
        policy = ProposalPolicy(self.history, reaction, self.database, self.rng)
        candidates: list[Candidate] = []
        seen: set[tuple[int, ...]] = set()
        for _ in range(proposals * 2):
            if len(candidates) >= proposals:
                break
            components = policy.propose()
            if components in seen:
                continue
            seen.add(components)
            candidate = self.database.candidate(reaction, components)
            if candidate and self.gate.accepts(candidate, check_similarity=False):
                candidates.append(candidate)
        predicted = self.proxy.predict(candidates)
        predicted.sort(key=lambda item: item.proxy_mean + 0.35 * item.proxy_std, reverse=True)
        novel = []
        seen_inchikeys: set[str] = set()
        for item in predicted[:similarity_shortlist]:
            if item.inchikey in seen_inchikeys or not self.gate.accepts(item, check_similarity=True):
                continue
            seen_inchikeys.add(item.inchikey)
            novel.append(item)
        self.archive.add_candidates(novel)
        if self.scorer and novel:
            scored = self.scorer.score(novel[:exact_shortlist])
            self.archive.add_scores(scored)
        top = self.archive.top(reaction, 500, scored_only=self.scorer is not None)
        selected = select_submission(top)
        if result_path:
            CandidateArchive.atomic_write_result(result_path, selected)
        return {
            "elapsed_seconds": time.monotonic() - started,
            "constructed": float(len(candidates)),
            "novel": float(len(novel)),
            "exact_scored": float(min(len(novel), exact_shortlist) if self.scorer else 0),
            "submission_score": float(sum(item.score for item in selected)),
        }
