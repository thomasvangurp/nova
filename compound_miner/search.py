from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .archive import CandidateArchive
from .database import ReactionDatabase
from .features import molecule_features
from .proxy import TargetProxy
from .scorer import NIMScorer
from .selection import select_submission
from .types import Candidate, ScoredCandidate
from .validity import HistoricalIndex, ValidityGate


_worker_database: ReactionDatabase | None = None
_worker_gate: ValidityGate | None = None
_worker_history: HistoricalIndex | None = None


def _initialize_construction_worker(database_path: str) -> None:
    global _worker_database, _worker_gate
    _worker_database = ReactionDatabase(database_path)
    _worker_gate = ValidityGate()


def _construct_candidate(arguments: tuple[int, tuple[int, ...]]) -> Candidate | None:
    reaction, components = arguments
    if _worker_database is None or _worker_gate is None:
        raise RuntimeError("construction worker is not initialized")
    candidate = _worker_database.candidate(reaction, components)
    if candidate is None or not _worker_gate.accepts(candidate, check_similarity=False):
        return None
    return candidate


def _construct_candidate_with_features(
    arguments: tuple[int, tuple[int, ...]],
) -> tuple[Candidate, np.ndarray] | None:
    candidate = _construct_candidate(arguments)
    if candidate is None:
        return None
    # Fuse proxy featurization into the construction worker.  Re-parsing every
    # product serially in TargetProxy.predict dominated the widened CPU funnel.
    return candidate, molecule_features(candidate.smiles)


def _check_historical_novelty(candidate: Candidate) -> bool:
    if _worker_history is None:
        raise RuntimeError("historical novelty worker is not initialized")
    return _worker_history.is_novel(candidate)


class ProposalPolicy:
    """Mix target-conditioned component priors, local substitutions, and exploration."""

    def __init__(
        self,
        history: pd.DataFrame,
        reaction: int,
        database: ReactionDatabase,
        rng: random.Random,
        exact_parents: list[ScoredCandidate] | None = None,
    ) -> None:
        self.frame = history[history["reaction"] == reaction].sort_values("score", ascending=False)
        self.reaction = reaction
        self.database = database
        self.rng = rng
        self.exact_parents = exact_parents or []
        if self.exact_parents:
            exact_scores = np.asarray([item.score for item in self.exact_parents])
            exact_scores -= exact_scores.max()
            weights = np.exp(np.clip(exact_scores / 0.02, -20, 0))
            self.exact_parent_weights = (weights / weights.sum()).tolist()
        else:
            self.exact_parent_weights = []
        self.roles = database.reactions[reaction].roles
        self.weighted_ids: list[tuple[list[int], list[float]]] = []
        for column in ("component_a", "component_b", "component_c")[: len(self.roles)]:
            grouped = self.frame.groupby(column)["score"].agg(["mean", "count"])
            if grouped.empty:
                self.weighted_ids.append(([], []))
                continue
            baseline = float(grouped["mean"].quantile(0.5))
            weights = np.exp(np.clip((grouped["mean"].to_numpy() - baseline) / 0.015, -10, 10)) * np.sqrt(grouped["count"].to_numpy())
            normalized = (weights / weights.sum()).tolist()
            # random.choices computes this O(number of known components) prefix
            # sum on every call when given raw weights.  Cache the identical
            # Python accumulation once so each draw becomes a binary search.
            self.weighted_ids.append(
                (
                    grouped.index.to_numpy(dtype=np.int64).tolist(),
                    list(itertools.accumulate(normalized)),
                )
            )

    def _weighted_component(self, index: int) -> int:
        identifiers, cumulative_weights = self.weighted_ids[index]
        if len(identifiers) == 0:
            return self.rng.choice(self.database.role_ids[self.roles[index]])
        return int(
            self.rng.choices(
                identifiers, cum_weights=cumulative_weights, k=1
            )[0]
        )

    def propose(self) -> tuple[int, ...]:
        draw = self.rng.random()
        if draw < 0.40 and self.exact_parents:
            # Once validator-matched labels exist, search substitutions around
            # those discoveries instead of continuing to trust the proxy.
            parent = self.rng.choices(
                self.exact_parents, weights=self.exact_parent_weights, k=1
            )[0].candidate
            components = list(parent.components)
            index = self.rng.randrange(len(components))
            components[index] = self._weighted_component(index)
            return tuple(components)
        if draw < 0.70 and not self.frame.empty:
            # Structural neighborhood: retain a successful context and substitute
            # one role using its target-conditioned prior.
            parent = self.frame.iloc[self.rng.randrange(min(len(self.frame), 3000))]
            components = [int(parent["component_a"]), int(parent["component_b"])]
            if len(self.roles) == 3:
                components.append(int(parent["component_c"]))
            index = self.rng.randrange(len(components))
            components[index] = self._weighted_component(index)
            return tuple(components)
        if draw < 0.92:
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
        construction_workers: int = 1,
        similarity_workers: int = 1,
    ) -> dict[str, float]:
        started = time.monotonic()
        exact_parents = self.archive.top(reaction, limit=100, scored_only=True)
        policy = ProposalPolicy(
            self.history, reaction, self.database, self.rng, exact_parents=exact_parents
        )
        if construction_workers > 1:
            component_sets: list[tuple[int, ...]] = []
            seen: set[tuple[int, ...]] = set()
            pool_target = proposals * 2
            for _ in range(pool_target * 2):
                if len(component_sets) >= pool_target:
                    break
                components = policy.propose()
                if components in seen:
                    continue
                seen.add(components)
                component_sets.append(components)
            proposed_tuples = len(component_sets)
            with ProcessPoolExecutor(
                max_workers=construction_workers,
                initializer=_initialize_construction_worker,
                initargs=(str(self.database.path),),
            ) as pool:
                constructed = pool.map(
                    _construct_candidate_with_features,
                    ((reaction, components) for components in component_sets),
                    chunksize=128,
                )
                constructed_with_features = [
                    item for item in constructed if item is not None
                ][:proposals]
                candidates = [item[0] for item in constructed_with_features]
                chemistry = np.vstack(
                    [item[1] for item in constructed_with_features]
                ) if constructed_with_features else None
        else:
            # Keep the established single-process trajectory bit-for-bit stable.
            # The parallel path is deliberately opt-in until its matched quality
            # replay passes, because harvesting a different number of failed
            # reactions changes all subsequent RNG decisions.
            candidates = []
            chemistry = None
            seen = set()
            proposed_tuples = 0
            for _ in range(proposals * 2):
                if len(candidates) >= proposals:
                    break
                components = policy.propose()
                if components in seen:
                    continue
                seen.add(components)
                proposed_tuples += 1
                candidate = self.database.candidate(reaction, components)
                if candidate and self.gate.accepts(candidate, check_similarity=False):
                    candidates.append(candidate)
        constructed_at = time.monotonic()
        predicted = self.proxy.predict(candidates, chemistry=chemistry)
        predicted.sort(key=lambda item: item.proxy_mean + 0.35 * item.proxy_std, reverse=True)
        predicted_at = time.monotonic()
        unique_shortlist = []
        seen_inchikeys: set[str] = set()
        for item in predicted[:similarity_shortlist]:
            if item.inchikey in seen_inchikeys:
                continue
            seen_inchikeys.add(item.inchikey)
            unique_shortlist.append(item)
        # Every item passed the full base gate during construction.  Re-run only
        # historical novelty here.  On Linux, fork lets workers share the large
        # immutable fingerprint index copy-on-write instead of rebuilding it.
        if similarity_workers > 1 and "fork" in mp.get_all_start_methods():
            global _worker_history
            _worker_history = self.gate.history
            with ProcessPoolExecutor(
                max_workers=similarity_workers,
                mp_context=mp.get_context("fork"),
            ) as pool:
                novelty = pool.map(
                    _check_historical_novelty,
                    unique_shortlist,
                    chunksize=64,
                )
                novel = [
                    item
                    for item, accepted in zip(unique_shortlist, novelty)
                    if accepted
                ]
        else:
            novel = [
                item
                for item in unique_shortlist
                if self.gate.history.is_novel(item)
            ]
        similarity_at = time.monotonic()
        self.archive.add_candidates(novel)
        if self.scorer and novel:
            scored = self.scorer.score(novel[:exact_shortlist])
            self.archive.add_scores(scored)
        scored_at = time.monotonic()
        top = self.archive.top(reaction, 500, scored_only=self.scorer is not None)
        selected = select_submission(top)
        if result_path:
            CandidateArchive.atomic_write_result(result_path, selected)
        return {
            "elapsed_seconds": time.monotonic() - started,
            "construction_seconds": constructed_at - started,
            "proxy_seconds": predicted_at - constructed_at,
            "similarity_seconds": similarity_at - predicted_at,
            "exact_seconds": scored_at - similarity_at,
            "proposed_tuples": float(proposed_tuples),
            "constructed": float(len(candidates)),
            "novel": float(len(novel)),
            "exact_scored": float(min(len(novel), exact_shortlist) if self.scorer else 0),
            "submission_score": float(sum(item.score for item in selected)),
        }
