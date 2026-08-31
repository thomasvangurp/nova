from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd


@dataclass(frozen=True)
class _Estimate:
    mean: float
    count: int


class ContextualDeltaPolicy:
    """Conservative context-conditioned reagent substitutions.

    A candidate replacement is evaluated only from effects that involve the
    replaced role. A strong fixed B-C pair therefore cannot make an unrelated A
    look good, which is the failure mode that dominated the first live rxn5 run.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        roles: int,
        smoothing: float = 8.0,
        unseen_penalty: float = 0.012,
        pool_size: int = 256,
    ) -> None:
        self.roles = roles
        self.smoothing = smoothing
        self.unseen_penalty = unseen_penalty
        self.global_mean = float(frame["score"].mean())
        columns = ("component_a", "component_b", "component_c")[:roles]
        collapsed = frame.groupby(list(columns), as_index=False)["score"].mean()
        self.role: list[dict[int, _Estimate]] = []
        self.pairs: dict[tuple[int, int], dict[tuple[int, int], _Estimate]] = {}
        self.pools: list[list[int]] = []
        for column in columns:
            grouped = collapsed.groupby(column)["score"].agg(["mean", "count"])
            estimates = {
                int(identifier): _Estimate(float(row["mean"]), int(row["count"]))
                for identifier, row in grouped.iterrows()
            }
            self.role.append(estimates)
            ranked = sorted(
                estimates,
                key=lambda identifier: (
                    self._smooth(estimates[identifier]),
                    estimates[identifier].count,
                    -identifier,
                ),
                reverse=True,
            )
            self.pools.append(ranked[:pool_size])
        for left in range(roles):
            for right in range(left + 1, roles):
                grouped = collapsed.groupby([columns[left], columns[right]])["score"].agg(
                    ["mean", "count"]
                )
                self.pairs[(left, right)] = {
                    (int(a), int(b)): _Estimate(float(row["mean"]), int(row["count"]))
                    for (a, b), row in grouped.iterrows()
                }

    def _smooth(self, estimate: _Estimate) -> float:
        return (
            estimate.mean * estimate.count + self.global_mean * self.smoothing
        ) / (estimate.count + self.smoothing)

    def _pair(self, left: int, left_value: int, right: int, right_value: int) -> _Estimate | None:
        if left > right:
            left, right = right, left
            left_value, right_value = right_value, left_value
        return self.pairs[(left, right)].get((left_value, right_value))

    def replacement_value(
        self, components: tuple[int, ...], index: int, candidate: int
    ) -> float:
        role_estimate = self.role[index].get(candidate)
        value = self._smooth(role_estimate) if role_estimate else self.global_mean
        known = 0
        for other, other_value in enumerate(components):
            if other == index:
                continue
            pair = self._pair(index, candidate, other, int(other_value))
            if pair is None:
                value -= self.unseen_penalty
                continue
            known += 1
            # Add only the interaction residual involving the candidate. The
            # fixed context pair is intentionally absent from this delta.
            value += self._smooth(pair) - self.global_mean
        if known == 0:
            value -= self.unseen_penalty
        return value

    @lru_cache(maxsize=32_768)
    def ranked_replacements(
        self, components: tuple[int, ...], index: int, limit: int = 64
    ) -> list[tuple[int, float]]:
        current = int(components[index])
        values = [
            (candidate, self.replacement_value(components, index, candidate))
            for candidate in self.pools[index]
            if candidate != current
        ]
        values.sort(key=lambda item: (-item[1], item[0]))
        return values[:limit]

    def propose_replacement(
        self,
        components: tuple[int, ...],
        index: int,
        rng: random.Random,
        top_k: int = 32,
        temperature: float = 0.01,
    ) -> int:
        ranked = self.ranked_replacements(components, index, top_k)
        if not ranked:
            return int(components[index])
        peak = ranked[0][1]
        weights = [math.exp(max(-30.0, (value - peak) / temperature)) for _, value in ranked]
        return int(rng.choices([identifier for identifier, _ in ranked], weights=weights, k=1)[0])
