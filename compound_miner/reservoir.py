from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterable

from .types import Candidate


class CandidateReservoir:
    """Bounded, deterministic retention for million-scale proxy streams.

    Each view is allowed to see the full stream.  Final quotas are filled in a
    fixed order with de-duplication, so strong exploitation candidates coexist
    with uncertainty and chemistry-agnostic coverage without timing-dependent
    asynchronous harvesting.
    """

    def __init__(
        self,
        capacity: int,
        exploit_fraction: float = 0.50,
        uncertainty_fraction: float = 0.25,
        mean_fraction: float = 0.15,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 <= exploit_fraction <= 1:
            raise ValueError("exploit_fraction must be between zero and one")
        if not 0 <= uncertainty_fraction <= 1:
            raise ValueError("uncertainty_fraction must be between zero and one")
        if not 0 <= mean_fraction <= 1:
            raise ValueError("mean_fraction must be between zero and one")
        if exploit_fraction + uncertainty_fraction + mean_fraction > 1:
            raise ValueError("reservoir fractions cannot sum above one")
        self.capacity = capacity
        self.quotas = {
            "exploit": int(capacity * exploit_fraction),
            "uncertainty": int(capacity * uncertainty_fraction),
            "mean": int(capacity * mean_fraction),
        }
        self.quotas["coverage"] = capacity - sum(self.quotas.values())
        self.heaps: dict[str, list[tuple[float, int, Candidate]]] = {
            name: [] for name in self.quotas
        }
        self.seen = 0

    @staticmethod
    def _coverage_score(candidate: Candidate) -> int:
        digest = hashlib.blake2b(candidate.name.encode(), digest_size=8).digest()
        # Keeping the largest negative hashes is equivalent to retaining the
        # smallest stable hashes, a deterministic uniform reservoir.
        return -int.from_bytes(digest, "big")

    def _offer(
        self, name: str, score: float | int, order: int, item: Candidate
    ) -> None:
        heap = self.heaps[name]
        entry = (score, -order, item)
        if len(heap) < self.capacity:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    def add(self, candidates: Iterable[Candidate]) -> None:
        for item in candidates:
            order = self.seen
            self.seen += 1
            self._offer(
                "exploit",
                item.proxy_mean + 0.35 * item.proxy_std,
                order,
                item,
            )
            self._offer("uncertainty", item.proxy_std, order, item)
            self._offer("mean", item.proxy_mean, order, item)
            self._offer("coverage", self._coverage_score(item), order, item)

    def selected(self) -> list[Candidate]:
        selected: list[Candidate] = []
        names: set[str] = set()
        for view in ("exploit", "uncertainty", "mean", "coverage"):
            quota = self.quotas[view]
            if quota <= 0:
                continue
            added = 0
            ranked = sorted(self.heaps[view], key=lambda entry: entry[:2], reverse=True)
            for _, _, item in ranked:
                if item.name in names:
                    continue
                names.add(item.name)
                selected.append(item)
                added += 1
                if added >= quota:
                    break
        return selected[: self.capacity]
