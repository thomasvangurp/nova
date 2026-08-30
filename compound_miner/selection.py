from __future__ import annotations

from typing import Iterable

from .types import ScoredCandidate
from .validity import maccs_entropy


def select_submission(
    scores: Iterable[ScoredCandidate], count: int = 20, minimum_entropy: float = 0.1
) -> list[ScoredCandidate]:
    unique: list[ScoredCandidate] = []
    names: set[str] = set()
    inchikeys: set[str] = set()
    for item in sorted(scores, key=lambda value: value.score, reverse=True):
        if item.candidate.name in names or item.candidate.inchikey in inchikeys:
            continue
        unique.append(item)
        names.add(item.candidate.name)
        inchikeys.add(item.candidate.inchikey)
    if len(unique) < count:
        raise ValueError(f"need {count} unique candidates, only {len(unique)} available")
    selected = unique[:count]
    if maccs_entropy(item.candidate for item in selected) >= minimum_entropy:
        return selected

    # Preserve the highest-score core and exchange its weakest member for the
    # candidate yielding the largest entropy gain until the exact gate passes.
    remaining = unique[count:]
    for replacement_index in range(count - 1, -1, -1):
        best = max(
            remaining,
            key=lambda candidate: maccs_entropy(
                [item.candidate for index, item in enumerate(selected) if index != replacement_index]
                + [candidate.candidate]
            ),
            default=None,
        )
        if best is None:
            break
        selected[replacement_index] = best
        remaining.remove(best)
        if maccs_entropy(item.candidate for item in selected) >= minimum_entropy:
            return sorted(selected, key=lambda value: value.score, reverse=True)
    raise ValueError("candidate archive cannot satisfy the MACCS entropy constraint")

