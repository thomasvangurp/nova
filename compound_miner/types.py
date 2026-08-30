from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def parse_reaction_name(name: str) -> tuple[int, tuple[int, ...]]:
    parts = name.split(":")
    if len(parts) not in (4, 5) or parts[0] != "rxn":
        raise ValueError(f"invalid reaction identifier: {name!r}")
    values = tuple(int(value) for value in parts[1:])
    return values[0], values[1:]


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    smiles: str
    reaction: int
    components: tuple[int, ...]
    inchikey: str
    heavy_atoms: int
    rotatable_bonds: int
    proxy_mean: float = 0.0
    proxy_std: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["components"] = list(self.components)
        return value


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    affinity_probability_binary: float
    affinity_pred_value: float
    score: float
    scorer: str
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.as_dict(),
            "affinity_probability_binary": self.affinity_probability_binary,
            "affinity_pred_value": self.affinity_pred_value,
            "score": self.score,
            "scorer": self.scorer,
            "elapsed_seconds": self.elapsed_seconds,
        }

