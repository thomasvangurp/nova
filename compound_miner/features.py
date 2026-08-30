from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator

from .types import Candidate


DESCRIPTOR_NAMES = (
    "heavy_atoms", "molecular_weight", "logp", "tpsa", "hbd", "hba",
    "rotatable", "rings", "aromatic_rings", "fraction_csp3", "formal_charge",
)


def molecule_features(smiles: str, fingerprint_size: int = 1024) -> np.ndarray:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_size).GetFingerprint(molecule)
    bits = np.zeros((fingerprint_size,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    descriptors = np.asarray(
        [
            molecule.GetNumHeavyAtoms(), Descriptors.MolWt(molecule), Crippen.MolLogP(molecule),
            Descriptors.TPSA(molecule), Lipinski.NumHDonors(molecule), Lipinski.NumHAcceptors(molecule),
            Descriptors.NumRotatableBonds(molecule), Lipinski.RingCount(molecule),
            Lipinski.NumAromaticRings(molecule), Lipinski.FractionCSP3(molecule),
            Chem.GetFormalCharge(molecule),
        ],
        dtype=np.float32,
    )
    return np.concatenate((bits, descriptors))


class ComponentEncoder:
    """Leakage-safe smoothed role and pair target encodings."""

    def __init__(self, smoothing: float = 8.0) -> None:
        self.smoothing = smoothing
        self.global_mean = 0.0
        self.role_maps: list[dict[tuple[int, int], float]] = []
        self.pair_maps: list[dict[tuple[int, int, int], float]] = []

    def fit(self, frame: pd.DataFrame) -> "ComponentEncoder":
        self.global_mean = float(frame["score"].mean())
        self.role_maps = []
        for column in ("component_a", "component_b", "component_c"):
            grouped = frame.groupby(["reaction", column])["score"].agg(["mean", "count"])
            encoded = (grouped["mean"] * grouped["count"] + self.global_mean * self.smoothing) / (grouped["count"] + self.smoothing)
            self.role_maps.append({(int(a), int(b)): float(value) for (a, b), value in encoded.items()})
        self.pair_maps = []
        for left, right in (("component_a", "component_b"), ("component_a", "component_c"), ("component_b", "component_c")):
            grouped = frame.groupby(["reaction", left, right])["score"].agg(["mean", "count"])
            encoded = (grouped["mean"] * grouped["count"] + self.global_mean * self.smoothing) / (grouped["count"] + self.smoothing)
            self.pair_maps.append({(int(a), int(b), int(c)): float(value) for (a, b, c), value in encoded.items()})
        return self

    def transform_rows(self, reactions: np.ndarray, components: np.ndarray) -> np.ndarray:
        result = np.full((len(reactions), 6), self.global_mean, dtype=np.float32)
        for row, (reaction, values) in enumerate(zip(reactions, components)):
            result[row, :3] = [mapping.get((int(reaction), int(value)), self.global_mean) for mapping, value in zip(self.role_maps, values)]
            result[row, 3:] = [
                self.pair_maps[0].get((int(reaction), int(values[0]), int(values[1])), self.global_mean),
                self.pair_maps[1].get((int(reaction), int(values[0]), int(values[2])), self.global_mean),
                self.pair_maps[2].get((int(reaction), int(values[1]), int(values[2])), self.global_mean),
            ]
        return result


def frame_components(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    reactions = frame["reaction"].to_numpy(dtype=np.int32)
    components = frame[["component_a", "component_b", "component_c"]].to_numpy(dtype=np.int64)
    return reactions, components

