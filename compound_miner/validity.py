from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, MACCSkeys, rdFingerprintGenerator

from .types import Candidate

BANNED_ATOMS = frozenset({"Se", "Na", "Fe", "Zn", "B", "Br", "Si"})


def _brenk_catalog() -> FilterCatalog.FilterCatalog:
    parameters = FilterCatalog.FilterCatalogParams()
    parameters.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(parameters)


class HistoricalIndex:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.inchikeys: set[str] = set()
        self.fingerprints: list[DataStructs.ExplicitBitVect] = []
        if csv_path:
            frame = pd.read_csv(csv_path, usecols=["SMILES", "InChI_Key"])
            self.inchikeys = set(frame["InChI_Key"].dropna().astype(str))
            molecules = [Chem.MolFromSmiles(value) for value in frame["SMILES"].dropna().astype(str)]
            molecules = [molecule for molecule in molecules if molecule is not None]
            generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            self.fingerprints = list(generator.GetFingerprints(molecules, numThreads=0))

    def is_novel(self, candidate: Candidate, maximum_similarity: float = 0.7) -> bool:
        if candidate.inchikey in self.inchikeys:
            return False
        if not self.fingerprints:
            return True
        molecule = Chem.MolFromSmiles(candidate.smiles)
        fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(molecule)
        return not any(value >= maximum_similarity for value in DataStructs.BulkTanimotoSimilarity(fingerprint, self.fingerprints))


class ValidityGate:
    def __init__(self, history: HistoricalIndex | None = None) -> None:
        self.history = history or HistoricalIndex()
        self.brenk = _brenk_catalog()

    @staticmethod
    def boltz_safe(molecule: Chem.Mol) -> bool:
        hydrated = AllChem.AddHs(molecule)
        return all(
            len(atom.GetSymbol().upper() + str(rank + 1)) <= 4
            for atom, rank in zip(hydrated.GetAtoms(), AllChem.CanonicalRankAtoms(hydrated))
        )

    def accepts(self, candidate: Candidate, check_similarity: bool = True) -> bool:
        molecule = Chem.MolFromSmiles(candidate.smiles)
        if molecule is None or candidate.heavy_atoms < 10:
            return False
        if not 1 <= candidate.rotatable_bonds <= 10:
            return False
        if any(atom.GetSymbol() in BANNED_ATOMS for atom in molecule.GetAtoms()):
            return False
        if self.brenk.HasMatch(molecule) or not self.boltz_safe(molecule):
            return False
        return not check_similarity or self.history.is_novel(candidate)


def maccs_entropy(candidates: Iterable[Candidate]) -> float:
    fingerprints = [np.asarray(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(candidate.smiles)), dtype=float) for candidate in candidates]
    if not fingerprints:
        return 0.0
    probabilities = np.mean(fingerprints, axis=0)
    return float(np.mean([-p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0.0 for p in probabilities]))

