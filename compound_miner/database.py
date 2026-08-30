from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from .types import Candidate


@dataclass(frozen=True, slots=True)
class Reaction:
    reaction_id: int
    smarts: str
    roles: tuple[int, ...]


def database_path() -> Path:
    configured = os.environ.get("NOVA_COMBINATORIAL_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).parents[1] / "combinatorial_db" / "molecules.sqlite"


def readonly_connection(path: Path | str | None = None) -> sqlite3.Connection:
    resolved = Path(path or database_path()).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"combinatorial database not found: {resolved}")
    return sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)


class ReactionDatabase:
    """In-memory reaction/component index backed by an immutable SQLite database."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or database_path()).expanduser().resolve()
        with readonly_connection(self.path) as connection:
            rows = connection.execute(
                "SELECT rxn_id, smarts, roleA, roleB, roleC FROM reactions ORDER BY CAST(rxn_id AS INTEGER)"
            ).fetchall()
            molecule_rows = connection.execute(
                "SELECT mol_id, smiles, role_mask FROM molecules"
            ).fetchall()
        self.reactions: dict[int, Reaction] = {}
        for row in rows:
            reaction_id = int(row[0])
            roles = tuple(int(role) for role in row[2:] if role not in (None, 0))
            self.reactions[reaction_id] = Reaction(reaction_id, str(row[1]), roles)
        self.molecules = {int(row[0]): (str(row[1]), int(row[2])) for row in molecule_rows}
        distinct_roles = {role for reaction in self.reactions.values() for role in reaction.roles}
        self.role_ids = {
            role: tuple(mol_id for mol_id, (_, mask) in self.molecules.items() if mask & role == role)
            for role in distinct_roles
        }
        self._compiled = {
            reaction_id: AllChem.ReactionFromSmarts(reaction.smarts)
            for reaction_id, reaction in self.reactions.items()
            if reaction_id not in (1, 3, 5)
        }

    def random_components(self, reaction_id: int, rng: random.Random) -> tuple[int, ...]:
        reaction = self.reactions[reaction_id]
        return tuple(rng.choice(self.role_ids[role]) for role in reaction.roles)

    def sample_component_sets(
        self, reaction_id: int, count: int, rng: random.Random
    ) -> Iterable[tuple[int, ...]]:
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(seen) < count and attempts < count * 5:
            attempts += 1
            components = self.random_components(reaction_id, rng)
            if components not in seen:
                seen.add(components)
                yield components

    def _ordered_smiles(self, reaction_id: int, components: tuple[int, ...]) -> tuple[str, ...] | None:
        reaction = self.reactions[reaction_id]
        if len(components) != len(reaction.roles):
            return None
        rows = [self.molecules.get(component) for component in components]
        if any(row is None for row in rows):
            return None
        # Search the few possible permutations. This exactly respects role masks and
        # avoids relying on database row order.
        import itertools

        for permutation in itertools.permutations(rows):
            if all(mask & role == role for (_, mask), role in zip(permutation, reaction.roles)):
                return tuple(smiles for smiles, _ in permutation)
        return None

    @staticmethod
    def _triazole(azide: str, alkyne: str) -> str | None:
        try:
            left = Chem.RWMol(Chem.MolFromSmiles(azide))
            right = Chem.RWMol(Chem.MolFromSmiles(alkyne))
            a1 = next(i for i, atom in enumerate(left.GetAtoms()) if atom.GetSymbol() == "*" and atom.GetIsotope() == 1)
            a2 = next(i for i, atom in enumerate(right.GetAtoms()) if atom.GetSymbol() == "*" and atom.GetIsotope() == 2)
            n1 = left.GetAtomWithIdx(a1).GetNeighbors()[0].GetIdx()
            n2 = right.GetAtomWithIdx(a2).GetNeighbors()[0].GetIdx()
            combined = Chem.RWMol(left)
            mapping: dict[int, int] = {}
            for index, atom in enumerate(right.GetAtoms()):
                if index != a2:
                    mapping[index] = combined.AddAtom(atom)
            for bond in right.GetBonds():
                begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a2 not in (begin, end):
                    combined.AddBond(mapping[begin], mapping[end], bond.GetBondType())
            combined.RemoveAtom(a1)
            combined.AddBond(n1 - int(n1 > a1), mapping[n2] - int(mapping[n2] > a1), Chem.BondType.SINGLE)
            Chem.SanitizeMol(combined)
            return Chem.MolToSmiles(combined)
        except (StopIteration, ValueError, RuntimeError, AttributeError):
            return None

    @staticmethod
    def _smarts(smarts: str, *smiles: str) -> str | None:
        try:
            reaction = AllChem.ReactionFromSmarts(smarts)
            molecules = tuple(Chem.MolFromSmiles(value) for value in smiles)
            if any(molecule is None for molecule in molecules):
                return None
            products = reaction.RunReactants(molecules)
            if not products:
                return None
            product = products[0][0]
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product)
        except (ValueError, RuntimeError, IndexError):
            return None

    def construct(self, reaction_id: int, components: tuple[int, ...]) -> str | None:
        ordered = self._ordered_smiles(reaction_id, components)
        if ordered is None:
            return None
        if reaction_id == 1:
            return self._triazole(*ordered)
        if reaction_id == 3:
            intermediate = self._triazole(ordered[0], ordered[1])
            return None if intermediate is None else self._smarts("[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]", intermediate, ordered[2])
        if reaction_id == 5:
            intermediate = self._smarts("[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]", ordered[0], ordered[1])
            return None if intermediate is None else self._smarts("[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]", intermediate, ordered[2])
        return self._smarts(self.reactions[reaction_id].smarts, *ordered)

    def candidate(self, reaction_id: int, components: tuple[int, ...]) -> Candidate | None:
        smiles = self.construct(reaction_id, components)
        if not smiles:
            return None
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        canonical = Chem.MolToSmiles(molecule)
        name = ":".join(("rxn", str(reaction_id), *(str(value) for value in components)))
        return Candidate(
            name=name,
            smiles=canonical,
            reaction=reaction_id,
            components=components,
            inchikey=Chem.MolToInchiKey(molecule),
            heavy_atoms=molecule.GetNumHeavyAtoms(),
            rotatable_bonds=int(Descriptors.NumRotatableBonds(molecule)),
        )

