from __future__ import annotations

import csv
import dataclasses
import functools
import gzip
import hashlib
import heapq
import json
import math
import shutil
import sqlite3
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .targets import TARGETS
from .selection import select_submission
from .types import Candidate, ScoredCandidate


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {value: index for index, value in enumerate(AMINO_ACIDS)}


@dataclasses.dataclass(frozen=True, slots=True)
class StudentExample:
    name: str
    smiles: str
    target: str
    protein_sequence: str | None = None
    reaction: int = 0
    components: tuple[int, ...] = ()
    objective: float | None = None
    experimental_affinity: float | None = None
    boltz_affinity: float | None = None
    binder_probability: float | None = None
    weight: float = 1.0
    source: str = "unknown"


@dataclasses.dataclass(slots=True)
class GraphBatch:
    atom_features: torch.Tensor
    edge_features: torch.Tensor
    edge_sources: torch.Tensor
    edge_targets: torch.Tensor
    graph_index: torch.Tensor
    graph_count: int

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            self.atom_features.to(device, non_blocking=True),
            self.edge_features.to(device, non_blocking=True),
            self.edge_sources.to(device, non_blocking=True),
            self.edge_targets.to(device, non_blocking=True),
            self.graph_index.to(device, non_blocking=True),
            self.graph_count,
        )

    def pin_memory(self) -> "GraphBatch":
        return GraphBatch(
            self.atom_features.pin_memory(),
            self.edge_features.pin_memory(),
            self.edge_sources.pin_memory(),
            self.edge_targets.pin_memory(),
            self.graph_index.pin_memory(),
            self.graph_count,
        )


def _bounded(value: int, maximum: int) -> int:
    return max(0, min(int(value), maximum))


def molecule_graph(smiles: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    atoms = np.asarray(
        [
            [
                _bounded(atom.GetAtomicNum(), 118),
                _bounded(atom.GetDegree(), 8),
                _bounded(atom.GetFormalCharge() + 5, 10),
                _bounded(int(atom.GetChiralTag()), 4),
                _bounded(int(atom.GetHybridization()), 8),
                _bounded(atom.GetTotalNumHs(), 8),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
            ]
            for atom in molecule.GetAtoms()
        ],
        dtype=np.int64,
    )
    edges: list[tuple[int, int]] = []
    edge_features: list[list[int]] = []
    for bond in molecule.GetBonds():
        feature = [
            _bounded(int(bond.GetBondType()), 4),
            _bounded(int(bond.GetStereo()), 6),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((left, right), (right, left)))
        edge_features.extend((feature, feature))
    if edges:
        sources, targets = np.asarray(edges, dtype=np.int64).T
        bond_array = np.asarray(edge_features, dtype=np.int64)
    else:
        sources = np.empty(0, dtype=np.int64)
        targets = np.empty(0, dtype=np.int64)
        bond_array = np.empty((0, 4), dtype=np.int64)
    return atoms, bond_array, sources, targets


def protein_features(
    sequence: str,
    msa: str | Path | None = None,
    esm_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Compact, cacheable target representation; safe to reuse for every ligand."""
    clean = "".join(value for value in sequence.upper() if value in AA_INDEX)
    # The first 64 values are dependency-free sequence/MSA statistics. The
    # remaining 480 are the cached ESM-2 t12 representation when available.
    result = np.zeros(544, dtype=np.float32)
    result[0] = math.log1p(len(clean)) / 10.0
    for value in clean:
        result[1 + AA_INDEX[value]] += 1.0
    if clean:
        result[1:21] /= len(clean)
    for left, right in zip(clean, clean[1:]):
        digest = hashlib.blake2b(f"{left}{right}".encode(), digest_size=2).digest()
        result[21 + int.from_bytes(digest, "little") % 32] += 1.0
    if len(clean) > 1:
        result[21:53] /= len(clean) - 1
    groups = ("AILMFWVY", "STNQ", "KRH", "DE", "CGP")
    for index, group in enumerate(groups):
        result[53 + index] = sum(value in group for value in clean) / max(1, len(clean))
    if msa:
        path = Path(msa)
        if path.exists():
            sequences: list[str] = []
            current: list[str] = []
            for line in path.read_text().splitlines():
                if line.startswith(">"):
                    if current:
                        sequences.append("".join(current))
                    current = []
                else:
                    current.append("".join(value for value in line if not value.islower()))
                if len(sequences) >= 256:
                    break
            if current and len(sequences) < 256:
                sequences.append("".join(current))
            result[58] = math.log1p(len(sequences)) / 10.0
            if sequences:
                lengths = np.asarray([len(value.replace("-", "")) for value in sequences])
                result[59] = float(np.mean(lengths)) / max(1, len(clean))
                result[60] = float(np.std(lengths)) / max(1, len(clean))
                result[61] = float(np.mean([value.count("-") / max(1, len(value)) for value in sequences]))
    if esm_embedding is not None:
        embedding = np.asarray(esm_embedding, dtype=np.float32).reshape(-1)
        if len(embedding) != 480:
            raise ValueError(f"expected a 480-dimensional ESM-2 embedding, got {len(embedding)}")
        result[64:] = embedding
    return result


def cache_protein_features(target: str, output: str | Path, msa: str | Path | None = None) -> Path:
    sequence = TARGETS[target]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        target=np.asarray(target),
        sequence_sha256=np.asarray(hashlib.sha256(sequence.encode()).hexdigest()),
        features=protein_features(sequence, msa),
    )
    return destination


def cache_esm2_embeddings(
    targets: dict[str, str], output: str | Path, device: str | None = None, batch_size: int = 8
) -> Path:
    """Encode each protein once; inference only ships the small cached vectors."""
    import esm

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.eval().to(selected_device)
    converter = alphabet.get_batch_converter()
    identifiers = sorted(targets)
    embeddings: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(identifiers), batch_size):
            subset = identifiers[start : start + batch_size]
            truncated = {identifier: targets[identifier][:1022] for identifier in subset}
            _, _, tokens = converter([(identifier, truncated[identifier]) for identifier in subset])
            output_values = model(tokens.to(selected_device), repr_layers=[12], return_contacts=False)["representations"][12]
            for row, identifier in enumerate(subset):
                length = len(truncated[identifier])
                embeddings[identifier] = output_values[row, 1 : length + 1].mean(0).float().cpu().numpy()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **embeddings)
    return destination


def load_esm2_embeddings(path: str | Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    with np.load(path) as values:
        return {name: values[name].astype(np.float32) for name in values.files}


def _graph_batch(smiles: Sequence[str]) -> GraphBatch:
    atoms: list[np.ndarray] = []
    bonds: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    graph_indices: list[np.ndarray] = []
    offset = 0
    for graph_index, value in enumerate(smiles):
        atom, bond, source, target = molecule_graph(value)
        atoms.append(atom)
        bonds.append(bond)
        sources.append(source + offset)
        targets.append(target + offset)
        graph_indices.append(np.full(len(atom), graph_index, dtype=np.int64))
        offset += len(atom)
    return GraphBatch(
        torch.from_numpy(np.concatenate(atoms)),
        torch.from_numpy(np.concatenate(bonds)),
        torch.from_numpy(np.concatenate(sources)),
        torch.from_numpy(np.concatenate(targets)),
        torch.from_numpy(np.concatenate(graph_indices)),
        len(smiles),
    )


class DistilledDataset(Dataset[StudentExample]):
    def __init__(self, examples: Sequence[StudentExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> StudentExample:
        return self.examples[index]


def collate_examples(
    examples: Sequence[StudentExample], esm_embeddings: dict[str, np.ndarray] | None = None
) -> dict[str, Any]:
    components = np.full((len(examples), 3), -1, dtype=np.int64)
    for row, example in enumerate(examples):
        components[row, : min(3, len(example.components))] = example.components[:3]

    def labels(name: str) -> torch.Tensor:
        return torch.tensor(
            [float("nan") if getattr(value, name) is None else float(getattr(value, name)) for value in examples],
            dtype=torch.float32,
        )

    embedding_map = esm_embeddings or {}
    target_matrix = np.vstack(
        [
            protein_features(
                value.protein_sequence or TARGETS[value.target],
                esm_embedding=embedding_map.get(value.target),
            )
            for value in examples
        ]
    )
    return {
        "examples": examples,
        "graphs": _graph_batch([value.smiles for value in examples]),
        "protein": torch.from_numpy(target_matrix),
        "reaction": torch.tensor([value.reaction for value in examples], dtype=torch.long),
        "components": torch.from_numpy(components),
        "objective": labels("objective"),
        "experimental_affinity": labels("experimental_affinity"),
        "boltz_affinity": labels("boltz_affinity"),
        "binder_probability": labels("binder_probability"),
        "weight": torch.tensor([value.weight for value in examples], dtype=torch.float32),
    }


class DistilledOracle(nn.Module):
    """Diffusion-free target-conditioned molecular graph scorer."""

    atom_cardinalities = (119, 9, 11, 5, 9, 9, 2, 2)
    bond_cardinalities = (5, 7, 2, 2)

    def __init__(self, hidden: int = 192, layers: int = 4, component_buckets: int = 16384) -> None:
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.component_buckets = component_buckets
        self.atom_embeddings = nn.ModuleList(nn.Embedding(size, hidden) for size in self.atom_cardinalities)
        self.bond_embeddings = nn.ModuleList(nn.Embedding(size, hidden) for size in self.bond_cardinalities)
        self.message_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(layers)
        )
        self.update_layers = nn.ModuleList(nn.GRUCell(hidden, hidden) for _ in range(layers))
        self.protein_encoder = nn.Sequential(nn.Linear(544, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.reaction_embedding = nn.Embedding(6, hidden)
        self.component_embeddings = nn.ModuleList(nn.Embedding(component_buckets, hidden) for _ in range(3))
        self.context_projection = nn.Sequential(nn.Linear(hidden * 5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.trunk = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
        )
        self.objective_head = nn.Linear(hidden, 2)
        self.experimental_head = nn.Linear(hidden, 1)
        self.boltz_affinity_head = nn.Linear(hidden, 2)
        self.binder_head = nn.Linear(hidden, 1)

    @staticmethod
    def _embed(features: torch.Tensor, embeddings: nn.ModuleList) -> torch.Tensor:
        result = embeddings[0](features[:, 0])
        for index in range(1, features.shape[1]):
            result = result + embeddings[index](features[:, index])
        return result

    def encode_ligand(self, graphs: GraphBatch) -> torch.Tensor:
        hidden = self._embed(graphs.atom_features, self.atom_embeddings)
        edge = self._embed(graphs.edge_features, self.bond_embeddings)
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            aggregated = torch.zeros_like(hidden)
            if graphs.edge_sources.numel():
                message = message_layer(torch.cat((hidden[graphs.edge_sources], edge), dim=-1))
                aggregated.index_add_(0, graphs.edge_targets, message)
            hidden = update_layer(aggregated, hidden)
        pooled_mean = torch.zeros((graphs.graph_count, self.hidden), device=hidden.device, dtype=hidden.dtype)
        pooled_mean.index_add_(0, graphs.graph_index, hidden)
        counts = torch.bincount(graphs.graph_index, minlength=graphs.graph_count).clamp_min(1).unsqueeze(1)
        pooled_mean = pooled_mean / counts
        pooled_max = torch.full_like(pooled_mean, -torch.inf)
        pooled_max.scatter_reduce_(0, graphs.graph_index[:, None].expand_as(hidden), hidden, reduce="amax", include_self=True)
        return 0.5 * (pooled_mean + pooled_max)

    def encode_context(self, protein: torch.Tensor, reaction: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
        values = [self.protein_encoder(protein), self.reaction_embedding(reaction.clamp(0, 5))]
        for role, embedding in enumerate(self.component_embeddings):
            # Reserve bucket zero for missing components.
            indices = torch.where(components[:, role] < 0, 0, components[:, role].remainder(self.component_buckets - 1) + 1)
            values.append(embedding(indices))
        return self.context_projection(torch.cat(values, dim=-1))

    def forward(
        self, graphs: GraphBatch, protein: torch.Tensor, reaction: torch.Tensor, components: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        ligand = self.encode_ligand(graphs)
        context = self.encode_context(protein, reaction, components)
        hidden = self.trunk(torch.cat((ligand, context, ligand * torch.sigmoid(context)), dim=-1))
        objective = self.objective_head(hidden)
        boltz_affinity = self.boltz_affinity_head(hidden)
        return {
            "objective_mean": objective[:, 0],
            "objective_logvar": objective[:, 1].clamp(-8, 5),
            "experimental_affinity": self.experimental_head(hidden).squeeze(-1),
            "boltz_affinity_mean": boltz_affinity[:, 0],
            "boltz_affinity_logvar": boltz_affinity[:, 1].clamp(-8, 5),
            "binder_logit": self.binder_head(hidden).squeeze(-1),
        }


def _scorer_weight(scorer: str) -> float:
    lowered = scorer.lower()
    if "validator" in lowered or ("r3" in lowered and "as100" in lowered and "ad3" in lowered):
        return 1.0
    if "r1-s20" in lowered:
        return 0.55
    if lowered.startswith("nim:"):
        return 0.20
    return 0.75


def examples_from_archive(path: str | Path, target: str = "P40261") -> list[StudentExample]:
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro&immutable=1", uri=True) as connection:
        rows = connection.execute(
            """SELECT name,smiles,reaction,components,score,affinity_pred_value,
                      affinity_probability_binary,scorer
               FROM candidates WHERE score IS NOT NULL"""
        ).fetchall()
    return [
        StudentExample(
            name=row[0], smiles=row[1], target=target, reaction=int(row[2]),
            components=tuple(json.loads(row[3])), objective=float(row[4]),
            boltz_affinity=None if row[5] is None else float(row[5]),
            binder_probability=None if row[6] is None else float(row[6]),
            weight=_scorer_weight(str(row[7] or "unknown")), source=str(row[7] or "archive"),
        )
        for row in rows
    ]


def examples_from_boltz_json(paths: Iterable[str | Path], target: str = "P40261") -> list[StudentExample]:
    deduplicated: dict[str, StudentExample] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        inference = payload.get("inference") or {}
        exact = not inference or (
            inference.get("recycling_steps", 3) >= 3
            and inference.get("sampling_steps", 100) >= 100
            and inference.get("diffusion_samples_affinity", 3) >= 3
        )
        for row in payload["scores"]:
            example = StudentExample(
                name=row["name"], smiles=row["smiles"], target=target, reaction=int(row["reaction"]),
                components=tuple(int(value) for value in row.get("components", ())),
                objective=float(row["score"]), boltz_affinity=float(row["affinity_pred_value"]),
                binder_probability=float(row["affinity_probability_binary"]),
                weight=1.0 if exact else 0.55, source=str(row.get("scorer", Path(path).name)),
            )
            previous = deduplicated.get(example.name)
            if previous is None or example.weight > previous.weight:
                deduplicated[example.name] = example
    return list(deduplicated.values())


def examples_from_public_csv(path: str | Path, target: str = "P40261") -> list[StudentExample]:
    grouped: dict[tuple[str, str], list[tuple[float, dict[str, str]]]] = {}
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                affinity = float(row["pchembl_value"])
                smiles = row["canonical_smiles"]
                if not smiles or not math.isfinite(affinity) or Chem.MolFromSmiles(smiles) is None:
                    continue
                accession = row.get("target_accession") or target
                grouped.setdefault((accession, smiles), []).append((affinity, row))
            except (KeyError, TypeError, ValueError):
                continue
    examples = []
    for (accession, smiles), measurements in grouped.items():
        row = measurements[0][1]
        examples.append(
            StudentExample(
                name=row.get("molecule_chembl_id", f"public:{len(examples)}"), smiles=smiles,
                target=accession, protein_sequence=row.get("protein_sequence") or TARGETS.get(accession),
                experimental_affinity=float(np.median([value for value, _ in measurements])),
                weight=min(3.0, math.sqrt(len(measurements))), source="chembl",
            )
        )
    return examples


def examples_from_bindingdb_tsv(path: str | Path, limit: int | None = None) -> list[StudentExample]:
    """Read the official BindingDB TSV without depending on one release's column order."""
    affinity_columns = ("Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)")
    results: dict[tuple[str, str], list[tuple[float, dict[str, str]]]] = {}
    with Path(path).open(errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        smiles_name = next((value for value in ("Ligand SMILES", "SMILES") if value in fields), None)
        accession_name = next(
            (
                value
                for value in (
                    "UniProt (SwissProt) Primary ID of Target Chain 1",
                    "UniProt (SwissProt) Primary ID of Target Chain",
                    "UniProt ID",
                )
                if value in fields
            ),
            None,
        )
        trembl_name = next(
            (value for value in ("UniProt (TrEMBL) Primary ID of Target Chain 1", "UniProt (TrEMBL) Primary ID of Target Chain") if value in fields),
            None,
        )
        chain_count_name = "Number of Protein Chains in Target (>1 implies a multichain complex)"
        sequence_name = next(
            (value for value in ("BindingDB Target Chain Sequence 1", "Target Chain Sequence", "Target Sequence") if value in fields), None
        )
        if not smiles_name or not (accession_name or trembl_name) or not sequence_name:
            raise ValueError("BindingDB TSV is missing SMILES, UniProt accession, or target sequence columns")
        for row in reader:
            if chain_count_name in fields and row.get(chain_count_name) not in ("", "1"):
                continue
            smiles = row.get(smiles_name, "").strip()
            accession = (
                row.get(accession_name, "") if accession_name else ""
            ).strip() or (row.get(trembl_name, "") if trembl_name else "").strip()
            accession = accession.split(",")[0].strip()
            sequence = row.get(sequence_name, "").replace(" ", "").strip()
            if not smiles or not accession or not sequence or Chem.MolFromSmiles(smiles) is None:
                continue
            values = []
            for column in affinity_columns:
                raw = row.get(column, "").strip()
                if not raw or raw[0] in "<>":
                    continue
                try:
                    nanomolar = float(raw)
                    if nanomolar > 0:
                        values.append(9.0 - math.log10(nanomolar))
                except ValueError:
                    continue
            if values:
                results.setdefault((accession, smiles), []).append((float(np.median(values)), {"sequence": sequence}))
            if limit is not None and len(results) >= limit:
                break
    examples = []
    for (accession, smiles), measurements in results.items():
        examples.append(
            StudentExample(
                name=f"bindingdb:{accession}:{len(examples)}", smiles=smiles, target=accession,
                protein_sequence=measurements[0][1]["sequence"],
                experimental_affinity=float(np.median([value for value, _ in measurements])),
                weight=min(3.0, math.sqrt(len(measurements))), source="bindingdb",
            )
        )
    return examples


def prepare_bindingdb_balanced(
    path: str | Path,
    output: str | Path,
    maximum_per_target: int = 64,
    minimum_per_target: int = 8,
) -> dict[str, Any]:
    """Create a deterministic target-balanced subset without loading 3M graphs."""
    affinity_columns = ("Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)")
    # target -> (max-heap by deterministic hash, selected records by SMILES)
    states: dict[str, tuple[list[tuple[int, str]], dict[str, dict[str, Any]]]] = {}
    rows_seen = 0
    usable_seen = 0
    with Path(path).open(errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows_seen += 1
            if row.get("Number of Protein Chains in Target (>1 implies a multichain complex)") not in ("", "1"):
                continue
            smiles = row.get("Ligand SMILES", "").strip()
            sequence = row.get("BindingDB Target Chain Sequence 1", "").replace(" ", "").strip().upper()
            accession = (
                row.get("UniProt (SwissProt) Primary ID of Target Chain 1", "").strip()
                or row.get("UniProt (TrEMBL) Primary ID of Target Chain 1", "").strip()
            ).split(",")[0]
            if not smiles or not sequence or not accession:
                continue
            values = []
            for column in affinity_columns:
                raw = row.get(column, "").strip()
                if not raw or raw[0] in "<>":
                    continue
                try:
                    nanomolar = float(raw)
                    if nanomolar > 0:
                        values.append(9.0 - math.log10(nanomolar))
                except ValueError:
                    continue
            if not values:
                continue
            usable_seen += 1
            affinity = float(np.median(values))
            heap, records = states.setdefault(accession, ([], {}))
            existing = records.get(smiles)
            if existing is not None:
                existing["sum"] += affinity
                existing["count"] += 1
                continue
            digest = hashlib.blake2b(f"{accession}\0{smiles}".encode(), digest_size=8).digest()
            rank = int.from_bytes(digest, "little")
            if len(records) >= maximum_per_target:
                worst_rank, worst_smiles = heap[0]
                if rank >= -worst_rank:
                    continue
                heapq.heappop(heap)
                del records[worst_smiles]
            records[smiles] = {"rank": rank, "sequence": sequence, "sum": affinity, "count": 1}
            heapq.heappush(heap, (-rank, smiles))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if destination.suffix == ".gz" else open
    fieldnames = ("target_accession", "protein_sequence", "molecule_chembl_id", "canonical_smiles", "pchembl_value", "measurement_count", "source")
    target_count = 0
    interaction_count = 0
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with opener(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for accession in sorted(states):
            records = states[accession][1]
            if len(records) < minimum_per_target:
                continue
            target_count += 1
            for index, (smiles, record) in enumerate(sorted(records.items())):
                writer.writerow(
                    {
                        "target_accession": accession,
                        "protein_sequence": record["sequence"],
                        "molecule_chembl_id": f"bindingdb:{accession}:{index}",
                        "canonical_smiles": smiles,
                        "pchembl_value": record["sum"] / record["count"],
                        "measurement_count": record["count"],
                        "source": "bindingdb",
                    }
                )
                interaction_count += 1
    temporary.replace(destination)
    return {
        "rows_seen": rows_seen,
        "usable_measurements": usable_seen,
        "targets": target_count,
        "interactions": interaction_count,
        "maximum_per_target": maximum_per_target,
        "minimum_per_target": minimum_per_target,
        "output": str(destination),
    }


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "nova-compound-miner/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_chembl_target(accession: str, output: str | Path) -> dict[str, Any]:
    target_query = urllib.parse.urlencode({"target_components__accession": accession, "limit": 100})
    targets = _request_json(f"https://www.ebi.ac.uk/chembl/api/data/target.json?{target_query}")["targets"]
    single = [value for value in targets if value.get("target_type") == "SINGLE PROTEIN"]
    if not single:
        raise RuntimeError(f"no ChEMBL single-protein target found for {accession}")
    target_id = single[0]["target_chembl_id"]
    uniprot = _request_json(f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(accession)}.json")
    sequence = uniprot["sequence"]["value"]
    activity_query = urllib.parse.urlencode({"target_chembl_id": target_id, "limit": 1000})
    activities = _request_json(f"https://www.ebi.ac.uk/chembl/api/data/activity.json?{activity_query}")["activities"]
    molecule_ids = sorted({value["molecule_chembl_id"] for value in activities if value.get("molecule_chembl_id")})
    structures: dict[str, str] = {}
    for start in range(0, len(molecule_ids), 100):
        molecule_query = urllib.parse.urlencode({"molecule_chembl_id__in": ",".join(molecule_ids[start : start + 100]), "limit": 100})
        payload = _request_json(f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?{molecule_query}")
        for molecule in payload["molecules"]:
            structure = molecule.get("molecule_structures") or {}
            if structure.get("canonical_smiles"):
                structures[molecule["molecule_chembl_id"]] = structure["canonical_smiles"]
    rows: list[dict[str, Any]] = []
    for activity in activities:
        molecule_id = activity.get("molecule_chembl_id")
        smiles = structures.get(str(molecule_id))
        pchembl = activity.get("pchembl_value")
        if not smiles or pchembl is None or activity.get("standard_relation") not in ("=", None):
            continue
        rows.append(
            {
                "target_chembl_id": target_id,
                "target_accession": accession,
                "protein_sequence": sequence,
                "molecule_chembl_id": molecule_id,
                "canonical_smiles": smiles,
                "standard_type": activity.get("standard_type"),
                "standard_relation": activity.get("standard_relation"),
                "standard_value": activity.get("standard_value"),
                "standard_units": activity.get("standard_units"),
                "pchembl_value": pchembl,
                "assay_chembl_id": activity.get("assay_chembl_id"),
            }
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return {"accession": accession, "target_chembl_id": target_id, "activities": len(activities), "usable_rows": len(rows), "output": str(destination)}


def scaffold_split(examples: Sequence[StudentExample], validation_fraction: float = 0.2) -> tuple[list[StudentExample], list[StudentExample]]:
    train: list[StudentExample] = []
    validation: list[StudentExample] = []
    cutoff = int(validation_fraction * 10_000)
    for example in examples:
        molecule = Chem.MolFromSmiles(example.smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or Chem.MolToSmiles(molecule)
        bucket = int.from_bytes(hashlib.sha256(scaffold.encode()).digest()[:4], "little") % 10_000
        (validation if bucket < cutoff else train).append(example)
    if not train or not validation:
        midpoint = max(1, int(len(examples) * (1 - validation_fraction)))
        ordered = sorted(examples, key=lambda value: hashlib.sha256(value.name.encode()).digest())
        train, validation = ordered[:midpoint], ordered[midpoint:]
    return train, validation


def target_split(
    examples: Sequence[StudentExample], validation_fraction: float = 0.2
) -> tuple[list[StudentExample], list[StudentExample]]:
    """Hold out whole proteins to measure genuine cross-target transfer."""
    targets = sorted({value.target for value in examples})
    if len(targets) < 2:
        return scaffold_split(examples, validation_fraction)
    cutoff = int(validation_fraction * 10_000)
    validation_targets = {
        target
        for target in targets
        if int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little") % 10_000 < cutoff
    }
    if not validation_targets or validation_targets == set(targets):
        count = max(1, min(len(targets) - 1, round(len(targets) * validation_fraction)))
        validation_targets = set(targets[:count])
    train = [value for value in examples if value.target not in validation_targets]
    validation = [value for value in examples if value.target in validation_targets]
    return train, validation


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value.to(device) if isinstance(value, GraphBatch) else value
        for key, value in batch.items()
    }


def _gaussian_loss(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
        return mean.sum() * 0
    residual = mean[mask] - target[mask]
    return torch.mean(weight[mask] * (0.5 * torch.exp(-logvar[mask]) * residual.square() + 0.5 * logvar[mask]))


def _ranking_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    prediction, target = prediction[mask], target[mask]
    if len(target) < 2:
        return prediction.sum() * 0
    delta = target[:, None] - target[None, :]
    useful = delta.abs() > 0.002
    if not useful.any():
        return prediction.sum() * 0
    predicted_delta = prediction[:, None] - prediction[None, :]
    return torch.nn.functional.softplus(-delta.sign()[useful] * predicted_delta[useful] / 0.02).mean()


def _run_epoch(model: DistilledOracle, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> float:
    model.train(optimizer is not None)
    total = 0.0
    batches = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(batch["graphs"], batch["protein"], batch["reaction"], batch["components"])
            loss = _gaussian_loss(output["objective_mean"], output["objective_logvar"], batch["objective"], batch["weight"])
            public_mask = torch.isfinite(batch["experimental_affinity"])
            if public_mask.any():
                loss = loss + torch.nn.functional.smooth_l1_loss(
                    output["experimental_affinity"][public_mask], batch["experimental_affinity"][public_mask]
                )
            affinity_loss = _gaussian_loss(
                output["boltz_affinity_mean"], output["boltz_affinity_logvar"], batch["boltz_affinity"], batch["weight"]
            )
            loss = loss + 0.35 * affinity_loss + 0.35 * _ranking_loss(output["objective_mean"], batch["objective"])
            binder_mask = torch.isfinite(batch["binder_probability"])
            if binder_mask.any():
                loss = loss + 0.20 * torch.nn.functional.binary_cross_entropy_with_logits(
                    output["binder_logit"][binder_mask], batch["binder_probability"][binder_mask]
                )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += float(loss.detach())
        batches += 1
    return total / max(1, batches)


def train_distilled_ensemble(
    public_examples: Sequence[StudentExample],
    objective_examples: Sequence[StudentExample],
    output: str | Path,
    ensemble_size: int = 5,
    pretrain_epochs: int = 30,
    finetune_epochs: int = 80,
    batch_size: int = 64,
    hidden: int = 192,
    layers: int = 4,
    device: str | None = None,
    esm_embeddings_path: str | Path | None = None,
    workers: int = 0,
) -> dict[str, Any]:
    if len(objective_examples) < 8:
        raise ValueError("at least eight Boltz-labelled examples are required")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    public_train, public_validation = target_split(public_examples) if public_examples else ([], [])
    objective_train, objective_validation = scaffold_split(objective_examples)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    esm_embeddings = load_esm2_embeddings(esm_embeddings_path)
    collator = functools.partial(collate_examples, esm_embeddings=esm_embeddings)
    members = []
    base_members = []
    for member in range(ensemble_size):
        torch.manual_seed(68 + member)
        model = DistilledOracle(hidden=hidden, layers=layers).to(selected_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        if public_train:
            generator = torch.Generator().manual_seed(1000 + member)
            loader = DataLoader(
                DistilledDataset(public_train), batch_size=batch_size, shuffle=True,
                generator=generator, collate_fn=collator, num_workers=workers,
                persistent_workers=workers > 0, pin_memory=selected_device.type == "cuda",
            )
            for _ in range(pretrain_epochs):
                _run_epoch(model, loader, optimizer, selected_device)
            base_checkpoint = destination / f"base_member_{member}.pt"
            torch.save(
                {"version": 1, "hidden": hidden, "layers": layers, "state_dict": model.state_dict()},
                base_checkpoint,
            )
            base_members.append(str(base_checkpoint.name))
            for module in (
                model.atom_embeddings, model.bond_embeddings, model.message_layers,
                model.update_layers, model.protein_encoder, model.experimental_head,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-5)
        generator = torch.Generator().manual_seed(2000 + member)
        loader = DataLoader(
            DistilledDataset(objective_train), batch_size=min(batch_size, len(objective_train)),
            shuffle=True, generator=generator, collate_fn=collator,
            num_workers=workers, persistent_workers=workers > 0,
            pin_memory=selected_device.type == "cuda",
        )
        for _ in range(finetune_epochs):
            _run_epoch(model, loader, optimizer, selected_device)
        checkpoint = destination / f"member_{member}.pt"
        torch.save(
            {"version": 1, "hidden": hidden, "layers": layers, "state_dict": model.state_dict(), "target": objective_examples[0].target},
            checkpoint,
        )
        members.append(str(checkpoint.name))
    manifest = {
        "version": 1,
        "members": members,
        "base_members": base_members,
        "public_train": len(public_train),
        "public_validation": len(public_validation),
        "objective_train": len(objective_train),
        "objective_validation": len(objective_validation),
        "objective_split": "murcko-scaffold-sha256",
        "public_split": "whole-target-sha256",
    }
    if esm_embeddings_path:
        shutil.copy2(esm_embeddings_path, destination / "protein_embeddings.npz")
        manifest["protein_embeddings"] = "protein_embeddings.npz"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    metrics = evaluate_distilled(destination, objective_validation, batch_size=batch_size, device=str(selected_device))
    if public_validation:
        metrics.update(evaluate_public_affinity(
            destination, public_validation, batch_size=batch_size, device=str(selected_device),
            training_baseline=float(np.mean([
                value.experimental_affinity for value in public_train
                if value.experimental_affinity is not None
            ])),
        ))
    return {**manifest, **metrics, "output": str(destination)}


def _load_models(path: str | Path, device: torch.device) -> list[DistilledOracle]:
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    models = []
    for name in manifest["members"]:
        payload = torch.load(directory / name, map_location=device, weights_only=True)
        model = DistilledOracle(hidden=int(payload["hidden"]), layers=int(payload["layers"])).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


def predict_distilled(
    model_path: str | Path, examples: Sequence[StudentExample], batch_size: int = 256, device: str | None = None
) -> list[dict[str, float]]:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = _load_models(model_path, selected_device)
    model_directory = Path(model_path)
    manifest = json.loads((model_directory / "manifest.json").read_text())
    embeddings_name = manifest.get("protein_embeddings")
    esm_embeddings = load_esm2_embeddings(model_directory / embeddings_name if embeddings_name else None)
    collator = functools.partial(collate_examples, esm_embeddings=esm_embeddings)
    predictions: list[dict[str, float]] = []
    loader = DataLoader(DistilledDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, selected_device)
            member_means = []
            member_variances = []
            for model in models:
                output = model(batch["graphs"], batch["protein"], batch["reaction"], batch["components"])
                member_means.append(output["objective_mean"])
                member_variances.append(torch.exp(output["objective_logvar"]))
            means = torch.stack(member_means)
            variances = torch.stack(member_variances)
            ensemble_mean = means.mean(0)
            total_variance = variances.mean(0) + means.var(0, correction=0)
            for mean, variance in zip(ensemble_mean.cpu(), total_variance.cpu()):
                predictions.append({"mean": float(mean), "std": float(torch.sqrt(variance.clamp_min(1e-12)))})
    return predictions


def evaluate_distilled(
    model_path: str | Path, examples: Sequence[StudentExample], top_k: int = 20, batch_size: int = 256, device: str | None = None
) -> dict[str, float]:
    if not examples:
        return {"validation_rows": 0.0}
    predicted = predict_distilled(model_path, examples, batch_size=batch_size, device=device)
    means = np.asarray([value["mean"] for value in predicted])
    truth = np.asarray([float(value.objective) for value in examples])
    count = min(top_k, len(examples))
    predicted_top = set(np.argsort(means)[-count:])
    actual_top = set(np.argsort(truth)[-count:])
    correlation = spearmanr(truth, means).statistic if len(examples) > 1 else float("nan")
    return {
        "validation_rows": float(len(examples)),
        "objective_mae": float(np.mean(np.abs(truth - means))),
        "spearman": float(correlation),
        "top_k": float(count),
        "top_k_recall": len(predicted_top & actual_top) / max(1, count),
        "predicted_top_mean": float(np.mean(truth[list(predicted_top)])),
        "oracle_top_mean": float(np.mean(truth[list(actual_top)])),
    }


def evaluate_public_affinity(
    model_path: str | Path,
    examples: Sequence[StudentExample],
    batch_size: int = 256,
    device: str | None = None,
    training_baseline: float | None = None,
) -> dict[str, float]:
    labelled = [value for value in examples if value.experimental_affinity is not None]
    if not labelled:
        return {"public_target_holdout_rows": 0.0}
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = _load_models(model_path, selected_device)
    directory = Path(model_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    embedding_name = manifest.get("protein_embeddings")
    embeddings = load_esm2_embeddings(directory / embedding_name if embedding_name else None)
    loader = DataLoader(
        DistilledDataset(labelled), batch_size=batch_size, shuffle=False,
        collate_fn=functools.partial(collate_examples, esm_embeddings=embeddings),
    )
    predictions = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, selected_device)
            values = [
                model(batch["graphs"], batch["protein"], batch["reaction"], batch["components"])["experimental_affinity"]
                for model in models
            ]
            predictions.extend(torch.stack(values).mean(0).cpu().tolist())
    truth = np.asarray([float(value.experimental_affinity) for value in labelled])
    predictions_array = np.asarray(predictions)
    by_target: dict[str, list[int]] = {}
    for index, example in enumerate(labelled):
        by_target.setdefault(example.target, []).append(index)
    within_target = []
    for indices in by_target.values():
        if len(indices) >= 4 and np.std(truth[indices]) > 0 and np.std(predictions_array[indices]) > 0:
            within_target.append(float(spearmanr(truth[indices], predictions_array[indices]).statistic))
    metrics = {
        "public_target_holdout_rows": float(len(labelled)),
        "public_target_holdout_mae": float(np.mean(np.abs(truth - predictions_array))),
        "public_target_holdout_spearman": float(spearmanr(truth, predictions_array).statistic),
        "public_target_holdout_macro_within_target_spearman": float(np.mean(within_target)) if within_target else float("nan"),
        "public_target_holdout_evaluable_targets": float(len(within_target)),
    }
    if training_baseline is not None:
        metrics["public_global_mean_baseline_mae"] = float(np.mean(np.abs(truth - training_baseline)))
    return metrics


def _scored_candidate(example: StudentExample) -> ScoredCandidate:
    molecule = Chem.MolFromSmiles(example.smiles)
    if molecule is None or example.objective is None:
        raise ValueError(f"example {example.name} lacks a valid molecule or objective")
    candidate = Candidate(
        name=example.name,
        smiles=example.smiles,
        reaction=example.reaction,
        components=example.components,
        inchikey=Chem.MolToInchiKey(molecule),
        heavy_atoms=molecule.GetNumHeavyAtoms(),
        rotatable_bonds=int(Descriptors.NumRotatableBonds(molecule)),
    )
    return ScoredCandidate(
        candidate,
        example.binder_probability or 0.0,
        example.boltz_affinity or 0.0,
        float(example.objective),
        example.source,
        0.0,
    )


def evaluate_cascade(
    model_path: str | Path,
    examples: Sequence[StudentExample],
    fraction: float = 0.01,
    minimum: int = 100,
    uncertainty_fraction: float = 0.25,
    submission_count: int = 20,
    minimum_entropy: float = 0.1,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure the decision quality retained under a bounded Boltz budget."""
    labelled = [value for value in examples if value.objective is not None]
    if len(labelled) < submission_count:
        raise ValueError(f"need at least {submission_count} labelled holdout molecules")
    import time

    started = time.monotonic()
    predictions = predict_distilled(model_path, labelled, device=device)
    prediction_seconds = time.monotonic() - started
    budget = min(len(labelled), max(minimum, math.ceil(len(labelled) * fraction), submission_count))
    uncertainty_count = min(budget, math.ceil(budget * uncertainty_fraction))
    exploit_count = budget - uncertainty_count
    mean_order = sorted(range(len(labelled)), key=lambda index: (-predictions[index]["mean"], labelled[index].name))
    uncertainty_order = sorted(
        range(len(labelled)),
        key=lambda index: (-predictions[index]["std"], -predictions[index]["mean"], labelled[index].name),
    )
    chosen = mean_order[:exploit_count]
    chosen_set = set(chosen)
    for index in uncertainty_order:
        if len(chosen) >= budget:
            break
        if index not in chosen_set:
            chosen.append(index)
            chosen_set.add(index)
    truth = np.asarray([float(value.objective) for value in labelled])
    errors = np.abs(truth - np.asarray([value["mean"] for value in predictions]))
    uncertainties = np.asarray([value["std"] for value in predictions])
    top_metrics: dict[str, float] = {}
    for top_k in sorted({submission_count, min(100, len(labelled))}):
        actual_top = set(np.argsort(truth)[-top_k:])
        top_metrics[f"top_{top_k}_recall"] = len(actual_top & chosen_set) / top_k
    all_scored = [_scored_candidate(value) for value in labelled]
    chosen_scored = [all_scored[index] for index in chosen]
    oracle_selection = select_submission(all_scored, submission_count, minimum_entropy)
    try:
        cascade_selection = select_submission(chosen_scored, submission_count, minimum_entropy)
        cascade_score = sum(value.score for value in cascade_selection)
        cascade_entropy_valid = True
    except ValueError:
        cascade_score = float("nan")
        cascade_entropy_valid = False
    oracle_score = sum(value.score for value in oracle_selection)
    uncertainty_correlation = spearmanr(uncertainties, errors).statistic if len(labelled) > 1 else float("nan")
    return {
        "pool": len(labelled),
        "boltz_budget": budget,
        "boltz_fraction": budget / len(labelled),
        "prediction_seconds": prediction_seconds,
        "student_molecules_per_second": len(labelled) / max(prediction_seconds, 1e-9),
        **top_metrics,
        "oracle_submission_score": oracle_score,
        "cascade_submission_score": cascade_score,
        "score_retention": cascade_score / oracle_score if cascade_entropy_valid and oracle_score else float("nan"),
        "score_regret": oracle_score - cascade_score if cascade_entropy_valid else float("nan"),
        "cascade_entropy_valid": cascade_entropy_valid,
        "uncertainty_error_spearman": float(uncertainty_correlation),
        "mean_absolute_error": float(np.mean(errors)),
    }


def select_for_boltz(
    model_path: str | Path,
    archive: str | Path,
    output: str | Path,
    target: str = "P40261",
    fraction: float = 0.01,
    minimum: int = 64,
    maximum: int = 1024,
    uncertainty_fraction: float = 0.25,
    device: str | None = None,
) -> dict[str, Any]:
    with sqlite3.connect(f"file:{Path(archive).resolve()}?mode=ro&immutable=1", uri=True) as connection:
        rows = connection.execute(
            """SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds,proxy_mean,proxy_std
               FROM candidates WHERE score IS NULL ORDER BY name"""
        ).fetchall()
    examples = [
        StudentExample(
            name=row[0], smiles=row[1], target=target, reaction=int(row[2]),
            components=tuple(json.loads(row[3])),
        )
        for row in rows
    ]
    predictions = predict_distilled(model_path, examples, device=device)
    count = min(maximum, max(minimum, math.ceil(len(rows) * fraction)))
    uncertainty_count = min(count, math.ceil(count * uncertainty_fraction))
    ranked_mean = sorted(range(len(rows)), key=lambda index: (-predictions[index]["mean"], rows[index][0]))
    ranked_uncertainty = sorted(range(len(rows)), key=lambda index: (-predictions[index]["std"], -predictions[index]["mean"], rows[index][0]))
    chosen = ranked_mean[: count - uncertainty_count]
    chosen_set = set(chosen)
    for index in ranked_uncertainty:
        if len(chosen) >= count:
            break
        if index not in chosen_set:
            chosen.append(index)
            chosen_set.add(index)
    columns = ("name", "smiles", "reaction", "components", "inchikey", "heavy_atoms", "rotatable_bonds", "proxy_mean", "proxy_std")
    records = []
    for index in chosen:
        record = dict(zip(columns, rows[index]))
        record["components"] = json.loads(record["components"])
        record["score"] = predictions[index]["mean"]
        record["student_mean"] = predictions[index]["mean"]
        record["student_std"] = predictions[index]["std"]
        records.append(record)
    payload = {"count": len(records), "pool": len(rows), "fraction": fraction, "scores": records}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    temporary.replace(destination)
    return {key: value for key, value in payload.items() if key != "scores"}
