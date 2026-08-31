#!/usr/bin/env python3
"""Extract frozen PSICHIC interaction fingerprints for a labelled JSON set."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

# Prefer the vendor CUDA extensions when they exist. The small compatibility
# layer is only for environments where PyG is present without torch-scatter or
# torch-sparse wheels for that exact PyTorch/CUDA release.
try:
    importlib.import_module("torch_scatter")
    importlib.import_module("torch_sparse")
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "compat"))

from compound_miner.targets import TARGETS


class ProteinBatch:
    def __init__(self, protein: dict, batch_size: int, device: torch.device) -> None:
        aa = protein["seq_feat"].float()
        evolutionary = protein["token_representation"].float()
        edge_index = protein["edge_index"].long()
        edge_weight = protein["edge_weight"].float()
        self.length = aa.shape[0]
        self.batch_size = batch_size
        self.prot_node_aa = aa.repeat(batch_size, 1).to(device)
        self.prot_node_evo = evolutionary.repeat(batch_size, 1).to(device)
        edge_count = edge_index.shape[1]
        offsets = torch.arange(batch_size).repeat_interleave(edge_count) * self.length
        self.prot_edge_index = (edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)).to(device)
        self.prot_edge_weight = edge_weight.repeat(batch_size).to(device)
        self.prot_batch = torch.arange(batch_size).repeat_interleave(self.length).to(device)

    def slice(self, count: int) -> "ProteinBatch":
        if count == self.batch_size:
            return self
        result = object.__new__(ProteinBatch)
        result.length = self.length
        result.batch_size = count
        keep = self.prot_edge_index[0] < count * self.length
        result.prot_node_aa = self.prot_node_aa[: count * self.length]
        result.prot_node_evo = self.prot_node_evo[: count * self.length]
        result.prot_edge_index = self.prot_edge_index[:, keep]
        result.prot_edge_weight = self.prot_edge_weight[keep]
        result.prot_batch = self.prot_batch[: count * self.length]
        return result


def prepare_graphs(graphs: dict[str, dict]) -> dict[str, dict]:
    output = {}
    for smiles, value in graphs.items():
        graph = dict(value)
        graph["atom_idx"] = value["atom_idx"].long().view(-1, 1)
        graph["atom_feature"] = value["atom_feature"].float()
        adjacency = value["bond_feature"].long()
        edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
        graph["atom_edge_index"] = edge_index
        graph["atom_edge_attr"] = adjacency[edge_index[0], edge_index[1]].long()
        graph["x_clique"] = value["x_clique"].long().view(-1, 1)
        graph["tree_edge_index"] = value["tree_edge_index"].long()
        graph["atom2clique_index"] = value["atom2clique_index"].long()
        output[smiles] = graph
    return output


def label_quality(row: dict) -> int:
    scorer = str(row.get("scorer", "")).lower()
    return 2 if "validator" in scorer or ("r3" in scorer and "as100" in scorer and "ad3" in scorer) else 1


def collate_molecules(graphs: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    values = {name: [] for name in ("mol_x", "mol_feat", "mol_ei", "mol_ea", "clique_x", "clique_ei", "atom2clique", "mol_batch", "clique_batch")}
    atom_offset = 0
    clique_offset = 0
    for batch_index, graph in enumerate(graphs):
        atom_count = graph["atom_idx"].shape[0]
        clique_count = graph["x_clique"].shape[0]
        values["mol_x"].append(graph["atom_idx"])
        values["mol_feat"].append(graph["atom_feature"])
        values["mol_ei"].append(graph["atom_edge_index"] + atom_offset)
        values["mol_ea"].append(graph["atom_edge_attr"])
        values["clique_x"].append(graph["x_clique"])
        values["clique_ei"].append(graph["tree_edge_index"] + clique_offset)
        values["atom2clique"].append(graph["atom2clique_index"] + torch.tensor([[atom_offset], [clique_offset]]))
        values["mol_batch"].append(torch.full((atom_count,), batch_index, dtype=torch.long))
        values["clique_batch"].append(torch.full((clique_count,), batch_index, dtype=torch.long))
        atom_offset += atom_count
        clique_offset += clique_count
    concatenate = torch.cat
    return {
        "mol_x": concatenate(values["mol_x"]).to(device),
        "mol_x_feat": concatenate(values["mol_feat"]).to(device),
        "bond_x": concatenate(values["mol_ea"]).to(device),
        "atom_edge_index": concatenate(values["mol_ei"], 1).to(device),
        "clique_x": concatenate(values["clique_x"]).to(device),
        "clique_edge_index": concatenate(values["clique_ei"], 1).to(device),
        "atom2clique_index": concatenate(values["atom2clique"], 1).to(device),
        "mol_batch": concatenate(values["mol_batch"]).to(device),
        "clique_batch": concatenate(values["clique_batch"]).to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psichic-root", type=Path, required=True)
    parser.add_argument("--records-json", type=Path, action="append", required=True)
    parser.add_argument("--target", default="P40261")
    parser.add_argument("--sequence")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    psichic_root = args.psichic_root.resolve()
    sys.path.insert(0, str(psichic_root.parent))
    sys.path.insert(0, str(psichic_root))
    os.environ.setdefault("DEVICE_OVERRIDE", "cuda" if torch.cuda.is_available() else "cpu")
    wrapper_module = importlib.import_module(f"{psichic_root.name}.wrapper")
    ligand_module = importlib.import_module(f"{psichic_root.name}.psichic_utils.ligand_init")
    startup_started = time.perf_counter()
    wrapper = wrapper_module.PsichicWrapper()
    wrapper.initialize_model(args.sequence or TARGETS[args.target])
    startup_seconds = time.perf_counter() - startup_started
    device = torch.device(wrapper.device)
    protein = dict(wrapper.protein_dict[next(iter(wrapper.protein_dict))])
    protein_batch = ProteinBatch(protein, args.batch_size, device)

    rows = {}
    for path in args.records_json:
        for row in json.loads(path.read_text())["scores"]:
            previous = rows.get(row["name"])
            quality = label_quality(row)
            previous_quality = label_quality(previous) if previous else 0
            if previous is None or quality > previous_quality:
                rows[row["name"]] = row
    ordered = [rows[name] for name in sorted(rows)]
    featurize_started = time.perf_counter()
    raw_graphs = ligand_module.ligand_init([row["smiles"] for row in ordered])
    graphs = prepare_graphs(raw_graphs)
    featurize_seconds = time.perf_counter() - featurize_started
    kept = [row for row in ordered if row["smiles"] in graphs]
    fingerprints = []
    scores = []
    warmup_started = time.perf_counter()
    if kept:
        warmup_chunk = kept[: min(len(kept), args.batch_size)]
        warmup_pack = collate_molecules([graphs[row["smiles"]] for row in warmup_chunk], device)
        warmup_protein = protein_batch.slice(len(warmup_chunk))
        with torch.inference_mode():
            wrapper.model(
                **warmup_pack,
                residue_x=warmup_protein.prot_node_aa,
                residue_evo_x=warmup_protein.prot_node_evo,
                residue_edge_index=warmup_protein.prot_edge_index,
                residue_edge_weight=warmup_protein.prot_edge_weight,
                prot_batch=warmup_protein.prot_batch,
                save_cluster=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(kept), args.batch_size):
            chunk = kept[start : start + args.batch_size]
            packed = collate_molecules([graphs[row["smiles"]] for row in chunk], device)
            current_protein = protein_batch.slice(len(chunk))
            prediction, _, _, _, _, _, attention = wrapper.model(
                **packed,
                residue_x=current_protein.prot_node_aa,
                residue_evo_x=current_protein.prot_node_evo,
                residue_edge_index=current_protein.prot_edge_index,
                residue_edge_weight=current_protein.prot_edge_weight,
                prot_batch=current_protein.prot_batch,
                save_cluster=False,
            )
            fingerprints.append(attention["interaction_fingerprint"].float().cpu().numpy())
            scores.append(prediction.squeeze(-1).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=np.asarray([row["name"] for row in kept]),
        smiles=np.asarray([row["smiles"] for row in kept]),
        reactions=np.asarray([int(row.get("reaction", 0)) for row in kept], dtype=np.int16),
        weights=np.asarray([
            1.0 if label_quality(row) == 2 else 0.55
            for row in kept
        ], dtype=np.float32),
        fingerprints=np.concatenate(fingerprints),
        psichic_score=np.concatenate(scores),
        objective=np.asarray([row.get("score", float("nan")) for row in kept], dtype=np.float32),
    )
    print(json.dumps({
        "count": len(kept),
        "features": int(fingerprints[0].shape[1]),
        "startup_seconds": startup_seconds,
        "warmup_seconds": warmup_seconds,
        "featurize_seconds": featurize_seconds,
        "inference_seconds": inference_seconds,
        "warmed_molecules_per_second": len(kept) / (featurize_seconds + inference_seconds),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
