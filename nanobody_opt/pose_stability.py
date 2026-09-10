"""Measure repeat-to-repeat stability of validator-shaped nanobody complexes.

BoltzGen's scalar metrics can hide pose switching.  This module aligns every
pair of refolded complexes on the target chain, then measures CDR3 RMSD,
interface-contact Jaccard, and configured-epitope occupancy.  Sequence indices
are used on the nanobody side so the result remains comparable when structure
residue numbering changes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nanobody_opt.regions import find_regions


def residue_numbers(spec: str) -> set[int]:
    numbers: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            start, end = (int(value) for value in part.split("..", 1))
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return numbers


def _heavy_atoms(residue: Any) -> list[Any]:
    return [atom for atom in residue if atom.element.name != "H"]


def _ca(residue: Any) -> np.ndarray | None:
    atom = residue.find_atom("CA", "*")
    if atom is None:
        return None
    return np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)


def _chains(
    path: Path, target_chain: str, nanobody_chain: str
) -> tuple[Any, Any]:
    import gemmi

    structure = gemmi.read_structure(str(path))
    if not structure:
        raise ValueError(f"no models in {path}")
    model = structure[0]
    try:
        return model[target_chain], model[nanobody_chain]
    except KeyError as error:
        raise ValueError(
            f"required chain {error.args[0]!r} absent from {path}"
        ) from error


def structure_features(
    path: Path,
    *,
    sequence: str,
    target_chain: str,
    nanobody_chain: str,
    epitope: set[int],
    cutoff: float = 5.0,
) -> dict[str, Any]:
    target, design = _chains(path, target_chain, nanobody_chain)
    target_residues = [residue for residue in target if _heavy_atoms(residue)]
    design_residues = [residue for residue in design if _heavy_atoms(residue)]
    if len(design_residues) != len(sequence):
        raise ValueError(
            f"{path} has {len(design_residues)} nanobody residues, expected "
            f"{len(sequence)}"
        )

    target_ca = {
        (int(residue.seqid.num), str(residue.seqid.icode)): coordinate
        for residue in target_residues
        if (coordinate := _ca(residue)) is not None
    }
    regions = find_regions(sequence)
    cdr3_ca = {
        index: coordinate
        for index, residue in enumerate(design_residues, start=1)
        if regions.cdr3[0] < index <= regions.cdr3[1]
        and (coordinate := _ca(residue)) is not None
    }

    cutoff_squared = cutoff * cutoff
    contacts: set[tuple[int, int]] = set()
    contacted_target: set[int] = set()
    contacted_design: set[int] = set()
    for design_index, design_residue in enumerate(design_residues, start=1):
        design_atoms = _heavy_atoms(design_residue)
        for target_residue in target_residues:
            target_atoms = _heavy_atoms(target_residue)
            if any(
                (design_atom.pos.x - target_atom.pos.x) ** 2
                + (design_atom.pos.y - target_atom.pos.y) ** 2
                + (design_atom.pos.z - target_atom.pos.z) ** 2
                <= cutoff_squared
                for design_atom in design_atoms
                for target_atom in target_atoms
            ):
                target_number = int(target_residue.seqid.num)
                contacts.add((design_index, target_number))
                contacted_target.add(target_number)
                contacted_design.add(design_index)

    occupancy = (
        len(contacted_target & epitope) / len(epitope) if epitope else 1.0
    )
    return {
        "path": str(path),
        "target_ca": target_ca,
        "cdr3_ca": cdr3_ca,
        "contacts": contacts,
        "contacted_target": contacted_target,
        "contacted_design": contacted_design,
        "epitope_occupancy": float(occupancy),
    }


def _aligned_rmsd(left: dict[str, Any], right: dict[str, Any]) -> float:
    target_keys = sorted(set(left["target_ca"]) & set(right["target_ca"]))
    cdr3_keys = sorted(set(left["cdr3_ca"]) & set(right["cdr3_ca"]))
    if len(target_keys) < 3 or len(cdr3_keys) < 3:
        raise ValueError("insufficient common target or CDR3 CA atoms")
    moving = np.asarray([right["target_ca"][key] for key in target_keys])
    reference = np.asarray([left["target_ca"][key] for key in target_keys])
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (moving - moving_center).T @ (reference - reference_center)
    left_vectors, _, right_vectors = np.linalg.svd(covariance)
    rotation = left_vectors @ right_vectors
    if np.linalg.det(rotation) < 0:
        left_vectors[:, -1] *= -1
        rotation = left_vectors @ right_vectors
    moving_cdr3 = np.asarray([right["cdr3_ca"][key] for key in cdr3_keys])
    reference_cdr3 = np.asarray([left["cdr3_ca"][key] for key in cdr3_keys])
    aligned = (moving_cdr3 - moving_center) @ rotation + reference_center
    return float(np.sqrt(np.mean(np.sum((aligned - reference_cdr3) ** 2, axis=1))))


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def summarize_sequence_structures(
    sequence: str,
    paths: list[Path],
    *,
    target_chain: str,
    nanobody_chain: str,
    epitope: set[int],
    cutoff: float = 5.0,
) -> dict[str, Any]:
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths if path.is_file()))
    features = [
        structure_features(
            path,
            sequence=sequence,
            target_chain=target_chain,
            nanobody_chain=nanobody_chain,
            epitope=epitope,
            cutoff=cutoff,
        )
        for path in unique_paths
    ]
    if not features:
        return {"structure_observation_count": 0}
    pairs = list(itertools.combinations(features, 2))
    contact_jaccards = [
        _jaccard(left["contacts"], right["contacts"])
        for left, right in pairs
    ]
    cdr3_rmsds = [_aligned_rmsd(left, right) for left, right in pairs]
    occupancies = [item["epitope_occupancy"] for item in features]
    design_frequency: dict[int, int] = {}
    target_frequency: dict[int, int] = {}
    for item in features:
        for position in item["contacted_design"]:
            design_frequency[position] = design_frequency.get(position, 0) + 1
        for position in item["contacted_target"]:
            target_frequency[position] = target_frequency.get(position, 0) + 1
    consensus_contacts = sorted(
        position
        for position, count in design_frequency.items()
        if count / len(features) >= 0.5
    )
    consensus_target_contacts = sorted(
        position
        for position, count in target_frequency.items()
        if count / len(features) >= 0.5
    )
    return {
        "structure_observation_count": len(features),
        "contact_cutoff_angstrom": cutoff,
        "contact_jaccard_min": float(min(contact_jaccards)) if pairs else 1.0,
        "contact_jaccard_median": (
            float(np.median(contact_jaccards)) if pairs else 1.0
        ),
        "cdr3_rmsd_max_angstrom": float(max(cdr3_rmsds)) if pairs else 0.0,
        "cdr3_rmsd_median_angstrom": (
            float(np.median(cdr3_rmsds)) if pairs else 0.0
        ),
        "epitope_occupancy_min": float(min(occupancies)),
        "epitope_occupancy_median": float(np.median(occupancies)),
        "consensus_nanobody_contact_positions": consensus_contacts,
        "consensus_target_contact_positions": consensus_target_contacts,
        "structure_paths": [str(path) for path in unique_paths],
    }


def exact_structure_paths(
    paths: list[Path], target: str
) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    seen_observations: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        if str(payload.get("target")) != target:
            raise ValueError(f"{path} targets {payload.get('target')!r}, not {target}")
        observation = str(payload.get("observation_id") or path.resolve())
        if observation in seen_observations:
            continue
        seen_observations.add(observation)
        for sequence, by_target in (payload.get("structures") or {}).items():
            record = by_target.get(target) if isinstance(by_target, dict) else None
            structure_path = record.get("path") if isinstance(record, dict) else record
            if structure_path:
                found.setdefault(str(sequence).strip().upper(), []).append(
                    Path(str(structure_path))
                )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="append", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())["protein_selection"]["nanobody"]
    if str(config["target"]) != args.target:
        raise SystemExit("pose config target does not match exact target")
    structures = exact_structure_paths(args.exact, args.target)
    summaries = {
        sequence: summarize_sequence_structures(
            sequence,
            paths,
            target_chain=str(config["structure_chain"]),
            nanobody_chain="B",
            epitope=residue_numbers(str(config["structure_binding_site"])),
            cutoff=args.cutoff,
        )
        for sequence, paths in structures.items()
    }
    payload = {"target": args.target, "sequences": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"POSE_STABILITY_READY sequences={len(summaries)} output={args.output}")


if __name__ == "__main__":
    main()
