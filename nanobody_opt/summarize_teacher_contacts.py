"""Infer a target epitope from consensus contacts of strong public teachers."""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml


def _compress_residues(residues: list[int]) -> str:
    if not residues:
        return ""
    groups: list[tuple[int, int]] = []
    start = previous = residues[0]
    for residue in residues[1:]:
        if residue == previous + 1:
            previous = residue
            continue
        groups.append((start, previous))
        start = previous = residue
    groups.append((start, previous))
    return ",".join(
        str(start) if start == end else f"{start}..{end}"
        for start, end in groups
    )


def summarize_contacts(
    contacts: list[dict[str, Any]], minimum_frequency: float
) -> dict[str, Any]:
    if not 0 < minimum_frequency <= 1:
        raise ValueError("minimum_frequency must be in (0, 1]")
    if not contacts:
        raise ValueError("no teacher contacts supplied")

    target_chains = {str(item.get("target_chain", "")) for item in contacts}
    cutoffs = {float(item.get("cutoff_angstrom", 0)) for item in contacts}
    if len(target_chains) != 1 or "" in target_chains:
        raise ValueError(f"teacher target chains disagree: {sorted(target_chains)}")
    if len(cutoffs) != 1 or 0 in cutoffs:
        raise ValueError(f"teacher contact cutoffs disagree: {sorted(cutoffs)}")

    frequency: Counter[int] = Counter()
    interface_sizes: list[int] = []
    for item in contacts:
        observed = {
            int(target_contact["residue_number"])
            for interface_residue in item.get("interface_residues", [])
            for target_contact in interface_residue.get("target_contacts", [])
        }
        frequency.update(observed)
        interface_sizes.append(int(item.get("interface_residue_count", 0)))

    required = math.ceil(len(contacts) * minimum_frequency)
    selected = sorted(
        residue for residue, count in frequency.items() if count >= required
    )
    if not selected:
        raise ValueError("no target residue meets the consensus threshold")
    return {
        "teacher_count": len(contacts),
        "target_chain": next(iter(target_chains)),
        "cutoff_angstrom": next(iter(cutoffs)),
        "minimum_frequency": minimum_frequency,
        "minimum_teacher_count": required,
        "recommended_binding_site": _compress_residues(selected),
        "recommended_residues": selected,
        "residue_teacher_counts": {
            str(residue): frequency[residue] for residue in sorted(frequency)
        },
        "interface_residue_counts": interface_sizes,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", required=True, help="glob of contact JSON files")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-frequency", type=float, default=0.7)
    parser.add_argument("--minimum-teachers", type=int, default=3)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.contacts))]
    if len(paths) < args.minimum_teachers:
        raise SystemExit(
            f"found {len(paths)} contact files; need {args.minimum_teachers}"
        )
    contacts = [json.loads(path.read_text()) for path in paths]
    result = summarize_contacts(contacts, args.minimum_frequency)
    config = yaml.safe_load(args.config.read_text())
    challenge = config["protein_selection"]["nanobody"]
    if result["target_chain"] != str(challenge["structure_chain"]):
        raise SystemExit("teacher target chain does not match live challenge")
    result.update({
        "purpose": "dynamic generation epitope; inferred from public teachers",
        "target": str(challenge["target"]),
        "structure": str(challenge["structure"]),
        "configured_binding_site": str(challenge["structure_binding_site"]),
        "contact_files": [str(path) for path in paths],
    })
    _atomic_json(args.output, result)
    print(
        f"TEACHER_EPITOPE_READY target={result['target']} "
        f"teachers={result['teacher_count']} "
        f"binding={result['recommended_binding_site']}"
    )


if __name__ == "__main__":
    main()
