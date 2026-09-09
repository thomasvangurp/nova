"""Extract nanobody-side interface residues from a predicted target complex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi


def _residue_numbers(spec: str) -> set[int]:
    numbers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            start, end = (int(value) for value in part.split("..", 1))
            numbers.update(range(start, end + 1))
        else:
            numbers.add(int(part))
    return numbers


def _atoms(residue: gemmi.Residue) -> list[gemmi.Atom]:
    return [atom for atom in residue if atom.element.name != "H"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--nanobody-chain", default="B")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--epitope", default="")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.cutoff <= 0:
        parser.error("--cutoff must be positive")
    structure = gemmi.read_structure(str(args.structure))
    if not structure:
        raise SystemExit(f"no models in {args.structure}")
    model = structure[0]
    try:
        target_chain = model[args.target_chain]
        nanobody_chain = model[args.nanobody_chain]
    except KeyError as exc:
        available = [chain.name for chain in model]
        raise SystemExit(
            f"chain {exc.args[0]!r} absent; available chains: {available}"
        ) from exc

    target_residues = [residue for residue in target_chain if _atoms(residue)]
    nanobody_residues = [residue for residue in nanobody_chain if _atoms(residue)]
    epitope = _residue_numbers(args.epitope)
    contacts = []
    cutoff_squared = args.cutoff * args.cutoff

    for sequence_index, nanobody_residue in enumerate(nanobody_residues, start=1):
        nanobody_atoms = _atoms(nanobody_residue)
        target_contacts = []
        minimum_squared = float("inf")
        for target_residue in target_residues:
            target_atoms = _atoms(target_residue)
            residue_minimum = min(
                (nanobody_atom.pos.x - target_atom.pos.x) ** 2
                + (nanobody_atom.pos.y - target_atom.pos.y) ** 2
                + (nanobody_atom.pos.z - target_atom.pos.z) ** 2
                for nanobody_atom in nanobody_atoms
                for target_atom in target_atoms
            )
            minimum_squared = min(minimum_squared, residue_minimum)
            if residue_minimum <= cutoff_squared:
                target_number = int(target_residue.seqid.num)
                target_contacts.append({
                    "residue_number": target_number,
                    "residue_name": target_residue.name,
                    "minimum_distance": round(residue_minimum ** 0.5, 3),
                    "configured_epitope": target_number in epitope,
                })
        if target_contacts:
            contacts.append({
                "sequence_index": sequence_index,
                "residue_number": int(nanobody_residue.seqid.num),
                "residue_name": nanobody_residue.name,
                "minimum_target_distance": round(minimum_squared ** 0.5, 3),
                "target_contacts": target_contacts,
            })

    payload = {
        "structure": str(args.structure),
        "target_chain": args.target_chain,
        "nanobody_chain": args.nanobody_chain,
        "cutoff_angstrom": args.cutoff,
        "configured_epitope": sorted(epitope),
        "nanobody_residue_count": len(nanobody_residues),
        "interface_residue_count": len(contacts),
        "interface_residues": contacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(contacts)}/{len(nanobody_residues)} nanobody residues contact "
          f"target chain {args.target_chain} within {args.cutoff:.1f} A")
    print("sequence indices:", ",".join(str(row["sequence_index"]) for row in contacts))


if __name__ == "__main__":
    main()
