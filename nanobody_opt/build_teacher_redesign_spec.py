"""Build a CDR-only inverse-folding spec from a retained public teacher pose.

The teacher is selected dynamically from ``select_structural_teachers.py``
output and located in an isolated exact-scoring work root.  Its public sequence
is never emitted as a candidate.  By default all three IMGT CDRs are marked for
redesign; when a contact report is supplied, observed interface residues are
held fixed and only the non-contact CDR positions are redesigned.  In both
cases the competitive target/VHH backbone pose is held fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


def _compress_positions(positions: list[int]) -> str:
    """Return sorted 1-based positions as compact inclusive ranges."""
    if not positions:
        raise ValueError("no design positions remain")
    groups: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        groups.append((start, previous))
        start = previous = position
    groups.append((start, previous))
    return ",".join(
        str(start) if start == end else f"{start}..{end}"
        for start, end in groups
    )


def cdr_ranges(
    sequence: str,
    cdr_sequences: list[str],
    excluded_positions: set[int] | None = None,
) -> str:
    """Return 1-based inclusive ranges for ordered CDR subsequences."""
    ranges: list[str] = []
    search_from = 0
    excluded = excluded_positions or set()
    for cdr in cdr_sequences:
        if not cdr:
            raise ValueError("CDR sequence may not be empty")
        start = sequence.find(cdr, search_from)
        if start < 0:
            raise ValueError(f"CDR {cdr!r} is absent from teacher sequence")
        end = start + len(cdr)
        positions = [
            position
            for position in range(start + 1, end + 1)
            if position not in excluded
        ]
        if positions:
            ranges.append(_compress_positions(positions))
        search_from = end
    if not ranges:
        raise ValueError("no design positions remain")
    return ",".join(ranges)


def contact_positions(path: Path) -> set[int]:
    """Load 1-based VHH sequence positions observed within the contact cutoff."""
    payload = json.loads(path.read_text())
    residues = payload.get("interface_residues")
    if not isinstance(residues, list):
        raise ValueError("contact report has no interface_residues list")
    positions = {int(residue["sequence_index"]) for residue in residues}
    if not positions:
        raise ValueError("contact report has no VHH interface positions")
    return positions


def locate_retained_complex(
    work_root: Path, sequence: str
) -> tuple[Path, int]:
    matches: list[tuple[Path, int]] = []
    for shard_output in sorted(work_root.glob("out*.json")):
        payload = json.loads(shard_output.read_text())
        present = any(
            sequence in by_sequence
            for by_sequence in payload.get("components", {}).values()
        )
        if not present:
            continue
        shard = int(payload["shard"])
        refolds = sorted(
            (work_root / f"shard{shard}" / "chunk0" / "outputs"
             / "intermediate_designs" / "refold_cif").glob("*.cif")
        )
        if len(refolds) != 1:
            raise ValueError(
                f"teacher shard {shard} has {len(refolds)} retained refolds"
            )
        matches.append((refolds[0], shard))
    if len(matches) != 1:
        raise ValueError(f"teacher sequence maps to {len(matches)} retained complexes")
    return matches[0]


def build_spec(
    structure: Path,
    target_chain: str,
    nanobody_chain: str,
    design_ranges: str,
) -> dict[str, Any]:
    return {
        "entities": [
            {
                "file": {
                    "path": str(structure),
                    "include": [
                        {"chain": {"id": target_chain}},
                        {"chain": {"id": nanobody_chain}},
                    ],
                    "design": [
                        {
                            "chain": {
                                "id": nanobody_chain,
                                "res_index": design_ranges,
                            }
                        }
                    ],
                }
            }
        ]
    }


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--teacher-rank", type=int, default=1)
    parser.add_argument("--target-chain", default="A")
    parser.add_argument("--nanobody-chain", default="B")
    parser.add_argument(
        "--contacts",
        type=Path,
        help="interface_contacts.py JSON; preserve every observed VHH contact",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.teachers.read_text())
    cases = payload.get("cases", [])
    if not 1 <= args.teacher_rank <= len(cases):
        parser.error(f"--teacher-rank must be between 1 and {len(cases)}")
    teacher = cases[args.teacher_rank - 1]
    sequence = str(teacher["sequence"])

    # AbNumber delegates IMGT numbering to ANARCI, both locked in MetaNano.
    from abnumber import Chain

    chain = Chain(sequence, scheme="imgt")
    excluded = contact_positions(args.contacts) if args.contacts else set()
    ranges = cdr_ranges(
        sequence,
        [str(chain.cdr1_seq), str(chain.cdr2_seq), str(chain.cdr3_seq)],
        excluded_positions=excluded,
    )
    structure, shard = locate_retained_complex(args.work_root, sequence)
    spec = build_spec(
        structure,
        target_chain=args.target_chain,
        nanobody_chain=args.nanobody_chain,
        design_ranges=ranges,
    )
    _atomic_yaml(args.output, spec)
    print(
        f"TEACHER_REDESIGN_SPEC_READY target={payload.get('target')} "
        f"teacher_rank={args.teacher_rank} shard={shard} cdrs={ranges} "
        f"preserved_contacts={len(excluded)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
