"""Generate minimal terminal variants of an exact-scored structural teacher.

The public teacher itself is never emitted.  Terminal substitutions preserve
the measured paratope and competitive docking geometry while satisfying the
historical exact-hash gate.  Every proposal is also checked against the live
sequence gates and the validator's current Algorithm-2 top-50 diversity set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

from nanobody_opt import gates


AMINO_ACIDS_WITHOUT_CYS = "ADEFGHIKLMNPQRSTVWY"


def terminal_positions(length: int, width: int) -> list[int]:
    """Return sorted zero-based positions in both sequence termini."""
    if width <= 0:
        raise ValueError("terminal width must be positive")
    if 2 * width > length:
        raise ValueError("terminal windows overlap")
    return list(range(width)) + list(range(length - width, length))


def scan_variants(
    teacher: str,
    width: int,
    existing: set[str],
    requirements: dict,
    passes_top50: Callable[[str], tuple[bool, float]],
) -> list[dict[str, object]]:
    """Enumerate deterministic single substitutions that pass cheap gates."""
    cases: list[dict[str, object]] = []
    for position in terminal_positions(len(teacher), width):
        for residue in AMINO_ACIDS_WITHOUT_CYS:
            if residue == teacher[position]:
                continue
            candidate = teacher[:position] + residue + teacher[position + 1:]
            if candidate in existing or gates.check_sequence(candidate, requirements):
                continue
            diverse, max_jaccard = passes_top50(candidate)
            if not diverse:
                continue
            cases.append({
                "sequence": candidate,
                "teacher_sequence": teacher,
                "mutation": f"{teacher[position]}{position + 1}{residue}",
                "max_jaccard_to_top50": max_jaccard,
            })
    return cases


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
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
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--teacher-rank", type=int, default=1)
    parser.add_argument("--terminal-width", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import pandas as pd
    import yaml

    from nanobody_opt import diversity

    teacher_payload = json.loads(args.teachers.read_text())
    teachers = teacher_payload.get("cases", [])
    if not 1 <= args.teacher_rank <= len(teachers):
        parser.error(f"--teacher-rank must be between 1 and {len(teachers)}")
    teacher = str(teachers[args.teacher_rank - 1]["sequence"])

    config = yaml.safe_load(args.config.read_text())
    nanobody = config["protein_selection"]["nanobody"]
    target = str(nanobody["target"])
    if target != str(teacher_payload.get("target")):
        raise SystemExit("teacher target does not match the live configuration")
    requirements = config["nanobody_requirements"]

    archive = pd.read_csv(args.archive).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive.sequence.astype(str).str.strip().str.upper()
    existing = set(archive.sequence)
    top = diversity.minmax_weighted_rank(
        archive, k=int(requirements["num_top_sequences"])
    )
    top_kmers = [diversity.kmers(sequence, 5) for sequence in top.sequence]
    threshold = float(requirements["max_similarity_score"])

    def passes_top50(sequence: str) -> tuple[bool, float]:
        return diversity.passes_diversity(
            sequence, top_kmers, threshold=threshold, k=5
        )

    cases = scan_variants(
        teacher,
        width=args.terminal_width,
        existing=existing,
        requirements=requirements,
        passes_top50=passes_top50,
    )
    payload = {
        "purpose": "R&D terminal scan; public teacher is never emitted",
        "target": target,
        "teacher_rank": args.teacher_rank,
        "terminal_width": args.terminal_width,
        "cases": cases,
    }
    _atomic_json(args.output, payload)
    print(
        f"TEACHER_TERMINAL_SCAN_READY target={target} teacher_rank={args.teacher_rank} "
        f"candidates={len(cases)} output={args.output}"
    )


if __name__ == "__main__":
    main()
