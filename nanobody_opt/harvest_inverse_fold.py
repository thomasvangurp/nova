"""Harvest fixed-backbone inverse-fold outputs with exact cheap gates.

BoltzGen can generate hundreds of sequences for one measured teacher pose much
faster than Boltz2 can refold them. This stage computes the exact BoltzGen
liability formula, current Layer-1 gates, archive uniqueness, and the cheap
top-50 screen before any expensive exact scoring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def choose_designed_sequence(polymers: list[str], teacher: str) -> str:
    """Select the sole polymer matching the teacher VHH length."""
    normalized = ["".join(sequence.split()).upper() for sequence in polymers]
    matches = [sequence for sequence in normalized if len(sequence) == len(teacher)]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {len(teacher)}-residue designed chain, got {len(matches)}"
        )
    return matches[0]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
    parser.add_argument("--design-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    import gemmi
    import pandas as pd
    import yaml
    from boltzgen.task.analyze.analyze_utils import compute_liability_scores

    from nanobody_opt import diversity, gates

    teacher = args.teacher.strip().upper()
    config = yaml.safe_load(args.config.read_text())
    nanobody = config["protein_selection"]["nanobody"]
    requirements = config["nanobody_requirements"]
    archive = pd.read_csv(args.archive).drop_duplicates("sequence", keep="last")
    archive["sequence"] = archive.sequence.astype(str).str.strip().str.upper()
    existing = set(archive.sequence)
    top = diversity.minmax_weighted_rank(
        archive[archive.liability_num_violations <= 50],
        k=int(requirements["num_top_sequences"]),
    )
    top_kmers = [diversity.kmers(sequence, 5) for sequence in top.sequence]
    threshold = float(requirements["max_similarity_score"])

    raw: dict[str, dict[str, Any]] = {}
    errors = 0
    for path in sorted(args.design_dir.glob("*.cif")):
        try:
            block = gemmi.cif.read_file(str(path)).sole_block()
            polymers = list(block.find_values("_entity_poly.pdbx_seq_one_letter_code"))
            sequence = choose_designed_sequence(polymers, teacher)
        except (ValueError, RuntimeError):
            errors += 1
            continue
        raw.setdefault(
            sequence,
            {
                "sequence": sequence,
                "source": str(path),
                "mutation_count": sum(a != b for a, b in zip(sequence, teacher)),
            },
        )

    liability = compute_liability_scores(
        list(raw), modality="peptide", peptide_type="linear"
    )
    candidates = []
    rejected: dict[str, int] = {}
    for sequence, item in raw.items():
        reason = None
        if sequence == teacher:
            reason = "public_teacher"
        elif sequence in existing:
            reason = "historical_exact"
        elif gates.check_sequence(sequence, requirements):
            reason = "layer1"
        else:
            diverse, max_jaccard = diversity.passes_diversity(
                sequence, top_kmers, threshold=threshold, k=5
            )
            item["max_jaccard_to_top50"] = max_jaccard
            if not diverse:
                reason = "top50_jaccard"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        item["liability_score"] = liability[sequence]["score"]
        item["liability_num_violations"] = len(liability[sequence]["violations"])
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            item["liability_score"],
            item["liability_num_violations"],
            item["mutation_count"],
            item["sequence"],
        )
    )
    selected = candidates[: args.limit]
    payload = {
        "purpose": "fixed-backbone candidates ranked by exact cheap gates",
        "target": str(nanobody["target"]),
        "teacher_sequence": teacher,
        "input_cif_count": len(list(args.design_dir.glob("*.cif"))),
        "parse_error_count": errors,
        "unique_sequence_count": len(raw),
        "eligible_count": len(candidates),
        "rejection_counts": rejected,
        "cases": selected,
    }
    _atomic_json(args.output, payload)
    print(
        f"INVERSE_FOLD_HARVEST_READY target={payload['target']} "
        f"cifs={payload['input_cif_count']} unique={len(raw)} "
        f"eligible={len(candidates)} selected={len(selected)}"
    )
    for item in selected[:10]:
        print(
            f"  liability={item['liability_score']}/{item['liability_num_violations']} "
            f"mutations={item['mutation_count']} {item['sequence']}"
        )


if __name__ == "__main__":
    main()
