"""Graft a measured teacher paratope onto validator-proven VHH frameworks.

CDRs alone do not determine their final conformation.  AbNumber therefore
copies all IMGT Vernier residues from the teacher as well as every framework
residue observed at the antigen interface.  The remaining scaffold comes from
public, low-liability VHHs that already survived a validator scoring pass.
Public donors and the public teacher are provenance only; exact historical and
current top-50 novelty gates are applied to every resulting hybrid.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nanobody_opt import diversity, gates
from nanobody_opt.build_teacher_redesign_spec import contact_positions
from nanobody_opt.simulate_round import METRICS


def select_donors(
    archive: pd.DataFrame,
    teacher_length: int,
    max_liability_score: float,
    limit: int,
) -> pd.DataFrame:
    """Select low-liability frameworks with strong measured target metrics."""
    missing = ({"sequence", "liability_score"} | set(METRICS)) - set(archive)
    if missing:
        raise ValueError(f"archive lacks donor columns: {sorted(missing)}")
    frame = archive.copy()
    frame["sequence"] = frame.sequence.astype(str).str.strip().str.upper()
    frame = frame[
        frame.sequence.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+")
        & (frame.sequence.str.len() == teacher_length)
        & (pd.to_numeric(frame.liability_score, errors="coerce")
           <= max_liability_score)
    ].drop_duplicates("sequence", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError("no same-length donors pass the liability ceiling")

    total = np.zeros(len(frame), dtype=np.float64)
    for metric, direction in METRICS.items():
        total += pd.to_numeric(frame[metric], errors="coerce").rank(
            ascending=(direction == "min"), method="average", na_option="bottom"
        ).to_numpy()
    frame["donor_rank_sum"] = total
    return frame.sort_values(
        ["donor_rank_sum", "liability_score", "sequence"], kind="stable"
    ).head(min(limit, len(frame)))


def graft_sequence(
    teacher_sequence: str,
    donor_sequence: str,
    preserved_raw_positions: set[int],
) -> tuple[str, list[str]]:
    """Return an IMGT CDR/Vernier/contact graft and copied positions."""
    from abnumber import Chain

    teacher = Chain(teacher_sequence, scheme="imgt")
    donor = Chain(donor_sequence, scheme="imgt")
    numbered = [position for position, _ in teacher]
    invalid = sorted(
        position for position in preserved_raw_positions
        if position < 1 or position > len(numbered)
    )
    if invalid:
        raise ValueError(f"contact positions outside teacher sequence: {invalid}")
    backmutations = [numbered[position - 1] for position in preserved_raw_positions]
    grafted = teacher.graft_cdrs_onto(
        donor,
        backmutate_vernier=True,
        backmutations=backmutations,
        name="teacher_paratope_low_liability_framework",
    )
    return grafted.seq, [str(position) for position in backmutations]


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
    parser.add_argument("--teachers", required=True, type=Path)
    parser.add_argument("--teacher-rank", type=int, default=1)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--contacts", action="append", type=Path, default=[],
        help="contact report to backmutate from the teacher; may be repeated",
    )
    parser.add_argument("--max-donor-liability", type=float, default=180.0)
    parser.add_argument("--donors", type=int, default=200)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    teacher_payload = json.loads(args.teachers.read_text())
    teachers = teacher_payload.get("cases", [])
    if not 1 <= args.teacher_rank <= len(teachers):
        parser.error(f"--teacher-rank must be between 1 and {len(teachers)}")
    teacher_sequence = str(teachers[args.teacher_rank - 1]["sequence"])
    archive = pd.read_csv(args.archive)
    config = yaml.safe_load(args.config.read_text())
    requirements = config["nanobody_requirements"]
    target = str(config["protein_selection"]["nanobody"]["target"])
    if target != str(teacher_payload.get("target")):
        raise SystemExit("teacher target does not match live configuration")

    preserved: set[int] = set()
    for report in args.contacts:
        preserved.update(contact_positions(report))
    donor_frame = select_donors(
        archive,
        teacher_length=len(teacher_sequence),
        max_liability_score=args.max_donor_liability,
        limit=args.donors,
    )

    archive_sequences = set(
        archive.sequence.dropna().astype(str).str.strip().str.upper()
    )
    top = diversity.minmax_weighted_rank(
        archive[archive.liability_num_violations <= 50],
        k=int(requirements["num_top_sequences"]),
    )
    top_kmers = [diversity.kmers(sequence, 5) for sequence in top.sequence]
    threshold = float(requirements["max_similarity_score"])

    cases = []
    rejected: dict[str, int] = {}
    seen = set()
    for donor in donor_frame.to_dict("records"):
        try:
            sequence, backmutations = graft_sequence(
                teacher_sequence, donor["sequence"], preserved
            )
        except (AssertionError, ValueError, RuntimeError):
            reason = "imgt_graft"
        else:
            reason = ""
            if sequence in seen:
                reason = "duplicate_hybrid"
            elif sequence in archive_sequences:
                reason = "historical_exact"
            elif gates.check_sequence(sequence, requirements):
                reason = "layer1"
            else:
                diverse, max_jaccard = diversity.passes_diversity(
                    sequence, top_kmers, threshold=threshold, k=5
                )
                if not diverse:
                    reason = "top50_jaccard"
            if not reason:
                seen.add(sequence)
                cases.append({
                    "sequence": sequence,
                    "donor_sequence": donor["sequence"],
                    "donor_rank_sum": float(donor["donor_rank_sum"]),
                    "donor_liability_score": float(donor["liability_score"]),
                    "donor_liability_num_violations": int(
                        donor["liability_num_violations"]
                    ),
                    "mutation_count_from_teacher": sum(
                        left != right
                        for left, right in zip(sequence, teacher_sequence)
                    ) + abs(len(sequence) - len(teacher_sequence)),
                    "max_jaccard_to_top50": float(max_jaccard),
                    "contact_backmutations": backmutations,
                })
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1

    cases.sort(key=lambda item: (
        item["donor_rank_sum"], item["donor_liability_score"], item["sequence"]
    ))
    cases = cases[: args.limit]
    payload = {
        "purpose": "R&D IMGT paratope graft onto validator-proven frameworks",
        "target": target,
        "teacher_sequence": teacher_sequence,
        "teacher_rank": args.teacher_rank,
        "max_donor_liability": args.max_donor_liability,
        "donor_count": len(donor_frame),
        "preserved_contact_positions": sorted(preserved),
        "vernier_backmutation": True,
        "rejection_counts": rejected,
        "cases": cases,
    }
    _atomic_json(args.output, payload)
    print(
        f"DONOR_GRAFTS_READY target={target} donors={len(donor_frame)} "
        f"eligible={len(cases)} rejected={sum(rejected.values())} "
        f"output={args.output}"
    )
    for item in cases[:10]:
        print(
            f"  donor_liability={item['donor_liability_score']:.0f}/"
            f"{item['donor_liability_num_violations']} "
            f"mutations={item['mutation_count_from_teacher']} "
            f"jaccard={item['max_jaccard_to_top50']:.3f} {item['sequence']}"
        )


if __name__ == "__main__":
    main()
