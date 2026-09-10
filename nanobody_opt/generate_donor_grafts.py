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


CONSERVATIVE_GROUPS = tuple(map(frozenset, (
    "AVLIM",   # aliphatic/hydrophobic
    "FYW",     # aromatic
    "STNQ",    # uncharged polar
    "DE",      # acidic
    "KRH",     # basic/basic-like
    "AGST",    # small
    "NQ",      # amide
)))
VHH_HALLMARKS = frozenset({"H42", "H49", "H50", "H52"})


def conservative_substitution(left: str, right: str) -> bool:
    """Whether a donor change remains in a broad physicochemical class."""
    return left != right and any(
        left in group and right in group for group in CONSERVATIVE_GROUPS
    )


def _liability_priority(residue: str) -> int:
    """Order likely validator-liability repairs before neutral donor changes."""
    if residue in "KRW":
        return 10
    if residue == "M":
        return 5
    if residue in "NQ":
        return 2
    return 0


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


def scaffold_candidates(donors: pd.DataFrame) -> list[dict[str, Any]]:
    """Project archived rows into Layer-2 inputs for current gate replay."""
    keep = [
        "sequence", "donor_rank_sum", "liability_score",
        "liability_num_violations", *METRICS,
    ]
    # METRICS also contains both liability columns.  Preserve a stable,
    # duplicate-free projection so the output stays easy to audit.
    columns = list(dict.fromkeys(column for column in keep if column in donors))
    return donors[columns].to_dict("records")


def graft_sequence(
    teacher_sequence: str,
    donor_sequence: str,
    preserved_raw_positions: set[int],
) -> tuple[str, list[str], list[str]]:
    """Return an IMGT CDR/Vernier/contact graft and copied positions."""
    from abnumber import Chain

    teacher = Chain(teacher_sequence, scheme="imgt")
    donor = Chain(donor_sequence, scheme="imgt")
    numbered = [position for position, _ in teacher]
    # AbNumber only exposes Vernier classification under Kabat numbering. Map
    # those residues back by raw sequence index, then express the explicit
    # backmutations in the IMGT numbering used for the CDR graft itself.
    kabat_teacher = Chain(
        teacher_sequence, scheme="kabat", cdr_definition="kabat"
    )
    vernier_raw_positions = {
        index
        for index, (position, _) in enumerate(kabat_teacher, start=1)
        if position.is_in_vernier()
    }
    invalid = sorted(
        position for position in preserved_raw_positions
        if position < 1 or position > len(numbered)
    )
    if invalid:
        raise ValueError(f"contact positions outside teacher sequence: {invalid}")
    all_raw_positions = preserved_raw_positions | vernier_raw_positions
    backmutations = [
        numbered[position - 1] for position in sorted(all_raw_positions)
    ]
    grafted = teacher.graft_cdrs_onto(
        donor,
        backmutate_vernier=False,
        backmutations=backmutations,
        name="teacher_paratope_low_liability_framework",
    )
    return (
        grafted.seq,
        [str(numbered[position - 1]) for position in sorted(preserved_raw_positions)],
        [str(numbered[position - 1]) for position in sorted(vernier_raw_positions)],
    )


def progressive_grafts(
    teacher_sequence: str,
    donor_sequence: str,
    preserved_raw_positions: set[int],
    swap_counts: list[int],
) -> list[tuple[str, list[str]]]:
    """Apply conservative donor residues to a protected teacher framework."""
    from abnumber import Chain

    teacher = Chain(teacher_sequence, scheme="imgt")
    donor = Chain(donor_sequence, scheme="imgt")
    kabat_teacher = Chain(
        teacher_sequence, scheme="kabat", cdr_definition="kabat"
    )
    teacher_items = list(teacher)
    donor_by_position = dict(donor)
    vernier_raw_positions = {
        index
        for index, (position, _) in enumerate(kabat_teacher, start=1)
        if position.is_in_vernier()
    }
    protected_raw = preserved_raw_positions | vernier_raw_positions
    changes = []
    for raw_position, (position, source) in enumerate(teacher_items, start=1):
        target = donor_by_position.get(position)
        if (
            raw_position in protected_raw
            or position.is_in_cdr()
            or str(position) in VHH_HALLMARKS
            or target is None
            or not conservative_substitution(source, target)
        ):
            continue
        changes.append((
            -_liability_priority(source),
            raw_position,
            source,
            target,
            str(position),
        ))
    changes.sort()

    variants = []
    for count in sorted(set(swap_counts)):
        if count <= 0 or count > len(changes):
            continue
        sequence = list(teacher_sequence)
        mutations = []
        for _, raw_position, source, target, position in changes[:count]:
            sequence[raw_position - 1] = target
            mutations.append(f"{position}:{source}>{target}")
        variants.append(("".join(sequence), mutations))
    return variants


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
    parser.add_argument(
        "--mode", choices=("full", "progressive", "scaffolds"), default="full",
        help=("whole framework graft, conservative teacher-side donor swaps, "
              "or current-gate replay inputs for the selected donor scaffolds"),
    )
    parser.add_argument(
        "--swap-counts", default="2,4,6,8,10,12",
        help="comma-separated progressive framework mutation counts",
    )
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

    if args.mode == "scaffolds":
        cases = scaffold_candidates(donor_frame)
        payload = {
            "purpose": "R&D current-validator replay of low-liability scaffolds",
            "target": target,
            "mode": args.mode,
            "max_donor_liability": args.max_donor_liability,
            "donor_count": len(donor_frame),
            "cases": cases,
        }
        _atomic_json(args.output, payload)
        print(
            f"SCAFFOLDS_READY target={target} candidates={len(cases)} "
            f"output={args.output}"
        )
        return

    archive_sequences = set(
        archive.sequence.dropna().astype(str).str.strip().str.upper()
    )
    top = diversity.minmax_weighted_rank(
        archive[archive.liability_num_violations <= 50],
        k=int(requirements["num_top_sequences"]),
    )
    top_kmers = [diversity.kmers(sequence, 5) for sequence in top.sequence]
    threshold = float(requirements["max_similarity_score"])
    try:
        swap_counts = [
            int(value) for value in args.swap_counts.split(",") if value.strip()
        ]
    except ValueError as exc:
        parser.error(f"invalid --swap-counts: {exc}")
    if args.mode == "progressive" and not swap_counts:
        parser.error("--swap-counts must contain at least one integer")

    cases = []
    rejected: dict[str, int] = {}
    seen = set()
    for donor in donor_frame.to_dict("records"):
        try:
            if args.mode == "full":
                sequence, contact_backmutations, vernier_backmutations = (
                    graft_sequence(teacher_sequence, donor["sequence"], preserved)
                )
                variants = [(sequence, [], contact_backmutations,
                             vernier_backmutations)]
            else:
                variants = [
                    (sequence, mutations, [], [])
                    for sequence, mutations in progressive_grafts(
                        teacher_sequence, donor["sequence"], preserved, swap_counts
                    )
                ]
        except (AssertionError, ValueError, RuntimeError):
            reason = "imgt_graft"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        if not variants:
            reason = "no_eligible_swaps"
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        for sequence, mutations, contact_backmutations, vernier_backmutations in variants:
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
                    "contact_backmutations": contact_backmutations,
                    "vernier_backmutations": vernier_backmutations,
                    "swap_count": len(mutations),
                    "swaps": mutations,
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
        "mode": args.mode,
        "swap_counts": swap_counts if args.mode == "progressive" else [],
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
