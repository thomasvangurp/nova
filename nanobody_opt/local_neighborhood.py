"""Propose target-preserving nanobody neighborhoods for exact scoring.

Seeds come from a current Layer-2-qualified shortlist. They may be recent
archive designs or novel, exact-scored leaders that have not reached the public
archive yet. Proposals either make one to three supported substitutions inside
a single seed's CDRs or transplant one complete observed CDR motif. This avoids
the independent many-position recombination that destroyed binding geometry in
the 2026-09-09 probe.

There is deliberately no archive-proxy ordering here: that proxy had Spearman
rho 0.045 on the exact-scored synthetic probe. The output is balanced across
parents and proposal types for Layer-2 filtering followed by exact scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt import diversity, gates
from nanobody_opt.regions import find_regions
from nanobody_opt.simulate_round import dense_rank_sums

METRICS = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
    "liability_score": "min",
    "liability_num_violations": "min",
}


def _layer2_candidates(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    return [
        candidate
        for candidate in payload.get("candidates", [])
        if candidate.get("layer2_passed") and candidate.get("sequence")
    ]


def _rank_sum(frame: pd.DataFrame) -> np.ndarray:
    missing = set(METRICS) - set(frame.columns)
    if missing:
        raise ValueError(f"archive lacks exact metrics: {sorted(missing)}")
    return dense_rank_sums(frame, METRICS)


def _motif(sequence: str, bounds: tuple[int, int]) -> str:
    return sequence[bounds[0] : bounds[1]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--layer2-seeds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        type=Path,
        help="CSV containing a sequence column to exclude; may be repeated",
    )
    parser.add_argument("--recent", type=int, default=300)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--per-seed", type=int, default=32)
    parser.add_argument("--donor-count", type=int, default=100)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-mutations", type=int, default=1)
    parser.add_argument("--max-mutations", type=int, default=3)
    parser.add_argument("--motif-fraction", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    if not 1 <= args.min_mutations <= args.max_mutations:
        parser.error("require 1 <= --min-mutations <= --max-mutations")
    if not 0 <= args.motif_fraction <= 1:
        parser.error("--motif-fraction must be between zero and one")
    for name in ("recent", "seed_count", "per_seed", "donor_count", "min_support"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    requirements = yaml.safe_load(args.config.read_text())["nanobody_requirements"]
    frame = pd.read_csv(args.archive).drop_duplicates("sequence", keep="first")
    length = int(frame.sequence.str.len().mode()[0])
    frame = frame[frame.sequence.str.len() == length].reset_index(drop=True)
    recent = frame.tail(min(args.recent, len(frame))).reset_index(drop=True)
    recent = recent.assign(recent_rank_sum=_rank_sum(recent))

    top = diversity.minmax_weighted_rank(
        frame, k=int(requirements.get("num_top_sequences", 50))
    )
    top_kmers = [diversity.kmers(sequence, 5) for sequence in top.sequence]
    recent_scores = dict(zip(recent.sequence, recent.recent_rank_sum))
    eligible: list[tuple[float, str, float, str]] = []
    for candidate in _layer2_candidates(args.layer2_seeds):
        sequence = str(candidate["sequence"]).strip().upper()
        if len(sequence) != length:
            continue
        selection_score = candidate.get("selection_score")
        if selection_score is not None:
            parent_score = float(selection_score)
            parent_source = "exact_shortlist"
        elif sequence in recent_scores:
            parent_score = float(recent_scores[sequence])
            parent_source = "recent_archive"
        else:
            continue
        diverse, max_jaccard = diversity.passes_diversity(
            sequence, top_kmers, threshold=0.9, k=5
        )
        if diverse:
            eligible.append((parent_score, sequence, max_jaccard, parent_source))
    eligible.sort(key=lambda item: (item[0], item[1]))
    seeds = eligible[: args.seed_count]
    if len(seeds) < args.seed_count:
        raise SystemExit(
            f"only {len(seeds)} recent seeds passed current Layer-2 and diversity; "
            f"requested {args.seed_count}"
        )

    donor_sequences = [
        sequence
        for _, sequence, _, _ in eligible[: min(args.donor_count, len(eligible))]
    ]
    if not donor_sequences:
        raise SystemExit("no current-gate donor sequences found")

    allowed: list[list[str]] = []
    donor_array = np.array([list(sequence) for sequence in donor_sequences])
    for position in range(length):
        counts = pd.Series(donor_array[:, position]).value_counts()
        allowed.append([
            str(residue)
            for residue, count in counts.items()
            if count >= args.min_support and residue in gates.ALLOWED_AAS
        ])

    existing = set(frame.sequence)
    for path in args.exclude:
        excluded = pd.read_csv(path)
        if "sequence" not in excluded:
            raise SystemExit(f"exclude file has no sequence column: {path}")
        existing.update(excluded.sequence.dropna().astype(str).str.strip().str.upper())
    seen: set[str] = set()
    proposals: list[dict[str, object]] = []
    rng = np.random.default_rng(args.random_seed)

    for parent_score, parent, parent_jaccard, parent_source in seeds:
        regions = find_regions(parent)
        cdr_positions = list(regions.cdr_positions())
        bounds = [regions.cdr1, regions.cdr2, regions.cdr3]
        attempts = 0
        accepted = 0
        while accepted < args.per_seed and attempts < args.per_seed * 500:
            attempts += 1
            sequence = list(parent)
            source = "local_substitution"

            if rng.random() < args.motif_fraction:
                source = "whole_cdr_motif"
                cdr_index = int(rng.integers(3))
                start, end = bounds[cdr_index]
                compatible = []
                for donor in donor_sequences:
                    donor_regions = find_regions(donor)
                    donor_bounds = [
                        donor_regions.cdr1, donor_regions.cdr2, donor_regions.cdr3
                    ][cdr_index]
                    donor_motif = _motif(donor, donor_bounds)
                    if len(donor_motif) == end - start and donor_motif != parent[start:end]:
                        compatible.append(donor_motif)
                if not compatible:
                    continue
                sequence[start:end] = list(compatible[int(rng.integers(len(compatible)))])
            else:
                mutation_count = int(rng.integers(
                    args.min_mutations, args.max_mutations + 1
                ))
                mutable = [
                    position for position in cdr_positions
                    if any(residue != parent[position] for residue in allowed[position])
                ]
                if len(mutable) < mutation_count:
                    continue
                positions = rng.choice(mutable, size=mutation_count, replace=False)
                for position_value in positions:
                    position = int(position_value)
                    options = [
                        residue for residue in allowed[position]
                        if residue != parent[position]
                    ]
                    sequence[position] = options[int(rng.integers(len(options)))]

            candidate = "".join(sequence)
            if candidate in existing or candidate in seen:
                continue
            failures = gates.check_sequence(candidate, requirements)
            if failures:
                continue
            diverse, max_jaccard = diversity.passes_diversity(
                candidate, top_kmers, threshold=0.9, k=5
            )
            if not diverse:
                continue

            mutations = [
                f"{before}{index + 1}{after}"
                for index, (before, after) in enumerate(zip(parent, candidate))
                if before != after
            ]
            seen.add(candidate)
            proposals.append({
                "sequence": candidate,
                "parent_sequence": parent,
                "parent_selection_score": parent_score,
                "parent_source": parent_source,
                "parent_max_jaccard_to_top50": parent_jaccard,
                "proposal_type": source,
                "edit_distance": len(mutations),
                "mutations": ";".join(mutations),
                "max_jaccard_to_top50": max_jaccard,
            })
            accepted += 1

        if accepted < args.per_seed:
            raise SystemExit(
                f"only generated {accepted}/{args.per_seed} candidates for seed "
                f"with selection score {parent_score:.1f}"
            )

    output = pd.DataFrame(proposals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"eligible current-gate seeds: {len(eligible)}")
    print("selected seeds:")
    for score, sequence, max_jaccard, source in seeds:
        print(
            f"  score={score:.1f} source={source} "
            f"jaccard={max_jaccard:.3f} {sequence}"
        )
    print(f"wrote {len(output)} balanced proposals to {args.output}")
    print(output.groupby(["proposal_type", "edit_distance"]).size().to_string())


if __name__ == "__main__":
    main()
