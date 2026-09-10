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
from collections import Counter, defaultdict
import json
import math
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


def residue_prior_weights(
    background_sequences: list[str], winner_sequences: list[str], length: int
) -> list[dict[str, float]]:
    """Return smoothed winner-vs-background enrichment per residue/position."""
    if not background_sequences or not winner_sequences:
        return [defaultdict(lambda: 1.0) for _ in range(length)]
    amino_acids = sorted(gates.ALLOWED_AAS)
    priors: list[dict[str, float]] = []
    for position in range(length):
        background = Counter(sequence[position] for sequence in background_sequences)
        winners = Counter(sequence[position] for sequence in winner_sequences)
        position_weights = {}
        for residue in amino_acids:
            winner_probability = (winners[residue] + 1) / (
                len(winner_sequences) + len(amino_acids)
            )
            background_probability = (background[residue] + 1) / (
                len(background_sequences) + len(amino_acids)
            )
            position_weights[residue] = float(
                np.clip(winner_probability / background_probability, 0.10, 10.0)
            )
        priors.append(position_weights)
    return priors


def _weighted_choice(
    rng: np.random.Generator, values: list[str], weights: list[float]
) -> str:
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    return values[int(rng.choice(len(values), p=probabilities))]


def _winner_sequences(path: Path | None, length: int) -> list[str]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    return [
        str(item["sequence"]).strip().upper()
        for item in payload.get("winners", [])
        if len(str(item.get("sequence", "")).strip()) == length
    ]


def _round_robin_by_parent(proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        grouped[str(proposal["parent_sequence"])].append(proposal)
    for items in grouped.values():
        items.sort(key=lambda item: (
            0
            if item.get("proposal_type") == "exhaustive_single_substitution"
            else 1,
            -float(item["proposal_prior_score"]),
            int(item["edit_distance"]),
            str(item["sequence"]),
        ))
    ordered = []
    parents = sorted(
        grouped,
        key=lambda parent: (
            float(grouped[parent][0]["parent_selection_score"]), parent
        ),
    )
    for index in range(max(map(len, grouped.values()), default=0)):
        ordered.extend(
            grouped[parent][index]
            for parent in parents
            if index < len(grouped[parent])
        )
    return ordered


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
    parser.add_argument(
        "--minimum-seeds",
        type=int,
        default=4,
        help="minimum current-valid elite parents required to continue",
    )
    parser.add_argument("--per-seed", type=int, default=32)
    parser.add_argument("--donor-count", type=int, default=100)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-mutations", type=int, default=1)
    parser.add_argument("--max-mutations", type=int, default=3)
    parser.add_argument(
        "--single-mutation-fraction",
        type=float,
        default=0.70,
        help="fraction of local proposals reserved for one-residue substitutions",
    )
    parser.add_argument("--motif-fraction", type=float, default=0.20)
    parser.add_argument(
        "--maximum-motif-mutations",
        type=int,
        default=3,
        help="reject whole-CDR transfers that change more positions than this",
    )
    parser.add_argument(
        "--maximum-exhaustive-singles",
        type=int,
        default=160,
        help="fail closed if deterministic supported-single coverage exceeds this",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--winner-priors",
        type=Path,
        help="analyze_epoch_winners.py JSON used as a smoothed proposal prior",
    )
    args = parser.parse_args()

    if not 1 <= args.min_mutations <= args.max_mutations:
        parser.error("require 1 <= --min-mutations <= --max-mutations")
    if not 0 <= args.motif_fraction <= 1:
        parser.error("--motif-fraction must be between zero and one")
    if not 0 <= args.single_mutation_fraction <= 1:
        parser.error("--single-mutation-fraction must be between zero and one")
    if args.maximum_motif_mutations < 1:
        parser.error("--maximum-motif-mutations must be positive")
    for name in (
        "recent", "seed_count", "minimum_seeds", "per_seed", "donor_count",
        "min_support", "maximum_exhaustive_singles",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_seeds > args.seed_count:
        parser.error("--minimum-seeds cannot exceed --seed-count")

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
    eligible: list[tuple[float, str, float, str, set[int]]] = []
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
            pose = candidate.get("pose_stability") or {}
            protected = {
                int(position) - 1
                for position in pose.get(
                    "consensus_nanobody_contact_positions", []
                )
                if 1 <= int(position) <= length
            }
            eligible.append((
                parent_score,
                sequence,
                max_jaccard,
                parent_source,
                protected,
            ))
    eligible.sort(key=lambda item: (item[0], item[1]))
    seeds = eligible[: args.seed_count]
    if len(seeds) < args.minimum_seeds:
        raise SystemExit(
            f"only {len(seeds)} recent seeds passed current Layer-2 and diversity; "
            f"require at least {args.minimum_seeds}"
        )

    donor_sequences = [
        sequence
        for _, sequence, _, _, _ in eligible[: min(args.donor_count, len(eligible))]
    ]
    if not donor_sequences:
        raise SystemExit("no current-gate donor sequences found")
    background_sequences = [
        str(sequence).strip().upper()
        for sequence in recent.sequence
        if len(str(sequence).strip()) == length
    ]
    priors = residue_prior_weights(
        background_sequences,
        _winner_sequences(args.winner_priors, length),
        length,
    )

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
    exhaustive_single_count = 0
    rng = np.random.default_rng(args.random_seed)

    for parent_score, parent, parent_jaccard, parent_source, protected in seeds:
        regions = find_regions(parent)
        cdr_positions = list(regions.cdr_positions())
        bounds = [regions.cdr1, regions.cdr2, regions.cdr3]
        attempts = 0
        accepted = 0

        def add_candidate(candidate: str, source: str) -> bool:
            nonlocal accepted
            if candidate in existing or candidate in seen:
                return False
            failures = gates.check_sequence(candidate, requirements)
            if failures:
                return False
            diverse, max_jaccard = diversity.passes_diversity(
                candidate, top_kmers, threshold=0.9, k=5
            )
            if not diverse:
                return False
            mutations = [
                f"{before}{index + 1}{after}"
                for index, (before, after) in enumerate(zip(parent, candidate))
                if before != after
            ]
            mutation_weights = [
                priors[index][after]
                for index, (before, after) in enumerate(zip(parent, candidate))
                if before != after
            ]
            proposal_prior_score = float(
                math.exp(np.mean(np.log(mutation_weights)))
                if mutation_weights else 1.0
            )
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
                "proposal_prior_score": proposal_prior_score,
                "protected_parent_contact_count": len(protected),
                "max_jaccard_to_top50": max_jaccard,
            })
            accepted += 1
            return True

        # Deterministic coverage floor: every novel, supported, one-residue CDR
        # substitution is emitted before stochastic exploration.  The exact
        # race recognizes this proposal_type and scores the whole bounded lane.
        single_candidates = []
        for position in cdr_positions:
            if position in protected:
                continue
            for residue in allowed[position]:
                if residue == parent[position]:
                    continue
                candidate = parent[:position] + residue + parent[position + 1 :]
                single_candidates.append((
                    -priors[position][residue],
                    position,
                    residue,
                    candidate,
                ))
        for _, _, _, candidate in sorted(single_candidates):
            if add_candidate(candidate, "exhaustive_single_substitution"):
                exhaustive_single_count += 1
                if exhaustive_single_count > args.maximum_exhaustive_singles:
                    raise SystemExit(
                        "supported single-substitution lane exceeds "
                        f"--maximum-exhaustive-singles={args.maximum_exhaustive_singles}"
                    )

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
                        changed = {
                            position
                            for position in range(start, end)
                            if donor_motif[position - start] != parent[position]
                        }
                        if (
                            changed & protected
                            or len(changed) > args.maximum_motif_mutations
                        ):
                            continue
                        motif_weights = [
                            priors[position][donor_motif[position - start]]
                            for position in changed
                        ]
                        prior = (
                            math.exp(np.mean(np.log(motif_weights)))
                            if motif_weights else 1.0
                        )
                        compatible.append((donor_motif, prior))
                if not compatible:
                    continue
                selected_motif = _weighted_choice(
                    rng,
                    [motif for motif, _ in compatible],
                    [weight for _, weight in compatible],
                )
                sequence[start:end] = list(selected_motif)
            else:
                if (
                    args.min_mutations == 1
                    and rng.random() < args.single_mutation_fraction
                ):
                    mutation_count = 1
                else:
                    lower = max(2, args.min_mutations)
                    mutation_count = (
                        int(rng.integers(lower, args.max_mutations + 1))
                        if lower <= args.max_mutations
                        else args.min_mutations
                    )
                mutable = [
                    position for position in cdr_positions
                    if position not in protected
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
                    sequence[position] = _weighted_choice(
                        rng,
                        options,
                        [priors[position][residue] for residue in options],
                    )

            candidate = "".join(sequence)
            add_candidate(candidate, source)

        if accepted < args.per_seed:
            raise SystemExit(
                f"only generated {accepted}/{args.per_seed} candidates for seed "
                f"with selection score {parent_score:.1f}"
            )

    proposals = _round_robin_by_parent(proposals)
    output = pd.DataFrame(proposals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"eligible current-gate seeds: {len(eligible)}")
    print("selected seeds:")
    for score, sequence, max_jaccard, source, protected in seeds:
        print(
            f"  score={score:.1f} source={source} "
            f"jaccard={max_jaccard:.3f} protected_contacts={len(protected)} "
            f"{sequence}"
        )
    print(f"wrote {len(output)} balanced proposals to {args.output}")
    print(f"exhaustive supported single substitutions: {exhaustive_single_count}")
    print(output.groupby(["proposal_type", "edit_distance"]).size().to_string())


if __name__ == "__main__":
    main()
