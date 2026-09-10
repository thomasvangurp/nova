"""Propose target-preserving nanobody neighborhoods for exact scoring.

Seeds come from a current Layer-2-qualified shortlist. The primary lane keeps
the best two parents per completed epoch for CDR-local exploration. A second
family-neighbour lane retains lower-ranked near neighbours and deterministically
tests supported framework/terminal singles and coherent short terminal
haplotypes. This covers useful precursors such as the epoch-25037 winner's
rank-nine D-terminal parent without returning to unconstrained recombination.

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
from nanobody_opt.mutation_network import (
    MutationNetwork,
    load_mutation_effects,
    load_observations,
    mutation_contexts,
)
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


def _winner_records(path: Path | None, length: int) -> list[dict[str, object]]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    records = []
    for item in payload.get("winners", []):
        sequence = str(item.get("sequence", "")).strip().upper()
        score = item.get("epoch_rank_sum")
        if len(sequence) != length or score is None:
            continue
        records.append({
            "sequence": sequence,
            "score": float(score),
            "epoch": item.get("epoch"),
            "source": "public_epoch_winner",
        })
    return records


def _round_robin_by_parent(proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        grouped[str(proposal["parent_sequence"])].append(proposal)
    source_priority = {
        "c_terminal_single_substitution": 0,
        "exhaustive_single_substitution": 0,
        "c_terminal_haplotype": 1,
        "n_terminal_single_substitution": 2,
        "n_terminal_haplotype": 3,
        "framework_single_substitution": 4,
        "framework_haplotype": 5,
        "whole_cdr_motif": 6,
        "local_substitution": 7,
    }
    def ranking_key(item: dict[str, object]) -> tuple[object, ...]:
        return (
            float(item.get("predecessor_potential_score", math.inf)),
            float(item.get("predecessor_consensus_score", math.inf)),
            source_priority.get(str(item.get("proposal_type")), 9),
            -float(item["proposal_prior_score"]),
            int(item["edit_distance"]),
            str(item["sequence"]),
        )

    for parent, items in list(grouped.items()):
        singles = sorted(
            (item for item in items if int(item["edit_distance"]) == 1),
            key=ranking_key,
        )
        multi = sorted(
            (item for item in items if int(item["edit_distance"]) > 1),
            key=ranking_key,
        )
        interleaved = []
        for index in range(max(len(singles), len(multi))):
            if index < len(singles):
                interleaved.append(singles[index])
            if index < len(multi):
                interleaved.append(multi[index])
        grouped[parent] = interleaved
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


def predecessor_evidence(
    sequence: str,
    predecessors: list[tuple[float, str]],
    *,
    distance_penalty: float = 5.0,
    consensus_size: int = 5,
) -> dict[str, float | int]:
    """Summarize all prior nearby evidence for one proposed sequence."""
    if not predecessors:
        raise ValueError("predecessor evidence requires at least one parent")
    if distance_penalty < 0 or consensus_size < 1:
        raise ValueError("invalid predecessor evidence configuration")
    observations = []
    for score, predecessor in predecessors:
        if len(predecessor) != len(sequence):
            continue
        distance = sum(
            left != right
            for left, right in zip(sequence, predecessor, strict=True)
        )
        observations.append((float(score) + distance_penalty * distance, distance))
    if not observations:
        raise ValueError("no length-compatible predecessor exists")
    observations.sort()
    nearest = min(distance for _, distance in observations)
    return {
        "predecessor_min_distance": nearest,
        "predecessor_potential_score": observations[0][0],
        "predecessor_consensus_score": float(np.mean([
            adjusted for adjusted, _ in observations[:consensus_size]
        ])),
        "predecessor_support_within_six": sum(
            distance <= 6 for _, distance in observations
        ),
    }


def supported_framework_candidates(
    parent: str,
    *,
    allowed: list[list[str]],
    background_sequences: list[str],
    cdr_positions: set[int],
    protected: set[int],
    terminal_width: int = 17,
    maximum_haplotype_mutations: int = 3,
) -> list[tuple[str, str]]:
    """Enumerate supported framework singles and coherent terminal haplotypes."""
    length = len(parent)
    if len(allowed) != length:
        raise ValueError("allowed residue table has the wrong length")
    if terminal_width < 1 or maximum_haplotype_mutations < 2:
        raise ValueError("invalid terminal haplotype configuration")
    framework = set(range(length)) - cdr_positions - protected
    c_terminal = (
        set(range(max(0, length - terminal_width), length)) & framework
    )
    n_terminal = set(range(0, min(10, length))) & framework
    candidates: dict[str, str] = {}

    position_order = sorted(
        framework,
        key=lambda position: (
            0 if position == length - 1 else
            1 if position in c_terminal else
            2 if position in n_terminal else 3,
            position,
        ),
    )
    for position in position_order:
        source = (
            "c_terminal_single_substitution"
            if position in c_terminal
            else "n_terminal_single_substitution"
            if position in n_terminal
            else "framework_single_substitution"
        )
        for residue in allowed[position]:
            if residue == parent[position]:
                continue
            candidate = parent[:position] + residue + parent[position + 1 :]
            candidates.setdefault(candidate, source)

    compatible = [
        sequence
        for sequence in background_sequences
        if len(sequence) == length
    ]
    for donor in compatible:
        for positions, source in (
            (c_terminal, "c_terminal_haplotype"),
            (n_terminal, "n_terminal_haplotype"),
            (framework, "framework_haplotype"),
        ):
            changes = [
                position
                for position in positions
                if donor[position] != parent[position]
            ]
            if not 2 <= len(changes) <= maximum_haplotype_mutations:
                continue
            candidate = list(parent)
            for position in changes:
                candidate[position] = donor[position]
            candidates.setdefault("".join(candidate), source)
    return list(candidates.items())


def balanced_proposal_order(
    core: list[dict[str, object]],
    framework: list[dict[str, object]],
    *,
    limit: int,
    framework_fraction: float,
) -> list[dict[str, object]]:
    """Reserve both proposal families, then fill unused capacity from either."""
    if limit < 1 or not 0 <= framework_fraction <= 1:
        raise ValueError("invalid balanced proposal budget")
    framework_limit = round(limit * framework_fraction)
    core_limit = limit - framework_limit
    core_selected = core[:core_limit]
    framework_selected = framework[:framework_limit]
    core_remaining = core[core_limit:]
    framework_remaining = framework[framework_limit:]
    selected: list[dict[str, object]] = []
    for index in range(max(len(core_selected), len(framework_selected))):
        if index < len(core_selected):
            selected.append(core_selected[index])
        if index < len(framework_selected):
            selected.append(framework_selected[index])
    used = {str(item["sequence"]) for item in selected}
    for item in [*core_remaining, *framework_remaining]:
        sequence = str(item["sequence"])
        if sequence in used:
            continue
        selected.append(item)
        used.add(sequence)
        if len(selected) >= limit:
            break
    for priority, item in enumerate(selected[:limit], start=1):
        item["screen_priority"] = priority
    return selected[:limit]


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
        "--neighbor-seed-count",
        type=int,
        default=0,
        help="family-neighbour parents reserved for framework/terminal proposals",
    )
    parser.add_argument(
        "--minimum-seeds",
        type=int,
        default=4,
        help="minimum current-valid elite parents required to continue",
    )
    parser.add_argument("--per-seed", type=int, default=32)
    parser.add_argument(
        "--proposal-limit",
        type=int,
        default=384,
        help="Layer-2 oversubscribed pool before the 320-candidate exact cutoff",
    )
    parser.add_argument("--framework-fraction", type=float, default=0.5)
    parser.add_argument("--framework-per-seed", type=int, default=64)
    parser.add_argument("--terminal-width", type=int, default=17)
    parser.add_argument("--maximum-terminal-haplotype-mutations", type=int, default=6)
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
    parser.add_argument(
        "--mutation-observations",
        type=Path,
        help="persistent locally scored graph observations from earlier tranches",
    )
    parser.add_argument(
        "--mutation-network-output",
        type=Path,
        help="write the scored parent/winner/local mutation graph used for ranking",
    )
    args = parser.parse_args()

    if not 1 <= args.min_mutations <= args.max_mutations:
        parser.error("require 1 <= --min-mutations <= --max-mutations")
    if not 0 <= args.motif_fraction <= 1:
        parser.error("--motif-fraction must be between zero and one")
    if not 0 <= args.single_mutation_fraction <= 1:
        parser.error("--single-mutation-fraction must be between zero and one")
    if not 0 <= args.framework_fraction <= 1:
        parser.error("--framework-fraction must be between zero and one")
    if args.maximum_motif_mutations < 1:
        parser.error("--maximum-motif-mutations must be positive")
    for name in (
        "recent", "seed_count", "minimum_seeds", "per_seed", "donor_count",
        "min_support", "maximum_exhaustive_singles", "proposal_limit",
        "framework_per_seed", "terminal_width",
        "maximum_terminal_haplotype_mutations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_seeds > args.seed_count:
        parser.error("--minimum-seeds cannot exceed --seed-count")
    if args.neighbor_seed_count < 0:
        parser.error("--neighbor-seed-count cannot be negative")

    config_payload = yaml.safe_load(args.config.read_text())
    requirements = config_payload["nanobody_requirements"]
    target = str(config_payload["protein_selection"]["nanobody"]["target"])
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
    eligible: list[tuple[float, str, float, str, set[int], str]] = []
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
                str(candidate.get("parent_selection_lane") or "epoch_elite"),
            ))
    eligible.sort(key=lambda item: (item[0], item[1]))
    core_seeds = [
        item for item in eligible if item[5] == "epoch_elite"
    ][: args.seed_count]
    if len(core_seeds) < args.minimum_seeds:
        raise SystemExit(
            f"only {len(core_seeds)} elite seeds passed current Layer-2 and diversity; "
            f"require at least {args.minimum_seeds}"
        )
    neighbor_seeds = [
        item for item in eligible if item[5] == "family_neighbor"
    ][: args.neighbor_seed_count]
    seeds = [*core_seeds, *neighbor_seeds]

    donor_sequences = [
        sequence
        for _, sequence, _, _, _, _ in eligible[: min(args.donor_count, len(eligible))]
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

    network_records = [
        {
            "sequence": sequence,
            "score": score,
            "source": f"parent:{lane}",
        }
        for score, sequence, _, _, _, lane in eligible
    ]
    network_records.extend(_winner_records(args.winner_priors, length))
    if args.mutation_observations is not None:
        network_records.extend(
            item for item in load_observations(
                args.mutation_observations,
                target=target,
            )
            if len(str(item.get("sequence", ""))) == length
        )
    effects = (
        load_mutation_effects(args.mutation_observations, target=target)
        if args.mutation_observations is not None
        else []
    )
    mutation_network = MutationNetwork(
        network_records,
        mutation_effects=effects,
    )
    if args.mutation_network_output is not None:
        mutation_network.save(args.mutation_network_output)

    allowed: list[list[str]] = []
    donor_array = np.array([list(sequence) for sequence in donor_sequences])
    for position in range(length):
        counts = pd.Series(donor_array[:, position]).value_counts()
        allowed.append([
            str(residue)
            for residue, count in counts.items()
            if count >= args.min_support and residue in gates.ALLOWED_AAS
        ])

    framework_allowed: list[list[str]] = []
    background_array = np.array([list(sequence) for sequence in background_sequences])
    for position in range(length):
        counts = pd.Series(background_array[:, position]).value_counts()
        framework_allowed.append([
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
    core_proposals: list[dict[str, object]] = []
    framework_proposals: list[dict[str, object]] = []
    exhaustive_single_count = 0
    rng = np.random.default_rng(args.random_seed)

    def add_candidate(
        parent_record: tuple[float, str, float, str, set[int], str],
        candidate: str,
        source: str,
        proposal_lane: str,
        destination: list[dict[str, object]],
    ) -> bool:
        (
            parent_score,
            parent,
            parent_jaccard,
            parent_source,
            protected,
            parent_lane,
        ) = parent_record
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
        changed = [
            (index, before, after)
            for index, (before, after) in enumerate(
                zip(parent, candidate, strict=True)
            )
            if before != after
        ]
        mutation_weights = [priors[index][after] for index, _, after in changed]
        proposal_prior_score = float(
            math.exp(np.mean(np.log(mutation_weights)))
            if mutation_weights else 1.0
        )
        evidence = mutation_network.evidence(candidate)
        neighbours = evidence.pop("predecessor_neighbours")
        contexts = mutation_contexts(
            parent,
            candidate,
            contact_positions=protected,
        )
        seen.add(candidate)
        destination.append({
            "sequence": candidate,
            "parent_sequence": parent,
            "parent_selection_score": parent_score,
            "parent_source": parent_source,
            "parent_selection_lane": parent_lane,
            "proposal_lane": proposal_lane,
            "parent_max_jaccard_to_top50": parent_jaccard,
            "proposal_type": source,
            "edit_distance": len(changed),
            "mutations": ";".join(
                f"{before}{index + 1}{after}"
                for index, before, after in changed
            ),
            "mutation_contexts": json.dumps(contexts, sort_keys=True),
            "proposal_prior_score": proposal_prior_score,
            "protected_parent_contact_count": len(protected),
            "max_jaccard_to_top50": max_jaccard,
            **evidence,
            "predecessor_neighbours": json.dumps(neighbours, sort_keys=True),
        })
        return True

    for parent_record in core_seeds:
        (
            parent_score,
            parent,
            _,
            _,
            protected,
            _,
        ) = parent_record
        regions = find_regions(parent)
        cdr_positions = list(regions.cdr_positions())
        bounds = [regions.cdr1, regions.cdr2, regions.cdr3]
        attempts = 0

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
            if add_candidate(
                parent_record,
                candidate,
                "exhaustive_single_substitution",
                "core_cdr",
                core_proposals,
            ):
                exhaustive_single_count += 1
                if exhaustive_single_count > args.maximum_exhaustive_singles:
                    raise SystemExit(
                        "supported single-substitution lane exceeds "
                        f"--maximum-exhaustive-singles={args.maximum_exhaustive_singles}"
                    )

        stochastic_accepted = 0
        while (
            stochastic_accepted < args.per_seed
            and attempts < args.per_seed * 500
        ):
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
            if add_candidate(
                parent_record,
                candidate,
                source,
                "core_cdr",
                core_proposals,
            ):
                stochastic_accepted += 1

        if stochastic_accepted < args.per_seed:
            raise SystemExit(
                f"only generated {stochastic_accepted}/{args.per_seed} "
                "stochastic candidates for seed "
                f"with selection score {parent_score:.1f}"
            )

    for parent_record in seeds:
        parent = parent_record[1]
        protected = parent_record[4]
        regions = find_regions(parent)
        candidates = supported_framework_candidates(
            parent,
            allowed=framework_allowed,
            background_sequences=background_sequences,
            cdr_positions=set(regions.cdr_positions()),
            protected=protected,
            terminal_width=args.terminal_width,
            maximum_haplotype_mutations=(
                args.maximum_terminal_haplotype_mutations
            ),
        )
        singles = [item for item in candidates if "haplotype" not in item[1]]
        haplotypes = [item for item in candidates if "haplotype" in item[1]]
        lane_limit = args.framework_per_seed // 2
        selected_counts = {"single": 0, "haplotype": 0}
        leftovers: list[tuple[str, str]] = []
        for lane, items in (("single", singles), ("haplotype", haplotypes)):
            for candidate, source in items:
                if selected_counts[lane] >= lane_limit:
                    leftovers.append((candidate, source))
                    continue
                if add_candidate(
                    parent_record,
                    candidate,
                    source,
                    "framework_terminal",
                    framework_proposals,
                ):
                    selected_counts[lane] += 1
        accepted = sum(selected_counts.values())
        for candidate, source in leftovers:
            if accepted >= args.framework_per_seed:
                break
            if add_candidate(
                parent_record,
                candidate,
                source,
                "framework_terminal",
                framework_proposals,
            ):
                accepted += 1

    proposals = balanced_proposal_order(
        _round_robin_by_parent(core_proposals),
        _round_robin_by_parent(framework_proposals),
        limit=args.proposal_limit,
        framework_fraction=args.framework_fraction,
    )
    output = pd.DataFrame(proposals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"eligible current-gate seeds: {len(eligible)}")
    print("selected seeds:")
    for score, sequence, max_jaccard, source, protected, lane in seeds:
        print(
            f"  score={score:.1f} source={source} lane={lane} "
            f"jaccard={max_jaccard:.3f} protected_contacts={len(protected)} "
            f"{sequence}"
        )
    print(f"wrote {len(output)} balanced proposals to {args.output}")
    print(f"exhaustive supported single substitutions: {exhaustive_single_count}")
    print(output.groupby(["proposal_type", "edit_distance"]).size().to_string())


if __name__ == "__main__":
    main()
