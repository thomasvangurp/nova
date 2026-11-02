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
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import hashlib
import json
import zlib
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt import diversity, gates
from nanobody_opt.basins import basin_id, structural_core
from nanobody_opt.mutation_network import (
    MutationNetwork,
    load_mutation_effects,
    load_network_observations,
    mutation_contexts,
)
from nanobody_opt.regions import find_regions
from nanobody_opt.simulate_round import dense_rank_sums
from nanobody_opt.terminal_policy import (
    TerminalPriors,
    basin_parent_recency_ranks,
    learn_terminal_priors,
    terminal_route_priority,
)

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


def _cdr3_top50_basin(cdr3: str, top50_cdr3s: list[str] | None) -> bool:
    """True when a CDR3 sits at >= 0.85 positional identity to any archived
    top-50 clonotype -- the validator's ``is_duplicate`` CDR3 rule. Fail-closed:
    an empty clonotype index or an empty CDR3 is treated as in-basin (blocked),
    so a missing index can never admit a top-50-adjacent child."""
    if not cdr3 or not top50_cdr3s:
        return True
    length = len(cdr3)
    for other in top50_cdr3s:
        if not other:
            continue
        shared = sum(1 for a, b in zip(cdr3, other) if a == b)
        if shared / max(length, len(other)) >= 0.85:
            return True
    return False


def _rank_sum(frame: pd.DataFrame) -> np.ndarray:
    missing = set(METRICS) - set(frame.columns)
    if missing:
        raise ValueError(f"archive lacks exact metrics: {sorted(missing)}")
    return dense_rank_sums(frame, METRICS)


def _motif(sequence: str, bounds: tuple[int, int]) -> str:
    return sequence[bounds[0] : bounds[1]]


# Where the field concentrates on one residue, deviating from it costs score,
# and the cost switches on sharply. Measured over 735 scored tickets across 17
# epochs, as mean z of leaderboard rank within each epoch (negative is better):
#
#   field consensus   positions   agree     deviate   penalty
#            <= 0.4           4   +0.05      -0.03      -0.08   (no cost)
#              0.7            4   -0.01      +0.03      +0.04   (negligible)
#              0.8            3   -0.07      +0.25      +0.31   (real)
#           >= 0.9          116   -0.01      +0.65      +0.65   (severe)
#
# So 0.80 is where the field stops tolerating deviation, and it is read off the
# data rather than chosen. Below it the field genuinely varies and mutating is
# how we search; at or above it, a deviation is a defect.
FIELD_CONSENSUS_THRESHOLD = 0.80

# Winners do not sit ON the consensus, they sit one step off it. Across the six
# molecules that won an epoch for the strongest competing coldkey, each deviates
# from settled consensus at a mean of 1.00 positions, while our own fielded
# molecules deviate at 1.73. So consensus is a pull, never a rule: a hard gate
# at any budget forbids some winner outright -- freezing every settled position
# reaches 3 of those 6, a one-deviation budget 4, a two-deviation budget 5.
# Nothing here is forbidden; the prior only makes deviation less likely where
# the field has settled.
def settled_site_options(
    position: int,
    *,
    parent: str,
    consensus: dict,
    allowed: dict,
    anchor: int | None,
    repair_only: bool,
) -> list[str]:
    """Residues this position may take in a proposal.

    Under ``repair_only`` a SETTLED position may only keep the parent residue or
    be repaired toward consensus.  A new deviation at a settled position is
    never proposed, because it is the most expensive thing we do: measured over
    1,427 live tickets, 116 of 127 positions are settled at >=80% field
    consensus, each deviation at one costs about 7-12 rank points, and a molecule
    carrying none averages final_score 132.7 -- better than the leading rival's
    own mean of 134.2.  That rival carries 0.74 such deviations against our
    1.21, which is most of the 27.8-point gap between us.

    With ``repair_only`` false this reproduces the previous behaviour exactly.
    """
    residue_at, share_at = consensus.get(position, ("", 0.0))
    settled = share_at >= FIELD_CONSENSUS_THRESHOLD
    if repair_only and settled and position != anchor:
        options = [parent[position]]
        if (
            residue_at
            and residue_at != parent[position]
            and residue_at in allowed[position]
        ):
            options.append(residue_at)
        return options
    settled_here = settled and position != anchor
    options = [
        residue for residue in allowed[position]
        if residue != parent[position] or settled_here
    ]
    if position == anchor and settled and residue_at in options:
        # A settled anchor is only chosen when the parent deviates there, so
        # spend the one forced change on the repair, not a sideways move.
        options = [residue_at]
    return options


def new_settled_deviations(
    parent: str, candidate: str, consensus: dict
) -> int:
    """Settled positions the parent had right and the candidate breaks.

    Lane-agnostic on purpose.  The CDR draw is not the only way a residue
    changes -- motif grafts, framework lanes, terminal haplotypes and basin
    routes all build sequences by other paths, so a rule enforced only at the
    draw leaks through every one of them.  This counts the breakage itself.
    """
    broken = 0
    for position, (residue, share) in consensus.items():
        if share < FIELD_CONSENSUS_THRESHOLD or not residue:
            continue
        if position >= len(parent) or position >= len(candidate):
            continue
        if parent[position] == residue and candidate[position] != residue:
            broken += 1
    return broken


def field_consensus(
    background_sequences: list[str], length: int
) -> dict[int, tuple[str, float]]:
    """Positions where the field has settled on one residue, and which.

    Built from the recent archive -- every sequence the field actually fielded,
    not only the winners. A residue can be the field's settled answer without
    being enriched among winners, and that is exactly the case the enrichment
    prior below cannot see.
    """
    consensus: dict[int, tuple[str, float]] = {}
    usable = [s for s in background_sequences if len(s) == length]
    if not usable:
        return consensus
    for position in range(length):
        counts = Counter(sequence[position] for sequence in usable)
        residue, hits = counts.most_common(1)[0]
        consensus[position] = (residue, hits / len(usable))
    return consensus


def residue_prior_weights(
    background_sequences: list[str],
    winner_sequences: list[str],
    length: int,
    *,
    follow_field: bool = True,
    our_sequences: list[str] | None = None,
    minimum_winner_support: int = 3,
) -> list[dict[str, float]]:
    """Return per-residue/position weights: winner enrichment, plus the field.

    The enrichment term alone is blind to consensus. It is
    P(residue | winners) / P(residue | field), so a residue the whole field has
    settled on scores about 1.0 and gets no pull -- it is common in both halves
    of the ratio. Position 99, the first CDR3 residue, is the case in point: the
    field is 81% G there and we fielded G only 48% of the time, because nothing
    in the prior ever argued for it. Our tickets carrying A at 99 scored a median
    179 against 156 for G, and 144 against 125 on confidence rank.

    ``follow_field`` multiplies in the field's own composition so a settled
    residue is preferred on its own merits, not only when winners over-use it.

    ``our_sequences`` replaces the DENOMINATOR with what we ourselves fielded.
    Measured over 189 epochs, the field denominator is the wrong reference: it
    contains us, so our own habits partly cancel out of the correction, and a
    residue that winners and the field both favour scores ~1.0 and gets no pull
    at all. Positions 113, 115, 121 and 123 are exactly that case -- they carry
    100, 71, 121 and 115 win appearances and we under-use every one of them.
    Against our own distribution those same residues pull properly.

    ``minimum_winner_support`` is an ASYMMETRIC floor: a residue may only be
    promoted above 1.0 once at least this many winners actually carry it at this
    position. Demotion is never gated. Without it the ratio amplifies rare
    residues through the +1 smoothing -- 126 Q->H scored a 4.11 pull on ZERO
    win appearances while the top miners use it 14x less than we do, which is
    where our over-use of positions 0, 125 and 126 came from.
    """
    if not background_sequences or not winner_sequences:
        return [defaultdict(lambda: 1.0) for _ in range(length)]
    amino_acids = sorted(gates.ALLOWED_AAS)
    reference = [s for s in (our_sequences or []) if len(s) == length]
    priors: list[dict[str, float]] = []
    for position in range(length):
        background = Counter(sequence[position] for sequence in background_sequences)
        winners = Counter(sequence[position] for sequence in winner_sequences)
        # The denominator is what we want to be pulled AWAY from: our own
        # fielded mix when we have it, the field otherwise.
        ours = Counter(sequence[position] for sequence in reference) if reference \
            else background
        ours_total = len(reference) if reference else len(background_sequences)
        position_weights = {}
        for residue in amino_acids:
            winner_probability = (winners[residue] + 1) / (
                len(winner_sequences) + len(amino_acids)
            )
            reference_probability = (ours[residue] + 1) / (
                ours_total + len(amino_acids)
            )
            enrichment = winner_probability / reference_probability
            if winners[residue] < minimum_winner_support:
                # Not enough winners carry it to justify promotion; it may still
                # be demoted, which costs nothing if we are wrong.
                enrichment = min(enrichment, 1.0)
            if follow_field:
                # The field's own frequency, normalised so an average residue
                # scores 1.0 and the term only tilts, never rescales.
                field_probability = (background[residue] + 1) / (
                    len(background_sequences) + len(amino_acids)
                )
                enrichment *= field_probability * len(amino_acids)
            position_weights[residue] = float(np.clip(enrichment, 0.10, 10.0))
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


#: Whether proposal ordering prefers candidates that break no settled position.
#: Set from --settled-deviation-policy; "off" restores the pre-change ordering.
_PREFER_ADHERENTS = True


def _proposal_ranking_key(item: dict[str, object]) -> tuple[object, ...]:
    source_priority = {
        "c_terminal_single_substitution": 0,
        "exhaustive_single_substitution": 0,
        "c_terminal_haplotype": 1,
        "n_terminal_single_substitution": 2,
        "n_terminal_haplotype": 3,
        "framework_single_substitution": 4,
        "framework_haplotype": 5,
        "lineage_terminal_single_substitution": 1,
        "lineage_cdr_single_substitution": 1,
        "lineage_haplotype": 2,
        "basin_terminal_single_substitution": 0,
        "basin_terminal_haplotype": 1,
        "whole_cdr_motif": 6,
        "local_substitution": 7,
    }
    return (
        # Settled-site breakage outranks everything else.  Each one costs about
        # 14 rank points, so a candidate that breaks none is worth more than any
        # ordering refinement below it.  Ranking on this is what makes the rule
        # graded rather than binary: uids fill from candidates that adhere, and
        # only fall back to ones that deviate when the strict pool runs out --
        # and then to the least-deviating first.
        int(item.get("new_settled_deviations", 0)) if _PREFER_ADHERENTS else 0,
        int(item.get("parent_basin_rank", 10**9)),
        float(item.get("predecessor_potential_score", math.inf)),
        float(item.get("predecessor_consensus_score", math.inf)),
        float(item.get("parent_selection_score", math.inf)),
        source_priority.get(str(item.get("proposal_type")), 9),
        -float(item["proposal_prior_score"]),
        int(item["edit_distance"]),
        str(item["sequence"]),
    )


def _deduplicate_proposals(
    proposals: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep the strongest parent attribution for every proposed sequence.

    A sequence can be reachable from several elite families.  Generation must
    not reserve it for whichever parent happens to be visited first: that made
    a discarded per-parent overflow consume the sequence globally and rendered
    genuine held-out winners unreachable.  Rank all attributions first and
    deduplicate only after proposal evidence has been computed.
    """
    selected: dict[str, dict[str, object]] = {}
    for proposal in proposals:
        sequence = str(proposal["sequence"])
        incumbent = selected.get(sequence)
        if incumbent is None or _proposal_ranking_key(
            proposal
        ) < _proposal_ranking_key(incumbent):
            selected[sequence] = proposal
    return list(selected.values())


def _round_robin_by_parent(proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        grouped[str(proposal["parent_sequence"])].append(proposal)

    for parent, items in list(grouped.items()):
        singles = sorted(
            (item for item in items if int(item["edit_distance"]) == 1),
            key=_proposal_ranking_key,
        )
        multi = sorted(
            (item for item in items if int(item["edit_distance"]) > 1),
            key=_proposal_ranking_key,
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


def _round_robin_by_basin(proposals: list[dict[str, object]]) -> list[dict[str, object]]:
    """Allocate proposal turns to structural cores before individual parents."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        identifier = str(
            proposal.get("parent_basin_core_id")
            or proposal.get("basin_core_id")
            or proposal.get("parent_sequence")
        )
        grouped[identifier].append(proposal)
    for identifier, items in grouped.items():
        grouped[identifier] = _round_robin_by_parent(items)
    core_order = sorted(
        grouped,
        key=lambda identifier: (
            min(
                int(item.get("parent_basin_rank", 10**9))
                for item in grouped[identifier]
            ),
            min(
                float(item.get("parent_selection_score", math.inf))
                for item in grouped[identifier]
            ),
            identifier,
        ),
    )
    ordered: list[dict[str, object]] = []
    for index in range(max(map(len, grouped.values()), default=0)):
        ordered.extend(
            grouped[identifier][index]
            for identifier in core_order
            if index < len(grouped[identifier])
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


def supported_lineage_candidates(
    parent: str,
    *,
    allowed: list[list[str]],
    background_sequences: list[str],
    cdr_positions: set[int],
    protected: set[int],
    terminal_width: int = 17,
    maximum_haplotype_mutations: int = 6,
    priors: list[dict[str, float]] | None = None,
) -> list[tuple[str, str]]:
    """Enumerate the complete supported terminal/CDR lineage grammar.

    Singles cover every historically observed residue at every mutable CDR and
    terminal position.  Haplotype proposals preserve combinations that were
    observed together in a historical sequence, including combinations that
    span the N terminus, a CDR, and the C terminus.  The former framework lane
    considered those regions separately, which made mixed lineage descendants
    unreachable even when every constituent residue was already observed.

    ``priors`` orders the singles by residue weight instead of by position
    index.  This lane is the ONLY producer for CDR positions -- the framework
    lane subtracts them -- and its output was emitted in ascending position
    order, so any downstream truncation cut CDR3 (99-115) last after the
    N terminus and CDR1/CDR2.  Measured over 189 epochs: position 113 carries
    a prior weight of 9.57 against a 10.0 ceiling and still reached only 2.73%
    of generated mutations against 9.68% of winners', a 3.5x shortfall that
    selection could not close despite enriching it 2.56x, more than any other
    position.  Ordering by weight lets the prior reach CDR positions at all;
    without it a saturated weight there is inert.
    """
    length = len(parent)
    if len(allowed) != length:
        raise ValueError("allowed residue table has the wrong length")
    if terminal_width < 1 or maximum_haplotype_mutations < 2:
        raise ValueError("invalid lineage haplotype configuration")
    terminal = set(range(min(10, length))) | set(
        range(max(0, length - terminal_width), length)
    )
    mutable = (terminal | cdr_positions) - protected
    candidates: dict[str, str] = {}
    haplotypes: list[tuple[float, str]] = []

    singles: list[tuple[float, int, str, str, str]] = []
    for position in sorted(mutable):
        source = (
            "lineage_cdr_single_substitution"
            if position in cdr_positions
            else "lineage_terminal_single_substitution"
        )
        for residue in allowed[position]:
            if residue == parent[position]:
                continue
            candidate = parent[:position] + residue + parent[position + 1 :]
            weight = 1.0
            if priors is not None and position < len(priors):
                weight = float(priors[position].get(residue, 1.0))
            # position and residue break ties so the order stays deterministic
            singles.append((-weight, position, residue, candidate, source))
    singles.sort()
    for _, _, _, candidate, source in singles:
        candidates.setdefault(candidate, source)

    for donor in background_sequences:
        if len(donor) != length:
            continue
        changes = [
            position
            for position in sorted(mutable)
            if donor[position] != parent[position]
        ]
        if not 2 <= len(changes) <= maximum_haplotype_mutations:
            continue
        candidate = list(parent)
        for position in changes:
            candidate[position] = donor[position]
        haplotypes.append((
            -_mean_prior_weight(priors, changes, donor),
            "".join(candidate),
        ))
    haplotypes.sort()
    for _, candidate in haplotypes:
        candidates.setdefault(candidate, "lineage_haplotype")
    return list(candidates.items())



def _mean_prior_weight(
    priors: list[dict[str, float]] | None,
    positions: list[int],
    donor: str,
) -> float:
    """Mean residue weight a haplotype carries; 1.0 when there is no prior."""
    if priors is None or not positions:
        return 1.0
    weights = [
        float(priors[position].get(donor[position], 1.0))
        for position in positions
        if position < len(priors)
    ]
    return sum(weights) / len(weights) if weights else 1.0


def supported_basin_terminal_candidates(
    parent: str,
    *,
    basin_sequences: list[str],
    protected: set[int],
    exploration_residues: dict[int, frozenset[str]] | None = None,
    terminal_width: int = 17,
    maximum_haplotype_mutations: int = 6,
) -> list[tuple[str, str, str, tuple[int, ...]]]:
    """Enumerate terminal variants observed within one masked binding core.

    Single substitutions provide combinatorial coverage of residues already
    seen in the basin.  Short haplotypes preserve co-observed terminal residue
    combinations.  Candidates remain distinct: sharing a core allocates budget
    but does not imply equal validator behavior.

    Each route carries the positions that changed (vs. its stored base) as
    its 4th element -- known for free at construction (the single
    substitution's own `position`, or the haplotype loop's own `changes`),
    so route_key/terminal_route_priority don't have to rediscover it via a
    full string zip+enumerate per route. Measured live: that rediscovery was
    43.6s + 22.5s of self time across ~7.4M calls on one basin -- the actual
    dominant cost once the sort keys' own sha256 hashing was removed.
    """
    length = len(parent)
    if terminal_width < 1 or maximum_haplotype_mutations < 2:
        raise ValueError("invalid basin terminal configuration")
    compatible = [
        sequence
        for sequence in basin_sequences
        if len(sequence) == length and structural_core(sequence) == structural_core(parent)
    ]
    if not compatible:
        return []
    mutable = (
        set(range(min(10, length)))
        | set(range(max(0, length - terminal_width), length))
    ) - protected
    candidates: dict[str, tuple[str, str, tuple[int, ...]]] = {}
    # Every historical terminal haplotype is a legitimate parent inside an
    # already qualified basin.  Mutating only the selected representative
    # would still miss winners that are one step from a lower-ranked sibling.
    # Exhaust the supported one-step shell around every observed haplotype.
    supported_residues = {
        position: sorted(
            {sequence[position] for sequence in compatible}
            | set((exploration_residues or {}).get(position, ()))
        )
        for position in mutable
    }
    for base in compatible:
        for position in sorted(mutable):
            for residue in supported_residues[position]:
                if residue == base[position] or residue not in gates.ALLOWED_AAS:
                    continue
                candidate = base[:position] + residue + base[position + 1 :]
                candidates.setdefault(
                    candidate,
                    ("basin_terminal_single_substitution", base, (position,)),
                )
    for donor in compatible:
        changes = [
            position for position in sorted(mutable)
            if donor[position] != parent[position]
        ]
        if not 2 <= len(changes) <= maximum_haplotype_mutations:
            continue
        candidate = list(parent)
        for position in changes:
            candidate[position] = donor[position]
        candidates.setdefault(
            "".join(candidate),
            ("basin_terminal_haplotype", parent, tuple(changes)),
        )
    return [
        (candidate, source, base, changed)
        for candidate, (source, base, changed) in candidates.items()
    ]


def bounded_basin_terminal_routes(
    routes: list[tuple[str, str, str]],
    *,
    base_scores: dict[str, float],
    limit: int,
    exploration_fraction: float = 0.20,
    terminal_priors: TerminalPriors | None = None,
    evidence_fraction: float = 0.50,
) -> list[tuple[str, str, str]]:
    """Bound a basin shell with legacy diversity and prior-evidence lanes."""
    if (
        limit < 1
        or not 0 <= exploration_fraction <= 1
        or not 0 <= evidence_fraction <= 1
    ):
        raise ValueError("invalid basin route budget")
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for route in routes:
        grouped[route[2]].append(route)

    def route_key(route: tuple[str, str, str, tuple[int, ...]]) -> tuple[object, ...]:
        candidate, source, base, changed = route
        terminal_depth = min(
            (min(position + 1, len(candidate) - position) for position in changed),
            default=len(candidate),
        )
        return (
            0 if source == "basin_terminal_single_substitution" else 1,
            len(changed),
            terminal_depth,
            # A unique, deterministic tie-break -- same property a hash
            # gives (total order regardless of input order), without
            # hashing: the candidate string itself already compares
            # deterministically. Measured live: route_key is the sort key
            # for ~5M calls across one basin-representative loop (profiled
            # on real epoch 25639 data), and the sha256 hexdigest this
            # replaced was over 40% of that cost.
            candidate,
        )

    for base in grouped:
        grouped[base].sort(key=route_key)
    bases = sorted(
        grouped,
        key=lambda base: (base_scores.get(base, math.inf), base),
    )
    exploit_limit = round(limit * (1.0 - exploration_fraction))
    legacy: list[tuple[str, str, str]] = []
    used: set[str] = set()
    for index in range(max(map(len, grouped.values()), default=0)):
        for base in bases:
            if index >= len(grouped[base]):
                continue
            route = grouped[base][index]
            if route[0] in used:
                continue
            legacy.append(route)
            used.add(route[0])
            if len(legacy) >= exploit_limit:
                break
        if len(legacy) >= exploit_limit:
            break

    remaining = [
        route for route in routes if route[0] not in used
    ]
    # Deterministic pseudo-random exploration order, not a meaningful
    # ranking -- any hash that is fast, deterministic and well-distributed
    # satisfies the intent. sha256 over up to ~1.6M routes measured at
    # ~22.7s per bounded_basin_terminal_routes call (profiled on real
    # epoch 25639 data); crc32 is non-cryptographic but exactly as
    # deterministic, and orders of magnitude cheaper per call. This DOES
    # pick a different specific set of exploration candidates than the old
    # sha256 order would -- a different valid pseudo-random draw, not a
    # regression -- confirmed with the user before swapping.
    remaining.sort(key=lambda route: (
        zlib.crc32(f"explore:{route[0]}".encode("ascii")),
        route[0],
    ))
    for route in remaining:
        if route[0] in used:
            continue
        legacy.append(route)
        used.add(route[0])
        if len(legacy) >= limit:
            break
    if terminal_priors is None or evidence_fraction == 0:
        return legacy

    base_rank = basin_parent_recency_ranks(grouped, terminal_priors)
    evidence = sorted(
        routes,
        key=lambda route: terminal_route_priority(
            route[0],
            route[2],
            changed=route[3],
            priors=terminal_priors,
            base_rank=base_rank,
        ),
    )
    evidence_limit = round(limit * evidence_fraction)
    legacy_limit = limit - evidence_limit
    selected = legacy[:legacy_limit]
    selected_sequences = {route[0] for route in selected}
    for route in evidence:
        if route[0] in selected_sequences:
            continue
        selected.append(route)
        selected_sequences.add(route[0])
        if len(selected) >= legacy_limit + evidence_limit:
            break
    for route in legacy[legacy_limit:]:
        if route[0] in selected_sequences:
            continue
        selected.append(route)
        selected_sequences.add(route[0])
        if len(selected) >= limit:
            break
    return selected


#: Set once per worker by _init_seed_worker -- identical for every parent,
#: so each worker receives it once, not once per parent.
_SEED_FRAMEWORK_ALLOWED: list[list[str]] = []
_SEED_BACKGROUND_SEQUENCES: list[str] = []
_SEED_TERMINAL_WIDTH: int = 0
_SEED_MAX_HAPLOTYPE_MUTATIONS: int = 0
_SEED_PRIORS: list[dict[str, float]] | None = None


def _init_seed_worker(framework_allowed, background_sequences, terminal_width,
                      maximum_haplotype_mutations, priors) -> None:
    global _SEED_FRAMEWORK_ALLOWED, _SEED_BACKGROUND_SEQUENCES, \
        _SEED_TERMINAL_WIDTH, _SEED_MAX_HAPLOTYPE_MUTATIONS, _SEED_PRIORS
    _SEED_FRAMEWORK_ALLOWED = framework_allowed
    _SEED_BACKGROUND_SEQUENCES = background_sequences
    _SEED_TERMINAL_WIDTH = terminal_width
    _SEED_MAX_HAPLOTYPE_MUTATIONS = maximum_haplotype_mutations
    _SEED_PRIORS = priors


def _seed_candidates_for_parent(
    parent_record: tuple,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The expensive, side-effect-free half of main()'s two per-seed loops:
    supported_framework_candidates and supported_lineage_candidates are both
    pure functions (no closures, no shared state) independent per parent --
    only the subsequent add_candidate calls touch shared dedup state, and
    those stay sequential in main(), fed by this function's output. Computed
    together since both need the same find_regions(parent) call and the same
    background_sequences/framework_allowed inputs."""
    parent = parent_record[1]
    protected = parent_record[4]
    regions = find_regions(parent)
    cdr_positions = set(regions.cdr_positions())
    framework_candidates = supported_framework_candidates(
        parent,
        allowed=_SEED_FRAMEWORK_ALLOWED,
        background_sequences=_SEED_BACKGROUND_SEQUENCES,
        cdr_positions=cdr_positions,
        protected=protected,
        terminal_width=_SEED_TERMINAL_WIDTH,
        maximum_haplotype_mutations=_SEED_MAX_HAPLOTYPE_MUTATIONS,
    )
    lineage_candidates = supported_lineage_candidates(
        parent,
        allowed=_SEED_FRAMEWORK_ALLOWED,
        background_sequences=_SEED_BACKGROUND_SEQUENCES,
        cdr_positions=cdr_positions,
        protected=protected,
        terminal_width=_SEED_TERMINAL_WIDTH,
        maximum_haplotype_mutations=_SEED_MAX_HAPLOTYPE_MUTATIONS,
        priors=_SEED_PRIORS,
    )
    return framework_candidates, lineage_candidates



_BASIN_TERMINAL_PRIORS: Any = None
_BASIN_BASE_SCORES: dict[str, float] = {}
_BASIN_TERMINAL_WIDTH: int = 0
_BASIN_MAX_HAPLOTYPE_MUTATIONS: int = 0
_BASIN_ROUTES_PER_CORE: int = 0


def _init_basin_worker(terminal_priors, base_scores, terminal_width,
                       maximum_haplotype_mutations, basin_routes_per_core) -> None:
    global _BASIN_TERMINAL_PRIORS, _BASIN_BASE_SCORES, _BASIN_TERMINAL_WIDTH, \
        _BASIN_MAX_HAPLOTYPE_MUTATIONS, _BASIN_ROUTES_PER_CORE
    _BASIN_TERMINAL_PRIORS = terminal_priors
    _BASIN_BASE_SCORES = base_scores
    _BASIN_TERMINAL_WIDTH = terminal_width
    _BASIN_MAX_HAPLOTYPE_MUTATIONS = maximum_haplotype_mutations
    _BASIN_ROUTES_PER_CORE = basin_routes_per_core


def _basin_terminal_routes_for_parent(
    args: tuple[tuple, list[str]],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """The expensive, side-effect-free half of main()'s basin-representative
    loop: enumerating and bounding terminal routes for one parent. Nothing
    here touches add_candidate's shared dedup state (seen/lineage_attributions
    /pending_evidence) -- that stays sequential in main(), fed by this
    function's output, exactly as before. This is the O(compatible_sequences
    x mutable_positions x amino_acids) nested loop that dominates a basin
    representative's cost; add_candidate's own per-candidate work afterward
    is flat and bounded by basin_routes_per_core, so it is cheap by
    comparison and not worth parallelizing separately."""
    parent_record, basin_sequences = args
    parent = parent_record[1]
    protected = parent_record[4]
    legacy_routes = supported_basin_terminal_candidates(
        parent,
        basin_sequences=basin_sequences,
        protected=protected,
        terminal_width=_BASIN_TERMINAL_WIDTH,
        maximum_haplotype_mutations=_BASIN_MAX_HAPLOTYPE_MUTATIONS,
    )
    legacy_candidates = bounded_basin_terminal_routes(
        legacy_routes,
        base_scores=_BASIN_BASE_SCORES,
        limit=_BASIN_ROUTES_PER_CORE,
    )
    evidence_routes = supported_basin_terminal_candidates(
        parent,
        basin_sequences=basin_sequences,
        protected=protected,
        exploration_residues=_BASIN_TERMINAL_PRIORS.exploration_residues,
        terminal_width=_BASIN_TERMINAL_WIDTH,
        maximum_haplotype_mutations=_BASIN_MAX_HAPLOTYPE_MUTATIONS,
    )
    evidence_candidates = bounded_basin_terminal_routes(
        evidence_routes,
        base_scores=_BASIN_BASE_SCORES,
        limit=_BASIN_ROUTES_PER_CORE,
        terminal_priors=_BASIN_TERMINAL_PRIORS,
        evidence_fraction=1.0,
    )
    return legacy_candidates, evidence_candidates


def _interleave_rankings(
    primary: list[dict[str, object]],
    evidence: list[dict[str, object]],
    *,
    limit: int,
    evidence_fraction: float,
) -> list[dict[str, object]]:
    """Merge two complete rankings while reserving a prefix share for each."""
    if limit < 1 or not 0 <= evidence_fraction <= 1:
        raise ValueError("invalid evidence ranking budget")
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    indices = {"primary": 0, "evidence": 0}
    evidence_count = 0
    for position in range(1, limit + 1):
        choose_evidence = evidence_count < math.floor(
            position * evidence_fraction
        )
        lanes = (
            ((evidence, "evidence"), (primary, "primary"))
            if choose_evidence
            else ((primary, "primary"), (evidence, "evidence"))
        )
        appended = False
        for rows, lane in lanes:
            index = indices[lane]
            while index < len(rows) and str(rows[index]["sequence"]) in used:
                index += 1
            indices[lane] = index
            if index >= len(rows):
                continue
            item = rows[index]
            indices[lane] += 1
            selected.append(item)
            used.add(str(item["sequence"]))
            evidence_count += int(lane == "evidence")
            appended = True
            break
        if not appended:
            break
    return selected


def ordered_basin_proposals(
    proposals: list[dict[str, object]],
    *,
    evidence_fraction: float,
    evidence_core_limit: int = 1,
    top_core_weight: int = 2,
) -> list[dict[str, object]]:
    """Balance cores while mixing proven legacy and terminal-prior rankings."""
    if evidence_core_limit < 0 or top_core_weight < 1:
        raise ValueError("invalid basin allocation policy")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for proposal in proposals:
        grouped[str(proposal["parent_basin_core_id"])].append(proposal)
    for identifier, items in grouped.items():
        primary = _round_robin_by_parent([
            item for item in items
            if item.get("basin_route_policy") == "legacy"
        ])
        evidence = sorted(
            (
                item for item in items
                if item.get("basin_route_policy") == "winner_position"
            ),
            key=lambda item: (
                int(item.get("basin_position_rank", 10**9)),
                int(item.get("basin_residue_rank", 10**9)),
                int(item.get("basin_parent_recency_rank", 10**9)),
                str(item["sequence"]),
            ),
        )
        basin_rank = min(
            int(item.get("parent_basin_rank", 10**9)) for item in items
        )
        grouped[identifier] = _interleave_rankings(
            primary,
            evidence,
            limit=len(items),
            evidence_fraction=(
                evidence_fraction
                if basin_rank <= evidence_core_limit
                else 0.0
            ),
        )
    core_order = sorted(
        grouped,
        key=lambda identifier: (
            min(
                int(item.get("parent_basin_rank", 10**9))
                for item in grouped[identifier]
            ),
            identifier,
        ),
    )
    weighted_core_order = [
        identifier
        for index, identifier in enumerate(core_order)
        for _ in range(top_core_weight if index == 0 else 1)
    ]
    ordered: list[dict[str, object]] = []
    offsets = {identifier: 0 for identifier in core_order}
    while len(ordered) < sum(map(len, grouped.values())):
        added = False
        for identifier in weighted_core_order:
            index = offsets[identifier]
            if index >= len(grouped[identifier]):
                continue
            ordered.append(grouped[identifier][index])
            offsets[identifier] += 1
            added = True
        if not added:
            break
    return ordered


def _lane_limits(per_seed: int, haplotype_fraction: float) -> dict[str, int]:
    """Split one seed's framework_terminal/lineage budget by proven lane yield.

    Both callers previously split 50/50 regardless of measured outcome. Over
    469k screened proposals (epochs 25021-25412) the haplotype half of these
    two lanes landed in the top 5%% of its epoch's screen 13-18%% of the time,
    the single-substitution half 3-4%%. haplotype_fraction reserves the larger
    share for the lane that has actually earned it.
    """
    haplotype_limit = round(per_seed * haplotype_fraction)
    return {"haplotype": haplotype_limit, "single": per_seed - haplotype_limit}


#: The shared, read-only network for _evidence_worker. Set as a plain module
#: global right before the pool is created, never passed through initargs:
#: under the "fork" start method a worker inherits it via copy-on-write from
#: the parent's already-built object, instead of pickling a network built
#: from hundreds of thousands of observations into every worker.
_EVIDENCE_NETWORK: MutationNetwork | None = None


def _evidence_worker(item: tuple[str, str, set[int]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    """One candidate's evidence() plus mutation_contexts(): identical to what
    add_candidate used to compute inline, just off the one core the GIL
    otherwise confines it to."""
    parent, candidate, protected = item
    evidence = _EVIDENCE_NETWORK.evidence(candidate)
    neighbours = evidence.pop("predecessor_neighbours")
    evidence["predecessor_neighbours"] = json.dumps(neighbours, sort_keys=True)
    contexts = mutation_contexts(parent, candidate, contact_positions=protected)
    evidence["mutation_contexts"] = json.dumps(contexts, sort_keys=True)
    return evidence


def _resolve_pending_evidence(
    pending: list[tuple[dict[str, object], str, str, set[int]]],
    mutation_network: MutationNetwork,
    *,
    workers: int,
    workers_minimum_candidates: int,
) -> None:
    """Fill in every accepted candidate's evidence()/mutation_contexts() at
    once, across the pool if the batch is worth it. Mutates each record dict
    in place; order does not matter since results merge by list position."""
    if not pending:
        return
    items = [(parent, candidate, protected) for _, parent, candidate, protected in pending]
    # Set unconditionally: _evidence_worker reads this global on BOTH the pool
    # and the serial fallback path. Setting it only inside the pool branch left
    # the serial path crashing on None -- masked in testing because module
    # globals persist across test functions in one process, so an earlier
    # test's pool run had already populated it before the serial test ran.
    global _EVIDENCE_NETWORK
    _EVIDENCE_NETWORK = mutation_network
    pool_size = max(1, workers)
    if pool_size > 1 and len(items) >= workers_minimum_candidates:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=pool_size, mp_context=context) as pool:
            results = list(pool.map(
                _evidence_worker, items,
                chunksize=max(1, len(items) // (pool_size * 4)),
            ))
    else:
        results = [_evidence_worker(item) for item in items]
    for (record, _, _, _), evidence in zip(pending, results):
        record.update(evidence)


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


def interleave_novel_lane(
    primary: list[dict[str, object]],
    novel: list[dict[str, object]],
    *,
    limit: int,
    novel_fraction: float,
) -> list[dict[str, object]]:
    """Add a bounded discovery lane without replacing the proven primary head."""
    if limit < 1 or not 0 <= novel_fraction <= 1:
        raise ValueError("invalid novel-lane allocation")
    novel_limit = round(limit * novel_fraction)
    primary_limit = limit - novel_limit
    primary_selected = primary[:primary_limit]
    primary_sequences = {str(item["sequence"]) for item in primary}
    unique_novel = [
        item for item in novel
        if str(item["sequence"]) not in primary_sequences
    ]
    novel_selected = unique_novel[:novel_limit]
    primary_remaining = primary[primary_limit:]
    novel_remaining = unique_novel[novel_limit:]
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    primary_index = 0
    novel_index = 0
    novel_count = 0
    for position in range(1, limit + 1):
        desired_novel = math.floor(position * novel_fraction)
        choose_novel = novel_count < desired_novel
        sources = (
            ((novel_selected, "novel"), (primary_selected, "primary"))
            if choose_novel
            else ((primary_selected, "primary"), (novel_selected, "novel"))
        )
        appended = False
        for rows, lane in sources:
            index = novel_index if lane == "novel" else primary_index
            while index < len(rows) and str(rows[index]["sequence"]) in used:
                index += 1
            if lane == "novel":
                novel_index = index
            else:
                primary_index = index
            if index >= len(rows):
                continue
            item = rows[index]
            selected.append(item)
            used.add(str(item["sequence"]))
            if lane == "novel":
                novel_index += 1
                novel_count += 1
            else:
                primary_index += 1
            appended = True
            break
        if not appended:
            break
    for item in [*primary_remaining, *novel_remaining]:
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


#: The supported-single lane is bounded by the archive, which grows every epoch,
#: so any fixed ceiling eventually trips on nothing but age -- it did, at 8192,
#: and the producer then failed every iteration while looking healthy. Scaling
#: with the proposal budget keeps the guard meaningful (a true runaway is orders
#: of magnitude larger) without it expiring.
EXHAUSTIVE_SINGLES_BUDGET_MULTIPLE = 20

#: Printed when a seed's neighbourhood cannot supply the requested novel
#: candidates. The caller treats it as "stop generating", never as an error.
EXHAUSTED_MARKER = "NEIGHBOURHOOD_EXHAUSTED"

#: Sidecar written beside --output when the neighbourhood is used up.
EXHAUSTED_SUFFIX = ".exhausted.json"


def exhaustive_singles_ceiling(
    configured: int | None, proposal_limit: int
) -> int:
    """The lane ceiling: explicit when given, else scaled to the budget.

    An explicit value is validated here because making the flag default to
    ``None`` took it out of the argument parser's positivity loop.  A ceiling of
    zero parses, starts, and then raises on the first supported single of every
    iteration -- the same look-healthy-and-produce-nothing failure the scaling
    default exists to remove, so it is refused at startup instead.
    """
    if configured is not None:
        if int(configured) < 1:
            raise ValueError(
                "--maximum-exhaustive-singles must be positive: a ceiling of "
                f"{configured} fails the supported-single lane on its first "
                "candidate, every iteration, for the rest of the epoch"
            )
        return int(configured)
    return max(EXHAUSTIVE_SINGLES_BUDGET_MULTIPLE * int(proposal_limit), 160)


#: Two parents this close are one family for seeding purposes. A conservative
#: proxy for the validator's clonotype rule, which needs an annotator this
#: environment does not carry.
SEED_FAMILY_IDENTITY = 0.95


def _identity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shared = sum(1 for a, b in zip(left, right) if a == b)
    return shared / max(len(left), len(right))


#: Share of the seed budget reserved for families other than the leading one.
#: Not an even split: the best family has earned the bulk of the search, and
#: spreading the budget evenly starves it. But a family with no seats produces
#: no proposals, is never scored and disappears -- which is how the pool came
#: to hold one clonotype while the validator scored two tickets of seven.
#: Exploit the leader, keep the others alive.
#:
#: Raised 0.35 -> 0.50 (2026-10-21, developability hoist): the field converged
#: on one high-compactness CDR3 basin and the old 65/35 split starved every
#: other family (SEED_SHORTAGE selected=6 floor=24), so there were too few
#: in-range parents to reach the ~4800-accepted-children target.
EXPLORATION_SEED_SHARE = 0.50


def round_robin_families(
    seeds,
    limit,
    family_identity=SEED_FAMILY_IDENTITY,
    exploration_share=EXPLORATION_SEED_SHARE,
    cdr3_by_sequence=None,
):
    """Allocate the seed budget by CDR3 family, winner-first.

    The leading family (the winner's clonotype, first in seed order) receives
    ``1 - exploration_share`` of the seats while every other family keeps a
    two-seat floor, so no family silently starves. The former flat 2-per-family
    pass spread the winner across every historical family and diluted the
    anchor. Metadata appended to seed tuples is intentionally avoided: callers
    retain the established nine-field tuple. Unknown CDR3s do not earn a
    fictitious distinct clonotype reservation.
    """
    if limit <= 0 or not seeds:
        return []
    from nanobody_opt_v2.select_elite_parents import _same_cdr3_family
    families = {}
    for seed in seeds:
        loop = (cdr3_by_sequence or {}).get(seed[1])
        if loop is None:
            try:
                start, end = find_regions(seed[1]).cdr3
                loop = seed[1][start:end]
            except ValueError:
                continue
        if loop:
            identifier = next((representative for representative in families
                               if _same_cdr3_family(loop, representative)), loop)
            families.setdefault(identifier, []).append(seed)
    if not families:
        return []
    leading_representative = next(iter(families))
    leading = families[leading_representative]
    others = [members for representative, members in families.items()
              if representative != leading_representative]
    leading_seats = max(2, round(limit * (1.0 - exploration_share)))
    # True round-robin across families (one seat per family per pass). The
    # leading family still goes first each pass, but it is CAPPED at
    # `leading_seats` and can no longer backfill past every other family's
    # seats (Gap 4: the old backfill gave ~22 leading + 2 others of 24).
    taken = []
    families = [leading, *others]
    leading_count = 0
    pass_index = 0
    while len(taken) < limit:
        # GAP 4: process the OTHER families first, and only then the leader, so
        # the leader is never appended in a pass where every other family has
        # already exhausted (which is how a single singleton family still let
        # the leader reach ~66.7%).
        others_progressed = False
        for family_index, members in enumerate(families):
            if family_index == 0:
                continue
            if pass_index >= len(members):
                continue
            if len(taken) >= limit:
                break
            taken.append(members[pass_index])
            others_progressed = True
        if not others_progressed and leading_count >= 1:
            # Every other family is exhausted; do not backfill the leader.
            break
        progressed = others_progressed
        if leading_count < leading_seats and pass_index < len(leading) and len(taken) < limit:
            taken.append(leading[pass_index])
            leading_count += 1
            progressed = True
        if not progressed:
            break
        pass_index += 1
    return taken[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--layer2-seeds", required=True, type=Path)
    parser.add_argument(
        "--top50-index", type=Path, default=None,
        help="build_top50_index.json; supplies the top-50 CDR3 clonotypes for "
             "the clonotype-distance floor in add_candidate",
    )
    parser.add_argument(
        "--parents-may-match-top-sequences",
        action="store_true",
        default=True,
        help=(
            "use public parents even when they crowd the target's top "
            "sequences. On by default: they are proposal centres, never "
            "submissions, and excluding them dropped 7 separated families to 4"
        ),
    )
    parser.add_argument(
        "--no-parents-matching-top-sequences",
        dest="parents_may_match_top_sequences",
        action="store_false",
    )
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
        "--generation-workers",
        type=int,
        default=min(200, os.cpu_count() or 1),
        help="process pool size for resolving each accepted candidate's "
             "evidence()/mutation_contexts() (pure Python, GIL-bound to one "
             "core otherwise); 1 disables the pool and resolves serially",
    )
    parser.add_argument(
        "--generation-workers-minimum-candidates",
        type=int,
        default=64,
        help="use the pool only above this many accepted candidates; below "
             "it, process-start overhead outweighs the parallel gain",
    )
    parser.add_argument(
        "--proposal-limit",
        type=int,
        default=384,
        help="Layer-2 oversubscribed pool before the 320-candidate exact cutoff",
    )
    parser.add_argument("--bounded-generation", action="store_true",
                        help="honor the proposal limit without legacy floors or extra lane tails")
    parser.add_argument("--framework-fraction", type=float, default=0.5)
    parser.add_argument(
        "--haplotype-lane-fraction",
        type=float,
        default=0.5,
        help="share of each per-seed framework_terminal/lineage budget reserved "
             "for multi-mutation haplotype candidates over single substitutions. "
             "Measured on 469k screened proposals (epochs 25021-25412): haplotype "
             "variants land in the top 5%% of a screen 13-18%% of the time versus "
             "3-4%% for the single-substitution half of the same lanes.",
    )
    parser.add_argument(
        "--basin-fraction",
        type=float,
        default=0.70,
        help="proposal share reserved across the strongest masked structural cores",
    )
    parser.add_argument(
        "--lineage-fraction",
        type=float,
        default=0.2,
        help="bounded share reserved for the broader terminal/CDR lineage grammar",
    )
    parser.add_argument("--framework-per-seed", type=int, default=64)
    parser.add_argument(
        "--basin-routes-per-core",
        type=int,
        default=192,
        help="bounded pre-gate terminal routes generated for each reserved core",
    )
    parser.add_argument(
        "--basin-evidence-fraction",
        type=float,
        default=0.40,
        help=(
            "share of each basin prefix selected by historical winner-position "
            "and residue evidence"
        ),
    )
    parser.add_argument(
        "--basin-evidence-core-limit",
        type=int,
        default=1,
        help="highest basin rank receiving winner-position allocation",
    )
    parser.add_argument(
        "--basin-top-core-weight",
        type=int,
        default=2,
        help="proposal turns for basin rank one per turn for every other basin",
    )
    parser.add_argument(
        "--basin-exploration-residues-per-position",
        type=int,
        default=5,
        help="top public residue priors admitted beyond same-core alleles",
    )
    parser.add_argument("--terminal-width", type=int, default=17)
    parser.add_argument("--maximum-terminal-haplotype-mutations", type=int, default=6)
    parser.add_argument("--donor-count", type=int, default=100)
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help=(
            "minimum historical residue observations; one preserves rare "
            "lineage mutations while the graph ranker controls allocation"
        ),
    )
    parser.add_argument(
        "--settled-deviation-policy",
        choices=("repair-only", "prefer", "off"),
        default="repair-only",
        help=(
            "repair-only: never propose a NEW deviation at a position the field "
            "has settled (>=80%% consensus), AND order proposals by breakage.  "
            "prefer: order by breakage but leave the draw unrestricted.  off: "
            "neither -- the exact pre-change behaviour, which is the rollback "
            "lever.  Measured over 1,427 live tickets, each such deviation costs "
            "~14 rank points and a molecule carrying none averages 132.7, better "
            "than the leading rival's mean of 134.2.  Other lanes can still "
            "produce one, so every candidate records new_settled_deviations and "
            "the ordering prefers the lowest: uids fill from candidates that "
            "adhere and fall back to the least-deviating"
        ),
    )
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
        default=None,
        help="fail closed if deterministic supported-single coverage exceeds this; "
        "defaults to EXHAUSTIVE_SINGLES_BUDGET_MULTIPLE x --proposal-limit",
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
    if not 0 <= args.basin_fraction <= 1:
        parser.error("--basin-fraction must be between zero and one")
    if not 0 <= args.basin_evidence_fraction <= 1:
        parser.error("--basin-evidence-fraction must be between zero and one")
    if not 0 <= args.lineage_fraction <= 1:
        parser.error("--lineage-fraction must be between zero and one")
    if not 0 <= args.haplotype_lane_fraction <= 1:
        parser.error("--haplotype-lane-fraction must be between zero and one")
    if args.maximum_motif_mutations < 1:
        parser.error("--maximum-motif-mutations must be positive")
    try:
        args.maximum_exhaustive_singles = exhaustive_singles_ceiling(
            args.maximum_exhaustive_singles, args.proposal_limit
        )
    except ValueError as error:
        parser.error(str(error))

    for name in (
        "recent", "seed_count", "minimum_seeds", "per_seed", "donor_count",
        "min_support", "proposal_limit",
        "framework_per_seed", "terminal_width",
        "basin_routes_per_core",
        "basin_exploration_residues_per_position",
        "basin_top_core_weight",
        "maximum_terminal_haplotype_mutations",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.minimum_seeds > args.seed_count:
        parser.error("--minimum-seeds cannot exceed --seed-count")
    if args.neighbor_seed_count < 0:
        parser.error("--neighbor-seed-count cannot be negative")
    if args.basin_evidence_core_limit < 0:
        parser.error("--basin-evidence-core-limit cannot be negative")

    # 4800-novel-target override (2026-10-19). The producer passes
    # --basin-routes-per-core 192 and --proposal-limit 5000, which cap the
    # frontier single-edit lane at ~2.3k proposals (~1.3k novel/iteration and
    # ~3.6 iterations to fill the 4800 screen budget). The measured frontier
    # (5GhR-adjacent) single-edit pass rate is 78.5%, so ~6.2k basin
    # proposals are needed to yield ~4.8k novel in one iteration. Raise both
    # floors here; the downstream exact-race still caps its own screen at
    # --candidate-limit, so this only widens the novelty pool -- it cannot
    # over-screen. These are FLOORS: a higher explicit producer value wins.
    args.basin_routes_per_core = max(args.basin_routes_per_core, 600)
    if not args.bounded_generation:
        args.proposal_limit = max(args.proposal_limit, 8000)
    # Bias generation toward the terminal/haplotype lanes (2026-10-19): across
    # 25810-25812 the top molecules come from c_terminal_haplotype,
    # n_terminal_haplotype, framework_haplotype and lineage_terminal_single
    # (exact_shortlist), while core_cdr (exhaustive/local, top-50-anchored) is
    # the weak lane. Raise the framework share over core in the baseline.
    args.framework_fraction = max(args.framework_fraction, 0.8)

    config_payload = yaml.safe_load(args.config.read_text())
    requirements = config_payload["nanobody_requirements"]
    target = str(config_payload["protein_selection"]["nanobody"]["target"])
    frame = pd.read_csv(args.archive).drop_duplicates("sequence", keep="first")
    mode = int(frame.sequence.str.len().mode()[0])
    # Top-50 CDR3 clonotypes for the add_candidate clonotype-distance floor.
    # Empty list (no index) leaves the floor inactive here; the producer always
    # passes build_top50_index.json, and the commit-time check_validator_novelty
    # remains the authoritative backstop either way.
    top50_cdr3s: list[str] = []
    if args.top50_index is not None and args.top50_index.is_file():
        try:
            _top50_payload = json.loads(args.top50_index.read_text())
            top50_cdr3s = [
                str(item).strip().upper()
                for item in (_top50_payload.get("cdr3s") or [])
                if str(item).strip()
            ]
        except (OSError, ValueError, KeyError, TypeError):
            top50_cdr3s = []
    parent_candidates = _layer2_candidates(args.layer2_seeds)
    # Existing producer caches predate duplicate metadata. Re-admit ONLY the
    # pinned historical bank, with a fresh gate, rather than gating merged seeds.
    from nanobody_opt_v2.select_elite_parents import (
        _validator_parent_pool, _developability_parent_pool,
    )
    reservoir_path = Path("/root/artifacts/agreedfix_historical_parents.json")
    reservoir_payload = json.loads(reservoir_path.read_text())
    if reservoir_payload.get("target") not in (None, target):
        raise SystemExit("historical reservoir target mismatch")
    historical = _validator_parent_pool(reservoir_payload["candidates"], target=target)
    # GAP 5: profile the pinned historical bank's CDR developability (cached)
    # so an unprofiled parent is never silently treated as in-range; fail-closed
    # parents route to repair, not to CDR3-preserving lanes.
    historical = _developability_parent_pool(
        historical, target=target, cache_dir=Path("/root/layer2_cache"),
    )
    historical = [dict(c, historical_reservoir=True, parent_selection_lane="frontier")
                  for c in historical if c["parent_validator_duplicate"] is False
                  and c.get("layer2_passed")]
    known = {c["sequence"]: c for c in parent_candidates}
    known.update({c["sequence"]: c for c in historical})
    parent_candidates = list(known.values())
    args._seed_candidates = parent_candidates
    lengths = sorted({len(str(c["sequence"]).strip()) for c in parent_candidates
                      if abs(len(str(c["sequence"]).strip()) - mode) <= 5})
    outputs = []
    seed_shortages = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    original_output = args.output
    original_network_output = args.mutation_network_output
    total_budget = args.proposal_limit
    counts = {length: sum(len(c["sequence"]) == length for c in parent_candidates)
              for length in lengths}
    # Concentrate the generation budget on the winner's length (2026-10-20):
    # the anchor clonotype is a single length, and spreading single-edit
    # generation across every +/-5 context diluted it. Give the winner's length
    # the bulk share; the remaining lengths split the rest by parent count.
    ranked_parents = sorted(
        (c for c in parent_candidates if c.get("selection_score") is not None),
        key=lambda c: float(c["selection_score"]),
    )
    winner_length = len(str(ranked_parents[0]["sequence"])) if ranked_parents else mode
    # Reduced 0.70 -> 0.50 (developability hoist): the old bias starved every
    # other length context, further shrinking the already-thin in-range seed
    # pool; the winner's length no longer monopolizes the generation budget.
    winner_share = 0.50
    winner_budget = int(total_budget * winner_share) if winner_length in lengths else 0
    other_lengths = [length for length in lengths if length != winner_length]
    other_weight = sum(counts[length] for length in other_lengths) or 1
    budgets = {length: 0 for length in lengths}
    if winner_length in lengths:
        budgets[winner_length] = winner_budget
    for length in other_lengths:
        budgets[length] = (total_budget - winner_budget) * counts[length] // other_weight
    if lengths:
        # NOTE: budget_length is a NEW variable, not the protein target "P20809"
        # which must stay intact for load_network_observations / load_mutation_effects.
        budget_length = winner_length if winner_length in lengths else max(lengths, key=lambda value: counts[value])
        budgets[budget_length] += total_budget - sum(budgets.values())
    for length in lengths:
        args.proposal_limit = budgets[length]
        if args.proposal_limit == 0:
            continue
        exact_frame = frame[frame.sequence.str.len() == length].reset_index(drop=True)
        if exact_frame.empty:
            # Public, Layer-2-qualified parents provide an exact-length background
            # even if this archive snapshot has no rows of that length.
            rows = [dict(c.get("metrics") or {}, sequence=c["sequence"])
                    for c in parent_candidates if len(c["sequence"]) == length]
            exact_frame = pd.DataFrame(rows)
        args.output = original_output.with_name(original_output.stem + f".length{length}.csv")
        args.mutation_network_output = (original_network_output.with_name(
            original_network_output.stem + f".length{length}" + original_network_output.suffix)
            if original_network_output else None)
        seed_shortages.append(_generate_exact_length(args, exact_frame, length, requirements, target, top50_cdr3s))
        outputs.append(pd.read_csv(args.output))
    args.output = original_output
    args.proposal_limit = total_budget
    args.mutation_network_output = original_network_output
    # Each context has its own proposal budget; preserve its emitted candidates.
    # Interleave lengths and do not impose a second post-generation cap.
    rows = []
    for index in range(max((len(output) for output in outputs), default=0)):
        rows.extend(output.iloc[index].to_dict() for output in outputs if index < len(output))
    output = pd.DataFrame(rows).drop_duplicates("sequence", keep="first") if rows else pd.DataFrame(columns=["sequence"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {"mode_length": mode, "length_window": [mode - 5, mode + 5],
              "contexts": lengths, "context_budgets": budgets,
              "seed_shortage": not lengths or any(seed_shortages),
              "fallback_required": not lengths or any(seed_shortages),
              "emitted": len(output), "proposal_floor": args.proposal_limit,
              "shortage": max(0, args.proposal_limit - len(output))}
    Path(str(args.output) + ".shortage.json").write_text(json.dumps(report, indent=2))
    print("GENERATION_SHORTAGE " + json.dumps(report), flush=True)


def _append_proposal_tail(proposals, tail, *, limit=None):
    """Keep unbounded legacy headroom, or fill only the explicit nomination budget."""
    emitted = {item["sequence"] for item in proposals}
    for item in tail:
        if limit is not None and len(proposals) >= limit:
            break
        if item["sequence"] not in emitted:
            proposals.append(item)
            emitted.add(item["sequence"])
    return proposals


def _generate_exact_length(args, frame, length, requirements, target, top50_cdr3s):
    exhausted_seeds = []
    recent = frame.tail(min(args.recent, len(frame))).reset_index(drop=True)
    recent = recent.assign(recent_rank_sum=_rank_sum(recent))

    top = diversity.minmax_weighted_rank(
        frame, k=int(requirements.get("num_top_sequences", 50))
    )
    top_kmers = [diversity.kmers(sequence, 5) for sequence in top.sequence]
    recent_scores = dict(zip(recent.sequence, recent.recent_rank_sum))
    eligible: list[
        tuple[float, str, float, str, set[int], str, str, int, bool]
    ] = []
    seed_metadata = {str(c["sequence"]): c for c in args._seed_candidates}
    for candidate in seed_metadata.values():
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
        crowds_top, max_jaccard = diversity.passes_diversity(
            sequence, top_kmers, threshold=0.9, k=5
        )
        # A parent is a proposal CENTRE, never a submission -- select_elite_parents
        # says so outright: "Parent sequences are public and are therefore never
        # submitted." Gating parents on the submission novelty rule excluded the
        # field's best molecules from our search precisely because they rank
        # highly: of 7 mutually separated public parent families available on
        # epoch 25098, only 4 survived to propose from. The proposals themselves
        # are still novelty-checked, by the exclusion file here and by
        # check_validator_novelty before anything is committed.
        if crowds_top or args.parents_may_match_top_sequences:
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
                str(candidate.get("basin_core_id") or basin_id(sequence)),
                int(candidate.get("basin_rank") or 10**9),
                bool(candidate.get("basin_allocation_reserved", False)),
            ))
    eligible.sort(key=lambda item: (
        0 if item[8] else 1,
        item[7],
        item[0],
        item[1],
    ))
    eligible.sort(key=lambda item: seed_metadata[item[1]].get("parent_validator_duplicate") is not False)
    reserved_seeds = [item for item in eligible if item[8]]
    reserved_sequences = {item[1] for item in reserved_seeds}
    legacy_seeds = [
        item for item in eligible
        if item[5] in ("epoch_elite", "frontier") and item[1] not in reserved_sequences
    ]
    seed_pool = [*reserved_seeds, *legacy_seeds]
    # Winner-first, not historical-first (2026-10-20): the most-recent-epoch
    # anchor must lead the seed pool so round_robin_families gives its family
    # the bulk share. Preserve basin-reserved seeds ahead of legacy seeds.
    seed_pool.sort(key=lambda item: (0 if item[8] else 1, item[0], item[1]))
    nondup = [item for item in seed_pool if seed_metadata[item[1]].get("parent_validator_duplicate") is False]
    other = [item for item in seed_pool if seed_metadata[item[1]].get("parent_validator_duplicate") is not False]
    # First reserve CDR3 seats entirely within nonduplicates; fill from the rest.
    cdr3_by_sequence = {sequence: c.get("parent_cdr3") for sequence, c in seed_metadata.items()
                        if c.get("parent_cdr3")}
    # Reserve a share of the multi-edit seed budget for duplicate parents
    # (including the min-multipass anchor): their single-edits are all
    # duplicates, but their 2-3-edit children cross back under the novelty
    # threshold, so the multi-edit lane must actually reach them.
    dup_seed_budget = max(1, args.seed_count // 4)
    # GAP 4: one round_robin_families call over the combined pool (nondup
    # first, so the leading family still leads). The old two-call split let the
    # leading family double-dip its 50% seat cap in each call, reaching ~91%
    # of the budget and starving every other family. A single call applies the
    # cap once across both the nondup and dup/repair families.
    combined_pool = [*nondup, *other]
    core_seeds = round_robin_families(
        combined_pool, args.seed_count,
        cdr3_by_sequence=cdr3_by_sequence,
    )
    # GAP 2: guarantee the multi-edit CDR3-repair lane at least one repair
    # parent, independent of the round-robin -- a nondup-heavy bank can exhaust
    # the seat budget before any out-of-range family is reached.
    repair_pool = [
        item for item in other
        if seed_metadata[item[1]].get("parent_developability_repair")
    ]
    if repair_pool and not any(
        seed_metadata[seed[1]].get("parent_developability_repair")
        for seed in core_seeds
    ):
        core_seeds = [repair_pool[0], *core_seeds[: args.seed_count - 1]]
    seed_shortage = len(core_seeds) < max(24, args.minimum_seeds)
    if seed_shortage:
        print(f"SEED_SHORTAGE length={length} selected={len(core_seeds)} "
              f"floor={max(24, args.minimum_seeds)}; historical single-edit fallback enabled", flush=True)
    if not core_seeds:
        pd.DataFrame(columns=["sequence"]).to_csv(args.output, index=False)
        return True
    neighbor_seeds = [
        item for item in eligible
        if item[5] == "family_neighbor" and item[1] not in {seed[1] for seed in core_seeds}
    ][: args.neighbor_seed_count]
    seeds = [*core_seeds, *neighbor_seeds]
    # GAP 2c: the framework and lineage lanes preserve the parent CDR3, so an
    # out-of-range (repair) parent must not seed them -- only the multi-edit
    # lane may touch a repair parent, and only with a CDR3 mismatch.
    framework_lineage_seeds = [
        seed for seed in seeds
        if seed_metadata[seed[1]].get("parent_cdr_developability_in_range", False)
        and not seed_metadata[seed[1]].get("parent_top50_identity_in_basin", True)
    ]

    # Single-edit seed pool: the full core_seeds pool -- no <0.95 pre-filter.
    # The <0.95 identity filter previously applied here was backwards: it
    # removed exactly the >=0.95 parents that dominate the rival 5GhREcsK's
    # family (104/106 of his molecules are >=0.95 identical to the current
    # top-50), collapsing the single-edit lane's within-1-edit match to his
    # basin from 71.7% to 1.9%. Seed single-edits from the same full
    # eligible/core pool the multi-edit/haplotype lane uses, so the lane
    # re-anchors on the competitor/frontier parents instead of our stale
    # <0.95 tail. No selection_score elite-ordering is re-introduced: this
    # only removes the identity pre-filter.
    # Single-edits of a KNOWN duplicate parent (>=0.95 to the HF top-50)
    # reproduce >=0.95 children the commit-time gate blocks, so the single-edit
    # lane is restricted to the winning family and other gate-clearing parents;
    # duplicates stay in the multi-edit lane (a multi-edit child can cross back
    # under 0.95). Ungated (competitor) parents count as gate-clearing.
    single_edit_seeds = [
        seed for seed in core_seeds
        if seed_metadata[seed[1]].get("parent_validator_duplicate") is not True
        and seed_metadata[seed[1]].get("parent_cdr_developability_in_range", False)
        and not seed_metadata[seed[1]].get("parent_top50_identity_in_basin", True)
    ]

    donor_sequences = [
        item[1] for item in eligible[: min(args.donor_count, len(eligible))]
    ]
    background_sequences = [
        str(sequence).strip().upper()
        for sequence in recent.sequence
        if len(str(sequence).strip()) == length
    ]
    if not donor_sequences:
        # Degrade instead of aborting: the whole epoch must not die because the
        # current-gate donor pool came back empty. Fall back to the recent
        # archive as the donor pool so generation still proposes.
        donor_sequences = list(dict.fromkeys(background_sequences))[
            : args.donor_count
        ]
    if not donor_sequences:
        raise SystemExit("no donor sequences found (archive and gates both empty)")
    all_background_sequences = [
        str(sequence).strip().upper()
        for sequence in frame.sequence
        if len(str(sequence).strip()) == length
    ]
    background_by_basin: dict[str, list[str]] = defaultdict(list)
    for sequence in all_background_sequences:
        background_by_basin[basin_id(sequence)].append(sequence)
    priors = residue_prior_weights(
        background_sequences,
        _winner_sequences(args.winner_priors, length),
        length,
    )
    consensus = field_consensus(background_sequences, length)
    repair_only = args.settled_deviation_policy == "repair-only"
    # The ordering key is a SEPARATE effect from the draw restriction: it drives
    # deduplication, the per-lane limits and the proposal-limit truncation, so
    # leaving it on while the draw is unrestricted still changes which molecules
    # are emitted.  "off" disables both, which is what a rollback needs.
    global _PREFER_ADHERENTS
    _PREFER_ADHERENTS = args.settled_deviation_policy != "off"
    settled = sum(
        1 for _, share in consensus.values() if share >= FIELD_CONSENSUS_THRESHOLD
    )
    print(
        f"field consensus: {settled} of {length} positions are settled at "
        f">={FIELD_CONSENSUS_THRESHOLD:.0%}; mutation will not displace a parent "
        f"that already agrees with one",
        flush=True,
    )

    network_records = [
        {
            "sequence": sequence,
            "score": score,
            "source": f"parent:{lane}",
        }
        for score, sequence, _, _, _, lane, _, _, _ in eligible
    ]
    network_records.extend(_winner_records(args.winner_priors, length))
    if args.mutation_observations is not None:
        # Narrow, length-scoped, streamed read: the full load_observations
        # pull exceeds the MySQL read timeout on the central store.
        network_records.extend(load_network_observations(
            args.mutation_observations,
            target=target,
            length=length,
        ))
    effects = (
        load_mutation_effects(args.mutation_observations, target=target)
        if args.mutation_observations is not None
        else []
    )
    mutation_network = MutationNetwork(
        network_records,
        mutation_effects=effects,
    )
    terminal_priors = learn_terminal_priors(
        network_records,
        length=length,
        terminal_width=args.terminal_width,
        exploration_residues_per_position=(
            args.basin_exploration_residues_per_position
        ),
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
    lineage_attributions: set[tuple[str, str, str]] = set()
    # (proposal dict, parent sequence, candidate sequence, protected positions):
    # mutation_network.evidence() and mutation_contexts() are read-only queries
    # against the shared network/consensus, deferred here and resolved for
    # every accepted candidate at once in _resolve_pending_evidence, in a
    # process pool -- evidence() alone was 48% of a 148s generation
    # (neighbours() 18%, its numpy distance sum 30%), all Python-bytecode-
    # bound to one core regardless of the 256 available.
    pending_evidence: list[tuple[dict[str, object], str, str, set[int]]] = []
    core_proposals: list[dict[str, object]] = []
    framework_proposals: list[dict[str, object]] = []
    lineage_proposals: list[dict[str, object]] = []
    basin_proposals: list[dict[str, object]] = []
    exhaustive_single_count = 0
    rng = np.random.default_rng(args.random_seed)

    def add_candidate(
        parent_record: tuple[
            float, str, float, str, set[int], str, str, int, bool
        ],
        candidate: str,
        source: str,
        proposal_lane: str,
        destination: list[dict[str, object]],
        *,
        allow_seen_sequence: bool = False,
        extra_evidence: dict[str, object] | None = None,
    ) -> bool:
        (
            parent_score,
            parent,
            parent_jaccard,
            parent_source,
            protected,
            parent_lane,
            parent_basin_core_id,
            parent_basin_rank,
            basin_reserved,
        ) = parent_record
        attribution = (parent, candidate, proposal_lane)
        if (
            candidate in existing
            or (
                attribution in lineage_attributions
                if allow_seen_sequence else candidate in seen
            )
        ):
            return False
        failures = gates.check_sequence(candidate, requirements)
        if failures:
            return False
        # FIX 2: clonotype-distance floor. A child whose CDR3 sits within 0.85
        # positional identity of any archived top-50 clonotype would be rejected
        # by entry_unique_for_protein_hf regardless of which lane produced it, so
        # admit no such child. This is the single authoritative floor for every
        # neighbourhood lane (single-edit, framework, lineage, basin-terminal,
        # historical fallback, and multi-edit): they all funnel through here.
        if top50_cdr3s:
            try:
                child_regions = find_regions(candidate)
                child_cdr3 = candidate[child_regions.cdr3[0]:child_regions.cdr3[1]]
            except Exception:
                # Unannotatable candidate: fail-closed (treat as in-basin).
                return False
            if _cdr3_top50_basin(child_cdr3, top50_cdr3s):
                return False
        # Cheap k-mer/Jaccard "crowded family" gate DISABLED (2026-10-19):
        # the authoritative duplicate gate is check_validator_novelty /
        # utils.nanobodies.is_duplicate -- Algorithm-2 alignment identity
        # (reject at whole-sequence identity >= 0.95) plus the CDR3 clonotype
        # rule. A 5-mer Jaccard of 0.9 corresponds to a single edit (identity
        # ~0.98), far stricter than the validator's bar: the validator still
        # admits 0.933-identity (9-edit) children such as rival 5GhREcsK's
        # single-edits. The k-mer gate rejected near-top-50 children before
        # they could reach the authoritative Algorithm-2 gate, so it is
        # removed; check_validator_novelty remains the only novelty authority
        # before commit. max_jaccard is still recorded for observability.
        _, max_jaccard = diversity.passes_diversity(
            candidate, top_kmers, threshold=0.9, k=5
        )
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
        if allow_seen_sequence:
            lineage_attributions.add(attribution)
        else:
            seen.add(candidate)
        record: dict[str, object] = {
            "sequence": candidate,
            "parent_sequence": parent,
            "parent_selection_score": parent_score,
            "parent_source": parent_source,
            "parent_selection_lane": parent_lane,
            "parent_basin_core_id": parent_basin_core_id,
            "parent_basin_rank": parent_basin_rank,
            "basin_allocation_reserved": basin_reserved,
            "basin_core_id": basin_id(candidate),
            "proposal_lane": proposal_lane,
            "parent_max_jaccard_to_top50": parent_jaccard,
            "proposal_type": source,
            "new_settled_deviations": new_settled_deviations(
                parent, candidate, consensus
            ),
            "edit_distance": len(changed),
            "mutations": ";".join(
                f"{before}{index + 1}{after}"
                for index, before, after in changed
            ),
            "proposal_prior_score": proposal_prior_score,
            "protected_parent_contact_count": len(protected),
            "max_jaccard_to_top50": max_jaccard,
            **(extra_evidence or {}),
        }
        destination.append(record)
        pending_evidence.append((record, parent, candidate, protected))
        return True

    # Multi-edit / haplotype lane: the full diverse core_seeds pool, including
    # >=0.95 frontier elite (a multi-edit child of a near-boundary parent can
    # still cross below 0.95). Single-edit generation runs in the dedicated
    # single_edit_seeds loop below, seeded from the same full core pool.
    for parent_record in core_seeds:
        (
            parent_score,
            parent,
            _,
            _,
            protected,
            _,
            _,
            _,
            _,
        ) = parent_record
        regions = find_regions(parent)
        cdr_positions = list(regions.cdr_positions())
        bounds = [regions.cdr1, regions.cdr2, regions.cdr3]
        cdr3_start, cdr3_end = regions.cdr3
        parent_cdr_in_range = bool(
            seed_metadata.get(parent, {}).get("parent_cdr_developability_in_range", False)
        )
        parent_in_top50_basin = bool(
            seed_metadata.get(parent, {}).get("parent_top50_identity_in_basin", True)
        )
        # Repair path: a parent whose CDR-only red flags are out of range OR
        # whose CDR3 clonotype is already in the top-50 basin cannot seed a
        # CDR3-preserving proposal. Both route into the forced-CDR3 branch below.
        needs_cdr3_repair = (not parent_cdr_in_range) or parent_in_top50_basin
        attempts = 0

        stochastic_accepted = 0
        while (
            stochastic_accepted < args.per_seed
            and attempts < args.per_seed * 500
        ):
            attempts += 1
            sequence = list(parent)
            source = "local_substitution"

            if rng.random() < args.motif_fraction and parent_cdr_in_range and not parent_in_top50_basin:
                # Gap 5: same-length motif transfers must not bypass the CDR3
                # repair -- an out-of-range parent goes straight to the
                # multi-edit lane below, where a true CDR3 mismatch is forced.
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
                lower = max(2, args.min_mutations)
                mutation_count = (
                    int(rng.integers(lower, args.max_mutations + 1))
                    if lower <= args.max_mutations
                    else args.min_mutations
                )
                # GAP 5: --max-mutations 1 must not bypass CDR3 repair. An
                # out-of-range or top-50-basin parent needs at least a 2-edit
                # (one forced CDR3 mismatch + one more) to move off the blocked
                # clonotype; a single framework edit would preserve it.
                if needs_cdr3_repair and mutation_count < 2:
                    mutation_count = 2
                # TWEAK 2: a top-50-basin parent needs 3 CDR3 mismatches (14/17 =
                # 0.824 < 0.85) to escape the clonotype dedup; 2 is still a dup.
                if parent_in_top50_basin and mutation_count < 3:
                    mutation_count = 3
                mutable = [
                    position for position in cdr_positions
                    if position not in protected
                    if any(
                        residue != parent[position]
                        for residue in settled_site_options(
                            position,
                            parent=parent,
                            consensus=consensus,
                            allowed=allowed,
                            anchor=None,
                            repair_only=repair_only,
                        )
                    )
                ]
                if len(mutable) < mutation_count:
                    continue
                # CDR3 repair: a parent whose CDR-only red flags (L/L3/C) are
                # out of range produces doomed framework-only edits (the child
                # inherits the out-of-range compactness). Force one edit into
                # CDR3 so the child's compactness can move back into range.
                cdr3_mutable = [p for p in mutable if cdr3_start <= p < cdr3_end]
                if needs_cdr3_repair and mutation_count >= 2:
                    if not cdr3_mutable:
                        # GAP 5: an out-of-range parent with no mutable CDR3
                        # position cannot be repaired; reject it rather than
                        # emitting an unchanged-CDR3 (doomed) child.
                        continue
                    if parent_in_top50_basin:
                        # Escaping a top-50 clonotype needs a CDR3 that drops
                        # below the validator's 0.85 CDR3-similarity bar: 3
                        # mismatches (14/17 = 0.824) is the measured recipe
                        # (J-region tail + leading residue); 2 (0.882) is still
                        # a dup. Force 3 CDR3 edits where the CDR3 can hold them.
                        if len(cdr3_mutable) < 3:
                            continue
                        forced_count = max(mutation_count, 3)
                        forced_count = min(forced_count, len(cdr3_mutable))
                        forced_positions = list(
                            rng.choice(cdr3_mutable, size=forced_count, replace=False)
                        )
                        rest = [p for p in mutable if p not in forced_positions]
                        positions = (
                            list(rng.choice(rest, size=mutation_count - forced_count, replace=False))
                            if mutation_count > forced_count else []
                        )
                        positions.extend(forced_positions)
                    else:
                        forced = int(rng.choice(cdr3_mutable, size=1, replace=False)[0])
                        rest = [p for p in mutable if p != forced]
                        positions = list(rng.choice(rest, size=mutation_count - 1, replace=False))
                        positions.append(forced)
                else:
                    positions = list(rng.choice(mutable, size=mutation_count, replace=False))
                # One position must genuinely change, or the proposal can come
                # back identical to its parent and be rejected as already seen.
                # That is not hypothetical: the producer measures consensus
                # against the most recent 600 submissions, where 18 of 19 CDR3
                # positions sit above the threshold, so letting every sampled
                # position hold its parent residue drove acceptance to 1 in 12.
                # The anchor keeps the candidate novel; every OTHER sampled
                # position may still hold, which is what stops the one-way drift
                # off consensus.
                # Which position is forced to change matters as much as that one
                # is. Measured over 380 of the strongest competitor's
                # epoch-to-epoch mutations against 727 of ours: both of us put
                # ~80% of mutations on unsettled positions, so WHERE to mutate is
                # not the difference. The difference is DIRECTION at the settled
                # ones -- they move toward consensus 52% of the time and away
                # 43%, while we move away 69% and toward only 29%. They repair
                # deviations; we manufacture them, and end up at 1.18 deviations
                # per ticket against their 0.82 and the field's 0.97. Every extra
                # deviation costs about 0.05-0.10 of field percentile.
                #
                # So the anchor prefers, in order:
                #   1. an unsettled position, where novelty costs nothing;
                #   2. a settled position where the PARENT ALREADY DEVIATES, so
                #      the forced change is a correction toward consensus;
                #   3. anything left, rather than fail to propose.
                # Without this the anchor manufactures exactly the deviation the
                # rest of this change exists to avoid.
                def _anchor_rank(position: int) -> int:
                    residue, share = consensus.get(position, ("", 0.0))
                    if share < FIELD_CONSENSUS_THRESHOLD:
                        return 0
                    return 1 if parent[position] != residue else 2

                anchor = int(min(positions, key=lambda value: (_anchor_rank(int(value)), int(value))))
                if needs_cdr3_repair and mutation_count >= 2 and cdr3_mutable:
                    # GAP 5 + top-50 basin: the forced CDR3 position must BE the
                    # anchor, so the mutation application actually changes the CDR3
                    # residue and the child leaves the out-of-range compactness or
                    # the blocked top-50 clonotype.
                    if parent_in_top50_basin:
                        anchor = int(min(
                            forced_positions,
                            key=lambda value: (_anchor_rank(int(value)), int(value)),
                        ))
                    else:
                        anchor = int(forced)
                for position_value in positions:
                    position = int(position_value)
                    # Everywhere else a mutation must change something, or the
                    # candidate is a duplicate of its parent. But at a position
                    # the field has settled, excluding the parent's residue means
                    # sampling there can ONLY move off consensus -- one-way drift
                    # that took us to 48% G at position 99 against the field's
                    # 81%. Leaving the parent residue on the table lets the
                    # weighted draw decline to displace it, while the prior still
                    # allows the deliberate single deviation the winners carry.
                    options = settled_site_options(
                        position,
                        parent=parent,
                        consensus=consensus,
                        allowed=allowed,
                        anchor=anchor,
                        repair_only=repair_only,
                    )
                    # Multi-edit lane: force a TRUE substitution. If the settled
                    # draw would retain the parent residue (a settled position),
                    # re-draw from the allowed alternatives so the child genuinely
                    # changes -- a duplicate anchor's child must move away from
                    # the parent/top-50 to escape the novelty gate.
                    if len(options) == 1 and options[0] == parent[position]:
                        alternatives = [
                            residue for residue in allowed[position]
                            if residue != parent[position]
                        ]
                        if alternatives:
                            options = alternatives
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
            # Saturation, not breakage. Every iteration excludes the candidates
            # of all previous iterations, so a neighbourhood that has been
            # revisited enough times simply runs out of novel molecules within
            # one mutation of its seeds -- there is no twelfth sequence left to
            # propose. Observed live at 43 iterations in one epoch with 13,504
            # candidates already excluded, which is what a fast scorer plus a
            # small per-iteration budget produces.
            #
            # Failing here made the producer retry, so it spun on a full
            # neighbourhood at 0% GPU for the rest of the epoch. The remaining
            # time is worth far more spent replicating what we already have:
            # reliability goes 0.16 -> 0.61 across eight observations, while a
            # forty-fourth pass over an exhausted basin adds nothing.
            exhausted_seeds.append(
                f"{stochastic_accepted}/{args.per_seed} at score {parent_score:.1f}"
            )

    # Single-edit lane: one-residue substitutions seeded from the full
    # core_seeds pool (see single_edit_seeds above). The deterministic
    # exhaustive coverage floor runs first, then stochastic single
    # substitutions. The edit-position set is the CDR positions extended with
    # the two boundary/framework positions the rival 5GhREcsK's QWALVES basin
    # differs from our core at, 46 and 61 (1-indexed): 61 (0-indexed 60) is
    # already inside CDR2, and 46 (0-indexed 45) is the FR2 residue on the
    # CDR2 boundary, added here so the lane can anchor on his basin.
    for parent_record in single_edit_seeds:
        (
            parent_score,
            parent,
            _,
            _,
            protected,
            _,
            _,
            _,
            _,
        ) = parent_record
        regions = find_regions(parent)
        cdr_positions = sorted(set(regions.cdr_positions()) | {45, 60})
        attempts = 0

        single_candidates = []
        for position in cdr_positions:
            if position in protected:
                continue
            for residue in settled_site_options(
                position,
                parent=parent,
                consensus=consensus,
                allowed=allowed,
                anchor=None,
                repair_only=repair_only,
            ):
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
            mutable = [
                position for position in cdr_positions
                if position not in protected
                if any(
                    residue != parent[position]
                    for residue in settled_site_options(
                        position,
                        parent=parent,
                        consensus=consensus,
                        allowed=allowed,
                        anchor=None,
                        repair_only=repair_only,
                    )
                )
            ]
            if not mutable:
                continue
            position = int(rng.choice(mutable, size=1, replace=False)[0])
            options = settled_site_options(
                position,
                parent=parent,
                consensus=consensus,
                allowed=allowed,
                anchor=position,
                repair_only=repair_only,
            )
            candidate = (
                parent[:position]
                + _weighted_choice(
                    rng,
                    options,
                    [priors[position][residue] for residue in options],
                )
                + parent[position + 1 :]
            )
            if add_candidate(
                parent_record,
                candidate,
                "local_substitution",
                "core_cdr",
                core_proposals,
            ):
                stochastic_accepted += 1

        if stochastic_accepted < args.per_seed:
            exhausted_seeds.append(
                f"{stochastic_accepted}/{args.per_seed} at score {parent_score:.1f}"
            )

    if exhausted_seeds:
        # A sidecar FILE, not only a printed marker: the producer streams child
        # output straight to its terminal rather than capturing it, so a caller
        # has nothing to grep. A file beside the output is the only signal that
        # survives.
        print(
            f"{EXHAUSTED_MARKER} {len(exhausted_seeds)} seed(s) short: "
            + "; ".join(exhausted_seeds[:4]),
            flush=True,
        )
        exhausted_path = Path(str(args.output) + EXHAUSTED_SUFFIX)
        exhausted_path.parent.mkdir(parents=True, exist_ok=True)
        exhausted_path.write_text(
            json.dumps({"seeds_short": exhausted_seeds, "per_seed": args.per_seed},
                       indent=2, sort_keys=True)
        )

    base_scores = {node.sequence: node.score for node in mutation_network.nodes}
    basin_representatives: dict[str, tuple[
        float, str, float, str, set[int], str, str, int, bool
    ]] = {}
    reserved_core_seeds = [item for item in core_seeds if item[8]]
    for parent_record in reserved_core_seeds or core_seeds:
        # GAP 2/5: basin-terminal routes mutate only terminal (non-CDR)
        # residues and preserve the CDR3, so an out-of-range (repair) parent
        # must not seed them -- only the multi-edit lane may repair its CDR3.
        # Fail-closed: an UNPROFILED parent (field absent) is treated as
        # out-of-range, never silently in-range.
        if not seed_metadata[parent_record[1]].get(
            "parent_cdr_developability_in_range", False
        ):
            continue
        # Top-50-basin parents are also excluded: terminal-only edits keep the
        # blocked CDR3 clonotype, so the child is rejected by
        # entry_unique_for_protein_hf exactly like the parent. Absent flag =
        # in-basin (fail-closed) so an unstamped parent never reopens the lane.
        if seed_metadata[parent_record[1]].get("parent_top50_identity_in_basin", True):
            continue
        basin_representatives.setdefault(parent_record[6], parent_record)
    # The nested enumeration inside supported_basin_terminal_candidates (every
    # historical basin member x every mutable position x every observed
    # residue, built via pure-Python string slicing) is independent per
    # parent and the single biggest cost in this script -- measured live at
    # 12m50s on a 224-core box using exactly 1 core. add_candidate's own
    # per-candidate work below is flat and bounded by basin_routes_per_core,
    # so only the route enumeration is dispatched to a pool; the dedup state
    # it feeds (seen/lineage_attributions/pending_evidence) stays sequential,
    # processed in the same parent order as the old loop.
    basin_parents = list(basin_representatives.values())
    basin_worker_inputs = [
        (parent_record, background_by_basin[parent_record[6]])
        for parent_record in basin_parents
    ]
    # --generation-workers-minimum-candidates (64) is the OTHER pool's
    # threshold, sized for per-CANDIDATE evidence resolution where there can
    # be thousands of cheap items. Live basin_representatives is typically a
    # handful of parents (observed: 8) -- each individually expensive (the
    # O(compatible x positions x residues) enumeration above), not numerous.
    # Gating on that threshold meant this pool never activated in production;
    # gate on parent count alone instead -- fork overhead (~1-2s) is trivial
    # against even a single parent's multi-second-to-minutes cost.
    if args.generation_workers > 1 and len(basin_parents) > 1:
        with ProcessPoolExecutor(
            max_workers=min(args.generation_workers, len(basin_parents)),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_basin_worker,
            initargs=(terminal_priors, base_scores, args.terminal_width,
                      args.maximum_terminal_haplotype_mutations,
                      args.basin_routes_per_core),
        ) as pool:
            basin_route_results = list(
                pool.map(_basin_terminal_routes_for_parent, basin_worker_inputs)
            )
    else:
        _init_basin_worker(terminal_priors, base_scores, args.terminal_width,
                           args.maximum_terminal_haplotype_mutations,
                           args.basin_routes_per_core)
        basin_route_results = [
            _basin_terminal_routes_for_parent(item) for item in basin_worker_inputs
        ]

    for parent_record, (legacy_candidates, evidence_candidates) in zip(
        basin_parents, basin_route_results
    ):
        parent = parent_record[1]
        basin_sequences = background_by_basin[parent_record[6]]
        parent_recency_rank = basin_parent_recency_ranks(
            basin_sequences,
            terminal_priors,
        )
        parent_proposals: list[dict[str, object]] = []
        for route_policy, candidates in (
            ("legacy", legacy_candidates),
            ("winner_position", evidence_candidates),
        ):
            for candidate, source, base_parent, changed in candidates:
                route_priority = terminal_route_priority(
                    candidate,
                    base_parent,
                    changed=changed,
                    priors=terminal_priors,
                    base_rank=parent_recency_rank,
                )
                attributed_parent = (
                    parent_record
                    if base_parent == parent
                    else (
                        base_scores.get(base_parent, parent_record[0]),
                        base_parent,
                        parent_record[2],
                        "historical_basin_member",
                        parent_record[4],
                        parent_record[5],
                        parent_record[6],
                        parent_record[7],
                        parent_record[8],
                    )
                )
                add_candidate(
                    attributed_parent,
                    candidate,
                    source,
                    "basin_terminal",
                    parent_proposals,
                    allow_seen_sequence=True,
                    extra_evidence={
                        "basin_route_policy": route_policy,
                        "basin_position_rank": route_priority[0],
                        "basin_residue_rank": route_priority[1],
                        "basin_parent_recency_rank": route_priority[2],
                        "basin_prior_public_epochs": (
                            terminal_priors.public_epoch_count
                        ),
                    },
                )
        parent_proposals = sorted(
            _deduplicate_proposals(parent_proposals), key=_proposal_ranking_key
        )
        basin_proposals.extend(parent_proposals)

    # supported_framework_candidates and supported_lineage_candidates are
    # both pure functions independent per parent (same reasoning as the
    # basin-terminal routes above); computed together here since both loops
    # below iterate the same `seeds` and need the same inputs. Only the
    # subsequent add_candidate calls in each loop touch shared dedup state,
    # so those stay sequential, fed by this pool's precomputed output.
    if args.generation_workers > 1 and len(framework_lineage_seeds) > 1:
        with ProcessPoolExecutor(
            max_workers=min(args.generation_workers, len(framework_lineage_seeds)),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_init_seed_worker,
            initargs=(framework_allowed, background_sequences,
                      args.terminal_width, args.maximum_terminal_haplotype_mutations,
                      priors),
        ) as pool:
            seed_results = list(pool.map(_seed_candidates_for_parent, framework_lineage_seeds))
    else:
        _init_seed_worker(framework_allowed, background_sequences,
                          args.terminal_width, args.maximum_terminal_haplotype_mutations,
                          priors)
        seed_results = [_seed_candidates_for_parent(item) for item in framework_lineage_seeds]

    for parent_record, (candidates, _lineage_candidates) in zip(framework_lineage_seeds, seed_results):
        singles = [item for item in candidates if "haplotype" not in item[1]]
        haplotypes = [item for item in candidates if "haplotype" in item[1]]
        lane_limits = _lane_limits(args.framework_per_seed, args.haplotype_lane_fraction)
        selected_counts = {"single": 0, "haplotype": 0}
        leftovers: list[tuple[str, str]] = []
        for lane, items in (("single", singles), ("haplotype", haplotypes)):
            for candidate, source in items:
                if selected_counts[lane] >= lane_limits[lane]:
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

    for parent_record, (_framework_candidates, candidates) in zip(framework_lineage_seeds, seed_results):
        parent_proposals: list[dict[str, object]] = []
        for candidate, source in candidates:
            add_candidate(
                parent_record,
                candidate,
                source,
                "lineage",
                parent_proposals,
                allow_seen_sequence=True,
            )
        parent_proposals = _deduplicate_proposals(parent_proposals)
        singles = sorted(
            (
                item for item in parent_proposals
                if int(item["edit_distance"]) == 1
            ),
            key=_proposal_ranking_key,
        )
        haplotypes = sorted(
            (
                item for item in parent_proposals
                if int(item["edit_distance"]) > 1
            ),
            key=_proposal_ranking_key,
        )
        lane_limits = _lane_limits(args.framework_per_seed, args.haplotype_lane_fraction)
        selected_counts = {"single": 0, "haplotype": 0}
        leftovers: list[dict[str, object]] = []
        for lane, items in (("single", singles), ("haplotype", haplotypes)):
            for item in items:
                if selected_counts[lane] >= lane_limits[lane]:
                    leftovers.append(item)
                    continue
                lineage_proposals.append(item)
                selected_counts[lane] += 1
        accepted = sum(selected_counts.values())
        for item in sorted(leftovers, key=_proposal_ranking_key):
            if accepted >= args.framework_per_seed:
                break
            lineage_proposals.append(item)
            accepted += 1

    fallback_proposals = []
    # Historical single fallback is a bounded safety net (2026-10-20): at most
    # 10% of this length's proposal budget, so a starved fresh pool is topped
    # up without being swamped by stale reservoir singles.
    fallback_budget = max(0, args.proposal_limit // 10)
    if seed_shortage:
        for parent_record in core_seeds:
            if len(fallback_proposals) >= fallback_budget:
                break
            if seed_metadata[parent_record[1]].get("parent_validator_duplicate") is not False:
                continue
            # GAP 5: a single-edit historical fallback preserves the parent
            # CDR3, so an out-of-range (repair) or unprofiled parent must not
            # seed it -- only the multi-edit lane may repair the CDR3. Default
            # False = fail-closed (unprofiled means not-in-range).
            if not seed_metadata[parent_record[1]].get(
                "parent_cdr_developability_in_range", False
            ):
                continue
            # Top-50-basin parents are equally barred: a single-residue child
            # keeps the blocked clonotype and dies at entry_unique_for_protein_hf.
            # Absent flag = in-basin (fail-closed).
            if seed_metadata[parent_record[1]].get("parent_top50_identity_in_basin", True):
                continue
            parent = parent_record[1]
            for position in range(length):
                if len(fallback_proposals) >= fallback_budget:
                    break
                if position in parent_record[4]:
                    continue
                for residue in sorted(gates.ALLOWED_AAS):
                    if len(fallback_proposals) >= fallback_budget:
                        break
                    if residue != parent[position]:
                        add_candidate(parent_record, parent[:position] + residue + parent[position + 1:],
                                      "historical_fallback_single", "core_cdr", fallback_proposals)

    _resolve_pending_evidence(
        pending_evidence, mutation_network,
        workers=args.generation_workers,
        workers_minimum_candidates=args.generation_workers_minimum_candidates,
    )

    # Order the basin lane by proposal quality. Retain its generated tail
    # for the exact novelty gate rather than truncating it before that gate.
    basin_proposals = sorted(
        _deduplicate_proposals(basin_proposals),
        key=_proposal_ranking_key,
    )

    baseline_proposals = balanced_proposal_order(
        _round_robin_by_basin(core_proposals),
        _round_robin_by_basin(framework_proposals),
        limit=args.proposal_limit,
        framework_fraction=args.framework_fraction,
    )
    baseline_with_lineage = interleave_novel_lane(
        baseline_proposals,
        _round_robin_by_basin(_deduplicate_proposals(lineage_proposals)),
        limit=args.proposal_limit,
        novel_fraction=args.lineage_fraction,
    )
    # Corrected lane priority (2026-10-19, epoch-25809 origin analysis): the
    # STRONG candidates come from exact_shortlist (current frontier elite) via
    # lineage_terminal_single / lineage_cdr_single / c-terminal + n-terminal
    # haplotypes, while the WEAK tail comes from historical_basin_member via
    # basin_terminal_single (stale archive basin). Put the lineage + framework
    # (strong) lanes first so they fill the budget, and let the stale-archive
    # basin lane backfill -- it no longer takes the primary 70% share.
    proposals = interleave_novel_lane(
        baseline_with_lineage,
        ordered_basin_proposals(
            _deduplicate_proposals(basin_proposals),
            evidence_fraction=args.basin_evidence_fraction,
            evidence_core_limit=args.basin_evidence_core_limit,
            top_core_weight=args.basin_top_core_weight,
        ),
        limit=args.proposal_limit,
        novel_fraction=0.0,
    )
    # The shortage reserve is additive after normal lane truncation; do not
    # discard its novelty headroom to satisfy an emitted-pool cap.
    # Lane budgets order the head; retain every already-generated proposal for
    # the exact novelty gate unless this cycle has an explicit proposal budget.
    _append_proposal_tail(
        proposals, [*core_proposals, *framework_proposals, *lineage_proposals,
                    *basin_proposals, *fallback_proposals],
        limit=args.proposal_limit if getattr(args, "bounded_generation", False) else None,
    )
    output = pd.DataFrame(proposals) if proposals else pd.DataFrame(columns=["sequence"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"eligible current-gate seeds: {len(eligible)}")
    print("selected seeds:")
    for (
        score, sequence, max_jaccard, source, protected, lane,
        core_identifier, core_rank, basin_reserved,
    ) in seeds:
        print(
            f"  score={score:.1f} source={source} lane={lane} "
            f"basin={core_identifier} basin_rank={core_rank} "
            f"basin_reserved={basin_reserved} "
            f"jaccard={max_jaccard:.3f} protected_contacts={len(protected)} "
            f"{sequence}"
        )
    print(f"wrote {len(output)} balanced proposals to {args.output}")
    print(f"exhaustive supported single substitutions: {exhaustive_single_count}")
    if not output.empty:
        print(output.groupby(["proposal_type", "edit_distance"]).size().to_string())

    return seed_shortage


if __name__ == "__main__":
    main()
