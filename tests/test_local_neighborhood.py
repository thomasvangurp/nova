import numpy as np

from nanobody_opt.local_neighborhood import (
    _deduplicate_proposals,
    _interleave_rankings,
    _round_robin_by_basin,
    _round_robin_by_parent,
    _weighted_choice,
    interleave_novel_lane,
    ordered_basin_proposals,
    supported_framework_candidates,
    supported_basin_terminal_candidates,
    supported_lineage_candidates,
    bounded_basin_terminal_routes,
    residue_prior_weights,
)
from nanobody_opt.terminal_policy import TerminalPriors


def test_winner_enrichment_weights_residues_without_banning_alternatives():
    priors = residue_prior_weights(
        ["AA", "AA", "CA", "CA"],
        ["AA", "AA", "AA"],
        length=2,
    )

    assert priors[0]["A"] > priors[0]["C"]
    assert priors[0]["C"] > 0


def test_weighted_choice_prefers_enriched_residue():
    rng = np.random.default_rng(7)
    choices = [
        _weighted_choice(rng, ["A", "B"], [100.0, 1.0])
        for _ in range(100)
    ]

    assert choices.count("A") > 95


def test_proposals_are_interleaved_across_structural_parents():
    proposals = [
        {
            "parent_sequence": parent,
            "parent_selection_score": score,
            "proposal_prior_score": prior,
            "edit_distance": 1,
            "sequence": f"{parent}{index}",
        }
        for parent, score in (("PARENT_A", 50), ("PARENT_B", 60))
        for index, prior in enumerate((1.0, 2.0))
    ]

    ordered = _round_robin_by_parent(proposals)

    assert [item["parent_sequence"] for item in ordered] == [
        "PARENT_A", "PARENT_B", "PARENT_A", "PARENT_B"
    ]
    assert [item["proposal_prior_score"] for item in ordered[:2]] == [2.0, 2.0]


def test_each_parent_interleaves_single_and_multi_edit_proposals():
    proposals = [
        {
            "parent_sequence": "PARENT",
            "parent_selection_score": 50,
            "proposal_prior_score": 1.0,
            "predecessor_potential_score": float(index),
            "edit_distance": edits,
            "sequence": f"SEQ{index}",
        }
        for index, edits in enumerate((1, 1, 1, 2, 3, 4))
    ]

    ordered = _round_robin_by_parent(proposals)

    assert [item["edit_distance"] for item in ordered] == [1, 2, 1, 3, 1, 4]


def test_proposal_allocation_round_robins_cores_before_parents():
    proposals = [
        {
            "sequence": f"{core}{parent}{index}",
            "parent_sequence": parent,
            "parent_basin_core_id": core,
            "parent_basin_rank": rank,
            "parent_selection_score": 50.0,
            "proposal_prior_score": 1.0,
            "predecessor_potential_score": 50.0,
            "predecessor_consensus_score": 50.0,
            "edit_distance": 1,
        }
        for core, rank in (("CORE_A", 1), ("CORE_B", 2))
        for parent in (f"{core}_P1", f"{core}_P2")
        for index in range(2)
    ]

    ordered = _round_robin_by_basin(proposals)

    assert [item["parent_basin_core_id"] for item in ordered[:4]] == [
        "CORE_A", "CORE_B", "CORE_A", "CORE_B"
    ]
    assert len({item["parent_sequence"] for item in ordered[:4]}) == 4


def test_duplicate_sequence_keeps_strongest_parent_attribution():
    common = {
        "sequence": "CANDIDATE",
        "proposal_prior_score": 1.0,
        "edit_distance": 1,
        "proposal_type": "lineage_terminal_single_substitution",
    }
    proposals = [
        {
            **common,
            "parent_sequence": "WEAK_PARENT",
            "parent_selection_score": 80.0,
            "predecessor_potential_score": 60.0,
            "predecessor_consensus_score": 70.0,
        },
        {
            **common,
            "parent_sequence": "ELITE_PARENT",
            "parent_selection_score": 40.0,
            "predecessor_potential_score": 45.0,
            "predecessor_consensus_score": 50.0,
        },
    ]

    selected = _deduplicate_proposals(proposals)

    assert len(selected) == 1
    assert selected[0]["parent_sequence"] == "ELITE_PARENT"


def test_novel_lane_is_bounded_without_destroying_primary_head():
    primary = [
        {"sequence": f"P{index}", "proposal_lane": "primary"}
        for index in range(20)
    ]
    novel = [
        {"sequence": f"N{index}", "proposal_lane": "lineage"}
        for index in range(20)
    ]

    selected = interleave_novel_lane(
        primary, novel, limit=10, novel_fraction=0.2
    )

    assert [item["sequence"] for item in selected] == [
        "P0", "P1", "P2", "P3", "N0", "P4", "P5", "P6", "P7", "N1",
    ]


def test_novel_lane_backfills_duplicate_primary_sequences():
    primary = [
        {"sequence": f"P{index}", "proposal_lane": "primary"}
        for index in range(10)
    ]
    novel = [
        {"sequence": "P0", "proposal_lane": "lineage"},
        {"sequence": "P1", "proposal_lane": "lineage"},
        {"sequence": "NOVEL", "proposal_lane": "lineage"},
    ]

    selected = interleave_novel_lane(
        primary, novel, limit=5, novel_fraction=0.2
    )

    assert [item["sequence"] for item in selected] == [
        "P0", "P1", "P2", "P3", "NOVEL",
    ]


def test_terminal_lane_never_changes_cdr_or_protected_positions():
    parent = "AAAAAAAAAAAAAAAAAAAA"
    allowed = [["A", "C"] for _ in parent]
    candidates = supported_framework_candidates(
        parent,
        allowed=allowed,
        background_sequences=["C" * len(parent)],
        cdr_positions={2, 3, 17},
        protected={1, 18},
        terminal_width=5,
    )

    for sequence, _ in candidates:
        assert sequence[1] == "A"
        assert sequence[2] == "A"
        assert sequence[3] == "A"
        assert sequence[17] == "A"
        assert sequence[18] == "A"


def test_lineage_lane_exhausts_supported_terminal_and_cdr_singles():
    parent = "A" * 20
    allowed = [["A"] for _ in parent]
    allowed[0] = ["A", "D"]
    allowed[5] = ["A", "R"]
    allowed[18] = ["A", "K"]

    candidates = dict(supported_lineage_candidates(
        parent,
        allowed=allowed,
        background_sequences=[],
        cdr_positions={5},
        protected=set(),
        terminal_width=4,
    ))

    assert parent[:0] + "D" + parent[1:] in candidates
    assert parent[:5] + "R" + parent[6:] in candidates
    assert parent[:18] + "K" + parent[19:] in candidates
    assert set(candidates.values()) == {
        "lineage_cdr_single_substitution",
        "lineage_terminal_single_substitution",
    }


def test_lineage_haplotype_can_span_both_terminals_and_cdr():
    parent = "A" * 20
    donor = "D" + "A" * 4 + "R" + "A" * 12 + "K" + "A"
    allowed = [sorted({left, right}) for left, right in zip(parent, donor)]

    candidates = dict(supported_lineage_candidates(
        parent,
        allowed=allowed,
        background_sequences=[donor],
        cdr_positions={5},
        protected=set(),
        terminal_width=4,
        maximum_haplotype_mutations=3,
    ))

    assert candidates[donor] == "lineage_haplotype"


def test_lineage_lane_never_changes_protected_positions():
    parent = "A" * 20
    donor = "D" + "A" * 4 + "R" + "A" * 12 + "K" + "A"
    allowed = [sorted({left, right}) for left, right in zip(parent, donor)]

    candidates = supported_lineage_candidates(
        parent,
        allowed=allowed,
        background_sequences=[donor],
        cdr_positions={5},
        protected={5},
        terminal_width=4,
        maximum_haplotype_mutations=3,
    )

    assert all(sequence[5] == "A" for sequence, _ in candidates)
    assert donor not in dict(candidates)


def test_basin_terminal_lane_recombines_only_within_masked_core():
    core = "A" * 100
    parent = "A" * 10 + core + "A" * 17
    same_core = "C" + "A" * 9 + core + "A" * 16 + "D"
    other_core = "D" * 10 + "C" * 100 + "E" * 17

    routes = supported_basin_terminal_candidates(
        parent,
        basin_sequences=[parent, same_core, other_core],
        protected=set(),
        maximum_haplotype_mutations=3,
    )
    candidates = {sequence: source for sequence, source, _ in routes}

    assert ("C" + parent[1:]) in candidates
    assert (parent[:-1] + "D") in candidates
    assert ("C" + parent[1:-1] + "D") in candidates
    assert all(sequence[10:-17] == core for sequence in candidates)


def test_basin_route_bound_reserves_winner_position_evidence():
    base = "A" * 30
    legacy = base[:-1] + "C"
    hotspot = "C" + base[1:]
    priors = TerminalPriors(
        position_rank={0: 1, 29: 2},
        residue_rank={0: {"C": 1}, 29: {"C": 1}},
        exploration_residues={},
        sequence_latest_epoch={base: 10},
        sequence_score={base: 50.0},
        winner_step_count={0: 3},
        public_epoch_count=10,
    )

    selected = bounded_basin_terminal_routes(
        [
            (legacy, "basin_terminal_single_substitution", base),
            (hotspot, "basin_terminal_single_substitution", base),
        ],
        base_scores={base: 50.0},
        limit=2,
        exploration_fraction=0,
        terminal_priors=priors,
        evidence_fraction=0.5,
    )

    assert {route[0] for route in selected} == {legacy, hotspot}


def test_interleaved_rankings_preserve_both_prefixes():
    primary = [{"sequence": f"P{index}"} for index in range(4)]
    evidence = [{"sequence": f"E{index}"} for index in range(4)]

    selected = _interleave_rankings(
        primary, evidence, limit=4, evidence_fraction=0.5
    )

    assert [item["sequence"] for item in selected] == ["P0", "E0", "P1", "E1"]


def test_top_basin_gets_evidence_and_an_extra_allocation_turn():
    def proposal(core, rank, policy, index):
        return {
            "sequence": f"{core}_{policy}_{index}",
            "parent_sequence": f"{core}_P{index}",
            "parent_basin_core_id": core,
            "parent_basin_rank": rank,
            "parent_selection_score": 50.0,
            "proposal_prior_score": 1.0,
            "predecessor_potential_score": 50.0,
            "predecessor_consensus_score": 50.0,
            "edit_distance": 1,
            "basin_route_policy": policy,
            "basin_position_rank": 1,
            "basin_residue_rank": 1,
            "basin_parent_recency_rank": index + 1,
        }

    proposals = [
        proposal(core, rank, policy, index)
        for core, rank in (("CORE_A", 1), ("CORE_B", 2))
        for policy in ("legacy", "winner_position")
        for index in range(3)
    ]
    ordered = ordered_basin_proposals(
        proposals,
        evidence_fraction=0.5,
        evidence_core_limit=1,
        top_core_weight=2,
    )

    assert [item["parent_basin_core_id"] for item in ordered[:3]] == [
        "CORE_A", "CORE_A", "CORE_B"
    ]
    assert [item["basin_route_policy"] for item in ordered[:3]] == [
        "legacy", "winner_position", "legacy"
    ]


def _seed(sequence, score=0.0):
    """A seed tuple shaped like the ones the eligible list carries."""
    return (score, sequence, 0.0, "exact_shortlist", set(), "epoch_elite", "core", 0, False)


BASE = "KWALVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQAPGKEREWVSLISSSGSTWMDYPDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCGVFLGAHPCPLEGEMDAFGAGTTVVVGQ"


def _kin(base, position, residue):
    out = list(base)
    out[position] = residue
    return "".join(out)


def _stranger(base, span):
    out = list(base)
    for i in range(20, 20 + span):
        out[i] = "G"
    return "".join(out)


def test_the_leading_family_keeps_the_bulk_but_others_get_seats():
    """Seeds were taken in pure score order, so the leading family filled all
    thirty-two slots and every ticket of every epoch came from one molecule's
    neighbourhood. An even split is the opposite error: it starves the family
    that has earned the search."""
    from nanobody_opt.local_neighborhood import round_robin_families

    leader = [_seed(_kin(BASE, 2, aa), i) for i, aa in enumerate("MVQTAWYCDEFG")]
    others = [_seed(_stranger(BASE, n), 90.0 + n) for n in (20, 40, 60)]

    taken = round_robin_families(leader + others, 10)
    sequences = [t[1] for t in taken]
    from_leader = sum(1 for s in sequences if s in {x[1] for x in leader})

    assert len(taken) == 10
    assert from_leader >= 6, "the leading family lost its share of the budget"
    for other in others:
        assert other[1] in sequences, "a family was left with no seats at all"
    assert sequences[0] == leader[0][1], "best-first within the leader was lost"


def test_a_single_family_still_fills_the_budget():
    from nanobody_opt.local_neighborhood import round_robin_families

    only = [_seed(_kin(BASE, 2, aa), i) for i, aa in enumerate("MVQTA")]
    assert len(round_robin_families(only, 4)) == 4


def test_the_budget_is_never_exceeded():
    from nanobody_opt.local_neighborhood import round_robin_families

    seeds = [_seed(_stranger(BASE, n), n) for n in range(5, 40, 5)]
    assert len(round_robin_families(seeds, 3)) == 3
    assert round_robin_families(seeds, 0) == []


def test_the_seed_list_is_actually_built_round_robin():
    """Wiring, not arithmetic: the collapse happened where core_seeds is built."""
    import inspect

    from nanobody_opt import local_neighborhood

    source = inspect.getsource(local_neighborhood)
    line = next(l for l in source.splitlines() if "core_seeds = " in l)
    assert "round_robin_families" in line or "round_robin_families(" in source[
        source.index("core_seeds = "): source.index("core_seeds = ") + 200
    ], "seeds are taken in score order again"
