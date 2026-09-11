import numpy as np

from nanobody_opt.local_neighborhood import (
    _deduplicate_proposals,
    _round_robin_by_parent,
    _weighted_choice,
    interleave_novel_lane,
    supported_framework_candidates,
    supported_lineage_candidates,
    residue_prior_weights,
)


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
