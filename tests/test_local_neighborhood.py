import numpy as np

from nanobody_opt.local_neighborhood import (
    _round_robin_by_parent,
    _weighted_choice,
    supported_framework_candidates,
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
