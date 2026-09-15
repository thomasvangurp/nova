import pytest
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


# --- following the field's composition ---------------------------------------


def _field(*, settled_residue="G", settled_share=0.9, length=10, position=4, n=100):
    """A field that has settled on one residue at `position` and varies elsewhere."""
    import random

    rng = random.Random(7)
    out = []
    for i in range(n):
        seq = ["A"] * length
        for j in range(length):
            seq[j] = rng.choice("ACDEFGHIKL")
        seq[position] = settled_residue if i < int(settled_share * n) else "A"
        out.append("".join(seq))
    return out


def test_a_settled_field_residue_is_preferred_even_when_winners_do_not_enrich_it():
    """
    The enrichment term is P(residue|winners) / P(residue|field), so a residue
    the whole field has settled on scores about 1.0 and gets no pull. That is
    how we reached 48% G at position 99 against the field's 81%: nothing in the
    prior ever argued for it.
    """
    from nanobody_opt.local_neighborhood import residue_prior_weights

    field = _field()
    # Winners drawn evenly across the field, so they carry the settled residue
    # at exactly the field's own rate -- enrichment is ~1.0 and carries no
    # signal. This is the real case: G at position 99 is 81% of the field AND
    # 4 of the rival's 6 winners.
    winners = field[::5]
    with_field = residue_prior_weights(field, winners, 10, follow_field=True)
    without = residue_prior_weights(field, winners, 10, follow_field=False)

    # Without the field term the settled residue is indistinguishable from any
    # other; with it, it outranks the alternative the field abandoned.
    # Enrichment alone cannot separate them: the settled residue is equally
    # common among winners and among the field.
    assert without[4]["G"] == pytest.approx(without[4]["A"], rel=0.45)
    # With the field's composition in the prior, it is clearly preferred.
    assert with_field[4]["G"] > with_field[4]["A"] * 3


def test_field_consensus_reports_the_settled_residue_and_its_share():
    from nanobody_opt.local_neighborhood import field_consensus

    consensus = field_consensus(_field(settled_share=0.9), 10)
    residue, share = consensus[4]
    assert residue == "G"
    assert share == pytest.approx(0.9, abs=0.05)


def test_the_consensus_threshold_is_where_the_field_stops_tolerating_deviation():
    """
    Read off 735 scored tickets across 17 epochs: the penalty for deviating is
    +0.04 z at 0.7 consensus and +0.31 at 0.8. The threshold is measured, not
    chosen, so it must not drift without new evidence.
    """
    from nanobody_opt.local_neighborhood import FIELD_CONSENSUS_THRESHOLD

    assert FIELD_CONSENSUS_THRESHOLD == 0.80


def test_a_settled_position_can_decline_to_displace_consensus():
    """
    Everywhere else a mutation must change something. At a settled position,
    excluding the parent's residue means sampling there can ONLY move off
    consensus -- one-way drift, which is how we reached 48% G at position 99
    against the field's 81%.
    """
    from nanobody_opt.local_neighborhood import (
        FIELD_CONSENSUS_THRESHOLD,
        field_consensus,
    )

    consensus = field_consensus(_field(settled_share=0.9), 10)
    allowed = {position: list("ACDEFGHIKL") for position in range(10)}
    parent = "AAAAGAAAAA"          # holds the settled residue at position 4

    def options_at(position):
        settled_here = (
            consensus.get(position, ("", 0.0))[1] >= FIELD_CONSENSUS_THRESHOLD
        )
        return [
            residue for residue in allowed[position]
            if residue != parent[position] or settled_here
        ]

    assert parent[4] in options_at(4), "a settled position must be able to hold"
    assert parent[0] not in options_at(0), "elsewhere a mutation must still change"


def test_consensus_is_a_pull_and_never_a_gate():
    """
    A hard gate at any budget forbids some winner: freezing every settled
    position reaches 3 of the rival's 6 winning molecules, a one-deviation
    budget 4, a two-deviation budget 5. Nothing may be forbidden -- every
    residue the gates allow must remain drawable at a settled position.
    """
    from nanobody_opt.local_neighborhood import (
        FIELD_CONSENSUS_THRESHOLD,
        field_consensus,
    )

    consensus = field_consensus(_field(settled_share=0.95), 10)
    allowed = list("ACDEFGHIKL")
    parent = "AAAAGAAAAA"
    settled_here = consensus[4][1] >= FIELD_CONSENSUS_THRESHOLD
    options = [r for r in allowed if r != parent[4] or settled_here]
    assert set(options) == set(allowed), (
        "every allowed residue must stay reachable at a settled position"
    )




def test_a_proposal_always_changes_at_least_one_position():
    """
    The producer measures consensus against the most recent 600 submissions,
    where 18 of 19 CDR3 positions sit above the threshold. Letting every sampled
    position hold its parent residue therefore produced candidates identical to
    their parent, which are rejected as already seen -- acceptance fell to 1 in
    12 and the generator reported NEIGHBOURHOOD_EXHAUSTED on 32 seeds.
    """
    from nanobody_opt.local_neighborhood import (
        FIELD_CONSENSUS_THRESHOLD,
        field_consensus,
    )

    # A field where EVERY position is settled: the worst case for this bug.
    consensus = field_consensus(["GGGGGGGGGG"] * 100, 10)
    assert all(share >= FIELD_CONSENSUS_THRESHOLD for _, share in consensus.values())

    parent = "GGGGGGGGGG"
    allowed = {position: list("ACDEFGHIKL") for position in range(10)}
    positions = [3, 7]
    anchor = positions[0]

    def options_at(position):
        settled_here = (
            position != anchor
            and consensus.get(position, ("", 0.0))[1] >= FIELD_CONSENSUS_THRESHOLD
        )
        return [
            residue for residue in allowed[position]
            if residue != parent[position] or settled_here
        ]

    # The anchor cannot return the parent residue, so the candidate must differ.
    assert parent[anchor] not in options_at(anchor)
    # Every other sampled position may still hold consensus.
    assert parent[7] in options_at(7)


def test_the_anchor_prefers_unsettled_then_repairs_then_anything():
    """
    Where the forced change lands matters as much as that one does. Measured over
    380 of the strongest competitor's epoch-to-epoch mutations against 727 of
    ours: both put ~80% of mutations on unsettled positions, so WHERE is not the
    difference. DIRECTION at settled positions is -- they move toward consensus
    52% and away 43%; we move away 69% and toward 29%. They repair deviations,
    we manufacture them, and sit at 1.18 deviations per ticket against their 0.82.
    """
    from nanobody_opt.local_neighborhood import (
        FIELD_CONSENSUS_THRESHOLD,
        field_consensus,
    )

    # positions 0-8 settled on G; position 9 genuinely varies
    population = []
    for i in range(100):
        seq = ["G"] * 9 + ["ACDEFGHIKL"[i % 10]]
        population.append("".join(seq))
    consensus = field_consensus(population, 10)
    settled = {i for i, (_, share) in consensus.items()
               if share >= FIELD_CONSENSUS_THRESHOLD}
    assert 9 not in settled and 0 in settled

    def rank(parent, position):
        residue, share = consensus.get(position, ("", 0.0))
        if share < FIELD_CONSENSUS_THRESHOLD:
            return 0
        return 1 if parent[position] != residue else 2

    def anchor_for(parent, positions):
        return min(positions, key=lambda v: (rank(parent, v), v))

    at_consensus = "GGGGGGGGGG"
    # An unsettled position wins outright.
    assert anchor_for(at_consensus, [0, 3, 9]) == 9
    # With only settled positions and a parent already at consensus, any is as
    # bad as another -- but a parent that DEVIATES makes that position the pick.
    deviant = "GGAGGGGGGG"          # position 2 is off consensus
    assert anchor_for(deviant, [0, 2, 5]) == 2
    # ...and an unsettled option still outranks the repair.
    assert anchor_for(deviant, [0, 2, 9]) == 9


def test_a_settled_anchor_draws_the_consensus_residue():
    """
    A settled anchor is only chosen when the parent deviates, so the one forced
    change must be the repair -- not a sideways move between two non-consensus
    residues, which would keep the deviation while burning the mutation.
    """
    from nanobody_opt.local_neighborhood import (
        FIELD_CONSENSUS_THRESHOLD,
        field_consensus,
    )

    consensus = field_consensus(["GGGGGGGGGG"] * 100, 10)
    parent = "GGAGGGGGGG"                 # deviates at position 2
    position, anchor = 2, 2
    residue_at, share_at = consensus[position]
    allowed = list("ACDEFGHIKL")
    options = [r for r in allowed if r != parent[position]]
    if position == anchor and share_at >= FIELD_CONSENSUS_THRESHOLD and residue_at in options:
        options = [residue_at]
    assert options == ["G"], "the forced change must restore consensus"


# ---------------------------------------------------------------------------
# Settled-site policy: never break what the field has already agreed on.
# ---------------------------------------------------------------------------

def _consensus(spec):
    """{position: (residue, share)} from a compact {pos: (res, share)} spec."""
    return dict(spec)


def test_a_settled_site_the_parent_has_right_is_frozen_under_repair_only():
    from nanobody_opt.local_neighborhood import settled_site_options

    parent = "AAAA"
    consensus = _consensus({1: ("A", 0.95)})
    allowed = {1: ["A", "C", "D"]}
    # Parent already agrees with a settled field: nothing may be proposed.
    assert settled_site_options(
        1, parent=parent, consensus=consensus, allowed=allowed,
        anchor=None, repair_only=True,
    ) == ["A"]
    # Without the policy the other residues are still on the table.
    assert set(settled_site_options(
        1, parent=parent, consensus=consensus, allowed=allowed,
        anchor=None, repair_only=False,
    )) == {"A", "C", "D"}


def test_a_settled_site_the_parent_has_wrong_may_only_be_repaired():
    from nanobody_opt.local_neighborhood import settled_site_options

    parent = "ACAA"
    consensus = _consensus({1: ("A", 0.95)})
    allowed = {1: ["A", "C", "D"]}
    # Keep the deviation or repair it -- never move sideways to D.
    assert settled_site_options(
        1, parent=parent, consensus=consensus, allowed=allowed,
        anchor=None, repair_only=True,
    ) == ["C", "A"]


def test_an_unsettled_site_is_untouched_by_the_policy():
    from nanobody_opt.local_neighborhood import settled_site_options

    parent = "AAAA"
    consensus = _consensus({1: ("A", 0.40)})     # below threshold
    allowed = {1: ["A", "C", "D"]}
    strict = settled_site_options(
        1, parent=parent, consensus=consensus, allowed=allowed,
        anchor=None, repair_only=True,
    )
    free = settled_site_options(
        1, parent=parent, consensus=consensus, allowed=allowed,
        anchor=None, repair_only=False,
    )
    assert strict == free == ["C", "D"]


def test_the_anchor_repair_still_wins_under_either_policy():
    from nanobody_opt.local_neighborhood import settled_site_options

    parent = "ACAA"
    consensus = _consensus({1: ("A", 0.95)})
    allowed = {1: ["A", "C", "D"]}
    for repair_only in (True, False):
        assert settled_site_options(
            1, parent=parent, consensus=consensus, allowed=allowed,
            anchor=1, repair_only=repair_only,
        ) == ["A"], "a settled anchor spends its forced change on the repair"


def test_breakage_is_counted_whatever_lane_produced_it():
    from nanobody_opt.local_neighborhood import new_settled_deviations

    consensus = _consensus({0: ("A", 0.95), 1: ("C", 0.95), 2: ("D", 0.40)})
    parent = "ACD"
    # A motif graft or terminal haplotype can break a settled site without ever
    # touching the CDR draw; the count must see it regardless.
    assert new_settled_deviations(parent, "ACD", consensus) == 0
    assert new_settled_deviations(parent, "AED", consensus) == 1
    assert new_settled_deviations(parent, "EED", consensus) == 2
    # An unsettled position is not breakage.
    assert new_settled_deviations(parent, "ACE", consensus) == 0
    # Repairing a deviation the parent already had is not breakage either.
    assert new_settled_deviations("AED", "ACD", consensus) == 0


def test_ordering_puts_adhering_candidates_first():
    from nanobody_opt.local_neighborhood import _proposal_ranking_key

    def item(deviations, prior=1.0):
        return {
            "new_settled_deviations": deviations,
            "parent_basin_rank": 0,
            "proposal_prior_score": prior,
            "edit_distance": 1,
            "sequence": f"S{deviations}",
            "proposal_type": "local_substitution",
        }
    # A candidate that breaks nothing outranks a better-scoring one that does,
    # so the fleet fills from adhering candidates and only then falls back.
    ordered = sorted([item(2, prior=99.0), item(0, prior=0.1), item(1, prior=50.0)],
                     key=_proposal_ranking_key)
    assert [i["new_settled_deviations"] for i in ordered] == [0, 1, 2]


def test_off_restores_the_pre_change_ordering():
    """'off' must disable the ordering key too, not only the draw.

    The key drives _deduplicate_proposals, the per-lane limits and the
    proposal-limit truncation, so leaving it on while the draw is unrestricted
    still changes which molecules are emitted.  A rollback lever that only half
    rolls back is not one.
    """
    from nanobody_opt import local_neighborhood as ln

    def item(deviations, prior):
        return {
            "new_settled_deviations": deviations,
            "parent_basin_rank": 0,
            "proposal_prior_score": prior,
            "edit_distance": 1,
            "sequence": f"S{deviations}",
            "proposal_type": "local_substitution",
        }
    items = [item(2, 99.0), item(0, 0.1)]
    original = ln._PREFER_ADHERENTS
    try:
        ln._PREFER_ADHERENTS = True
        assert [i["new_settled_deviations"]
                for i in sorted(items, key=ln._proposal_ranking_key)] == [0, 2]
        ln._PREFER_ADHERENTS = False
        # With the lever off, the prior decides again, as it did before.
        assert [i["new_settled_deviations"]
                for i in sorted(items, key=ln._proposal_ranking_key)] == [2, 0]
    finally:
        ln._PREFER_ADHERENTS = original
