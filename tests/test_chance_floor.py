"""The "has a chance of winning" floor shared by the ranker and the publisher's order check.

Epochs 25668-25670: thirteen clean (tier-0) tickets with a win probability of ~1e-45 filled every
slot ahead of tickets with 0.05-0.001 because the chance test was ``p > 0``.  The published
median replay score fell from ~1,340 to ~1,500 while the same epochs' threshold picks scored
~1,290-1,330.  The floor lives in consensus_policy.CHANCE_FLOOR and fleet's invariant must use the
same value, or every publication cycle is rejected ("tier 0 follows tier 2").
"""
import pytest

from nanobody_opt import consensus_policy, fleet


def mk(probability, tier=0, sequence="S"):
    return {"sequence": sequence, "win_probability": probability, "settled_deviations": tier}


def order(candidates):
    return consensus_policy.ordered_candidates(candidates, "tier")


def test_floor_is_defined_once_and_shared():
    assert consensus_policy.CHANCE_FLOOR == 1e-3
    assert fleet.CHANCE_FLOOR is consensus_policy.CHANCE_FLOOR


def test_a_clean_ticket_with_no_real_chance_never_outranks_one_that_has_one():
    ranked = order([mk(1e-45, 0, "clean-but-hopeless"), mk(0.01, 1, "has-a-chance")])
    assert [c["sequence"] for c in ranked] == ["has-a-chance", "clean-but-hopeless"]


def test_chance_first_then_tier_then_probability():
    ranked = order([mk(1e-9, 0), mk(0.05, 1), mk(0.0, 0), mk(0.002, 2), mk(0.01, 0)])
    assert [(c["win_probability"], c["settled_deviations"]) for c in ranked] == [
        (0.01, 0), (0.05, 1), (0.002, 2), (1e-9, 0), (0.0, 0)]


def test_the_floor_itself_counts_as_a_chance():
    ranked = order([mk(consensus_policy.CHANCE_FLOOR * 0.99, 0, "below"),
                    mk(consensus_policy.CHANCE_FLOOR, 2, "at-floor")])
    assert [c["sequence"] for c in ranked] == ["at-floor", "below"]


def test_publisher_invariant_accepts_what_the_ranker_writes():
    ranked = order([mk(1e-9, 0), mk(0.05, 1), mk(0.0, 0), mk(0.002, 2), mk(0.01, 0), mk(0.04, 2)])
    fleet._assert_non_increasing_win_probability(ranked)


def test_publisher_invariant_still_rejects_a_candidate_with_a_chance_behind_one_without():
    with pytest.raises(fleet.SelectionOrderError):
        fleet._assert_non_increasing_win_probability([mk(1e-9, 0), mk(0.05, 1)])


def test_publisher_invariant_still_rejects_a_tier_ladder_running_backwards():
    with pytest.raises(fleet.SelectionOrderError):
        fleet._assert_non_increasing_win_probability([mk(0.05, 2), mk(0.04, 1)])
