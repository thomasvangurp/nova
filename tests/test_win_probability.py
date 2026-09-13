"""Selection by chance-of-a-winning-draw rather than by best average."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import win_probability as wp
from nanobody_opt.win_probability import (
    MINIMUM_SPREAD,
    observation_spread,
    pooled_spread,
    rank_by_win_probability,
    win_probability,
)


def test_spread_needs_two_observations():
    assert observation_spread([120.0]) is None
    assert observation_spread([]) is None
    assert observation_spread([100.0, 140.0]) > 0


def test_a_wider_candidate_can_beat_a_better_average():
    """The point of the estimator: mean 130 sd 50 beats mean 120 sd 10 at t=60."""
    steady = win_probability([110.0, 120.0, 130.0], 60.0)
    volatile = win_probability([60.0, 130.0, 200.0], 60.0)
    assert volatile > steady
    # ... and the mean would have ordered them the other way round.
    assert sum([110.0, 120.0, 130.0]) / 3 < sum([60.0, 130.0, 200.0]) / 3


def test_threshold_at_the_mean_is_a_coin_flip():
    assert abs(win_probability([100.0, 120.0], 110.0) - 0.5) < 1e-9


def test_spread_floor_stops_infinite_confidence():
    """Two coincident observations must not certify a candidate."""
    coincident = win_probability([100.0, 100.0], 99.0)
    assert 0.0 < coincident < 0.5
    # The floor is what keeps it off zero: sd 0 would make this exactly 0.
    assert coincident > win_probability([100.0, 100.0], 99.0 - 10 * MINIMUM_SPREAD)


def test_single_observation_borrows_the_cohort_spread():
    candidates = [
        {"sequence": "A", "per_observation_median_rank_sum": [150.0, 170.0, 190.0]},
        {"sequence": "B", "per_observation_median_rank_sum": [120.0]},
    ]
    ranked = rank_by_win_probability(candidates, 100.0)
    borrowed = next(item for item in ranked if item["sequence"] == "B")
    measured = next(item for item in ranked if item["sequence"] == "A")
    assert borrowed["observation_spread_estimated"] is True
    assert measured["observation_spread_estimated"] is False
    assert borrowed["observation_spread"] == pooled_spread([20.0])
    # Borrowing a real spread, not the floor, keeps it comparable to its cohort.
    assert borrowed["observation_spread"] > MINIMUM_SPREAD


def test_ranking_is_best_first_and_annotated():
    candidates = [
        {"sequence": "steady", "score_median": 118.0,
         "per_observation_median_rank_sum": [110.0, 120.0, 125.0]},
        {"sequence": "volatile", "score_median": 130.0,
         "per_observation_median_rank_sum": [55.0, 130.0, 205.0]},
    ]
    ranked = rank_by_win_probability(candidates, 60.0)
    assert [item["sequence"] for item in ranked] == ["volatile", "steady"]
    assert ranked[0]["win_probability"] > ranked[1]["win_probability"]
    assert ranked[0]["win_threshold"] == 60.0
    # The input is not mutated: the caller may still need the original ordering.
    assert "win_probability" not in candidates[0]


def test_no_observations_cannot_win():
    assert win_probability([], 100.0) == 0.0


def test_a_two_observation_spread_leans_on_the_cohort():
    """One degree of freedom is not evidence of a wide draw distribution.

    Spread is an asset in this estimator, so a candidate whose spread is merely
    badly measured outranks one whose spread is real. Epoch 25093 ranked a
    two-observation candidate first on a spread of 91.9 against ~45 for the
    five-observation candidates.
    """
    pooled = 45.0
    wide_but_unknown = wp.shrunk_spread(91.9, pooled, observations=2)
    wide_and_known = wp.shrunk_spread(91.9, pooled, observations=9)
    assert wide_but_unknown < wide_and_known
    assert pooled < wide_but_unknown < 91.9


def test_a_well_observed_spread_is_mostly_its_own():
    assert wp.shrunk_spread(80.0, 40.0, observations=9) > wp.shrunk_spread(
        80.0, 40.0, observations=3
    )


def test_a_single_observation_cannot_stand_out_on_a_lucky_draw():
    """One draw lands about 34 rank points from its own mean, so it is not
    grounds to spend a ticket on."""
    cohort = 130.0
    lucky = wp.shrunk_mean(90.0, cohort, observations=1)
    earned = wp.shrunk_mean(90.0, cohort, observations=9)
    assert earned < lucky < cohort


def test_shrinkage_never_invents_a_spread_below_the_floor():
    assert wp.shrunk_spread(0.1, 0.1, observations=9) >= wp.MINIMUM_SPREAD
    assert wp.shrunk_spread(None, 0.0, observations=1) >= wp.MINIMUM_SPREAD


def test_the_ranker_applies_the_shrinkage_not_just_the_raw_sample():
    """Wiring, not arithmetic: the epoch-25093 ordering came out of the ranker.

    A wider spread still ranks higher after shrinking -- that is the point of
    variance seeking -- but a two-observation candidate must not collect the
    full advantage of a spread carrying one degree of freedom.
    """
    cohort = [
        {"per_observation_median_rank_sum": [120.0, 130.0, 125.0, 135.0, 128.0],
         "sequence": "STEADY"},
        {"per_observation_median_rank_sum": [118.0, 132.0, 124.0, 136.0, 127.0],
         "sequence": "STEADY2"},
        {"per_observation_median_rank_sum": [65.0, 191.0], "sequence": "NOISY"},
    ]
    ranked = wp.rank_by_win_probability(cohort, 60.0)
    noisy = next(item for item in ranked if item["sequence"] == "NOISY")

    sample = wp.observation_spread([65.0, 191.0])
    assert noisy["observation_spread"] < sample, "the sample spread was used raw"
    assert noisy["observation_spread_sample"] == sample

    unshrunk = wp.win_probability([65.0, 191.0], 60.0, spread=sample)
    assert noisy["win_probability"] < unshrunk, (
        "the two-observation candidate kept the full advantage of a "
        "one-degree-of-freedom spread"
    )
