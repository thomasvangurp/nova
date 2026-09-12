"""Selection by chance-of-a-winning-draw rather than by best average."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
