"""Selection by chance-of-a-winning-draw rather than by best average."""

from __future__ import annotations

import itertools
import pytest
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import win_probability as wp
from nanobody_opt.win_probability import (
    MINIMUM_SPREAD,
    empirical_fresh_draw_probabilities,
    observation_spread,
    pooled_spread,
    rank_by_fresh_draw_probability,
    rank_by_win_probability,
    win_probability,
)


def test_empirical_fresh_draw_probability_is_exact_and_sums_to_one():
    candidates = [
        {"sequence": "A", "per_observation_median_rank_sum": [1.0, 3.0]},
        {"sequence": "B", "per_observation_median_rank_sum": [2.0, 4.0]},
    ]
    probabilities = empirical_fresh_draw_probabilities(candidates)

    # A wins three of the four equally likely draw pairs.
    assert probabilities == {"A": 0.75, "B": 0.25}
    assert sum(probabilities.values()) == 1.0


def test_empirical_fresh_draw_ties_split_the_win():
    candidates = [
        {"sequence": "A", "per_observation_median_rank_sum": [1.0]},
        {"sequence": "B", "per_observation_median_rank_sum": [1.0]},
        {"sequence": "C", "per_observation_median_rank_sum": [2.0]},
    ]

    assert empirical_fresh_draw_probabilities(candidates) == {
        "A": 0.5,
        "B": 0.5,
        "C": 0.0,
    }


def test_optimized_empirical_probability_matches_brute_force_with_ties():
    generator = random.Random(20260922)
    for size in range(2, 7):
        for _ in range(20):
            candidates = [
                {
                    "sequence": f"S{index}",
                    "per_observation_median_rank_sum": [
                        float(generator.randrange(1, 6))
                        for _ in range(generator.randrange(1, 4))
                    ],
                }
                for index in range(size)
            ]
            expected = {item["sequence"]: 0.0 for item in candidates}
            distributions = [
                item["per_observation_median_rank_sum"] for item in candidates
            ]
            outcomes = 1
            for distribution in distributions:
                outcomes *= len(distribution)
            for draw in itertools.product(*distributions):
                best = min(draw)
                winners = [index for index, value in enumerate(draw) if value == best]
                for index in winners:
                    expected[candidates[index]["sequence"]] += (
                        1.0 / len(winners) / outcomes
                    )

            actual = empirical_fresh_draw_probabilities(candidates)

            for sequence in expected:
                assert abs(actual[sequence] - expected[sequence]) < 1e-12


def test_fresh_draw_ranker_uses_head_to_head_probability_not_threshold_tail():
    candidates = [
        {
            "sequence": "RELIABLE",
            "score_median": 3.0,
            "per_observation_median_rank_sum": [2.0, 3.0, 4.0],
        },
        {
            "sequence": "LOTTERY",
            "score_median": 50.0,
            "per_observation_median_rank_sum": [1.0, 50.0, 100.0],
        },
    ]

    ranked = rank_by_fresh_draw_probability(candidates)

    assert [item["sequence"] for item in ranked] == ["RELIABLE", "LOTTERY"]
    assert ranked[0]["win_probability"] == 2 / 3
    assert ranked[0]["win_probability_method"] == ("empirical-independent-fresh-draw")


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
        {
            "sequence": "steady",
            "score_median": 118.0,
            "per_observation_median_rank_sum": [110.0, 120.0, 125.0],
        },
        {
            "sequence": "volatile",
            "score_median": 130.0,
            "per_observation_median_rank_sum": [55.0, 130.0, 205.0],
        },
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
        {
            "per_observation_median_rank_sum": [120.0, 130.0, 125.0, 135.0, 128.0],
            "sequence": "STEADY",
        },
        {
            "per_observation_median_rank_sum": [118.0, 132.0, 124.0, 136.0, 127.0],
            "sequence": "STEADY2",
        },
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


def _reference_fresh_draw_probabilities(candidates, *, scores_key="per_observation_median_rank_sum"):
    """The pre-vectorisation algorithm (linear scan, no sorting), kept only to
    differentially test that the bisect-based rewrite is exact."""
    sequences = [str(item.get("sequence", "")) for item in candidates]
    distributions = [[float(v) for v in (item.get(scores_key) or [])] for item in candidates]
    probabilities = {s: 0.0 for s in sequences}
    eligible = [i for i, s in enumerate(distributions) if s]
    if not eligible:
        return probabilities
    for candidate_index in eligible:
        own = distributions[candidate_index]
        probability = 0.0
        for score in own:
            tie_polynomial = [1.0]
            scalar = 1.0
            for opponent_index in eligible:
                if opponent_index == candidate_index:
                    continue
                opponent = distributions[opponent_index]
                denominator = float(len(opponent))
                worse = sum(value > score for value in opponent) / denominator
                equal = sum(value == score for value in opponent) / denominator
                if equal == 0.0:
                    scalar *= worse
                    if scalar == 0.0:
                        break
                    continue
                expanded = [0.0] * (len(tie_polynomial) + 1)
                for ties, coefficient in enumerate(tie_polynomial):
                    expanded[ties] += coefficient * worse
                    expanded[ties + 1] += coefficient * equal
                tie_polynomial = expanded
            if scalar:
                probability += scalar * sum(
                    c / (t + 1) for t, c in enumerate(tie_polynomial)
                ) / len(own)
        probabilities[sequences[candidate_index]] = probability
    return probabilities


def test_bisect_rewrite_matches_the_reference_implementation_with_ties(monkeypatch):
    import random
    from nanobody_opt.win_probability import empirical_fresh_draw_probabilities

    rng = random.Random(20260929)
    for trial in range(200):
        n = rng.randint(1, 12)
        # A small integer range forces frequent exact ties across candidates,
        # the case the sorted/scalar split exists to handle correctly.
        candidates = [
            {"sequence": f"seq{i}",
             "per_observation_median_rank_sum": [float(rng.randint(0, 6)) for _ in range(rng.randint(1, 6))]}
            for i in range(n)
        ]
        got = empirical_fresh_draw_probabilities(candidates)
        want = _reference_fresh_draw_probabilities(candidates)
        for seq in want:
            assert got[seq] == pytest.approx(want[seq], abs=1e-12), (trial, candidates)


def test_bisect_rewrite_matches_the_reference_implementation_with_floats(monkeypatch):
    import random
    from nanobody_opt.win_probability import empirical_fresh_draw_probabilities

    rng = random.Random(1)
    for trial in range(100):
        n = rng.randint(1, 10)
        candidates = [
            {"sequence": f"seq{i}",
             "per_observation_median_rank_sum": [rng.uniform(50, 300) for _ in range(rng.randint(1, 8))]}
            for i in range(n)
        ]
        got = empirical_fresh_draw_probabilities(candidates)
        want = _reference_fresh_draw_probabilities(candidates)
        for seq in want:
            assert got[seq] == pytest.approx(want[seq], abs=1e-12), (trial, candidates)
