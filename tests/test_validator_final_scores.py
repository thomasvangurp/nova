"""Pin the corrected validator ``final_score`` formula.

The validator's ``final_score`` is NOT the minmax weighted rank (rank_binders)
and NOT the sum of all ten metric dense ranks. It ranks each of the three metric
categories within the field and sums those three category ranks. These tests
guard ``simulate_round.validator_final_scores`` against silently reverting to
either of the older proxies.
"""
import numpy as np
import pandas as pd

from nanobody_opt.simulate_round import METRIC_CATEGORIES, METRICS, validator_final_scores


def _manual_final_score(frame: pd.DataFrame) -> np.ndarray:
    """Hand-computed reference: rank each category-sum, then sum the three."""
    total = np.zeros(len(frame))
    for category, names in METRIC_CATEGORIES.items():
        cat_sum = np.zeros(len(frame))
        for name in names:
            cat_sum += frame[name].rank(
                method="dense", ascending=(METRICS[name] == "min")
            ).to_numpy()
        total += pd.Series(cat_sum).rank(
            method="dense", ascending=True
        ).to_numpy()
    return total


def _sum_of_ten(frame: pd.DataFrame) -> np.ndarray:
    """The old (wrong) objective: sum all ten metric dense ranks directly."""
    total = np.zeros(len(frame))
    for name, direction in METRICS.items():
        total += frame[name].rank(
            method="dense", ascending=(direction == "min")
        ).to_numpy()
    return total


def _field() -> pd.DataFrame:
    return pd.DataFrame({
        "design_iiptm": [0.5, 0.8, 0.9],
        "design_ptm": [0.6, 0.7, 0.95],
        "design_to_target_iptm": [0.5, 0.7, 0.85],
        "min_design_to_target_pae": [3.0, 2.0, 1.0],
        "interaction_pae": [12.0, 9.0, 5.0],
        "plip_hbonds_refolded": [2.0, 4.0, 6.0],
        "plip_saltbridge_refolded": [1.0, 2.0, 3.0],
        "delta_sasa_refolded": [500.0, 700.0, 900.0],
        "liability_score": [200.0, 100.0, 50.0],
        "liability_num_violations": [20.0, 10.0, 5.0],
    })


def test_matches_three_category_rank_sum():
    frame = _field()
    np.testing.assert_allclose(
        validator_final_scores(frame), _manual_final_score(frame)
    )


def test_orders_by_category_rank_not_sum_of_ten():
    frame = _field()
    got = validator_final_scores(frame)
    old = _sum_of_ten(frame)
    # Row 2 is best on every metric, row 0 worst.
    assert got[0] > got[1] > got[2]
    # The corrected objective is a distinct statistic from the sum-of-ten;
    # assert it is not byte-identical so a reversion to the old proxy fails.
    assert not np.allclose(got, old)


def test_returns_lowest_score_for_best_sequence():
    frame = _field()
    got = validator_final_scores(frame)
    assert int(np.argmin(got)) == 2
    assert int(np.argmax(got)) == 0
