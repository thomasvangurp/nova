import pandas as pd

from nanobody_opt.train_parent_relative_ranker import (
    _screen_order,
    allocation_metrics,
    feature_matrix,
)
from nanobody_opt.simulate_round import METRICS


def _row(sequence, parent, child_score, improvement, screen_order):
    row = {
        "sequence": sequence,
        "parent_sequence": parent,
        "proposal_type": "exhaustive_single_substitution",
        "edit_distance": 1,
        "proposal_prior_score": 1.0,
        "parent_selection_score": 50.0,
        "observation_count": 1,
        "child_rank_sum": child_score,
        "target_improvement": improvement,
        "screen_order": screen_order,
    }
    row.update({f"parent_{name}": 1.0 for name in METRICS})
    return row


def test_screen_order_matches_live_exhaustive_first_policy():
    rows = [
        {"sequence": "AAA", "proposal_type": "local_substitution"},
        {"sequence": "BBB", "proposal_type": "exhaustive_single_substitution"},
        {"sequence": "CCC", "proposal_type": "whole_cdr_motif"},
    ]

    assert _screen_order(rows) == {"BBB": 0, "AAA": 1, "CCC": 2}


def test_features_change_with_parent_mutation_context():
    frame = pd.DataFrame([
        _row("AAA", "ACA", 40, 10, 0),
        _row("AAA", "ADA", 40, 10, 1),
    ])

    matrix = feature_matrix(frame, length=3)

    assert matrix.shape[0] == 2
    assert (matrix[0] != matrix[1]).any()


def test_allocation_metrics_compare_model_against_actual_live_order():
    frame = pd.DataFrame([
        _row("AAA", "ACA", 80, -10, 0),
        _row("ACA", "AAA", 20, 40, 1),
        _row("ADA", "AAA", 50, 5, 2),
    ])

    metrics = allocation_metrics(
        frame,
        predicted_mean=[-10, 40, 5],
        probability_improves=[0.0, 1.0, 0.7],
        budgets=(1,),
    )

    assert metrics["model_regret_at_1"] == 0
    assert metrics["heuristic_regret_at_1"] == 60
