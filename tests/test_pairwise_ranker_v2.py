import pandas as pd

from nanobody_opt.train_pairwise_ranker_v2 import (
    pair_feature_matrix,
    sampled_epoch_pairs,
)


def test_pair_sampling_always_includes_winner_comparisons():
    frame = pd.DataFrame({
        "sequence": ["AAA", "ACA", "ADA", "AEA"],
        "final_score": [40, 10, 30, 20],
    })

    pairs = sampled_epoch_pairs(frame, maximum=10, seed=1)

    assert {(0, 1), (1, 2), (1, 3)}.issubset(set(pairs))


def test_pair_features_are_directional_and_sparse():
    forward = pair_feature_matrix(["AAA"], ["ACA"], length=3)
    reverse = pair_feature_matrix(["ACA"], ["AAA"], length=3)

    assert forward.shape == reverse.shape
    assert forward.nnz > 0
    assert (forward != reverse).nnz > 0
