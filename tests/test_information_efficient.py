from compound_miner.information_efficient import (
    expected_positive_improvement,
    posterior_standard_error,
    select_confirmation_candidates,
)


def test_unlucky_single_pass_keeps_a_nonzero_chance_to_enter():
    probability, improvement = expected_positive_improvement(0.091, 0.100, 0.006)
    assert 0.05 < probability < 0.10
    assert improvement > 0


def test_empirical_noise_can_only_widen_the_posterior():
    floor = posterior_standard_error(2, 0.0)
    assert posterior_standard_error(2, 0.02) > floor


def test_confirmation_is_one_adaptive_pass_and_stops_at_three():
    rows = [
        {"name": "complete", "passes": 3, "mean": 0.12, "sd": 0.01, "components": [1, 2]},
        {"name": "frontier", "passes": 2, "mean": 0.101, "sd": 0.004, "components": [1, 3]},
        {"name": "uncertain", "passes": 1, "mean": 0.096, "sd": 0.02, "components": [4, 5]},
    ]
    selected = select_confirmation_candidates(
        rows, threshold=0.100, incumbent_components=[[1, 2]], cap=2,
    )
    assert {item.name for item in selected} == {"frontier", "uncertain"}
    assert "complete" not in {item.name for item in selected}


def test_information_and_diversity_seats_survive_pure_mean_ranking():
    rows = [
        {"name": f"value-{index}", "passes": 1, "mean": 0.105 - index * 0.0001,
         "sd": 0.003, "components": [1, 2]}
        for index in range(20)
    ]
    rows += [
        {"name": "high-uncertainty", "passes": 1, "mean": 0.097,
         "sd": 0.025, "components": [1, 2]},
        {"name": "novel-components", "passes": 1, "mean": 0.098,
         "sd": 0.004, "components": [90, 91]},
    ]
    selected = select_confirmation_candidates(
        rows, threshold=0.100, incumbent_components=[[1, 2]], cap=10,
    )
    by_name = {item.name: item for item in selected}
    assert by_name["high-uncertainty"].reason == "information"
    assert by_name["novel-components"].reason == "diversity"

