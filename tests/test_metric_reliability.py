from nanobody_opt import metric_reliability as mr


def _repeats(rng, true_sd, noise_sd, groups=60, per_group=7):
    import random

    generator = random.Random(rng)
    out = {}
    for index in range(groups):
        theta = generator.gauss(0, true_sd)
        out[f"s{index}"] = [
            {"m": theta + generator.gauss(0, noise_sd)} for _ in range(per_group)
        ]
    return out


def test_reliability_recovers_a_known_value():
    # reliability = var_true / (var_true + var_noise)
    measured = mr.reliability_from_observations(_repeats(1, 3.0, 1.0), ["m"])["m"]
    assert abs(measured - 0.9) < 0.08


def test_a_pure_noise_metric_scores_zero():
    measured = mr.reliability_from_observations(_repeats(2, 0.0, 1.0), ["m"])["m"]
    assert measured < 0.05


def test_deterministic_metric_scores_one():
    repeats = {f"s{i}": [{"m": float(i)} for _ in range(5)] for i in range(20)}
    assert mr.reliability_from_observations(repeats, ["m"])["m"] > 0.99


def test_shrinkage_moves_unreliable_metrics_to_the_population_value():
    records = {
        "A": [{"reliable": 10.0, "noisy": 8.0}],
        "B": [{"reliable": 20.0, "noisy": 2.0}],
    }
    shrunk = mr.shrink_observations(
        records, reliability={"reliable": 1.0, "noisy": 0.0},
        metrics=["reliable", "noisy"],
    )

    # the reliable metric is untouched; the uninformative one becomes identical
    # for every candidate, so it contributes a constant to every rank sum
    assert shrunk["A"][0]["reliable"] == 10.0
    assert shrunk["B"][0]["reliable"] == 20.0
    assert shrunk["A"][0]["noisy"] == shrunk["B"][0]["noisy"] == 5.0


def test_partial_reliability_shrinks_proportionally():
    records = {"A": [{"m": 10.0}], "B": [{"m": 0.0}]}
    shrunk = mr.shrink_observations(records, reliability={"m": 0.5}, metrics=["m"])

    # halfway toward the population median of 5
    assert shrunk["A"][0]["m"] == 7.5
    assert shrunk["B"][0]["m"] == 2.5


def test_reliability_file_is_validated(tmp_path):
    import json

    import pytest

    path = tmp_path / "reliability.json"
    path.write_text(json.dumps({"reliability": {"design_iiptm": 1.4}}))
    with pytest.raises(ValueError, match="within"):
        mr.load_reliability(path)
