import pytest

from nanobody_opt.publish_checkpoint import validate_exact_robustness


def test_robustness_gate_requires_eight_independent_observations():
    with pytest.raises(ValueError, match="fewer than 8"):
        validate_exact_robustness({"observation_count": 7, "competitive_rate": 1})


def test_robustness_gate_requires_half_of_live_field_replays_to_be_competitive():
    with pytest.raises(ValueError, match="below 50%"):
        validate_exact_robustness({"observation_count": 8, "competitive_rate": 0.49})

    assert validate_exact_robustness(
        {"observation_count": 8, "competitive_rate": 0.5}
    ) == (8, 0.5)
