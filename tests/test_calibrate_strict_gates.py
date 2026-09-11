import numpy as np
import pytest

from nanobody_opt import calibrate_strict_gates as calibrate


def _distribution(centre, spread=30.0, n=500, seed=3):
    """A cohort whose gates have real tail support, as a live cohort must."""
    rng = np.random.default_rng(seed)
    return {
        "total": rng.normal(centre, spread, n),
        "confidence": rng.normal(centre * 0.7, spread, n),
        "physical_interaction": rng.normal(centre * 0.25, spread * 0.4, n),
        "developability": rng.normal(centre * 0.05, spread * 0.2, n),
    }


def test_an_equivalent_scorer_reproduces_the_existing_thresholds():
    production = _distribution(100.0)
    replacement = _distribution(100.0, seed=11)

    result = calibrate.calibrate(production, replacement)

    for gate, (_, current) in calibrate.SCALE_DEPENDENT_GATES.items():
        assert abs(result["gates"][gate] - current) < 6.0, gate


def test_an_offset_scorer_moves_the_threshold_by_its_offset():
    production = _distribution(100.0)
    replacement = _distribution(120.0, seed=11)

    result = calibrate.calibrate(production, replacement)
    record = result["provenance"]["maximum_p90_rank_sum"]

    # the gate follows the distribution rather than staying at 90, so the
    # fraction of candidates that qualify is preserved
    assert record["shift"] > 10.0
    assert abs(record["replacement_pass_rate"] - record["production_pass_rate"]) < 0.02


def test_selectivity_not_the_number_is_what_is_preserved():
    production = _distribution(100.0)
    replacement = _distribution(80.0, seed=11)

    result = calibrate.calibrate(production, replacement)
    record = result["provenance"]["maximum_p90_rank_sum"]

    assert record["shift"] < 0.0
    assert abs(record["replacement_pass_rate"] - record["production_pass_rate"]) < 0.02


def test_quantile_of_reports_the_pass_rate():
    values = np.array([10.0, 20.0, 30.0, 40.0])

    assert calibrate.quantile_of(values, 25.0) == 0.5
    assert calibrate.quantile_of(values, 5.0) == 0.0

    with pytest.raises(ValueError, match="no observations"):
        calibrate.quantile_of(np.array([]), 10.0)


def test_pose_gates_are_not_silently_recalibrated():
    result = calibrate.calibrate(_distribution(100.0), _distribution(100.0, seed=5))

    for gate in calibrate.POSE_GATES:
        assert gate not in result["gates"]


def test_dense_rank_matches_the_validator_helper():
    import pandas as pd

    from nanobody_opt.simulate_round import METRICS, rank_sum_in_field

    rng = np.random.default_rng(5)
    field = pd.DataFrame(
        {name: rng.normal(1.0, 0.2, 30) for name in METRICS}
    )
    mine = {name: float(rng.normal(1.0, 0.2)) for name in METRICS}

    compiled = calibrate._compiled_field(field)
    total = sum(
        calibrate._dense_rank(compiled[name], mine[name], direction)
        for name, direction in METRICS.items()
    )

    assert total == rank_sum_in_field(mine, field)


def test_both_cohorts_are_scored_against_the_same_fields():
    import pandas as pd

    from nanobody_opt.simulate_round import METRICS

    rng = np.random.default_rng(7)
    fields = {
        epoch: pd.DataFrame({name: rng.normal(1.0, 0.2, 30) for name in METRICS})
        for epoch in (25050, 25051, 25052)
    }
    # a cohort re-scored offline carries no epoch of its own
    observations = pd.DataFrame(
        [{"epoch": 99999, **{name: 1.0 for name in METRICS}} for _ in range(4)]
    )

    scores = calibrate.score_distributions(observations, fields, reference_epochs=3)

    assert len(scores["total"]) == 4
    assert all(np.isfinite(scores["total"]))


def test_a_gate_with_too_little_tail_support_keeps_its_production_value():
    production = _distribution(150.0, spread=20.0)
    replacement = _distribution(170.0, spread=20.0, n=200, seed=11)

    # competitive_score_threshold passes well under 1% of observations, so 200
    # observations cannot place it; it must not move on that evidence
    result = calibrate.calibrate(production, replacement)
    record = result["provenance"]["competitive_score_threshold"]

    assert record["applied"] is False
    assert "competitive_score_threshold" not in result["gates"]
    assert "need 20" in record["reason"]


def test_a_well_supported_gate_still_moves():
    production = _distribution(100.0)
    replacement = _distribution(120.0, n=2000, seed=11)

    result = calibrate.calibrate(
        production, replacement, minimum_tail_observations=20
    )

    assert result["provenance"]["maximum_p90_physical_interaction_rank_sum"][
        "applied"
    ] is True
