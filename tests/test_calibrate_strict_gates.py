import numpy as np
import pytest

from nanobody_opt import calibrate_strict_gates as calibrate


def _distribution(centre, spread=20.0, n=500, seed=3):
    rng = np.random.default_rng(seed)
    return {
        "total": rng.normal(centre, spread, n),
        "confidence": rng.normal(centre * 0.7, spread, n),
        "physical_interaction": rng.normal(centre * 0.25, spread * 0.4, n),
        "developability": rng.normal(centre * 0.05, spread * 0.2, n),
    }


def test_an_equivalent_scorer_reproduces_the_existing_thresholds():
    production = _distribution(150.0)
    replacement = _distribution(150.0, seed=11)

    result = calibrate.calibrate(production, replacement)

    for gate, (_, current) in calibrate.SCALE_DEPENDENT_GATES.items():
        assert abs(result["gates"][gate] - current) < 6.0, gate


def test_an_offset_scorer_moves_the_threshold_by_its_offset():
    production = _distribution(150.0)
    replacement = _distribution(170.0, seed=11)

    result = calibrate.calibrate(production, replacement)
    record = result["provenance"]["maximum_p90_rank_sum"]

    # the gate follows the distribution rather than staying at 90, so the
    # fraction of candidates that qualify is preserved
    assert record["shift"] > 10.0
    assert abs(record["replacement_pass_rate"] - record["production_pass_rate"]) < 0.02


def test_selectivity_not_the_number_is_what_is_preserved():
    production = _distribution(150.0)
    replacement = _distribution(120.0, seed=11)

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
    result = calibrate.calibrate(_distribution(150.0), _distribution(150.0, seed=5))

    for gate in calibrate.POSE_GATES:
        assert gate not in result["gates"]
