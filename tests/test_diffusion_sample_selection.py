import numpy as np
import pytest

from nanobody_opt.diffusion_sample_selection import select_diffusion_samples


def test_validator_rules_select_independent_best_samples():
    selected = select_diffusion_samples({
        "iptm": np.asarray([0.90, 0.50, 0.60]),
        "ptm": np.asarray([0.90, 0.50, 0.60]),
        "design_to_target_iptm": np.asarray([0.40, 0.90, 0.60]),
        "design_ptm": np.asarray([0.40, 0.90, 0.60]),
    })

    assert selected.sample_count == 3
    assert selected.writer_sample_index == 0
    assert selected.analyze_sample_index == 1
    assert selected.sample_selection_disagrees is True
    assert selected.writer_scores == pytest.approx((0.90, 0.50, 0.60))
    assert selected.analyze_scores == pytest.approx((0.40, 0.90, 0.60))


def test_validator_rule_rejects_inconsistent_sample_arrays():
    with pytest.raises(ValueError, match="different lengths"):
        select_diffusion_samples({
            "iptm": [0.9, 0.8],
            "ptm": [0.9],
            "design_to_target_iptm": [0.8, 0.7],
            "design_ptm": [0.8, 0.7],
        })


def test_reduced_diffusion_sample_count_is_refused():
    import pytest

    from nanobody_opt.diffusion_sample_selection import (
        EXPECTED_DIFFUSION_SAMPLES,
        require_expected_sample_count,
    )

    assert require_expected_sample_count(
        {"sample_count": EXPECTED_DIFFUSION_SAMPLES}
    ) == EXPECTED_DIFFUSION_SAMPLES

    with pytest.raises(ValueError, match="maximum over samples"):
        require_expected_sample_count({"sample_count": 2})
    with pytest.raises(ValueError, match="diffusion sample count"):
        require_expected_sample_count({})
