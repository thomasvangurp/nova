import pytest

from nanobody_opt.publish_checkpoint import (
    checkpoint_is_better,
    validate_exact_robustness,
    validate_provisional_exact_evidence,
)


def test_robustness_gate_requires_eight_independent_observations():
    with pytest.raises(ValueError, match="fewer than 8"):
        validate_exact_robustness({"observation_count": 7, "competitive_rate": 1})


def _robust_candidate(**overrides):
    candidate = {
        "observation_count": 8,
        "competitive_rate": 0.75,
        "observation_competitive_rate": 0.75,
        "score_p90": 90,
        "p90_category_rank_sums": {
            "confidence": 60,
            "physical_interaction": 40,
            "developability": 15,
        },
        "pose_stability": {
            "structure_observation_count": 8,
            "contact_jaccard_median": 0.35,
            "cdr3_rmsd_max_angstrom": 5.0,
            "epitope_occupancy_min": 0.20,
        },
    }
    candidate.update(overrides)
    return candidate


def test_robustness_gate_requires_three_quarters_of_replays_to_be_competitive():
    with pytest.raises(ValueError, match="below 75%"):
        validate_exact_robustness(_robust_candidate(competitive_rate=0.74))

    with pytest.raises(ValueError, match="independent-observation.*below 75%"):
        validate_exact_robustness(
            _robust_candidate(observation_competitive_rate=0.74)
        )

    assert validate_exact_robustness(
        _robust_candidate()
    ) == (8, 0.75, 0.75)


def test_robustness_gate_enforces_total_and_category_tails():
    with pytest.raises(ValueError, match="p90 rank sum 91.*exceeds 90"):
        validate_exact_robustness(_robust_candidate(score_p90=91))
    with pytest.raises(ValueError, match="p90 confidence.*exceeds 70"):
        validate_exact_robustness(_robust_candidate(
            p90_category_rank_sums={
                "confidence": 71,
                "physical_interaction": 30,
                "developability": 10,
            }
        ))


def test_robustness_gate_requires_repeat_stable_complexes():
    with pytest.raises(ValueError, match="fewer than 8 retained"):
        validate_exact_robustness(_robust_candidate(
            pose_stability={"structure_observation_count": 7}
        ))
    with pytest.raises(ValueError, match="contact Jaccard.*below"):
        validate_exact_robustness(_robust_candidate(
            pose_stability={
                "structure_observation_count": 8,
                "contact_jaccard_median": 0.34,
                "cdr3_rmsd_max_angstrom": 4,
                "epitope_occupancy_min": 0.5,
            }
        ))


def test_provisional_gate_accepts_first_complete_exact_observation():
    assert validate_provisional_exact_evidence({
        "observation_count": 1,
        "score_p90": 42,
        "competitive_rate": 1,
        "observation_competitive_rate": 1,
    }) == (1, 1.0, 1.0)


def _checkpoint(sequence="A", score=50, observations=1, epoch=25036):
    return {
        "epoch": epoch,
        "sequences": [sequence],
        "selection": {"score": score},
        "exact": {"observation_count": observations},
    }


def test_checkpoint_replacement_is_monotonic_within_an_epoch():
    previous = _checkpoint()
    assert checkpoint_is_better(previous, _checkpoint(score=49))
    assert not checkpoint_is_better(previous, _checkpoint(score=51))
    assert checkpoint_is_better(previous, _checkpoint(observations=8))
    assert checkpoint_is_better(previous, _checkpoint(epoch=25037, score=99))
