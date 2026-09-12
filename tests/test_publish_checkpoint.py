import pytest

from nanobody_opt import publication_gates
from nanobody_opt.publish_checkpoint import (
    checkpoint_is_better,
    validate_exact_robustness,
    validate_provisional_exact_evidence,
)


GATES = publication_gates.gate_defaults()
MINIMUM_OBSERVATIONS = int(GATES["--minimum-observations"])
MINIMUM_STRUCTURES = int(GATES["--minimum-structure-observations"])
MINIMUM_COMPETITIVE_RATE = float(GATES["--minimum-competitive-rate"])
MAXIMUM_P90 = float(GATES["--maximum-p90-rank-sum"])
MAXIMUM_CONFIDENCE_P90 = float(GATES["--maximum-p90-confidence-rank-sum"])
MINIMUM_JACCARD = float(GATES["--minimum-median-contact-jaccard"])


def _robust_candidate(**overrides):
    """A candidate that sits exactly on every gate, whatever the gates now are."""
    candidate = {
        "observation_count": MINIMUM_OBSERVATIONS,
        "competitive_rate": MINIMUM_COMPETITIVE_RATE,
        "observation_competitive_rate": MINIMUM_COMPETITIVE_RATE,
        "score_p90": MAXIMUM_P90,
        "p90_category_rank_sums": {
            "confidence": MAXIMUM_CONFIDENCE_P90,
            "physical_interaction": float(
                GATES["--maximum-p90-physical-interaction-rank-sum"]
            ),
            "developability": float(GATES["--maximum-p90-developability-rank-sum"]),
        },
        "pose_stability": {
            "structure_observation_count": MINIMUM_STRUCTURES,
            "contact_jaccard_median": MINIMUM_JACCARD,
            "cdr3_rmsd_max_angstrom": float(GATES["--maximum-cdr3-rmsd"]),
            "epitope_occupancy_min": float(GATES["--minimum-epitope-occupancy"]),
        },
    }
    candidate.update(overrides)
    return candidate


def test_every_threshold_comes_from_publication_gates():
    # These numbers lived here as literals until the day the gates were relaxed
    # elsewhere: ranking then qualified candidates this writer went on refusing,
    # so the miner held a checkpoint it would not publish. One source of truth.
    assert validate_exact_robustness(_robust_candidate()) == (
        MINIMUM_OBSERVATIONS, MINIMUM_COMPETITIVE_RATE, MINIMUM_COMPETITIVE_RATE
    )


def test_robustness_gate_requires_the_configured_observation_floor():
    with pytest.raises(ValueError, match=f"fewer than {MINIMUM_OBSERVATIONS}"):
        validate_exact_robustness(_robust_candidate(
            observation_count=MINIMUM_OBSERVATIONS - 1
        ))


def test_robustness_gate_requires_the_configured_competitive_rates():
    if MINIMUM_COMPETITIVE_RATE > 0:
        with pytest.raises(ValueError, match="competitive rate"):
            validate_exact_robustness(_robust_candidate(
                competitive_rate=MINIMUM_COMPETITIVE_RATE - 0.01
            ))
    with pytest.raises(ValueError, match="competitive rate"):
        validate_exact_robustness(_robust_candidate(competitive_rate=float("nan")))
    with pytest.raises(ValueError, match="independent-observation"):
        validate_exact_robustness(_robust_candidate(
            observation_competitive_rate=float("nan")
        ))


def test_robustness_gate_enforces_total_and_category_tails():
    with pytest.raises(ValueError, match="p90 rank sum"):
        validate_exact_robustness(_robust_candidate(score_p90=MAXIMUM_P90 + 1))
    with pytest.raises(ValueError, match="p90 confidence"):
        validate_exact_robustness(_robust_candidate(
            p90_category_rank_sums={
                "confidence": MAXIMUM_CONFIDENCE_P90 + 1,
                "physical_interaction": 1,
                "developability": 1,
            }
        ))


def test_robustness_gate_requires_repeat_stable_complexes():
    with pytest.raises(ValueError, match=f"fewer than {MINIMUM_STRUCTURES} retained"):
        candidate = _robust_candidate()
        candidate["pose_stability"]["structure_observation_count"] = (
            MINIMUM_STRUCTURES - 1
        )
        validate_exact_robustness(candidate)
    with pytest.raises(ValueError, match="contact Jaccard.*below"):
        candidate = _robust_candidate()
        candidate["pose_stability"]["contact_jaccard_median"] = MINIMUM_JACCARD - 0.01
        validate_exact_robustness(candidate)


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
