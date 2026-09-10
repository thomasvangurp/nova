import copy

import pytest

from nanobody_opt.audit_validator_score import replay_validator_score
from nanobody_opt.simulate_round import METRICS


HOTKEY = "mine"
TARGET = "P20809"


def _metrics(maximum: float, minimum: float) -> dict[str, float]:
    return {
        name: float(maximum if direction == "max" else minimum)
        for name, direction in METRICS.items()
    }


def _payload() -> dict:
    mine = {
        "sequence": "MINE",
        "sequence_hash": "mine-hash",
        "protein_name": TARGET,
        **_metrics(3, 1),
        "confidence_rank": 5.0,
        "physical_interaction_rank": 3.0,
        "developability_rank": 2.0,
        "final_score": 10.0,
    }
    other1 = {
        "sequence": "OTHER1",
        "sequence_hash": "other-1",
        "protein_name": TARGET,
        **_metrics(2, 2),
    }
    other2 = {
        "sequence": "OTHER2",
        "sequence_hash": "other-2",
        "protein_name": TARGET,
        **_metrics(1, 3),
    }
    return {
        "success": True,
        "data": [
            {"hotkey": HOTKEY, "nanobodies": [mine]},
            {"hotkey": "other-1", "nanobodies": [other1]},
            {"hotkey": "other-2", "nanobodies": [other2]},
        ],
    }


def test_replays_published_validator_score_exactly():
    audit = replay_validator_score(_payload(), epoch=25026, hotkey=HOTKEY)

    assert audit["status"] == "exact_match"
    assert audit["field_size"] == 3
    assert audit["local_replay"] == {
        "categories": {
            "confidence": 5.0,
            "physical_interaction": 3.0,
            "developability": 2.0,
        },
        "final_score": 10.0,
    }


def test_refuses_api_or_local_scoring_drift():
    payload = copy.deepcopy(_payload())
    payload["data"][0]["nanobodies"][0]["final_score"] = 11.0

    with pytest.raises(ValueError, match="replay mismatch"):
        replay_validator_score(payload, epoch=25026, hotkey=HOTKEY)


def test_labels_checkpoint_metrics_as_independent_observation():
    checkpoint = {
        "epoch": 25026,
        "target": TARGET,
        "sequences": ["MINE"],
        "exact": {"metrics": _metrics(2, 2)},
    }

    audit = replay_validator_score(
        _payload(), epoch=25026, hotkey=HOTKEY, checkpoint=checkpoint
    )

    comparison = audit["checkpoint_comparison"]
    assert comparison["epoch_matches"]
    assert comparison["target_matches"]
    assert comparison["sequence_matches"]
    assert "independent stochastic observations" in comparison["semantics"]
