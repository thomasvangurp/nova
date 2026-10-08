import json

import pytest

from nanobody_dash import compound


def test_dynamic_floor_excludes_own_coldkey_and_uses_calibration():
    cache = {
        "1": [
            {"coldkey": "ours", "reaction": 3, "score": 3.0},
            {"coldkey": "rival", "reaction": 3, "score": 1.0},
        ],
        "2": [
            {"coldkey": "ours", "reaction": 3, "score": 3.0},
            {"coldkey": "rival", "reaction": 3, "score": 1.0},
        ],
        "_self_calibration": {
            "coldkeys": ["ours"],
            "evidence": [
                {"epoch": 1, "reaction": 3, "optimism": 0.10},
                {"epoch": 2, "reaction": 3, "optimism": 0.10},
            ],
        },
    }

    floors, details = compound._dynamic_floors_from_cache(cache, next_epoch=3)

    assert floors[3] == 1.0682
    assert details[3]["winner"] == 1.0
    assert details[3]["owned_coldkeys_excluded"] == 1
    assert details[3]["own_evidence_epochs"] == 2


def test_final_fleet_rejection_is_not_counted_as_an_allocation():
    lines = [
        '{"at": "12:00:00", "event": "window_open", "epoch": 100, "reaction": 3}',
        '{"event": "allocated", "uids": 1, "epoch": 100, "reaction": 3, '
        '"best": 1.7, "worst": 1.7, "entropy_min": 0.26, "distinct": 20}',
        json.dumps({
            "event": "fleet_below_baseline", "raw_floor": 1.8, "worst_raw": 1.7,
            "prediction_offset": 0.09, "realized_floor": 1.71,
            "worst_predicted_realized": 1.61,
        }),
    ]

    parsed = compound._parse_loop_log("\n".join(lines))

    assert parsed["allocated"] == []
    assert parsed["blocked"] == [{
        "epoch": 100, "reaction": 3, "selected_score": 1.7, "baseline": 1.8,
        "prediction_offset": 0.09, "predicted_realized": 1.61,
        "realized_floor": 1.71, "final_gate": True,
    }]


def test_strategy_verdict_does_not_call_relative_lift_a_breakthrough():
    ab = {
        "arms": {
            "control": {"n": 40, "mean_source_delta": -0.08, "ge_entry": 0},
            "treatment": {"n": 200, "mean_source_delta": -0.06, "ge_entry": 0},
            "comparison": {"mean_source_delta_difference": 0.02},
        },
        "verdict": {"minimum_per_arm": 20, "status": "better"},
    }
    lanes = {"lanes": {
        "competitor_motif": {"max": 0.09},
        "frontier": {"max": 0.11},
    }}

    verdict = compound._strategy_verdict(ab, lanes)

    assert verdict["status"] == "relative_only"
    assert verdict["label"] == "relative signal; no entry hit"
    assert verdict["best_lane"] == "frontier"


@pytest.mark.parametrize("policy", ["field-recombination-v1", "field-recombination-v2"])
def test_field_recombination_uses_pre_proxy_random_arm_as_control(policy):
    lanes = {
        "active_strategy": policy,
        "active_geometry": "known-block-triple-pair-novel",
        "block_share": 300,
        "entry_score": 0.12,
        "lanes": {
            "block_bandit/field_recombination/random": {
                "scored": 25, "mean": 0.07, "max": 0.10, "ge_entry": 0,
            },
            "block_bandit/field_recombination/proxy": {
                "scored": 100, "mean": 0.08, "max": 0.13, "ge_entry": 1,
            },
            "block_bandit": {"max": 0.13},
        },
    }

    verdict = compound._strategy_verdict({}, lanes)

    assert verdict["mode"] == "field_recombination"
    assert verdict["status"] == "breakthrough"
    assert verdict["relative_lift"] == pytest.approx(0.01)
    assert verdict["control_n"] == 25
    assert verdict["treatment_n"] == 100
