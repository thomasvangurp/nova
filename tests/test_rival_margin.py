from tools import rival_margin


def rows(reaction: int, count: int = 20) -> list[dict]:
    return [
        {"epoch": epoch, "coldkey": "rival", "reaction": reaction, "score": 1.0}
        for epoch in range(1, count + 1)
    ]


def test_opportunity_gate_uses_no_lookahead_calibration():
    floor, info = rival_margin.required_margin(
        3, rows(3), 21, optimism=0.10, optimism_samples=[0.05],
    )

    gate = info["opportunity_gate"]
    assert gate["enabled"] is True
    assert gate["target_win_probability"] == 0.25
    assert gate["residual_samples"] >= rival_margin.OPPORTUNITY_MIN_RESIDUALS
    assert floor < gate["standard_margin"]
    assert gate["adjustment"] >= -rival_margin.RESIDUAL


def test_non_opportunity_reaction_keeps_standard_margin():
    floor, info = rival_margin.required_margin(
        2, rows(2), 21, optimism=0.10, optimism_samples=[0.01],
    )

    assert "opportunity_gate" not in info
    assert floor == info["margin"]
    assert floor == 1.104


def test_opportunity_gate_requires_enough_historical_residuals():
    floor, info = rival_margin.required_margin(
        1, rows(1, count=5), 6, optimism=0.10, optimism_samples=[0.01],
    )

    assert info["opportunity_gate"]["enabled"] is False
    assert info["opportunity_gate"]["reason"] == "insufficient_no_lookahead_residuals"
    assert floor == info["margin"]
