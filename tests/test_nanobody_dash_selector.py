from __future__ import annotations

import json

from nanobody_dash.build import selector_replay_table


def test_selector_replay_table_highlights_the_deployed_policy(tmp_path):
    result = tmp_path / "selector_results.json"
    result.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "policy": "fresh-draw-win-probability",
                        "held_out_decisions": 144,
                        "oracle_pick_rate": 38 / 144,
                        "mean_regret": 83.5,
                        "median_regret": 65.0,
                    },
                    {
                        "policy": "validator-plain",
                        "held_out_decisions": 144,
                        "oracle_pick_rate": 32 / 144,
                        "mean_regret": 93.3,
                        "median_regret": 83.5,
                    },
                ]
            }
        )
    )

    html = selector_replay_table(result)

    assert "fresh-draw probability (selected)" in html
    assert "38/144" in html
    assert '<tr class="us">' in html
    assert "plain aggregate (old)" in html
