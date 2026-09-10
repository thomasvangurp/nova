import json

from nanobody_opt.filter_layer2 import _load_candidates


def test_load_candidates_accepts_exact_scorer_cases(tmp_path):
    input_path = tmp_path / "cases.json"
    input_path.write_text(
        json.dumps(
            {
                "target": "P20809",
                "cases": [
                    {"sequence": " qvqlvesgg ", "mutation": "Q3K"},
                    {"sequence": "QVQLVESGG", "mutation": "duplicate"},
                    {"sequence": "EVQLVESGG", "mutation": "L4M"},
                ],
            }
        )
    )

    assert _load_candidates(input_path) == [
        {"sequence": "QVQLVESGG", "mutation": "Q3K"},
        {"sequence": "EVQLVESGG", "mutation": "L4M"},
    ]
