import hashlib
import json

from nanobody_opt import filter_layer2
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


def test_execution_failure_is_retried_instead_of_cached(tmp_path, monkeypatch):
    sequence = "QVQLVESGG"
    digest = filter_layer2._developability_cache_digest(sequence)
    (tmp_path / f"{digest}.json").write_text(
        json.dumps(
            {
                "sequence": sequence,
                "developability": {"passed": False, "reason": "TNP crashed"},
            }
        )
    )
    measured = {"passed": True, "flags": {"PSH": "green"}}

    async def fake_developability(sequences, concurrency):
        assert sequences == [sequence]
        assert concurrency == 1
        return [measured]

    monkeypatch.setattr(filter_layer2, "_developability", fake_developability)

    assert filter_layer2._cached_developability(
        [{"sequence": sequence}], tmp_path, concurrency=1
    ) == [measured]
    cached = json.loads((tmp_path / f"{digest}.json").read_text())
    assert cached["developability"] == measured


def test_unmeasured_results_never_count_as_passes_or_biological_rejections():
    assert filter_layer2.developability_rejection({"passed": False, "reason": "failed"}) == "developability_unavailable"
    assert filter_layer2.developability_rejection({"passed": True}) == "developability_unavailable"
    assert filter_layer2.developability_rejection({"passed": False, "flags": {"C": "red"}}) == "developability"
    assert filter_layer2.developability_rejection({"passed": True, "flags": {"C": "green"}}) == ""


def test_cache_requires_current_version_and_matching_input(tmp_path):
    seq = "TEST"
    payload = {"sequence": seq, "developability": {"passed": True, "flags": {"C": "green"}}}
    (tmp_path / (hashlib.sha256(seq.encode()).hexdigest() + ".json")).write_text(json.dumps(payload))
    assert filter_layer2.cached_developability_result(seq, tmp_path) is None
    path = tmp_path / (filter_layer2._developability_cache_digest(seq) + ".json")
    path.write_text(json.dumps({**payload, "sequence": "different"}))
    assert filter_layer2.cached_developability_result(seq, tmp_path) is None
    path.write_text(json.dumps(payload))
    assert filter_layer2.cached_developability_result(seq, tmp_path)["passed"] is True
