import pytest

from nanobody_opt.score_candidate_liability import attach_liability


def test_attaches_exact_liability_and_sorts():
    cases = [
        {"sequence": "AAAA", "donor_rank_sum": 1},
        {"sequence": "CCCC", "donor_rank_sum": 2},
    ]
    scores = {
        "AAAA": {"score": 20, "violations": [{}, {}]},
        "CCCC": {"score": 10, "violations": [{}]},
    }
    got = attach_liability(cases, scores)
    assert [item["sequence"] for item in got] == ["CCCC", "AAAA"]
    assert got[0]["liability_num_violations"] == 1


def test_rejects_incomplete_measurement():
    with pytest.raises(ValueError, match="missing liability"):
        attach_liability([{"sequence": "AAAA"}], {})
