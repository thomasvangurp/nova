from nanobody_opt.audit_pairwise_shortlist import (
    _diagnosis,
    supported_routes,
)
from nanobody_opt.backtest_mutation_ranker import Parent


def _parent(sequence: str, score: float = 20.0) -> Parent:
    return Parent(sequence, 1, score, (0.0,) * 10)


def test_supported_routes_recognize_terminal_single_and_recombination():
    parent = _parent("AAAAAA")
    donors = ["CCCCCS", "CCCCTT"]

    single = supported_routes(
        "AAAAAS", [parent], donors, historical={parent.sequence}
    )
    double = supported_routes(
        "AAAATT", [parent], donors, historical={parent.sequence}
    )

    assert single.reachable
    assert single.minimum_edits == 1
    assert "exhaustive_single" in single.sources
    assert "c_terminal_shell" in single.sources
    assert double.reachable
    assert double.minimum_edits == 2
    assert "c_terminal_shell" in double.sources
    assert "position_recombination" in double.sources


def test_supported_routes_fail_closed_for_historical_or_unsupported_sequence():
    parent = _parent("AAAAAA")
    donors = ["CCCCCS"]

    historical = supported_routes(
        "AAAAAS", [parent], donors, historical={"AAAAAA", "AAAAAS"}
    )
    unsupported = supported_routes(
        "AAAAAW", [parent], donors, historical={parent.sequence}
    )

    assert not historical.reachable
    assert historical.historically_excluded
    assert not unsupported.reachable
    assert not unsupported.historically_excluded


def test_diagnosis_separates_discovery_from_allocation_failure():
    assert _diagnosis(False, False, False) == "operator_unreachable"
    assert _diagnosis(True, False, False) == "generation_miss"
    assert _diagnosis(True, True, False) == "allocation_miss"
    assert _diagnosis(True, True, True) == "included"
