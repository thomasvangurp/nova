"""Regression tests for the six developability-hoist deploy gaps (2026-10-21).

Every test imports the SHIPPING tree: ``nanobody_opt_v2`` for the producer and
parent selection, ``nanobody_opt`` for the workers (``local_neighborhood``).
Source-level value regressions (winner_share, selection_policy, launch flags)
read the shipped files directly, matching the C7 fieldname-extraction precedent.
Bug tags map to TESTING.md G14-G16.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from sn68_source import ROOT, import_shipping


EXPECTED_POLICY = "agreedfix-2026-10-20-complete+developability-hoist"
LAUNCH_FILES = (
    ROOT / "go_live_v2.sh",
    ROOT / "restart_v2.sh",
    ROOT / "deploy" / "producer" / "producer_argv.txt",
)

spec = importlib.util.spec_from_file_location("sn68_ci_guardrails", ROOT / "scripts" / "ci_guardrails.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def selector(monkeypatch):
    return import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)


@pytest.fixture
def local_neighborhood(monkeypatch):
    return import_shipping("nanobody_opt.local_neighborhood", monkeypatch)


def _parent(seq: str, *, score: float, pct: float, in_range: bool, duplicate: bool = False) -> dict:
    return {
        "sequence": seq,
        "parent_cdr3": "CDR3-" + seq[:6],
        "validator_final_score": score,
        "public_epoch_percentile": pct,
        "parent_validator_duplicate": duplicate,
        "parent_cdr_developability_in_range": in_range,
        "parent_multipass_mean": None,
        "historical_reservoir": False,
    }


# ---------------------------------------------------------------------------
# Gap 1 -- launch config must retain shortlist-deferred TNP.
# ---------------------------------------------------------------------------
def test_gap1_launch_config_retains_shortlist_deferred_tnp():
    """The user restored shortlist TNP; redeployment must preserve it."""
    for path in LAUNCH_FILES:
        text = path.read_text()
        assert not gate._LAYER2_IGBLAST_BANNED.search(text), (
            f"{path.name} re-enables full-pool TNP")


def test_gap1_launch_config_gate_detects_both_flag_forms():
    for separator in (" ", "\n"):
        assert gate._LAYER2_IGBLAST_BANNED.search("--layer2-igblast-only" + separator + "0")
        assert not gate._LAYER2_IGBLAST_BANNED.search("--layer2-igblast-only" + separator + "1")


# ---------------------------------------------------------------------------
# Gap 2 -- out-of-range CDR parents are repair-pool routed, not normal seats.
# ---------------------------------------------------------------------------
def test_gap2_cdr_developability_in_range_classifies_red_region(selector):
    in_range = {
        "total_cdr_length": 30.0,
        "cdr3_length": 14.0,
        "cdr3_compactness": 1.0,
    }
    assert selector._cdr_developability_in_range(in_range) is True
    for bad in (
        {"total_cdr_length": 45.0, "cdr3_length": 14.0, "cdr3_compactness": 1.0},
        {"total_cdr_length": 30.0, "cdr3_length": 25.0, "cdr3_compactness": 1.0},
        {"total_cdr_length": 30.0, "cdr3_length": 14.0, "cdr3_compactness": 2.0},
        {"total_cdr_length": 30.0, "cdr3_length": 14.0},  # missing compactness
    ):
        assert selector._cdr_developability_in_range(bad) is False
    assert selector._cdr_developability_in_range(None) is False
    assert selector._cdr_developability_in_range("not-a-dict") is False


def test_gap2_out_of_range_parent_routed_to_repair_pool_not_normal_seats(selector):
    """An out-of-range nondup is stamped duplicate+repair and routed to the
    multi-edit repair lane, which keeps a seat even when in-range nondups fill
    the limit (GAP 2d: repair is never silently dropped)."""
    in_a = _parent("A" * 127, score=100.0, pct=0.1, in_range=True)
    in_b = _parent("B" * 127, score=90.0, pct=0.2, in_range=True)
    in_c = _parent("C" * 127, score=80.0, pct=0.3, in_range=True)
    in_d = _parent("D" * 127, score=70.0, pct=0.4, in_range=True)
    repair = _parent("E" * 127, score=60.0, pct=0.5, in_range=False)
    selected = selector._frontier_parents(
        [in_a, in_b, in_c, in_d, repair], {}, before_epoch=100, limit=4
    )
    by_seq = {item["sequence"]: item for item in selected}
    # The out-of-range parent survives, stamped as the repair-lane parent.
    assert "E" * 127 in by_seq
    repair_item = by_seq["E" * 127]
    assert repair_item["parent_developability_repair"] is True
    assert repair_item["parent_validator_duplicate"] is True
    # One in-range nondup seat was yielded to the repair lane, and only one.
    assert "D" * 127 not in by_seq
    assert {"A" * 127, "B" * 127, "C" * 127} <= set(by_seq)


# ---------------------------------------------------------------------------
# Gap 3 -- accepted-count replenishment targets ~4800 by default.
# ---------------------------------------------------------------------------
def test_gap3_child_floor_default_is_4800():
    from nanobody_opt_v2 import run_live_producer as producer
    assert producer._parser().parse_args([]).child_floor == 4800


# ---------------------------------------------------------------------------
# Gap 4 -- winner's length no longer monopolizes the generation budget, and
# the historical reservoir cycles distinct CDR3 families instead of letting the
# leading family backfill dominate.
# ---------------------------------------------------------------------------
def test_gap4_winner_share_is_half():
    source = (ROOT / "nanobody_opt" / "local_neighborhood.py").read_text()
    assert re.search(r"\bwinner_share = 0\.50\b", source), "winner share must be 0.50"
    assert re.search(r"\bwinner_share = 0\.70\b", source) is None, "0.70 bias must be gone"


def test_gap4_historical_reservoir_round_robins_families():
    source = (ROOT / "nanobody_opt_v2" / "run_live_producer.py").read_text()
    # Cycle distinct historical CDR3s before revisiting a family: the index
    # loops over families' i-th member, so the leading family cannot fill the
    # head of the pool on its own.
    assert "for family in families.values() if i < len(family)" in source
    assert "Parent order cycles distinct historical CDR3s before revisiting a family" in source


# ---------------------------------------------------------------------------
# Gap 5 -- the forced CDR3 position is the mutation anchor, so the residue
# actually changes and the child's compactness can move back into range.
# ---------------------------------------------------------------------------
def test_gap5_forced_cdr3_anchor_changes_the_residue(local_neighborhood):
    settled = local_neighborhood.settled_site_options
    parent = "A" * 20
    position = 5
    consensus = {position: ("G", 0.90)}  # settled; parent deviates (A != G)
    allowed = {position: ["A", "G", "K"]}
    # At the forced anchor the only residue offered is the consensus residue --
    # the parent residue cannot be retained, so the CDR3 genuinely changes.
    assert settled(position, parent=parent, consensus=consensus, allowed=allowed,
                   anchor=position, repair_only=True) == ["G"]
    # A settled NON-anchor position under repair_only may keep the parent residue
    # (optionally repair toward consensus), but never proposes a new deviation.
    assert settled(position, parent=parent, consensus=consensus, allowed=allowed,
                   anchor=None, repair_only=True) == ["A", "G"]


# ---------------------------------------------------------------------------
# Gap 6 -- the selection_policy bump is coherent: the producer's staleness
# check and the selector's written policy must agree on the hoisted value.
# ---------------------------------------------------------------------------
def test_gap6_selection_policy_bump_consistent_across_producer_and_selector():
    producer_source = (ROOT / "nanobody_opt_v2" / "run_live_producer.py").read_text()
    selector_source = (ROOT / "nanobody_opt_v2" / "select_elite_parents.py").read_text()
    producer_expected = re.findall(r'get\(\s*"selection_policy"\)\s*!=\s*"([^"]+)"', producer_source)
    selector_written = re.findall(r'"selection_policy"\s*:\s*"([^"]+)"', selector_source)
    assert producer_expected == [EXPECTED_POLICY], "producer staleness check must expect the hoisted policy"
    assert selector_written and set(selector_written) == {EXPECTED_POLICY}, "selector must write the hoisted policy"


# ---------------------------------------------------------------------------
# Dual-tree import gate self-tests (G15).
# ---------------------------------------------------------------------------
def test_dual_tree_scan_flags_legacy_v2_imports(tmp_path):
    bad = tmp_path / "test_bad.py"
    bad.write_text("from nanobody_opt.run_live_producer import X\nimport nanobody_opt.select_elite_parents\n")
    problems = gate.scan_dual_tree_imports(tmp_path)
    assert any("test_bad.py" in p and "run_live_producer" in p for p in problems)
    assert any("test_bad.py" in p and "select_elite_parents" in p for p in problems)


def test_dual_tree_scan_ignores_v2_and_worker_imports(tmp_path):
    ok = tmp_path / "test_ok.py"
    ok.write_text(
        "from nanobody_opt_v2 import run_live_producer\n"
        "import nanobody_opt_v2.select_elite_parents\n"
        "from nanobody_opt import local_neighborhood, check_validator_novelty\n"
        "import nanobody_opt.mutation_network\n"
    )
    assert gate.scan_dual_tree_imports(tmp_path) == []


def test_launch_config_gate_reports_clean_repo():
    assert gate.check_launch_config(ROOT) == []
