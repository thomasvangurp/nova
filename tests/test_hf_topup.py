"""R1 HF/rival parent top-up: bounded on the writer side, rank-merged on the reader side.

The central property (ARCHITECTURE_REDESIGN R3): the producer-scale top-50
(~20-370) and the HF/rival top-up list (~1e5, sum10 vs the full HF archive)
are never compared by raw score value.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanobody_opt import build_top50_index as b50
from nanobody_opt import hf_topup as H

ROOT = Path(__file__).resolve().parents[1]
L = 20


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mp():
    return _load("multipass_all_rivals_topup_t", ROOT / "tools" / "multipass_all_rivals.py")


def _seq(i: int, base: str = "A") -> str:
    """Distinct length-L sequences, pairwise far apart."""
    rng = np.random.default_rng(i)
    return "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), L))


# --- merge_by_rank: positions only -----------------------------------------
def test_merge_takes_no_scores():
    params = set(inspect.signature(H.merge_by_rank).parameters)
    assert params == {"primary", "topup", "slots", "stride"}


def test_merge_interleaves_reserved_slots_and_keeps_primary_order():
    primary = [f"P{i}" for i in range(50)]
    topup = [f"T{i}" for i in range(30)]
    merged, pos = H.merge_by_rank(primary, topup, slots=10, stride=4)
    assert len(merged) == 60
    assert pos == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    assert [merged[i] for i in pos] == topup[:10]
    assert [s for s in merged if s.startswith("P")] == primary
    assert merged[:8] == ["P0", "P1", "P2", "T0", "P3", "P4", "P5", "T1"]


def test_merge_skips_topup_already_in_primary_and_handles_short_lists():
    merged, pos = H.merge_by_rank(["A", "B"], ["A", "X", "Y"], slots=10, stride=4)
    assert merged == ["A", "B", "X", "Y"] and pos == [2, 3]
    merged, pos = H.merge_by_rank(["A", "B", "C"], [], slots=10, stride=4)
    assert merged == ["A", "B", "C"] and pos == []
    with pytest.raises(ValueError):
        H.merge_by_rank(["A"], ["B"], stride=1)


class _FakeConn:
    """Stands in for db.connect(): returns producer replay rows for the SQL."""

    def __init__(self, rows):
        self.rows = rows
        self.sql: list[str] = []
        self.params: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.sql.append(sql)
        self.params.append(params)
        rows = self.rows

        class _R:
            def fetchall(self_inner):
                return rows
        return _R()


def _fake_db(monkeypatch, connect):
    """Install a stand-in nanobody_opt.db (the real one needs MySQLdb)."""
    import types

    import nanobody_opt
    fake = types.ModuleType("nanobody_opt.db")
    fake.connect = connect
    monkeypatch.setitem(sys.modules, "nanobody_opt.db", fake)
    monkeypatch.setattr(nanobody_opt, "db", fake, raising=False)


def _live(monkeypatch, producer_rows, topup):
    conn = _FakeConn(producer_rows)
    _fake_db(monkeypatch, lambda *a, **k: conn)
    report: dict = {}
    out = b50._live_top50_from_db("unused", limit=5, extra=2, min_distance=6,
                                  topup=topup, report=report)
    return out, report, conn


def test_live_top50_merges_by_rank_not_by_score(monkeypatch):
    """However the top-up's own scores compare with the producer's (both
    'better' and 'worse' on raw value), its slots are the same: only its rank
    within its own list matters, because no score crosses the boundary."""
    producer = [(_seq(i), 20.0 + i, 12) for i in range(12)]  # ~20-370 scale
    hf = [_seq(100 + i) for i in range(3)]
    # Producer-side read only ever sees producer rows; the top-up arrives as
    # an ordered list. Same order -> same output, irrespective of the values
    # the top-up was ranked by (1e5 vs 1.0) -- they never reach this function.
    out_a, rep_a, conn = _live(monkeypatch, producer, hf)
    out_b, rep_b, _ = _live(monkeypatch, producer, list(hf))
    assert out_a == out_b
    assert rep_a["producer_top50"] == [r[0] for r in producer[:5]]
    # limit 5: one stride-4 slot, then the rest after the primary list
    assert rep_a["topup_positions"] == [3, 6, 7]
    assert out_a[3] == hf[0]
    # The producer-scale SQL is unchanged: REPLAY_SCALE_SOURCES only, no
    # multipass label, so the two scales are not pooled in AVG(score).
    sql = conn.sql[0]
    assert "multipass_sum10_pass" not in sql
    assert set(b50.REPLAY_SCALE_SOURCES) <= set(conn.params[0])
    assert "public_multipass_rivals" not in conn.params[0]


def test_live_top50_without_topup_is_unchanged(monkeypatch):
    producer = [(_seq(i), 20.0 + i, 12) for i in range(12)]
    out, rep, _ = _live(monkeypatch, producer, None)
    out2, _, _ = _live(monkeypatch, producer, [])
    assert out == out2 == rep["producer_top50"] + rep["filler"]


def test_live_top50_still_fails_closed_on_empty_producer_half(monkeypatch):
    with pytest.raises(SystemExit, match="TOP50_EMPTY"):
        _live(monkeypatch, [], [_seq(1)])


def test_topup_failure_degrades_to_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    _fake_db(monkeypatch, boom)
    monkeypatch.setattr(H, "DISABLE_FILE", Path("/nonexistent/switch"))
    monkeypatch.setattr("nanobody_opt.failure_log.record_failure",
                        lambda *a, **k: None, raising=False)
    seqs, info = b50._hf_rival_topup(pd.DataFrame({"sequence": []}), [])
    assert seqs == [] and info["status"] == "error"


def test_topup_kill_switch(monkeypatch, tmp_path):
    switch = tmp_path / "DISABLE_TOP50_TOPUP"
    switch.write_text("")
    monkeypatch.setattr(H, "DISABLE_FILE", switch)
    seqs, info = b50._hf_rival_topup(pd.DataFrame({"sequence": []}), [])
    assert seqs == [] and info["status"] == "disabled"


# --- the derived score: same formula as the writer's -----------------------
def test_sum10_table_matches_the_writer(mp):
    assert H.SUM_OF_TEN_DIRS == mp.SUM_OF_TEN_DIRS


def test_sum10_against_field_equals_writer_formula_per_row(mp):
    rng = np.random.default_rng(0)
    cols = list(H.SUM_OF_TEN_DIRS)
    field = pd.DataFrame(rng.integers(0, 30, size=(200, len(cols))).astype(float),
                         columns=cols, index=[f"f{i}" for i in range(200)])
    rows = pd.DataFrame(rng.integers(0, 30, size=(15, len(cols))).astype(float),
                        columns=cols, index=[f"r{i}" for i in range(15)])
    got = H.sum10_against_field(rows, field)
    for name, row in rows.iterrows():
        expect = mp.sum_of_ten(pd.concat([field, row.to_frame().T]))[name]
        assert got[name] == pytest.approx(float(expect))


# --- writer side: only the missing passes, within budget -------------------
def test_plan_scores_only_missing_passes():
    plan = H.plan_topup(["a", "b", "c", "d"], {"a": 10, "b": 7, "c": 0},
                        exclude={"d"}, depth=10, budget=500)
    assert plan == {"b": 3, "c": 10}


def test_plan_respects_budget_and_order():
    plan = H.plan_topup([f"s{i}" for i in range(10)], {}, depth=10, budget=25)
    assert plan == {"s0": 10, "s1": 10, "s2": 5}
    assert H.pass_batches({"x": 2, "y": 1}) == [["x", "y"], ["x"]]
    assert H.pass_batches({}) == []


def test_eligible_parents_drops_near_validator_top50():
    ref = ["AAAAAAAAAA"]
    cands = ["AAAAAAAAAC", "CCCCCCAAAA", "CCCCCCCCCCC"]
    assert H.eligible_parents(cands, ref, 6) == ["CCCCCCAAAA", "CCCCCCCCCCC"]
    assert H.eligible_parents(cands, ref, 6, length=10) == ["CCCCCCAAAA"]
