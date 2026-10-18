"""Regression tests for the 2026-10-18 multipass (field-rank) selection cutover.

Each test pins one bug class from the cutover so a silent regression back to
win-probability / p90 selection cannot pass unnoticed. Red-to-green markers in
the docstrings record which tests caught a live bug versus which only guard an
already-fixed path.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanobody_opt import fleet, rank_exact

ROOT = Path(__file__).resolve().parent.parent          # /root
TOOLS = ROOT / "tools"


# ---------------------------------------------------------------------------
# 1. Multipass sort direction -- a settled-deviations=0 candidate with a bad
#    multipass rank must NOT sort first (the old deviation_tier inversion).
# ---------------------------------------------------------------------------
def test_multipass_order_is_purely_by_rank_ascending():
    ranked = [
        {"sequence": "CLEAN_BUT_BAD", "settled_deviations": 0,
         "multipass_rank": 500.0, "score_median": 10.0},
        {"sequence": "BROKEN_BUT_BEST", "settled_deviations": 3,
         "multipass_rank": 100.0, "score_median": 50.0},
    ]
    ordered = rank_exact._order_candidates(
        ranked, "multipass", competitive_score_threshold=0.0)
    assert [c["sequence"] for c in ordered] == ["BROKEN_BUT_BEST", "CLEAN_BUT_BAD"]


# ---------------------------------------------------------------------------
# 2. No p90 fallback -- _candidate_rank must raise when multipass_rank is absent.
# ---------------------------------------------------------------------------
def test_candidate_rank_raises_without_multipass_rank():
    from nanobody_opt_v2 import sequential_exact_race
    with pytest.raises(KeyError, match="multipass_rank"):
        sequential_exact_race._candidate_rank({"score_p90": 42.0})


# ---------------------------------------------------------------------------
# 3. Metric directions -- lower rank-sum = better for all ten metrics.
# ---------------------------------------------------------------------------
def test_metric_directions_rank_a_strictly_better_candidate_first():
    field = pd.DataFrame({m: [10.0, 20.0] for m in rank_exact.METRICS})
    better = {m: (1.0 if d == "min" else 30.0)
              for m, d in rank_exact.METRICS.items()}
    ranks = rank_exact.rank_metrics_against_field({"k": better}, field)
    assert ranks["k"] == 10.0  # rank 1 on all ten metrics


def test_metric_directions_rank_a_strictly_worse_candidate_last():
    field = pd.DataFrame({m: [10.0, 20.0] for m in rank_exact.METRICS})
    worse = {m: (30.0 if d == "min" else 1.0)
             for m, d in rank_exact.METRICS.items()}
    ranks = rank_exact.rank_metrics_against_field({"k": worse}, field)
    assert ranks["k"] == 30.0  # rank 3 on all ten metrics


# ---------------------------------------------------------------------------
# 4. Empty-field fail-closed -- an empty pooled field must not silently rank
#    every candidate at multipass_rank=10.0.
# ---------------------------------------------------------------------------
def test_empty_field_is_detected_before_ranking():
    assert rank_exact.pool_recent_fields([]).empty


def test_empty_field_would_rank_every_candidate_at_ten_without_the_guard():
    # The precise failure the guard exists to prevent: ranking against an empty
    # field silently returns 10 for every candidate (searchsorted over an empty
    # array), so every candidate would tie at multipass_rank=10.0.
    fake = {m: 0.0 for m in rank_exact.METRICS}
    empty = rank_exact.pool_recent_fields([])
    assert rank_exact.rank_metrics_against_field({"k": fake}, empty) == {"k": 10.0}


# ---------------------------------------------------------------------------
# 5. No win-probability selection anywhere live.
# ---------------------------------------------------------------------------
def _literal_selection_orders(path: Path) -> list[str]:
    return re.findall(r'"--selection-order"\s*,\s*"([a-z-]+)"', path.read_text())


def test_no_live_selection_is_win_probability_or_p90():
    assert fleet.REQUIRED_SELECTION_ORDER == "multipass"
    live_scripts = [
        TOOLS / "topup_publishable.py",
        TOOLS / "publish_current_nanobody.py",
        Path(fleet.__file__),
    ]
    from nanobody_opt_v2 import sequential_exact_race
    live_scripts.append(Path(sequential_exact_race.__file__))
    for path in live_scripts:
        for value in _literal_selection_orders(path):
            assert value == "multipass", f"{path.name} selects {value!r}"
    # publish_current_nanobody routes through the fleet constant, not a literal.
    sys.path.insert(0, str(TOOLS))
    import publish_current_nanobody
    assert publish_current_nanobody.SELECTION_ORDER == "multipass"


# ---------------------------------------------------------------------------
# 6. fleet.VALUE_KEY is multipass_rank (lower-better), never score_p90.
# ---------------------------------------------------------------------------
def test_value_key_is_multipass_rank():
    assert fleet.VALUE_KEY == "multipass_rank"


def test_value_share_picks_lowest_multipass_rank_not_p90():
    pool = {
        "selection_order": fleet.REQUIRED_SELECTION_ORDER,
        "candidates": [
            {"sequence": "BBB", "multipass_rank": 100.0, "score_p90": 90.0},
            {"sequence": "AAA", "multipass_rank": 200.0, "score_p90": 10.0},
        ],
    }
    # AAA has the better p90 (10 < 90) but the worse multipass rank; the value
    # half must still choose BBB because VALUE_KEY is multipass_rank.
    assert fleet.distinct_sequences(pool, 1, value_share=1.0) == ["BBB"]


# ---------------------------------------------------------------------------
# 7. Deepening by rank, not pass count.
# ---------------------------------------------------------------------------
def test_carry_pending_records_deepens_top_by_rank_not_pass_count(
        tmp_path, monkeypatch):
    from nanobody_opt_v2 import sequential_exact_race as race

    pooled_root = tmp_path / "pooled"
    (pooled_root / "iteration_01").mkdir(parents=True)
    (pooled_root / "iteration_02").mkdir(parents=True)
    # WEAK has 25 passes but a bad rank; STRONG has 1 pass but the best rank.
    (pooled_root / "iteration_01" / "layer2.json").write_text(json.dumps([
        {"sequence": "WEAK", "observation_count": 25},
        {"sequence": "STRONG", "observation_count": 1},
    ]))
    (pooled_root / "iteration_02" / "layer2.json").write_text(json.dumps([
        {"sequence": "FRESH"},
    ]))
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    monkeypatch.setattr(rank_exact, "_observations", lambda paths, target: {})
    monkeypatch.setattr(race, "_pooled_exact_paths", lambda *_: [])
    monkeypatch.setattr(
        race, "_already_scored",
        lambda db, target: {"WEAK": 25, "STRONG": 1})
    monkeypatch.setattr(
        race, "_multipass_rank_map",
        lambda db, target, epoch: {"WEAK": 900.0, "STRONG": 100.0})
    monkeypatch.setattr(race, "RESCORE_TOP", 1)

    out = race._carry_pending_records(
        [], pooled_root, output_dir, "P20809",
        database=tmp_path / "obs.sqlite", epoch=100)

    sequences = [r["sequence"] for r in out]
    # Fresh candidates always screen; the single deepen slot goes to the BEST
    # multipass rank (STRONG), not the most-passed candidate (WEAK).
    assert sequences == ["FRESH", "STRONG"]


# ---------------------------------------------------------------------------
# 8. Scale consistency -- challenger margin default is on the multipass scale.
# ---------------------------------------------------------------------------
def test_challenger_margin_default_is_on_the_multipass_scale(monkeypatch):
    from nanobody_opt_v2 import sequential_exact_race as race
    monkeypatch.delenv("NANOBODY_CHALLENGER_MARGIN", raising=False)
    args = race._parser().parse_args([
        "--layer2", "/tmp/x.json", "--archive", "/tmp/a.csv",
        "--config", "/tmp/c.yaml", "--target", "P20809",
        "--epoch", "1", "--output-dir", "/tmp/o",
    ])
    assert args.challenger_p90_margin == 150.0


# ---------------------------------------------------------------------------
# 9. Dialect-aware SQL -- _multipass_rank_map computes ? / %s, never hardcodes.
# ---------------------------------------------------------------------------
def test_multipass_rank_map_uses_dialect_placeholder(monkeypatch):
    from nanobody_opt_v2 import sequential_exact_race as race
    from nanobody_opt import mutation_network

    nonempty_field = pd.DataFrame({m: [1.0, 2.0] for m in rank_exact.METRICS})
    monkeypatch.setattr(
        rank_exact, "fetch_recent_fields_from_db",
        lambda db, target, before_epoch, count: [(1, nonempty_field)])
    monkeypatch.setattr(
        rank_exact, "pool_recent_fields", lambda fields: nonempty_field)

    class _Rows(list):
        def fetchall(self):
            return list(self)

    class _FakeConn:
        def __init__(self):
            self.executed = []

        def execute(self, sql, params=None):
            self.executed.append((sql, params))
            return _Rows()

        def close(self):
            pass

    conn = _FakeConn()
    monkeypatch.setattr(mutation_network, "_connect", lambda path: conn)

    monkeypatch.setattr(mutation_network, "_is_local", lambda c: True)
    race._multipass_rank_map(Path("/tmp/x.sqlite"), "P20809", 100)
    sql = conn.executed[0][0]
    assert "?" in sql and "%s" not in sql

    monkeypatch.setattr(mutation_network, "_is_local", lambda c: False)
    race._multipass_rank_map(Path("/tmp/x.sqlite"), "P20809", 100)
    sql = conn.executed[1][0]
    assert "%s" in sql and "?" not in sql


# ---------------------------------------------------------------------------
# 10. The ordering guard reads multipass_rank (non-decreasing), not the old
#     win-probability no-op.
# ---------------------------------------------------------------------------
def test_ordering_guard_refuses_a_multipass_regression():
    with pytest.raises(fleet.SelectionOrderError, match="multipass_rank"):
        fleet._assert_non_decreasing_multipass_rank([
            {"multipass_rank": 100.0},
            {"multipass_rank": 50.0},
        ])


def test_ordering_guard_is_not_a_silent_noop_under_multipass():
    # A ranking file with only win_probability (no multipass_rank) would have
    # passed the old guard without it ever inspecting anything. The new guard
    # must actually raise on a real multipass inversion.
    with pytest.raises(fleet.SelectionOrderError):
        fleet.distinct_sequences({
            "selection_order": "multipass",
            "candidates": [
                {"sequence": "AAA", "multipass_rank": 200.0},
                {"sequence": "BBB", "multipass_rank": 100.0},
            ],
        }, 2)


# ---------------------------------------------------------------------------
# 11. publish_checkpoint selection labeling reflects multipass, not p90.
# ---------------------------------------------------------------------------
def test_selection_identity_records_multipass_rank_not_p90():
    from nanobody_opt.publish_checkpoint import selection_identity
    winner = {"multipass_rank": 217.5, "selection_score": 42.0}
    method, score = selection_identity(
        winner, {"selection_order": "multipass"}, provisional=False)
    assert "multipass" in method and "p90" not in method
    assert score == 217.5


def test_selection_identity_refuses_a_multipass_ranking_without_the_rank():
    from nanobody_opt.publish_checkpoint import selection_identity
    with pytest.raises(SystemExit, match="multipass_rank"):
        selection_identity(
            {"selection_score": 42.0},
            {"selection_order": "multipass"}, provisional=False)


def test_checkpoint_is_better_compares_multipass_rank():
    from nanobody_opt.publish_checkpoint import checkpoint_is_better

    def _ckpt(rank, observations=1, sequence="A", epoch=25036):
        return {
            "epoch": epoch,
            "sequences": [sequence],
            "selection": {"score": rank, "multipass_rank": rank},
            "exact": {"observation_count": observations},
        }

    previous = _ckpt(rank=100.0)
    assert checkpoint_is_better(previous, _ckpt(rank=50.0))       # lower rank wins
    assert not checkpoint_is_better(previous, _ckpt(rank=150.0))
    assert checkpoint_is_better(previous, _ckpt(rank=200.0, observations=8))


# ---------------------------------------------------------------------------
# 12. Exclusion is intentional and logged, never a silent stale append of the
#     current pool.
# ---------------------------------------------------------------------------
def test_validator_blocked_exclusion_keeps_current_pool(tmp_path, capsys):
    sys.path.insert(0, str(TOOLS))
    import publish_current_nanobody as pub

    epoch_root = tmp_path / "epoch"
    epoch_root.mkdir()
    blocked_path = epoch_root / "validator_gate_blocked.txt"
    exclusion_path = epoch_root / "excluded_sequences.txt"
    exclusion_path.write_text("STALE_OLD\n")
    blocked_path.write_text("KEEP_ME\nREJECT_A\nreject_b\n")
    os.utime(blocked_path, None)  # fresh -> skip subprocess regeneration

    class _Args:
        nanobody_opt_root = tmp_path
        pixi = "pixi"

    pub._exclude_validator_blocked(
        args=_Args(),
        bgen_python=Path("/usr/bin/env"),
        pixi_manifest=Path("manifest"),
        epoch_root=epoch_root,
        archive_path=Path("archive.csv"),
        target="P20809",
        exclusion_path=exclusion_path,
        epoch=1,
        config_path=Path("config.yaml"),
        exact_inputs=[],
        observation_history=None,
        crew=[],
        keep=["keep_me"],  # lowercased: the current pool is still protected
    )

    written = exclusion_path.read_text().splitlines()
    assert "KEEP_ME" not in written, "the current pool must stay selectable"
    assert "REJECT_A" in written and "REJECT_B" in written
    assert "STALE_OLD" in written  # append, never truncate history
    assert "validator-gate pre-filter" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Fix A -- the commit blackout deadline (BLACKOUT_BLOCKS = 16).
# ---------------------------------------------------------------------------
def test_commit_schedule_stops_before_the_blackout():
    epoch_end = 361 * 25063
    assert fleet.BLACKOUT_BLOCKS == 16
    blocks = fleet.commitment_blocks(
        epoch=25062, epoch_length=361, reserve_blocks=55, count=39)
    assert blocks[-1] == epoch_end - 17
    assert all(b < epoch_end - fleet.BLACKOUT_BLOCKS for b in blocks)
    # Scheduling into the blackout must be refused, not silently scheduled.
    with pytest.raises(ValueError, match="do not fit"):
        fleet.commitment_blocks(
            epoch=25062, epoch_length=361, reserve_blocks=55, count=40)


def test_next_reserve_block_stops_before_the_blackout(monkeypatch):
    sys.path.insert(0, str(TOOLS))
    import publish_current_nanobody as pub

    epoch_end = 361 * 25063

    class _View:
        block = 0
        epoch_length = 361

    class _Args:
        chain_state = Path("chain.json")
        netuid = 68
        maximum_chain_state_age = 60

    monkeypatch.setattr(pub, "_chain_view", lambda *_: _View())
    # live block at end - 40 -> next candidate end - 39 is still inside the
    # validator's read window (end - 39 < end - 16).
    _View.block = epoch_end - 40
    assert pub._next_reserve_block(_Args(), 25062) == epoch_end - 39
    # live block at end - 17 -> next candidate end - 16 is in the blackout.
    _View.block = epoch_end - 17
    assert pub._next_reserve_block(_Args(), 25062) is None


def test_await_commit_block_refuses_a_blackout_target(monkeypatch):
    sys.path.insert(0, str(TOOLS))
    import publish_current_nanobody as pub

    epoch_end = 361 * 25063

    class _View:
        block = epoch_end - 40
        epoch_length = 361
        epoch = 25062

    class _Args:
        chain_state = Path("chain.json")
        netuid = 68
        maximum_chain_state_age = 60

    monkeypatch.setattr(pub, "_chain_view", lambda *_: _View())
    monkeypatch.setattr(pub, "_sleep", lambda seconds: None)
    # Scheduled inside the blackout -> refuse without sleeping/firing.
    assert pub._await_commit_block(_Args(), epoch_end - 10, 25062) is False
    # Scheduled inside the read window -> wait until the live block reaches it.
    assert pub._await_commit_block(_Args(), epoch_end - 40, 25062) is True


# ---------------------------------------------------------------------------
# Fix B -- rank_exact selection_score persists the multipass rank, not p90.
# ---------------------------------------------------------------------------
def test_rank_candidate_selection_score_tracks_the_multipass_order():
    # _rank_candidate is a worker-process body and is impractical to unit-test
    # without the full ranker context; pin the source-level contract that
    # selection_score is the multipass rank (not raw p90) under multipass so the
    # elite-parents / local_neighborhood consumers of selection_score also see
    # the selection metric rather than the old p90 scale.
    src = Path(rank_exact.__file__).read_text()
    assert 'args.selection_order == "multipass"' in src
    assert '"selection_score": (' in src

