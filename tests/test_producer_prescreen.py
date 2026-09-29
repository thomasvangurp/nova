import argparse
import json
from pathlib import Path

from nanobody_opt import run_live_producer as producer


def _args(tmp_path: Path, **extra) -> argparse.Namespace:
    return argparse.Namespace(runtime_root=tmp_path, chain_state=tmp_path / "chain_state.json",
                              prescreen_minimum_blocks=60, **extra)


def _state(offset: int) -> dict:
    return {"epoch": 25411, "block": 25411 * 361 + offset, "epoch_length": 361,
            "nanobody_target": "P20809"}


def test_prescreen_runs_the_next_epochs_iteration_at_its_start(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(producer, "_run_iteration",
                        lambda *, args, state, iteration: calls.append((args, state, iteration)))
    assert producer.run_prescreen(_args(tmp_path), _state(250)) is True
    (early, state, iteration), = calls
    assert state["epoch"] == 25412 and state["block"] % 361 == 0
    assert iteration == 1
    synthetic = json.loads(early.chain_state.read_text())
    assert synthetic["epoch"] == 25412 and synthetic["block"] == 25412 * 361
    root = tmp_path / "live" / "epoch25412" / "producer"
    assert early.chain_state.parent == root
    assert json.loads((root / "prescreen.json").read_text())["iterations"] == [1]


def test_prescreen_continues_numbering_and_stops_near_the_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "_run_iteration", lambda **_: None)
    root = tmp_path / "live" / "epoch25412" / "producer"
    (root / "iteration_01").mkdir(parents=True)
    assert producer.run_prescreen(_args(tmp_path), _state(250)) is True
    assert json.loads((root / "prescreen.json").read_text())["iterations"] == [2]
    # 361 - 320 = 41 blocks left is under the 60-block minimum
    assert producer.run_prescreen(_args(tmp_path), _state(320)) is False


def test_real_epoch_start_rebuilds_leaderboard_inputs_once(tmp_path):
    root = tmp_path / "producer"
    root.mkdir()
    (root / "prescreen.json").write_text(json.dumps({"iterations": [1], "refreshed": False}))
    for name in producer.PRESCREEN_STALE_INPUTS:
        (root / name).write_text("{}")
    producer.refresh_prescreen_inputs(root)
    assert not any((root / name).exists() for name in producer.PRESCREEN_STALE_INPUTS)
    (root / "winner_priors.json").write_text("{}")      # rebuilt by the real epoch
    producer.refresh_prescreen_inputs(root)
    assert (root / "winner_priors.json").exists()


def test_prescreen_is_off_unless_requested():
    assert producer._parser().parse_args([]).prescreen_next_epoch is False
