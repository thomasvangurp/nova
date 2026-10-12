"""Strict qualification looks at the top of a fresh ranking, not at all ~1,000 candidates.

The strict pass used to take ~25 minutes, so every qualified ranking reflected a database snapshot that old and
nothing written after about block 200 of an epoch (the second half of the confirmation, the top-up) reached the
published tickets.  --strict-pool-size restricts the per-candidate qualification to the top N of a fresh
provisional-mode ranking; history and shrinkage still use the full pool (rank_exact --only-sequences).
"""
import argparse
import json
import subprocess
from pathlib import Path

from tools import publish_current_nanobody as pub


def _args(tmp_path, size=100):
    return argparse.Namespace(nanobody_opt_root=tmp_path / "nanobody_opt", strict_pool_size=size)


def _call(tmp_path, *, size=100, crew=13, previous=None):
    epoch_root = tmp_path / "epoch"
    epoch_root.mkdir(exist_ok=True)
    return pub._strict_pool(
        args=_args(tmp_path, size), bgen_python=Path("python"), epoch_root=epoch_root,
        archive_path=tmp_path / "archive.csv", target="P20809", exclusion_path=tmp_path / "ex.txt",
        epoch=25674, shortlist=1000, config_path=tmp_path / "config.yaml", exact_inputs=(),
        observation_history=tmp_path / "db.sqlite", crew=[object()] * crew, previous=previous,
    ), epoch_root


def _fake_rank(monkeypatch, n):
    seen = {}

    def fake_run(command):
        seen["command"] = command
        out = Path(command[command.index("--output") + 1])
        out.write_text(json.dumps({"candidates": [{"sequence": f"SEQ{i:03d}"} for i in range(n)]}))

    monkeypatch.setattr(pub, "_run", fake_run)
    return seen


def test_rank_command_only_adds_only_sequences_when_asked(tmp_path):
    base = dict(bgen_python=Path("python"), nanobody_opt_root=tmp_path, exact_inputs=(), archive=tmp_path / "a.csv",
                target="P20809", output=tmp_path / "o.json", exclusion=tmp_path / "e.txt", epoch=25674,
                shortlist=1000, config=tmp_path / "c.yaml", strict=True, observation_history=tmp_path / "db")
    assert "--only-sequences" not in pub._rank_command(**base)
    command = pub._rank_command(**base, only_sequences=tmp_path / "pool.txt")
    assert command[command.index("--only-sequences") + 1] == str(tmp_path / "pool.txt")


def test_pool_is_the_top_of_a_fresh_provisional_ranking(tmp_path, monkeypatch):
    seen = _fake_rank(monkeypatch, 400)
    path, epoch_root = _call(tmp_path, size=100)
    assert path == epoch_root / "strict_pool_sequences.txt"
    assert path.read_text().split() == [f"SEQ{i:03d}" for i in range(100)]
    assert "--only-sequences" not in seen["command"]          # the pool itself is a plain provisional ranking
    assert "--pose-config" not in seen["command"]             # ... without the expensive qualification


def test_size_zero_keeps_the_old_whole_shortlist_behaviour(tmp_path, monkeypatch):
    seen = _fake_rank(monkeypatch, 400)
    path, _ = _call(tmp_path, size=0)
    assert path is None and "command" not in seen              # no ranking is even run


def test_a_failed_pool_ranking_falls_back_to_the_whole_shortlist(tmp_path, monkeypatch):
    def boom(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pub, "_run", boom)
    path, _ = _call(tmp_path)
    assert path is None


def test_a_pool_too_thin_to_fill_the_fleet_is_not_used(tmp_path, monkeypatch):
    _fake_rank(monkeypatch, 20)
    path, _ = _call(tmp_path, crew=13)                         # 20 < 3 * 13
    assert path is None


def test_what_this_epoch_already_published_stays_selectable(tmp_path, monkeypatch):
    _fake_rank(monkeypatch, 400)
    monkeypatch.setattr(pub, "_live_sequences", lambda previous, epoch: ["PUBLISHED_A", "SEQ001", "PUBLISHED_B"])
    path, _ = _call(tmp_path, size=100, previous={"epoch": 25674})
    lines = path.read_text().split()
    assert lines[:100] == [f"SEQ{i:03d}" for i in range(100)]
    assert lines[100:] == ["PUBLISHED_A", "PUBLISHED_B"]       # appended once, no duplicate of SEQ001


def test_the_flag_defaults_on_with_a_pool_of_one_hundred():
    parser = pub._parser()
    assert parser.parse_args([]).strict_pool_size == 100
    assert parser.parse_args(["--strict-pool-size", "0"]).strict_pool_size == 0
