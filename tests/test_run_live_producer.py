from pathlib import Path
import hashlib
import json
import sqlite3
import time

import pytest

from nanobody_opt.run_live_producer import (
    _chain_state,
    _parser,
    maintain_runtime_storage,
)


def test_live_producer_uses_fresh_miner_chain_state(tmp_path):
    path = tmp_path / "chain.json"
    path.write_text(json.dumps({
        "epoch": 25034,
        "block": 25034 * 361 + 20,
        "epoch_length": 361,
        "nanobody_target": "P20809",
        "updated_at_unix": time.time(),
    }))

    state = _chain_state(path, maximum_age=90)

    assert state["epoch"] == 25034
    assert state["nanobody_target"] == "P20809"


def test_live_producer_rejects_stale_chain_state(tmp_path):
    path = tmp_path / "chain.json"
    path.write_text(json.dumps({
        "epoch": 25034,
        "block": 25034 * 361,
        "epoch_length": 361,
        "updated_at_unix": time.time() - 1000,
    }))

    with pytest.raises(ValueError, match="stale"):
        _chain_state(path, maximum_age=90)


def test_live_producer_uses_elite_multi_epoch_parent_defaults():
    args = _parser().parse_args([])

    assert args.parent_epochs == 0
    assert args.parents_per_epoch == 2
    assert args.parent_limit == 64
    assert args.seed_count == 32
    assert args.neighbor_seed_count == 32
    assert args.neighbor_parent_limit == 32
    assert args.recent_parent_epoch_window == 16
    assert args.recent_anchor_parent_limit == 16
    assert args.recent_neighbor_parent_limit == 16
    assert args.basin_core_limit == 5
    assert args.basin_parent_limit == 32
    assert args.minimum_seeds == 8
    assert args.per_seed == 12
    assert args.proposal_limit == 384
    assert args.basin_fraction == 0.70
    assert args.basin_routes_per_core == 192
    # None on purpose: the generator scales the lane ceiling to the proposal
    # budget. A pinned 8192 expired as the archive grew and failed every
    # iteration on nothing but age.
    assert args.maximum_exhaustive_singles is None
    assert args.candidate_limit == 320
    assert args.screen_tranche_size == 20
    assert args.maximum_initial_screen_p90 == 90.0
    assert args.weak_initial_fallback_after == 80
    assert args.finalists == 5
    assert args.exact_scorer.as_posix() == "/root/nanobody_opt/score_boltzgen_batch.py"
    assert args.basin_shrinkage_strength == 2.0
    assert args.minimum_basin_samples == 3
    assert args.challenger_p90_margin == 20.0
    assert args.single_mutation_fraction == 0.75


def test_the_producer_omits_the_lane_ceiling_unless_it_is_set():
    """Passing a fixed ceiling defeats the generator's scaling default."""
    import nanobody_opt.run_live_producer as rlp

    args = rlp._parser().parse_args([])
    assert args.maximum_exhaustive_singles is None
    # ...and the flag is only forwarded when explicitly set.
    source = (Path(__file__).resolve().parents[1]
              / "nanobody_opt" / "run_live_producer.py").read_text()
    assert "if args.maximum_exhaustive_singles is not None" in source


def test_storage_maintenance_prunes_only_old_regenerable_rank_matrices(tmp_path):
    old = tmp_path / "live/epoch10/producer/iteration_01/race"
    current = tmp_path / "live/epoch11/producer/iteration_01/race"
    (old / "rounds").mkdir(parents=True)
    (current / "rounds").mkdir(parents=True)
    for path in (
        old / "screen_ranking.json",
        old / "rounds/stage_ranking.json",
        old / "qualified_after_rounds.json",
        current / "screen_ranking.json",
    ):
        path.write_bytes(b"x" * 1024)

    report = maintain_runtime_storage(
        tmp_path,
        11,
        minimum_free_gib=0,
        maximum_persisted_ranking_mib=0.0001,
    )

    assert report["removed_files"] == 2
    assert not (old / "screen_ranking.json").exists()
    assert not (old / "rounds/stage_ranking.json").exists()
    assert (old / "qualified_after_rounds.json").exists()
    assert (current / "screen_ranking.json").exists()


def test_storage_maintenance_refuses_new_work_below_reserve(tmp_path, monkeypatch):
    import nanobody_opt.run_live_producer as producer
    from types import SimpleNamespace

    monkeypatch.setattr(
        producer.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )
    with pytest.raises(RuntimeError, match="disk reserve violated"):
        maintain_runtime_storage(tmp_path, 11, minimum_free_gib=1)


def _sealed_database(path: Path, exact: Path) -> None:
    digest = hashlib.sha256(exact.read_bytes()).hexdigest()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE exact_runs(
              target TEXT, run_id TEXT, source_path TEXT, file_sha256 TEXT,
              expected_rows INTEGER, ingested_rows INTEGER, complete INTEGER
            );
            CREATE TABLE exact_run_observations(
              target TEXT, run_id TEXT, observation_id TEXT
            );
            CREATE TABLE observation_artifacts(observation_id TEXT);
            """
        )
        connection.execute(
            "INSERT INTO exact_runs VALUES(?,?,?,?,?,?,?)",
            ("P20809", "run-1", str(exact), digest, 1, 1, 1),
        )
        connection.execute(
            "INSERT INTO exact_run_observations VALUES(?,?,?)",
            ("P20809", "run-1", "observation-1"),
        )
        connection.execute(
            "INSERT INTO observation_artifacts VALUES(?)", ("observation-1",)
        )


def test_storage_maintenance_removes_only_database_sealed_old_epochs(tmp_path):
    old = tmp_path / "live/epoch10/producer/iteration_01/race/exact_all.json"
    recent = tmp_path / "live/epoch20/producer/iteration_01/race/exact_all.json"
    old.parent.mkdir(parents=True)
    recent.parent.mkdir(parents=True)
    old.write_text('{"run": 1}')
    recent.write_text('{"run": 2}')
    database = tmp_path / "observations.sqlite"
    _sealed_database(database, old)

    report = maintain_runtime_storage(
        tmp_path,
        30,
        minimum_free_gib=0,
        observation_database=database,
        retained_epochs=16,
    )

    assert report["removed_epoch_dirs"] == 1
    assert not (tmp_path / "live/epoch10").exists()
    assert (tmp_path / "live/epoch20").exists()


def test_storage_maintenance_keeps_old_epoch_when_audit_row_is_missing(tmp_path):
    old = tmp_path / "live/epoch10/producer/iteration_01/race/exact_all.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"run": 1}')
    database = tmp_path / "observations.sqlite"
    _sealed_database(database, old)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM observation_artifacts")

    report = maintain_runtime_storage(
        tmp_path,
        30,
        minimum_free_gib=0,
        observation_database=database,
        retained_epochs=16,
    )

    assert report["removed_epoch_dirs"] == 0
    assert (tmp_path / "live/epoch10").exists()
