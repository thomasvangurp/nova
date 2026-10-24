import json

import pytest

from nanobody_opt_v2 import run_live_producer as producer
from sn68_source import import_shipping


def state(epoch):
    return {"epoch": epoch, "epoch_length": 361, "block": epoch * 361,
            "nanobody_target": "TEST"}


@pytest.mark.parametrize("limit,expected", [(None, ["A", "B", "C", "D"]), (3, ["A", "B", "C"])])
def test_child_lane_tail_respects_explicit_proposal_budget(monkeypatch, limit, expected):
    generator = import_shipping("nanobody_opt.local_neighborhood", monkeypatch)
    result = generator._append_proposal_tail(
        [{"sequence": "A"}], [{"sequence": s} for s in ["A", "B", "C", "D"]], limit=limit)
    assert [c["sequence"] for c in result] == expected


def test_prepare_once_preserves_oversupply_for_fresh_epoch_gate(tmp_path, monkeypatch):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    calls = []
    def prepare(**kw):
        calls.append(kw)
        assert kw["prepare_only"]
        assert producer._STAGED.deadline.epoch == 71
        assert producer._STAGED.nice is True
        assert kw["args"].generation_proposal_budget == 8000
        assert kw["args"].child_floor == 8000
        assert kw["args"].only_unseen_candidates
        assert sum(producer._generation_quotas(kw["args"], 8000).values()) == 8000
        producer._atomic_json(kw["iteration_root_override"] / "layer2_dedup.json", {
            "candidates": [{"sequence": f"SEQ{i}"} for i in range(6500)],
        })
    monkeypatch.setattr(producer, "_run_iteration", prepare)
    first = producer._prepare_epoch_pool(args, state(71), 72)
    assert producer._prepare_epoch_pool(args, state(72), 72) == first
    assert len(calls) == 1
    pool = json.loads(first.read_text())
    assert len(pool["candidates"]) == 6500
    assert pool["exact_admission_limit"] == 4800
    assert pool["screen_budget_policy"] == "unseen-candidates-v2"
    assert pool["prepared_during_epoch"] == 71
    assert pool["origin_epoch"] == 72


def test_live_admission_filters_first_then_uses_strongest_4800(tmp_path, monkeypatch):
    import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.complete_screen_until_epoch = 72
    args.complete_screen_pool = tmp_path / "pool.json"
    producer._atomic_json(args.complete_screen_pool, {
        "target": "TEST", "origin_epoch": 72, "exact_admission_limit": 4800,
        "candidates": [{"sequence": f"SEQ{i}", "parent_multipass_mean": 8000 - i}
                       for i in range(8000)],
    })
    monkeypatch.setattr(producer, "_prepare_epoch", lambda **kw: (tmp_path / "cfg", tmp_path / "archive", "TEST"))
    # Reject 2000 of the strongest candidates at the actual epoch's novelty gate.
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: {f"SEQ{i}" for i in range(6000)})
    monkeypatch.setattr(producer, "_complete_pool_coverage", lambda **kw: {"complete": False})
    calls = []
    monkeypatch.setattr(producer, "_run_race", lambda **kw: calls.append(kw))
    producer._run_complete_screen_pool(args, state(72))
    assert len(calls) == 1
    pool = json.loads(calls[0]["layer2"].read_text())
    assert len(pool["candidates"]) == 4800
    assert set(pool["sequences"]) == {f"SEQ{i}" for i in range(1200, 6000)}
    assert pool["novelty_rejected"] == 2000
    assert pool["deferred"] == 1200
    assert calls[0]["args"].screen_all_candidates
    assert calls[0]["args"].finalists == 64


@pytest.mark.parametrize("confirmed,epoch,expected", [(False, 71, []), (True, 70, []), (True, 71, [72])])
def test_next_pool_waits_for_current_confirmation(tmp_path, monkeypatch, confirmed, epoch, expected):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.epoch_cycle_start = 72
    if confirmed:
        producer._atomic_json(tmp_path / f"live/epoch{epoch}/producer/full_scan_confirmation.json", {})
    monkeypatch.setattr(producer, "_chain_state", lambda *a: state(epoch))
    calls = []
    monkeypatch.setattr(producer, "_prepare_epoch_pool", lambda a, s, e: calls.append(e))
    producer._prepare_following_epoch_pool(args, state(epoch))
    assert calls == expected


def test_next_pool_overlaps_confirmation_without_blocking_or_duplicate_preparation(tmp_path, monkeypatch):
    import threading
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.epoch_cycle_start = 72
    entered, release = threading.Event(), threading.Event()
    calls = []
    def prepare(a, current, following):
        calls.append((current["epoch"], following))
        entered.set()
        assert release.wait(2)
    monkeypatch.setattr(producer, "_prepare_epoch_pool", prepare)
    producer._start_following_epoch_pool(args, state(72))
    assert entered.wait(1)
    try:
        producer._start_following_epoch_pool(args, state(72))
        producer._atomic_json(tmp_path / "live/epoch72/producer/full_scan_confirmation.json", {})
        producer._prepare_following_epoch_pool(args, state(72))
        assert calls == [(72, 73)]
    finally:
        release.set()
        producer._EPOCH_POOL_PREP[(str(tmp_path.resolve()), 73)]["thread"].join(2)
    assert producer._EPOCH_POOL_PREP[(str(tmp_path.resolve()), 73)]["complete"]


def test_failed_speculative_preparation_is_recoverable(tmp_path, monkeypatch):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.epoch_cycle_start = 72
    def fail(*a):
        raise RuntimeError("interrupted preparation")
    monkeypatch.setattr(producer, "_prepare_epoch_pool", fail)
    producer._start_following_epoch_pool(args, state(72))
    record = producer._EPOCH_POOL_PREP[(str(tmp_path.resolve()), 73)]
    record["thread"].join(2)
    assert record["error_type"] == "RuntimeError" and not record["complete"]
    monkeypatch.setattr(producer, "_prepare_epoch_pool", lambda *a: None)
    producer._start_following_epoch_pool(args, state(72))
    record = producer._EPOCH_POOL_PREP[(str(tmp_path.resolve()), 73)]
    record["thread"].join(2)
    assert record["complete"]


@pytest.mark.parametrize("offset,expected", [(0, []), (220, [73])])
def test_race_starts_next_pool_during_confirmation_only(tmp_path, monkeypatch, offset, expected):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.epoch_cycle_start = 72
    current = {**state(72), "block": 72 * 361 + offset}
    monkeypatch.setattr(producer, "_chain_state", lambda *a: current)
    monkeypatch.setattr(producer, "_budget_plan", lambda *a: {
        "replication_reserve_blocks": 127, "confirmation_candidates": 64,
        "gpu_seconds_per_candidate": 4.8, "load_seconds": 45,
    })
    monkeypatch.setattr(producer, "_screen_budget", lambda *a: (700, 4.8))
    starts = []
    monkeypatch.setattr(producer, "_start_following_epoch_pool", lambda a, s: starts.append(s["epoch"] + 1))
    monkeypatch.setattr(producer, "_run", lambda command: None)
    producer._run_race(args=args, state=current, iteration=1, config=tmp_path / "config.yaml",
        archive=tmp_path / "archive.csv", target="TEST", producer_root=tmp_path,
        iteration_root=tmp_path / "iteration_01", layer2=tmp_path / "layer2.json")
    assert starts == expected


@pytest.mark.parametrize("epoch,route", [(71, "full"), (72, "cycle"), (73, "cycle")])
def test_cycle_routes_without_interrupting_existing_full_pool(tmp_path, monkeypatch, epoch, route):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.once = True
    args.complete_screen_pool = tmp_path / "pool.json"
    args.complete_screen_until_epoch = 71
    args.epoch_cycle_start = 72
    monkeypatch.setattr(producer, "_chain_state", lambda *a: state(epoch))
    monkeypatch.setattr(producer, "maintain_runtime_storage", lambda *a, **kw: {})
    monkeypatch.setattr(producer, "_prepare_following_epoch_pool", lambda *a: None)
    monkeypatch.setattr(producer, "_run_complete_screen_pool", lambda *a: "full")
    monkeypatch.setattr(producer, "_run_epoch_cycle", lambda *a: "cycle")
    assert producer._one_pass(args, {}, set(), set()) == route


def test_fallback_cannot_overshoot_screen_floor_and_retries_rejections(tmp_path, monkeypatch):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.generation_proposal_budget = 8000
    parent = {"sequence": "AA", "parent_cdr3": "family", "parent_validator_duplicate": False,
              "parent_top50_identity_in_basin": False, "layer2_passed": True,
              "parent_cdr_developability_in_range": True}
    producer._atomic_json(tmp_path / "artifacts/agreedfix_historical_parents.json", {"candidates": [parent]})
    destination = tmp_path / "pool.json"
    producer._atomic_json(destination, {"candidates": []})
    monkeypatch.setattr(selector, "_validator_parent_pool", lambda pool, **kw: pool)
    monkeypatch.setattr(selector, "_developability_parent_pool", lambda pool, **kw: pool)
    attempts = []
    def run(command):
        source = json.loads(open(command[command.index("--input") + 1]).read())
        attempts.append([c["sequence"] for c in source["candidates"]])
        producer._atomic_json(producer.Path(command[command.index("--output") + 1]), source)
    monkeypatch.setattr(producer, "_run", run)
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: set(attempts[-1]) if len(attempts) > 1 else set())
    producer._historical_child_fallback(args=args, target="TEST", config=tmp_path / "cfg",
                                       hf_archive=tmp_path / "archive", iteration_root=tmp_path,
                                       destination=destination, floor=2)
    assert attempts == [["CA", "DA"], ["EA", "FA"]]
    assert len(json.loads(destination.read_text())["candidates"]) == 2


def test_fresh_budget_ignores_cache_and_survives_restart(tmp_path, monkeypatch):
    from nanobody_opt_v2 import screen_budget
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    monkeypatch.setattr(race, "_prior_evaluation_sequences", lambda *a, **kw: {"OLD"})
    parser = race._parser()
    args = parser.parse_args([
        "--layer2", "pool.json", "--archive", "archive.csv", "--config", "config.yaml",
        "--target", "TEST", "--epoch", "101", "--output-dir", str(tmp_path / "race"),
        "--pooled-exact-root", str(tmp_path), "--gpus", "0", "--carry-pending-candidates",
        "--screen-all-candidates", "--fresh-screen-budget", "3",
    ])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(race, "_parser", lambda: parser)
    records = [{"sequence": s} for s in ["OLD", "NEW", "THIRD", "EXTRA"]]
    monkeypatch.setattr(race, "_candidate_records", lambda p: records)
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: False)
    race.main()
    assert json.loads((tmp_path / "race/screen_candidates.json").read_text())["sequences"] == ["NEW", "THIRD", "EXTRA"]
    # Charge before launch, so even a failed/interrupted GPU dispatch cannot
    # be silently retried outside the cap. Cheap rejected records never enter it.
    assert screen_budget.reserve(["OLD", "NEW"], tmp_path, 101, 3) == ["OLD", "NEW"]
    assert screen_budget.pending(records, tmp_path, 101, 3) == [{"sequence": "THIRD"}]
    assert screen_budget.reserve(["NEW", "THIRD", "EXTRA"], tmp_path, 101, 3) == ["THIRD"]
    assert screen_budget.reserve(["EXTRA"], tmp_path, 101, 3) == []
    assert len(screen_budget.snapshot(tmp_path, 101, 3)["reserved"]) == 3
    with pytest.raises(ValueError):
        screen_budget.snapshot(tmp_path, 102, 3)


def test_fresh_coverage_counts_only_this_epoch_screen_not_history_or_repeats(tmp_path, monkeypatch):
    from nanobody_opt_v2 import screen_budget
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    monkeypatch.setattr(race, "_already_scored", lambda *a, **kw: pytest.fail("cache is not budget"))
    current = tmp_path / "iteration_01/race/screen/tranche_01/exact_all.json"
    current.parent.mkdir(parents=True)
    current.write_text("{}")
    def observations(paths, target):
        assert paths == [current]
        return {"A": [{}]}
    monkeypatch.setattr(rank, "_observations", observations)
    screen_budget.reserve(["A", "B"], tmp_path, 101, 3)
    pool = {"candidates": [{"sequence": s} for s in ["A", "B", "C"]],
            "screen_budget_policy": screen_budget.POLICY, "exact_admission_limit": 3}
    report = producer._complete_pool_coverage(pool=pool, producer_root=tmp_path,
        database=tmp_path / "db", target="TEST", epoch=101, until_epoch=101)
    assert (report["screened"], report["attempted"], report["pending"]) == (1, 2, 1)
    assert report["uncompleted_dispatches"] == 1
    assert not report["complete"]
    screen_budget.reserve(["C"], tmp_path, 101, 3)
    report = producer._complete_pool_coverage(pool=pool, producer_root=tmp_path,
        database=tmp_path / "db", target="TEST", epoch=101, until_epoch=101)
    assert report["budget_exhausted"] and report["complete"]
    assert report["screened"] == 1  # never claim failed dispatches succeeded


def test_live_race_updates_epoch_coverage_before_returning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from nanobody_opt_v2 import screen_budget
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    pool = tmp_path / "full_scan_pool.json"
    producer._atomic_json(pool, {
        "target": "TEST", "origin_epoch": 101,
        "screen_budget_policy": screen_budget.POLICY, "exact_admission_limit": 3,
        "candidates": [{"sequence": s} for s in ["A", "B", "C"]],
    })
    parser = race._parser()
    args = parser.parse_args([
        "--layer2", str(pool), "--archive", "archive.csv", "--config", "config.yaml",
        "--target", "TEST", "--epoch", "101", "--gpus", "0",
        "--output-dir", str(tmp_path / "iteration_01/race"),
        "--pooled-exact-root", str(tmp_path), "--fresh-screen-budget", "3",
        "--screen-all-candidates", "--screen-tranche-size", "2", "--pipeline-depth", "0",
    ])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(race, "_parser", lambda: parser)
    monkeypatch.setattr(race, "_prior_evaluation_sequences", lambda *a, **kw: set())
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: True)
    monkeypatch.setattr(race, "_remaining_blocks", lambda *a: 300)
    monkeypatch.setattr(rank, "_observations", lambda paths, target: {"A": [{}]} if paths else {})
    snapshots = []
    def start(**kw):
        snapshots.append(json.loads((tmp_path / "full_scan_coverage.json").read_text()))
        completed = kw["output_dir"] / "exact_all.json"
        def wait():
            producer._atomic_json(completed, {})
        return SimpleNamespace(wait=wait, completed=completed, tranche=kw["tranche"])
    monkeypatch.setattr(race, "_start_screen", start)
    class StopAfterCoverage(Exception):
        pass
    def rank_after_coverage(*a, **kw):
        snapshots.append(json.loads((tmp_path / "full_scan_coverage.json").read_text()))
        raise StopAfterCoverage
    monkeypatch.setattr(race.subprocess, "run", rank_after_coverage)
    with pytest.raises(StopAfterCoverage):
        race.main()
    assert [(r["attempted"], r["screened"], r["uncompleted_dispatches"])
            for r in snapshots] == [(2, 0, 2), (2, 1, 1)]
    assert all(r["pending"] == 1 and not r["complete"] for r in snapshots)


@pytest.mark.parametrize("failure", ["wrong_epoch", "wrong_target", "write_failed"])
def test_coverage_refresh_failure_does_not_abort_scoring(tmp_path, monkeypatch, capsys, failure):
    from types import SimpleNamespace
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    producer._atomic_json(tmp_path / "full_scan_pool.json", {
        "target": "OTHER" if failure == "wrong_target" else "TEST",
        "origin_epoch": 100 if failure == "wrong_epoch" else 101,
        "screen_budget_policy": "unseen-candidates-v2",
    })
    called = []
    def write(**kw):
        called.append(kw)
        raise OSError("simulated unavailable report destination")
    monkeypatch.setattr(producer, "_complete_pool_coverage", write)
    race._refresh_full_scan_coverage(SimpleNamespace(
        fresh_screen_budget=3, pooled_exact_root=tmp_path, epoch=101,
        target="TEST", mutation_database=tmp_path / "db"))
    assert bool(called) == (failure == "write_failed")
    assert "EXACT_COVERAGE_REFRESH_FAILED epoch=101" in capsys.readouterr().out
    assert not (tmp_path / "full_scan_coverage.json").exists()


def test_unseen_admission_replaces_old_candidates_before_budget_cut(tmp_path, monkeypatch):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.complete_screen_until_epoch = 72
    args.complete_screen_pool = tmp_path / "pool.json"
    producer._atomic_json(args.complete_screen_pool, {
        "target": "TEST", "origin_epoch": 72, "exact_admission_limit": 3,
        "screen_budget_policy": "unseen-candidates-v2",
        "candidates": [{"sequence": s, "parent_multipass_mean": i}
                       for i,s in enumerate(["OLD1","OLD2","NEW1","NEW2","NEW3","NEW4"])],
    })
    monkeypatch.setattr(producer, "_prepare_epoch", lambda **kw: (tmp_path / "cfg", tmp_path / "archive", "TEST"))
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: {"OLD1","OLD2","NEW1","NEW2","NEW3","NEW4"})
    monkeypatch.setattr(race, "_prior_evaluation_sequences", lambda *a, **kw: {"OLD1","OLD2"})
    monkeypatch.setattr(producer, "_complete_pool_coverage", lambda **kw: {"complete": False})
    calls=[]
    monkeypatch.setattr(producer, "_run_race", lambda **kw: calls.append(kw))
    producer._run_complete_screen_pool(args, state(72))
    pool=json.loads(calls[0]["layer2"].read_text())
    assert pool["sequences"] == ["NEW1","NEW2","NEW3"]
    assert pool["previously_evaluated_excluded"] == 2
    assert pool["deferred"] == 1
    assert calls[0]["args"].fresh_screen_budget == 3


def test_prior_evaluation_query_includes_public_measurements_and_fails_closed(tmp_path, monkeypatch):
    import sqlite3
    from nanobody_opt import mutation_network
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    path=tmp_path / "db"
    c=sqlite3.connect(path)
    c.executescript("CREATE TABLE nodes (id INTEGER,sequence TEXT,target TEXT); CREATE TABLE observations (node_id INTEGER, source TEXT);")
    c.executemany("INSERT INTO nodes VALUES (?,?,?)", [(1,"A","TEST"),(2,"B","TEST"),(3,"C","TEST"),(4,"D","OTHER")])
    c.executemany("INSERT INTO observations VALUES (?,?)", [(1,"local_exact_raw"),(2,"public_multipass_rivals"),(3,"proxy"),(4,"local_exact_raw")])
    c.commit();c.close()
    monkeypatch.setattr(mutation_network, "_connect", lambda p: sqlite3.connect(p))
    monkeypatch.setattr(mutation_network, "_is_local", lambda c: True)
    assert race._prior_evaluation_sequences(path,"TEST",sequences=["A","B","C","D"]) == {"A","B"}
    assert race._prior_evaluation_sequences(path,"TEST") == {"A","B"}
    with pytest.raises(RuntimeError, match="lookup unavailable"):
        race._prior_evaluation_sequences(tmp_path / "missing","TEST")
