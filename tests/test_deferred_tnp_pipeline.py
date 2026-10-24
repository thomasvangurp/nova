"""Deferred TNP must cover preparation without weakening publication gates."""
import json
from pathlib import Path

import pytest

from nanobody_opt_v2 import run_live_producer as producer
from sn68_source import import_shipping


@pytest.mark.parametrize("deferred", [0, 1])
def test_competitor_seed_filter_honors_deferred_mode(tmp_path, monkeypatch, deferred):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.budget_mode = "fixed"
    args.layer2_igblast_only = deferred
    root = tmp_path / "live/epoch100/producer"
    iteration = root / "iteration_01"
    iteration.mkdir(parents=True)
    for name in ("winner_priors.json", "competitor_parents.json"):
        (root / name).write_text("{}")
    producer._stamp_artifact_policy(root / "competitor_parents.json")
    (root / "elite_parents.json").write_text(json.dumps({
        "selection_policy": "agreedfix-2026-10-20-complete+developability-hoist",
    }))
    for name in ("hf_archive.csv", "top50_index.json"):
        (iteration / name).write_text("{}")
    monkeypatch.setattr(producer, "_prepare_epoch", lambda **kw: (
        root / "config.yaml", root / "archive.csv", "TEST",
    ))
    monkeypatch.setattr(producer, "_budget_plan", lambda *a: {"replication_reserve_blocks": 1})
    monkeypatch.setattr(producer, "_adopt_staged_prep", lambda **kw: False)
    commands = []

    class ReachedSeedFilter(Exception):
        pass

    def run(command):
        commands.append(command)
        raise ReachedSeedFilter

    monkeypatch.setattr(producer, "_run", run)
    with pytest.raises(ReachedSeedFilter):
        producer._run_iteration(args=args, iteration=1, state={
            "epoch": 100, "block": 36100, "epoch_length": 361, "nanobody_target": "TEST",
        })
    assert commands[0][commands[0].index("--input") + 1] == str(root / "competitor_parents.json")
    assert ("--igblast-only" in commands[0]) == bool(deferred)


@pytest.mark.parametrize("deferred", [0, 1])
def test_fallback_filter_honors_deferred_mode(tmp_path, monkeypatch, deferred):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.layer2_igblast_only = deferred
    (tmp_path / "artifacts").mkdir()
    parent = {
        "sequence": "A" * 120, "parent_cdr3": "test-family",
        "parent_validator_duplicate": False, "parent_top50_identity_in_basin": False,
        "layer2_passed": True, "parent_cdr_developability_in_range": True,
    }
    (tmp_path / "artifacts/agreedfix_historical_parents.json").write_text(json.dumps({
        "target": "TEST", "candidates": [parent],
    }))
    destination = tmp_path / "pool.json"
    destination.write_text('{"candidates": []}')
    monkeypatch.setattr(selector, "_validator_parent_pool", lambda pool, **kw: pool)
    monkeypatch.setattr(selector, "_developability_parent_pool", lambda pool, **kw: pool)
    commands = []

    def run(command):
        commands.append(command)
        source = json.loads(open(command[command.index("--input") + 1]).read())
        producer._atomic_json(tmp_path / "fallback_0.layer2.json", source)

    monkeypatch.setattr(producer, "_run", run)
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: {
        c["sequence"] for c in json.loads(kw["input_path"].read_text())["candidates"]
    })
    producer._historical_child_fallback(
        args=args, target="TEST", config=tmp_path / "config.yaml",
        hf_archive=tmp_path / "archive.csv", iteration_root=tmp_path,
        destination=destination, floor=1,
    )
    assert ("--igblast-only" in commands[0]) == bool(deferred)
    assert json.loads(destination.read_text())["candidates"]


def test_race_preserves_explicit_tranche_budget(monkeypatch):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    parser = race._parser()
    args = parser.parse_args([
        "--layer2", "unused.json", "--archive", "unused.csv", "--config", "unused.yaml",
        "--target", "TEST", "--epoch", "100", "--output-dir", "unused",
        "--gpus", "0", "--maximum-screen-tranches", "1",
    ])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(race, "_parser", lambda: parser)

    class ReachedCandidates(Exception):
        pass

    def records(path):
        assert args.maximum_screen_tranches == 1
        raise ReachedCandidates

    monkeypatch.setattr(race, "_candidate_records", records)
    with pytest.raises(ReachedCandidates):
        race.main()


def test_multipass_batching_preserves_independent_fold_ranks(monkeypatch, tmp_path):
    import pandas as pd
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    network = import_shipping("nanobody_opt.mutation_network", monkeypatch)
    field = pd.DataFrame({m: [1., 2., 3.] for m in rank.METRICS})
    folds = {"A": [{m: v for m in rank.METRICS} for v in [1., 4.]],
             "B": [{m: 2. for m in rank.METRICS}]}
    rows = [(seq, json.dumps(fold)) for seq, fs in folds.items() for fold in fs]

    class Connection:
        def execute(self, *a):
            return self
        def fetchall(self):
            return rows
        def close(self):
            pass

    monkeypatch.setattr(network, "_connect", lambda *a: Connection())
    monkeypatch.setattr(network, "_is_local", lambda *a: True)
    monkeypatch.setattr(rank, "fetch_recent_fields_from_db", lambda *a, **kw: [(1, field)])
    monkeypatch.setattr(rank, "pool_recent_fields", lambda *a: field)
    original = rank.rank_metrics_against_field
    expected = {seq: sum(original({str(i): f for i, f in enumerate(fs)}, field).values()) / len(fs)
                for seq, fs in folds.items()}
    sizes = []

    def ranked(batch, background):
        sizes.append(len(batch))
        return original(batch, background)

    monkeypatch.setattr(rank, "rank_metrics_against_field", ranked)
    assert race._multipass_rank_map(tmp_path / "observations.sqlite", "TEST", 100) == expected
    assert sizes == [3]


def test_history_reads_only_requested_candidates_across_batches(monkeypatch):
    import sqlite3
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    network = import_shipping("nanobody_opt.mutation_network", monkeypatch)
    monkeypatch.setattr(network, "_is_local", lambda *a: True)
    connection = sqlite3.connect(":memory:")
    connection.executescript("""
        CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT);
        CREATE INDEX nodes_sequence ON nodes(target, sequence);
        CREATE TABLE observations (node_id INTEGER, observation_id TEXT, source TEXT, metrics_json TEXT);
        CREATE INDEX observations_node ON observations(node_id);
    """)
    for i in range(503):
        connection.execute("INSERT INTO nodes VALUES (?, 'TEST', ?)", (i, str(i)))
        connection.execute("INSERT INTO observations VALUES (?, ?, 'live_exact_screen', '{}')", (i, str(i)))
    wanted = [str(i) for i in range(501)]
    assert dict(race._exact_history_rows(connection, "TEST", sequences=wanted)) == {s: 1 for s in wanted}
    assert dict(race._exact_history_rows(connection, "TEST", sequences=wanted, metrics=True)) == {s: "{}" for s in wanted}
    assert race._exact_history_rows(connection, "TEST", sequences=[]) == []
    connection.close()


def test_race_uses_historical_folds_only_for_current_exact_candidates(monkeypatch, tmp_path):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    command = race._rank_command(
        exact=[tmp_path / "exact.json"], archive=tmp_path / "archive.csv",
        config=tmp_path / "config.yaml", target="TEST", epoch=100,
        output=tmp_path / "ranking.json", ledger=None, strict=False,
        observation_history=tmp_path / "history.sqlite",
    )
    assert "--current-exact-candidates-only" in command
    assert command[command.index("--minimum-observations") + 1] == "1"
    assert "--maximum-public-edits" not in command
    assert "--liability-quantile" not in command


def test_publisher_scope_uses_only_fresh_current_target_proofs(monkeypatch, tmp_path):
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    root = tmp_path / "epoch100"
    for number, target, fresh in [(1, "TEST", True), (2, "OTHER", True), (3, "TEST", False)]:
        path = root / f"producer/iteration_{number:02d}/layer2_dedup.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"target": target, "sequences": [f"SEQUENCE{number}"]}))
        if fresh:
            producer._stamp_artifact_policy(path)
    result = publisher._producer_candidate_scope(root, "TEST")
    assert result.read_text().splitlines() == ["SEQUENCE1"]
    assert publisher._producer_candidate_scope(root, "ABSENT") is None


def test_measured_parent_quality_precedes_source_label(monkeypatch):
    from collections import Counter
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    rows = [
        {"sequence": "WEAK", "parent_sequence": "PWEAK", "parent_source": "exact_shortlist",
         "parent_multipass_mean": 1805, "screen_priority": 1},
        {"sequence": "STRONG", "parent_sequence": "PSTRONG", "source": "single_edit",
         "parent_multipass_mean": 894, "screen_priority": 1000},
        {"sequence": "UNKNOWN", "parent_sequence": "PUNKNOWN", "parent_source": "exact_shortlist"},
    ]
    ordered = race._ordered_candidate_records(rows)
    assert [x["sequence"] for x in ordered] == ["STRONG", "WEAK", "UNKNOWN"]
    metadata = {x["sequence"]: x for x in rows}
    assert race._balanced_screen_tranche(
        ["WEAK", "UNKNOWN", "STRONG"], metadata, Counter(), 1,
    ) == ["STRONG"]


def test_parent_annotation_uses_measured_values_not_lane_indices(monkeypatch, tmp_path):
    competitor = import_shipping("nanobody_opt.competitor_lane", monkeypatch)
    args = producer._parser().parse_args([])
    (tmp_path / "elite_parents.json").write_text(json.dumps({"candidates": [
        {"sequence": "PARENT", "parent_multipass_mean": 882.4},
    ]}))
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"candidates": [
        {"sequence": "CHILD", "parent_sequence": "PARENT"},
        {"sequence": "UNKNOWN_CHILD", "parent_sequence": "UNKNOWN"},
    ]}))
    monkeypatch.setattr(competitor, "multipass_scores", lambda *a, **kw: {"PARENT": 894.35})
    producer._annotate_parent_screen_scores(
        args=args, epoch=100, target="P20809", producer_root=tmp_path, pool=pool,
    )
    rows = json.loads(pool.read_text())["candidates"]
    assert rows[0]["parent_multipass_mean"] == 894.35
    assert rows[0]["parent_score_source"] == "live_public_multipass"
    assert "parent_multipass_mean" not in rows[1]


@pytest.mark.parametrize("bounded", [False, True])
def test_multiedit_seeds_do_not_promote_unmeasured_competitor_children(tmp_path, monkeypatch, bounded):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.budget_mode = "fixed"
    if bounded:
        args.generation_proposal_budget = 8000
    root = tmp_path / "live/epoch100/producer"
    iteration = root / "iteration_01"
    iteration.mkdir(parents=True)
    elite = [{"sequence": "ELITE" + str(i), "selection_score": 900 + i} for i in range(8)]
    children = [{"sequence": "UNMEASURED" + str(i)} for i in range(8)]
    for name, rows in [("elite_parents.json", elite), ("elite_parents_layer2.json", elite),
                       ("competitor_parents.json", children), ("competitor_parents_layer2.json", children)]:
        (root / name).write_text(json.dumps({
            "selection_policy": "agreedfix-2026-10-20-complete+developability-hoist",
            "candidates": rows,
        }))
    (root / "winner_priors.json").write_text("{}")
    producer._stamp_artifact_policy(root / "competitor_parents.json")
    producer._stamp_artifact_policy(root / "competitor_parents_layer2.json")
    for name in ("hf_archive.csv", "top50_index.json"):
        (iteration / name).write_text("{}")
    monkeypatch.setattr(producer, "_prepare_epoch", lambda **kw: (
        root / "config.yaml", root / "archive.csv", "TEST",
    ))
    monkeypatch.setattr(producer, "_budget_plan", lambda *a: {"replication_reserve_blocks": 1})
    monkeypatch.setattr(producer, "_adopt_staged_prep", lambda **kw: False)

    class ReachedGeneration(Exception):
        pass

    def run(command):
        assert "nanobody_opt.local_neighborhood" in command
        assert ("--bounded-generation" in command) == bounded
        expected = producer._generation_quotas(args, 8000)["children"] if bounded else 8000
        assert int(command[command.index("--proposal-limit") + 1]) == expected
        seeds = json.loads(open(command[command.index("--layer2-seeds") + 1]).read())
        assert seeds["candidates"] == elite
        raise ReachedGeneration

    monkeypatch.setattr(producer, "_run", run)
    with pytest.raises(ReachedGeneration):
        producer._run_iteration(args=args, iteration=1, state={
            "epoch": 100, "block": 36100, "epoch_length": 361, "nanobody_target": "TEST",
        })


def test_budget_learns_the_rounds_the_shared_scorer_actually_emits():
    from nanobody_opt_v2.compute_budget import TimingModel
    rows = [{"phase": "replication_round_shared", "candidates": n,
             "gpu_count": 8, "seconds": 2 + 3 * (n // 8)}
            for _ in range(10) for n in [16, 40]]
    model = TimingModel(45, 4.8, 30).fit_from_measurements(rows, 8)
    assert model.load_seconds == pytest.approx(2.3)
    assert model.gpu_seconds_per_candidate == pytest.approx(3.45)


def test_priority_parent_lane_cannot_bypass_measured_multipass_order(monkeypatch):
    lane = import_shipping("nanobody_opt.competitor_lane", monkeypatch)
    assert lane._merge_priority(["WEAK", "UNMEASURED", "STRONG"], {
        "WEAK": 1805, "STRONG": 684.15,
    }) == ["STRONG", "WEAK"]


def _comparison_fixture(monkeypatch, tmp_path):
    from types import SimpleNamespace
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    root = tmp_path / "epoch100"
    root.mkdir()
    archive = root / "archive.csv"
    archive.write_text("sequence\nHISTORICAL\n")
    config = root / "config.yaml"
    config.write_text("current rules")
    exclusion = root / "excluded.txt"
    exclusion.write_text("RETIRED\n")
    for name in ("check_validator_novelty.py", "filter_layer2.py"):
        (tmp_path / name).write_text("current implementation")
    pool = {"STORED", "NEW", "INVALID_NOVELTY", "INVALID_LAYER2", "HISTORICAL", "RETIRED"}
    monkeypatch.setattr(publisher, "_stored_comparison_sequences", lambda *a: set(pool))
    commands = []
    def run(command):
        commands.append(command)
        data = json.loads(Path(command[command.index("--input") + 1]).read_text())
        seqs = set(data["sequences"])
        output = Path(command[command.index("--output") + 1])
        if "check-validator-novelty" in command:
            passed = seqs - {"INVALID_NOVELTY"}
            publisher._atomic_json(output, {"input_count": len(seqs), "sequences": sorted(passed)})
        else:
            assert "filter-layer2" in command
            assert "--igblast-only" not in command
            publisher._atomic_json(output, {"input_count": len(seqs), "candidates": [
                {"sequence": s, "layer2_passed": True} for s in sorted(seqs - {"INVALID_LAYER2"})]})
    monkeypatch.setattr(publisher, "_run", run)
    kwargs = dict(args=SimpleNamespace(nanobody_opt_root=tmp_path, pixi=Path("pixi"),
                     layer2_concurrency=8, layer2_cache_dir=tmp_path / "cache"),
                  epoch_root=root, target="TEST", database=tmp_path / "history.sqlite",
                  archive=archive, config=config, exclusion=exclusion,
                  pixi_manifest=tmp_path / "pixi.toml")
    return publisher, kwargs, pool, commands


def test_publisher_prechecks_stored_and_new_before_merit_cutoff(monkeypatch, tmp_path):
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    scope = publisher._prechecked_comparison_scope(**kwargs)
    assert set(scope.read_text().splitlines()) == {"STORED", "NEW"}
    assert len(commands) == 2
    # Both origins pass the exact same full current gates; neither invalid
    # high-merit cached entry can consume a shortlist position.
    precheck_input = json.loads((kwargs["epoch_root"] / "comparison_precheck_input.json").read_text())
    assert set(precheck_input["sequences"]) == pool - {"HISTORICAL", "RETIRED"}
    assert "--summary-only" in commands[0]
    command = publisher._rank_command(
        bgen_python=Path("python"), nanobody_opt_root=tmp_path, exact_inputs=[],
        archive=kwargs["archive"], target="TEST", output=tmp_path / "ranking.json",
        exclusion=kwargs["exclusion"], epoch=100, shortlist=1, config=kwargs["config"],
        strict=False, observation_history=kwargs["database"], only_sequences=scope)
    assert command[command.index("--only-sequences") + 1] == str(scope)
    assert "--observation-history" in command
    assert "--current-exact-candidates-only" not in command
    assert int(command[command.index("--minimum-observations") + 1]) == 20
    for cmd in [*commands, command]:
        assert "--scorer" not in cmd and "--gpus" not in cmd
    from nanobody_opt_v2.screen_budget import pending
    assert pending([{"sequence": "NEW"}], tmp_path, 100, 4800) == [{"sequence": "NEW"}]


def test_comparison_precheck_reuses_proofs_and_checks_new_arrivals(monkeypatch, tmp_path):
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    publisher._prechecked_comparison_scope(**kwargs)
    commands.clear()
    publisher._prechecked_comparison_scope(**kwargs)
    assert not commands
    pool.add("LATER_CONFIRMED")
    scope = publisher._prechecked_comparison_scope(**kwargs)
    assert len(commands) == 2
    inputs = json.loads((kwargs["epoch_root"] / "comparison_precheck_input.json").read_text())
    assert inputs["sequences"] == ["LATER_CONFIRMED"]
    assert set(scope.read_text().splitlines()) == {"STORED", "NEW", "LATER_CONFIRMED"}
    kwargs["exclusion"].write_text("RETIRED\nSTORED\n")
    scope = publisher._prechecked_comparison_scope(**kwargs)
    assert "STORED" not in scope.read_text().splitlines()


@pytest.mark.parametrize("change", ["archive", "config", "implementation", "epoch", "target"])
def test_comparison_proof_invalidates_when_current_rules_change(monkeypatch, tmp_path, change):
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    publisher._prechecked_comparison_scope(**kwargs)
    commands.clear()
    if change in ("archive", "config"):
        with kwargs[change].open("a") as handle:
            handle.write("CHANGED\n")
    elif change == "implementation":
        (tmp_path / "check_validator_novelty.py").write_text("changed implementation")
    elif change == "epoch":
        old = kwargs["epoch_root"]
        new = tmp_path / "epoch101"
        new.mkdir()
        (new / "comparison_precheck.json").write_bytes((old / "comparison_precheck.json").read_bytes())
        kwargs["epoch_root"] = new
    else:
        kwargs["target"] = "OTHER"
    publisher._prechecked_comparison_scope(**kwargs)
    assert len(commands) == 2
    inputs = json.loads((kwargs["epoch_root"] / "comparison_precheck_input.json").read_text())
    assert len(inputs["sequences"]) == 4


def test_empty_comparison_precheck_cannot_reuse_stale_scope(monkeypatch, tmp_path):
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    scope = publisher._prechecked_comparison_scope(**kwargs)
    pool.clear()
    pool.add("INVALID_NOVELTY")
    with pytest.raises(RuntimeError, match="no comparison candidate"):
        publisher._prechecked_comparison_scope(**kwargs)
    assert scope.read_text() == ""


def test_failed_comparison_precheck_does_not_cache_success(monkeypatch, tmp_path):
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    def fail(command):
        raise RuntimeError("gate interrupted")
    monkeypatch.setattr(publisher, "_run", fail)
    with pytest.raises(RuntimeError, match="gate interrupted"):
        publisher._prechecked_comparison_scope(**kwargs)
    assert not (kwargs["epoch_root"] / "comparison_precheck.json").exists()


def test_cold_prechecks_do_not_block_cached_fleet_or_truncate_bank(monkeypatch, tmp_path):
    from nanobody_opt.filter_layer2 import _developability_cache_digest
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    pool.update(f"COLD_{i}" for i in range(90))
    cache = kwargs["args"].layer2_cache_dir
    cache.mkdir()
    (cache / (_developability_cache_digest("STORED") + ".json")).write_text(json.dumps({
        "sequence": "STORED", "developability": {"passed": True, "flags": {}},
    }))
    scope = publisher._prechecked_comparison_scope(**kwargs)
    assert scope.read_text().splitlines() == ["STORED"]
    first = json.loads((kwargs["epoch_root"] / "comparison_precheck.json").read_text())
    assert first["pending_count"] > 32
    commands.clear()
    # A retry after a failed publication must deliver the checked fleet before
    # making it wait for cold measurements again.
    assert publisher._prechecked_comparison_scope(**kwargs, ready_artifact=False) == scope
    assert commands == []
    for _ in range(4):
        commands.clear()
        scope = publisher._prechecked_comparison_scope(**kwargs)
        assert len(commands) <= 1  # same-archive novelty proof is reused
    assert set(scope.read_text().splitlines()) == pool - {
        "HISTORICAL", "RETIRED", "INVALID_NOVELTY", "INVALID_LAYER2",
    }
    assert json.loads((kwargs["epoch_root"] / "comparison_precheck.json").read_text())["pending_count"] == 0


def test_interrupted_layer2_keeps_novelty_proof_but_not_eligibility(monkeypatch, tmp_path):
    publisher, kwargs, pool, commands = _comparison_fixture(monkeypatch, tmp_path)
    run = publisher._run
    def interrupted(command):
        if "filter-layer2" in command:
            raise RuntimeError("interrupted Layer-2")
        run(command)
    monkeypatch.setattr(publisher, "_run", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        publisher._prechecked_comparison_scope(**kwargs)
    proof = json.loads((kwargs["epoch_root"] / "comparison_precheck.json").read_text())
    assert proof["eligible_sequences"] == []
    assert "STORED" not in proof["checked_sequences"]
    monkeypatch.setattr(publisher, "_run", run)
    commands.clear()
    assert set(publisher._prechecked_comparison_scope(**kwargs).read_text().splitlines()) == {"STORED", "NEW"}
    assert len(commands) == 1 and "filter-layer2" in commands[0]


@pytest.mark.parametrize("aggregate", ["absent", "stale", "current"])
def test_stored_comparison_uses_only_sealed_own_evidence(monkeypatch, tmp_path, aggregate):
    import sqlite3
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    network = import_shipping("nanobody_opt.mutation_network", monkeypatch)
    database = tmp_path / "history.sqlite"
    c = sqlite3.connect(database)
    c.executescript("""
        CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT);
        CREATE TABLE observations (node_id INTEGER, observation_id TEXT, source TEXT, metrics_json TEXT);
        CREATE TABLE exact_run_observations (observation_id TEXT, target TEXT, run_id TEXT);
        CREATE TABLE exact_runs (target TEXT, run_id TEXT, complete INTEGER);
    """)
    for i, (seq, target, count, source, complete) in enumerate([
        ("READY", "TEST", 20, "local_exact_raw", 1),
        ("SHALLOW", "TEST", 19, "local_exact_raw", 1),
        ("RIVAL", "TEST", 20, "public_validator_api", 1),
        ("UNSEALED", "TEST", 20, "local_exact_raw", 0),
        ("OTHER", "OTHER", 20, "local_exact_raw", 1),
    ]):
        c.execute("INSERT INTO nodes VALUES (?, ?, ?)", (i, target, seq))
        c.execute("INSERT INTO exact_runs VALUES (?, ?, ?)", (target, str(i), complete))
        for j in range(count):
            oid = f"{i}-{j}"
            c.execute("INSERT INTO observations VALUES (?, ?, ?, '{}')", (i, oid, source))
            c.execute("INSERT INTO exact_run_observations VALUES (?, ?, ?)", (oid, target, str(i)))
    if aggregate != "absent":
        c.executescript("""
            CREATE TABLE exact_observation_aggregate (target TEXT, sequence TEXT, count_value INTEGER);
            CREATE TABLE exact_observation_meta (key TEXT, value INTEGER);
            INSERT INTO exact_observation_meta VALUES ('valid', 1);
        """)
        watermark = c.execute("SELECT MAX(rowid) FROM exact_run_observations").fetchone()[0]
        c.execute("INSERT INTO exact_observation_meta VALUES ('source_max_rowid', ?)",
                  (watermark if aggregate == "current" else watermark - 1,))
        c.execute("INSERT INTO exact_observation_aggregate VALUES ('TEST', ?, 20)",
                  ("READY" if aggregate == "current" else "WRONG",))
    c.commit()
    c.close()
    monkeypatch.setattr(network, "_connect", lambda *a: sqlite3.connect(database))
    monkeypatch.setattr(network, "_is_local", lambda *a: True)
    assert publisher._stored_comparison_sequences(database, "TEST", 20) == {"READY"}


def test_scope_change_invalidates_old_publisher_noop_state(monkeypatch, tmp_path):
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    common = dict(epoch=100, target="TEST", exact_fingerprint="same",
                  ledger_fingerprint="same", fleet_fingerprint="same")
    assert not publisher._publisher_state_matches({"schema_version": 3, **common}, **common)
    path = tmp_path / "state.json"
    publisher._write_publisher_state(path, **common, strict_fingerprint="same")
    assert publisher._publisher_state_matches(json.loads(path.read_text()), **common)


@pytest.mark.parametrize("artifact,expected", [
    (None, True),
    ({"epoch": 99, "submissions": [{"sequence": "RETIRED"}]}, True),
    ({"epoch": 100, "submissions": [{"sequence": "READY"}]}, True),
    ({"epoch": 100, "submissions": [{"sequence": "INELIGIBLE"}]}, False),
])
def test_current_artifact_retention_uses_real_sequence_list_contract(monkeypatch, tmp_path, artifact, expected):
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    scope = tmp_path / "scope.txt"
    scope.write_text("READY\n")
    assert publisher._artifact_passes_scope(artifact, 100, scope) is expected


@pytest.mark.parametrize("interrupted", [False, True])
def test_database_publication_delivers_checked_fleet_before_cold_backlog(
    monkeypatch, tmp_path, interrupted,
):
    """Exercise the complete orchestrator with real artifact/helper contracts.

    External gates/ranking are simulated, but scope, retry, fleet construction,
    artifact retention, and state writes run together as they do on the host.
    """
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    from test_publish_current_nanobody import _Host, EPOCH
    from nanobody_opt.filter_layer2 import _developability_cache_digest

    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    args = host.args(strict_qualification=False)
    args.exact = []
    args.observation_database = tmp_path / "observations.sqlite"
    cold = {f"COLD{i:03d}" for i in range(80)}
    bank = set(host.candidate_sequences) | cold
    rejected = host.candidate_sequences[0]
    monkeypatch.setattr(publisher, "_publication_exact_inputs", lambda *a: ())
    monkeypatch.setattr(publisher, "_database_exact_fingerprints", lambda *a: ("bank", "bank", 1))
    monkeypatch.setattr(publisher, "_stored_comparison_sequences", lambda *a: set(bank))
    args.layer2_cache_dir.mkdir()
    for sequence in host.candidate_sequences:
        (args.layer2_cache_dir / (_developability_cache_digest(sequence) + ".json")).write_text(
            json.dumps({"sequence": sequence, "developability": {"passed": True, "flags": {}}})
        )
    comparison_batches = []
    fail_rank_once = interrupted

    def run(command):
        nonlocal fail_rank_once
        def value(flag):
            return command[command.index(flag) + 1]

        if "check-validator-novelty" in command or "filter-layer2" in command:
            host.commands.append(list(command))
            source = json.loads(Path(value("--input")).read_text())
            sequences = set(source.get("sequences", [])) or {
                row["sequence"] for row in source.get("candidates", [])
            }
            if "check-validator-novelty" in command:
                result = {"input_count": len(sequences), "sequences": sorted(sequences - {rejected})}
            else:
                if Path(value("--input")).name == "comparison_precheck_layer2_input.json":
                    comparison_batches.append(sequences)
                result = {"input_count": len(sequences), "candidates": [
                    {"sequence": sequence, "layer2_passed": True} for sequence in sorted(sequences)
                ]}
            publisher._atomic_json(Path(value("--output")), result)
        elif any(str(part).endswith("rank_exact.py") for part in command):
            if fail_rank_once:
                fail_rank_once = False
                raise RuntimeError("publication interrupted after precheck")
            allowed = set(Path(value("--only-sequences")).read_text().splitlines())
            assert rejected not in allowed
            host.run(command)
            output = Path(value("--output"))
            result = json.loads(output.read_text())
            result["candidates"] = [row for row in result["candidates"] if row["sequence"] in allowed]
            publisher._atomic_json(output, result)
        else:
            host.run(command)

    monkeypatch.setattr(publisher, "_run", run)
    if interrupted:
        with pytest.raises(RuntimeError, match="publication interrupted"):
            publisher._publish_once(args)
        assert not host.artifact.exists()
    publisher._publish_once(args)
    artifact = json.loads(host.artifact.read_text())
    assert artifact["epoch"] == EPOCH
    assert artifact["submission_count"] == 3
    assert len(comparison_batches) == 1
    assert not comparison_batches[0] & cold
    assert rejected not in artifact["sequence_pool"]
    assert set(artifact["sequence_pool"]) <= bank - cold

    # The published fleet survives the real retention helper while the next
    # cycle advances cold eligibility without requiring any new GPU evidence.
    publisher._publish_once(args)
    assert len(comparison_batches) == 2
    assert len(comparison_batches[1]) == 32
    assert comparison_batches[1] <= cold
    assert json.loads(host.artifact.read_text())["submission_count"] == 3
    assert not any("fast_score" in " ".join(command) for command in host.commands)
