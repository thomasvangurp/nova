"""Deferred TNP must cover preparation without weakening publication gates."""
import json

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
