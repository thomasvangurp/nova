import json
from collections import Counter

from nanobody_opt import sequential_exact_race as race
from nanobody_opt.mutation_network import load_lineage_edges


def test_candidate_major_repeats_use_one_isolated_gpu_process_each(
    tmp_path, monkeypatch
):
    commands = []
    monkeypatch.setattr(race, "_run_parallel", lambda group: commands.extend(group))

    race._score_repeats(
        sequence="QWQLVESGG",
        repeat_numbers=list(range(2, 9)),
        root=tmp_path,
        config=tmp_path / "config.yaml",
        target="P20809",
        gpus=[str(index) for index in range(10)],
    )

    assert len(commands) == 7
    assert [command[command.index("--gpus") + 1] for command in commands] == [
        str(index) for index in range(7)
    ]
    assert all(command[-1] == "--capture-structures" for command in commands)


def test_candidate_loader_requires_layer2_pass_and_deduplicates(tmp_path):
    source = tmp_path / "layer2.json"
    source.write_text(json.dumps({"candidates": [
        {"sequence": " aaa ", "layer2_passed": True},
        {"sequence": "AAA", "layer2_passed": True},
        {"sequence": "BBB", "layer2_passed": False},
        {"sequence": "CCC"},
        {
            "sequence": "DDD",
            "proposal_type": "exhaustive_single_substitution",
        },
    ]}))

    assert race._candidate_sequences(source, 10) == ["DDD", "AAA", "CCC"]


def test_candidate_loader_preserves_graph_ranked_screen_priority(tmp_path):
    source = tmp_path / "layer2.json"
    source.write_text(json.dumps({"candidates": [
        {
            "sequence": "AAA",
            "screen_priority": 2,
            "proposal_type": "exhaustive_single_substitution",
        },
        {
            "sequence": "BBB",
            "screen_priority": 1,
            "proposal_type": "c_terminal_single_substitution",
        },
    ]}))

    assert race._candidate_sequences(source, 10) == ["BBB", "AAA"]


def test_screen_budget_is_split_into_publishable_twenty_candidate_tranches():
    sequences = [f"SEQ{index}" for index in range(320)]

    batches = race._batches(sequences, 20)

    assert len(batches) == 16
    assert all(len(batch) == 20 for batch in batches)
    assert [item for batch in batches for item in batch] == sequences


def test_next_screen_tranche_balances_actual_parent_counts():
    pending = ["AAAA", "AAAC", "CCCC", "CCCA"]
    metadata = {
        "AAAA": {"parent_sequence": "AAAT"},
        "AAAC": {"parent_sequence": "AAAT"},
        "CCCC": {"parent_sequence": "CCCT"},
        "CCCA": {"parent_sequence": "CCCT"},
    }

    selected = race._balanced_screen_tranche(
        pending,
        metadata,
        Counter({"AAAT": 2, "CCCT": 0}),
        3,
    )

    assert selected == ["CCCC", "CCCA", "AAAA"]


def test_next_screen_tranche_balances_basin_before_individual_parent():
    pending = ["A1", "A2", "A3", "B1", "B2"]
    metadata = {
        "A1": {"parent_sequence": "PA1", "parent_basin_core_id": "CORE_A", "parent_basin_rank": 1},
        "A2": {"parent_sequence": "PA2", "parent_basin_core_id": "CORE_A", "parent_basin_rank": 1},
        "A3": {"parent_sequence": "PA3", "parent_basin_core_id": "CORE_A", "parent_basin_rank": 1},
        "B1": {"parent_sequence": "PB1", "parent_basin_core_id": "CORE_B", "parent_basin_rank": 2},
        "B2": {"parent_sequence": "PB2", "parent_basin_core_id": "CORE_B", "parent_basin_rank": 2},
    }

    selected = race._balanced_screen_tranche(
        pending, metadata, Counter(), 4
    )

    assert selected == ["A1", "B1", "A2", "B2"]


def test_balanced_screen_preserves_priority_without_parent_metadata():
    assert race._balanced_screen_tranche(
        ["AAAA", "CCCC", "DDDD"], {}, Counter(), 2
    ) == ["AAAA", "CCCC"]


def test_completed_screen_tranche_resumes_its_original_candidates(tmp_path):
    source = tmp_path / "candidates.json"
    source.write_text(json.dumps({"sequences": ["AAAA", "AAAC"]}))
    completed = tmp_path / "exact_all.json"
    completed.write_text("{}")

    selected = race._resumable_screen_tranche(
        ["AAAA", "AAAC", "CCCC"],
        {
            "AAAA": {"parent_sequence": "AAAT"},
            "AAAC": {"parent_sequence": "AAAT"},
            "CCCC": {"parent_sequence": "CCCT"},
        },
        Counter(),
        2,
        source=source,
        completed=completed,
    )

    assert selected == ["AAAA", "AAAC"]


def test_screen_tranche_loads_each_gpu_pipeline_only_once():
    assert race._screen_execution_shape(20, 10) == (20, 2)
    assert race._screen_execution_shape(7, 10) == (7, 1)


def test_replacement_scorer_is_forwarded_to_wave_runner(tmp_path):
    scorer = tmp_path / "fast_scorer.py"
    command = race._repeat_command(
        input_path=tmp_path / "input.json",
        config=tmp_path / "config.yaml",
        target="P20809",
        output_dir=tmp_path / "repeat",
        gpu="3",
        scorer=scorer,
    )

    assert command[command.index("--scorer") + 1] == str(scorer)


def test_chain_reserve_prevents_new_gpu_wave(tmp_path):
    state = tmp_path / "chain.json"
    state.write_text(json.dumps({
        "epoch": 25034,
        "block": 25034 * 361 + 310,
        "epoch_length": 361,
    }))

    assert race._chain_allows_work(state, 25034, reserve_blocks=55) is False
    assert race._chain_allows_work(state, 25035, reserve_blocks=55) is False


def test_adaptive_challenger_only_spends_repeat_wave_inside_margin():
    assert race._challenger_can_overtake(
        screen_p90=105.0, incumbent_p90=85.0, margin=20.0
    )
    assert not race._challenger_can_overtake(
        screen_p90=105.1, incumbent_p90=85.0, margin=20.0
    )


def test_initial_repeat_requires_strict_candidate_or_one_explicit_fallback():
    candidates = {
        "WEAK": {"score_p90": 121.0},
        "STRONG": {"score_p90": 84.0},
    }

    assert race._initial_repeat_candidate(
        ["WEAK", "STRONG"],
        candidates,
        repeated=set(),
        maximum_screen_p90=90.0,
        allow_weak_fallback=False,
    ) == ("STRONG", False)
    assert race._initial_repeat_candidate(
        ["WEAK"],
        candidates,
        repeated=set(),
        maximum_screen_p90=90.0,
        allow_weak_fallback=False,
    ) is None
    assert race._initial_repeat_candidate(
        ["WEAK"],
        candidates,
        repeated=set(),
        maximum_screen_p90=90.0,
        allow_weak_fallback=True,
    ) == ("WEAK", True)


def test_terminal_one_shot_score_is_shrunk_toward_its_basin(tmp_path):
    ranking = tmp_path / "ranking.json"
    ranking.write_text(json.dumps({
        "sequences": ["LUCKY", "A2", "A3", "SOLID"],
        "candidates": [
            {"sequence": "LUCKY", "score_p90": 40.0},
            {"sequence": "A2", "score_p90": 160.0},
            {"sequence": "A3", "score_p90": 170.0},
            {"sequence": "SOLID", "score_p90": 80.0},
        ],
    }))
    metadata = {
        sequence: {
            "proposal_lane": "basin_terminal",
            "parent_basin_core_id": "CORE_A",
        }
        for sequence in ("LUCKY", "A2", "A3")
    }
    metadata["SOLID"] = {"proposal_lane": "core_cdr"}

    payload = race._apply_basin_shrinkage(
        ranking, metadata, prior_strength=2.0, minimum_samples=3
    )

    by_sequence = {item["sequence"]: item for item in payload["candidates"]}
    assert by_sequence["LUCKY"]["score_p90"] == 40.0
    assert by_sequence["LUCKY"]["basin_adjusted_score"] == 120.0
    assert payload["sequences"][0] == "SOLID"


def test_best_confirmed_p90_uses_best_candidate_not_file_order(tmp_path):
    ranking = tmp_path / "ranking.json"
    ranking.write_text(json.dumps({"candidates": [
        {"sequence": "AAA", "score_p90": 88.0},
        {"sequence": "BBB", "score_p90": 72.0},
    ]}))

    assert race._best_confirmed_p90(ranking) == 72.0


def test_provisional_tranche_ranking_skips_slow_pose_and_api_replay(tmp_path):
    command = race._rank_command(
        exact=[tmp_path / "exact.json"],
        archive=tmp_path / "archive.csv",
        config=tmp_path / "config.yaml",
        target="P20809",
        epoch=25036,
        output=tmp_path / "ranking.json",
        ledger=None,
        strict=False,
    )

    assert "--pose-config" not in command
    assert "--api-field-count" not in command
    assert command[command.index("--selection-order") + 1] == "p90"
    assert command[command.index("--limit") + 1] == "10000"


def test_screen_ranking_persists_true_proposal_lineage(tmp_path):
    ranking = tmp_path / "ranking.json"
    ranking.write_text(json.dumps({"candidates": [{
        "sequence": "AAAC",
        "selection_score": 54.0,
        "observation_count": 1,
        "score_median": 50.0,
        "score_p90": 54.0,
        "metrics": {},
    }]}))
    database = tmp_path / "mutation.sqlite"

    race._store_ranking(
        database,
        ranking,
        epoch=25045,
        target="P20809",
        source="live_exact_screen",
        candidate_metadata={"AAAC": {
            "parent_sequence": "AAAA",
            "proposal_type": "lineage_terminal_single_substitution",
            "mutations": "A4C",
            "proposal_lane": "lineage",
            "screen_priority": 17,
        }},
    )

    assert load_lineage_edges(database, target="P20809") == [{
        "parent_sequence": "AAAA",
        "sequence": "AAAC",
        "epoch": 25045,
        "source": "live_exact_screen",
        "proposal_type": "lineage_terminal_single_substitution",
        "mutations": ["A4C"],
        "metadata": {
            "proposal_lane": "lineage",
            "score_median": 50.0,
            "score_p90": 54.0,
            "screen_priority": 17,
        },
    }]


def test_predict_batch_is_pinned_so_a_bigger_tranche_cannot_change_scores():
    # Scores depend on which candidates share one predict call, so growing the
    # tranche for throughput must not silently regroup the predict batch.
    assert race._screen_execution_shape(200, 10) == (200, 20)
    assert race._screen_execution_shape(200, 10, 2) == (200, 2)
    assert race._screen_execution_shape(20, 10, 2) == (20, 2)


def test_already_scored_tranche_starts_no_process_and_resumes(tmp_path):
    from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL

    root = tmp_path / "tranche_01"
    root.mkdir()
    (root / "exact_all.json").write_text(
        json.dumps({"scorer_protocol": SCORER_PROTOCOL})
    )

    job = race._start_screen(
        index=1,
        tranche=["AAA"],
        source=root / "candidates.json",
        config=tmp_path / "config.yaml",
        target="P20809",
        output_dir=root,
        gpus=["0"],
    )

    assert job.wait() == root / "exact_all.json"
    assert job.scoring_seconds == 0.0


def test_prefetched_tranche_matches_the_serial_selection():
    # Dispatching one tranche ahead is only safe because tranche composition
    # never reads the ranking that the overlapped bookkeeping produces.
    metadata = {
        sequence: {"parent_basin_core_id": core, "parent_basin_rank": rank}
        for sequence, core, rank in (
            ("AAA", "core_1", 1),
            ("BBB", "core_2", 2),
            ("CCC", "core_1", 1),
            ("DDD", "core_3", 3),
        )
    }
    pending = ["AAA", "BBB", "CCC", "DDD"]
    counts: Counter[str] = Counter()

    serial = race._balanced_screen_tranche(pending, metadata, counts, 2)
    for sequence in serial:
        counts[str(metadata[sequence]["parent_basin_core_id"])] += 1
    remaining = [item for item in pending if item not in serial]
    serial_next = race._balanced_screen_tranche(remaining, metadata, counts, 2)

    enriched = {
        sequence: {
            **record,
            "basin_adjusted_score": 42.0,
            "basin_screen_median": 42.0,
        }
        for sequence, record in metadata.items()
    }
    prefetched = race._balanced_screen_tranche(remaining, enriched, counts, 2)

    assert prefetched == serial_next


def test_replication_rounds_reobserve_the_whole_shortlist(tmp_path, monkeypatch):
    ranking = tmp_path / "screen_ranking.json"
    ranking.write_text(json.dumps({"sequences": ["AAA", "BBB", "CCC", "DDD"]}))

    shortlist = race._replication_shortlist(
        ranking, exclude={"BBB"}, limit=2
    )
    assert shortlist == ["AAA", "CCC"]

    calls = []

    def _fake_screen(**kwargs):
        calls.append(kwargs)
        return kwargs["output_dir"] / "exact_all.json"

    monkeypatch.setattr(race, "_run_screen", _fake_screen)

    completed = race._run_replication_rounds(
        shortlist=shortlist,
        rounds=3,
        root=tmp_path,
        config=tmp_path / "config.yaml",
        target="P20809",
        gpus=["0", "1"],
        scorer=tmp_path / "scorer.py",
        chain_state=None,
        epoch=25056,
        reserve_blocks=55,
        predict_batch_size=2,
        timings=tmp_path / "timings.jsonl",
    )

    assert completed == 3
    # every round observes the entire shortlist, so each candidate gains the
    # same number of independent observations
    assert [call["candidate_limit"] for call in calls] == [2, 2, 2]
    assert [call["output_dir"].name for call in calls] == [
        "round_01", "round_02", "round_03"
    ]
    assert {call["predict_batch_size"] for call in calls} == {2}
    source = json.loads((tmp_path / "rounds" / "shortlist.json").read_text())
    assert source["sequences"] == ["AAA", "CCC"]
    records = [
        json.loads(line)
        for line in (tmp_path / "timings.jsonl").read_text().splitlines()
    ]
    assert [record["phase"] for record in records] == ["replication_round"] * 3


def test_replication_rounds_stop_at_the_commitment_reserve(tmp_path, monkeypatch):
    state = tmp_path / "chain.json"
    state.write_text(json.dumps({
        "epoch": 25056,
        "block": 25056 * 361 + 310,
        "epoch_length": 361,
    }))
    monkeypatch.setattr(race, "_run_screen", lambda **kwargs: tmp_path)

    completed = race._run_replication_rounds(
        shortlist=["AAA"],
        rounds=5,
        root=tmp_path,
        config=tmp_path / "config.yaml",
        target="P20809",
        gpus=["0"],
        scorer=tmp_path / "scorer.py",
        chain_state=state,
        epoch=25056,
        reserve_blocks=55,
        predict_batch_size=None,
        timings=tmp_path / "timings.jsonl",
    )

    assert completed == 0


def test_strict_gates_are_overridable_per_scorer(tmp_path):
    baseline = race.strict_rank_arguments()
    assert baseline[baseline.index("--maximum-p90-rank-sum") + 1] == "90"

    overridden = race.strict_rank_arguments({"maximum_p90_rank_sum": 104})
    assert overridden[overridden.index("--maximum-p90-rank-sum") + 1] == "104"
    # every other gate is preserved exactly
    assert len(overridden) == len(baseline)
    assert (
        overridden[overridden.index("--minimum-observations") + 1]
        == baseline[baseline.index("--minimum-observations") + 1]
    )


def test_unknown_strict_gate_is_refused():
    import pytest

    with pytest.raises(ValueError, match="unknown strict rank gate"):
        race.strict_rank_arguments({"maximum_p90_invented_gate": 1})


def test_strict_gate_overrides_load_from_either_shape(tmp_path):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"maximum_p90_rank_sum": 104}))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"gates": {"maximum_p90_rank_sum": 104}}))

    assert race.load_strict_gate_overrides(flat) == {"maximum_p90_rank_sum": 104}
    assert race.load_strict_gate_overrides(wrapped) == {
        "maximum_p90_rank_sum": 104
    }
    assert race.load_strict_gate_overrides(None) == {}


def test_screening_stops_early_enough_to_leave_room_for_replication(tmp_path):
    # 100 blocks remain before the commitment reserve would bite
    state = tmp_path / "chain.json"
    state.write_text(json.dumps({
        "epoch": 25060,
        "block": 25060 * 361 + 361 - 100,
        "epoch_length": 361,
    }))

    # screening alone may continue
    assert race._chain_allows_work(state, 25060, reserve_blocks=55) is True
    # but not once the replication rounds are reserved as well
    assert race._chain_allows_work(state, 25060, reserve_blocks=55 + 68) is False


def test_layer2_gate_keeps_only_candidates_with_full_evidence(tmp_path, monkeypatch):
    def _fake_run(command, check=False):
        output = command[command.index("--output") + 1]
        __import__("pathlib").Path(output).write_text(json.dumps({
            "candidates": [
                {"sequence": "AAA", "layer2_passed": True},
                {"sequence": "BBB", "layer2_passed": False},
                {"sequence": "CCC", "igblast_passed": True},
            ]
        }))
        return __import__("subprocess").CompletedProcess(command, 0)

    monkeypatch.setattr(race.subprocess, "run", _fake_run)

    kept = race._layer2_gate(
        ["AAA", "BBB", "CCC"],
        command=["tnp", "--input", "{input}", "--output", "{output}"],
        root=tmp_path,
        target="P20809",
    )

    # BBB failed TNP; CCC only has IgBLAST evidence, which cannot be published
    assert kept == ["AAA"]
    source = json.loads((tmp_path / "layer2_gate" / "shortlist.json").read_text())
    assert source["sequences"] == ["AAA", "BBB", "CCC"]


def test_layer2_gate_fails_closed(tmp_path, monkeypatch):
    import subprocess as sp

    def _boom(command, check=False):
        raise sp.CalledProcessError(1, command)

    monkeypatch.setattr(race.subprocess, "run", _boom)

    # publishing on IgBLAST evidence alone is worse than losing replication
    assert race._layer2_gate(
        ["AAA"],
        command=["tnp", "--input", "{input}", "--output", "{output}"],
        root=tmp_path,
        target="P20809",
    ) == []


def test_layer2_gate_is_inert_without_a_command(tmp_path):
    assert race._layer2_gate(
        ["AAA", "BBB"], command=[], root=tmp_path, target="P20809"
    ) == ["AAA", "BBB"]


def test_pooled_exact_paths_reach_earlier_generations(tmp_path):
    producer = tmp_path / "producer"
    for iteration in (1, 2):
        root = producer / f"iteration_{iteration:02d}" / "race" / "screen" / "t"
        root.mkdir(parents=True)
        (root / "exact_all.json").write_text("{}")
    current = producer / "iteration_03" / "race"
    current.mkdir(parents=True)
    local = producer / "iteration_03" / "race" / "screen" / "t"
    local.mkdir(parents=True)
    local_path = local / "exact_all.json"
    local_path.write_text("{}")

    pooled = race._pooled_exact_paths([local_path], producer, current)

    # the closing generation sees every earlier generation's observations,
    # and its own are not duplicated
    assert len(pooled) == 3
    assert pooled.count(local_path) == 1
    assert any("iteration_01" in str(path) for path in pooled)


def test_pooled_exact_paths_are_inert_without_a_root(tmp_path):
    local = [tmp_path / "exact_all.json"]

    assert race._pooled_exact_paths(local, None, tmp_path) == local
