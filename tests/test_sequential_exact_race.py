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
