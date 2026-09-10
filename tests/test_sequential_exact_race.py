import json

from nanobody_opt import sequential_exact_race as race


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


def test_screen_tranche_loads_each_gpu_pipeline_only_once():
    assert race._screen_execution_shape(20, 10) == (20, 2)
    assert race._screen_execution_shape(7, 10) == (7, 1)


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
