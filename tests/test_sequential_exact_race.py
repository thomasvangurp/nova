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
