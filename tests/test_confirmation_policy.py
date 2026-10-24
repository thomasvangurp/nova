"""Budget, real-fold quality, and SHIPPING execution guards for broad confirmation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobody_opt_v2 import confirmation_policy as policy
from nanobody_opt_v2 import sequential_exact_race as race
from nanobody_opt_v2.compute_budget import TimingModel
from tools.benchmark_confirmation_policy import benchmark, CONTROL


def make_plan(width=64, gpus=8, rounds=20):
    return policy.plan(candidates=width, rounds=rounds, rounds_per_stage=5,
                       keep_fraction=.5, minimum_keep=20, gpus=gpus,
                       model=TimingModel(9.43, 5.52, 0))


@pytest.mark.parametrize("width", [1, 13, 20, 32, 40, 64, 96, 200])
@pytest.mark.parametrize("gpus", [1, 4, 8, 13])
def test_breadth_never_buys_evaluations_or_time_by_lowering_final_depth(width, gpus):
    plan = make_plan(width, gpus)
    assert plan["planned_evaluations"] <= plan["evaluation_cap"]
    assert plan["planned_seconds"] <= plan["seconds_cap"] + 1e-7
    assert sum(r for _, r in plan["schedule"]) == 20
    assert plan["schedule"][-1][0] >= min(width, 20)


def test_half_pruning_schedule_uses_three_then_eight_then_twenty_one_observations():
    p = make_plan()
    assert p["schedule"] == [(104, 2), (40, 5), (20, 13)]
    assert p["planned_evaluations"] == 668
    assert p["evaluation_cap"] == 680


def test_replay_control_matches_actual_shipping_launch_allocation():
    from nanobody_opt_v2.run_live_producer import _parser
    root = Path(__file__).resolve().parents[1]
    args = _parser().parse_args((root / "deploy/producer/producer_argv.txt").read_text().splitlines()[3:])
    assert CONTROL == dict(candidates=args.epoch_confirmation_finalists,
                           rounds=args.replication_rounds,
                           rounds_per_stage=args.replication_rounds_per_stage,
                           keep_fraction=args.replication_keep_fraction,
                           minimum_keep=args.replication_minimum_keep)
    p = policy.plan(**CONTROL, gpus=8, model=TimingModel(9.43, 5.52, 0))
    assert p["schedule"] == [(147, 2), (40, 5), (20, 13)]
    assert p["planned_evaluations"] == 754 and p["evaluation_cap"] == 755
    assert args.confirmation_policy == "broad"


def test_second_chances_reach_bad_first_draws_without_admitting_excluded_entries():
    ordered = [f"eligible_{i}" for i in range(300)]
    metadata = {"eligible_290": {"parent_multipass_mean": 700},
                "eligible_291": {"parent_multipass_mean": 850},
                "NOT_ELIGIBLE": {"parent_multipass_mean": 1}}
    selected, reasons = policy.shortlist(ordered, width=104, metadata=metadata,
        observations={s: 1 for s in ordered}, epoch=10, broadened=True)
    assert len(set(selected)) == 104
    assert selected[:98] == ordered[:98]
    assert reasons["eligible_290"] == reasons["eligible_291"] == "parent_rescue"
    assert "random_audit" in reasons.values()
    assert "NOT_ELIGIBLE" not in selected
    again = policy.shortlist(ordered, width=104, metadata=metadata,
        observations={s: 1 for s in ordered}, epoch=10, broadened=True)
    assert again == (selected, reasons)


def test_short_window_preserves_depth_and_refuses_impossible_work():
    model = TimingModel(10, 5, 0)
    fitted = policy.fit(make_plan()["schedule"], available=104, usable_seconds=500,
                        model=model, gpus=8)
    assert sum(r for _, r in fitted) == 20
    assert policy.seconds(fitted, model, 8) <= 500
    assert policy.fit(make_plan()["schedule"], available=104, usable_seconds=10,
                      model=model, gpus=8) == []


@pytest.mark.parametrize("fixture", ["30folds", "40folds"])
def test_real_fold_heldout_quality_does_not_regress_at_equal_budget(fixture):
    source = Path(__file__).parent / "fixtures/confirmation_replay" / (fixture + ".json")
    result = benchmark(json.loads(source.read_text())["folds"])
    assert result["passed"], result
    assert result["heldout_folds"] >= 9


def test_actual_runner_preserves_stage_depth_budget_and_resume(tmp_path, monkeypatch):
    calls, prunes = [], []
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: True)
    monkeypatch.setattr(race, "_gpu_shared", lambda: False)
    monkeypatch.setattr(race, "_append_timing", lambda *a: None)
    def score(**kwargs):
        seqs = json.loads(kwargs["source"].read_text())["sequences"]
        calls.append(len(seqs))
        out = kwargs["output_dir"] / "exact_all.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
    monkeypatch.setattr(race, "_run_screen", score)
    def prune(alive, keep):
        prunes.append((len(calls), len(alive), keep))
        return alive[:keep]
    kwargs = dict(shortlist=[f"c{i}" for i in range(104)], rounds=20, root=tmp_path,
        config=tmp_path / "config", target="TEST", gpus=[str(i) for i in range(8)],
        scorer=tmp_path / "scorer", chain_state=None, epoch=10, reserve_blocks=36,
        predict_batch_size=1, timings=tmp_path / "timings", rerank=prune,
        stage_schedule=make_plan()["schedule"])
    assert race._run_replication_rounds(**kwargs) == 20
    assert calls == [104] * 2 + [40] * 5 + [20] * 13
    assert sum(calls) == 668
    assert prunes == [(2, 104, 40), (7, 40, 20)]
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: False)
    assert race._run_replication_rounds(**kwargs) == 20
    assert len(calls) == 20  # Completed work is not rescored on restart.


def test_prepare_gates_before_gpu_and_freezes_plan_on_restart(tmp_path, monkeypatch):
    args = SimpleNamespace(output_dir=tmp_path, epoch=10, finalists=64,
        replication_rounds=20, replication_rounds_per_stage=5, replication_keep_fraction=.5,
        replication_minimum_keep=20, replication_load_seconds=9.43,
        replication_seconds_per_observation=5.52, replication_finish_guard_seconds=30,
        layer2_gate_command="gate", target="TEST", chain_state=None, reserve_blocks=36,
        config=tmp_path / "config", archive=tmp_path / "archive")
    args.config.write_text("test config")
    args.archive.write_text("test archive")
    ranking = tmp_path / "rank.json"
    ranking.write_text(json.dumps({"sequences": [f"c{i}" for i in range(300)]}))
    monkeypatch.setattr(race, "_remaining_blocks", lambda *a: 150)
    monkeypatch.setattr(race, "_append_timing", lambda *a: None)
    monkeypatch.setattr(race, "_layer2_gate", lambda seqs, **kw: [s for s in seqs if s != "c0"])
    selected, stages, _ = race._prepare_broad_confirmation(args=args, ranking=ranking,
        exclude={"c1"}, metadata={}, gpus=list("01234567"), timings=tmp_path / "timings")
    assert "c0" not in selected and "c1" not in selected
    assert len(selected) == stages[0][0] == 104
    receipt = json.loads((tmp_path / "confirmation_plan.json").read_text())
    assert receipt["eligibility"]["refilled_candidates"] >= 1
    ranking.write_text(json.dumps({"sequences": ["different"]}))
    again, again_stages, _ = race._prepare_broad_confirmation(args=args, ranking=ranking,
        exclude=set(), metadata={}, gpus=list("01234567"), timings=tmp_path / "timings")
    assert again == selected and again_stages == stages
    args.config.write_text("changed config")
    with pytest.raises(ValueError, match="resume plan mismatch"):
        race._prepare_broad_confirmation(args=args, ranking=ranking,
            exclude=set(), metadata={}, gpus=list("01234567"), timings=tmp_path / "timings")


def test_completed_rounds_cannot_report_healthy_without_qualified_new_candidates(tmp_path):
    output = tmp_path / "qualified.json"
    output.write_text(json.dumps({"candidates": [
        {"sequence": "new", "observation_count": 19},
        {"sequence": "old", "observation_count": 20}]}))
    result = race._confirmation_health(root=tmp_path, epoch=10, required=20,
        rounds_completed=20, shortlist_count=104, minimum_keep=20,
        qualified_paths=[output], fresh_sequences={"new"})
    assert result["status"] == "degraded_no_new_confirmed_candidates"
    assert result["qualified_candidates"] == 1
    assert result["fresh_confirmed_candidates"] == 0


def test_producer_forwards_policy_to_the_actual_race(monkeypatch, tmp_path):
    from nanobody_opt_v2 import run_live_producer as producer
    args = producer._parser().parse_args([])
    args.budget_mode = "fixed"
    args.gpus = "0,1,2,3,4,5,6,7"
    state = {"epoch": 10, "block": 10 * 361 + 100, "epoch_length": 361}
    monkeypatch.setattr(producer, "_chain_state", lambda *a: state)
    monkeypatch.setattr(producer, "_budget_plan", lambda *a: {
        "confirmation_candidates": 64, "replication_reserve_blocks": 75})
    monkeypatch.setattr(producer, "_start_following_epoch_pool", lambda *a: None)
    commands = []
    monkeypatch.setattr(producer, "_run", commands.append)
    producer._run_race(args=args, state=state, iteration=1, config=tmp_path / "config",
        archive=tmp_path / "archive", target="TEST", producer_root=tmp_path,
        iteration_root=tmp_path / "iteration_01", layer2=tmp_path / "layer2",
        force_confirmation=True)
    command = commands[0]
    assert command[command.index("--confirmation-policy") + 1] == "broad"
    assert command[command.index("--minimum-publishable-observations") + 1] == str(args.minimum_publishable_observations)
