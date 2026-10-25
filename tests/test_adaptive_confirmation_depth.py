"""Deadline fallback must reach publication without weakening eligibility."""
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobody_opt import confirmation_depth as depth, publication_gates as gates
from nanobody_opt import filter_layer2
from nanobody_opt.publish_checkpoint import validate_exact_robustness
from nanobody_opt_v2 import confirmation_policy as policy
from nanobody_opt_v2 import sequential_exact_race as race
from nanobody_opt_v2.compute_budget import TimingModel
from tools import publish_current_nanobody as publisher


@pytest.mark.parametrize("seconds,minimum", [(1000, 20), (300, 10), (10, 10)])
def test_fit_preserves_fleet_breadth_then_uses_ten_total_passes(seconds, minimum):
    model = TimingModel(4.83, 6.095, 0)
    original = [(147, 2), (40, 5), (20, 13)]
    fitted, required = policy.fit_depth(original, available=147, usable_seconds=seconds,
                                        model=model, gpus=8, minimum_keep=20)
    assert required == minimum
    assert policy.seconds(fitted, model, 8) <= seconds
    assert policy.evaluation_count(fitted) <= policy.evaluation_count(original)
    if seconds == 10:
        assert fitted == []
    else:
        assert fitted[-1][0] >= 20
        assert sum(r for _, r in fitted) == (20 if minimum == 20 else 9)


def prepare(tmp_path, monkeypatch, remaining, cached_pass=False):
    args = SimpleNamespace(output_dir=tmp_path / "race", pooled_exact_root=tmp_path,
        epoch=100, finalists=64, replication_rounds=20, replication_rounds_per_stage=5,
        replication_keep_fraction=.65, replication_minimum_keep=20,
        replication_load_seconds=4.83, replication_seconds_per_observation=6.095,
        replication_finish_guard_seconds=30, layer2_gate_command=f"gate --cache-dir {tmp_path}",
        target="TEST", chain_state=None, reserve_blocks=36,
        config=tmp_path / "config", archive=tmp_path / "archive",
        adaptive_confirmation_depth=True, minimum_publishable_observations=20)
    args.config.write_text("config"); args.archive.write_text("archive")
    ranking = tmp_path / "rank.json"
    ranking.write_text(json.dumps({"sequences": [f"candidate{i}" for i in range(300)]}))
    monkeypatch.setattr(race, "_remaining_blocks", lambda *a: remaining)
    monkeypatch.setattr(filter_layer2, "cached_developability_result",
                        lambda *a: {"passed": True} if cached_pass else None)
    calls = []
    def gate(batch, **kwargs):
        assert kwargs["timeout_seconds"] > 0
        calls.append((len(batch), kwargs["timeout_seconds"]))
        return batch
    monkeypatch.setattr(race, "_layer2_gate", gate)
    result = race._prepare_broad_confirmation(args=args, ranking=ranking, exclude=set(),
        metadata={}, gpus=list(range(8)), timings=tmp_path / "timings")
    return args, ranking, result, calls


def test_cold_gate_short_window_runs_checks_and_freezes_ten_pass_plan(tmp_path, monkeypatch):
    args, ranking, (selected, schedule, model), calls = prepare(tmp_path, monkeypatch, 70)
    assert calls and len(selected) >= 20
    assert sum(r for _, r in schedule) == 9
    receipt = json.loads((args.output_dir / "confirmation_plan.json").read_text())
    assert receipt["gate_budget_seconds"] == 120
    assert receipt["confirmation_depth"]["minimum_observations"] == 10
    assert depth.read(tmp_path, epoch=100, target="TEST")["minimum_observations"] == 10
    # Resuming later cannot widen or deepen a frozen tournament.
    monkeypatch.setattr(race, "_remaining_blocks", lambda *a: 300)
    again = race._prepare_broad_confirmation(args=args, ranking=ranking, exclude=set(),
        metadata={}, gpus=list(range(8)), timings=tmp_path / "timings")
    assert again[:2] == (selected, schedule)


def test_ample_window_retains_full_depth(tmp_path, monkeypatch):
    args, _, (selected, schedule, _), calls = prepare(tmp_path, monkeypatch, 150, True)
    assert calls and len(selected) == 147
    assert sum(r for _, r in schedule) == 20
    assert args._effective_minimum_observations == 20


def test_receipt_cannot_lower_another_epoch_or_target_or_below_ten(tmp_path):
    value = depth.decision(epoch=100, target="TEST", minimum=10, schedule=[(20, 9)])
    path = tmp_path / "confirmation_depth.json"
    path.write_text(json.dumps(value))
    assert depth.read(tmp_path, epoch=101, target="TEST") is None
    assert depth.read(tmp_path, epoch=100, target="OTHER") is None
    value["minimum_observations"] = 5
    path.write_text(json.dumps(value))
    assert depth.read(tmp_path, epoch=100, target="TEST") is None


@pytest.mark.parametrize("strict", [False, True])
def test_publisher_ranking_uses_epoch_floor_and_resets_it(tmp_path, strict):
    baseline = gates.gate_defaults()
    with gates.evidence_floor(10):
        changed = gates.gate_defaults()
        assert {k for k in baseline if baseline[k] != changed[k]} == {"--minimum-observations"}
        command = publisher._rank_command(bgen_python=tmp_path / "python",
            nanobody_opt_root=tmp_path, exact_inputs=[], archive=tmp_path / "archive",
            target="TEST", output=tmp_path / "out", exclusion=tmp_path / "excluded",
            epoch=100, shortlist=100, config=tmp_path / "config", strict=strict)
        assert command[command.index("--minimum-observations") + 1] == "10"
        assert publisher._strict_minimum_observations() == 10
    assert gates.gate_defaults() == baseline


def test_checkpoint_accepts_ten_but_keeps_quality_and_integrity_gates():
    g = gates.gate_defaults()
    winner = {"observation_count": 10, "competitive_rate": 1., "observation_competitive_rate": 1.,
        "score_p90": 1., "p90_category_rank_sums": {"confidence": 1., "physical_interaction": 1., "developability": 1.},
        "pose_stability": {"structure_observation_count": 1, "contact_jaccard_median": 1.,
                           "cdr3_rmsd_max_angstrom": 0., "epitope_occupancy_min": 1.}}
    assert validate_exact_robustness(winner, minimum_observations=10)[0] == 10
    with pytest.raises(ValueError, match="fewer than 20"):
        validate_exact_robustness(winner)
    with pytest.raises(ValueError, match="fewer than 10"):
        validate_exact_robustness({**winner, "observation_count": 9}, minimum_observations=10)
    with pytest.raises(ValueError, match="p90 rank sum"):
        validate_exact_robustness({**winner, "score_p90": float(g["--maximum-p90-rank-sum"]) + 1}, minimum_observations=10)
    winner["pose_stability"]["contact_jaccard_median"] = 0
    with pytest.raises(ValueError, match="contact Jaccard"):
        validate_exact_robustness(winner, minimum_observations=10)


def test_changed_archive_revalidation_yields_partial_current_scope(tmp_path, monkeypatch):
    root, opt = tmp_path / "epoch100", tmp_path / "opt"
    root.mkdir(); opt.mkdir()
    for name in ("check_validator_novelty.py", "filter_layer2.py"):
        (opt / name).write_text("gate")
    archive, config, exclusion = tmp_path / "archive", tmp_path / "config", tmp_path / "exclusion"
    archive.write_text("sequence\nPUBLISHED\n"); config.write_text("config"); exclusion.write_text("")
    (root / "comparison_precheck.json").write_text(json.dumps({"key": {"archive": "old"},
        "checked_sequences": ["OLD_INVALID"], "eligible_sequences": ["OLD_INVALID", "OLD_VALID"]}))
    bank = {f"FRESH{i:05d}" for i in range(10000)} | {"OLD_INVALID", "OLD_VALID"}
    monkeypatch.setattr(publisher, "_stored_comparison_sequences", lambda *a: bank)
    novelty_batches = []
    def run(command):
        seqs = json.loads(Path(command[command.index("--input") + 1]).read_text())["sequences"]
        output = Path(command[command.index("--output") + 1])
        if "check-validator-novelty" in command:
            novelty_batches.append(seqs)
            publisher._atomic_json(output, {"input_count": len(seqs), "sequences": [s for s in seqs if s != "OLD_INVALID"]})
        else:
            publisher._atomic_json(output, {"input_count": len(seqs), "candidates": [
                {"sequence": s, "igblast_passed": True} for s in seqs]})
    args = SimpleNamespace(nanobody_opt_root=opt, layer2_cache_dir=tmp_path / "cache",
                           layer2_concurrency=2, pixi=Path("pixi"), adaptive_confirmation_depth=True)
    publisher._prechecked_comparison_scope(args=args, epoch_root=root, target="TEST",
        database=tmp_path / "db", archive=archive, config=config, exclusion=exclusion,
        pixi_manifest=tmp_path / "pixi.toml", epoch=100, cheap_only=True,
        minimum_observations=1, runner=run)
    proof = json.loads((root / "comparison_precheck.json").read_text())
    assert len(novelty_batches[0]) == 4096
    assert "OLD_INVALID" in novelty_batches[0] and "OLD_VALID" in novelty_batches[0]
    assert "OLD_INVALID" not in proof["eligible_sequences"]
    assert "OLD_VALID" in proof["eligible_sequences"]
    assert proof["eligible_count"] > 0 and proof["pending_count"] > 0
    # Every admission still has current-archive proof; backlog is not omitted.
    assert set(proof["eligible_sequences"]) <= set(proof["checked_sequences"])
    assert proof["checked_count"] + proof["pending_count"] == len(bank)


def test_shipping_enables_policy_without_changing_new_evaluation_cap():
    from nanobody_opt_v2 import run_live_producer as producer
    root = Path(__file__).resolve().parents[1]
    args = producer._parser().parse_args((root / "deploy/producer/producer_argv.txt").read_text().splitlines()[3:])
    assert args.adaptive_confirmation_depth and args.epoch_candidate_budget == 4800
    task = next(l for l in (root / "pixi.toml").read_text().splitlines() if l.startswith("publish-current-nanobody ="))
    assert "--adaptive-confirmation-depth" in task


def test_running_precheck_refreshes_changed_archive_between_batches(tmp_path, monkeypatch):
    from nanobody_opt_v2 import cached_confirmation
    archive, config, root = tmp_path / "archive", tmp_path / "config", tmp_path / "scope"
    archive.write_text("initial archive"); config.write_text("config")
    observed = []
    def batch(**kwargs):
        observed.append(kwargs["archive"].read_text())
        publisher._atomic_json(root / "comparison_precheck.json",
                              {"eligible_count": 1, "pending_count": int(len(observed) == 1)})
        if len(observed) == 1:
            archive.write_text("updated archive")
    monkeypatch.setattr(publisher, "_prechecked_comparison_scope", batch)
    args = SimpleNamespace(runtime_root=tmp_path, ledger=None, mutation_database=tmp_path / "db",
                           layer2_concurrency=2, adaptive_confirmation_depth=True)
    cached_confirmation.prepare(args=args, state={"epoch": 100, "nanobody_target": "TEST"},
                                archive=archive, config=config, root=root, runner=None)
    assert observed == ["initial archive", "updated archive"]


def test_publisher_floor_resets_after_a_failed_attempt(monkeypatch):
    def failed(args):
        gates.set_evidence_floor(10)
        raise RuntimeError("interrupted publication")
    monkeypatch.setattr(publisher, "_publish_once_current_epoch", failed)
    with pytest.raises(RuntimeError):
        publisher._publish_once(SimpleNamespace())
    assert publisher._strict_minimum_observations() == 20


@pytest.mark.parametrize("stage", ["provisional", "qualified"])
def test_consumer_requires_epoch_bound_ten_pass_receipt(stage):
    from test_publish_current_nanobody import _checkpoint, EPOCH, TARGET, REQUIREMENTS
    from compound_miner.nanobody_provider import validate_nanobody_checkpoint, NanobodyCheckpointError
    payload = _checkpoint(stage=stage)
    payload["exact"]["observation_count"] = 10
    payload["confirmation_depth"] = depth.decision(epoch=EPOCH, target=TARGET,
                                                  minimum=10, schedule=[])
    def validate():
        return validate_nanobody_checkpoint(payload, expected_epoch=EPOCH, expected_target=TARGET,
            min_sequence_length=REQUIREMENTS["min_sequence_length"],
            max_sequence_length=REQUIREMENTS["max_sequence_length"])
    validate()
    payload["exact"]["observation_count"] = 9
    with pytest.raises(NanobodyCheckpointError):
        validate()
    payload["exact"]["observation_count"] = 10
    payload["confirmation_depth"]["epoch"] += 1
    with pytest.raises(NanobodyCheckpointError, match="confirmation-depth"):
        validate()


def test_actual_publisher_builds_and_validates_ten_pass_fleet(tmp_path, monkeypatch):
    from test_publish_current_nanobody import _Host, EPOCH, TARGET, REQUIREMENTS
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    args = host.args()
    args.adaptive_confirmation_depth = True
    receipt = depth.decision(epoch=EPOCH, target=TARGET, minimum=10, schedule=[(20, 9)])
    publisher._atomic_json(args.runtime_root / "live" / f"epoch{EPOCH}" / "producer/confirmation_depth.json", receipt)
    original = publisher._run
    checkpoints = []
    def run(command):
        result = original(command)
        if any(str(x).endswith("rank_exact.py") for x in command):
            output = Path(command[command.index("--output") + 1])
            data = json.loads(output.read_text())
            for candidate in data.get("candidates", []):
                candidate["observation_count"] = 10
            publisher._atomic_json(output, data)
        if any(str(x).endswith("publish_checkpoint.py") for x in command):
            assert command[command.index("--minimum-observations") + 1] == "10"
            output = Path(command[command.index("--output") + 1])
            data = json.loads(output.read_text())
            data["confirmation_depth"] = receipt
            publisher._atomic_json(output, data)
            checkpoints.append(data)
        return result
    monkeypatch.setattr(publisher, "_run", run)
    publisher._publish_once(args)
    artifact = json.loads(host.artifact.read_text())
    assert len(artifact["submissions"]) == 3
    assert artifact["confirmation_depth"]["minimum_observations"] == 10
    assert checkpoints and all(c["exact"]["observation_count"] == 10 for c in checkpoints)
    assert publisher._strict_minimum_observations() == 20
    assert all(publisher._is_current_qualified(c, target=TARGET, epoch=EPOCH,
                                               requirements=REQUIREMENTS) for c in checkpoints)


def test_actual_race_uses_nine_rounds_and_qualifies_ten_observations(tmp_path, monkeypatch):
    from nanobody_opt import rank_exact
    root = tmp_path / "producer"
    out = root / "iteration_02/race"
    root.mkdir()
    config, archive, layer2 = root / "config", root / "archive", root / "layer2"
    config.write_text("config"); archive.write_text("archive")
    sequences = [f"candidate{i}" for i in range(300)]
    layer2.write_text(json.dumps({"candidates": [{"sequence": s} for s in sequences]}))
    parser = race._parser()
    args = parser.parse_args(["--layer2", str(layer2), "--archive", str(archive),
        "--config", str(config), "--target", "TEST", "--epoch", "100",
        "--gpus", "0,1,2,3,4,5,6,7", "--output-dir", str(out),
        "--pooled-exact-root", str(root), "--replication-only", "--replication-rounds", "20",
        "--finalists", "64", "--confirmation-policy", "broad", "--adaptive-confirmation-depth",
        "--replication-load-seconds", "4.83", "--replication-seconds-per-observation", "6.095",
        "--replication-finish-guard-seconds", "30", "--reserve-blocks", "36",
        "--layer2-gate-command", f"gate --cache-dir {tmp_path}"])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(race, "_parser", lambda: parser)
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: True)
    monkeypatch.setattr(race, "_remaining_blocks", lambda *a: 70)
    monkeypatch.setattr(race, "_exact_paths", lambda *a: [root / "exact.json"])
    monkeypatch.setattr(race, "_pooled_exact_paths", lambda *a: [root / "exact.json"])
    monkeypatch.setattr(rank_exact, "_observations", lambda *a: {s: [{}] for s in sequences})
    monkeypatch.setattr(race, "_apply_basin_shrinkage", lambda *a, **kw: None)
    monkeypatch.setattr(race, "_store_ranking", lambda *a, **kw: None)
    monkeypatch.setattr(filter_layer2, "cached_developability_result", lambda *a: None)
    def rank(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        race._atomic_json(output, {"sequences": sequences, "candidates": [
            {"sequence": s, "observation_count": 1, "selection_score": 1000 + i}
            for i, s in enumerate(sequences)]})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(race.subprocess, "run", rank)
    monkeypatch.setattr(race, "_layer2_gate", lambda seqs, **kw: seqs)
    def repeat(**kwargs):
        assert kwargs["rounds"] == 9
        assert sum(r for _, r in kwargs["stage_schedule"]) == 9
        return 9
    monkeypatch.setattr(race, "_run_replication_rounds", repeat)
    def strict(**kwargs):
        assert kwargs["strict_gate_overrides"]["--minimum-observations"] == 10
        race._atomic_json(kwargs["output"], {"candidates": [
            {"sequence": s, "observation_count": 10, "multipass_rank": 1000}
            for s in sequences[:20]]})
        return True
    monkeypatch.setattr(race, "_try_strict_rank", strict)
    monkeypatch.setattr(race, "_best_confirmed_rank", lambda *a: 1000)
    race.main()
    complete = json.loads((out / "complete.json").read_text())
    assert complete["replication_rounds_completed"] == 9
    assert complete["confirmation_health"]["required_observations"] == 10
    assert complete["confirmation_health"]["qualified_candidates"] == 20
