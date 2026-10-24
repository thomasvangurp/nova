import argparse
import json
from pathlib import Path

import pytest

from nanobody_opt_v2 import cached_confirmation as cached
from nanobody_opt_v2 import run_live_producer as producer
from sn68_source import import_shipping


def proof(root, *, epoch, target, archive, config, opt_root, allowed):
    root.mkdir(parents=True, exist_ok=True)
    from tools.publish_current_nanobody import PUBLISHER_STATE_SCHEMA_VERSION
    data = {"key": {"schema_version": PUBLISHER_STATE_SCHEMA_VERSION,
                    "policy": "incremental-layer2-v1",
                    "epoch": epoch, "target": target, "confirmation_only": True,
                    "minimum_observations": 1, "archive": cached.digest(archive),
                    "config": cached.digest(config), "gates": {name: cached.digest(opt_root / name)
                        for name in ("check_validator_novelty.py", "filter_layer2.py")}},
            "eligible_sequences": list(allowed), "checked_sequences": list(allowed),
            "pending_count": 0}
    (root / "comparison_precheck.json").write_text(json.dumps(data))
    return data


@pytest.mark.parametrize("changed", [None, "epoch", "target", "archive", "config", "gate"])
def test_cached_scope_requires_current_complete_eligibility_proof(tmp_path, changed):
    archive, config, opt = tmp_path / "archive", tmp_path / "config", tmp_path / "opt"
    archive.write_text("archive"); config.write_text("config"); opt.mkdir()
    for name in ("check_validator_novelty.py", "filter_layer2.py"):
        (opt / name).write_text("gate")
    root = tmp_path / "scope"
    d = proof(root, epoch=100, target="TEST", archive=archive, config=config,
              opt_root=opt, allowed={"CACHED"})
    d["eligible_sequences"].append("UNCHECKED")
    (root / "comparison_precheck.json").write_text(json.dumps(d))
    if changed in ("archive", "config"):
        (archive if changed == "archive" else config).write_text("changed")
    if changed == "gate": (opt / "filter_layer2.py").write_text("changed")
    allowed, status = cached.read_scope(root, epoch=101 if changed == "epoch" else 100,
        target="OTHER" if changed == "target" else "TEST", archive=archive, config=config, opt_root=opt)
    assert allowed == ({"CACHED"} if changed is None else set())
    assert status["ready"] == (changed is None)


def test_cached_precheck_includes_thin_history_and_drains_all_batches_without_fresh_budget(tmp_path, monkeypatch):
    publisher = import_shipping("tools.publish_current_nanobody", monkeypatch)
    root, opt = tmp_path / "scope", tmp_path / "nanobody_opt"
    opt.mkdir()
    for name in ("check_validator_novelty.py", "filter_layer2.py"):
        (opt / name).write_text("gate")
    archive, config = tmp_path / "archive.csv", tmp_path / "config.yaml"
    archive.write_text("sequence\nPUBLISHED\n"); config.write_text("configuration")
    bank = {f"CACHED{i}" for i in range(2050)} | {"PUBLISHED", "INVALID"}
    floors = []
    def stored(db, target, minimum):
        floors.append(minimum)
        return bank
    monkeypatch.setattr(publisher, "_stored_comparison_sequences", stored)
    batches = []
    def run(command):
        inp = json.loads(Path(command[command.index("--input")+1]).read_text())
        output = Path(command[command.index("--output")+1])
        seqs = inp["sequences"]
        if "check-validator-novelty" in command:
            publisher._atomic_json(output, {"input_count": len(seqs), "sequences": [s for s in seqs if s != "INVALID"]})
        else:
            assert "--igblast-only" in command
            batches.append(len(seqs))
            publisher._atomic_json(output, {"input_count": len(seqs), "candidates": [
                {"sequence": s, "igblast_passed": True} for s in seqs]})
    args = argparse.Namespace(runtime_root=tmp_path, ledger=None,
        mutation_database=tmp_path / "db", layer2_concurrency=2)
    cached.prepare(args=args, state={"epoch": 100, "nanobody_target": "TEST"},
                   archive=archive, config=config, root=root, runner=run)
    allowed, status = cached.read_scope(root, epoch=100, target="TEST", archive=archive, config=config, opt_root=opt)
    assert len(allowed) == 2050 and status["pending"] == 0
    assert batches == [2048, 2] and floors == [1, 1]
    assert not list(tmp_path.rglob("fresh_screen_budget.json"))


def test_confirmation_compares_cached_partial_folds_without_rescreening_or_repeating_deep_history(tmp_path, monkeypatch):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    from nanobody_opt_v2 import screen_budget
    archive, config, layer2 = tmp_path / "archive.csv", tmp_path / "config.yaml", tmp_path / "layer2.json"
    archive.write_text("sequence\nPUBLISHED\n"); config.write_text("configuration")
    layer2.write_text(json.dumps({"candidates": [{"sequence": "FRESH"}]}))
    root, out = tmp_path / "producer", tmp_path / "producer/iteration_02/race"
    proof(root / "cached_confirmation", epoch=100, target="TEST", archive=archive, config=config,
          opt_root=Path(race.__file__).resolve().parents[1] / "nanobody_opt",
          allowed={"OLD1", "OLD5", "DEEP"})
    screen_budget.reserve(["FRESH"], root, 100, 4800)
    parser = race._parser()
    args = parser.parse_args([
        "--layer2", str(layer2), "--archive", str(archive), "--config", str(config),
        "--target", "TEST", "--epoch", "100", "--gpus", "0", "--output-dir", str(out),
        "--pooled-exact-root", str(root), "--fresh-screen-budget", "4800",
        "--screen-all-candidates", "--replication-only", "--replication-rounds", "20",
        "--finalists", "2", "--maximum-metric-percentile", "1",
        "--layer2-gate-command", "full-gates {input} {output}"])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(race, "_parser", lambda: parser)
    monkeypatch.setattr(race, "_prior_evaluation_sequences", lambda *a, **kw: set())
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: True)
    monkeypatch.setattr(race, "_remaining_blocks", lambda *a: 300)
    monkeypatch.setattr(race, "_pooled_exact_paths", lambda *a: [tmp_path / "fresh_exact.json"])
    monkeypatch.setattr(rank, "_observations", lambda *a: {"FRESH": [{}]})
    monkeypatch.setattr(race, "_apply_basin_shrinkage", lambda *a, **kw: None)
    def run(command, **kw):
        assert "--current-exact-candidates-only" not in command
        scope = Path(command[command.index("--only-sequences") + 1])
        assert set(scope.read_text().splitlines()) == {"FRESH", "OLD1", "OLD5", "DEEP"}
        rows = [{"sequence": s, "observation_count": depth, "multipass_rank": score}
                for s, depth, score in [("DEEP", 20, 800), ("OLD5", 5, 1144), ("FRESH", 1, 1300), ("OLD1", 1, 2008)]]
        producer._atomic_json(out / "screen_ranking.json", {"sequences": [r["sequence"] for r in rows], "candidates": rows})
    monkeypatch.setattr(race.subprocess, "run", run)
    gates = []
    monkeypatch.setattr(race, "_layer2_gate", lambda rows, **kw: gates.append(rows) or rows)
    confirmations = []
    monkeypatch.setattr(race, "_run_replication_rounds", lambda **kw: confirmations.append(kw["shortlist"]) or 0)
    race.main()
    assert gates == confirmations == [["OLD5", "FRESH"]]
    assert len(screen_budget.snapshot(root, 100, 4800)["reserved"]) == 1


def test_interrupted_candidate_assembly_cannot_be_reused(tmp_path):
    p = tmp_path / "candidates.csv"
    p.write_text("sequence\nPARTIAL\n")
    producer._stamp_artifact_policy(p)  # Legacy code stamped before finishing.
    assert not producer._assembled_candidates(p)
    p.write_text("sequence\nCOMPLETE\n")
    producer._finish_candidate_assembly(p)
    assert producer._assembled_candidates(p)
    p.write_text("sequence\nCHANGED\n")
    assert not producer._assembled_candidates(p)


def test_seen_children_do_not_use_the_new_proposal_limit(tmp_path, monkeypatch):
    lane = import_shipping("nanobody_opt.competitor_lane", monkeypatch)
    parents = ["A" * 127, "C" * 127]
    monkeypatch.setattr(lane, "_load_pool", lambda _: parents)
    monkeypatch.setattr(lane, "multipass_scores", lambda *a, **kw: dict(zip(parents, [700, 900])))
    kwargs = dict(archive=tmp_path / "archive", positions=[1, 2], priority=parents,
                  mutation_database=tmp_path / "db", epoch=100)
    previous = lane.generate(limit=2, **kwargs)
    seen = {r["sequence"] for r in previous["candidates"]}
    fresh = lane.generate(limit=2, excluded_sequences=seen, **kwargs)
    assert len(fresh["candidates"]) == 2
    assert not seen & {r["sequence"] for r in fresh["candidates"]}
    assert fresh["previously_evaluated_excluded"] == 2
    assert fresh["unseen_before_limit"]
