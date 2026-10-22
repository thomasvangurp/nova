import json
from pathlib import Path

import pytest

from nanobody_opt_v2 import run_live_producer as producer
from sn68_source import import_shipping


@pytest.mark.parametrize("full,expected", [(False, 4800), (True, 11002)])
def test_full_pool_admission_does_not_silently_truncate(monkeypatch, tmp_path, full, expected):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    parser = race._parser()
    args = parser.parse_args([
        "--layer2", "pool.json", "--archive", "archive.csv", "--config", "config.yaml",
        "--target", "TEST", "--epoch", "101", "--output-dir", str(tmp_path),
        "--gpus", "0", *(["--screen-all-candidates"] if full else []),
    ])
    monkeypatch.setattr(parser, "parse_args", lambda: args)
    monkeypatch.setattr(race, "_parser", lambda: parser)
    monkeypatch.setattr(race, "_candidate_records", lambda p: [
        {"sequence": f"candidate{i}"} for i in range(11002)])
    monkeypatch.setattr(race, "_chain_allows_work", lambda *a: False)
    race.main()
    assert len(json.loads((tmp_path / "screen_candidates.json").read_text())["sequences"]) == expected


def test_carryover_keeps_prior_raw_results_and_excludes_own_output(monkeypatch, tmp_path):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    prior, current = tmp_path / "epoch100/producer", tmp_path / "epoch101/producer"
    old = prior / "iteration_01/race/screen/tranche_01/exact_all.json"
    own = current / "iteration_02/race"
    local = own / "screen/tranche_01/exact_all.json"
    for p in [old, local]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    (current / "pooled_roots.json").write_text(json.dumps({"roots": [str(prior), str(prior)]}))
    assert race._pooled_exact_paths([], current, own) == [old]
    assert set(race._pooled_exact_paths([local], current, own)) == {old, local}


def test_complete_pass_spends_no_folds_rescreening_old_candidates(monkeypatch, tmp_path):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    monkeypatch.setattr(rank, "_observations", lambda *a: {"OLD": [{}]})
    monkeypatch.setattr(race, "_already_scored", lambda *a, **kw: {"OLD": 1})
    monkeypatch.setattr(race, "_multipass_rank_map", lambda *a, **kw: pytest.fail("no re-screen ranking"))
    records = [{"sequence": "OLD"}, {"sequence": "NEW"}]
    assert race._carry_pending_records(records, tmp_path, tmp_path / "race", "TEST",
                                       Path("db"), 101, rescore_top=0) == [{"sequence": "NEW"}]


@pytest.mark.parametrize("epoch,complete,force", [(100, False, False), (101, False, False), (101, True, True)])
def test_fixed_pool_rechecks_novelty_and_confirms_only_after_coverage(
        monkeypatch, tmp_path, epoch, complete, force):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.complete_screen_pool = tmp_path / "pool.json"
    args.complete_screen_until_epoch = 101
    args.complete_screen_pool.write_text(json.dumps({
        "target": "TEST", "origin_epoch": 100,
        "candidates": [{"sequence": "novel"}, {"sequence": "blocked"}],
    }))
    state = {"epoch": epoch, "epoch_length": 361, "block": epoch * 361,
             "nanobody_target": "TEST"}
    monkeypatch.setattr(producer, "_prepare_epoch", lambda **kw: (
        tmp_path / "config", tmp_path / "archive", "TEST"))
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: {"novel"})
    monkeypatch.setattr(producer, "_complete_pool_coverage", lambda **kw: {
        "complete": complete, "eligible": 1, "screened": int(complete), "pending": int(not complete)})
    calls = []
    def run_race(**kw):
        calls.append(kw)
        if force:
            producer._atomic_json(kw["iteration_root"] / "race/complete.json", {
                "replication_rounds_completed": 20, "incumbent_rank": 950,
            })
    monkeypatch.setattr(producer, "_run_race", run_race)
    producer._run_complete_screen_pool(args, state)
    assert len(calls) == 1
    call = calls[0]
    assert call["force_confirmation"] == force
    assert call["args"].screen_all_candidates
    assert call["args"].maximum_screen_tranches_override == 0
    assert json.loads(call["layer2"].read_text())["sequences"] == ["novel"]
    roots = json.loads((call["producer_root"] / "pooled_roots.json").read_text())["roots"]
    assert len(roots) == epoch - 100
    if force:
        result = json.loads((call["producer_root"] / "full_scan_confirmation.json").read_text())
        assert result["complete"]
        assert result["best_confirmed_multipass"] == 950
        assert not result["score_target_met"]


def test_coverage_counts_complete_observations_not_attempts(monkeypatch, tmp_path):
    race = import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    rank = import_shipping("nanobody_opt.rank_exact", monkeypatch)
    monkeypatch.setattr(race, "_already_scored", lambda *a, **kw: {"stored": 1})
    monkeypatch.setattr(rank, "_observations", lambda *a: {"raw": [{}], "outside": [{}]})
    report = producer._complete_pool_coverage(
        pool={"candidates": [{"sequence": s} for s in ["stored", "raw", "failed"]]},
        producer_root=tmp_path, database=tmp_path / "db", target="TEST", epoch=101, until_epoch=101)
    assert report["screened"] == 2
    assert report["pending"] == 1
    assert not report["complete"]


def test_current_epoch_spends_its_remaining_time_on_first_pass(monkeypatch, tmp_path):
    args = producer._parser().parse_args([])
    args.budget_mode = "adaptive"
    args.gpus = "0,1,2,3,4,5,6,7"
    args.replication_rounds = 20
    args.complete_screen_until_epoch = 101
    args.finalists = 64
    state = {"epoch": 100, "epoch_length": 361}
    before = producer._budget_plan(args, state, tmp_path / "epoch100/producer")
    after = producer._budget_plan(args, {**state, "epoch": 101}, tmp_path / "epoch101/producer")
    assert before["replication_reserve_blocks"] == 0
    assert before["confirmation_rounds"] == 0
    assert after["replication_reserve_blocks"] > 0
    assert after["confirmation_rounds"] == 20
