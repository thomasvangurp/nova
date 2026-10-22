import json

import pytest

from nanobody_opt_v2 import run_live_producer as producer
from sn68_source import import_shipping


def state(epoch):
    return {"epoch": epoch, "epoch_length": 361, "block": epoch * 361,
            "nanobody_target": "TEST"}


@pytest.mark.parametrize("limit,expected", [(None, ["A", "B", "C", "D"]), (3, ["A", "B", "C"])])
def test_child_lane_tail_respects_explicit_proposal_budget(monkeypatch, limit, expected):
    generator = import_shipping("nanobody_opt.local_neighborhood", monkeypatch)
    result = generator._append_proposal_tail(
        [{"sequence": "A"}], [{"sequence": s} for s in ["A", "B", "C", "D"]], limit=limit)
    assert [c["sequence"] for c in result] == expected


def test_prepare_once_preserves_oversupply_for_fresh_epoch_gate(tmp_path, monkeypatch):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    calls = []
    def prepare(**kw):
        calls.append(kw)
        assert kw["prepare_only"]
        assert kw["args"].generation_proposal_budget == 8000
        assert kw["args"].child_floor == 4800
        assert sum(producer._generation_quotas(kw["args"], 8000).values()) == 8000
        producer._atomic_json(kw["iteration_root_override"] / "layer2_dedup.json", {
            "candidates": [{"sequence": f"SEQ{i}"} for i in range(6500)],
        })
    monkeypatch.setattr(producer, "_run_iteration", prepare)
    first = producer._prepare_epoch_pool(args, state(71), 72)
    assert producer._prepare_epoch_pool(args, state(72), 72) == first
    assert len(calls) == 1
    pool = json.loads(first.read_text())
    assert len(pool["candidates"]) == 6500
    assert pool["exact_admission_limit"] == 4800
    assert pool["prepared_during_epoch"] == 71
    assert pool["origin_epoch"] == 72


def test_live_admission_filters_first_then_uses_strongest_4800(tmp_path, monkeypatch):
    import_shipping("nanobody_opt_v2.sequential_exact_race", monkeypatch)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.complete_screen_until_epoch = 72
    args.complete_screen_pool = tmp_path / "pool.json"
    producer._atomic_json(args.complete_screen_pool, {
        "target": "TEST", "origin_epoch": 72, "exact_admission_limit": 4800,
        "candidates": [{"sequence": f"SEQ{i}", "parent_multipass_mean": 8000 - i}
                       for i in range(8000)],
    })
    monkeypatch.setattr(producer, "_prepare_epoch", lambda **kw: (tmp_path / "cfg", tmp_path / "archive", "TEST"))
    # Reject 2000 of the strongest candidates at the actual epoch's novelty gate.
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: {f"SEQ{i}" for i in range(6000)})
    monkeypatch.setattr(producer, "_complete_pool_coverage", lambda **kw: {"complete": False})
    calls = []
    monkeypatch.setattr(producer, "_run_race", lambda **kw: calls.append(kw))
    producer._run_complete_screen_pool(args, state(72))
    assert len(calls) == 1
    pool = json.loads(calls[0]["layer2"].read_text())
    assert len(pool["candidates"]) == 4800
    assert set(pool["sequences"]) == {f"SEQ{i}" for i in range(1200, 6000)}
    assert pool["novelty_rejected"] == 2000
    assert pool["deferred"] == 1200
    assert calls[0]["args"].screen_all_candidates
    assert calls[0]["args"].finalists == 64


@pytest.mark.parametrize("confirmed,epoch,expected", [(False, 71, []), (True, 70, []), (True, 71, [72])])
def test_next_pool_waits_for_current_confirmation(tmp_path, monkeypatch, confirmed, epoch, expected):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.epoch_cycle_start = 72
    if confirmed:
        producer._atomic_json(tmp_path / f"live/epoch{epoch}/producer/full_scan_confirmation.json", {})
    monkeypatch.setattr(producer, "_chain_state", lambda *a: state(epoch))
    calls = []
    monkeypatch.setattr(producer, "_prepare_epoch_pool", lambda a, s, e: calls.append(e))
    producer._prepare_following_epoch_pool(args, state(epoch))
    assert calls == expected


@pytest.mark.parametrize("epoch,route", [(71, "full"), (72, "cycle"), (73, "cycle")])
def test_cycle_routes_without_interrupting_existing_full_pool(tmp_path, monkeypatch, epoch, route):
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.once = True
    args.complete_screen_pool = tmp_path / "pool.json"
    args.complete_screen_until_epoch = 71
    args.epoch_cycle_start = 72
    monkeypatch.setattr(producer, "_chain_state", lambda *a: state(epoch))
    monkeypatch.setattr(producer, "maintain_runtime_storage", lambda *a, **kw: {})
    monkeypatch.setattr(producer, "_prepare_following_epoch_pool", lambda *a: None)
    monkeypatch.setattr(producer, "_run_complete_screen_pool", lambda *a: "full")
    monkeypatch.setattr(producer, "_run_epoch_cycle", lambda *a: "cycle")
    assert producer._one_pass(args, {}, set(), set()) == route


def test_fallback_cannot_overshoot_screen_floor_and_retries_rejections(tmp_path, monkeypatch):
    selector = import_shipping("nanobody_opt_v2.select_elite_parents", monkeypatch)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.generation_proposal_budget = 8000
    parent = {"sequence": "AA", "parent_cdr3": "family", "parent_validator_duplicate": False,
              "parent_top50_identity_in_basin": False, "layer2_passed": True,
              "parent_cdr_developability_in_range": True}
    producer._atomic_json(tmp_path / "artifacts/agreedfix_historical_parents.json", {"candidates": [parent]})
    destination = tmp_path / "pool.json"
    producer._atomic_json(destination, {"candidates": []})
    monkeypatch.setattr(selector, "_validator_parent_pool", lambda pool, **kw: pool)
    monkeypatch.setattr(selector, "_developability_parent_pool", lambda pool, **kw: pool)
    attempts = []
    def run(command):
        source = json.loads(open(command[command.index("--input") + 1]).read())
        attempts.append([c["sequence"] for c in source["candidates"]])
        producer._atomic_json(producer.Path(command[command.index("--output") + 1]), source)
    monkeypatch.setattr(producer, "_run", run)
    monkeypatch.setattr(producer, "_validator_novel_sequences", lambda **kw: set(attempts[-1]) if len(attempts) > 1 else set())
    producer._historical_child_fallback(args=args, target="TEST", config=tmp_path / "cfg",
                                       hf_archive=tmp_path / "archive", iteration_root=tmp_path,
                                       destination=destination, floor=2)
    assert attempts == [["CA", "DA"], ["EA", "FA"]]
    assert len(json.loads(destination.read_text())["candidates"]) == 2
