import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from nanobody_opt import publication_ledger as ledger
from nanobody_dash.submission_outcomes import export
from tools import publish_current_nanobody as publisher


def fixture(tmp_path):
    raw = tmp_path / "published.txt"
    raw.write_text("UNUSED\nACTUAL\nPOSSIBLE\nUNKNOWN\n")
    history = tmp_path / "history"
    ledger.save(history / "epoch10.json", {"epoch": 10, "target": "TEST", "submissions": [
        {"sequence": s} for s in ["UNUSED", "ACTUAL", "POSSIBLE"]]})
    proof = tmp_path / "outcomes.json"
    ledger.save(proof, {"schema": ledger.SCHEMA, "target": "TEST", "generated_at": 1000,
                       "covered_epochs": [10, 11], "possibly_submitted_hashes": [
                           ledger.identity("ACTUAL"), ledger.identity("POSSIBLE")]})
    kwargs = dict(epoch=12, target="TEST", history=history, outcomes=proof,
                  output=tmp_path / "effective.txt", now=1001)
    return raw, kwargs


def test_unused_publication_is_released_but_actual_possible_and_unknown_are_retained(tmp_path):
    raw, kwargs = fixture(tmp_path)
    before = raw.read_bytes()
    result = ledger.effective(raw, **kwargs)
    assert result.read_text().splitlines() == ["ACTUAL", "POSSIBLE", "UNKNOWN"]
    assert raw.read_bytes() == before
    assert json.loads(result.with_suffix(".receipt.json").read_text())["released_unused"] == 1


@pytest.mark.parametrize("change", ["stale", "future", "target", "gap", "missing"])
def test_missing_or_incomplete_outcome_evidence_cannot_release_candidates(tmp_path, change):
    raw, kwargs = fixture(tmp_path)
    proof = ledger.read(kwargs["outcomes"])
    if change == "stale":proof["generated_at"] = 1
    if change == "future":proof["generated_at"] = 2000
    if change == "target":proof["target"] = "OTHER"
    if change == "gap":proof["covered_epochs"] = [10, 12];kwargs["epoch"] = 13
    if change == "missing":proof = {}
    ledger.save(kwargs["outcomes"], proof)
    result = ledger.effective(raw, **kwargs)
    assert result.read_bytes() == raw.read_bytes()


def test_republished_candidate_waits_for_its_new_epoch_outcome(tmp_path):
    raw, kwargs = fixture(tmp_path)
    assert "UNUSED" not in ledger.effective(raw, **kwargs).read_text()
    ledger.record(raw, epoch=12, target="TEST", sequences=["UNUSED"])
    assert "UNUSED" in ledger.effective(raw, **{**kwargs, "epoch": 13}).read_text()
    proof = ledger.read(kwargs["outcomes"])
    proof["covered_epochs"].append(12)
    proof["possibly_submitted_hashes"].append(ledger.identity("UNUSED"))
    ledger.save(kwargs["outcomes"], proof)
    assert "UNUSED" in ledger.effective(raw, **{**kwargs, "epoch": 13}).read_text()


@pytest.mark.parametrize("events", [[], {"hash": "unknown"}])
def test_malformed_event_history_retains_exclusions(tmp_path, events):
    raw, kwargs = fixture(tmp_path)
    ledger.save(raw.with_suffix(".publication_events.json"), {"TEST": events})
    assert ledger.effective(raw, **kwargs) == raw


def test_durable_event_is_written_before_the_artifact_becomes_visible(tmp_path, monkeypatch):
    raw, _ = fixture(tmp_path)
    args = SimpleNamespace(pi_handoff=True, ledger=raw, artifact=tmp_path / "artifact.json")
    artifact = {"epoch": 12, "target": "TEST", "submissions": [{"sequence": "NEW"}],
                "sequence_pool": ["NEW", "SPARE"]}
    calls = []
    def publish(path, value):
        events = ledger.read(raw.with_suffix(".publication_events.json"))["TEST"]
        assert events[ledger.identity("NEW")] == events[ledger.identity("SPARE")] == 12
        assert {"NEW", "SPARE"} <= set(raw.read_text().splitlines())
        calls.append(value)
    monkeypatch.setattr(publisher, "_atomic_json", publish)
    publisher._write_submission_artifact(args, artifact)
    assert calls == [artifact]


def test_current_epoch_spares_remain_selectable_but_do_not_leak_into_next_epoch(tmp_path):
    raw = tmp_path / "published.txt"
    raw.write_text("ASSIGNED\nSPARE\nOLD\n")
    artifact = {"epoch": 12, "submissions": [{"sequence": "ASSIGNED"}],
                "sequence_pool": ["ASSIGNED", "SPARE"]}
    current = publisher.fleet_module.write_exclusion_file(
        tmp_path / "current.txt", raw, publisher._live_publication_pool(artifact, 12))
    assert current.read_text().splitlines() == ["OLD"]
    following = publisher.fleet_module.write_exclusion_file(
        tmp_path / "following.txt", raw, publisher._live_publication_pool(artifact, 13))
    assert set(following.read_text().splitlines()) == {"ASSIGNED", "SPARE", "OLD"}


def test_pi_export_protects_entire_inputs_and_requires_finalized_archives(tmp_path):
    database = tmp_path / "dashboard.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE entries(epoch INTEGER,sequence TEXT)")
        db.executemany("INSERT INTO entries VALUES(?,?)", [(10, "ACTUAL"), (11, "OTHER")])
    archive = tmp_path / "archive"
    ledger.save(archive / "epoch10/nanobody_result.json", {"epoch": 10, "target": "TEST",
        "submissions": [{"sequence": "ACTUAL"}], "sequence_pool": ["ACTUAL", "POSSIBLE"]})
    ledger.save(archive / "epoch12/nanobody_result.json", {"epoch": 12, "target": "TEST",
        "submissions": [{"sequence": "PENDING"}], "sequence_pool": ["PENDING"]})
    output = tmp_path / "proof.json"
    export(database, archive, output, target="TEST", now=1000)
    proof = ledger.read(output)
    assert proof["covered_epochs"] == [10]
    assert set(proof["possibly_submitted_hashes"]) == {
        ledger.identity(s) for s in ["ACTUAL", "OTHER", "POSSIBLE", "PENDING"]}
    assert "UNUSED" not in output.read_text()


def test_pi_job_pushes_atomically_and_does_not_promote_a_failed_transfer(tmp_path, monkeypatch):
    from nanobody_dash import pi_refresh, submission_outcomes
    monkeypatch.setattr(pi_refresh, "DATA", tmp_path)
    monkeypatch.setattr(submission_outcomes, "export", lambda *a, **k: None)
    commands = []
    monkeypatch.setattr(pi_refresh, "run", lambda cmd, timeout: commands.append(cmd) or SimpleNamespace(returncode=0))
    assert pi_refresh.submission_outcomes("test-host", 22, ["-o", "BatchMode=yes"])
    assert commands[0][:2] == ["scp", "-O"]
    assert commands[-1][-1].startswith("mv /root/artifacts/pi_submission_outcomes.json.next ")
    commands.clear()
    monkeypatch.setattr(pi_refresh, "run", lambda cmd, timeout: commands.append(cmd) or SimpleNamespace(returncode=1))
    assert not pi_refresh.submission_outcomes("test-host", 22, [])
    assert len(commands) == 1


def test_producer_uses_the_same_effective_exclusions_as_publication(tmp_path, monkeypatch):
    from nanobody_opt_v2 import run_live_producer as producer
    raw, kwargs = fixture(tmp_path)
    args = producer._parser().parse_args([])
    args.runtime_root = tmp_path
    args.pi_handoff = True
    args.once = True
    args.epoch_cycle_start = 12
    args.ledger = raw
    state = {"epoch": 12, "nanobody_target": "TEST"}
    effective = ledger.effective(raw, **kwargs)
    monkeypatch.setattr(producer, "_chain_state", lambda *a: state)
    monkeypatch.setattr(ledger, "effective", lambda *a, **k: effective)
    monkeypatch.setattr(producer, "maintain_runtime_storage", lambda *a, **k: {})
    seen = []
    monkeypatch.setattr(producer, "_run_epoch_cycle", lambda a, s: seen.append(a.ledger) or producer.PRODUCER_ADVANCED)
    producer._one_pass(args, {}, set(), set())
    assert seen == [effective]
    assert args.ledger == raw
