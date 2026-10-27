"""Audit durability and confidentiality without native models or live resources."""
import copy
import asyncio
import errno
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from nanobody_opt import pr100_native_journal as journal
from nanobody_opt import rules


REQ = {"psh_strict_rejection_threshold": 89}
SEQUENCE = "AAA"
SENTINEL = "private-runtime-message-must-never-be-written"


def profile():
    return {"passed": True, "total_cdr_length": 30, "cdr3_length": 12,
            "cdr3_compactness": 1.0, "surface_hydrophobic_patches": 80,
            "positive_charge_patches": 0.5, "negative_charge_patches": 0.5,
            "aggrescan3d": {"status": "ok", "passed": True,
                "sequence_hash": rules.sequence_hash(SEQUENCE),
                "ph_scale": "default-scale", "surface_method": "FreeSASA-2.0.1-NACCESS-LR100",
                "mode": "static-no-minimization", "radius_angstrom": "10.000",
                "input_structure_sha256": "a" * 64, "normalized_structure_sha256": "b" * 64,
                "residue_count": len(SEQUENCE), "a3d_average": "-0.700000000000001",
                "a3d_maximum": "1.234567890123456789"}}


def row():
    return {"sequence": SEQUENCE, "sequence_hash": "not-trusted",
            "source": "fresh_baseline_proposal", "reference_hash": "c" * 64,
            "pr100_annotation": {"annotation_identity": "d" * 64}}


@pytest.fixture
def cheap_context(monkeypatch):
    monkeypatch.setattr(journal, "_context", lambda *args: {
        "schema": journal.SCHEMA, "observational": True, "reusable": False,
        "model_weight_identity": "incomplete", "runtime_identity_status": "partial",
        "input_sha256": "e" * 64})


def records(directory):
    return [json.loads(line) for line in (directory / "native-observations.jsonl").read_text().splitlines()
            if line.strip()]


def test_real_gate_pass_and_decimal_precision_are_preserved():
    original = profile()
    before = copy.deepcopy(original)
    observed = journal.observation(original, REQ, SEQUENCE)
    assert original == before
    assert observed["outcome"] == "passed"
    assert observed["gate_reason"] == rules.developability_reason(original, REQ, sequence=SEQUENCE) == ""
    assert observed["missing_metrics"] == []
    assert observed["aggrescan3d"]["a3d_maximum"] == "1.234567890123456789"
    assert observed["execution_censored"] is False


@pytest.mark.parametrize("changes,reason,outcome", [
    ({"surface_hydrophobic_patches": 90}, "psh_threshold", "rejected"),
    ({"red_flags": [SENTINEL]}, "developability_red_flags", "rejected"),
    ({"passed": False}, "developability_failed", "rejected"),
    ({"error": True, "error_kind": "timeout"}, "developability_unavailable", "inconclusive"),
    ({"surface_hydrophobic_patches": float("nan")}, "developability_unavailable", "inconclusive"),
    ({"aggrescan3d": None}, "aggrescan_unavailable_or_failed", "inconclusive"),
])
def test_rejected_and_censored_outcomes_are_distinct(changes, reason, outcome):
    value = {**profile(), **changes}
    observed = journal.observation(value, REQ, SEQUENCE)
    assert observed["outcome"] == outcome
    assert observed["gate_reason"] == reason
    assert observed["execution_censored"] is (outcome == "inconclusive")
    assert SENTINEL not in json.dumps(observed, allow_nan=False)


def test_unknown_result_and_bad_provenance_are_not_measured_rejections():
    assert journal.observation(None, REQ, SEQUENCE)["outcome"] == "inconclusive"
    value = profile()
    value["aggrescan3d"]["mode"] = SENTINEL
    result = journal.observation(value, REQ, SEQUENCE)
    assert result["gate_reason"] == "aggrescan_provenance_mismatch"
    assert result["outcome"] == "inconclusive"
    assert result["aggrescan3d"]["mode"] == "unknown"
    assert SENTINEL not in json.dumps(result)


@pytest.mark.parametrize("average,maximum", [
    ("-0.49695852943914703", "2.3138981898258155"),  # both caps, live incident
    ("-0.5540617053334963", "1.5485080163280531"),  # average only
    ("-0.7380241848048523", "1.584922997654891"),  # maximum only
    (str(rules.MAX_AVERAGE + Decimal("0.000000000000000001")), str(rules.MAX_PEAK)),
    (str(rules.MAX_AVERAGE), str(rules.MAX_PEAK + Decimal("0.000000000000000001"))),
])
def test_completed_a3d_cap_failure_is_rejected_without_changing_acceptance(
        tmp_path, cheap_context, average, maximum):
    value = profile()
    value["passed"] = False
    value["aggrescan3d"].update(passed=False, a3d_average=average, a3d_maximum=maximum)
    before = copy.deepcopy(value)
    assert rules.developability_reason(value, REQ, sequence=SEQUENCE) == "aggrescan_unavailable_or_failed"

    observed = journal.observation(value, REQ, SEQUENCE)
    assert observed["outcome"] == "rejected"
    assert observed["gate_reason"] == "aggrescan_threshold"
    assert observed["execution_censored"] is False
    assert observed["backend_passed"] is False
    assert observed["aggrescan3d"]["passed"] is False
    assert value == before
    assert rules.developability_reason(value, REQ, sequence=SEQUENCE) == "aggrescan_unavailable_or_failed"

    sink = journal.Journal(tmp_path, {}, REQ)
    assert sink.save(row(), 1, observed)
    saved = records(tmp_path)[-1]["observation"]
    assert saved["outcome"] == "rejected"
    assert saved["gate_reason"] == "aggrescan_threshold"
    assert sink.progress()["outcomes"] == {"passed": 0, "rejected": 1, "inconclusive": 0}


@pytest.mark.parametrize("changes", [
    {"status": "error"}, {"status": "disabled"},
    {"sequence_hash": "f" * 64}, {"residue_count": len(SEQUENCE) + 1},
    {"mode": "unknown"}, {"surface_method": "unknown"}, {"ph_scale": "unknown"},
    {"radius_angstrom": "9"}, {"input_structure_sha256": "invalid"},
    {"normalized_structure_sha256": "invalid"},
    {"a3d_average": None}, {"a3d_maximum": "NaN"}, {"a3d_maximum": "-1"},
])
def test_unverifiable_a3d_failure_stays_inconclusive(changes):
    value = profile()
    value["passed"] = False
    value["aggrescan3d"].update({"passed": False, "a3d_average": "-0.4", **changes})
    observed = journal.observation(value, REQ, SEQUENCE)
    assert observed["outcome"] == "inconclusive"
    assert observed["execution_censored"] is True


@pytest.mark.parametrize("error_kind", ["timeout", "no_profile", "profile_exception"])
def test_a3d_metrics_do_not_hide_execution_errors(error_kind):
    value = {**profile(), "error": True, "error_kind": error_kind}
    value["aggrescan3d"].update(passed=False, a3d_average="-0.4")
    observed = journal.observation(value, REQ, SEQUENCE)
    assert observed["outcome"] == "inconclusive"
    assert observed["error_kind"] == error_kind


def test_false_a3d_flag_with_no_cap_failure_is_not_reclassified():
    value = profile()
    value["passed"] = False
    value["aggrescan3d"].update(passed=False, a3d_average=str(rules.MAX_AVERAGE),
                               a3d_maximum=str(rules.MAX_PEAK))
    observed = journal.observation(value, REQ, SEQUENCE)
    assert observed["outcome"] == "inconclusive"
    assert observed["gate_reason"] == "aggrescan_unavailable_or_failed"


def test_contradictory_error_kind_is_censored_even_with_passed_flag():
    value = {**profile(), "error_kind": "timeout"}
    observed = journal.observation(value, REQ, SEQUENCE)
    assert observed["gate_reason"] == ""
    assert observed["outcome"] == "inconclusive"
    assert observed["error"] is True


def test_overflowing_integer_metric_is_missing_not_an_observer_exception():
    value = {**profile(), "total_cdr_length": 10 ** 400}
    result = journal.observation(value, REQ, SEQUENCE)
    assert "total_cdr_length" in result["missing_metrics"]
    assert result["outcome"] == "inconclusive"


def test_no_free_text_or_nonfinite_numbers_survive_projection():
    value = profile()
    value.update(reason=SENTINEL, error=SENTINEL, error_kind=SENTINEL,
                 flags={"stderr": SENTINEL}, environment={"AUTH": SENTINEL},
                 positive_charge_patches=float("inf"))
    value["aggrescan3d"].update(error=SENTINEL, reason=SENTINEL, a3d_total="NaN",
                               input_structure_sha256=SENTINEL, ph_scale=SENTINEL)
    result = journal.observation(value, REQ, SEQUENCE)
    assert SENTINEL not in json.dumps(result, allow_nan=False)
    assert result["error_kind"] == "unknown"
    assert "positive_charge_patches" in result["missing_metrics"]
    assert "a3d_total" not in result["aggrescan3d"]
    assert result["outcome"] == "inconclusive"


def test_busy_journal_lock_is_reported_without_blocking(tmp_path, cheap_context):
    import fcntl
    import time
    sink = journal.Journal(tmp_path, {}, REQ)
    with sink.path.open('ab') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        assert sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE)) is False
        assert time.monotonic() - started < 1
    assert sink.progress()['status'] == 'degraded'
    assert sink.progress()['observations_durable'] == 0


def test_completed_observation_is_fsynced_before_save_returns(tmp_path, cheap_context, monkeypatch):
    calls = []
    real_fsync = journal.os.fsync
    def fsync(fd):
        calls.append(fd)
        real_fsync(fd)
    monkeypatch.setattr(journal.os, "fsync", fsync)
    sink = journal.Journal(tmp_path, {}, REQ)
    calls.clear()
    assert sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    assert calls
    header, saved = records(tmp_path)
    assert header["record_type"] == "attempt"
    assert saved["sequence_hash"] == rules.sequence_hash(SEQUENCE)
    assert saved["repeat"] == 1 and saved["reference_hash"] == "c" * 64
    assert saved["annotation_identity"] == "d" * 64
    assert saved["context_digest"] == sink.context_digest
    for record in (header, saved):
        digest = record.pop("record_digest")
        assert rules.digest(record) == digest
    progress = sink.progress()
    assert progress["observations_durable"] == 1
    assert progress["outcomes"] == {"passed": 1, "rejected": 0, "inconclusive": 0}
    assert progress["reusable"] is False


def test_retry_is_new_attempt_and_never_reuses_previous_measurements(tmp_path, cheap_context):
    first = journal.Journal(tmp_path, {}, REQ)
    assert first.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    retry = journal.Journal(tmp_path, {}, REQ)
    assert retry.attempt_id != first.attempt_id
    assert retry.progress()["previous_records_audit_only"] == 1
    assert retry.progress()["observations_durable"] == 0
    assert retry.progress()["outcomes"]["passed"] == 0
    assert retry.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    assert retry.progress()["observations_durable"] == 1


def test_distinct_repeats_and_duplicate_callback_accounting(tmp_path, cheap_context):
    sink = journal.Journal(tmp_path, {}, REQ)
    value = journal.observation(profile(), REQ, SEQUENCE)
    assert sink.save(row(), 1, value)
    assert not sink.save(row(), 1, value)
    assert sink.save(row(), 2, value)
    progress = sink.progress()
    assert progress["observations_durable"] == 2
    assert progress["unique_sequences_durable"] == 1
    assert progress["first_pass_outcomes"]["passed"] == 1
    assert progress["repeat_observations_durable"] == 1
    assert progress["errors"] == {"duplicate_observation": 1}


def test_torn_tail_preserves_prefix_and_new_attempt_is_separated(tmp_path, cheap_context):
    first = journal.Journal(tmp_path, {}, REQ)
    assert first.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    path = tmp_path / "native-observations.jsonl"
    prefix = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"record_type":"obser')
    retry = journal.Journal(tmp_path, {}, REQ)
    assert path.read_bytes().startswith(prefix + b'{"record_type":"obser\n')
    assert retry.progress()["previous_records_audit_only"] == 1
    assert retry.progress()["ignored_torn_tail_bytes"] > 0
    assert retry.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    third = journal.Journal(tmp_path, {}, REQ)
    assert third.progress()["previous_records_audit_only"] == 2
    assert third.progress()["invalid_history_records"] == 1
    assert third.progress()["observations_durable"] == 0


def test_corrupt_digest_cannot_be_recovered_as_observation(tmp_path, cheap_context):
    first = journal.Journal(tmp_path, {}, REQ)
    first.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    path = tmp_path / "native-observations.jsonl"
    lines = path.read_text().splitlines()
    value = json.loads(lines[1])
    value["repeat"] = 500
    path.write_text(lines[0] + "\n" + json.dumps(value) + "\n")
    retry = journal.Journal(tmp_path, {}, REQ)
    assert retry.progress()["previous_records_audit_only"] == 0
    assert retry.progress()["invalid_history_records"] == 1


def test_fsync_error_is_counted_without_claiming_durability_or_raising(tmp_path, cheap_context, monkeypatch):
    sink = journal.Journal(tmp_path, {}, REQ)
    def fail(_fd):
        raise OSError(errno.ENOSPC, SENTINEL)
    monkeypatch.setattr(journal.os, "fsync", fail)
    assert not sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    progress = sink.progress()
    assert progress["status"] == "degraded"
    assert progress["observations_seen"] == 1 and progress["observations_durable"] == 0
    assert progress["errno_counts"] == {str(errno.ENOSPC): 1}
    assert SENTINEL not in json.dumps(progress)


def test_runtime_identity_failure_does_not_halt_science(tmp_path, monkeypatch):
    def fail(*_args):
        raise RuntimeError(SENTINEL)
    monkeypatch.setattr(journal, "_context", fail)
    sink = journal.Journal(tmp_path, {}, REQ)
    assert sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    assert sink.progress()["status"] == "degraded"
    assert sink.context["runtime_identity_status"] == "unavailable"
    assert SENTINEL not in sink.path.read_text()


def test_save_reprojects_untrusted_observation_and_row_fields(tmp_path, cheap_context):
    sink = journal.Journal(tmp_path, {}, REQ)
    observed = journal.observation(profile(), REQ, SEQUENCE)
    observed.update(reason=SENTINEL, arbitrary=SENTINEL)
    observed["metrics"][SENTINEL] = 4
    observed["aggrescan3d"]["reason"] = SENTINEL
    candidate = row()
    candidate.update(source=SENTINEL, reference_hash=SENTINEL)
    candidate["pr100_annotation"]["annotation_identity"] = SENTINEL
    assert sink.save(candidate, 1, observed)
    assert SENTINEL not in sink.path.read_text()
    assert SEQUENCE not in sink.path.read_text()


def test_context_binds_input_challenge_requirements_and_source(tmp_path, monkeypatch):
    root = tmp_path / "release"
    root.mkdir()
    monkeypatch.setattr(journal, "ROOT", root)
    monkeypatch.setattr(journal, "SOURCE_FILES", {"adapter": "adapter.py", "worker": "worker.py"})
    (root / "adapter.py").write_text("adapter")
    (root / "worker.py").write_text("worker")
    (root / "build_stamp.json").write_text(json.dumps({"git_sha": "a" * 40}))
    (tmp_path / "input.json").write_text("input-one")
    monkeypatch.setattr(journal.importlib.metadata, "version", lambda _: "1.2.3")
    monkeypatch.setattr(journal.shutil, "which", lambda _: None)
    job = {"source_sha": "a" * 40, "challenge": {"epoch": 25960}}
    context = journal._context(tmp_path, job, REQ)
    assert context["source_revision"] == "a" * 40
    assert context["input_sha256"] == hashlib.sha256(b"input-one").hexdigest()
    assert context["challenge_sha256"] == rules.digest(job["challenge"])
    assert context["requirements_sha256"] == rules.digest(REQ)
    assert context["source_sha256"]["worker"] == hashlib.sha256(b"worker").hexdigest()
    before = rules.digest(context)
    (tmp_path / "input.json").write_text("input-two")
    assert rules.digest(journal._context(tmp_path, job, REQ)) != before
    assert context["model_weight_identity"] == "incomplete" and context["reusable"] is False
    monkeypatch.setattr(journal.importlib.metadata, "version", lambda _: SENTINEL)
    assert SENTINEL not in json.dumps(journal._context(tmp_path, job, REQ))


def test_symlink_journal_is_never_read_or_written(tmp_path, cheap_context):
    other = tmp_path / "unrelated"
    other.write_text(SENTINEL)
    (tmp_path / "native-observations.jsonl").symlink_to(other)
    sink = journal.Journal(tmp_path, {}, REQ)
    assert not sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    assert other.read_text() == SENTINEL
    assert SENTINEL not in json.dumps(sink.progress())


def test_completed_record_survives_abrupt_process_exit(tmp_path):
    # A process exits without finally/close hooks after the first observation;
    # no native model, executor, SSH or production process is involved.
    program = """
import os,sys
from nanobody_opt import pr100_native_journal as j
j._context=lambda *a:{'schema':j.SCHEMA,'observational':True,'reusable':False,'model_weight_identity':'incomplete'}
s=j.Journal(sys.argv[1],{}, {})
assert s.save({'sequence':'AAA'},1,j.observation({'error':True,'error_kind':'timeout'}, {}, 'AAA'))
os._exit(23)
"""
    completed = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=15)
    assert completed.returncode == 23
    saved = records(tmp_path)[1]
    assert saved["observation"]["outcome"] == "inconclusive"
    assert saved["observation"]["error_kind"] == "timeout"
    digest = saved.pop("record_digest")
    assert rules.digest(saved) == digest


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.001)
    assert predicate()


def test_async_initialization_stall_never_blocks_submission_progress_or_close(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def context(*_args):
        entered.set()
        release.wait(5)
        return {"schema": journal.SCHEMA, "observational": True, "reusable": False}
    monkeypatch.setattr(journal, "_context", context)
    start = time.monotonic()
    sink = journal.AsyncJournal(tmp_path, {}, REQ, queue_size=2)
    assert time.monotonic() - start < .25
    try:
        assert entered.wait(1)
        value = journal.observation(profile(), REQ, SEQUENCE)
        start = time.monotonic()
        assert sink.save(row(), 1, value)
        assert sink.save(row(), 2, value)
        assert not sink.save(row(), 3, value)
        for _ in range(25):
            progress = sink.progress()
        assert time.monotonic() - start < .25
        assert progress["observations_pending"] == 2
        assert progress["observations_durable"] == 0
        assert progress["observations_dropped"] == 1
        assert progress["async_errors"] == {"queue_full": 1}
        start = time.monotonic()
        closed = sink.close(timeout=.02)
        assert time.monotonic() - start < .25
        assert closed["observations_pending"] == 2 and closed["writer_alive"] is True
        assert not sink.save(row(), 4, value)
    finally:
        release.set()
        sink.close(timeout=1)
    assert sink.progress()["observations_durable"] == 2
    assert sink.progress()["observations_pending"] == 0


def test_async_stalled_fsync_does_not_block_another_profile(tmp_path, cheap_context, monkeypatch):
    sink = journal.AsyncJournal(tmp_path, {}, REQ)
    wait_until(lambda: sink.progress()["writer_ready"])
    entered, release = threading.Event(), threading.Event()
    real_fsync = journal.os.fsync
    def fsync(fd):
        entered.set()
        release.wait(5)
        return real_fsync(fd)
    monkeypatch.setattr(journal.os, "fsync", fsync)
    value = journal.observation(profile(), REQ, SEQUENCE)
    try:
        assert sink.save(row(), 1, value)
        assert entered.wait(1)
        async def other_profile():
            await asyncio.sleep(0)
            assert sink.save(row(), 2, value)
            return "profile-completed"
        start = time.monotonic()
        assert asyncio.run(other_profile()) == "profile-completed"
        progress = sink.progress()
        assert time.monotonic() - start < .25
        assert progress["observations_durable"] == 0
        assert progress["observations_pending"] == 2
        start = time.monotonic()
        assert sink.close(timeout=.02)["observations_pending"] == 2
        assert time.monotonic() - start < .25
    finally:
        release.set()
        sink.close(timeout=1)
    assert sink.progress()["observations_durable"] == 2
    assert len(records(tmp_path)) == 3


def test_async_snapshot_is_separate_allowlisted_and_detached_from_caller(tmp_path, cheap_context):
    baseline = b'{"status":"running","completed":0}'
    (tmp_path / "result.json").write_bytes(baseline)
    sink = journal.AsyncJournal(tmp_path, {}, REQ)
    observed = journal.observation(profile(), REQ, SEQUENCE)
    snapshot = {"status": "running", "input_count": 3, "completed": 1, "not_started": 1,
                "not_completed": 2, "in_flight": 1, "native_classification_complete": False,
                "measurement_outcomes": {"passed": 1, "rejected": 0, "inconclusive": 0,
                                         SENTINEL: 1}, "observation_errors": 0, "reason": SENTINEL}
    candidate = row()
    assert sink.save(candidate, 1, observed, progress=snapshot)
    snapshot["completed"] = 999
    observed["metrics"]["surface_hydrophobic_patches"] = float("nan")
    candidate["source"] = SENTINEL
    progress = sink.close(timeout=1)
    assert progress["observations_durable"] == 1 and progress["observations_pending"] == 0
    assert progress["writer_done"] is True
    assert (tmp_path / "result.json").read_bytes() == baseline
    saved = json.loads((tmp_path / "native-progress.json").read_text())
    assert saved["completed"] == 1 and saved["not_started"] == 1
    assert saved["not_completed"] == 2 and saved["in_flight"] == 1
    assert saved["native_classification_complete"] is False
    assert saved["observation_journal"]["observations_durable"] == 1
    assert SENTINEL not in json.dumps(saved)
    assert records(tmp_path)[1]["observation"]["metrics"]["surface_hydrophobic_patches"] == 80
    assert not sink.save(row(), 2, observed)
    assert sink.progress()["observations_dropped"] == 1


def test_async_write_failure_is_explicit_and_does_not_strand_pending_counts(tmp_path, cheap_context, monkeypatch):
    def fail(*_args):
        raise OSError(errno.ENOSPC, SENTINEL)
    monkeypatch.setattr(journal.Journal, "save", fail)
    sink = journal.AsyncJournal(tmp_path, {}, REQ)
    assert sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    result = sink.close(timeout=1)
    assert result["observations_accepted"] == 1
    assert result["observations_durable"] == 0 and result["observations_pending"] == 0
    assert result["observations_write_failed"] == 1
    assert result["async_errors"] == {"save": 1}
    assert result["async_errno_counts"] == {str(errno.ENOSPC): 1}
    assert SENTINEL not in json.dumps(result)


def test_async_initialization_exception_accounts_for_already_queued_records(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def fail(*_args):
        entered.set()
        release.wait(5)
        raise OSError(errno.EIO, SENTINEL)
    monkeypatch.setattr(journal, "Journal", fail)
    sink = journal.AsyncJournal(tmp_path, {}, REQ)
    try:
        assert entered.wait(1)
        assert sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
    finally:
        release.set()
        result = sink.close(timeout=1)
    assert result["observations_pending"] == 0 and result["observations_write_failed"] == 1
    assert result["observations_durable"] == 0 and result["writer_done"] is True
    assert result["async_errors"] == {"initialization": 1}
    assert SENTINEL not in json.dumps(result)


def test_async_submission_contention_is_fail_open_and_progress_lock_free(tmp_path, cheap_context):
    sink = journal.AsyncJournal(tmp_path, {}, REQ)
    sink._submit_lock.acquire()
    try:
        start = time.monotonic()
        assert not sink.save(row(), 1, journal.observation(profile(), REQ, SEQUENCE))
        result = sink.close(timeout=0)
        assert time.monotonic() - start < .25
        assert result["observations_dropped"] == 1
        assert result["async_errors"] == {"submission_busy": 1}
    finally:
        sink._submit_lock.release()
        sink.close(timeout=1)


@pytest.mark.parametrize("timeout", [-1, 1.01, float("nan"), True])
def test_async_close_timeout_must_be_bounded(tmp_path, cheap_context, timeout):
    sink = journal.AsyncJournal(tmp_path, {}, REQ)
    try:
        with pytest.raises(ValueError):
            sink.close(timeout=timeout)
    finally:
        sink.close(timeout=1)


def test_async_stalled_daemon_does_not_delay_normal_process_exit(tmp_path):
    program = """
import sys, threading
from nanobody_opt import pr100_native_journal as j
j._context=lambda *a: threading.Event().wait()
s=j.AsyncJournal(sys.argv[1],{}, {})
assert s.save({'sequence':'AAA'},1,j.observation({'error':True}, {}, 'AAA'))
assert s.close(timeout=.01)['observations_pending']==1
sys.exit(23)
"""
    completed = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=3)
    assert completed.returncode == 23


def test_async_acknowledged_record_survives_abrupt_process_exit(tmp_path):
    program = """
import os,sys
from nanobody_opt import pr100_native_journal as j
j._context=lambda *a:{'schema':j.SCHEMA,'observational':True,'reusable':False}
s=j.AsyncJournal(sys.argv[1],{}, {})
assert s.save({'sequence':'AAA'},1,j.observation({'error':True,'error_kind':'timeout'}, {}, 'AAA'))
assert s.close(timeout=1)['observations_durable']==1
os._exit(23)
"""
    completed = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=3)
    assert completed.returncode == 23
    saved = records(tmp_path)[1]
    assert saved["observation"]["outcome"] == "inconclusive"
    digest = saved.pop("record_digest")
    assert rules.digest(saved) == digest
