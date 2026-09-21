"""The submit host must be able to run the whole epoch itself.

For five epochs (25243-25247) the fleet committed nothing. Three separate
causes -- a stale producer artifact, the loop process dying, and the loop being
a foreground job on a laptop -- all produced the same silent outcome, because
the writer had no fallback and no supervisor. These tests pin the seams that
now make each of those recoverable.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loop = _load("pi_epoch_loop_under_test", ROOT / "tools/pi_epoch_loop.py")
submit = _load("pi_submit_under_test", ROOT / "deploy/submit_host/pi_submit.py")


def _artifact(epoch, uids=(1, 2)):
    return dict(epoch=epoch,
                submissions=[dict(uid=u, sequence="QVQ%d" % u) for u in uids])


def test_a_stale_artifact_is_still_refused_by_default():
    # Preferring fresh remains the default: staleness must be an explicit,
    # last-resort decision by the caller, never a silent downgrade.
    with pytest.raises(SystemExit):
        submit.sequences_by_uid(_artifact(25246), 25247)


def test_a_stale_artifact_is_usable_when_the_caller_allows_it():
    got = submit.sequences_by_uid(_artifact(25246), 25247, allow_stale=True)
    assert got == {1: "QVQ1", 2: "QVQ2"}


def test_a_fresh_artifact_needs_no_permission():
    assert submit.sequences_by_uid(_artifact(25247), 25247) == {1: "QVQ1", 2: "QVQ2"}


def test_uids_without_a_sequence_are_dropped_not_committed_blank():
    artifact = dict(epoch=7, submissions=[dict(uid=1, sequence=""),
                                          dict(uid=2, sequence="QVQ2"),
                                          dict(uid=3)])
    assert submit.sequences_by_uid(artifact, 7) == {2: "QVQ2"}


def test_the_commit_reserve_leaves_room_before_the_blackout():
    # The validator stops reading at end-16. Committing must start earlier than
    # that, with enough blocks left to place one extrinsic per uid per block.
    assert loop.COMMIT_RESERVE > 16
    assert loop.COMMIT_RESERVE - 16 >= len("123456789")


def test_local_dispatch_does_not_pass_the_ssh_only_flag(monkeypatch):
    # `-n` is an ssh flag. Passing it to `bash -lc` would silently change the
    # command, so the local branch must not reuse the remote argv shape.
    seen = {}
    monkeypatch.setattr(loop.subprocess, "run",
                        lambda argv, **kw: seen.setdefault("argv", argv))
    loop.run(loop.LOCAL, "echo hi")
    assert seen["argv"] == ["bash", "-lc", "echo hi"]
    assert "-n" not in seen["argv"]


def test_remote_dispatch_still_detaches_stdin(monkeypatch):
    # Without -n an ssh child inherits the loop's stdin and can swallow it,
    # which has hung this loop before.
    seen = {}
    monkeypatch.setattr(loop.subprocess, "run",
                        lambda argv, **kw: seen.setdefault("argv", argv))
    loop.run(["ssh", "host"], "echo hi")
    assert seen["argv"] == ["ssh", "host", "-n", "echo hi"]


def test_the_producer_probe_reads_the_epoch_it_pulled(monkeypatch, tmp_path):
    artifact = Path("/opt/nova-submit/nanobody_result.json")
    monkeypatch.setattr(loop, "run",
                        lambda where, cmd, timeout=240: type("R", (), {"returncode": 0})())
    if not artifact.is_file():
        pytest.skip("probe reads the submit host's own artifact path")
    assert isinstance(loop.producer_epoch_here(None, 0, 3), int)


def test_a_failed_pull_reports_unknown_rather_than_zero(monkeypatch):
    # Returning 0 would compare as "older than this epoch" and look like a
    # stale producer; None is the honest answer and keeps the loop waiting.
    monkeypatch.setattr(loop, "run",
                        lambda where, cmd, timeout=240: type("R", (), {"returncode": 1})())
    assert loop.producer_epoch_here(None, 25248, 3) is None
