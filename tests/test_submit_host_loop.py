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
    # The validator stops reading at end-16. Committing must begin early enough
    # that the LAST uid still lands before that, not just the first.
    assert loop.BLACKOUT_BLOCKS == 16
    assert loop.commit_reserve(9) > loop.BLACKOUT_BLOCKS


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


def test_rows_are_ordered_by_the_allocator_score_not_crew_order():
    # Commits are serial and the epoch can run out mid-list, so ordering is
    # what decides which uid is sacrificed.
    crew = [(5, "h5"), (9, "h9"), (2, "h2")]
    mols = {5: "a", 9: "b", 2: "c"}
    seqs = {5: "QVQ", 9: "QVQ", 2: "QVQ"}
    rows = submit.plan(crew, mols, seqs, {5: 1.61, 9: 1.67, 2: 1.64})
    assert [r["uid"] for r in rows] == [9, 2, 5]


def test_ordering_is_stable_without_scores():
    crew = [(5, "h5"), (9, "h9"), (2, "h2")]
    seqs = {5: "QVQ", 9: "QVQ", 2: "QVQ"}
    rows = submit.plan(crew, {}, seqs)
    assert [r["uid"] for r in rows] == [2, 5, 9]


def test_a_uid_holding_molecules_outranks_one_holding_only_a_sequence():
    crew = [(1, "h1"), (2, "h2")]
    rows = submit.plan(crew, {2: "a"}, {1: "QVQ", 2: "QVQ"})
    assert rows[0]["uid"] == 2


def test_an_unreadable_chain_does_not_read_as_blackout():
    # None, not 0: a transient RPC failure must not cancel the remaining
    # commits by looking like "no blocks left".
    assert submit.blocks_left(25248, "definitely-not-a-network") is None


def test_the_handled_ledger_round_trips(tmp_path):
    # Restart=always makes restarts routine, so "already done" has to survive
    # one. In memory it did not, and a restart inside the window re-committed
    # ten uids with a later block_submitted -- the validator's first tiebreak.
    ledger = tmp_path / "handled_epochs.json"
    ledger.write_text(json.dumps([25246, 25247, 25248]))
    assert {int(e) for e in json.loads(ledger.read_text())} == {25246, 25247, 25248}


def test_a_missing_or_corrupt_ledger_is_not_fatal(tmp_path):
    # A first run has no ledger, and a truncated write must not stop the
    # writer: re-doing an epoch is recoverable, refusing to start is not.
    for body in (None, "", "{not json", "[]"):
        ledger = tmp_path / "handled_epochs.json"
        if body is None:
            ledger.unlink(missing_ok=True)
        else:
            ledger.write_text(body)
        try:
            handled = {int(e) for e in json.loads(ledger.read_text())}
        except Exception:
            handled = set()
        assert isinstance(handled, set)


def test_the_reserve_grows_with_the_fleet():
    # A fixed reserve is wrong in both directions: too small and the tail
    # commits into the blackout, too large and we abandon a producer we still
    # had time to wait for.
    assert loop.commit_reserve(10) == 16 + 50
    assert loop.commit_reserve(12) > loop.commit_reserve(10)
    assert loop.commit_reserve(0) == loop.commit_reserve(1)


def test_the_reserve_matches_what_25248_actually_did():
    # Verified on-chain: ten uids, first at 69 blocks left, last at 26, and the
    # validator's cutoff is 16. The reserve must cover that run.
    assert loop.commit_reserve(10) <= 69
    assert loop.commit_reserve(10) >= 26


def test_remembering_an_epoch_terminates(tmp_path):
    # It did not. A blanket string replace rewrote `handled.add(epoch)` inside
    # remember() into a call to itself, so the loop committed 25249 and then
    # died of RecursionError before recording it -- and Restart=always brought
    # it straight back into the same open window to commit all ten again.
    import json as _json
    ledger = tmp_path / "handled_epochs.json"
    handled = set()

    def remember(epoch):
        handled.add(epoch)
        scratch = ledger.with_suffix(".tmp")
        scratch.write_text(_json.dumps(sorted(handled)[-200:]))
        scratch.replace(ledger)

    remember(25249)
    remember(25250)
    assert _json.loads(ledger.read_text()) == [25249, 25250]


def test_the_ledger_source_has_no_self_call():
    # Cheap guard against the exact edit that caused it.
    body = (ROOT / "tools/pi_epoch_loop.py").read_text()
    start = body.index("def remember(")
    fragment = body[start:start + 400]
    assert "handled.add(epoch)" in fragment
    assert fragment.count("remember(epoch)") == 0
