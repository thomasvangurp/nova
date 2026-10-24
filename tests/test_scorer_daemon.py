"""job_from_command's --sequences parsing must match fast_score_batch.py's own.

Regression for the 2026-10-10 bug: job_from_command used to hand
{"sequence": ...} dict items straight through as "sequences" instead of
extracting the string, same as fast_score_batch.py's own main() does. The
worker's _deduplicate_nanobodies then tried to hash a dict, raised inside
run_chunk, logged "continuing" and silently dropped the sequence -- the
daemon answered "ok" with nothing scored, and _merge correctly reported
"MERGE LOST SEQUENCES: submitted N, merged 0". Every real caller
(score_boltzgen_waves._wave_command, the live race) writes dict items, so
this made every daemon-routed round fail and silently fall back to the slow
subprocess path -- the daemon provided zero speedup despite being enabled.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobody_opt import scorer_daemon as sd


def _command(sequences_path: Path, **extra: str) -> list[str]:
    command = [
        "/root/release_python.sh",
        "/root/fastscore/pinned-x/fast_score_batch.py",
        "--sequences", str(sequences_path),
        "--config", "/root/config.yaml",
        "--target", "P20809",
        "--gpus", "0,1",
        "--out", "/root/out.json",
    ]
    for flag, value in extra.items():
        command.extend([flag, value])
    return command


def test_job_from_command_extracts_dict_items(tmp_path: Path) -> None:
    """{"sequences": [{"sequence": "..."}]} -- score_boltzgen_waves's own shape."""
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": [
        {"sequence": "AAAA"}, {"sequence": "CCCC"},
    ]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA", "CCCC"]
    assert all(isinstance(s, str) for s in job["sequences"])


def test_job_from_command_accepts_bare_strings(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": ["AAAA", "CCCC"]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA", "CCCC"]


def test_job_from_command_accepts_bare_list(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps([{"sequence": "AAAA"}]))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA"]


def test_job_from_command_accepts_cases_key(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"cases": [{"sequence": "AAAA"}]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA"]


def test_job_from_command_rejects_empty_sequences(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": []}))
    with pytest.raises(ValueError, match="no sequences parsed"):
        sd.job_from_command(_command(path))


def test_job_from_command_drops_empty_items(tmp_path: Path) -> None:
    """A blank string item is dropped, matching fast_score_batch.py's own
    `isinstance(s, str) and s` filter (a dict item missing "sequence" is a
    KeyError in both, by design -- an outer CLI's argv must fail loudly, not
    parse into a plausible-looking job with silently wrong data)."""
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": [{"sequence": "AAAA"}, ""]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA"]


def _real_worker(code):
    import subprocess
    import sys
    import threading
    worker = sd.ShardWorker.__new__(sd.ShardWorker)
    worker.gpu = "test"
    worker._pipe = threading.Lock()
    worker.proc = subprocess.Popen(
        [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        start_new_session=True,
    )
    return worker


def test_dead_worker_cannot_hide_behind_child_holding_reply_pipe():
    import time
    worker = _real_worker(
        "import subprocess,sys,os; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "sys.stdout.write('{'); sys.stdout.flush(); os._exit(1)"
    )
    start = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="exited"):
            worker._read_reply(3)
        assert time.monotonic() - start < 2
    finally:
        worker.close()


def test_disconnect_cancels_live_worker_and_releases_its_process_group():
    import socket
    worker = _real_worker("import time; time.sleep(30)")
    server, client = socket.socketpair()
    try:
        assert not sd._client_disconnected(server)
        client.close()
        with pytest.raises(RuntimeError, match="client disconnected"):
            worker.run([], "TEST", lambda: sd._client_disconnected(server))
        assert worker.proc.poll() is not None
    finally:
        server.close()
        client.close()
        worker.close()


def test_partial_reply_has_a_deadline():
    worker = _real_worker("import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(30)")
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            worker._read_reply(0.3)
    finally:
        worker.close()


def test_session_wires_disconnect_detection_into_scoring(monkeypatch):
    import socket
    server, client = socket.socketpair()
    daemon = sd.Daemon("unused")
    def handle(job, cancelled):
        assert not cancelled()
        client.close()
        assert cancelled()
        return {"ok": True}
    monkeypatch.setattr(daemon, "handle", handle)
    sd._send(client, {"op": "score", "job": {}})
    daemon._session(server)
    assert daemon.served == 1


def test_worker_reports_system_exit_and_remains_available(tmp_path):
    import subprocess
    import sys
    scorer = tmp_path / "fake_scorer.py"
    scorer.write_text("def _worker(args):\n    raise SystemExit(7)\n")
    (tmp_path / "resident_scorer.py").write_text(
        "def bootstrap(path): pass\ndef install_all(module): pass\n")
    (tmp_path / "chunk_reuse.py").write_text(
        "ENABLED = {}\ndef install_all(module): return {}\n")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from nanobody_opt.scorer_daemon import shard_worker_main; "
        "raise SystemExit(shard_worker_main(sys.argv[2]))"
    )
    request = json.dumps({"args": [0, 1, 2, {}], "target": "TEST"}) + "\n"
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(scorer)],
                            input=request * 2, capture_output=True, text=True, timeout=10)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0
    assert replies[0] == {"ready": True}
    assert len(replies) == 3
    assert all(not reply["ok"] and "SystemExit" in reply["error"] for reply in replies[1:])
