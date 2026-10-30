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


@pytest.fixture
def reservation(tmp_path, monkeypatch):
    path = tmp_path / 'reservation.json'
    monkeypatch.setattr(sd, 'GPU_RESERVATION_FILE', str(path))
    return path


def write_reservation(path, gpus):
    path.write_text(json.dumps({'schema':'pr100-gpu-reservation-v1',
                               'owner':'pr100-live-controller', 'gpus':gpus}))


def fake_daemon(monkeypatch):
    spawned, scored = [], []
    class Worker:
        def __init__(self, gpu, *_args): spawned.append(gpu)
        def alive(self): return True
        def run(self, args, *_args):
            scored.append(args[1])
            return {'ok':True}
    monkeypatch.setattr(sd, 'ShardWorker', Worker)
    monkeypatch.setattr(sd, 'workers_per_gpu', lambda:1)
    daemon = sd.Daemon('unused')
    daemon._module = object()
    monkeypatch.setattr(daemon, '_rerun_short_shards', lambda *a, **kw:None)
    monkeypatch.setattr(daemon, '_merge', lambda *a, **kw:None)
    return daemon, spawned, scored


def daemon_job(tmp_path, gpus):
    return {'gpus':gpus, 'target':'TEST', 'sequences':['AAAA'],
            'out_path':str(tmp_path/'out.json'), 'shard_root':str(tmp_path/'work'),
            'observation_id':'test', 'batch_size':1, 'subnet_config':{}}


def test_reservation_absent_preserves_legacy_worker_allocation(reservation, monkeypatch):
    daemon, spawned, _ = fake_daemon(monkeypatch)
    daemon.start_workers(['1','5'])
    assert sorted(spawned) == ['1','5']


def test_reserved_start_workers_refuses_whole_request_before_any_spawn(reservation, monkeypatch):
    write_reservation(reservation, [1,2,3,4])
    daemon, spawned, _ = fake_daemon(monkeypatch)
    with pytest.raises(RuntimeError, match='gpu_reserved_for_pr100'):
        daemon.start_workers(['5','1'])
    assert spawned == [] and daemon._workers == {}


def test_reserved_handle_does_not_create_workers_or_output(reservation, monkeypatch, tmp_path):
    write_reservation(reservation, [1,2,3,4])
    daemon, spawned, scored = fake_daemon(monkeypatch)
    with pytest.raises(RuntimeError, match='gpu_reserved_for_pr100'):
        daemon.handle(daemon_job(tmp_path, '1,5'))
    assert spawned == scored == [] and not (tmp_path/'work').exists()


def test_reservation_reread_blocks_reuse_of_already_resident_worker(reservation, monkeypatch, tmp_path):
    daemon, spawned, scored = fake_daemon(monkeypatch)
    assert daemon.handle(daemon_job(tmp_path, '5'))['ok']
    assert spawned == scored == ['5']
    write_reservation(reservation, [5])
    with pytest.raises(RuntimeError, match='gpu_reserved_for_pr100'):
        daemon.handle(daemon_job(tmp_path, '5'))
    assert spawned == scored == ['5']


def test_nonreserved_job_still_scores(reservation, monkeypatch, tmp_path):
    write_reservation(reservation, [1,2,3,4])
    daemon, spawned, scored = fake_daemon(monkeypatch)
    assert daemon.handle(daemon_job(tmp_path, '5'))['ok']
    assert spawned == scored == ['5']


@pytest.mark.parametrize('value', [None, [], {}, {'schema':'bad'},
    {'schema':'pr100-gpu-reservation-v1','owner':'wrong','gpus':[1]},
    {'schema':'pr100-gpu-reservation-v1','owner':'pr100-live-controller','gpus':[True]},
    {'schema':'pr100-gpu-reservation-v1','owner':'pr100-live-controller','gpus':['1']},
    {'schema':'pr100-gpu-reservation-v1','owner':'pr100-live-controller','gpus':[1,1]},
    {'schema':'pr100-gpu-reservation-v1','owner':'pr100-live-controller','gpus':[-1]},
    {'schema':'pr100-gpu-reservation-v1','owner':'pr100-live-controller','gpus':[{}]}])
def test_malformed_reservation_fails_closed(reservation, monkeypatch, value):
    reservation.write_text(json.dumps(value))
    daemon, spawned, _ = fake_daemon(monkeypatch)
    with pytest.raises(RuntimeError, match='gpu_reservation_invalid'):
        daemon.start_workers(['5'])
    assert not spawned


@pytest.mark.parametrize('mode', ['invalid_json','directory','broken_link','oversized'])
def test_unreadable_reservation_fails_closed(reservation, mode):
    if mode=='invalid_json': reservation.write_text('{')
    elif mode=='directory': reservation.mkdir()
    elif mode=='broken_link': reservation.symlink_to(reservation.parent/'missing')
    else: reservation.write_text(' ' * 16385)
    with pytest.raises(RuntimeError, match='gpu_reservation_unreadable'):
        sd.ensure_unreserved_gpus(['5'])


@pytest.mark.parametrize('gpu', ['01', ' 1 ', 1])
def test_reserved_index_aliases_cannot_bypass_policy(reservation, gpu):
    write_reservation(reservation, [1])
    with pytest.raises(RuntimeError, match='gpu_reserved_for_pr100'):
        sd.ensure_unreserved_gpus([gpu])


def test_gpu_uuid_cannot_bypass_index_reservation(reservation):
    write_reservation(reservation, [1])
    with pytest.raises(RuntimeError, match='gpu_reservation_requires_device_indices'):
        sd.ensure_unreserved_gpus(['GPU-unknown'])


def test_daemon_does_not_accept_adapter_lease_exemption(reservation, monkeypatch):
    from nanobody_opt import pr100_live_gpu
    write_reservation(reservation, [1])
    monkeypatch.setattr(pr100_live_gpu, 'inherited_lease_fd', lambda *a:pytest.fail('daemon must never exempt resident workers'))
    daemon, spawned, _ = fake_daemon(monkeypatch)
    with pytest.raises(RuntimeError, match='gpu_reserved_for_pr100'):
        daemon.start_workers(['1'])
    assert spawned == []


def test_model_reuse_repeats_forward_without_caching_results(monkeypatch):
    from types import SimpleNamespace
    from nanobody_opt import resident_scorer as resident
    constructed, forwards = [], []
    class Stage:
        def __init__(self, config, overrides=None): constructed.append(config)
        def forward(self, request):
            forwards.append(request)
            return len(forwards)
    module = SimpleNamespace(ResidentStage=Stage)
    monkeypatch.setattr(resident, '_CACHE', {})
    monkeypatch.setattr(resident, '_INSTALLED', False)
    assert resident.install(module)
    first = module.ResidentStage('same-config', {'steps': 50})
    second = module.ResidentStage('same-config', {'steps': 50})
    assert first is second and len(constructed) == 1
    assert first.forward('draw-one') == 1
    assert second.forward('draw-two') == 2
    assert forwards == ['draw-one', 'draw-two']


def test_repeated_daemon_requests_get_distinct_observation_identity(tmp_path):
    source = tmp_path/'input.json'; source.write_text(json.dumps({'sequences': ['AAAA']}))
    first = sd.job_from_command(_command(source))
    second = sd.job_from_command(_command(source))
    assert first['sequences'] == second['sequences']
    assert first['observation_id'] != second['observation_id']


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


@pytest.mark.parametrize("schema,replicate", [("base-v1", False), ("replicates-v1", False), ("replicates-v1", True)])
def test_master_and_slave_worker_arguments_preserve_their_protocol(schema, replicate):
    job = {"worker_argument_schema": schema, "allow_replicates": replicate,
           "sequences": ["A"] * 7, "_shard_count": 3, "subnet_config": {},
           "batch_size": 2, "observation_id": "test"}
    args = sd.Daemon._shard_args(1, "0", ["A", "A"], "out", job, "work")
    assert len(args) == (16 if schema == "base-v1" else 18)
    if schema == "replicates-v1":
        assert args[-2:] == [replicate, [1, 4]]


def test_replica_flag_reaches_daemon_job(tmp_path):
    source = tmp_path / "seq.json"
    source.write_text(json.dumps({"sequences": ["AAAA", "AAAA"]}))
    job = sd.job_from_command(_command(source) + ["--allow-replicates"])
    assert job["allow_replicates"] is True
    assert job["sequences"] == ["AAAA", "AAAA"]


def test_owner_watch_tolerates_launcher_and_stops_after_daemon_dies():
    import subprocess
    import sys
    import time
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    worker = _real_worker(
        "from nanobody_opt.scorer_daemon import _watch_parent; import time; "
        f"_watch_parent({owner.pid}, 0.05); print('ready', flush=True); time.sleep(30)"
    )
    try:
        assert worker._read_reply(5) == "ready"
        time.sleep(0.15)
        assert worker.proc.poll() is None  # owner is deliberately not getppid()
        owner.terminate()
        owner.wait(timeout=2)
        worker.proc.wait(timeout=2)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)
        worker.close()
