import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "producer_guard_under_test",
        ROOT / "deploy/slot_host/ensure_vast_running.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load()


def test_running_producer_is_a_noop(monkeypatch):
    monkeypatch.setattr(guard, "_instances", lambda _binary: [
        {"id": 7, "label": "prod", "actual_status": "running"}
    ])
    monkeypatch.setattr(
        guard.subprocess, "run",
        lambda *a, **k: pytest.fail("a running producer must not be restarted"),
    )

    assert guard.ensure_running("vastai", "prod") == {
        "event": "producer_healthy", "id": 7, "state": "running"
    }


def test_exited_producer_is_restarted(monkeypatch):
    observations = iter([
        [{"id": 7, "label": "prod", "actual_status": "exited"}],
        [{"id": 7, "label": "prod", "actual_status": "running"}],
    ])
    monkeypatch.setattr(guard, "_instances", lambda _binary: next(observations))
    calls = []
    monkeypatch.setattr(
        guard.subprocess, "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )

    result = guard.ensure_running("vastai", "prod", wait_seconds=0)

    assert calls == [["vastai", "start", "instance", "7"]]
    assert result["event"] == "producer_restarted"


def test_missing_or_ambiguous_label_never_rents_or_starts(monkeypatch):
    for rows in ([], [
        {"id": 7, "label": "prod", "actual_status": "exited"},
        {"id": 8, "label": "prod", "actual_status": "running"},
    ]):
        monkeypatch.setattr(guard, "_instances", lambda _binary, rows=rows: rows)
        monkeypatch.setattr(
            guard.subprocess, "run",
            lambda *a, **k: pytest.fail("an ambiguous guard must fail closed"),
        )
        with pytest.raises(RuntimeError):
            guard.ensure_running("vastai", "prod", wait_seconds=0)


def test_direct_endpoint_precedes_proxy():
    assert guard.producer_endpoints({
        "public_ipaddr": "203.0.113.10",
        "ports": {"22/tcp": [{"HostPort": "41222"}]},
        "ssh_host": "ssh.example.invalid",
        "ssh_port": 18000,
    }) == [
        ("direct", "203.0.113.10", 41222),
        ("proxy", "ssh.example.invalid", 18000),
    ]


def test_disable_marker_suppresses_recovery(tmp_path, capsys):
    marker = tmp_path / "disabled"
    marker.touch()

    assert guard.main([
        "--vastai", "vastai", "--label", "prod",
        "--disable-marker", str(marker),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "producer_guard_disabled"
