"""Regressions for unattended dashboard freshness and truthful failures."""
import json
from pathlib import Path
import subprocess

import pytest

from nanobody_dash import compound, pi_refresh as pi, publish_vercel as publisher
from tools import build_multipass_comparison as builder


@pytest.fixture
def paths(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "public").mkdir()
    monkeypatch.setattr(pi, "ROOT", tmp_path)
    monkeypatch.setattr(pi, "DATA", tmp_path / "data")
    monkeypatch.setattr(pi, "COMPARISON", tmp_path / "comparison.json")
    monkeypatch.setenv("NOVA_DASH_LOCAL_PI", "1")
    return tmp_path


def test_recent_export_publishes_newest_before_older_failure(tmp_path, monkeypatch):
    out = tmp_path / "comparison.json"
    out.write_text(json.dumps({"25868": {"method": {}}}))
    called = []
    def replay(epoch):
        called.append(epoch)
        if epoch == 25873:
            raise TimeoutError()
        return {"method": {}}
    monkeypatch.setattr(builder, "rescale_epoch", replay)
    with pytest.raises(TimeoutError):
        builder.build(out, lo=None, hi=25874, recent=2)
    assert called == [25874, 25873]
    assert set(json.loads(out.read_text())) == {"25868", "25874"}


@pytest.mark.parametrize("body", [{}, [], {"25870": {"method": {}}}, {"25875": {}}])
def test_invalid_or_regressed_download_preserves_live(paths, body):
    before = {"25874": {"method": {}}}
    pi.COMPARISON.write_text(json.dumps(before))
    pending = paths / "download"
    pending.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        pi.accept_comparison(pending)
    assert json.loads(pi.COMPARISON.read_text()) == before


def test_partial_export_timeout_still_pulls_progress(paths, monkeypatch):
    monkeypatch.setattr(pi, "master_endpoint", lambda: ("ssh9.vast.ai", 10616))
    monkeypatch.setattr(pi, "latest_leaderboard", lambda: 25874)
    def run(command, timeout):
        if command[0] == "ssh":
            assert "timeout --kill-after" in command[-1]
            raise subprocess.TimeoutExpired(command, timeout)
        Path(command[-1]).write_text(json.dumps({"25874": {"method": {}}}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(pi, "run", run)
    result = pi.multipass()
    assert result["export_returncode"] == 124
    assert result["ok"] is False
    assert pi.latest_multipass(pi.read_json(pi.COMPARISON)) == 25874


def test_fast_refresh_renders_despite_collection_timeout(paths, monkeypatch):
    called = []
    def module(name, timeout, *args):
        called.append(name)
        if name.endswith("collect"):
            raise subprocess.TimeoutExpired(name, timeout)
        (paths / "public/index.html.next").write_text("new dashboard")
        (paths / "public/version.json.next").write_text('{}')
        return True
    monkeypatch.setattr(pi, "module", module)
    result = pi.refresh()
    assert result["built"] is True and result["ok"] is False
    assert called == ["nanobody_dash.collect", "nanobody_dash.build"]
    assert (paths / "public/index.html").read_text() == "new dashboard"


def test_publish_health_reports_failure_not_attempt(paths, monkeypatch):
    monkeypatch.setattr(publisher, "publish", lambda: True)
    monkeypatch.setattr(publisher, "_state", lambda: {"ok": False})
    assert pi.publish() == {"ok": False, "attempted": True,
                            "published_at": None, "reason": "vercel_deploy_failed"}


def test_local_publish_never_uses_ssh_or_exposes_cli_output(paths, monkeypatch, capsys):
    dest = paths / "vercel"
    dest.mkdir()
    (dest / ".vercel").mkdir()
    (paths / "public/index.html").write_text("new dashboard")
    monkeypatch.setattr(publisher, "ROOT", paths)
    monkeypatch.setattr(publisher, "PI_PUBLISH_DIR", str(dest))
    def run(command, **kwargs):
        assert command[:3] == ["npx", "--no-install", "vercel@65.1.0"]
        assert kwargs["capture_output"]
        return subprocess.CompletedProcess(command, 1, "PRIVATE_RESPONSE", "PRIVATE_DIAGNOSTIC")
    monkeypatch.setattr(publisher.subprocess, "run", run)
    assert publisher._deploy_local() is False
    assert "PRIVATE" not in capsys.readouterr().out
    assert (dest / ".vercel").is_dir()


def test_failed_deploy_keeps_previous_success_hash_and_retries(paths, monkeypatch):
    live = paths / "public/index.html"
    live.write_text("new")
    state = paths / "data/publish.json"
    state.write_text(json.dumps({"ok": True, "content_hash": "old", "published_at": 12}))
    monkeypatch.setattr(publisher, "LIVE", live)
    monkeypatch.setattr(publisher, "STATE", state)
    monkeypatch.setattr(publisher, "_deploy_local", lambda: False)
    assert publisher.publish(force=True)
    assert publisher._state()["content_hash"] == "old"
    assert publisher._state()["published_at"] == 12
    assert publisher._state()["ok"] is False


def test_compound_pi_reads_are_local(paths, monkeypatch):
    def run(command, **kwargs):
        assert command == ["cat", "/opt/nova-submit/loop.log"]
        return subprocess.CompletedProcess(command, 0, "data", "")
    monkeypatch.setattr(compound.subprocess, "run", run)
    assert compound._ssh(compound.PI_HOST, "cat", "/opt/nova-submit/loop.log") == "data"


def test_health_failure_preserves_last_success_without_exception_text(paths, monkeypatch, capsys):
    health = paths / "data/dashboard_refresh_health.json"
    health.write_text(json.dumps({"last_success_at": 123}))
    def fail():
        raise RuntimeError("PRIVATE_DIAGNOSTIC")
    monkeypatch.setattr(pi, "refresh", fail)
    monkeypatch.setattr(pi.sys, "argv", ["pi_refresh", "refresh"])
    assert pi.main() == 1
    assert pi.read_json(health)["last_success_at"] == 123
    assert pi.read_json(health)["reason"] == "RuntimeError"
    assert "PRIVATE" not in capsys.readouterr().out


def test_ui_distinguishes_export_from_complete_measurement(paths, monkeypatch):
    pi.COMPARISON.write_text(json.dumps({
        "25872": {"ours": {"mean": 849}}, "25874": {"ours": {"mean": None}}}))
    assert "export 25874" in pi.status_text()
    assert "measured fleet 25872" in pi.status_text()


def test_dashboard_is_in_tested_release_inventory():
    from scripts.release_manifest import SOURCE_PATHS, runtime_file
    assert "nanobody_dash" in SOURCE_PATHS
    assert runtime_file("nanobody_dash/pi_refresh.py")


def test_timers_use_independent_units_without_mac_paths():
    root = Path(__file__).resolve().parents[1]
    for job in ("refresh", "publish", "multipass", "money", "compound", "proxy"):
        timer = (root / f"deploy/dashboard/nova-dashboard-{job}.timer").read_text()
        assert f"Unit=nova-dashboard@{job}.service" in timer
    service = (root / "deploy/dashboard/nova-dashboard@.service").read_text()
    assert "KillMode=control-group" in service
    assert "/Users/" not in service
