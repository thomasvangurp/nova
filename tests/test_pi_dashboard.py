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
    monkeypatch.setenv("NOVA_DASH_MASTER_RELEASE_DIR", "/root/releases/" + "a" * 40)
    monkeypatch.setattr(pi, 'refresh_api_status', lambda: None)
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


def test_pi_aws_policy_grants_only_cost_reads():
    root = Path(__file__).resolve().parents[1]
    policy = json.loads((root / "deploy/dashboard/aws-cost-reader-policy.json").read_text())
    assert policy["Statement"] == [{
        "Sid": "ReadDashboardCostHistoryAndForecast", "Effect": "Allow",
        "Action": ["ce:GetCostAndUsage", "ce:GetCostForecast"], "Resource": "*"}]


def test_only_money_job_receives_encrypted_aws_credential():
    root = Path(__file__).resolve().parents[1]
    unit = (root / "deploy/dashboard/nova-dashboard@.service").read_text()
    dropin = (root / "deploy/dashboard/nova-dashboard-money-aws.conf").read_text()
    assert "LoadCredential" not in unit
    assert "LoadCredentialEncrypted=aws-dashboard:/etc/nova/dashboard-aws.cred" in dropin
    assert "AWS_SHARED_CREDENTIALS_FILE=%d/aws-dashboard" in dropin
    assert "AWS_CONFIG_FILE=/dev/null" in dropin
    assert "NOVA_AWS_PROFILE=nova-dashboard" in dropin
    assert "AWS_SECRET_ACCESS_KEY" not in dropin


def test_aws_cli_uses_service_credential_locator_and_drops_ambient_keys(monkeypatch):
    from nanobody_dash import infra
    monkeypatch.setenv("NOVA_AWS_BIN", "/opt/nova-dashboard/.venv/bin/aws")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/run/credentials/money/aws-dashboard")
    monkeypatch.setenv("AWS_PROFILE", "unrelated")
    def run(command, **kwargs):
        assert command[-2:] == ["--profile", "nova-dashboard"]
        assert "--no-cli-pager" not in command
        assert kwargs["env"]["AWS_PAGER"] == ""
        assert kwargs["env"]["AWS_SHARED_CREDENTIALS_FILE"] == "/run/credentials/money/aws-dashboard"
        assert "AWS_PROFILE" not in kwargs["env"]
        assert kwargs["stderr"] == subprocess.DEVNULL
        return subprocess.CompletedProcess(command, 0, b'{"Account":"668100850015"}')
    monkeypatch.setattr(infra.subprocess, "run", run)
    assert infra._aws_json(["sts", "get-caller-identity"], profile="nova-dashboard") == {"Account":"668100850015"}


def test_aws_collector_rejects_mutation_before_starting_cli(monkeypatch):
    from nanobody_dash import infra
    def unexpected(*args, **kwargs):
        pytest.fail("a write must not reach the AWS CLI")
    monkeypatch.setattr(infra.subprocess, "run", unexpected)
    with pytest.raises(infra.InfraError):
        infra._aws_json(["ec2", "run-instances"], profile="nova-dashboard")


def test_master_export_requires_pinned_release_and_local_store(monkeypatch):
    from nanobody_dash.master_refresh import export_command
    for value in ("", "/root", "/root/releases/current", "/root/releases/" + "a" * 40 + ";bad"):
        monkeypatch.setenv("NOVA_DASH_MASTER_RELEASE_DIR", value)
        with pytest.raises(ValueError, match="release_required"):
            export_command()
    monkeypatch.setenv("NOVA_DASH_MASTER_RELEASE_DIR", "/root/dashboard-releases/" + "a" * 40)
    command = export_command()
    assert "NOVA_LOCAL_SQLITE=1" in command
    assert "-m nanobody_dash.master_refresh" in command
    assert "cd /root &&" not in command
    assert "timeout --kill-after=10 1100" in export_command(replay=True)


def test_master_reference_sync_precedes_export_and_excludes_current_epoch(monkeypatch, tmp_path):
    from nanobody_dash import master_refresh
    from nanobody_opt import rules, seed_mutation_database as seed, mutation_network
    calls = []
    def fetch(epochs, **kwargs):
        assert list(epochs) == list(range(rules.ACTIVATION_EPOCH, rules.ACTIVATION_EPOCH + 3))
        assert kwargs["target"] == rules.target_for_epoch(rules.ACTIVATION_EPOCH + 3)
        calls.append("fetch")
        return [{"observation_id": "fixture"}]
    monkeypatch.setattr(seed, "fetch_observations", fetch)
    monkeypatch.setattr(mutation_network, "update_observation_store", lambda *a: calls.append("store") or 1)
    monkeypatch.setattr(builder, "build", lambda *a, **k: calls.append("export") or {
        str(rules.ACTIVATION_EPOCH + 2): {"ours": {"mean": None}}})
    result = master_refresh.refresh(epoch=rules.ACTIVATION_EPOCH + 3,
                                    database=tmp_path/"db", output=tmp_path/"out")
    assert calls == ["fetch", "store", "export"]
    assert result["reference_rows"] == 1
    assert result["multipass_epoch"] == rules.ACTIVATION_EPOCH + 2


def test_replay_lease_excludes_live_gpus_and_shared_daemon(tmp_path):
    from nanobody_dash import replay_refresh as replay
    for gpu in (True, -1, 0, 1, 2, 3, 4, 8):
        with pytest.raises(ValueError, match="background_pool"):
            replay.lease_job(tmp_path, gpu=gpu, epoch=25965)
    job = replay.lease_job(tmp_path, gpu=5, epoch=25965)
    assert job["gpu"] == 5 and job["ttl_seconds"] == 1050
    assert job["lock_directory"] == "/root/pr100-controller/gpu-locks"
    assert "--owned-daemon" in job["command"]
    assert "--direct" not in job["command"]
    assert job["command"][job["command"].index("--gpus")+1] == "5"
    assert "CUDA_VISIBLE_DEVICES" not in job["env"]
    assert job['command'][:3] == ['taskset', '-c', '104-111']
    assert "TOKEN" not in " ".join(job["env"])


def test_replay_pending_uses_same_target_and_missing_depth(monkeypatch, tmp_path):
    import sqlite3
    from nanobody_dash import replay_refresh as replay
    from nanobody_opt import rules
    from tools import collect_slave_multipass_seqs as collector
    db = tmp_path/"data.sqlite"
    monkeypatch.setattr(replay, "STATE", tmp_path/'queue')
    monkeypatch.setattr(replay, "priority_cohort", lambda *a: [])
    with sqlite3.connect(db) as connection:
        connection.executescript("CREATE TABLE nodes(id INTEGER,target TEXT);"
                                 "CREATE TABLE observations(node_id INTEGER,epoch INTEGER,source TEXT);")
        connection.executemany("INSERT INTO nodes VALUES (?,?)", [(1, rules.TARGET), (2, "OLD")])
        connection.executemany("INSERT INTO observations VALUES (?,?,?)", [
            (1, 25965, "public_validator_api"), (1, 25967, "public_validator_api"),
            (2, 25966, "public_validator_api")])
    monkeypatch.setattr(collector, "leaderboard_sequences", lambda epoch: ["COVERED", "MISSING"] if epoch == 25965 else pytest.fail("wrong epoch"))
    monkeypatch.setattr(collector, "pass_counts", lambda seqs, target: {"COVERED": 10, "MISSING": 9})
    assert replay.pending(db, 25967) == (25965, ["MISSING"])


def test_failed_replay_never_imports_rows(monkeypatch, tmp_path):
    from nanobody_dash import replay_refresh as replay
    from tools import replay_context, ingest_slave_multipass
    monkeypatch.setattr(replay, "STATE", tmp_path)
    monkeypatch.setattr(replay, "pending", lambda *a: (25965, ["FIXTURE"]))
    monkeypatch.setattr(replay_context, "prepare", lambda *a, **k: None)
    monkeypatch.setattr(ingest_slave_multipass, "load_rows", lambda *a: pytest.fail("failed lease must not import"))
    def launch(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text())
        Path(job["receipt"]).write_text(json.dumps({"status": "contention_abort", "child_exit_code": None}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(replay.subprocess, "run", launch)
    assert replay.run(epoch=25967)["reason"] == "replay_contention_abort"


def test_pi_replay_does_not_forward_remote_strings(paths, monkeypatch):
    monkeypatch.setattr(pi, "master_endpoint", lambda: ("host", 22))
    monkeypatch.setattr(pi, "run", lambda *a: subprocess.CompletedProcess([], 1,
        json.dumps({"ok": False, "epoch": "PRIVATE", "rows": "PRIVATE", "reason": "PRIVATE"}).encode()))
    assert pi.auxiliary("replay") == {"ok": False, "epoch": None, "rows": None, "reason": "replay_pending"}


@pytest.mark.parametrize("duplicate", [False, True])
def test_replay_import_requires_all_distinct_passes(monkeypatch, tmp_path, duplicate):
    from nanobody_dash import replay_refresh as replay
    from tools import replay_context, ingest_slave_multipass, collect_slave_multipass_seqs as collector
    from nanobody_opt import mutation_network, rules
    monkeypatch.setattr(replay, "STATE", tmp_path)
    monkeypatch.setattr(replay, "pending", lambda *a: (25965, ["FIXTURE"]))
    monkeypatch.setattr(replay_context, "prepare", lambda *a, **k: None)
    rows = [{"sequence": "FIXTURE", "epoch": 25965, "target": rules.TARGET,
             "metrics": {"fixture": 1}, "allow_replicates": True,
             "observation_id": str(n), "pass_index": 1 if duplicate else n}
            for n in range(1, 11)]
    monkeypatch.setattr(ingest_slave_multipass, "load_rows", lambda *a: rows)
    monkeypatch.setattr(ingest_slave_multipass, "validate", lambda rows: [])
    imported = []
    monkeypatch.setattr(mutation_network, "update_observation_store", lambda db, value: imported.extend(value) or len(value))
    monkeypatch.setattr(collector, "pass_counts", lambda *a: {"FIXTURE": 10})
    def launch(command, **kwargs):
        job = json.loads(Path(command[-1]).read_text())
        Path(job["receipt"]).write_text(json.dumps({"status": "process_finished", "child_exit_code": 0}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(replay.subprocess, "run", launch)
    if duplicate:
        with pytest.raises(ValueError, match="duplicate_passes"):
            replay.run(epoch=25967)
        assert imported == []
    else:
        assert replay.run(epoch=25967)["rows"] == 10
        assert imported == rows


def test_replay_finishes_queued_fleet_across_epoch_changes(monkeypatch, tmp_path):
    from nanobody_dash import replay_refresh as replay
    from nanobody_opt import rules
    from tools import collect_slave_multipass_seqs as collector
    monkeypatch.setattr(replay, 'STATE', tmp_path)
    (tmp_path/'pending_cohort.json').write_text(json.dumps({'target': rules.TARGET, 'items': [
        {'epoch': 25966, 'sequence': 'COVERED'}, {'epoch': 25965, 'sequence': 'RIVAL'}]}))
    monkeypatch.setattr(collector, 'pass_counts', lambda *a: {'COVERED': 10, 'RIVAL': 5})
    monkeypatch.setattr(replay, 'priority_cohort', lambda *a: pytest.fail('must finish pending cohort'))
    assert replay.pending(tmp_path/'unused.sqlite', 25968) == (25965, ['RIVAL'])


def test_replay_interleaves_commits_with_main_rival(monkeypatch, tmp_path):
    from nanobody_dash import replay_refresh as replay, collect
    from nanobody_opt import rules
    from tools.rescale_multipass_history import OUR_COLDKEY
    monkeypatch.setattr(replay, 'DATABASE', tmp_path/'data.sqlite')
    (tmp_path/'dashboard_committed_sequences.json').write_text(json.dumps({
        'epoch': 25966, 'target': rules.TARGET,
        'submissions': [{'sequence': 'OUR_A'}, {'sequence': 'OUR_B'}]}))
    def row(coldkey, score, sequence):
        return {'coldkey': coldkey, 'final_score': score, 'nanobodies': [
            {'protein_name': rules.TARGET, 'sequence': sequence}]}
    monkeypatch.setattr(collect, 'fetch_epoch', lambda epoch: [
        row('MAIN', 1, 'RIVAL_A'), row('MAIN', 3, 'RIVAL_B'),
        row('OTHER', 20, 'LATER'), row(OUR_COLDKEY, 10, 'OUR_OLD')])
    assert replay.priority_cohort(25967, 25965) == [
        {'epoch': 25966, 'sequence': 'OUR_A'}, {'epoch': 25965, 'sequence': 'RIVAL_A'},
        {'epoch': 25966, 'sequence': 'OUR_B'}, {'epoch': 25965, 'sequence': 'RIVAL_B'},
        {'epoch': 25965, 'sequence': 'OUR_OLD'}]


@pytest.mark.parametrize('receipts', [True, False])
def test_submission_status_requires_per_uid_receipts_and_archived_inputs(tmp_path, receipts):
    from nanobody_dash.submission_status import latest
    from nanobody_opt import rules
    archive = tmp_path/'archive/epoch25966'
    archive.mkdir(parents=True)
    (archive/'nanobody_result.json').write_text(json.dumps({
        'epoch': 25966, 'target': rules.TARGET,
        'submissions': [{'uid': 1, 'sequence': 'ONE'}, {'uid': 2, 'sequence': 'TWO'}]}))
    events = [{'event': 'plan', 'epoch': 25966, 'committing': 2, 'with_sequence': 2}]
    if receipts:
        events.extend({'event': 'commit', 'uid': uid, 'rc': 0, 'sequence': True} for uid in (1, 2))
    events.append({'event': 'done', 'committed': 2, 'failed': 0})
    log = tmp_path/'loop.log'
    log.write_text('\n'.join(json.dumps(v) for v in events))
    status, data = latest(log, tmp_path/'archive')
    if receipts:
        assert status == {'epoch': 25966, 'committed': 2, 'planned': 2, 'failed': 0,
                          'source': 'pi_commit_receipts'}
        assert len(data['submissions']) == 2
    else:
        assert status == data == {}


def test_ui_separates_commit_receipts_from_api_evaluation(paths, monkeypatch):
    monkeypatch.setattr(pi, 'latest_leaderboard', lambda: 25965)
    monkeypatch.setattr(pi, 'latest_api_epoch', lambda: 25965)
    (paths/'data/dashboard_submission_status.json').write_text(json.dumps({
        'epoch': 25966, 'committed': 13, 'planned': 13}))
    text = pi.status_text()
    assert 'Pi commits 25966: 13/13' in text
    assert 'API leaderboard 25965' in text
    assert 'measured fleet pending' in text


def test_parallel_replay_uses_only_three_distinct_background_leases(monkeypatch):
    from nanobody_dash import replay_refresh as replay
    seen = []
    def choose(db, epoch, *, exclude):
        values = [s for s in ('A', 'B', 'C', 'D') if s not in exclude]
        return epoch-1, values[:1]
    def run(**kwargs):
        seen.append((kwargs['gpu'], tuple(kwargs['selection'][1])))
        return {'ok': True, 'epoch': kwargs['epoch']-1, 'rows': 10}
    monkeypatch.setattr(replay, 'pending', choose)
    monkeypatch.setattr(replay, 'plan_pair_benchmark', lambda epoch: None)
    monkeypatch.setattr(replay, 'run', run)
    result = replay.run_parallel(25967)
    assert sorted(seen) == [(5, ('A',)), (6, ('B',)), (7, ('C',))]
    assert result['ok'] and result['rows'] == 30


def test_commit_receipts_survive_uid_independent_archive(tmp_path):
    from nanobody_dash.submission_status import latest
    archive = tmp_path/'archive/epoch25966'
    archive.mkdir(parents=True)
    (archive/'nanobody_result.json').write_text(json.dumps({'epoch': 25966, 'pool': []}))
    log = tmp_path/'loop.log'
    log.write_text('\n'.join(json.dumps(v) for v in [
        {'event': 'plan', 'epoch': 25966, 'committing': 1, 'with_sequence': 1},
        {'event': 'commit', 'uid': 1, 'rc': 0, 'sequence': True},
        {'event': 'done', 'committed': 1, 'failed': 0}]))
    status, sequences = latest(log, tmp_path/'archive')
    assert status['committed'] == status['planned'] == 1
    assert sequences == {}


def test_ui_reports_chain_and_incomplete_replay_separately(paths, monkeypatch):
    monkeypatch.setattr(pi, 'latest_chain_epoch', lambda: 25967)
    monkeypatch.setattr(pi, 'latest_leaderboard', lambda: 25966)
    monkeypatch.setattr(pi, 'latest_api_epoch', lambda: 25966)
    pi.COMPARISON.write_text(json.dumps({'25966': {
        'ours': {'mean': None, 'scored_sequences': 1, 'fleet_size': 13},
        'rivals': {'rival': {'scored_sequences': 0, 'fleet_size': 4}}}}))
    text = pi.status_text()
    assert 'chain 25967' in text and 'API leaderboard 25966' in text
    assert 'replay 25966: own 1/13; rivals 0/4 sequences at ≥10 passes' in text
    assert 'measured fleet pending' in text


def test_successful_partial_plan_does_not_hide_unfilled_registered_slots(paths):
    from nanobody_dash.submission_status import latest
    log = paths/'loop.log'
    events = [{'event': 'plan', 'epoch': 25967, 'crew': 13,
               'committing': 10, 'with_sequence': 10}]
    events.extend({'event': 'commit', 'uid': uid, 'rc': 0, 'sequence': True}
                  for uid in range(10))
    events.append({'event': 'done', 'committed': 10, 'failed': 0})
    log.write_text('\n'.join(json.dumps(row) for row in events))
    status, sequences = latest(log, paths/'archive')
    assert status['registered'] == 13 and status['committed'] == status['planned'] == 10
    assert status['failed'] == 0 and sequences == {}
    (paths/'data/dashboard_submission_status.json').write_text(json.dumps(status))
    text = pi.status_text()
    assert 'Pi commits 25967: 10/13 registered (10 planned)' in text
    assert 'Pi commits 25967: 10/10' not in text


def test_chain_status_rejects_stale_slot_record(paths, monkeypatch):
    import sqlite3
    monkeypatch.setattr(pi.time, 'time', lambda: 1000)
    with sqlite3.connect(pi.DATA/'nanobody_dash.sqlite') as db:
        db.executescript('CREATE TABLE slots(epoch INTEGER,time REAL);'
                         'INSERT INTO slots VALUES (25967,399);')
    assert pi.latest_chain_epoch() is None
    with sqlite3.connect(pi.DATA/'nanobody_dash.sqlite') as db:
        db.execute('UPDATE slots SET time=999')
    assert pi.latest_chain_epoch() == 25967


def test_validator_fallback_cannot_advance_api_label(paths, monkeypatch):
    monkeypatch.setattr(pi, 'latest_leaderboard', lambda: 25967)
    monkeypatch.setattr(pi, 'latest_api_epoch', lambda: 25966)
    text = pi.status_text()
    assert 'API leaderboard 25966' in text and 'validator rows 25967' in text
    assert 'API leaderboard 25967' not in text


def test_old_target_complete_fleet_is_not_current_replay(paths):
    pi.COMPARISON.write_text(json.dumps({
        '25954': {'ours': {'mean': 123, 'scored_sequences': 13}},
        '25965': {'ours': {'mean': None, 'scored_sequences': 1, 'fleet_size': 3}},
        '25966': {'ours': {'mean': None, 'scored_sequences': 0, 'fleet_size': 9}}}))
    text = pi.status_text()
    assert 'measured fleet pending' in text
    assert 'earlier replay 25965: 1 sequences at ≥10 passes' in text
    assert 'measured fleet 25954' not in text


def test_requested_pair_benchmark_reuses_two_compatible_pending_candidates(tmp_path, monkeypatch):
    from nanobody_dash import replay_refresh as replay
    from nanobody_opt import rules
    from tools import collect_slave_multipass_seqs as collector
    monkeypatch.setattr(replay, 'STATE', tmp_path)
    monkeypatch.setattr(collector, 'pass_counts', lambda *a: {})
    (tmp_path/'pair_benchmark.request').touch()
    (tmp_path/'pending_cohort.json').write_text(json.dumps({'target': rules.TARGET, 'items': [
        {'epoch': 25966, 'sequence': 'ONE'}, {'epoch': 25966, 'sequence': 'LONGER'},
        {'epoch': 25966, 'sequence': 'TWO'}]}))
    replay.plan_pair_benchmark(25968)
    assert replay.pending(tmp_path/'unused', 25969) == (25966, ['ONE', 'TWO'])
    assert replay.pending(tmp_path/'unused', 25969, exclude=['ONE', 'TWO']) == (25966, ['LONGER'])


def test_pair_benchmark_does_not_postpone_partial_single_candidate(tmp_path, monkeypatch):
    import hashlib
    from nanobody_dash import replay_refresh as replay
    from nanobody_opt import rules
    from tools import collect_slave_multipass_seqs as collector
    monkeypatch.setattr(replay, 'STATE', tmp_path)
    monkeypatch.setattr(collector, 'pass_counts', lambda *a: {})
    (tmp_path/'pair_benchmark.request').touch()
    (tmp_path/'pending_cohort.json').write_text(json.dumps({'target': rules.TARGET, 'items': [
        {'epoch': 25966, 'sequence': 'ONE'}, {'epoch': 25966, 'sequence': 'TWO'}]}))
    batch = hashlib.sha256(json.dumps([25966, 10, ['ONE']]).encode()).hexdigest()[:16]
    output = tmp_path/f'epoch25966/{batch}/work/epoch25966/replicates10/job/length3/pass1/exact_all.json'
    output.parent.mkdir(parents=True); output.write_text('{}')
    replay.plan_pair_benchmark(25968)
    assert not (tmp_path/'pair_benchmark.json').exists()
