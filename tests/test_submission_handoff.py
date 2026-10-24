"""Regress epoch25914: completion on the master must precede the Pi snapshot."""
import hashlib
import json
import shlex
import time
import tomllib
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from nanobody_opt import submission_timing as timing
from nanobody_opt.epoch_deadline import EpochDeadline, EpochDeadlineExceeded
from nanobody_opt_v2 import run_live_producer as producer
from tools import pi_epoch_loop as pi
from tools import publish_current_nanobody as publisher

ROOT = Path(__file__).resolve().parents[1]


def shipping_args(tmp_path):
    args = producer._parser().parse_args(
        (ROOT / 'deploy/producer/producer_argv.txt').read_text().splitlines()[3:])
    args.runtime_root = tmp_path
    args.gpus = '0,1,2,3,4,5,6,7'
    args.netuid = 68
    return args


def write_clock(path, epoch, offset):
    path.write_text(json.dumps({'epoch': epoch, 'block': epoch * 361 + offset,
                               'epoch_length': 361, 'updated_at_unix': time.time()}))


def test_shipping_profiles_finish_before_the_actual_pi_pull(tmp_path):
    args = shipping_args(tmp_path)
    assert args.pi_handoff and args.epoch_candidate_budget == 4800
    config = tomllib.loads((ROOT / 'pixi.toml').read_text())
    def tasks(obj):
        for k, v in obj.items():
            if k == 'publish-current-nanobody':
                yield v['cmd'] if isinstance(v, dict) else v
            if isinstance(v, dict):
                yield from tasks(v)
    commands = list(tasks(config))
    assert len(commands) == 1
    assert '--pi-handoff' in shlex.split(commands[0])
    p = timing.configured_plan(args)
    # The real Pi's commit reserve + final-pull margin, not the master's old 36.
    pi_pull = 361 - pi.commit_reserve(13) - 60 / 12
    scoring_end = 361 - producer._handoff_reserve(args)
    publication_end = scoring_end + timing.PUBLICATION_SECONDS / 12
    transferred = publication_end + timing.TRANSFER_SECONDS / 12
    assert transferred <= pi_pull
    assert p['publication_reserve_blocks'] == 46
    assert producer._handoff_reserve(args) == 76
    # This is the incident's old contract and must fail the same invariant.
    assert 361 - 36 + timing.PUBLICATION_SECONDS / 12 > pi_pull


@pytest.mark.parametrize('crew', [1, 12, 13, 14, 20])
def test_live_fleet_growth_moves_producer_deadline_before_pi_cutoff(tmp_path, crew):
    args = shipping_args(tmp_path)
    p = timing.configured_plan(args, crew)
    path = tmp_path / 'artifacts/submission_handoff.json'
    path.parent.mkdir()
    path.write_text(json.dumps({**p, 'epoch': 100}))
    reserve = producer._handoff_reserve(args)
    assert reserve >= pi.commit_reserve(crew) + 5 + 5 + 30
    assert timing.configured_plan(args)['crew_size'] == max(13, crew)


def test_real_adaptive_budget_reserves_the_whole_handoff(tmp_path):
    args = shipping_args(tmp_path)
    root = tmp_path / 'live/epoch100/producer'
    state = {'epoch': 100, 'epoch_length': 361, 'block': 36100}
    plan = producer._budget_plan(args, state, root)
    assert plan['screen_cutoff_block'] + plan['replication_reserve_blocks'] <= 361 - 76
    receipt = json.loads((root / 'budget.json').read_text())
    assert receipt['effective_handoff_reserve_blocks'] == 76
    assert receipt['submission_handoff']['pi_pull_reserve_blocks'] == 41


def test_confirm_publish_pull_flow_and_rollover(tmp_path, monkeypatch):
    args = shipping_args(tmp_path)
    args.chain_state = tmp_path / 'clock.json'
    epoch = 25914
    write_clock(args.chain_state, epoch, 284)
    # Confirmation finishes just before its actual shipping deadline.
    assert EpochDeadline(args.chain_state, epoch, producer._handoff_reserve(args)).remaining() > 0
    completed = tmp_path / f'live/epoch{epoch}/producer/full_scan_confirmation.json'
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps({'epoch': epoch, 'rounds': 20}))
    confirmation = timing.confirmation_digest(tmp_path, epoch)
    p = timing.configured_plan(args)
    # Spend the entire allowed publication window, using the publisher's
    # real pre-publication guard, then transfer before the Pi's snapshot.
    write_clock(args.chain_state, epoch, 314)
    monkeypatch.setattr(publisher, '_COMMAND_DEADLINE',
                        EpochDeadline(args.chain_state, epoch, p['publication_reserve_blocks']))
    monkeypatch.setattr(publisher, '_chain_view', lambda *a: type('Chain', (), {'epoch': epoch})())
    publisher._assert_same_epoch(args, epoch, 'test handoff')
    artifact = {'epoch': epoch, 'submission_handoff': timing.publication_receipt(
        epoch=epoch, timing=p, confirmation=confirmation, block=epoch*361+314, epoch_length=361)}
    source, pulled = tmp_path/'master.json', tmp_path/'pi.json'
    publisher._atomic_json(source, artifact)
    write_clock(args.chain_state, epoch, 319)
    pulled.write_bytes(source.read_bytes())
    receipt = pi.submission_timing.artifact_receipt(pulled, epoch)
    assert receipt['confirmation_complete_at_start']
    assert receipt['confirmation_sha256'] == hashlib.sha256(completed.read_bytes()).hexdigest()
    assert receipt['artifact_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    # A late publisher must leave that committed input intact, and the next
    # epoch cannot mistake it for its own completed confirmation.
    write_clock(args.chain_state, epoch, 320)
    with pytest.raises(EpochDeadlineExceeded):
        publisher._assert_same_epoch(args, epoch, 'late publication')
    write_clock(args.chain_state, epoch+1, 1)
    with pytest.raises(EpochDeadlineExceeded, match='epoch changed'):
        publisher._COMMAND_DEADLINE.remaining()
    assert not pi.submission_timing.artifact_receipt(pulled, epoch+1)['confirmation_complete_at_start']
    assert timing.confirmation_digest(tmp_path, epoch+1) is None
    assert EpochDeadline(args.chain_state, epoch+1, 76).remaining() > 0


def test_fallback_cannot_claim_confirmation_that_finished_later(tmp_path):
    epoch = 100
    before = timing.confirmation_digest(tmp_path, epoch)
    assert before is None
    completed = tmp_path / 'live/epoch100/producer/full_scan_confirmation.json'
    completed.parent.mkdir(parents=True)
    completed.write_text(json.dumps({'epoch': epoch}))
    receipt = timing.publication_receipt(epoch=epoch, timing=timing.plan(),
        confirmation=before, block=100*361+240, epoch_length=361)
    assert receipt['confirmation_complete_at_start'] is False
    assert receipt['confirmation_sha256'] is None


@pytest.mark.parametrize('future, expected_reserve', [(False, 76), (True, 0)])
def test_pool_preparation_keeps_the_real_epoch_clock(tmp_path, monkeypatch, future, expected_reserve):
    args = shipping_args(tmp_path)
    state = {'epoch': 100, 'block': 100*361+240, 'epoch_length': 361,
             'nanobody_target': 'TEST'}
    class ObservedDeadline(Exception):
        pass
    def prepare(**kwargs):
        assert kwargs['prepare_only']
        assert producer._STAGED.deadline.epoch == 100
        assert producer._STAGED.deadline.reserve_blocks == expected_reserve
        raise ObservedDeadline()
    monkeypatch.setattr(producer, '_run_iteration', prepare)
    with pytest.raises(ObservedDeadline):
        producer._prepare_epoch_pool(args, state, 101 if future else 100)


def test_parallel_reserve_has_retry_room_without_charging_every_uid():
    assert pi.commit_reserve(13) == 36
    assert timing.commit_reserve(13, serial=True) == 81
    assert pi.commit_reserve(20) == pi.commit_reserve(13)
    assert pi.commit_reserve(21) > pi.commit_reserve(20)
    # 98s was the slowest measured process in thirty retained epochs.
    assert (pi.commit_reserve(13) - timing.BLACKOUT_BLOCKS) * 12 >= 2 * 98
    assert timing.LATE_COMMIT_MARGIN_SECONDS == 60


def test_pi_preparation_does_not_extend_the_chain_deadline(tmp_path, monkeypatch):
    now = [10000.0]
    monkeypatch.setattr(pi, 'ROOT', tmp_path)
    monkeypatch.setattr(pi.time, 'time', lambda: now[0])
    monkeypatch.setattr(pi.time, 'sleep', lambda seconds: now.__setitem__(0, now[0]+seconds))
    monkeypatch.setattr(pi, 'say', lambda **kwargs: None)
    # Model five minutes spent preparing crew/artifacts before waiting.
    monkeypatch.setattr(pi, 'publish', lambda *a: now.__setitem__(0, now[0]+300))
    monkeypatch.setattr(pi, 'producer_epoch_here', lambda *a: 100)
    monkeypatch.setattr(pi, 'artifact_confirmation', lambda *a: (13, 13))
    monkeypatch.setattr(pi, 'check_producer_identity', lambda *a: False)
    monkeypatch.setattr(pi, 'refresh_crew_for_commit', lambda a,v,*rest: v)
    pulls = []
    def pull(*a, **k):
        pulls.append(now[0])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pi, 'run', pull)
    args = SimpleNamespace(on_submit_host=True, late_commit=True,
        late_commit_margin_seconds=60, confirmation_min_observations=20, dry_run=True)
    pi.handle_epoch(args, {'crew': list(range(13)), 'reaction': 1},
        100, 100*361+225, 101*361, set(), {'producer_stopped': False})
    expected = 10000 + (361-225-36)*12 - 60
    assert pulls == [expected]


def test_silent_commit_process_cannot_block_the_next_epoch(tmp_path, monkeypatch):
    child = tmp_path / 'child.pid'
    marker = tmp_path / 'escaped'
    child_code = f'import time; from pathlib import Path; time.sleep(5); Path({str(marker)!r}).touch()'
    code = ('import subprocess,sys,time; from pathlib import Path; '
            f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
            f'Path({str(child)!r}).write_text(str(p.pid)); time.sleep(30)')
    # A real silent subprocess exercises the old blocking stdout iterator.
    command = shlex.join([sys.executable, '-c', code])
    real_popen = subprocess.Popen
    def started_child(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        # Login-shell startup varies by runner. Begin the timeout experiment
        # only after the child exists; startup speed is not the property here.
        until = time.monotonic() + 10
        while not child.exists() and process.poll() is None and time.monotonic() < until:
            time.sleep(.02)
        return process
    monkeypatch.setattr(pi.subprocess, 'Popen', started_child)
    started = time.monotonic()
    rc, _ = pi.stream(pi.LOCAL, command, timeout=.5)
    assert rc == 124 and time.monotonic()-started < 12
    assert child.exists()
    probe = subprocess.run(['ps', '-o', 'stat=', '-p', child.read_text()], capture_output=True, text=True)
    assert not probe.stdout.strip() or probe.stdout.strip().startswith('Z')
    assert not marker.exists()
    assert pi.stream(pi.LOCAL, shlex.join([sys.executable, '-c', 'pass']), timeout=3)[0] == 0
