import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from nanobody_opt.epoch_deadline import EpochDeadline, EpochDeadlineExceeded, run
from sn68_source import import_shipping


def clock(path, epoch=72, offset=20):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'epoch': epoch, 'epoch_length': 361,
                                    'block': epoch * 361 + offset,
                                    'updated_at_unix': time.time()}))
    temporary.replace(path)


def test_rollover_cancels_blocked_command_and_its_child(tmp_path):
    state, marker, child_marker = [tmp_path / x for x in ('clock.json', 'child.pid', 'escaped')]
    clock(state)
    child = ('import time; from pathlib import Path; time.sleep(4); '
             f'Path({str(child_marker)!r}).touch()')
    command = ('import subprocess,sys,time; from pathlib import Path; '
               f'p=subprocess.Popen([sys.executable,"-c",{child!r}]); '
               f'Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(30)')
    def rollover():
        end = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < end:
            time.sleep(.02)
        clock(state, epoch=73)
    thread = threading.Thread(target=rollover)
    thread.start()
    start = time.monotonic()
    with pytest.raises(EpochDeadlineExceeded, match='epoch changed'):
        run([sys.executable, '-c', command], deadline=EpochDeadline(state, 72), timeout=8)
    thread.join(timeout=6)
    assert marker.exists()
    assert time.monotonic() - start < 4
    # SIGKILL must reach descendants even if their parent exits on SIGTERM.
    child_pid = int(marker.read_text())
    probe = subprocess.run(['ps', '-o', 'stat=', '-p', str(child_pid)], capture_output=True, text=True)
    assert not probe.stdout.strip() or probe.stdout.strip().startswith('Z')
    assert not child_marker.exists()
    # The next epoch can immediately start its own work.
    assert run([sys.executable, '-c', 'pass'], deadline=EpochDeadline(state, 73)).returncode == 0


def test_time_budget_cancels_work_even_when_clock_stops_advancing(tmp_path):
    state = tmp_path / 'clock.json'
    clock(state)
    with pytest.raises(EpochDeadlineExceeded, match='time budget'):
        run([sys.executable, '-c', 'import time; time.sleep(30)'],
            deadline=EpochDeadline(state, 72), timeout=.1)


def test_handoff_reserve_refuses_to_start_and_stale_clock_fails_closed(tmp_path):
    state = tmp_path / 'clock.json'
    clock(state, offset=330)
    with pytest.raises(EpochDeadlineExceeded, match='handoff'):
        run(['must-not-execute'], deadline=EpochDeadline(state, 72, 36))
    clock(state)
    data = json.loads(state.read_text()); data['updated_at_unix'] -= 200
    state.write_text(json.dumps(data))
    with pytest.raises(EpochDeadlineExceeded, match='stale'):
        EpochDeadline(state, 72).remaining()


def test_publisher_thin_strict_pool_never_expands_to_entire_screen(monkeypatch, tmp_path):
    publisher = import_shipping('tools.publish_current_nanobody', monkeypatch)
    def rank(command):
        Path(command[command.index('--output') + 1]).write_text(json.dumps({
            'candidates': [{'sequence': 'BEST'}, {'sequence': 'NEXT'}]}))
    monkeypatch.setattr(publisher, '_run', rank)
    path = publisher._strict_pool(
        args=SimpleNamespace(strict_pool_size=100, nanobody_opt_root=tmp_path),
        bgen_python=Path(sys.executable), epoch_root=tmp_path, archive_path=tmp_path/'archive.csv',
        target='TEST', exclusion_path=tmp_path/'ledger', epoch=72, shortlist=100,
        config_path=tmp_path/'config.yaml', exact_inputs=[], observation_history=None,
        crew=list(range(13)), previous=None, candidate_scope=tmp_path/'eleven_thousand.txt')
    assert path is not None
    assert path.read_text().splitlines() == ['BEST', 'NEXT']


def test_publication_requires_twenty_independent_folds():
    from nanobody_opt.publication_gates import gate_defaults
    assert int(gate_defaults()['--minimum-observations']) == 20


def test_slave_sees_producer_finalists_when_publisher_is_stale(monkeypatch, tmp_path):
    from tools import topup_slave as slave
    monkeypatch.setattr(slave, 'LIVE_ROOT', tmp_path)
    path = tmp_path/'epoch72/producer/iteration_02/race/rounds/stage_ranking.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'candidates': [{'sequence': 'BEST', 'observation_count': 19}]}))
    assert slave.ranking_sources(72)['pool'] == [('BEST', 19)]
    assert slave.ranking_sources(73)['pool'] == []


def test_background_replay_yields_until_confirmation_and_current_publication(tmp_path):
    from tools.live_submission_priority import critical
    artifacts = tmp_path/'artifacts'; artifacts.mkdir()
    state = artifacts/'chain_state.json'
    clock(state, offset=160)
    assert critical(tmp_path)
    completed = tmp_path/'live/epoch72/producer/full_scan_confirmation.json'
    completed.parent.mkdir(parents=True); completed.write_text('{}')
    artifact = artifacts/'nanobody_result.json'
    artifact.write_text(json.dumps({'epoch': 71, 'submissions': [{}]}))
    assert critical(tmp_path)
    artifact.write_text(json.dumps({'epoch': 72, 'submissions': [{}]}))
    assert not critical(tmp_path)
    clock(state, epoch=73, offset=160)
    assert critical(tmp_path)


def saved_finalists(root, epoch=71, target='TEST', depth=20, count=1):
    from nanobody_opt_v2.run_live_producer import GENERATION_POLICY
    iteration = root / f'live/epoch{epoch}/producer/iteration_02'
    rounds = iteration / 'race/rounds'
    rounds.mkdir(parents=True)
    sequences = [f'CANDIDATE{i:03}' for i in range(count)]
    layer2 = iteration / 'layer2_dedup.json'
    layer2.write_text(json.dumps({'target': target, 'sequences': sequences}))
    Path(str(layer2) + '.policy').write_text(GENERATION_POLICY)
    (rounds / 'stage_ranking.json').write_text(json.dumps({
        'target': target, 'candidates': [
            {'sequence': sequence, 'observation_count': depth, 'selection_score': i + 1}
            for i, sequence in enumerate(sequences)]}))
    exact = rounds / 'round_20/exact_all.json'
    exact.parent.mkdir(); exact.write_text('{}')
    return layer2, exact


def test_completed_unpublished_finalists_survive_rollover(monkeypatch, tmp_path):
    publisher = import_shipping('tools.publish_current_nanobody', monkeypatch)
    _, exact = saved_finalists(tmp_path)
    saved_finalists(tmp_path, epoch=72, count=0)
    epoch_root = tmp_path / 'live/epoch72'
    scope = publisher._producer_candidate_scope(epoch_root, 'TEST')
    assert scope.read_text().splitlines() == ['CANDIDATE000']
    current = publisher._current_epoch_exact_inputs(tmp_path, 72)
    assert publisher._publication_exact_inputs(tmp_path, 72, 'TEST') == (*current, exact)
    assert publisher._previous_confirmed_pool(epoch_root, 'TEST')[0] == {'CANDIDATE000'}


@pytest.mark.parametrize('invalid', ['shallow', 'target', 'policy', 'old'])
def test_carryover_rejects_ineligible_or_unbounded_history(monkeypatch, tmp_path, invalid):
    publisher = import_shipping('tools.publish_current_nanobody', monkeypatch)
    layer2, _ = saved_finalists(tmp_path, epoch=70 if invalid == 'old' else 71,
                               target='OTHER' if invalid == 'target' else 'TEST',
                               depth=19 if invalid == 'shallow' else 20)
    if invalid == 'policy':
        Path(str(layer2) + '.policy').write_text('obsolete')
    assert publisher._previous_confirmed_pool(tmp_path / 'live/epoch72', 'TEST') == (set(), ())


def test_carryover_caps_finalists_and_never_reads_old_screen_pool(monkeypatch, tmp_path):
    publisher = import_shipping('tools.publish_current_nanobody', monkeypatch)
    _, exact = saved_finalists(tmp_path, count=100)
    screen = exact.parents[2] / 'screen/exact_all.json'
    screen.parent.mkdir(); screen.write_text('{}')
    sequences, files = publisher._previous_confirmed_pool(tmp_path / 'live/epoch72', 'TEST')
    assert len(sequences) == 64
    assert 'CANDIDATE000' in sequences and 'CANDIDATE099' not in sequences
    assert files == (exact,)
