"""The supervisor must bound every phase and always leave a receipt behind."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    'pr100_live_supervisor', ROOT / 'scripts' / 'pr100_live_supervisor.py')
sup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sup)


def clock_file(directory, *, epoch=25957, block=None, length=361, age=0.0):
    block = epoch * length if block is None else block
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = Path(directory) / 'clock.json'
    path.write_text(json.dumps({'block': block, 'epoch': block // length,
                                'epoch_length': length,
                                'updated_at_unix': time.time() - age}))
    return path


def spec_for(directory, command, **overrides):
    directory = Path(directory)
    spec = {'command': command, 'cwd': str(directory), 'environment': {'PATH': os.environ['PATH']},
            'wall_seconds': 30, 'deadline_unix': time.time() + 30,
            'receipt_path': str(directory / 'supervisor-receipt.json'), 'cpu_slots': 2,
            'clock_path': str(clock_file(directory)), 'epoch': 25957, 'epoch_length': 361,
            'reserve_blocks': 46}
    spec.update(overrides)
    return spec


def test_blocks_remaining_counts_down_and_detects_stale_or_rolled_clocks(tmp_path):
    path = clock_file(tmp_path, epoch=25957, block=25957 * 361 + 100)
    left = sup.blocks_remaining(path, 25957)
    assert 260 < left <= 261
    clock_file(tmp_path, epoch=25957, age=10_000)
    assert sup.blocks_remaining(tmp_path / 'clock.json', 25957) is None
    clock_file(tmp_path, block=25958 * 361)
    assert sup.blocks_remaining(tmp_path / 'clock.json', 25957) == 0.0
    (tmp_path / 'clock.json').write_text('not json')
    assert sup.blocks_remaining(tmp_path / 'clock.json', 25957) is None


def test_successful_command_writes_a_completed_receipt_and_captures_streams(tmp_path):
    spec = spec_for(tmp_path, [sys.executable, '-c',
                               'import sys; print("out"); print("err", file=sys.stderr)'])
    receipt = sup.supervise(spec, poll=0.05)
    assert receipt['reason'] == 'completed' and receipt['exit_code'] == 0
    assert receipt['killed'] is False
    stored = json.loads(Path(spec['receipt_path']).read_text())
    assert stored == receipt
    assert 'out' in Path(receipt['stdout_path']).read_text()
    assert 'err' in receipt['stderr_tail']


def test_nonzero_exit_is_reported_with_a_bounded_sanitized_stderr_tail(tmp_path):
    program = ('import sys; print("authorization: abcdefghijklmnop", file=sys.stderr); '
               'sys.exit(3)')
    spec = spec_for(tmp_path, [sys.executable, '-c', program])
    receipt = sup.supervise(spec, poll=0.05)
    assert receipt['reason'] == 'nonzero_exit' and receipt['exit_code'] == 3
    assert '<redacted>' in receipt['stderr_tail']
    assert 'abcdefghijklmnop' not in receipt['stderr_tail']
    assert len(receipt['stderr_tail']) <= sup.TAIL_BYTES


def test_wall_seconds_cutoff_kills_a_process_group_that_ignores_term(tmp_path):
    program = ('import signal, time\n'
               'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
               'time.sleep(120)\n')
    spec = spec_for(tmp_path, [sys.executable, '-c', program], wall_seconds=0.5)
    receipt = sup.supervise(spec, poll=0.05, grace=0.5)
    assert receipt['reason'] == 'wall_seconds'
    assert receipt['killed'] is True and receipt['killed_hard'] is True
    assert receipt['exit_code'] is not None
    assert json.loads(Path(spec['receipt_path']).read_text())['killed'] is True


def test_deadline_and_reserve_cutoffs_stop_the_phase(tmp_path):
    sleeper = [sys.executable, '-c', 'import time; time.sleep(120)']
    spec = spec_for(tmp_path / 'a', sleeper, wall_seconds=60,
                    deadline_unix=time.time() + 0.5)
    receipt = sup.supervise(spec, poll=0.05, grace=0.3)
    assert receipt['reason'] == 'deadline_unix' and receipt['killed'] is True


    spec = spec_for(tmp_path / 'b', sleeper, wall_seconds=60)
    # Written after the spec, whose own helper would otherwise reset the clock file.
    clock_file(tmp_path / 'b', block=25957 * 361 + 361 - 10)
    receipt = sup.supervise(spec, poll=0.05, grace=0.3)
    assert receipt['reason'] == 'reserve_before_start'
    assert json.loads(Path(spec['receipt_path']).read_text())['reason'] == 'reserve_before_start'


def test_a_child_that_outlives_the_leader_is_cleaned_up(tmp_path):
    program = (f'import subprocess, sys\n'
               f'subprocess.Popen([{sys.executable!r}, "-c", "import time; time.sleep(60)"])\n')
    spec = spec_for(tmp_path, [sys.executable, '-c', program])
    receipt = sup.supervise(spec, poll=0.05, grace=0.3)
    assert receipt['reason'] == 'completed'
    assert receipt.get('orphans_killed') is True


def test_an_invalid_spec_still_produces_a_receipt(tmp_path):
    spec = spec_for(tmp_path, [sys.executable, '-c', 'pass'])
    del spec['wall_seconds']
    receipt = sup.supervise(spec, poll=0.05)
    assert receipt['reason'] == 'supervisor_error'
    assert receipt['error_type'] == 'ValueError'
    assert Path(spec['receipt_path']).is_file()


def test_cli_writes_a_receipt_for_an_unreadable_spec(tmp_path):
    (tmp_path / 'spec.json').write_text('{not json')
    code = sup.main(['--spec', str(tmp_path / 'spec.json')])
    assert code == 2
    receipt = json.loads((tmp_path / 'supervisor-receipt.json').read_text())
    assert receipt['reason'] == 'spec_unreadable'


def test_cli_runs_a_command_end_to_end(tmp_path):
    spec = spec_for(tmp_path, [sys.executable, '-c', 'print("ok")'])
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    code = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / 'pr100_live_supervisor.py'),
                           '--spec', str(tmp_path / 'spec.json')],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode
    assert code == 0
    assert json.loads(Path(spec['receipt_path']).read_text())['reason'] == 'completed'


def test_a_clock_that_goes_stale_mid_run_does_not_kill_a_child_within_the_grace(tmp_path, monkeypatch):
    # A brief controller-to-host outage stops the clock updates; the healthy child must keep running
    # and only the hard deadline or wall bound may end it.
    monkeypatch.setattr(sup, 'MAXIMUM_CLOCK_AGE', 1.0)
    monkeypatch.setattr(sup, 'CLOCK_GRACE_SECONDS', 30.0)
    spec = spec_for(tmp_path, [sys.executable, '-c', 'import time; time.sleep(3)'])
    receipt = sup.supervise(spec, poll=0.05, grace=0.3)
    assert receipt['reason'] == 'completed' and receipt['killed'] is False


def test_a_clock_lost_beyond_the_grace_kills_the_child(tmp_path, monkeypatch):
    monkeypatch.setattr(sup, 'MAXIMUM_CLOCK_AGE', 1.0)
    monkeypatch.setattr(sup, 'CLOCK_GRACE_SECONDS', 1.0)
    spec = spec_for(tmp_path, [sys.executable, '-c', 'import time; time.sleep(60)'], wall_seconds=30)
    receipt = sup.supervise(spec, poll=0.05, grace=0.3)
    assert receipt['reason'] == 'clock_unavailable' and receipt['killed'] is True
