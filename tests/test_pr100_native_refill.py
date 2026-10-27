"""Inert scheduling comparisons; these never execute a scientific profile."""
import asyncio
from collections import Counter, deque
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from nanobody_opt import pr100_filter, pr100_live_worker as worker, pr100_native_refill as refill, rules
from tests.test_pr100_native_screen_integration import CHALLENGE, profile


class Simulation:
    def __init__(self, duration, outcome, *, deadline, budget=300, cleanup=10):
        self.duration, self.outcome = duration, outcome
        self.deadline, self.budget, self.cleanup = deadline, budget, cleanup
        self.now = 0.; self.calls = Counter(); self.trace = []; self.maximum = 0

    def clock(self):
        return self.now

    def start(self, sequence, timeout):
        self.calls[sequence] += 1; draw = self.calls[sequence]
        duration = self.duration(sequence, draw)
        outcome = self.outcome(sequence, draw)
        if duration > timeout:
            duration, outcome = timeout, 'timeout'
        value = (profile(sequence, 80. if outcome == 'pass' else 160.)
                 if outcome in ('pass', 'reject') else {'error': True, 'error_kind': outcome})
        self.trace.append({'sequence': sequence, 'draw': draw, 'start': self.now,
                           'end': self.now + duration, 'outcome': outcome, 'timeout': timeout})
        return self.now + duration, value

    def batch(self, sequences, workers, **_kwargs):
        begin = self.now
        timeout = min(self.budget, self.deadline - self.now - self.cleanup)
        events = [self.start(sequence, timeout) for sequence in sequences]
        self.maximum = max(self.maximum, len(events))
        self.now = max(end for end, _ in events)
        return [value for _, value in events], begin, self.now - begin

    def refill(self, next_calls, on_result, workers, *, clock):
        active = []; serial = 0; completed = 0
        while True:
            calls = next_calls(workers - len(active))
            assert len(calls) <= workers - len(active)
            for call in calls:
                assert all(call['sequence'] != item[2]['sequence'] for item in active)
                end, value = self.start(call['sequence'], call['timeout_seconds'])
                heapq.heappush(active, (end, serial, call, value)); serial += 1
            self.maximum = max(self.maximum, len(active))
            if not active:
                break
            self.now = active[0][0]
            done = []
            while active and active[0][0] == self.now:
                done.append(heapq.heappop(active))
            for _, _, call, value in done:
                call['record'](call['sequence'], value)
                on_result(call, value); completed += 1
        return {'admitted_calls': serial, 'completed_calls': completed,
                'maximum_in_flight': self.maximum, 'elapsed_seconds': self.now}


def simulated_phase(directory, monkeypatch, *, enabled, durations, outcomes,
                    count=600, width=96, deadline=1000, fresh_limit=4800, carried=None):
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'resume_eligible': True, 'runtime': 'unchanged'}
    monkeypatch.setattr(worker, 'worker_count', lambda _job: width)
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *_args: identity)
    monkeypatch.setattr(pr100_filter, 'helper', lambda _name: SimpleNamespace(tool_paths=None))
    monkeypatch.setattr(rules, 'validate_evidence', lambda *_args: None)
    rows = [{'sequence': str(i), 'pr100_annotation': {}} for i in range(count)]
    for index, observed in (carried or {}).items():
        sequence = str(index)
        rows[index]['native_resume'] = {
            'schema': 'pr100-native-confirmation-checkpoint-v1',
            'sequence_hash': rules.sequence_hash(sequence), 'identity': identity,
            'measurements': [profile(sequence) for _ in range(observed)],
            'required_cpu_observations': 3, 'execution_host': 'slave'}
    pr100_filter.atomic(directory / 'input.json', {'challenge': CHALLENGE,
        'epoch': CHALLENGE['epoch'], 'target': CHALLENGE['target'],
        'archive_sha256': 'a' * 64, 'archive_checked_at_unix': 1, 'candidates': rows})
    simulation = Simulation(durations, outcomes, deadline=deadline)
    job = {'challenge': CHALLENGE, 'native_tools': '/tools', 'native_host': 'slave',
        'workers': width, 'repeats': 3, 'phase_budget_seconds': deadline,
        'candidate_budget_seconds': 300, 'native_minimum_call_window_seconds': 130,
        'native_cleanup_reserve_seconds': 10, 'fresh_admission_limit': fresh_limit,
        'native_interleave_confirmations': True, 'native_work_conserving': enabled}
    injection = {'refill': simulation.refill} if enabled else {'measure': simulation.batch}
    result = worker.native_phase(job, directory, clock=simulation.clock, **injection)
    return result, simulation


def summary(result, simulation, width):
    busy = sum(row['end'] - row['start'] for row in simulation.trace)
    return {'wall_seconds': simulation.now, 'fresh': result['fresh_evaluations'],
        'repeats': result['repeat_evaluations'], 'fully_confirmed': sum(
            row['repeats'] >= 3 for row in result['candidates']),
        'inconclusive': result['measurement_outcomes']['inconclusive'],
        'maximum_in_flight': simulation.maximum,
        'slot_utilization': busy / (width * simulation.now) if simulation.now else 0.,
        'profile_calls': len(simulation.trace)}


@pytest.mark.parametrize('width', [32, 96])
def test_stragglers_do_not_hold_free_slots_and_confirmation_yield(tmp_path, monkeypatch, width):
    results = {}
    for enabled in (False, True):
        result, simulation = simulated_phase(tmp_path / str(enabled), monkeypatch,
            enabled=enabled, width=width, deadline=1000,
            durations=lambda sequence, _draw: 250 if int(sequence) % width == 0 else 60,
            outcomes=lambda sequence, _draw: 'pass' if int(sequence) % 37 == 0 else 'reject')
        results[enabled] = summary(result, simulation, width)
        assert simulation.maximum <= width
        assert all(row['start'] + 140 <= 1000 and row['end'] <= 990 for row in simulation.trace)
        assert all(count <= 3 for count in simulation.calls.values())
    assert results[True]['fresh'] > results[False]['fresh']
    assert results[True]['fully_confirmed'] >= results[False]['fully_confirmed']
    assert results[True]['slot_utilization'] > results[False]['slot_utilization']


def test_unlimited_window_preserves_identical_original_draws_and_verdicts(tmp_path, monkeypatch):
    results = []
    for enabled in (False, True):
        result, simulation = simulated_phase(tmp_path / str(enabled), monkeypatch,
            enabled=enabled, count=130, width=32, deadline=5000,
            durations=lambda sequence, draw: 20 + (int(sequence) * 37 + draw * 17) % 230,
            outcomes=lambda sequence, draw: 'pass' if int(sequence) % 13 == 0 and draw < 3 else 'reject')
        results.append((result, simulation))
    old, new = results
    assert new[1].calls == old[1].calls
    assert new[0]['terminal_native_outcomes'] == old[0]['terminal_native_outcomes']
    assert new[0]['measurement_outcomes'] == old[0]['measurement_outcomes']
    assert new[0]['candidates'] == old[0]['candidates']
    assert new[1].now < old[1].now


def test_rejection_and_timeout_never_receive_another_draw(tmp_path, monkeypatch):
    result, simulation = simulated_phase(tmp_path, monkeypatch, enabled=True,
        count=40, width=8, deadline=1500,
        durations=lambda sequence, draw: 400 if int(sequence) % 3 == 0 and draw == 2 else 40,
        outcomes=lambda sequence, draw: 'pass' if draw == 1 else 'reject')
    assert all(count == 2 for count in simulation.calls.values())
    assert result['measurement_outcomes']['inconclusive'] == 14
    assert len(result['terminal_native_outcomes']) == 26
    assert len(result['deferred_confirmations']) == 14
    assert all(len(row['native_resume']['measurements']) == 1 for row in result['deferred_confirmations'])


@pytest.mark.parametrize('deadline', [130, 140, 260, 430])
def test_late_cutoff_does_not_start_below_minimum_or_overrun_cleanup(tmp_path, monkeypatch, deadline):
    result, simulation = simulated_phase(tmp_path, monkeypatch, enabled=True, count=20,
        width=4, deadline=deadline, durations=lambda _sequence, _draw: 400,
        outcomes=lambda _sequence, _draw: 'pass')
    assert all(row['start'] + 140 <= deadline for row in simulation.trace)
    assert all(row['end'] <= deadline - 10 for row in simulation.trace)
    assert all(count == 1 for count in simulation.calls.values())
    assert result['candidates'] == []


def test_carried_observations_fresh_cap_and_completed_checkpoint_zero_redraw(tmp_path, monkeypatch):
    result, simulation = simulated_phase(tmp_path, monkeypatch, enabled=True,
        count=12, width=4, deadline=1000, fresh_limit=3, carried={0: 3, 1: 2, 2: 1},
        durations=lambda _sequence, _draw: 40, outcomes=lambda _sequence, _draw: 'pass')
    assert simulation.calls['0'] == 0 and simulation.calls['1'] == 1 and simulation.calls['2'] == 2
    assert result['fresh_evaluations'] == result['fresh_admitted_unique'] == 3
    assert result['repeat_evaluations'] == 9
    assert result['carried_cpu_observations'] == 6
    assert len(result['candidates']) == 6
    assert all(row['native_resume']['execution_host'] == 'slave'
               for row in result['candidates'] if 'native_resume' in row)


def test_actual_async_executor_refills_before_straggler_and_after_cleanup():
    queue = deque(str(i) for i in range(5)); events = []; active = set(); completed = []
    def next_calls(free):
        return [{'sequence': queue.popleft(), 'timeout_seconds': 130,
                 'record': lambda *_args: None, 'record_error': lambda *_args: None}
                for _ in range(min(free, len(queue)))]
    async def develop(sequences, concurrency, **_kwargs):
        sequence = sequences[0]; active.add(sequence); events.append(('start', sequence))
        assert concurrency == 1 and len(active) <= 2
        try:
            await asyncio.sleep(.08 if sequence == '0' else .005)
            return [{'value': sequence}]
        finally:
            active.remove(sequence); events.append(('clean', sequence))
    def done(call, value):
        assert call['sequence'] not in active
        completed.append(value['value'])
    result = refill.run(next_calls, done, 2, develop=develop)
    assert events.index(('start', '4')) < events.index(('clean', '0'))
    assert len(completed) == 5 and result['maximum_in_flight'] == 2


def test_actual_executor_callback_failure_joins_active_threads():
    queue = deque(['fast', 'slow']); clean = []
    def next_calls(free):
        return [{'sequence': queue.popleft(), 'timeout_seconds': 130,
                 'record': lambda *_args: None, 'record_error': lambda *_args: None}
                for _ in range(min(free, len(queue)))]
    async def develop(sequences, *_args, **_kwargs):
        sequence = sequences[0]
        task = asyncio.create_task(asyncio.to_thread(time.sleep, .005 if sequence == 'fast' else .04))
        try:
            await asyncio.shield(task)
            return [{}]
        finally:
            while not task.done():
                try: await asyncio.shield(task)
                except asyncio.CancelledError: continue
            clean.append(sequence)
    def fail(*_args):
        raise RuntimeError('inert persistence failure')
    with pytest.raises(RuntimeError, match='persistence'):
        refill.run(next_calls, fail, 2, develop=develop)
    assert set(clean) == {'fast', 'slow'}
    assert not any(thread.name.startswith('pr100-native-refill') for thread in threading.enumerate())


def test_actual_executor_does_not_enqueue_more_than32_default_threads():
    queue = deque(str(i) for i in range(40)); barrier = threading.Barrier(40, timeout=3)
    def next_calls(free):
        return [{'sequence': queue.popleft(), 'timeout_seconds': 130,
                 'record': lambda *_args: None, 'record_error': lambda *_args: None}
                for _ in range(min(free, len(queue)))]
    async def develop(sequences, *_args, **_kwargs):
        await asyncio.to_thread(barrier.wait)
        return [{}]
    result = refill.run(next_calls, lambda *_args: None, 40, develop=develop)
    assert result['completed_calls'] == result['maximum_in_flight'] == 40
    assert not any(thread.name.startswith('pr100-native-refill') for thread in threading.enumerate())


def test_actual_call_rechecks_absolute_window_after_admission_persistence():
    queued = [False]; outcomes = []; now = [0.]
    def next_calls(_free):
        if queued[0]: return []
        queued[0] = True; now[0] = 11.
        return [{'sequence': 'inert', 'timeout_seconds': 140, 'deadline': 140,
            'minimum_window': 130, 'record': lambda *_args: None,
            'record_error': lambda *_args: None}]
    async def forbidden(*_args, **_kwargs):
        pytest.fail('closed admission reached the scientific adapter')
    result = refill.run(next_calls, lambda call, value: outcomes.append(value), 1,
                        develop=forbidden, clock=lambda: now[0])
    assert len(outcomes) == 1 and isinstance(outcomes[0], refill.NotStarted)
    assert result['completed_calls'] == 0 and result['never_started_calls'] == 1


def test_late_prepared_call_remains_fresh_never_started(tmp_path, monkeypatch):
    def late(self, next_calls, on_result, workers, *, clock):
        delayed = [False]
        def wrapped(free):
            calls = next_calls(free)
            if not delayed[0]: self.now += 11.; delayed[0] = True
            return calls
        async def forbidden(*_args, **_kwargs):
            pytest.fail('late prepared call reached native code')
        return refill.run(wrapped, on_result, workers, clock=clock, develop=forbidden)
    monkeypatch.setattr(Simulation, 'refill', late)
    result, simulation = simulated_phase(tmp_path, monkeypatch, enabled=True,
        count=4, width=2, deadline=150, durations=lambda *_args: 40,
        outcomes=lambda *_args: 'pass')
    assert result['completed'] == result['fresh_evaluations'] == result['fresh_admitted_unique'] == 0
    assert result['repeat_evaluations'] == result['in_flight'] == 0
    assert result['fresh_never_started_sequence_hashes'] == sorted(rules.sequence_hash(str(i)) for i in range(4))
    assert len(result['deferred_candidates']) == 4
    assert result['terminal_native_outcomes'] == {} and result['measurement_outcomes']['inconclusive'] == 0


def test_refill_child_tree_is_owned_and_completed_checkpoint_survives_cutoff(tmp_path):
    from tests.test_pr100_live_supervisor import spec_for, sup
    root = Path(__file__).resolve().parents[1]
    pr100_filter.atomic(tmp_path / 'input.json', {'challenge': CHALLENGE,
        'epoch': CHALLENGE['epoch'], 'target': CHALLENGE['target'],
        'archive_sha256': 'a' * 64, 'archive_checked_at_unix': 1,
        'candidates': [{'sequence': value, 'pr100_annotation': {}} for value in ('fast', 'slow')]})
    program = r'''
import asyncio,json,os,pathlib,subprocess,sys,time
from types import SimpleNamespace
from nanobody_opt import pr100_live_worker as worker,pr100_filter,rules
from tests.test_pr100_native_screen_integration import CHALLENGE,profile
base=pathlib.Path.cwd()
worker.worker_count=lambda _:2
worker.native_resume_identity=lambda *_a:{'resume_eligible':True,'runtime':'inert'}
pr100_filter.helper=lambda _:SimpleNamespace(tool_paths=None)
rules.validate_evidence=lambda *_a:None
async def develop(sequences, concurrency, **kwargs):
 sequence=sequences[0]
 if sequence=='fast':
  await asyncio.sleep(.02); value=profile(sequence)
  kwargs['on_result'](sequence,value)
  return [value]
 child_code="import os,signal,subprocess,sys,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)']);open('owned-pids.json','w').write(json.dumps([os.getpid(),child.pid]));time.sleep(60)"
 child=await asyncio.create_subprocess_exec(sys.executable,'-c',child_code)
 await child.wait()
 return [{}]
pr100_filter.develop=develop
worker.native_phase({'challenge':CHALLENGE,'native_tools':'/inert','workers':2,'repeats':3,
 'native_host':'slave','phase_budget_seconds':300,'candidate_budget_seconds':300,
 'native_minimum_call_window_seconds':130,'native_cleanup_reserve_seconds':10,
 'native_interleave_confirmations':True,'native_work_conserving':True},base)
'''
    spec = spec_for(tmp_path, [sys.executable, '-B', '-c', program], wall_seconds=2.,
                    deadline_unix=time.time() + 15)
    spec['environment']['PYTHONPATH'] = str(root)
    path = tmp_path / 'spec.json'; path.write_text(json.dumps(spec))
    executed = subprocess.run([sys.executable, '-B', str(root / 'scripts/pr100_live_supervisor.py'),
        '--spec', str(path)], capture_output=True, timeout=20)
    assert executed.returncode == 1
    receipt = json.loads((tmp_path / 'supervisor-receipt.json').read_text())
    assert receipt['reason'] == 'wall_seconds' and receipt['killed'] is True
    records = json.loads((tmp_path / 'native-confirmations.json').read_text())['records']
    complete = next(row for row in records if row.get('sequence') == 'fast')
    assert len(complete['native_resume']['measurements']) == 3
    assert complete['native_resume']['execution_host'] == 'slave'
    if sys.platform == 'linux':
        assert receipt['cleanup']['remaining_descendants'] == 0
        assert receipt['cleanup']['fixed_point_reached'] is True
    assert not sup._group_alive(receipt['pid'])
    for pid in json.loads((tmp_path / 'owned-pids.json').read_text()):
        with pytest.raises(ProcessLookupError): os.kill(pid, 0)
