"""Inert scheduling/receipt tests. No model, GPU or native profiling runs here."""
import hashlib
import json
from pathlib import Path
import signal
from unittest.mock import Mock

import pytest

from nanobody_opt import pr100_filter, pr100_live_gpu as gpu, pr100_live_worker as worker
from nanobody_opt import pr100_wide_screen as screen
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL
from nanobody_opt.simulate_round import METRICS


class Clock:
    def __init__(self):
        self.now, self.jump = 0.0, 0.0
    def monotonic(self): return self.now
    def wall(self): return 1000.0 + self.now + self.jump
    def sleep(self, seconds): self.now += seconds


def sequences(count):
    return [f'S{i:06d}' for i in range(count)]


def job(directory, **over):
    return {'phase': 'screen', 'challenge': {'epoch': 25960, 'target': 'TEST'},
            'gpus': [0, 1, 2, 3], 'wall_seconds': 1500, 'deadline_unix': 2500,
            'output_directory': str(directory / 'score'), 'release': str(directory / 'release'),
            'source_sha': 'a' * 40, 'scorer_settings': str(directory / 'settings.json'),
            'scorer': 'registered', 'gpu_lock_directory': str(directory / 'locks'), **over}


class Process:
    def __init__(self, gpu_job, *, hanging=False, lease=None):
        self.gpu_job, self.alive, self.lease = gpu_job, hanging, lease
        self.pid, self.signals = 100 + gpu_job['gpu'], []
    def poll(self): return None if self.alive else 0
    def send_signal(self, value): self.signals.append(value)


class Control:
    def __init__(self): self.initialized, self.cleanups, self.fail = 0, [], False
    def dedicated_controller(self): self.initialized += 1
    def stop_descendants(self, processes, *, timeout):
        self.cleanups.append((list(processes), timeout))
        for process in processes:
            if process.lease:
                # The live lease contract retains its kernel lock through cleanup.
                with pytest.raises(gpu.Unavailable):
                    with gpu.lease_locks(process.gpu_job['gpu'], process.gpu_job['lock_directory']):
                        pass
                process.lease.__exit__(None, None, None)
            process.alive = False
        return {'remaining_descendants': int(self.fail), 'fixed_point_reached': not self.fail}


class Launcher:
    def __init__(self, clock, *, cost=0, hangs=(), unavailable=(), bad_mode_wave=None,
                 changed_settings_wave=None):
        self.clock, self.cost = clock, cost
        self.hangs, self.unavailable = set(hangs), set(unavailable)
        self.bad_mode_wave, self.changed_settings_wave = bad_mode_wave, changed_settings_wave
        self.jobs, self.inputs, self.processes = [], [], []
    def __call__(self, path, log):
        public = json.loads(path.read_text())
        command = public['command']
        input_path = Path(command[command.index('--input') + 1])
        output = Path(command[command.index('--output') + 1])
        values = json.loads(input_path.read_text())['sequences']
        wave = int(path.parent.name.split('-')[1])
        key = (wave, public['gpu'])
        assert 1 <= len(values) <= 128
        assert command[command.index('--mode') + 1] == 'fresh'
        assert command[command.index('--passes') + 1] == '1'
        assert public['deadline_unix'] <= 2500
        self.jobs.append(public); self.inputs.extend(values)
        self.clock.sleep(self.cost)
        lease = None
        if key not in self.unavailable:
            lease = gpu.lease_locks(public['gpu'], public['lock_directory'])
            lease.__enter__()
        process = Process(public, hanging=key in self.hangs, lease=lease)
        self.processes.append(process)
        pr100_filter.atomic(Path(public['receipt']), {'status': 'unavailable' if key in self.unavailable else 'process_finished'})
        if key in self.hangs or key in self.unavailable:
            return process
        output.mkdir()
        settings = {'scorer_sha256': 'b' * 64, 'steps': 36 if wave == self.changed_settings_wave else 35}
        (output / 'scorer_settings.json').write_text(json.dumps(settings))
        groups = []
        for length in sorted(set(map(len, values))):
            found = [s for s in values if len(s) == length]
            group = output / f'length{length}'
            group.mkdir()
            value = {'target': 'TEST', 'n': len(found), 'scorer_protocol': SCORER_PROTOCOL,
                     'per_nanobody_components': {'0': {s: {'TEST': {key: 1.0 for key in METRICS}}
                                                        for s in found}},
                     'structures': {s: {'TEST': {'path': 'inert', 'writer_path': 'inert',
                                               'sample_selection': {'sample_count': 5}}} for s in found}}
            pr100_filter.atomic(group / 'exact_all.json', value)
            groups.append({'length': length, 'valid': True, 'candidates': len(found)})
        pr100_filter.atomic(output / 'report.json', {'status': 'complete', 'target': 'TEST',
            'mode': 'replicates' if wave == self.bad_mode_wave else 'fresh', 'passes': 1,
            'registered_scorer_sha256': 'b' * 64, 'groups': groups})
        return process


def run(directory, count, *, config=None, launch_options=None, payload_over=None, control=None):
    directory.mkdir(exist_ok=True)
    pr100_filter.atomic(directory / 'input.json', {'sequences': sequences(count), **(payload_over or {})})
    clock, control = Clock(), control or Control()
    launcher = Launcher(clock, **(launch_options or {}))
    result = screen.screen_phase(config or job(directory), directory, launch=launcher,
                                 control=control, clock=clock.monotonic, wall_clock=clock.wall,
                                 sleep=clock.sleep)
    return result, launcher, control, clock


@pytest.mark.parametrize('count', [13, 512, 513, 4800])
def test_every_unique_input_is_fresh_scored_once_without_shard_truncation(tmp_path, count):
    result, launched, control, _ = run(tmp_path, count)
    assert result['status'] == 'complete'
    assert result['input_count'] == result['fresh_evaluations_completed'] == count
    assert result['scored_candidates'] == count and result['cache_hits'] == 0
    assert result['submitted_candidates'] == count and result['not_submitted'] == 0
    assert len(launched.inputs) == len(set(launched.inputs)) == count
    assert set(launched.inputs) == set(sequences(count))
    assert len(result['waves']) == (count + 511) // 512
    assert control.initialized == 1 and len(control.cleanups) == len(result['waves'])
    assert result['gpus_attempted'] == min(4, count)
    assert result['gpu_leases_attempted'] == len(launched.jobs)
    assert all(timeout <= 4 for _, timeout in control.cleanups)
    stored = json.loads((tmp_path / 'score/report.json').read_text())
    assert sum(g['candidates'] for g in stored['groups']) == count
    assert all((tmp_path / 'score' / g['file']).is_file() for g in stored['groups'])


def test_absolute_deadline_retains_completed_waves_and_leaves_rest_unsubmitted(tmp_path):
    result, launched, control, clock = run(tmp_path, 4800,
        config=job(tmp_path, deadline_unix=1250), launch_options={'cost': 20})
    assert result['status'] == 'partial' and result['stop_reason'] == 'deadline'
    assert result['fresh_evaluations_completed'] == result['submitted_candidates'] == 1024
    assert result['not_submitted'] == 3776 and clock.now <= 250
    assert len(launched.jobs) == 8 and len(control.cleanups) == 2
    assert json.loads((tmp_path / 'result.json').read_text()) == result


def test_partial_wave_is_kept_and_missing_candidates_are_not_rescored(tmp_path):
    result, launched, control, clock = run(tmp_path, 513,
        config=job(tmp_path, wave_wall_seconds=110), launch_options={'hangs': {(0, 1)}})
    assert result['status'] == 'partial'
    assert result['fresh_evaluations_completed'] == 385
    assert result['submitted_without_valid_result'] == 128
    assert len(launched.inputs) == len(set(launched.inputs)) == 513
    assert signal.SIGTERM in launched.processes[1].signals
    assert all(not process.alive for process in launched.processes)
    assert clock.now < 110 and len(control.cleanups) == 2


def test_unavailable_lease_does_not_count_candidate_requests_as_fresh_results(tmp_path):
    result, launched, _, _ = run(tmp_path, 513, launch_options={'unavailable': {(0, 2)}})
    assert result['scored_candidates'] == 385
    assert result['submitted_candidates'] == 513 and result['submitted_without_valid_result'] == 128
    assert len(launched.inputs) == len(set(launched.inputs))


@pytest.mark.parametrize('change', ['replicates', 'settings'])
def test_bad_later_wave_cannot_relabel_or_replace_previous_fresh_results(tmp_path, change):
    options = {'bad_mode_wave' if change == 'replicates' else 'changed_settings_wave': 1}
    result, _, _, _ = run(tmp_path, 1025, launch_options=options)
    assert result['status'] == 'partial' and result['stop_reason'] == 'invalid_result'
    assert result['scored_candidates'] == 512 and result['not_submitted'] == 1


def test_previously_evaluated_inputs_are_excluded_before_submission(tmp_path):
    previous = [hashlib.sha256(s.encode()).hexdigest() for s in sequences(17)]
    result, launched, _, _ = run(tmp_path, 513, payload_over={'previously_evaluated_hashes': previous})
    assert result['input_count'] == 513 and result['eligible_input_count'] == 496
    assert result['fresh_evaluations_completed'] == 496
    assert result['excluded_previously_evaluated'] == 17 and result['cache_hits'] == 0
    assert not set(sequences(17)).intersection(launched.inputs)


def test_existing_output_cannot_be_counted_as_new_or_overwritten(tmp_path):
    out = tmp_path / 'score'; out.mkdir()
    (out / 'report.json').write_text('{"scored_candidates":4800}')
    with pytest.raises(FileExistsError):
        run(tmp_path, 4800)
    assert json.loads((out / 'report.json').read_text())['scored_candidates'] == 4800


def test_duplicate_input_refuses_before_any_launch(tmp_path):
    with pytest.raises(ValueError, match='duplicate'):
        run(tmp_path, 2, payload_over={'sequences': ['INERT', 'INERT']})


def test_absolute_deadline_is_required_and_finite(tmp_path):
    settings = job(tmp_path); settings.pop('deadline_unix')
    with pytest.raises(KeyError): run(tmp_path, 513, config=settings)
    for invalid in (float('inf'), float('nan'), -1, True):
        with pytest.raises(ValueError): run(tmp_path, 513, config=job(tmp_path, deadline_unix=invalid))


def test_expired_deadline_starts_no_gpu_work(tmp_path):
    result, launched, control, _ = run(tmp_path, 4800, config=job(tmp_path, deadline_unix=999))
    assert result['scored_candidates'] == result['submitted_candidates'] == 0
    assert result['not_submitted'] == 4800 and not launched.jobs and not control.cleanups


def test_failed_containment_prevents_later_wave(tmp_path):
    control = Control(); control.fail = True
    result, launched, _, _ = run(tmp_path, 1025, control=control)
    assert result['stop_reason'] == 'containment_failure'
    assert result['scored_candidates'] == 512 and len(launched.jobs) == 4


@pytest.mark.parametrize('jump', [-5000, 5000])
def test_clock_movement_cannot_extend_absolute_or_monotonic_budget(tmp_path, jump):
    pr100_filter.atomic(tmp_path / 'input.json', {'sequences': sequences(4800)})
    clock, control = Clock(), Control()
    launcher = Launcher(clock, cost=20)
    def launch(path, log):
        process = launcher(path, log)
        if len(launcher.jobs) == 1:
            clock.jump = jump
        return process
    result = screen.screen_phase(job(tmp_path, deadline_unix=1250), tmp_path,
        launch=launch, control=control, clock=clock.monotonic, wall_clock=clock.wall, sleep=clock.sleep)
    assert result['status'] == 'partial' and result['stop_reason'] == 'deadline'
    assert clock.now <= 250
    assert result['fresh_evaluations_completed'] == (128 if jump > 0 else 1024)


def test_launch_failure_cleans_existing_lease_and_preserves_completed_group(tmp_path):
    pr100_filter.atomic(tmp_path / 'input.json', {'sequences': sequences(513)})
    clock, control = Clock(), Control()
    launcher = Launcher(clock)
    def launch(path, log):
        if launcher.jobs:
            raise OSError('inert launch failure')
        return launcher(path, log)
    result = screen.screen_phase(job(tmp_path), tmp_path, launch=launch, control=control,
        clock=clock.monotonic, wall_clock=clock.wall, sleep=clock.sleep)
    assert result['scored_candidates'] == result['submitted_candidates'] == 128
    assert result['stop_reason'] == 'runner_launch_or_poll_failed'
    assert len(control.cleanups) == 1
    with gpu.lease_locks(0, tmp_path / 'locks'):
        pass


def test_signal_stops_admission_and_restores_the_previous_handler(tmp_path):
    pr100_filter.atomic(tmp_path / 'input.json', {'sequences': sequences(513)})
    clock, control = Clock(), Control()
    launcher = Launcher(clock)
    previous = signal.getsignal(signal.SIGTERM)
    def launch(path, log):
        process = launcher(path, log)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return process
    result = screen.screen_phase(job(tmp_path), tmp_path, launch=launch, control=control,
        clock=clock.monotonic, wall_clock=clock.wall, sleep=clock.sleep)
    assert result['stop_reason'] == 'interrupted' and result['scored_candidates'] == 128
    assert len(launcher.jobs) == 1 and signal.getsignal(signal.SIGTERM) == previous


def test_wave_job_builder_refuses_the_old_silent_truncation(tmp_path):
    with pytest.raises(ValueError, match='probe limit'):
        worker.score_jobs(job(tmp_path), tmp_path, sequences(513))
    jobs = worker.score_jobs(job(tmp_path), tmp_path, sequences(13))
    assert sum(j['candidate_count'] for j in jobs) == 13
    assert all(j['deadline_unix'] == 2500 for j in jobs)


def test_large_score_phase_dispatches_to_bounded_worker(tmp_path, monkeypatch):
    pr100_filter.atomic(tmp_path / 'input.json', {'sequences': sequences(513)})
    called = []
    monkeypatch.setattr(screen, 'screen_phase', lambda *a, **k: called.append(a) or {'status': 'inert'})
    assert worker.score_phase(job(tmp_path), tmp_path) == {'status': 'inert'}
    assert len(called) == 1


def test_lease_absolute_deadline_refuses_expired_work_without_gpu_admission(tmp_path):
    value = {'gpu': 0, 'ttl_seconds': 120, 'deadline_unix': 1005,
             'command': ['/inert'], 'scorer': 'registered', 'env': {},
             'lock_directory': str(tmp_path / 'locks'), 'receipt': str(tmp_path / 'receipt.json'),
             'cwd': str(tmp_path)}
    snapshot, launch, control = Mock(), Mock(), Mock()
    result = gpu.run_reserved(value, snapshot=snapshot, launch=launch, control=control,
                              clock=lambda: 0, wall_clock=lambda: 1000)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'epoch_deadline_before_admission'
    snapshot.assert_not_called(); launch.assert_not_called()


def test_transfer_accepts_wave_groups_without_allowing_path_traversal(tmp_path):
    import re
    from nanobody_opt import pr100_live_controller as controller
    assert "(?:wave-[0-9]+/)?gpu-" in controller.TRANSPORT
    pattern = r'(?:wave-[0-9]+/)?gpu-[0-9]+/length[0-9]+/exact_all[.]json'
    assert re.fullmatch(pattern, 'wave-0001/gpu-2/length7/exact_all.json')
    assert not re.fullmatch(pattern, '../wave-0001/gpu-2/length7/exact_all.json')
