"""Bounded fresh GPU screening in waves; no eligibility or publication decisions.

Uses the same pinned fresh one-pass probe and cooperative GPU leases as the
small-fleet score phase. Completed probe groups survive a later deadline. A
fresh private output directory prevents previous artifacts becoming new work.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from nanobody_opt import pr100_filter, pr100_live_gpu, pr100_live_worker as worker

MIN_WAVE_SECONDS = 110.0  # probe's 60 s minimum + 30 s lease + 20 s phase reserves
CLEANUP_SECONDS = 6.0
MAX_WAVE_SECONDS = 1800.0
_GROUP = re.compile(r'gpu-([0-9]+)/length([0-9]+)/exact_all[.]json')


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')
    return float(value)


def _inputs(payload):
    sequences = payload.get('sequences')
    if (not isinstance(sequences, list) or not sequences
            or any(type(s) is not str or not s or s != s.strip().upper() for s in sequences)):
        raise ValueError('screen input must contain canonical nonempty sequences')
    if len(set(sequences)) != len(sequences):
        raise ValueError('duplicate sequence in screen input')
    excluded = payload.get('previously_evaluated_hashes', [])
    if (not isinstance(excluded, list) or any(not isinstance(h, str)
            or not re.fullmatch(r'[0-9a-f]{64}', h) for h in excluded)):
        raise ValueError('invalid previous-evaluation exclusions')
    excluded = set(excluded)
    fresh = [s for s in sequences if hashlib.sha256(s.encode()).hexdigest() not in excluded]
    return sequences, fresh


def _launch(path, log):
    return subprocess.Popen([sys.executable, '-B', '-m', 'nanobody_opt.pr100_live_gpu',
                             '--job', str(path)], stdin=subprocess.DEVNULL,
                            stdout=log, stderr=log, start_new_session=True)


def _collect(out, jobs, target):
    """Count actual validated unique payload entries, never declared shard sizes."""
    from nanobody_opt.score_boltzgen_waves import _validated_wave
    from nanobody_opt.simulate_round import METRICS
    report = worker.collect_scores(out, jobs)
    scorer = report.get('registered_scorer_sha256')
    if report['groups'] and (not isinstance(scorer, str) or not re.fullmatch(r'[0-9a-f]{64}', scorer)):
        raise ValueError('invalid registered scorer identity')
    found = set()
    groups = []
    for group in report['groups']:
        match = _GROUP.fullmatch(group['file'])
        if match is None:
            raise ValueError('invalid score group path')
        gpu, length = map(int, match.groups())
        probe = json.loads((out / f'gpu-{gpu}' / 'report.json').read_text())
        if probe.get('mode') != 'fresh' or probe.get('passes') != 1 or probe.get('target') != target:
            raise ValueError('probe is not a fresh one-pass observation of this target')
        shard = json.loads((out / f'shard-gpu-{gpu}.json').read_text())['sequences']
        expected = [s for s in shard if len(s) == length]
        path = out / group['file']
        if not expected or not path.resolve().is_relative_to(out.resolve()):
            raise ValueError('unexpected score group')
        value = _validated_wave(path, expected, target, require_structures=True)
        rows = value.get('per_nanobody_components', {}).get('0', {})
        if set(rows) != set(expected) or found.intersection(rows):
            raise ValueError('duplicate or unexpected fresh observations')
        for sequence, by_target in rows.items():
            metrics = by_target.get(target)
            if not isinstance(metrics, dict) or any(
                    type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key])
                    for key in METRICS):
                raise ValueError('incomplete or nonfinite fresh metrics')
        found.update(rows)
        groups.append({**group, 'candidates': len(rows),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report.update(groups=groups, scored_candidates=len(found))
    return report, found


def screen_phase(job, directory, *, launch=None, clock=time.monotonic,
                 wall_clock=time.time, sleep=time.sleep, control=None):
    """Screen all unique eligible inputs once, while the absolute deadline allows.

    Prior-evaluation hashes, when supplied, are excluded before lease submission.
    Without that registry, freshness means unique fresh evaluations in this run;
    this worker cannot infer whether an input was evaluated in an earlier epoch.
    The caller's supervisor still owns the live chain clock and epoch rollover.
    """
    directory = Path(directory)
    if not isinstance(job.get('source_sha'), str) or not re.fullmatch(r'[0-9a-f]{40}', job['source_sha']):
        raise ValueError('invalid source revision')
    began = clock()
    wall = _positive(job['wall_seconds'], 'wall_seconds')
    absolute = _positive(job['deadline_unix'], 'deadline_unix')
    deadline = began + max(0.0, min(wall, absolute - wall_clock()))
    wave_wall = _positive(job.get('wave_wall_seconds', MAX_WAVE_SECONDS), 'wave_wall_seconds')
    if not MIN_WAVE_SECONDS <= wave_wall <= MAX_WAVE_SECONDS:
        raise ValueError('wave_wall_seconds outside probe bounds')
    gpus = job['gpus']
    worker.selection.shard([{'sequence': 'INERT'}], gpus)  # shared GPU-list validation
    input_bytes = (directory / 'input.json').read_bytes()
    payload = json.loads(input_bytes)
    sequences, eligible = _inputs(payload)
    per_gpu = job.get('wave_candidates_per_gpu', worker.SCORE_SHARD_LIMIT)
    if type(per_gpu) is not int or not 1 <= per_gpu <= worker.SCORE_SHARD_LIMIT:
        raise ValueError('wave_candidates_per_gpu outside probe bounds')
    wave_size = per_gpu * len(gpus)
    out = Path(job['output_directory'])
    # Never resume cached group files as newly completed fresh evaluations.
    out.mkdir(parents=True, exist_ok=False)
    launch = launch or _launch
    control = control or pr100_live_gpu._controller()
    control.dedicated_controller()
    stopping = []
    previous = {sig: signal.signal(sig, lambda *_: stopping.append(True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    groups, shards, waves, scored = [], [], [], set()
    settings, scorer_hash = None, None
    submitted = 0
    lease_attempts, lease_finishes = {}, {}
    stop_reason = None

    def remaining():
        # Wall-clock forward movement shortens the run; backward movement can
        # never extend the monotonic deadline fixed at entry.
        return max(0.0, min(deadline - clock(), absolute - wall_clock()))

    def checkpoint(status):
        report = {'schema': 'pr100-wide-score-v1', 'groups': list(groups), 'shards': list(shards),
                  'registered_scorer_sha256': scorer_hash, 'waves': list(waves),
                  'gpus_attempted': len(lease_attempts),
                  'gpus_finished': sum(lease_finishes.get(gpu, 0) == count
                                       for gpu, count in lease_attempts.items()),
                  'gpu_leases_attempted': sum(lease_attempts.values()),
                  'gpu_leases_finished': sum(lease_finishes.values()),
                  'scored_candidates': len(scored), 'status': status,
                  'elapsed_seconds': clock() - began, 'deadline_unix': absolute,
                  'mode': 'fresh', 'passes': 1, 'cache_hits': 0,
                  'fresh_evaluations_completed': len(scored), 'submitted_candidates': submitted,
                  'input_count': len(sequences), 'eligible_input_count': len(eligible),
                  'excluded_previously_evaluated': len(sequences) - len(eligible),
                  'not_submitted': len(eligible) - submitted,
                  'submitted_without_valid_result': submitted - len(scored),
                  'stop_reason': stop_reason,
                  'prior_evaluation_registry_supplied': 'previously_evaluated_hashes' in payload,
                  'input_sha256': hashlib.sha256(input_bytes).hexdigest(),
                  'source_sha': job['source_sha']}
        pr100_filter.atomic(out / 'report.json', report)
        result = {key: value for key, value in report.items() if key not in ('groups', 'shards')}
        result.update(challenge=job['challenge'], epoch=job['challenge']['epoch'],
                      candidates=[s for s in eligible if s in scored],
                      not_run=len(eligible) - submitted)
        pr100_filter.atomic(directory / 'result.json', result)
        return result

    result = checkpoint('running')
    try:
        for wave_index, offset in enumerate(range(0, len(eligible), wave_size)):
            if stopping or remaining() < MIN_WAVE_SECONDS:
                stop_reason = 'interrupted' if stopping else 'deadline'
                break
            wave_directory = out / f'wave-{wave_index:04d}'
            wave_directory.mkdir()
            wave_seconds = min(wave_wall, remaining())
            wave_deadline = clock() + wave_seconds

            def wave_remaining():
                return max(0.0, min(remaining(), wave_deadline - clock()))
            # Avoid another nested 'score' in report paths; score_jobs owns the
            # wave directory and writes exactly its familiar per-GPU layout.
            wave_job = {**job, 'output_directory': str(wave_directory),
                        'wall_seconds': wave_seconds,
                        'deadline_unix': min(absolute, wall_clock() + wave_seconds - CLEANUP_SECONDS)}
            batch = eligible[offset:offset + wave_size]
            jobs = worker.score_jobs(wave_job, wave_directory, batch)
            runners, attempted_jobs = [], []
            error_code, cleanup = None, None
            try:
                for gpu_job in jobs:
                    if stopping or wave_remaining() < CLEANUP_SECONDS + 10:
                        if stopping or remaining() < CLEANUP_SECONDS + 10:
                            stop_reason = 'interrupted' if stopping else 'deadline'
                        break
                    path = wave_directory / f"gpu-job-{gpu_job['gpu']}.json"
                    pr100_filter.atomic(path, {k: v for k, v in gpu_job.items() if k != 'candidate_count'})
                    log = open(wave_directory / f"gpu-runner-{gpu_job['gpu']}.log", 'ab')
                    try:
                        process = launch(path, log)
                    except BaseException:
                        log.close()
                        raise
                    runners.append((process, log))
                    attempted_jobs.append(gpu_job)
                    submitted += gpu_job['candidate_count']
                    lease_attempts[gpu_job['gpu']] = lease_attempts.get(gpu_job['gpu'], 0) + 1
                while any(process.poll() is None for process, _ in runners):
                    if stopping or wave_remaining() <= CLEANUP_SECONDS:
                        if stopping or remaining() <= CLEANUP_SECONDS:
                            stop_reason = 'interrupted' if stopping else 'deadline'
                        break
                    sleep(min(.5, max(.01, wave_remaining() - CLEANUP_SECONDS)))
            except Exception:
                error_code = 'runner_launch_or_poll_failed'
            finally:
                for process, _ in runners:
                    try:
                        if process.poll() is None:
                            process.send_signal(signal.SIGTERM)
                    except (OSError, ValueError):
                        pass
                # Parent is a dedicated subreaper, so this reaches owned probe
                # sessions too. No unrelated resident GPU worker is touched.
                if runners:
                    try:
                        cleanup = control.stop_descendants([p for p, _ in runners],
                                                           timeout=min(4.0, wave_remaining()))
                    except Exception:
                        cleanup = {'remaining_descendants': None, 'fixed_point_reached': False,
                                   'error_code': 'owned_cleanup_failed'}
                for _, log in runners:
                    log.close()
            wave_receipt = {'wave': wave_index, 'submitted_candidates': sum(j['candidate_count'] for j in attempted_jobs),
                            'error_code': error_code, 'cleanup': cleanup}
            try:
                report, found = _collect(wave_directory, attempted_jobs, job['challenge']['target'])
                if scored.intersection(found):
                    raise ValueError('a fresh input was scored twice')
                if found:
                    incoming_settings = (wave_directory / 'scorer_settings.json').read_bytes()
                    if (settings is not None and settings != incoming_settings
                            or scorer_hash is not None and scorer_hash != report['registered_scorer_sha256']):
                        raise ValueError('scorer identity changed between waves')
                    settings, scorer_hash = incoming_settings, report['registered_scorer_sha256']
                    (out / 'scorer_settings.json').write_bytes(settings)
                scored.update(found)
                groups.extend({**g, 'wave': wave_index, 'file': f"wave-{wave_index:04d}/" + g['file']}
                              for g in report['groups'])
                shards.extend({**s, 'wave': wave_index} for s in report['shards'])
                for shard in report['shards']:
                    if shard.get('probe_status') == 'complete':
                        lease_finishes[shard['gpu']] = lease_finishes.get(shard['gpu'], 0) + 1
                wave_receipt.update(status=report['status'], scored_candidates=len(found))
            except (OSError, ValueError, KeyError, TypeError):
                wave_receipt.update(status='invalid', scored_candidates=0, error_code='wave_result_invalid')
                stop_reason = 'invalid_result'
            waves.append(wave_receipt)
            if cleanup and (cleanup.get('remaining_descendants') or not cleanup.get('fixed_point_reached')):
                stop_reason = 'containment_failure'
            elif error_code:
                stop_reason = error_code
            checkpoint('partial')
            if stop_reason:
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result = checkpoint('complete' if len(scored) == len(eligible) and not stop_reason else 'partial')
    return result
