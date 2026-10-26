#!/usr/bin/env python3
"""CPU-only CLI/repeat/warm parity experiment; no cache or production writes.

Use only after independent review of population, runtime closure and source.
This verifies a supplied runtime file inventory, not its completeness.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanobody_opt.pr100_native_benchmark import SCHEMA, MODES, FIELDS, A3D_FIELDS, digest, compare_results
FROZEN_POPULATION = '1fe6a4bab9b41a055268036e89f088c998f4fe1046d56e68a2b75c20e3dd638d'


def sha_file(path, *, deadline=None):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('runtime verification deadline')
            hasher.update(block)
    return hasher.hexdigest()


def verify_runtime(runtime, *, deadline=None):
    if runtime.get('schema') != 'pr100-native-runtime-manifest-v1':
        raise ValueError('native manifest schema')
    if runtime.get('closure_reviewed') is not True:
        raise ValueError('dependency closure not independently reviewed')
    from scripts.release_manifest import RESEARCH_FILES, verify
    stage_manifest = ROOT / 'build_stamp.json'
    if sha_file(stage_manifest, deadline=deadline) != runtime['source_stage_manifest_sha256']:
        raise ValueError('staged source manifest drift')
    stage = json.loads(stage_manifest.read_text())
    if stage.get('research_files') != sorted(RESEARCH_FILES):
        raise ValueError('research source not included in reviewed stage')
    verify(ROOT, stage, expected_sha=runtime['source_stage_git_sha'])
    if runtime.get('cuda_visible_devices') != '' or runtime.get('refinement_threads') != 1:
        raise ValueError('only the frozen CPU/thread1 comparison is permitted')
    for required in ('python', 'tnp', 'nanobody_builder_cli', 'aggrescan', 'freesasa',
                     'hmmscan', 'hmm', 'germline_reference', 'logging_shim'):
        if runtime['entrypoints'][required] not in runtime['files']:
            raise ValueError('missing runtime entrypoint')
    if len(runtime.get('model_weights', [])) != 4 or any(
            p not in runtime['files'] for p in runtime['model_weights']):
        raise ValueError('all four model checkpoints must be pinned')
    if runtime['entrypoints']['python'] != str(Path(sys.executable).resolve()):
        raise ValueError('wrong interpreter')
    for path, expected in runtime['files'].items():
        if not re.fullmatch('[0-9a-f]{64}', expected) or sha_file(path, deadline=deadline) != expected:
            raise ValueError('native file drift')
    for name, expected in runtime['distributions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('native package version drift')
    for name, key in (('TNP', 'tnp'), ('NanoBodyBuilder2', 'nanobody_builder_cli'), ('hmmscan', 'hmmscan')):
        resolved = shutil.which(name, path=runtime['path'])
        if not resolved or Path(resolved).resolve() != Path(runtime['entrypoints'][key]).resolve():
            raise ValueError('native executable resolution drift')


def validate_inputs(population, rows):
    if population.get('schema') != SCHEMA or population.get('population_sha256') != digest(population['records']):
        raise ValueError('population identity mismatch')
    if len(population['records']) != 24 or population['profile_count'] != 72:
        raise ValueError('unregistered volume')
    expected = {r['sequence_hash'] for r in population['records']}
    if len(expected) != 24 or len(rows) != 24:
        raise ValueError('duplicate or missing inputs')
    sequences = {}
    for row in rows:
        sequence = row['sequence']
        if not isinstance(sequence, str) or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{80,160}', sequence):
            raise ValueError('invalid sequence')
        key = hashlib.sha256(sequence.encode()).hexdigest()
        if row['sequence_hash'] != key or key not in expected or key in sequences:
            raise ValueError('sequence identity mismatch')
        sequences[key] = sequence
    for row in population['records']:
        if sorted(row['mode_order']) != sorted(MODES):
            raise ValueError('invalid matched order')
    return sequences


def cpu_seconds():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def safe_profile(profile):
    """Discard backend text; retain measured values and fixed provenance."""
    a3d = profile.get('aggrescan3d', {})
    if a3d.get('status') != 'ok':
        raise ValueError('Aggrescan did not produce a measured profile')
    result = {k: profile[k] for k in FIELDS}
    result['aggrescan3d'] = {k: a3d[k] for k in A3D_FIELDS}
    values = [result[k] for k in FIELDS] + list(result['aggrescan3d'].values())
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('nonfinite native measurement')
    for key in ('input_structure_sha256', 'normalized_structure_sha256', 'sequence_hash'):
        value = a3d[key]
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('invalid structure provenance')
        result['aggrescan3d'][key] = value
    for key, expected in (('ph_scale', 'default-scale'),
                          ('surface_method', 'FreeSASA-2.0.1-NACCESS-LR100'),
                          ('mode', 'static-no-minimization')):
        if a3d.get(key) != expected:
            raise ValueError('native provenance changed')
        result['aggrescan3d'][key] = expected
    result['aggrescan3d'].update(status='ok', passed=a3d['passed'], radius_angstrom=a3d['radius_angstrom'])
    if type(a3d['passed']) is not bool or a3d['radius_angstrom'] != 10:
        raise ValueError('invalid native decision provenance')
    result['aggrescan3d']['citation_doi'] = '10.1093/nar/gkv359'
    return result


def worker(index, tasks, runtime, workdir, events):
    os.setsid()
    # Native libraries may print arbitrary text. Both output channels are
    # suppressed before imports; only parsed allowlisted records leave.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    workdir = Path(workdir)
    os.chdir(workdir)
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(workdir))
    os.environ['PATH'] = runtime['path']
    sys.path.insert(0, str(Path(runtime['entrypoints']['logging_shim']).parent))
    from nanobody_opt.pr100_filter import activate_filter, helper
    from nanobody_opt import rules
    activate_filter()
    from metanano.filters.developability import DevelopabilityFilter
    from metanano.utils.aggrescan3d_wrapper import Aggrescan3DWrapper
    from metanano.utils.tnp_wrapper import ProfileExecution
    from nanobody_opt.tnp_daemon import _TNPProfiler
    cfg = helper('aggrescan3d').developability_config().developability
    cfg.aggrescan3d.executable = runtime['entrypoints']['aggrescan']
    cfg.aggrescan3d.freesasa_executable = runtime['entrypoints']['freesasa']
    native = DevelopabilityFilter(cfg)
    native._tnp.tnp_executable = runtime['entrypoints']['tnp']
    scorer = Aggrescan3DWrapper(cfg.aggrescan3d.executable, cfg.aggrescan3d.freesasa_executable)
    requirements = rules.pinned_config(rules.ACTIVATION_EPOCH)['nanobody_requirements']
    warm = None
    for task in tasks:
        for mode in task['mode_order']:
            key, sequence = task['sequence_hash'], task['sequence']
            before, cpu = time.monotonic(), cpu_seconds()
            events.put({'event': 'begin', 'worker': index, 'sequence_hash': key, 'mode': mode, 'started': before})
            a3d_seconds = initialization_seconds = 0.0
            execution = ProfileExecution(timeout=300)

            def aggregation(seq, path, shared_execution=execution):
                nonlocal a3d_seconds
                started = time.monotonic()
                try:
                    return scorer.profile(seq, Path(path), shared_execution)
                finally:
                    a3d_seconds += time.monotonic() - started

            try:
                native._tnp.aggrescan3d = aggregation
                if mode == 'warm':
                    if warm is None:
                        start = time.monotonic()
                        warm = _TNPProfiler(str(workdir), structure_scorer=aggregation)
                        initialization_seconds = time.monotonic() - start
                    warm._structure_scorer = aggregation
                    profile = warm.profile_one(sequence, task['ordinal'])
                    execution.remaining()
                else:
                    result = native._tnp.profile(sequence, name='probe', execution=execution)
                    if result is None:
                        raise ValueError('TNP profile missing')
                    profile = result.to_profile_dict()
                measured = native.analyze_profile(sequence, profile).to_dict()
                clean = safe_profile(measured)
                reason = rules.developability_reason(measured, requirements, sequence=sequence) or 'passed'
                record = {'status': 'ok', 'profile': clean, 'reason': reason, 'eligible': reason == 'passed'}
            except BaseException as error:
                record = {'status': 'failed', 'error_class': type(error).__name__}
            record.update(event='result', sequence_hash=key, mode=mode, worker=index,
                          wall_seconds=time.monotonic() - before, cpu_seconds=cpu_seconds() - cpu,
                          a3d_seconds=a3d_seconds, initialization_seconds=initialization_seconds,
                          self_high_water_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          largest_reaped_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            events.put(record)
    events.put({'event': 'done', 'worker': index})


def descendants(pid=None):
    """Read PID ancestry only, never command lines or process environments."""
    pid = os.getpid() if pid is None else pid
    found = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            tasks = list((Path('/proc') / str(parent) / 'task').iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            if not task.name.isdecimal():
                continue
            try:
                children = (task / 'children').read_text().split()
            except FileNotFoundError:
                continue
            for child in children:
                if not child.isdecimal():
                    raise ValueError('invalid kernel PID record')
                child = int(child)
                if child not in found:
                    found.add(child)
                    pending.append(child)
    return found


def process_rss(pid):
    try:
        pages = int((Path('/proc') / str(pid) / 'statm').read_text().split()[1])
    except FileNotFoundError:
        return 0
    return pages * os.sysconf('SC_PAGE_SIZE')


def stop_descendants():
    """Linux subreaper owns even children that start new POSIX sessions."""
    for _ in range(4):
        for child in descendants():
            try:
                os.kill(child, signal.SIGSTOP)
            except ProcessLookupError:
                pass
    for child in descendants():
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            time.sleep(.02)
    return len(descendants())


def containment_rehearsal():
    """At most ten seconds, zero native/model imports, two sleeping children."""
    if sys.platform != 'linux':
        raise ValueError('Linux containment rehearsal required')
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise ValueError('subreaper unavailable')
    started = time.monotonic()
    code = ('import subprocess,sys,time; '
            'subprocess.Popen([sys.executable,"-c","import time; time.sleep(300)"],'
            'start_new_session=True); time.sleep(300)')
    process = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    observed = set()
    try:
        while time.monotonic() - started < 3:
            observed = descendants()
            if len(observed) >= 2:
                break
            time.sleep(.02)
        separate_sessions = len({os.getsid(pid) for pid in observed}) >= 2
    finally:
        remaining = stop_descendants()
        process.poll()
    return {'scope': 'Linux_process_containment_only', 'native_calls': 0, 'gpu_calls': 0,
            'observed_descendants': len(observed), 'distinct_sessions': separate_sessions,
            'remaining_descendants': remaining, 'elapsed_seconds': time.monotonic() - started,
            'passed': len(observed) >= 2 and separate_sessions and remaining == 0
                      and time.monotonic() - started < 10}


def run(population, sequences, runtime, output):
    begin, before_cpu = time.monotonic(), cpu_seconds()
    if sys.platform != 'linux' or not hasattr(os, 'sched_setaffinity'):
        raise ValueError('Linux subreaper and CPU affinity required')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise ValueError('cannot own escaped native process groups')
    limits = population['limits']
    if population['population_sha256'] != FROZEN_POPULATION:
        raise ValueError('population differs from reviewed registration')
    if limits != {'wall_seconds': 1200, 'cpu_affinity_count': 8,
                  'memory_bytes': 24 * 1024 ** 3, 'workers': 4, 'profile_seconds': 300}:
        raise ValueError('resource registration changed')
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 8:
        raise ValueError('eight available CPU slots required')
    os.sched_setaffinity(0, cpus[:8])
    verify_runtime(runtime, deadline=begin + 1190)
    if output.exists():
        raise ValueError('output already exists')
    output.mkdir(parents=True)
    (output / 'population.json').write_text(json.dumps(population, indent=2))
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    ctx, processes, records, active, done = mp.get_context('spawn'), [], [], {}, set()
    events = ctx.Queue()
    peak_rss, status = 0, 'complete'
    try:
        for index in range(4):
            private = output / ('worker' + str(index))
            private.mkdir()
            tasks = [dict(row, sequence=sequences[row['sequence_hash']]) for row in population['records'][index::4]]
            process = ctx.Process(target=worker, args=(index, tasks, runtime, str(private), events))
            process.start()
            processes.append(process)
        while len(done) != 4:
            now = time.monotonic()
            # Reserve ten seconds inside the registered wall cap for cleanup.
            if now - begin > limits['wall_seconds'] - 10:
                status = 'wall_timeout'
                break
            if any(now - started > limits['profile_seconds'] for started in active.values()):
                status = 'profile_timeout'
                break
            rss = sum(process_rss(pid) for pid in descendants() | {os.getpid()})
            peak_rss = max(peak_rss, rss)
            if rss > limits['memory_bytes']:
                status = 'memory_cap'
                break
            if any(p.exitcode not in (None, 0) for p in processes):
                status = 'worker_failed'
                break
            try:
                event = events.get(timeout=.05)
            except queue.Empty:
                continue
            index = event['worker']
            if event['event'] == 'begin':
                active[index] = event['started']
            elif event['event'] == 'done':
                done.add(index)
            else:
                active.pop(index, None)
                records.append(event)
                (output / 'measurements.json').write_text(json.dumps(records, allow_nan=False))
    finally:
        for p in processes:
            p.join(timeout=.1)
        remaining = stop_descendants()
    report = compare_results(population, records)
    report.update(status=status, remaining_descendants=remaining,
                  sampled_process_tree_rss_upper_sum_bytes=peak_rss,
                  population_sha256=population['population_sha256'], runtime_sha256=digest(runtime),
                  gpu_calls=0, cache_reads=False, publication_enabled=False)
    try:
        verify_runtime(runtime, deadline=begin + 1197)
        report['runtime_unchanged'] = True
    except Exception:
        report.update(runtime_unchanged=False, status='runtime_drift')
    elapsed = time.monotonic() - begin
    report.update(elapsed_seconds=elapsed, actual_cpu_seconds=cpu_seconds() - before_cpu,
                  allocated_cpu_seconds=8 * elapsed)
    if remaining or elapsed > limits['wall_seconds']:
        report['status'] = 'containment_or_wall_cap_failure'
    if report['status'] != 'complete' or report.get('runtime_unchanged') is not True:
        report['decision'] = 'NO_GO_EXECUTION'
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    if sys.argv[1:] == ['--self-test-containment']:
        try:
            report = containment_rehearsal()
        except Exception as error:
            report = {'passed': False, 'error_class': type(error).__name__}
        print(json.dumps(report))
        return 0 if report['passed'] else 1
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('population', 'input', 'runtime', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        population = json.loads(args.population.read_text())
        rows = json.loads(args.input.read_text())
        runtime = json.loads(args.runtime.read_text())
        sequences = validate_inputs(population, rows)
        report = run(population, sequences, runtime, args.output.resolve())
        print(json.dumps({k: report[k] for k in ('status', 'decision', 'received', 'missing', 'gate_flips',
                         'elapsed_seconds', 'actual_cpu_seconds', 'remaining_descendants')}))
        return 0 if report['decision'] == 'GO_TO_LARGER_EXPERIMENT_ONLY' else 1
    except Exception as error:
        print(json.dumps({'status': 'failed_preflight', 'error_class': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
