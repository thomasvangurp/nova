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
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanobody_opt.pr100_native_benchmark import SCHEMA, MODES, FIELDS, A3D_FIELDS, digest, compare_results
FROZEN_POPULATION = '1fe6a4bab9b41a055268036e89f088c998f4fe1046d56e68a2b75c20e3dd638d'
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                               'MKL_NUM_THREADS', 'OPENMM_CPU_THREADS')}


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


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
    if runtime.get('cpu_thread_settings') != THREADS:
        raise ValueError('native CPU thread configuration mismatch')
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
    # These are generated by the pinned pure analyzer, never backend log text.
    if type(profile.get('passed')) is not bool:
        raise ValueError('missing native decision')
    flags = profile.get('red_flags') or []
    if any(not isinstance(flag, str) or len(flag) > 512 or
           not any(flag.startswith(field + ' ') for field in FIELDS) for flag in flags):
        raise ValueError('unclassified native flag')
    result.update(passed=profile['passed'], red_flags=list(flags))
    return result


def verify_loaded_identity(runtime, destination, checked):
    """Compare actual import/bytecode/library origins to pre-frozen file hashes."""
    expected = dict(runtime['files'])
    stage = json.loads((ROOT / 'build_stamp.json').read_text())
    expected.update({str((ROOT / name).resolve()): value for name, value in stage['files'].items()})
    origins, candidates, unresolved = {}, set(), []
    for name, module in list(sys.modules.items()):
        for field in ('__file__', '__cached__'):
            value = getattr(module, field, None)
            if not isinstance(value, str):
                continue
            path = Path(value).resolve()
            if field == '__cached__' and not path.is_file():
                continue
            key = str(path)
            if key not in expected:
                unresolved.append({'kind': field, 'module': name,
                                   'origin_locator_sha256': hashlib.sha256(key.encode()).hexdigest()})
                continue
            origins[name + ':' + field] = key
            candidates.add(key)
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        key = str(Path(parts[5]).resolve())
        if key not in expected:
            unresolved.append({'kind': 'mapped_file', 'origin_locator_sha256': hashlib.sha256(key.encode()).hexdigest()})
        else:
            mapped.add(key); candidates.add(key)
    for key in candidates - checked:
        if sha_file(key) != expected[key]:
            raise ValueError('loaded runtime file drift')
        checked.add(key)
    receipt = {'module_origins': origins, 'mapped_files': sorted(mapped),
               'unregistered_origins': unresolved, 'checked_file_count': len(checked)}
    atomic_json(destination, receipt)
    if unresolved:
        raise ValueError('loaded runtime origin is not frozen')
    return receipt


def worker(index, tasks, runtime, workdir, events):
    if os.getpid() != os.getsid(0):
        os.setsid()
    # Native libraries may print arbitrary text. Both output channels are
    # suppressed before imports; only parsed allowlisted records leave.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    workdir = Path(workdir)
    os.chdir(workdir)
    os.environ.update(THREADS)
    os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(workdir))
    os.environ['PATH'] = runtime['path']
    sys.path.insert(0, str(Path(runtime['entrypoints']['logging_shim']).parent))
    events.put({'event': 'phase', 'worker': index, 'stage': 'imports'})
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
    checked_origins = set()
    verify_loaded_identity(runtime, workdir / 'loaded_identity.json', checked_origins)
    warm = None
    for task in tasks:
        for mode_position, mode in enumerate(task['mode_order'], 1):
            key, sequence = task['sequence_hash'], task['sequence']
            before, cpu = time.monotonic(), cpu_seconds()
            events.put({'event': 'begin', 'worker': index, 'sequence_hash': key, 'mode': mode, 'started': before})
            a3d_seconds = initialization_seconds = 0.0
            execution = ProfileExecution(timeout=300)
            stage = 'initialize' if mode == 'warm' else 'cli_native'

            def observe_stage(value):
                nonlocal stage
                stage = value

            def aggregation(seq, path, shared_execution=execution):
                nonlocal a3d_seconds
                observe_stage('aggrescan')
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
                        warm = _TNPProfiler(str(workdir), structure_scorer=aggregation,
                                            stage_observer=observe_stage)
                        initialization_seconds = time.monotonic() - start
                    warm._structure_scorer = aggregation
                    warm._stage_observer = observe_stage
                    profile = warm.profile_one(sequence, task['ordinal'])
                    execution.remaining()
                else:
                    result = native._tnp.profile(sequence, name='probe', execution=execution)
                    if result is None:
                        raise ValueError('TNP profile missing')
                    profile = result.to_profile_dict()
                if any(key not in profile for key in FIELDS):
                    raise ValueError('incomplete measured profile')
                stage = 'analyze'
                measured = native.analyze_profile(sequence, profile).to_dict()
                stage = 'sanitize'
                clean = safe_profile(measured)
                reason = rules.developability_reason(measured, requirements, sequence=sequence) or 'passed'
                record = {'status': 'ok', 'profile': clean, 'reason': reason, 'eligible': reason == 'passed'}
            except Exception as error:
                timed_out = isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or time.monotonic() - before >= 300
                record = {'status': 'profile_timeout' if timed_out else 'failed',
                          'error_class': type(error).__name__, 'stage': stage}
            except BaseException as error:
                events.put({'event': 'fatal', 'worker': index, 'sequence_hash': key,
                            'mode': mode, 'stage': stage, 'error_class': type(error).__name__})
                raise
            record.update(event='result', sequence_hash=key, mode=mode, worker=index,
                          wall_seconds=time.monotonic() - before, cpu_seconds=cpu_seconds() - cpu,
                          a3d_seconds=a3d_seconds, initialization_seconds=initialization_seconds,
                          mode_position=mode_position,
                          self_high_water_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          largest_reaped_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            events.put({'event': 'profile_computed', 'worker': index, 'sequence_hash': key,
                        'mode': mode, 'started': before,
                        'profile_completed_at': before + record['wall_seconds']})
            audit_started = time.monotonic()
            if record['status'] == 'profile_timeout':
                record['identity_status'] = 'not_checked_after_timeout'
            else:
                try:
                    verify_loaded_identity(runtime, workdir / 'loaded_identity.json', checked_origins)
                    record['identity_status'] = 'verified'
                except Exception as error:
                    record.update(identity_status='failed', identity_error_class=type(error).__name__)
                    if record['status'] == 'ok':
                        record.update(status='identity_failed', stage='loaded_identity')
            record['identity_audit_seconds_excluded_from_profile_time'] = time.monotonic() - audit_started
            events.put(record)
            if record['status'] == 'profile_timeout' or record.get('identity_status') == 'failed':
                return  # Every mode aborts the entire trial; never reuse timed-out warm state.
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


def dedicated_controller():
    if os.getpid() != os.getsid(0) or descendants():
        raise ValueError('dedicated session leader with no existing descendants required')
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise ValueError('subreaper unavailable')


def available_memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            values['host_available_bytes'] = int(line.split()[1]) * 1024
    root = Path('/sys/fs/cgroup')
    if (root / 'memory.max').exists() and (root / 'memory.current').exists():
        maximum = (root / 'memory.max').read_text().strip()
        if maximum.isdecimal():
            values['cgroup_headroom_bytes'] = max(0, int(maximum) - int((root / 'memory.current').read_text()))
    values['available_bytes'] = min(values.values())
    return values


def stop_descendants(processes=(), *, timeout=5):
    """Stop to a bounded fixed point, then kill/reap only this controller's tree."""
    deadline = time.monotonic() + timeout
    previous, stable, scans = None, 0, 0
    while time.monotonic() < deadline - .5:
        current = descendants()
        scans += 1
        for child in current:
            try:
                os.kill(child, signal.SIGSTOP)
            except ProcessLookupError:
                pass
        stable = stable + 1 if current == previous else 0
        previous = current
        if stable >= 2:
            break
        time.sleep(.005)
    victims = descendants()
    for child in victims:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Reap direct registered subprocesses first; never steal their wait status.
    exitcodes = {}
    for process in processes:
        try:
            process.wait(timeout=max(.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        exitcodes[str(process.pid)] = process.poll()
    while time.monotonic() < deadline:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            time.sleep(.01)
    return {'remaining_descendants': len(descendants()), 'fixed_point_reached': stable >= 2,
            'freeze_scan_count': scans, 'killed_descendants': len(victims),
            'worker_exitcodes': exitcodes}


def containment_rehearsal():
    """Dedicated Linux process; two bounded cases, zero native imports permitted."""
    if sys.platform != 'linux':
        raise ValueError('Linux containment rehearsal required')
    if not sys.flags.no_site:
        raise ValueError('containment rehearsal requires interpreter -S')
    dedicated_controller()
    started = time.monotonic()
    forbidden = {'torch', 'openmm', 'ImmuneBuilder', 'theraprofnano'}
    if forbidden & set(sys.modules):
        raise ValueError('native modules already imported')
    guard_hits = []
    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in forbidden:
                guard_hits.append(fullname.split('.')[0])
                raise ImportError('native import forbidden in containment rehearsal')
    guard = Guard()
    sys.meta_path.insert(0, guard)
    cases = []
    try:
        for kind in ('detached_sleeper', 'bounded_fork_race'):
            count = 1 if kind == 'detached_sleeper' else 16
            code = ('import subprocess,sys,time\n'
                    f'for i in range({count}):\n'
                    ' subprocess.Popen([sys.executable,"-S","-c","import time; time.sleep(300)"],start_new_session=True)\n'
                    ' time.sleep(.02)\n'
                    'time.sleep(300)\n')
            process = subprocess.Popen([sys.executable, '-S', '-c', code], start_new_session=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            observed, sessions = set(), set()
            try:
                target = 2 if kind == 'detached_sleeper' else 3
                setup_deadline = min(started + 7, time.monotonic() + 1.5)
                while time.monotonic() < setup_deadline:
                    observed = descendants()
                    if len(observed) >= target:
                        break
                    time.sleep(.005)
                resolved = 0
                for pid in observed:
                    try:
                        sessions.add(os.getsid(pid)); resolved += 1
                    except ProcessLookupError:
                        pass
            finally:
                cleanup = stop_descendants([process], timeout=max(.1, min(2, started + 9 - time.monotonic())))
            cases.append(dict(kind=kind, observed_descendants=len(observed),
                              resolved_session_pids=resolved, distinct_sessions=len(sessions),
                              fork_count_permitted=count, **cleanup))
    finally:
        sys.meta_path.remove(guard)
    elapsed = time.monotonic() - started
    return {'scope': 'Linux_process_containment_only', 'native_calls_permitted': 0,
            'gpu_calls_permitted': 0, 'controller_native_import_guard_installed': True,
            'guarded_module_roots': sorted(forbidden),
            'controller_native_import_guard_hits': guard_hits, 'cases': cases,
            'remaining_descendants': len(descendants()), 'elapsed_seconds': elapsed,
            'passed': len(cases) == 2 and all(c['fixed_point_reached'] and
                c['remaining_descendants'] == 0 and c['distinct_sessions'] >= 2
                and c['observed_descendants'] >= (2 if c['kind'] == 'detached_sleeper' else 3)
                for c in cases) and not guard_hits and elapsed < 10}


class FileEvents:
    """Atomic private event files avoid a killed writer blocking a shared pipe."""
    def __init__(self, root):
        self.root = Path(root)

    def put(self, event):
        kind = event['event']
        if kind in ('result', 'fatal'):
            if kind == 'fatal':
                event = dict(event, status='fatal')
            name = event['sequence_hash'] + '-' + event['mode'] + '.json'
            atomic_json(self.root / 'results' / name, event)
            (self.root / 'active.json').unlink(missing_ok=True)
        elif kind in ('begin', 'profile_computed'):
            atomic_json(self.root / 'active.json', event)
        elif kind == 'done':
            atomic_json(self.root / 'done.json', event)
            (self.root / 'active.json').unlink(missing_ok=True)
        elif kind == 'phase':
            atomic_json(self.root / 'phase.json', event)
        else:
            raise ValueError('unknown private worker event')


def worker_main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    events = FileEvents(spec['workdir'])
    try:
        worker(spec['index'], spec['tasks'], spec['runtime'], spec['workdir'], events)
        return 0
    except BaseException as error:
        atomic_json(Path(spec['workdir']) / 'failure.json',
                    {'stage': 'worker_startup_or_fatal', 'error_class': type(error).__name__})
        return 1


def run(population, sequences, runtime, output):
    begin, before_cpu = time.monotonic(), cpu_seconds()
    if output.exists():
        raise ValueError('output already exists')
    output.mkdir(parents=True)
    processes, records, directories, conditions = [], {}, [], []
    owned, verified, started, peak_rss = False, False, False, 0
    prior_term_handler = None
    cleanup = {'remaining_descendants': None, 'fixed_point_reached': False, 'worker_exitcodes': {}}
    report = {'scope': 'native_reuse_research_only', 'conditions': conditions,
              'native_profiles_permitted': 72, 'gpu_calls_permitted': 0,
              'cache_reads_permitted': False, 'publication_enabled_by_configuration': False,
              'kernel_enforced_aggregate_memory_limit': False,
              'rss_sampling_interval_seconds': .05, 'rss_sum_double_counts_shared_pages': True,
              'population_sha256': population['population_sha256'], 'runtime_sha256': digest(runtime)}
    try:
        if sys.platform != 'linux' or not hasattr(os, 'sched_setaffinity'):
            raise ValueError('Linux subreaper and CPU affinity required')
        dedicated_controller(); owned = True
        def interrupted(signum, frame):
            raise InterruptedError('trial termination requested')
        prior_term_handler = signal.signal(signal.SIGTERM, interrupted)
        if population['population_sha256'] != FROZEN_POPULATION:
            raise ValueError('population differs from reviewed registration')
        limits = population['limits']
        expected_limits = dict(wall_seconds=1200, cpu_affinity_count=8, workers=4, profile_seconds=300,
                               rss_abort_threshold_bytes=24*1024**3,
                               minimum_available_memory_bytes=96*1024**3,
                               available_memory_abort_bytes=64*1024**3)
        if limits != expected_limits or population.get('runtime_configuration_revision') != 2:
            raise ValueError('resource registration changed')
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < 8:
            raise ValueError('eight available CPU slots required')
        os.sched_setaffinity(0, cpus[:8])
        report['start_memory'] = available_memory()
        if report['start_memory']['available_bytes'] < limits['minimum_available_memory_bytes']:
            raise ValueError('insufficient measured memory headroom')
        verify_runtime(runtime, deadline=begin + 1190); verified = True
        atomic_json(output / 'population.json', population)
        atomic_json(output / 'runtime.json', runtime)
        for index in range(4):
            private = output / ('worker' + str(index)); private.mkdir()
            (private / 'results').mkdir()
            tasks = [dict(row, sequence=sequences[row['sequence_hash']]) for row in population['records'][index::4]]
            spec = dict(index=index, tasks=tasks, runtime=runtime, workdir=str(private))
            atomic_json(private / 'spec.json', spec)
            child_env = dict(THREADS, PATH=runtime['path'], LANG='C.UTF-8',
                             PYTHONPATH=str(ROOT) + ':' + str(Path(runtime['entrypoints']['logging_shim']).parent),
                             PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='',
                             ROCR_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                             TMPDIR=str(private), MPLCONFIGDIR=str(private/'mpl-cache'),
                             XDG_CACHE_HOME=str(private/'cache'), TORCH_HOME=str(private/'torch-cache'))
            if runtime.get('library_path'):
                child_env['LD_LIBRARY_PATH'] = runtime['library_path']
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(private / 'spec.json')],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       start_new_session=True, env=child_env)
            processes.append(process); directories.append((private, tasks))
        started = True
        while True:
            done = 0
            for index, (private, tasks) in enumerate(directories):
                for task in tasks:
                    for mode in MODES:
                        key = (task['sequence_hash'], mode)
                        path = private / 'results' / (key[0] + '-' + mode + '.json')
                        if key not in records and path.is_file():
                            records[key] = json.loads(path.read_text())
                if (private / 'done.json').is_file():
                    done += 1
                active = private / 'active.json'
                if active.is_file():
                    try:
                        active_row = json.loads(active.read_text())
                    except FileNotFoundError:
                        active_row = None
                    if active_row and (active_row['sequence_hash'], active_row['mode']) not in records:
                        if active_row.get('profile_completed_at', time.monotonic()) - active_row['started'] >= 300:
                            records[(active_row['sequence_hash'], active_row['mode'])] = {
                                'sequence_hash': active_row['sequence_hash'], 'mode': active_row['mode'],
                                'worker': index, 'status': 'profile_timeout', 'stage': 'controller_deadline',
                                'wall_seconds': time.monotonic() - active_row['started'], 'cpu_seconds': None}
                if (private / 'failure.json').is_file():
                    conditions.append('worker_failed')
                if processes[index].poll() is not None and not (private / 'done.json').is_file():
                    if not any(r['worker'] == index and r['status'] == 'profile_timeout' for r in records.values()):
                        conditions.append('worker_incomplete_exit')
            if any(r['status'] == 'profile_timeout' for r in records.values()):
                conditions.append('profile_timeout')
            if any(r['status'] in ('fatal', 'identity_failed') or r.get('identity_status') == 'failed'
                   for r in records.values()):
                conditions.append('fatal_worker_exception')
            atomic_json(output / 'measurements.json', list(records.values()))
            memory = available_memory()
            report['last_memory'] = memory
            rss = sum(process_rss(pid) for pid in descendants() | {os.getpid()})
            peak_rss = max(peak_rss, rss)
            if rss > limits['rss_abort_threshold_bytes']:
                conditions.append('sampled_rss_abort')
            if memory['available_bytes'] < limits['available_memory_abort_bytes']:
                conditions.append('available_memory_abort')
            if time.monotonic() - begin >= 1190:
                conditions.append('wall_timeout')
            atomic_json(output / 'progress.json', {'received': len(records), 'done_workers': done,
                        'elapsed_seconds': time.monotonic()-begin, 'conditions': list(dict.fromkeys(conditions))})
            if conditions or done == 4:
                break
            time.sleep(.05)
    except BaseException as error:
        conditions.append('execution_error' if started else 'preflight_error')
        report['error_class'] = type(error).__name__
        report['error_stage'] = 'controller_execution' if started else 'preflight'
    finally:
        if owned:
            report['pre_cleanup_worker_exitcodes'] = {str(p.pid): p.poll() for p in processes}
            try:
                cleanup = stop_descendants(processes, timeout=max(.1, min(5, begin+1196-time.monotonic())))
            except Exception as error:
                conditions.append('cleanup_error')
                report['cleanup_error_class'] = type(error).__name__
            if cleanup['remaining_descendants'] or not cleanup['fixed_point_reached']:
                conditions.append('containment_failure')
        if verified:
            try:
                verify_runtime(runtime, deadline=begin + 1197)
                report['runtime_unchanged'] = True
            except TimeoutError:
                conditions.append('post_verification_timeout'); report['runtime_unchanged'] = None
            except Exception as error:
                conditions.append('post_verification_failed'); report['runtime_unchanged'] = False
                report['post_verification_error_class'] = type(error).__name__
    try:
        report.update(compare_results(population, list(records.values())))
    except Exception as error:
        conditions.append('result_integrity_error')
        report.update(decision='NO_GO_EXECUTION', result_error_class=type(error).__name__)
    elapsed = time.monotonic() - begin
    if elapsed > 1200:
        conditions.append('wall_cap_failure')
    report.update(status='complete' if not conditions else conditions[0], conditions=list(dict.fromkeys(conditions)),
                  elapsed_seconds=elapsed, actual_cpu_seconds=cpu_seconds()-before_cpu,
                  allocated_cpu_seconds=8*elapsed, sampled_process_tree_rss_upper_sum_bytes=peak_rss,
                  cleanup=cleanup, remaining_descendants=cleanup['remaining_descendants'])
    if conditions:
        report['decision'] = 'NO_GO_EXECUTION'
    atomic_json(output / 'report.json', report)
    if prior_term_handler is not None:
        signal.signal(signal.SIGTERM, prior_term_handler)
    return report


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        return worker_main(sys.argv[2])
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
    output_preexisted = args.output.exists()
    try:
        population = json.loads(args.population.read_text())
        rows = json.loads(args.input.read_text())
        runtime = json.loads(args.runtime.read_text())
        sequences = validate_inputs(population, rows)
        report = run(population, sequences, runtime, args.output.resolve())
        print(json.dumps({k: report.get(k) for k in ('status', 'decision', 'conditions', 'received', 'missing', 'complete_pairs',
                         'failure_counts', 'status_mismatches', 'gate_flips', 'elapsed_seconds',
                         'actual_cpu_seconds', 'remaining_descendants')}))
        return 0 if report['decision'] == 'GO_TO_LARGER_EXPERIMENT_ONLY' else 1
    except Exception as error:
        receipt = {'status': 'failed_execution' if args.output.exists() and not output_preexisted else 'failed_preflight',
                   'decision': 'NO_GO_EXECUTION', 'error_class': type(error).__name__}
        if not output_preexisted:
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                if not (args.output/'report.json').exists():
                    atomic_json(args.output/'report.json', receipt)
            except OSError:
                receipt['failure_receipt_write_failed'] = True
        print(json.dumps(receipt))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
