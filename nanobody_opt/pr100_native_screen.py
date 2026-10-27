"""Resident CPU / CUDA screens: never authoritative publication evidence."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = 'pr100-native-screen-v1'
BACKENDS = ('resident_cpu', 'cuda_refine')
PINNED_SOURCES = {
    'ImmuneBuilder.refine': 'e33d4bdc3e8354296ede6bde0e2fb438368e5e2fbe87ec0d7b387be0d60ebaf0',
    'ImmuneBuilder.NanoBodyBuilder2': '927ac1cb381de0266fc4bd27828b48ca0443703b6b00c13490062e48e7c727dc',
    'theraprofnano.CDR_Profiler.CDR3_Conf_Assigner': '3b77f628bab30927b8934362c2759c06949d5393b4f5401e956334ab21033c44',
    'theraprofnano.Hydrophobicity_and_Charge_Profiler.Hydrophobicity_and_Charge_Assigner': '71c4fcd4fbe3aa2892598a54dfb1d208265a31f30f455efc8a9c85ea10af7c46',
}
PINNED_VERSIONS = {'torch': '2.7.1', 'openmm': '8.5.1', 'pdbfixer': '1.12.0', 'numpy': '1.26.4'}
_STATE = {}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic(path, value):
    from nanobody_opt.pr100_live_gpu import atomic as write
    write(path, value)

def asset_stat(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]

def capture_weights():
    spec = importlib.util.find_spec('ImmuneBuilder.NanoBodyBuilder2')
    root, captured = Path(spec.origin).resolve().parent / 'trained_model', {}
    for index in range(1, 5):
        path = root / f'nanobody_model_{index}'
        before = asset_stat(path)
        captured[str(index)] = dict(path=str(path), sha256=file_hash(path), stat=before)
        if asset_stat(path) != before:
            raise ValueError('model_changed_during_hash')
    return captured

def check_weights(captured, root):
    if set(captured) != {'1', '2', '3', '4'}:
        raise ValueError('incomplete_model_identity')
    for index, asset in captured.items():
        expected = (root / f'nanobody_model_{index}').resolve()
        if expected != Path(asset['path']).resolve() or asset_stat(expected) != asset['stat']:
            raise ValueError('model_identity_changed')

def protocol(backend):
    if backend not in BACKENDS:
        raise ValueError('invalid_screen_backend')
    return {'schema': SCHEMA, 'backend': backend, 'authoritative': False,
            'prediction_device': 'cpu', 'torch_threads': 1,
            'refinement_platform': 'CUDA' if backend == 'cuda_refine' else 'CPU',
            'checks': True, 'tries': 3, 'iterations': 6,
            'sources': PINNED_SOURCES, 'versions': PINNED_VERSIONS}

def settings(job):
    value = job.get('native_screen', {})
    if not isinstance(value, dict) or type(value.get('enabled', False)) is not bool:
        raise ValueError('invalid_screen_configuration')
    backend = value.get('backend', 'resident_cpu')
    protocol(backend)
    workers, cpu = value.get('workers_per_gpu', 4), value.get('workers', job.get('workers', 48))
    minimum, gpus = value.get('gpu_minimum_free_mib', 8192), value.get('gpus', job.get('gpus', []))
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('screen_context_limit')
    if type(cpu) is not int or not 1 <= cpu <= 64:
        raise ValueError('screen_cpu_limit')
    if type(minimum) is not int or not 8192 <= minimum <= 200000:
        raise ValueError('screen_memory_limit')
    if (not isinstance(gpus, list) or len(gpus) > 4 or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError('invalid_screen_devices')
    return dict(enabled=value.get('enabled', False), backend=backend,
                workers_per_gpu=workers, workers=cpu, gpu_minimum_free_mib=minimum, gpus=gpus)

@contextmanager
def refinement_observer(native, builder, *, cuda):
    """Private process only; original retry/check and model-fallback code stays."""
    original, alias, simulation = native.refine, builder.refine, native.app.Simulation
    if alias is not original:
        raise ValueError('unexpected_refinement_alias')
    evidence = {'calls': [], 'platforms': []}
    def observed_simulation(*args, **kwargs):
        if cuda:
            if len(args) > 3 or kwargs.get('platform') is not None:
                raise ValueError('unexpected_explicit_platform')
            kwargs = dict(kwargs, platform=native.Platform.getPlatformByName('CUDA'))
        value = simulation(*args, **kwargs)
        platform = value.context.getPlatform()
        if platform.getName() != ('CUDA' if cuda else 'CPU'):
            raise ValueError('platform_mismatch')
        props = {name: platform.getPropertyValue(value.context, name)
                 for name in ('Precision', 'DeviceIndex', 'Threads', 'DeterministicForces')
                 if name in platform.getPropertyNames()}
        if cuda and (props.get('Precision') != 'single' or props.get('DeviceIndex') != '0'):
            raise ValueError('unmeasured_cuda_configuration')
        if not cuda and props.get('Threads') != '1':
            raise ValueError('unmeasured_cpu_configuration')
        evidence['platforms'].append({'name': platform.getName(), 'properties': props})
        return value
    def observed_refine(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        if (bound.arguments.get('check_for_strained_bonds') is not True or
                bound.arguments.get('tries') != 3 or bound.arguments.get('n') != 6 or
                bound.arguments.get('n_threads') != 1):
            raise ValueError('native_refinement_settings_changed')
        started, event = time.monotonic(), {'native_success': False}
        try:
            if cuda:
                bound.arguments['n_threads'] = -1
                result = original(*bound.args, **bound.kwargs)
            else:
                result = original(*args, **kwargs)
            event['native_success'] = bool(result)
            return result
        finally:
            event['seconds'] = time.monotonic() - started
            evidence['calls'].append(event)
    native.app.Simulation = observed_simulation
    native.refine = builder.refine = observed_refine
    try:
        yield evidence
    finally:
        native.app.Simulation = simulation
        native.refine, builder.refine = original, alias

def _initialize_worker(request):
    global _STATE
    _STATE = {'request': request, 'initialization_stage': 'environment'}
    try:
        cuda = request['settings']['backend'] == 'cuda_refine'
        if not cuda and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('resident_cpu_requires_hidden_cuda')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENMM_CPU_THREADS'):
            os.environ[key] = '1'
        _STATE['initialization_stage'] = 'native_import'
        modules = {name: importlib.import_module(name) for name in PINNED_SOURCES}
        _STATE['initialization_stage'] = 'source_identity'
        sources = {name: file_hash(module.__file__) for name, module in modules.items()}
        if sources != PINNED_SOURCES:
            raise ValueError('unregistered_native_source')
        _STATE['initialization_stage'] = 'version_identity'
        versions = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
        if versions != PINNED_VERSIONS:
            raise ValueError('unregistered_native_version')
        import torch
        torch.set_num_threads(1)
        native, builder = modules['ImmuneBuilder.refine'], modules['ImmuneBuilder.NanoBodyBuilder2']
        weights_root = Path(builder.__file__).resolve().parent / 'trained_model'
        _STATE['initialization_stage'] = 'model_identity'
        check_weights(request['model_assets'], weights_root)
        weights = {i: asset['sha256'] for i, asset in request['model_assets'].items()}
        def no_download(*args, **kwargs):
            raise ValueError('native_model_download_forbidden')
        builder.download_file = no_download
        context = refinement_observer(native, builder, cuda=cuda)
        evidence = context.__enter__()
        from nanobody_opt.tnp_daemon import _TNPProfiler
        available = torch.cuda.is_available
        _STATE['initialization_stage'] = 'model_load'
        try:
            torch.cuda.is_available = lambda: False
            profiler = _TNPProfiler(request['temporary_directory'])
        finally:
            torch.cuda.is_available = available
        if profiler.builder.device != 'cpu':
            raise ValueError('prediction_must_remain_cpu')
        check_weights(request['model_assets'], weights_root)
        _STATE['initialization_stage'] = 'analyzer_setup'
        from nanobody_opt.pr100_filter import activate_filter, helper
        activate_filter()
        from metanano.filters.developability import DevelopabilityFilter
        from metanano.utils.aggrescan3d_wrapper import Aggrescan3DWrapper
        config = helper('aggrescan3d').developability_config().developability
        tools = Path(request['native_tools'])
        config.aggrescan3d.executable = str(tools / 'aggrescan3d-rs')
        config.aggrescan3d.freesasa_executable = str(tools / 'freesasa')
        identity = dict(sources=sources, model_weights=weights, versions=versions,
                        python=list(sys.version_info[:3]), prediction_device='cpu',
                        accelerator_source_sha256=file_hash(__file__),
                        hydrogen_context_platform_observed=False,
                        model_hash_scope='coordinator_hash_worker_path_stat_before_after_load')
        _STATE.update(profiler=profiler, analyzer=DevelopabilityFilter(config),
                      scorer=Aggrescan3DWrapper(config.aggrescan3d.executable, config.aggrescan3d.freesasa_executable),
                      context=context, evidence=evidence, identity=identity)
    except Exception as error:
        _STATE['initialization_error'] = type(error).__name__

def _screen_one(sequence):
    request = _STATE['request']
    key = hashlib.sha256(sequence.encode()).hexdigest()
    backend = request['settings']['backend']
    record = dict(schema=SCHEMA, evidence_backend=backend, authoritative=False,
                  sequence_hash=key, epoch=request['challenge']['epoch'],
                  challenge_sha256=digest(request['challenge']),
                  protocol_sha256=digest(protocol(backend)), request_sha256=request['request_sha256'])
    started = time.monotonic()
    try:
        if _STATE.get('initialization_error'):
            raise RuntimeError('accelerator_initialization_failed')
        from metanano.utils.tnp_wrapper import ProfileExecution
        from nanobody_opt import rules
        from scripts.pr100_native_probe import safe_profile
        execution = ProfileExecution(timeout=request['candidate_timeout_seconds'])
        profiler, evidence = _STATE['profiler'], _STATE['evidence']
        evidence['calls'].clear(); evidence['platforms'].clear()
        profiler._structure_scorer = lambda seq, path: _STATE['scorer'].profile(seq, Path(path), execution)
        value = profiler.profile_one(sequence, 0)
        execution.remaining()
        if not evidence['calls'] or not evidence['calls'][-1]['native_success'] or not evidence['platforms']:
            raise ValueError('refinement_not_verified')
        profile = safe_profile(_STATE['analyzer'].analyze_profile(sequence, value).to_dict())
        reason = rules.developability_reason(profile, request['requirements'], sequence=sequence)
        record.update(profile=profile, screen_outcome='rejected' if reason else 'passed', screen_reason=reason,
                      runtime_identity_sha256=digest(_STATE['identity']),
                      provenance={**_STATE['identity'], 'gpu': request.get('gpu'),
                                  'refinement': dict(evidence), 'final_refinement_success': True})
    except Exception as error:
        record.update(screen_outcome='inconclusive', error=True,
                      error_kind=_STATE.get('initialization_error', type(error).__name__),
                      initialization_stage=_STATE.get('initialization_stage') if _STATE.get('initialization_error') else None)
    record['elapsed_seconds'] = time.monotonic() - started
    atomic(Path(request['results_directory']) / (key + '.json'), record)
    return key

def _shard(request):
    import multiprocessing
    request['temporary_directory'] = str(Path(request['results_directory']).parent / 'temporary')
    Path(request['temporary_directory']).mkdir(parents=True, exist_ok=True)
    with multiprocessing.get_context('spawn').Pool(min(request['workers'], len(request['sequences'])),
            initializer=_initialize_worker, initargs=(request,)) as pool:
        for _ in pool.imap_unordered(_screen_one, request['sequences'], chunksize=1):
            pass

def _coordinate(request):
    """Dedicated subreaper owns scientific workers and cooperative leases."""
    from nanobody_opt.pr100_live_gpu import _controller, ENV_KEYS
    control = _controller()
    control.dedicated_controller()
    root, children = Path(request['directory']), []
    receipt = {'schema': SCHEMA, 'status': 'running'}
    def interrupted(*_):
        raise TimeoutError('screen_deadline')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, request['timeout_seconds'] - 10)
    started = time.monotonic()
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        request['model_assets'] = capture_weights()
        atomic(root / 'runtime-assets.json', request['model_assets'])
        configured = request['settings']
        cuda = configured['backend'] == 'cuda_refine'
        devices = configured['gpus'] if cuda else [None]
        for index, gpu in enumerate(devices):
            sequences = request['sequences'][index::len(devices)]
            if not sequences:
                continue
            directory = root / ('cpu' if gpu is None else f'gpu-{gpu}')
            directory.mkdir()
            cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
            workers = configured['workers_per_gpu'] if cuda else min(cores, configured['workers'])
            shard = {**request, 'sequences': sequences, 'gpu': gpu, 'workers': workers,
                     'results_directory': str(directory / 'results')}
            atomic(directory / 'request.json', shard)
            env = {key: value for key, value in os.environ.items() if key in ENV_KEYS}
            env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            command = [sys.executable, '-B', '-m', 'nanobody_opt.pr100_native_screen',
                       '--shard', str(directory / 'request.json')]
            if cuda:
                lease = {'gpu': gpu, 'ttl_seconds': request['timeout_seconds'] - 14,
                         'minimum_free_mib': configured['gpu_minimum_free_mib'], 'command': command,
                         'env': env, 'cwd': request['release'], 'lock_directory': request['gpu_lock_directory'],
                         'scorer': request['scorer'], 'receipt': str(directory / 'lease.json')}
                atomic(directory / 'lease-job.json', lease)
                command = [sys.executable, '-B', '-m', 'nanobody_opt.pr100_live_gpu',
                           '--job', str(directory / 'lease-job.json')]
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''
            children.append(subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL, start_new_session=True, env=env))
        while any(child.poll() is None for child in children):
            time.sleep(.1)
        receipt['status'] = 'finished'
    except BaseException as error:
        receipt.update(status='interrupted', error_kind=type(error).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt['cleanup'] = control.stop_descendants(children, timeout=4)
        receipt['elapsed_seconds'] = time.monotonic() - started
        atomic(root / 'coordinator.json', receipt)

def _guard(request, *, launch=subprocess.Popen, control=None):
    """Survive coordinator death; adopt and remove its detached children."""
    from nanobody_opt.pr100_live_gpu import _controller
    control = control or _controller()
    control.dedicated_controller()
    children, receipt = [], {'schema': SCHEMA, 'status': 'running'}
    def interrupted(*_):
        raise TimeoutError('screen_guard_deadline')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, request['timeout_seconds'] - 5)
    try:
        command = [sys.executable, '-B', '-m', 'nanobody_opt.pr100_native_screen',
                   '--coordinate', str(Path(request['directory']) / 'request.json')]
        child = launch(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, start_new_session=True)
        children.append(child)
        receipt['coordinator_exit_code'] = child.wait()
        receipt['status'] = 'finished'
    except BaseException as error:
        receipt.update(status='interrupted', error_kind=type(error).__name__)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt['cleanup'] = control.stop_descendants(children, timeout=4)
        atomic(Path(request['directory']) / 'guard.json', receipt)

def valid_record(record, *, request, sequence):
    try:
        return _valid_record(record, request=request, sequence=sequence)
    except (ValueError, TypeError, KeyError, AttributeError):
        return False

def _valid_record(record, *, request, sequence):
    backend = request['settings']['backend']
    key = hashlib.sha256(sequence.encode()).hexdigest()
    provenance = record.get('provenance', {})
    identity = {k: provenance[k] for k in ('sources', 'model_weights', 'versions', 'python',
                'prediction_device', 'accelerator_source_sha256', 'hydrogen_context_platform_observed', 'model_hash_scope') if k in provenance}
    weights = identity.get('model_weights', {})
    calls = provenance.get('refinement', {}).get('calls', [])
    platforms = provenance.get('refinement', {}).get('platforms', [])
    expected = 'CUDA' if backend == 'cuda_refine' else 'CPU'
    if (len(identity) != 8 or identity.get('versions') != PINNED_VERSIONS
            or identity.get('model_hash_scope') != 'coordinator_hash_worker_path_stat_before_after_load'
            or identity.get('prediction_device') != 'cpu' or set(weights) != {'1', '2', '3', '4'}
            or weights != request.get('expected_model_weights')
            or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in weights.values())
            or not calls or calls[-1].get('native_success') is not True or not platforms
            or any(p.get('name') != expected for p in platforms)
            or record.get('profile', {}).get('aggrescan3d', {}).get('sequence_hash') != key):
        return False
    for platform in platforms:
        props = platform.get('properties', {})
        if backend == 'cuda_refine' and (props.get('Precision') != 'single' or props.get('DeviceIndex') != '0'):
            return False
        if backend == 'resident_cpu' and props.get('Threads') != '1':
            return False
    return (record.get('schema') == SCHEMA and record.get('evidence_backend') == backend
            and record.get('authoritative') is False and record.get('sequence_hash') == key
            and record.get('epoch') == request['challenge']['epoch']
            and record.get('challenge_sha256') == digest(request['challenge'])
            and record.get('request_sha256') == request['request_sha256']
            and record.get('protocol_sha256') == digest(protocol(backend))
            and record.get('runtime_identity_sha256') == digest(identity)
            and record.get('screen_outcome') in ('passed', 'rejected') and not record.get('error')
            and provenance.get('sources') == PINNED_SOURCES
            and provenance.get('accelerator_source_sha256') == file_hash(__file__)
            and provenance.get('final_refinement_success') is True)

def screen_batch(sequences, *, job, directory, timeout_seconds, on_result=None):
    """All returned profiles are screens; missing/error rows need fresh CPU."""
    sequences = list(sequences)
    began, configured = time.monotonic(), settings(job)
    backend = configured['backend']
    empty = {'schema': SCHEMA, 'status': 'unavailable', 'profiles': {},
             'provenance': {'authoritative': False, 'evidence_backend': backend}}
    if not configured['enabled'] or not sequences or (backend == 'cuda_refine' and not configured['gpus']):
        return {**empty, 'reason': 'disabled_or_no_devices'}
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 25 <= timeout_seconds <= 420):
        return {**empty, 'reason': 'insufficient_bounded_window'}
    if len(set(sequences)) != len(sequences):
        raise ValueError('duplicate_screen_input')
    from nanobody_opt import rules
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    request = {'schema': SCHEMA, 'sequences': list(sequences), 'challenge': job['challenge'],
               'requirements': rules.pinned_config(job['challenge']['epoch'])['nanobody_requirements'],
               'settings': configured, 'timeout_seconds': float(timeout_seconds), 'directory': str(root),
               'native_tools': job['native_tools'], 'release': job['release'],
               'scorer': job.get('scorer'), 'gpu_lock_directory': job.get('gpu_lock_directory'),
               'candidate_timeout_seconds': min(300.0, float(timeout_seconds))}
    request['request_sha256'] = digest(request)
    atomic(root / 'request.json', request)
    command = [sys.executable, '-B', '-m', 'nanobody_opt.pr100_native_screen', '--guard', str(root / 'request.json')]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        process.wait(timeout=timeout_seconds + 5)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=2)
        raise
    profiles = {}
    try:
        receipt = json.loads((root / 'guard.json').read_text())
        cleanup = receipt.get('cleanup', {})
        if cleanup.get('remaining_descendants') != 0 or cleanup.get('fixed_point_reached') is not True:
            return {**empty, 'reason': 'accelerator_cleanup_unverified'}
        devices = configured['gpus'] if backend == 'cuda_refine' else [None]
        assets = json.loads((root / 'runtime-assets.json').read_text())
        request['expected_model_weights'] = {key: value['sha256'] for key, value in assets.items()}
        for index, gpu in enumerate(devices):
            shard = root / ('cpu' if gpu is None else f'gpu-{gpu}')
            for sequence in sequences[index::len(devices)]:
                key = hashlib.sha256(sequence.encode()).hexdigest()
                try:
                    record = json.loads((shard / 'results' / (key + '.json')).read_text())
                except (OSError, ValueError):
                    continue
                if not valid_record(record, request=request, sequence=sequence):
                    continue
                fields = ('evidence_backend', 'authoritative', 'sequence_hash', 'epoch', 'challenge_sha256',
                          'protocol_sha256', 'runtime_identity_sha256', 'screen_outcome', 'screen_reason', 'provenance')
                profile = {**record['profile'], **{name: record[name] for name in fields}}
                profiles[sequence] = profile
                if on_result is not None:
                    on_result(sequence, profile)
    except (OSError, ValueError, KeyError, TypeError):
        return {**empty, 'reason': 'accelerator_receipt_unavailable'}
    return {**empty, 'status': 'complete' if len(profiles) == len(sequences) else 'partial' if profiles else 'unavailable',
            'profiles': profiles, 'elapsed_seconds': time.monotonic() - began,
            'candidate_count': len(sequences), 'measured_count': len(profiles),
            'provenance': {**empty['provenance'], 'request_sha256': request['request_sha256'],
                           'source_sha': job['source_sha'], 'workers_per_gpu': configured['workers_per_gpu'],
                           'gpus': configured['gpus'], 'cleanup': cleanup}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--coordinate', type=Path)
    group.add_argument('--shard', type=Path)
    group.add_argument('--guard', type=Path)
    args = parser.parse_args()
    request = json.loads((args.coordinate or args.shard or args.guard).read_text())
    if args.guard:
        _guard(request)
    elif args.coordinate:
        _coordinate(request)
    else:
        _shard(request)

if __name__ == '__main__':
    main()
