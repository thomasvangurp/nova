"""Registered, leased upstream native evidence. Disabled until qualification.

The prediction device, preparation threads and CPU final refinement are distinct
parts of the execution policy. This route never promotes CUDA-refinement screens.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types
import uuid
import weakref
import zlib

from nanobody_opt import pr100_validator_reference as reference

ROOT = Path(__file__).resolve().parents[1]
BACKEND = 'native_reference'
MODULE = 'nanobody_opt.pr100_native_reference'
SCHEMA = 'pr100-registered-native-profile-v1'
REGISTRY = ROOT / 'config/pr100_native_reference.json'
RESIDENT_MODULE = 'nanobody_opt.pr100_registered_resident'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    from nanobody_opt.pr100_live_gpu import atomic as write
    write(path, value)


def registration(identity):
    registry = json.loads(REGISTRY.read_text())
    reference.require(registry.get('schema') == 'pr100-native-reference-registry-v1', 'invalid_registry')
    rows = registry.get('registrations')
    reference.require(isinstance(rows, list), 'invalid_registry')
    matches = [row for row in rows if reference.digest(row) == identity]
    reference.require(len(matches) == 1, 'unregistered_native_reference')
    row = matches[0]
    policy = row.get('policy', {})
    reference.require(policy.get('prediction') == 'auto_cuda'
        and type(policy.get('preparation_threads')) is int
        and 1 <= policy['preparation_threads'] <= 32, 'invalid_reference_policy')
    refinement = reference.validate_refinement_policy(policy.get('final_refinement'))
    if refinement['platform'] == 'CUDA':
        reference.require(policy['preparation_threads'] == 1, 'invalid_cuda_preparation_policy')
        reference.validate_cuda_plugin_identity(row.get('cuda_plugin'))
        reference.require(row.get('qualification_status') == 'qualified'
            and reference.sha256(row.get('qualification_report_sha256'))
            and reference.sha256(row.get('resident_module_sha256')), 'unqualified_cuda_registration')
    if 'resident_module_sha256' in row:
        reference.require(reference.sha256(row['resident_module_sha256']), 'unregistered_resident_module')
    for key in ('runtime_sha256', 'observer_sha256', 'implementation_sha256',
                'environment_manifest_sha256', 'bundle_manifest_sha256'):
        reference.require(reference.sha256(row.get(key)), 'missing_registration_identity')
    reference.require(row.get('actual_validator_worker_attested') is False,
                      'unsupported_validator_attestation')
    return row


def _policy_trace(receipt, policy):
    for event in receipt['trace']:
        if event.get('event') == 'context_created' and event.get('role') in ('missing_atoms', 'hydrogens'):
            reference.require(event.get('platform') == 'CPU'
                and event.get('properties', {}).get('Threads') == str(policy['preparation_threads']),
                'preparation_policy_not_observed')


def _heavy_pdb_rows(text):
    """Normalize only serial/index and hydrogens; preserve ordered heavy science."""
    reference.require(isinstance(text, str) and len(text) <= 2_000_000, 'invalid_structure_text')
    rows = []
    for line in text.splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        name = line[12:16].strip()
        if re.fullmatch(r'[123 ]*H.*', name):
            continue
        reference.require(len(line) >= 78, 'truncated_structure_atom')
        xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
        reference.require(all(math.isfinite(v) for v in xyz), 'invalid_structure_coordinates')
        rows.append((line[:6], name, line[16:17], line[17:20].strip(), line[21:22],
                     line[22:26].strip(), line[26:27], line[76:78].strip(), *xyz))
    reference.require(bool(rows), 'missing_heavy_atoms')
    return rows


def pack_artifact(text):
    raw = text.encode()
    reference.require(len(raw) <= 2_000_000, 'oversized_scientific_artifact')
    return {'encoding': 'zlib-base64-v1', 'raw_bytes': len(raw),
            'data': base64.b64encode(zlib.compress(raw, level=6)).decode()}


def unpack_artifact(value):
    try:
        reference.require(isinstance(value, dict) and value.get('encoding') == 'zlib-base64-v1'
            and type(value.get('raw_bytes')) is int and 0 < value['raw_bytes'] <= 2_000_000
            and isinstance(value.get('data'), str) and len(value['data']) <= 2_000_000,
            'invalid_scientific_artifact_envelope')
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(value['data'], validate=True), 2_000_001)
        reference.require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
            and len(raw) == value['raw_bytes'], 'invalid_scientific_artifact_size')
        return raw.decode()
    except (ValueError, zlib.error, UnicodeError) as error:
        raise ValueError('invalid_scientific_artifact') from error


def validate_structure_binding(receipt):
    """Bind observed model -> exact TNP H-removal -> final PDB -> TNP/A3D."""
    models = [e for e in receipt['trace'] if e.get('event') == 'model_saved']
    geometries = [e for e in receipt['trace'] if e.get('event') == 'geometry_snapshot'
                  and e.get('stage') == 'final_model']
    reference.require(len(models) == len(geometries) == 1, 'missing_final_geometry')
    model, geometry = models[0], geometries[0]
    reference.require(model['pid'] == geometry['pid']
        and geometry['event_index'] == model['event_index'] + 1, 'final_geometry_order_changed')
    binding = receipt['structure_binding']
    reference.require(binding['transformation'] == 'published_tnp_hydrogen_removal_pdbio'
        and binding['raw_model_sha256'] == model['pdb_sha256'], 'unbound_original_model')
    original, final = (unpack_artifact(binding[key]) for key in ('observed_geometry_pdb', 'final_pdb'))
    reference.require(hashlib.sha256(original.encode()).hexdigest() == geometry['pdb_sha256']
        and hashlib.sha256(final.encode()).hexdigest() == receipt['structure_sha256'],
        'structure_artifact_changed')
    reference.require(_heavy_pdb_rows(original) == _heavy_pdb_rows(final), 'heavy_structure_changed')
    output = binding['tnp_result_json']
    reference.require(isinstance(output, str) and len(output) <= 2_000_000
        and hashlib.sha256(output.encode()).hexdigest() == binding['tnp_result_sha256'],
        'tnp_output_changed')
    name = 'reference_' + receipt['sequence_hash'][:16]
    profile = json.loads(output)[name]
    reference.require(all(profile[source] == receipt['observation']['metrics'][target]
        for source, target in METRICS.items()), 'tnp_output_metrics_changed')
    reference.require(receipt['request'].get('structure_binding_sha256') == reference.digest(binding),
                      'structure_binding_request_changed')


def measurement_identity(value):
    """Validated scientific identity for rejecting mixed-runtime draw sets."""
    if not authoritative_measurement(value):
        return None
    binding = value['native_reference']
    return {'backend': BACKEND, 'registration_sha256': binding['registration_sha256'],
            'runtime_identity_sha256': binding['receipt']['runtime_identity_sha256'],
            'policy_sha256': reference.digest(binding['policy'])}


def authoritative_measurement(value):
    """Pure evidence check against source-controlled approval; no file-path trust."""
    try:
        reference.require(isinstance(value, dict) and value.get('schema') == SCHEMA
            and value.get('evidence_backend') == BACKEND and value.get('authoritative') is True,
            'not_registered_native_evidence')
        binding = value['native_reference']; row = registration(binding['registration_sha256'])
        receipt, request = binding['receipt'], binding['receipt']['request']
        reference.require(binding['policy'] == row['policy']
            and request['registration_sha256'] == binding['registration_sha256']
            and request['policy'] == row['policy']
            and request['implementation_sha256'] == row['implementation_sha256'],
            'reference_policy_changed')
        if (row['policy']['final_refinement']['platform'] == 'CUDA'
                or request.get('execution') == 'resident-v1'):
            reference.require(request.get('execution') == 'resident-v1'
                and reference.sha256(row.get('resident_module_sha256'))
                and request.get('resident_module_sha256') == row['resident_module_sha256']
                and reference.sha256(request.get('model_state_start_sha256'))
                and request['model_state_start_sha256'] == request.get('model_state_end_sha256')
                and isinstance(request.get('resident_batch_id'), str)
                and re.fullmatch('[0-9a-f]{32}', request['resident_batch_id']) is not None
                and type(request.get('resident_batch_draws')) is int
                and 1 <= request['resident_batch_draws'] <= 32,
                'resident_execution_changed')
        reference.require(receipt['request_sha256'] == reference.digest(request), 'request_changed')
        reference.validate_receipt(receipt, expected_source_sha=request['source_sha'],
            expected_sequence_hash=value['sequence_hash'], expected_request_sha256=reference.digest(request),
            expected_weights=row['weights'], expected_numpy=row['numpy'],
            expected_observer_sha256=row['observer_sha256'], expected_cuda_available=True,
            expected_runtime_identity_sha256=row['runtime_sha256'],
            expected_final_refinement=row['policy']['final_refinement'],
            expected_preparation_threads=row['policy']['preparation_threads'],
            expected_cuda_plugin=row.get('cuda_plugin'))
        _policy_trace(receipt, row['policy'])
        validate_structure_binding(receipt)
        metrics = receipt['observation']['metrics']
        reference.require(value['passed'] is receipt['observation']['passed']
            and all(value.get(key) == metrics[key] for key in METRICS.values())
            and value.get('aggrescan3d') == receipt['aggregation'], 'measurement_receipt_changed')
        aggregation = receipt['aggregation']
        reference.require(aggregation.get('status') == 'ok'
            and aggregation.get('sequence_hash') == value['sequence_hash']
            and aggregation.get('input_structure_sha256') == receipt['structure_sha256']
            and str(aggregation['a3d_average']) == str(metrics['a3d_average'])
            and str(aggregation['a3d_maximum']) == str(metrics['a3d_maximum']), 'aggregation_binding_changed')
        return True
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        return False


METRICS = {'Total CDR Length': 'total_cdr_length', 'CDR3 Length': 'cdr3_length',
           'CDR3 Compactness': 'cdr3_compactness', 'PSH': 'surface_hydrophobic_patches',
           'PPC': 'positive_charge_patches', 'PNC': 'negative_charge_patches'}


def _registered_runtime(raw_runtime, row):
    """Bind an added sidecar without changing the original sealed contract."""
    if row['policy']['final_refinement']['platform'] == 'CUDA':
        reference.validate_cuda_plugin_identity(row.get('cuda_plugin'))
        reference.require('cuda_plugin' not in raw_runtime, 'base_runtime_already_extended')
        return {**raw_runtime, 'cuda_plugin': dict(row['cuda_plugin'])}
    return raw_runtime


def verify_files(settings, row, *, full=False):
    """Verify the sealed closure once per phase, scientific assets per lease."""
    bundle, environment = Path(settings['bundle']), Path(settings['environment'])
    reference.require(sha(__file__) == row['implementation_sha256'], 'runner_bytes_changed')
    observer = ROOT / 'nanobody_opt/pr100_context_observer.py'
    reference.require(sha(observer) == row['observer_sha256'], 'observer_bytes_changed')
    raw_runtime = json.loads((bundle / 'runtime-contract.json').read_text())
    if row['policy']['final_refinement']['platform'] == 'CUDA':
        reference.require(settings.get('resident') is True, 'cuda_requires_registered_resident')
        cuda_plugin_paths(settings, row)
    runtime = _registered_runtime(raw_runtime, row)
    reference.require(reference.digest(runtime) == row['runtime_sha256'], 'runtime_changed')
    reference.validate_runtime(runtime, expected_weights=row['weights'], expected_numpy=row['numpy'],
                               expected_cuda_plugin=row.get('cuda_plugin'))
    if settings.get('resident', False):
        reference.require(reference.sha256(row.get('resident_module_sha256'))
            and sha(ROOT / (RESIDENT_MODULE.replace('.', '/') + '.py')) == row['resident_module_sha256'],
            'resident_module_changed')
    roots = [bundle / 'filter-source', environment / 'lib/python3.12/site-packages']
    for name, expected in runtime['sources'].items():
        candidates = [root / (name.replace('.', '/') + '.py') for root in roots]
        path = next((p for p in candidates if p.is_file()), None)
        reference.require(path is not None and sha(path) == expected, 'scientific_source_changed')
    for index, expected in row['weights'].items():
        reference.require(sha(roots[1] / 'ImmuneBuilder/trained_model' / ('nanobody_model_' + index))
                          == expected, 'model_weights_changed')
    for name, expected in runtime['native_tools'].items():
        reference.require(sha(bundle / 'tools' / name) == expected, 'native_tool_changed')
    reference.require(sha(environment / 'bin/python3.12') == runtime['interpreter_sha256']
        and sha(environment / 'bin/TNP') == runtime['tnp_cli']['file_sha256'], 'entrypoint_changed')
    manifests = [(environment.parent / 'runtime-manifest.json', environment.parent,
                  row['environment_manifest_sha256']),
                 (bundle / 'bundle-manifest.json', bundle, row['bundle_manifest_sha256'])]
    for path, root, expected in manifests:
        reference.require(sha(path) == expected, 'sealed_manifest_changed')
        if full:
            for relative, digest in json.loads(path.read_text())['files'].items():
                target = root / relative
                reference.require(target.resolve().is_relative_to(root.resolve())
                                  and sha(target) == digest, 'sealed_file_changed')
    return runtime


def cuda_plugin_paths(settings, row):
    """Only the registered, sealed sidecar; no automatic plugin discovery."""
    identity = reference.validate_cuda_plugin_identity(row.get('cuda_plugin'))
    value = settings.get('cuda_plugin_root')
    reference.require(isinstance(value, str) and Path(value).is_absolute(), 'missing_cuda_plugin_root')
    root = Path(value).resolve()
    path = root / 'plugin-manifest.json'
    reference.require(not path.is_symlink() and sha(path) == identity['manifest_sha256'],
                      'cuda_plugin_manifest_changed')
    manifest = json.loads(path.read_text())
    reference.require(manifest.get('schema') == 'pr100-openmm-cuda-plugin-v1'
        and all(manifest.get(key) == identity[key] for key in ('distribution', 'version', 'wheel_sha256')),
        'cuda_plugin_manifest_identity_changed')
    def inside(relative):
        reference.require(isinstance(relative, str) and not Path(relative).is_absolute()
            and '..' not in Path(relative).parts, 'invalid_cuda_plugin_path')
        result = root / relative
        reference.require(result.resolve().is_relative_to(root), 'cuda_plugin_path_escape')
        return result
    files = manifest.get('files')
    reference.require(isinstance(files, dict) and files, 'missing_cuda_plugin_files')
    for relative, expected in files.items():
        target = inside(relative)
        reference.require(re.fullmatch(r'.+\.so(?:\.\d+)*', target.name) is not None
            and reference.sha256(expected) and target.is_file() and not target.is_symlink()
            and sha(target) == expected, 'cuda_plugin_file_changed')
    directory = inside(manifest.get('plugin_directory'))
    reference.require(directory.is_dir() and (directory / 'libOpenMMCUDA.so').is_file(),
                      'missing_cuda_plugin_directory')
    libraries = manifest.get('library_directories')
    reference.require(isinstance(libraries, list), 'missing_cuda_library_directories')
    library_paths = [inside(relative) for relative in libraries]
    reference.require(all(path.is_dir() for path in library_paths), 'missing_cuda_library_directory')
    for folder in [directory, *library_paths]:
        for candidate in folder.iterdir():
            if re.fullmatch(r'.+\.so(?:\.\d+)*', candidate.name):
                reference.require(candidate.relative_to(root).as_posix() in files,
                                  'unsealed_cuda_library')
    return {'plugin_directory': str(directory), 'library_directories': [str(path) for path in library_paths]}


def settings_for(job):
    settings = job.get('native_reference', {})
    reference.require(isinstance(settings, dict) and type(settings.get('enabled', False)) is bool,
                      'invalid_native_reference_settings')
    if not settings.get('enabled', False):
        return None
    row = registration(settings.get('registration_sha256'))
    resident = settings.get('resident', False)
    reference.require(type(resident) is bool, 'invalid_resident_setting')
    if resident:
        reference.require(reference.sha256(row.get('resident_module_sha256')), 'unregistered_resident_module')
        batch_draws = settings.get('resident_batch_draws', 32)
        reference.require(type(batch_draws) is int and 1 <= batch_draws <= 32,
                          'invalid_resident_batch_draws')
        settings = {**settings, 'resident_batch_draws': batch_draws}
    if row['policy']['final_refinement']['platform'] == 'CUDA':
        reference.require(resident, 'cuda_requires_registered_resident')
        value = settings.get('cuda_plugin_root')
        reference.require(isinstance(value, str) and Path(value).is_absolute(), 'missing_cuda_plugin_root')
    gpus, allowlist = settings.get('gpus'), settings.get('gpu_allowlist')
    reference.require(isinstance(gpus, list) and isinstance(allowlist, list)
        and all(type(g) is int and 0 <= g < 64 for g in gpus + allowlist), 'invalid_reference_gpus')
    chosen = sorted(set(gpus) & set(allowlist))
    reference.require(bool(chosen), 'no_allowed_reference_gpu')
    workers = settings.get('workers_per_gpu', 1)
    reference.require(type(workers) is int and 1 <= workers <= 8, 'invalid_reference_workers')
    for key in ('bundle', 'environment', 'lock_directory', 'scorer'):
        reference.require(isinstance(settings.get(key), str) and Path(settings[key]).is_absolute(),
                          'missing_reference_path:' + key)
    return {**settings, 'resident': resident, 'gpus': chosen, 'workers_per_gpu': workers, 'registration': row}


class Inconclusive(dict):
    """Trusted in-process runner outcome; serialized errors are not this type."""


def failure(kind, *, started=False):
    reference.require(kind in ('reference_admission_window_closed', 'reference_gpu_unavailable',
                               'reference_profile_inconclusive'), 'unknown_reference_error')
    return Inconclusive(schema='pr100-native-execution-error-v1', error=True, error_kind=kind,
                        evidence_backend=BACKEND, authoritative=False, native_started=started)


class Adapter:
    """Coalesce singleton refill calls into bounded, exclusively leased GPU batches."""

    def __init__(self, job, directory, *, launcher=None, verify=True):
        self.settings = settings_for(job)
        reference.require(self.settings is not None, 'native_reference_disabled')
        self.source_sha = job['source_sha']
        reference.require(re.fullmatch('[0-9a-f]{40}', self.source_sha) is not None, 'invalid_source_sha')
        self.directory = Path(directory) / 'native-reference'
        self.directory.mkdir(parents=True, exist_ok=True)
        self._states = weakref.WeakKeyDictionary()
        self._launcher = launcher or self._launch
        if verify:
            verify_files(self.settings, self.settings['registration'], full=True)

    @property
    def capacity(self):
        return len(self.settings['gpus']) * self.settings['workers_per_gpu']

    async def develop(self, sequences, concurrency, *, on_result=None, on_result_error=None,
                      task_timeout_seconds=300, cpu_only_manager=False):
        reference.require(type(concurrency) is int and concurrency > 0, 'invalid_concurrency')
        reference.require(type(task_timeout_seconds) in (int, float)
            and math.isfinite(task_timeout_seconds) and 10 <= task_timeout_seconds <= 300,
            'invalid_reference_timeout')
        loop = asyncio.get_running_loop()
        state = self._states.setdefault(loop, {'queue': deque(), 'available': set(self.settings['gpus']),
                                                'tasks': set(), 'waiting_lanes': 0, 'closed': False})
        reference.require(not state['closed'], 'reference_adapter_closed')
        waiting = []
        for sequence in sequences:
            future = loop.create_future()
            state['queue'].append((sequence, future, loop.time() + task_timeout_seconds))
            waiting.append((sequence, future))
        self._pump(state)
        values = []
        try:
            for sequence, future in waiting:
                value = await future
                values.append(value)
                if on_result:
                    try:
                        on_result(sequence, value)
                    except Exception as error:
                        if on_result_error:
                            on_result_error(type(error).__name__)
                        else:
                            raise
            return values
        except BaseException:
            await self.close()
            raise
        finally:
            for _, future in waiting:
                if not future.done():
                    future.cancel()

    def _pump(self, state):
        while not state['closed'] and state['queue'] and state['available']:
            gpu = min(state['available']); state['available'].remove(gpu)
            state['waiting_lanes'] += 1
            task = asyncio.create_task(self._lane(state, gpu))
            state['tasks'].add(task); task.add_done_callback(state['tasks'].discard)

    def _batch_limit(self, state):
        return min(self.settings['workers_per_gpu'],
                   max(1, math.ceil(len(state['queue']) / max(1, state['waiting_lanes']))))

    async def _lane(self, state, gpu):
        batch = []
        try:
            await asyncio.sleep(0)  # Coalesce simultaneously admitted refill slots.
            limit = self._batch_limit(state)
            state['waiting_lanes'] -= 1
            while state['queue'] and len(batch) < limit:
                item = state['queue'].popleft()
                if not item[1].cancelled():
                    batch.append(item)
            if not batch:
                return
            timeout = min(item[2] for item in batch) - asyncio.get_running_loop().time()
            if timeout < 10:
                values = [failure('reference_admission_window_closed') for _ in batch]
            else:
                values = await self._launcher(gpu, [item[0] for item in batch], min(timeout, 300))
                reference.require(len(values) == len(batch), 'incomplete_reference_batch')
                # A clean refusal before any child existed may try another GPU.
                # Never retry started, partial or uncertain work on another device.
                tried = {gpu}
                while all(value.get('error_kind') == 'reference_gpu_unavailable'
                          and value.get('native_started') is False for value in values):
                    alternatives = state['available'] - tried
                    remaining = min(item[2] for item in batch) - asyncio.get_running_loop().time()
                    if not alternatives or remaining < 10:
                        break
                    alternative = min(alternatives); tried.add(alternative)
                    state['available'].remove(alternative)
                    try:
                        values = await self._launcher(alternative, [item[0] for item in batch], min(remaining, 300))
                        reference.require(len(values) == len(batch), 'incomplete_reference_batch')
                    finally:
                        state['available'].add(alternative)
            for (_, future, _), value in zip(batch, values):
                if not future.done():
                    future.set_result(value)
        except BaseException as error:
            # A missing/dirty lease receipt or runtime drift stops new admissions
            # on every GPU. A different device does not make an uncertain owner safe.
            state['closed'] = True
            while state['queue']:
                _, future, _ = state['queue'].popleft()
                if not future.done():
                    future.set_exception(error)
            for _, future, _ in batch:
                if not future.done():
                    future.set_exception(error)
        finally:
            state['available'].add(gpu)
            self._pump(state)

    async def close(self):
        state = self._states.get(asyncio.get_running_loop())
        if state:
            state['closed'] = True
            for _, future, _ in state['queue']:
                future.cancel()
            state['queue'].clear()
            for task in list(state['tasks']):
                task.cancel()
            if state['tasks']:
                await asyncio.gather(*list(state['tasks']), return_exceptions=True)

    async def _launch(self, gpu, sequences, timeout):
        directory = self.directory / uuid.uuid4().hex
        directory.mkdir()
        request = {'settings': self.settings, 'source_sha': self.source_sha,
                   'sequences': sequences, 'gpu': gpu, 'timeout_seconds': timeout - 5,
                   'directory': str(directory), 'source_root': str(ROOT)}
        atomic(directory / 'request.json', request)
        environment = self.settings['environment']; bundle = self.settings['bundle']
        lease = {'gpu': gpu, 'ttl_seconds': timeout, 'minimum_free_mib': 2048,
            'lock_directory': self.settings['lock_directory'], 'scorer': self.settings['scorer'],
            'receipt': str(directory / 'lease.json'), 'cwd': str(directory),
            'command': [bundle + '/bin/native-python', '-B', '-m', MODULE,
                        '--batch', str(directory / 'request.json')],
            'env': {'PATH': environment + '/bin:' + bundle + '/tools:/usr/bin:/bin',
                'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1', 'HOME': '/root',
                'LANG': 'C.UTF-8', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'}}
        atomic(directory / 'lease-job.json', lease)
        proc = await asyncio.create_subprocess_exec(sys.executable, '-B', '-m',
            'nanobody_opt.pr100_live_gpu', '--job', str(directory / 'lease-job.json'),
            cwd=ROOT, env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(ROOT),
                          'PYTHONDONTWRITEBYTECODE': '1'},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            await asyncio.wait_for(asyncio.shield(proc.wait()), timeout + 10)
        except BaseException:
            if proc.returncode is None:
                proc.terminate()
                # The lease owner performs fixed-point descendant cleanup. Never
                # SIGKILL that owner and then admit a replacement behind its back.
                await asyncio.wait_for(asyncio.shield(proc.wait()), 12)
            raise
        try:
            receipt = json.loads((directory / 'lease.json').read_text())
            cleanup = receipt.get('cleanup', {})
            clean = (cleanup.get('remaining_descendants') == 0 and cleanup.get('fixed_point_reached') is True)
            if ('child_pid' not in receipt and 'cleanup' not in receipt
                    and receipt.get('status') == 'unavailable' and receipt.get('released') is True):
                return [failure('reference_gpu_unavailable') for _ in sequences]
            reference.require(clean, 'unverified_reference_cleanup')
            values = []
            for index, sequence in enumerate(sequences):
                path = directory / str(index) / 'measurement.json'
                if not path.is_file():
                    values.append(failure('reference_profile_inconclusive', started=True)); continue
                value = json.loads(path.read_text())
                reference.require(value.get('sequence_hash') == hashlib.sha256(sequence.encode()).hexdigest()
                    and authoritative_measurement(value), 'invalid_registered_measurement')
                values.append(value)
            return values
        except (OSError, json.JSONDecodeError):
            raise RuntimeError('missing_reference_lease_receipt') from None


def _namespace_wrappers(bundle):
    for package in ('metanano', 'metanano.utils'):
        module = types.ModuleType(package)
        module.__path__ = [str(bundle / 'filter-source' / package.replace('.', '/'))]
        sys.modules[package] = module
    for name in ('metanano.utils.tnp_wrapper', 'metanano.utils.aggrescan3d_wrapper'):
        path = bundle / 'filter-source' / (name.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)


def draw(request, index):
    """One ordinary TNP CLI draw plus native A3D on that exact structure."""
    from nanobody_opt.pr100_live_gpu import inherited_lease_fd
    reference.require(inherited_lease_fd([request['gpu']]) is not None, 'missing_gpu_lease')
    settings = request['settings']; row = registration(settings['registration_sha256'])
    runtime = verify_files(settings, row)
    bundle = Path(settings['bundle']); environment = Path(settings['environment'])
    directory = Path(request['directory']) / str(index); directory.mkdir()
    sequence = request['sequences'][index]; key = hashlib.sha256(sequence.encode()).hexdigest()
    reference.require(re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{50,300}', sequence) is not None, 'invalid_sequence')
    observed = directory / 'observer'; observed.mkdir()
    (observed / 'sitecustomize.py').write_text(
        'from nanobody_opt.pr100_context_observer import install\ninstall()\n')
    trace_path = directory / 'trace.jsonl'
    env = dict(os.environ)
    env.update(NOVA_PR100_CONTEXT_TRACE=str(trace_path),
        OPENMM_CPU_THREADS=str(row['policy']['preparation_threads']),
        PYTHONPATH=str(observed) + ':' + str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    name = 'reference_' + key[:16]; profiles = directory / 'tnp'; profiles.mkdir()
    started = time.monotonic()
    with (directory / 'private-stdout.log').open('xb') as stdout, (directory / 'private-stderr.log').open('xb') as stderr:
        completed = subprocess.run([str(environment / 'bin/TNP'), '--name', name,
            '--output', str(profiles), '--seq', sequence], cwd=directory, env=env,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            timeout=request['timeout_seconds'])
    reference.require(completed.returncode == 0, 'native_cli_failed')
    result_text = (profiles / ('TNP_Results_SingleSeqEntry_' + name + '.json')).read_text()
    raw = json.loads(result_text)[name]
    metrics = {target: raw[source] for source, target in METRICS.items()}
    structure = profiles / 'Final_Models' / (name + '_NanoBodyBuilder2_Model.pdb')
    structure_sha = sha(structure)
    raw_trace = trace_path.read_bytes()
    # The original full trace remains an immutable host artifact. Ordered atom
    # identity/coordinate evidence is carried by hashed PDBs below, so repeatedly
    # embedding atom_ids on every intermediate geometry event is unnecessary.
    trace = [{key: value for key, value in json.loads(line).items() if key != 'atom_ids'}
             for line in raw_trace.splitlines()]
    reference.validate_trace(trace, sequence_hash=key, expected_observer_sha256=row['observer_sha256'],
                             expected_cuda_available=True)
    models = [e for e in trace if e.get('event') == 'model_saved']
    geometries = [e for e in trace if e.get('event') == 'geometry_snapshot' and e.get('stage') == 'final_model']
    reference.require(len(models) == len(geometries) == 1, 'missing_final_geometry')
    geometry_name = geometries[0]['pdb_file']
    reference.require(Path(geometry_name).name == geometry_name, 'invalid_geometry_artifact_path')
    structure_binding = {'transformation': 'published_tnp_hydrogen_removal_pdbio',
        'raw_model_sha256': models[0]['pdb_sha256'],
        'observed_geometry_pdb': pack_artifact((directory / 'coordinates' / geometry_name).read_text()),
        'final_pdb': pack_artifact(structure.read_text()), 'tnp_result_json': result_text,
        'tnp_result_sha256': hashlib.sha256(result_text.encode()).hexdigest()}
    _namespace_wrappers(bundle)
    from metanano.utils.tnp_wrapper import ProfileExecution
    from metanano.utils.aggrescan3d_wrapper import Aggrescan3DWrapper
    remaining = request['timeout_seconds'] - (time.monotonic() - started)
    reference.require(remaining > 1, 'aggregation_window_closed')
    aggregation = Aggrescan3DWrapper(str(bundle / 'tools/aggrescan3d-rs'),
        str(bundle / 'tools/freesasa')).profile(sequence, structure, ProfileExecution(min(60, remaining)))
    reference.require(aggregation.get('status') == 'ok'
        and aggregation.get('sequence_hash') == key
        and aggregation.get('input_structure_sha256') == structure_sha, 'aggregation_failed')
    metrics.update(a3d_average=str(aggregation['a3d_average']), a3d_maximum=str(aggregation['a3d_maximum']))
    binding = {'source_sha': request['source_sha'], 'sequence_hash': key,
        'registration_sha256': settings['registration_sha256'], 'policy': row['policy'],
        'implementation_sha256': row['implementation_sha256'], 'draw_id': str(directory),
        'aggregation_loader': 'exact_source_wrappers_without_application_bootstrap',
        'raw_trace_sha256': hashlib.sha256(raw_trace).hexdigest(),
        'structure_binding_sha256': reference.digest(structure_binding)}
    receipt = {'schema': reference.SCHEMA, 'source_sha': request['source_sha'], 'sequence_hash': key,
        'request': binding, 'request_sha256': reference.digest(binding),
        'reference_contract_sha256': reference.digest(reference.contract()), 'runtime': runtime,
        'runtime_identity_sha256': row['runtime_sha256'], 'trace': trace, 'trace_sha256': reference.digest(trace),
        'observation': {'passed': reference.measured_gate(metrics), 'metrics': metrics},
        'actual_validator_worker_attested': False, 'structure_sha256': structure_sha,
        'structure_binding': structure_binding, 'aggregation': aggregation}
    value = {'schema': SCHEMA, 'evidence_backend': BACKEND, 'authoritative': True, 'sequence_hash': key,
        'passed': receipt['observation']['passed'], **{key: metrics[key] for key in METRICS.values()},
        'aggrescan3d': aggregation, 'native_reference': {'registration_sha256': settings['registration_sha256'],
            'policy': row['policy'], 'receipt': receipt}, 'native_started': True,
        'elapsed_seconds': time.monotonic() - started}
    verify_files(settings, row)  # Drift invalidates even an otherwise valid profile.
    reference.require(authoritative_measurement(value), 'native_evidence_validation_failed')
    atomic(directory / 'measurement.json', value)


def batch(request):
    from nanobody_opt.pr100_live_gpu import inherited_lease_fd
    fd = inherited_lease_fd([request['gpu']])
    reference.require(fd is not None, 'missing_gpu_lease')
    reference.require(1 <= len(request['sequences']) <= request['settings']['workers_per_gpu'] <= 8,
                      'unbounded_reference_batch')
    request_path = Path(request['directory']) / 'request.json'
    def execute(index):
        completed = subprocess.run([sys.executable, '-B', '-m', MODULE, '--draw', str(request_path),
            '--index', str(index)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, pass_fds=(fd,), timeout=request['timeout_seconds'] + 2)
        return completed.returncode
    with ThreadPoolExecutor(max_workers=len(request['sequences'])) as pool:
        return list(pool.map(execute, range(len(request['sequences']))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--batch', type=Path); group.add_argument('--draw', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    try:
        path = args.batch or args.draw; request = json.loads(path.read_text())
        if args.batch:
            batch(request)
        else:
            draw(request, args.index)
        return 0
    except BaseException:
        # No third-party exception strings or environment values cross this boundary.
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
