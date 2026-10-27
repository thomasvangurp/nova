"""Qualified annotation child, separate from the native profiling runtime.

The existing phase supervisor owns this child and every descendant. Never
start a separate session or hide a child failure behind a successful verdict.
"""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from nanobody_opt import rules

SCHEMA = 'pr100-qualified-novelty-v1'
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_KEYS = frozenset({'python', 'path', 'python_path', 'identity_sha256'})


def validate_config(runtime):
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_KEYS:
        raise ValueError('dedicated novelty runtime configuration is required')
    for key in ('python', 'path', 'python_path'):
        value = runtime[key]
        if not isinstance(value, str) or not value or '\x00' in value or '\n' in value:
            raise ValueError('invalid novelty runtime path')
    if not Path(runtime['python']).is_absolute():
        raise ValueError('novelty interpreter must be absolute')
    if not isinstance(runtime['identity_sha256'], str) or not re.fullmatch(
            r'[0-9a-f]{64}', runtime['identity_sha256']):
        raise ValueError('qualified novelty runtime identity is required')
    return runtime


def runtime_identity():
    """Bind actual annotation tools, alignment backend and search sources."""
    from nanobody_opt import pr100_filter
    from abnumber import Chain
    if not callable(getattr(Chain, 'batch', None)):
        raise RuntimeError('novelty runtime requires abnumber Chain.batch')
    pr100_filter.validate_annotation_runtime()
    packages = {}
    for name in ('abnumber', 'anarci', 'anarcii', 'biopython', 'numpy', 'pandas',
                 'pydantic', 'pydantic_core', 'annotated-types', 'typing_extensions',
                 'typing-inspection', 'parasail'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name != 'parasail':
                raise RuntimeError('novelty runtime dependency unavailable') from None
            packages[name] = None
    assets = {}
    prefix = Path(sys.prefix).resolve()
    for name in ('assets-manifest.json', 'pydantic-assets-manifest.json', 'hmmer-assets-manifest.json'):
        raw = (prefix / name).read_bytes()
        manifest = json.loads(raw)
        if manifest.get('schema') != 'pr100-public-novelty-assets-v1' or not manifest.get('files'):
            raise RuntimeError('novelty runtime asset manifest unavailable')
        for relative, record in manifest['files'].items():
            path = (prefix / relative).resolve()
            if (not path.is_relative_to(prefix) or Path(relative).is_absolute()
                    or '..' in Path(relative).parts
                    or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']):
                raise RuntimeError('novelty runtime asset identity mismatch')
        assets[name] = hashlib.sha256(raw).hexdigest()
    sources = {}
    for name in ('nanobody_opt/check_validator_novelty.py',
                 'nanobody_opt/pr100_novelty_runtime.py',
                 'NOVA-nanobody-filter/metanano/config.py',
                 'NOVA-nanobody-filter/metanano/search/search_engine.py',
                 'NOVA-nanobody-filter/metanano/search/index_manager.py',
                 'NOVA-nanobody-filter/metanano/utils/alignment.py',
                 'NOVA-nanobody-filter/metanano/utils/cdr_utils.py',
                 'NOVA-nanobody-filter/metanano/utils/kmer.py'):
        sources[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {'schema': SCHEMA, 'annotation_identity': pr100_filter.annotation_identity(),
            'packages': packages, 'assets': assets, 'sources': sources,
            'python_version': list(sys.version_info[:3]),
            'python_binary_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            'execution_environment_sha256': rules.digest({key: os.environ.get(key) for key in (
                'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')}),
            'archive_batch_worker_limit': min(8, max(1, (os.cpu_count() or 1) // 2))}


def validate_runtime(positive_sequence):
    """Exercise batch annotation on an already-native-validated input row."""
    identity = runtime_identity()
    from abnumber import Chain
    try:
        chains, _ = Chain.batch({'qualified-positive': positive_sequence}, scheme='imgt')
        chain = chains.get('qualified-positive')
        if chain is None or not all(str(getattr(chain, key + '_seq') or '')
                                    for key in ('cdr1', 'cdr2', 'cdr3')):
            raise ValueError('missing positive annotation')
    except Exception:
        raise RuntimeError('novelty batch annotation preflight failed') from None
    return identity


def evaluate(request):
    """Child protocol; a claimed runtime identity is never authoritative."""
    import pandas as pd
    from nanobody_opt import check_validator_novelty as novelty
    rows = request['rows']
    if not isinstance(rows, list) or not rows:
        raise ValueError('novelty requires a positive native input')
    if len({rules.sequence_hash(r['sequence']) for r in rows}) != len(rows):
        raise ValueError('duplicate novelty input')
    identity = validate_runtime(rows[0]['sequence'])
    runtime_hash = rules.digest(identity)
    if runtime_hash != request['runtime_sha256']:
        raise ValueError('novelty runtime identity mismatch')
    raw = Path(request['archive']).read_bytes()
    archive_hash = hashlib.sha256(raw).hexdigest()
    if archive_hash != request['archive_sha256'] or rules.digest(rows) != request['input_sha256']:
        raise ValueError('novelty input binding mismatch')
    frame = pd.read_csv(io.BytesIO(raw))
    novelty._init_worker(frame, set(frame.sequence.dropna().astype(str).str.strip().str.upper()),
                         all_archive=True)
    index = novelty._WORKER_ENGINE._index_manager
    cdr_available = sum(index.get_record(i).cdrs is not None for i in range(index.size()))
    verdicts = []
    for row in rows:
        value = novelty._search_one(row)
        verdicts.append({'sequence_hash': rules.sequence_hash(row['sequence']),
                         **{key: value[key] for key in ('validator_novel', 'exact_historical',
                                                       'coarse_candidates', 'maximum_identity')}})
    return {'schema': SCHEMA, 'status': 'complete', 'runtime_sha256': runtime_hash,
            'archive_sha256': archive_hash, 'input_sha256': rules.digest(rows),
            'archive_rows': len(frame), 'archive_cdr_available': cdr_available, 'verdicts': verdicts}


def check(rows, archive, *, runtime, run=subprocess.run):
    """Reconstruct candidates from original input only after a bound reply."""
    runtime = validate_config(runtime)
    if not rows:
        raise ValueError('novelty requires a positive native input')
    archive = Path(archive).resolve()
    request = {'rows': rows, 'archive': str(archive),
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'input_sha256': rules.digest(rows), 'runtime_sha256': runtime['identity_sha256']}
    environment = {'PATH': runtime['path'], 'PYTHONPATH': str(ROOT) + ':' + runtime['python_path'],
                   'PYTHONDONTWRITEBYTECODE': '1', 'HOME': str(Path.home()), 'LANG': 'C.UTF-8',
                   'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '', 'ROCR_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    # Inherits the supervised session/process tree. The outer phase deadline
    # and fixed-point cleanup remain authoritative for all annotation tools.
    result = run([runtime['python'], '-B', '-m', 'nanobody_opt.pr100_novelty_runtime'],
                 input=json.dumps(request, allow_nan=False), text=True, capture_output=True,
                 cwd=ROOT, env=environment)
    if result.returncode != 0:
        raise RuntimeError('qualified novelty subprocess failed')
    try:
        reply = json.loads(result.stdout)
        if (reply.get('schema') != SCHEMA or reply.get('status') != 'complete'
                or any(reply.get(k) != request[k] for k in
                       ('runtime_sha256', 'archive_sha256', 'input_sha256'))
                or rules.digest(rows) != request['input_sha256']
                or hashlib.sha256(archive.read_bytes()).hexdigest() != request['archive_sha256']
                or type(reply.get('archive_rows')) is not int or reply['archive_rows'] < 0):
            raise ValueError('novelty reply binding mismatch')
        verdicts = reply['verdicts']
        expected = [rules.sequence_hash(r['sequence']) for r in rows]
        if not isinstance(verdicts, list) or [v.get('sequence_hash') for v in verdicts] != expected:
            raise ValueError('novelty reply input mismatch')
        kept = []
        for row, value in zip(rows, verdicts, strict=True):
            if (type(value.get('validator_novel')) is not bool
                    or type(value.get('exact_historical')) is not bool
                    or type(value.get('coarse_candidates')) is not int or value['coarse_candidates'] < 0
                    or isinstance(value.get('maximum_identity'), bool)
                    or not isinstance(value.get('maximum_identity'), (int, float))
                    or not math.isfinite(value['maximum_identity']) or not 0 <= value['maximum_identity'] <= 1
                    or value['validator_novel'] and value['exact_historical']):
                raise ValueError('invalid novelty verdict')
            if value['validator_novel']:
                kept.append({**row, 'novelty': {**{k: value[k] for k in (
                    'validator_novel', 'exact_historical', 'coarse_candidates', 'maximum_identity')},
                    'runtime_sha256': runtime['identity_sha256']}})
        return kept, reply['archive_rows']
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError('invalid qualified novelty reply') from None


def main():
    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = evaluate(request)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception:
        print(json.dumps({'schema': SCHEMA, 'status': 'failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
