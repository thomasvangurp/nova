#!/usr/bin/env python3
"""Bounded technical PR100 GPU probe; never publishes or writes observations.

Use an already-tested staged release and the host's registered scorer. This
tests target/length support and raw metrics, not biological eligibility or a
production-scale promotion. Backend logs stay in memory; only safe diagnostics
and measurements are persisted. Each subprocess owns its cancellable group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def bounded(command, env, timeout):
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {'exit': None, 'timed_out': True}
    text = (stdout + stderr).decode(errors='replace')
    return {'exit': process.returncode, 'timed_out': False,
            'error_classes': sorted(set(re.findall(r'\b([A-Za-z]+(?:Error|Exception)):', text))),
            'cuda_oom': 'CUDA out of memory' in text,
            'missing_model_cache': 'LocalEntryNotFoundError' in text,
            'replicate_flag_unsupported': 'unrecognized arguments: --allow-replicates' in text}


def validate_result(path, sequences, passes, target, unpack):
    from nanobody_opt.score_boltzgen_waves import _validated_wave
    expected = [hashlib.sha256(s.encode()).hexdigest()+str(i)
                for i, s in enumerate(s for s in sequences for _ in range(passes))]
    payload = _validated_wave(path, expected, target, require_structures=True)
    records = unpack(payload['per_nanobody_components']['0'], sequences, passes, target)
    for by_target in payload['structures'].values():
        record = by_target[target]
        for key in ['confidence', 'writer']:
            structure = Path(record[key+'_path']).resolve()
            if not structure.is_relative_to(path.parent.resolve()):
                raise ValueError('structure escaped the private probe directory')
            if hashlib.sha256(structure.read_bytes()).hexdigest() != record[key+'_sha256']:
                raise ValueError('retained structure digest mismatch')
    return {'records': len(records), 'structure_records': len(payload['structures']),
            # Different vectors are a useful diagnostic, not proof of
            # independent draws. Seed provenance needs a separate test.
            'distinct_metric_vectors_min': min(len({json.dumps(m, sort_keys=True)
                for s, _, m in records if s == sequence}) for sequence in sequences),
            'valid': len(records) == len(sequences)*passes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9]+', args.gpu) or not 1 <= args.passes <= 20 or not 60 <= args.timeout <= 3600:
        parser.error('invalid bounded probe limits')
    args.release = args.release.resolve()
    sys.path.insert(0, str(args.release))
    from scripts.release_manifest import verify
    from nanobody_opt import rules
    from tools.slave_runtime.slave_multipass_writeback import unpack
    import yaml

    manifest = json.loads((args.release/'build_stamp.json').read_text())
    verify(args.release, manifest, expected_sha=manifest['git_sha'])
    settings = json.loads(args.settings.read_text())
    scorer = Path(settings['scorer'])
    scorer_hash = hashlib.sha256(scorer.read_bytes()).hexdigest()
    if scorer_hash != settings['scorer_sha256']:
        raise ValueError('registered scorer pin mismatch')
    if args.output.exists():
        raise ValueError('probe output already exists')
    args.output.mkdir(parents=True)
    sequences = json.loads(args.input.read_text())['sequences']
    if not 1 <= len(sequences) <= 128 or len(set(sequences)) != len(sequences):
        raise ValueError('probe must contain 1..128 distinct sequences')
    config = args.output/'config.yaml'
    config.write_text(yaml.safe_dump(rules.pinned_config(rules.ACTIVATION_EPOCH)))
    allowed = ['scorer', 'scorer_sha256', 'python', 'worker_argument_schema',
               'fold_sampling_steps', 'fold_diffusion_samples', 'design_sampling_steps',
               'fold_matmul_precision']
    private_settings = {k: settings[k] for k in allowed if k in settings}
    # A new-target standard-design probe must not use an old synthesis
    # reference, the production daemon, or its live socket.
    private_settings.update(use_daemon=False, fold_diffusion_samples=5)
    settings_path = args.output/'scorer_settings.json'
    settings_path.write_text(json.dumps(private_settings))
    env = {k: v for k, v in os.environ.items()
           if k in ['PATH', 'HOME', 'USER', 'LANG', 'LD_LIBRARY_PATH', 'PYTHONPATH']}
    env.update(NANOBODY_SCORER_DAEMON='0',
               NANOBODY_SCORER_SOCKET=str(args.output/'unused.sock'),
               NANOBODY_FAST_SCORER_CONFIG=str(settings_path),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CUDA_VISIBLE_DEVICES=args.gpu)
    started = time.monotonic()
    report = {'candidate_sha': manifest['git_sha'], 'registered_scorer_sha256': scorer_hash,
              'target': rules.TARGET, 'scope': 'technical_only_not_publication_qualified',
              'scoring_equivalence_qualified': False,
              'submission_attempts': 0, 'observation_store_writes': 0,
              'candidate_count': len(sequences), 'passes': args.passes, 'groups': [],
              'status': 'running'}
    report_path = args.output/'report.json'
    for length in sorted({len(s) for s in sequences}):
        group = [s for s in sequences if len(s) == length]
        work = args.output/f'length{length}'
        work.mkdir()
        inp = work/'input.json'
        inp.write_text(json.dumps({'sequences': [{'sequence': s} for s in group for _ in range(args.passes)]}))
        command = [sys.executable, str(args.release/'nanobody_opt/score_boltzgen_fast_adapter.py'),
                   '--sequences', str(inp), '--config', str(config), '--target', rules.TARGET,
                   '--gpus', args.gpu, '--shards-per-gpu', '1', '--batch-size', '2',
                   '--work-root', str(work/'work'), '--out', str(work/'exact_all.json'),
                   '--structure-dir', str(work/'structures'), '--require-structures',
                   '--allow-replicates', '--settings', str(settings_path)]
        remaining = args.timeout - (time.monotonic() - started)
        if remaining <= 0:
            report['status'] = 'timed_out'
            break
        before = time.monotonic()
        result = bounded(command, env, remaining)
        result.update(length=length, candidates=len(group), seconds=round(time.monotonic()-before, 3))
        if result['exit'] == 0:
            try:
                result.update(validate_result(work/'exact_all.json', group, args.passes, rules.TARGET, unpack))
            except (OSError, ValueError, KeyError, TypeError) as error:
                result.update(valid=False, validation_error=type(error).__name__)
        else:
            result['valid'] = False
        report['groups'].append(result)
        report['elapsed_seconds'] = round(time.monotonic()-started, 3)
        report_path.write_text(json.dumps(report, indent=2))
        if not result['valid']:
            report['status'] = 'failed'
            break
    else:
        report['status'] = 'complete'
    report['elapsed_seconds'] = round(time.monotonic()-started, 3)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
