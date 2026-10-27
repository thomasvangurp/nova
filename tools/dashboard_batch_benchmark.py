"""Isolated ordinary prediction-batch experiment; never writes replay storage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from tools.slave_runtime import slave_multipass_writeback as worker

PASSES = 10


def research_command(inp, output, gpu, config, target, prediction_batch_size):
    if prediction_batch_size != 4:
        raise ValueError('only_explicit_batch_four_experiment_supported')
    command = worker.replicate_command(inp, output.parent, str(gpu), config, target)
    command.remove('--allow-replicates')
    command[command.index('--batch-size') + 1] = str(prediction_batch_size)
    return command


def records(path, sequences, target, replica, prediction_batch_size):
    payload = json.loads(path.read_text())
    if (payload.get('predict_batch_size') != prediction_batch_size
            or payload.get('diagnostic_seed') is not None
            or not isinstance(payload.get('observation_id'), str)
            or not payload['observation_id']):
        raise ValueError('experimental_prediction_identity_mismatch')
    return worker.ordinary_records(path, sequences, target, replica), payload['observation_id']


def prepare_context(directory, sequences, config, field, controls):
    """Keep experimental draws distinct from every canonical batch-two path."""
    if len(sequences) != len(set(sequences)) or len(sequences) != 4:
        raise ValueError('four_distinct_control_sequences_required')
    if len({len(s) for s in sequences}) != 1:
        raise ValueError('compatible_lengths_required')
    from nanobody_opt import score_boltzgen_fast_adapter as adapter
    settings = adapter.load_settings()
    pin = Path(settings.get('scorer', adapter.DEFAULT_SCORER))
    adapter.require_pinned_scorer(pin, settings.get('scorer_sha256'))
    identity = {'kind': 'research_prediction_batch_four', 'canonical_import': False,
                'predict_batch_size': 4, 'passes': PASSES, 'sequences': sequences,
                'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                'field_sha256': hashlib.sha256(field.read_bytes()).hexdigest(),
                'settings_sha256': hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
                'controls': [str(p) for p in controls],
                'scorer_sha256': hashlib.sha256(pin.read_bytes()).hexdigest(),
                'worker_sha256': hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
                'experiment_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    marker = directory/'context.json'
    if marker.exists() and json.loads(marker.read_text()).get('signature') != signature:
        raise ValueError('research_context_changed_preserve_previous_artifacts')
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({'signature': signature, **identity}, sort_keys=True))
    return identity


def comparison(experimental, controls, field):
    def distribution(values):
        return {'n': len(values), 'mean': statistics.mean(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                'minimum': min(values), 'maximum': max(values)}
    result = []
    exp = worker.reference_ranks(experimental, field)
    base = worker.reference_ranks(controls, field)
    for sequence in sorted({s for s, _, _, _ in exp}):
        x = [score for s, _, _, score in exp if s == sequence]
        y = [score for s, _, _, score in base if s == sequence]
        if len(x) != PASSES or len(y) != PASSES:
            raise ValueError('complete_ten_draw_distributions_required')
        result.append({'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                       'experimental_batch4': distribution(x),
                       'control_batch2': distribution(y),
                       'mean_difference': statistics.mean(x)-statistics.mean(y)})
    return result


def run(directory, controls, gpu, lease):
    import yaml
    worker.verify_owned_lease(lease, str(gpu))
    if len(controls) != 2:
        raise ValueError('two_complete_pair_controls_required')
    sequences, control_records, control_warm = [], [], []
    config, field_path = controls[0]/'config.yaml', controls[0]/'field.json'
    epoch = int(controls[0].parent.name.removeprefix('epoch'))
    field = json.loads(field_path.read_text())
    target = worker.validate_context(epoch, yaml.safe_load(config.read_text()), field)
    control_ids = set()
    for control in controls:
        if (control.parent.name != f'epoch{epoch}'
                or (control/'config.yaml').read_bytes() != config.read_bytes()
                or (control/'field.json').read_bytes() != field_path.read_bytes()):
            raise ValueError('control_context_mismatch')
        seqs = json.loads((control/'sequences.json').read_text())
        if len(seqs) != 2:
            raise ValueError('batch_two_controls_required')
        sequences.extend(seqs)
        seen = set()
        for path in control.glob('work/epoch*/replicates*/*/length*/pass*/exact_all.json'):
            if '.qualified-' in str(path) or '.stale-' in str(path):
                continue
            replica = int(path.parent.name.removeprefix('pass'))
            data, observation = records(path, seqs, target, replica, 2)
            if replica in seen or observation in control_ids:
                raise ValueError('duplicate_control_draw')
            seen.add(replica); control_ids.add(observation); control_records.extend(data)
            timing = json.loads((path.parent/'timing.json').read_text())
            if replica > 1:
                control_warm.append(timing['elapsed_seconds']/2)
        if seen != set(range(1, PASSES+1)):
            raise ValueError('incomplete_pair_control')
    sequences.sort()
    identity = prepare_context(directory, sequences, config, field_path, controls)
    completed, identities, durations = [], set(), []
    with worker.owned_daemon():
        for replica in range(1, PASSES+1):
            work = directory/f'pass{replica}'; work.mkdir(parents=True, exist_ok=True)
            output = work/'exact_all.json'; saved = None
            if output.is_file():
                try:
                    saved = records(output, sequences, target, replica, 4)
                except (ValueError, OSError, KeyError, TypeError):
                    # Only a complete result in this exact context may resume.
                    saved = None
            if saved is None:
                inp = work/'input.json'; inp.write_text(json.dumps({'sequences': sequences}))
                command = research_command(inp, output, gpu, config, target, 4)
                started = time.monotonic()
                result = subprocess.run(command, capture_output=True, timeout=600,
                                        env=dict(os.environ, NANOBODY_SCORER_DAEMON='1'))
                if result.returncode:
                    raise RuntimeError('experimental_scorer_failed')
                saved = records(output, sequences, target, replica, 4)
                (work/'timing.json').write_text(json.dumps({'elapsed_seconds': time.monotonic()-started}))
            rows, observation = saved
            if observation in identities or observation in control_ids:
                raise ValueError('experimental_draw_identity_reused')
            identities.add(observation); completed.extend(rows)
            timing = work/'timing.json'
            durations.append(json.loads(timing.read_text()).get('elapsed_seconds') if timing.exists() else None)
            (directory/'progress.json').write_text(json.dumps({'kind': identity['kind'],
                'canonical_import': False, 'completed_calls': replica, 'candidates': 4,
                'fresh_draw_ids': len(identities), 'call_seconds': durations}))
    warm = [v for v in durations[1:] if isinstance(v, (int, float))]
    result = {'kind': identity['kind'], 'canonical_import': False, 'target': target,
              'epoch': epoch, 'predict_batch_size': 4, 'candidates': 4, 'calls': PASSES,
              'fresh_draw_ids': len(identities), 'cold_seconds': durations[0],
              'mean_warm_seconds': statistics.mean(warm) if warm else None,
              'control_seconds_per_candidate': statistics.mean(control_warm),
              'scorer_sha256': identity['scorer_sha256'],
              'distributions': comparison(completed, control_records, field),
              'promotion_status': 'requires_separate_contract_and_parity_review'}
    temporary = directory/'result.json.next'; temporary.write_text(json.dumps(result, sort_keys=True))
    temporary.replace(directory/'result.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--control', type=Path, action='append', required=True)
    parser.add_argument('--gpu', type=int, choices=(5, 6, 7), required=True)
    parser.add_argument('--lease-receipt', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.directory, args.control, args.gpu, args.lease_receipt)
        print(json.dumps({k: result[k] for k in ('kind', 'canonical_import', 'calls', 'candidates')}))
        return 0
    except Exception as error:
        # Scorer subprocess output and paths remain private on any failure.
        print(json.dumps({'ok': False, 'reason': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
