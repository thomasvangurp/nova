"""Replay queued observations with an explicit epoch target and measured field."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/root')
from tools.slave_runtime.slave_multipass_writeback import validate_context, score_group, reference_ranks
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--field-json', required=True, type=Path)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--work-root', type=Path, default=Path('/root/replay_queue_work'))
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text())
    field_bytes = args.field_json.read_bytes()
    field = json.loads(field_bytes)
    epoch = int(batch['epoch'])
    target = validate_context(epoch, yaml.safe_load(args.config.read_text()), field)
    if batch.get('target') != target:
        raise ValueError('queue target differs from replay target')
    needs = {str(i['sequence']).strip().upper(): int(i['need']) for i in batch['items']}
    tiers = {str(i['sequence']).strip().upper(): int(i['tier']) for i in batch['items']}
    if not needs or any(n < 1 for n in needs.values()):
        raise ValueError('queue must contain positive pass requirements')
    batch_id = str(batch['batch_id'])
    if not batch_id.isdecimal():
        raise ValueError('invalid queue batch identifier')
    root = args.work_root / batch_id
    rows = []
    for passes in sorted(set(needs.values())):
        sequences = sorted(s for s, n in needs.items() if n == passes)
        records = score_group(sequences, passes, root/f'passes{passes}', args.gpus, args.config, target)
        for sequence, replicate, metrics, score in reference_ranks(records, field):
            rows.append({'observation_id': f'replay_queue:{batch_id}:{sequence}:pass{replicate}',
                'sequence': sequence, 'target': target, 'epoch': epoch,
                'source': 'public_multipass_rivals', 'score_kind': 'multipass_sum10_pass',
                'score': score, 'metrics': metrics, 'observation_count': 1,
                'writer': 'replay_queue_slave', 'formula': 'sum10', 'allow_replicates': True,
                'passes': passes, 'pass_index': replicate, 'batch_id': batch_id,
                'queue_tier': tiers[sequence], 'field': 'recent_validator_epochs',
                'field_epochs': field['epochs'], 'field_rows': len(field['rows']),
                'field_sha256': hashlib.sha256(field_bytes).hexdigest()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.ndjson.part')
    temporary.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    os.replace(temporary, args.out)
    print(f'REPLAY_QUEUE_DONE rows={len(rows)} target={target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
