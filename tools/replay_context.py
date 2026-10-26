"""Copy the exact epoch config and same-target reference for remote replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt.rules import target_for_epoch


def prepare(epoch, output, *, live=Path('/root/live'), db=Path('/root/artifacts/nanobody_mutations.sqlite')):
    import yaml
    from nanobody_opt.rank_exact import fetch_recent_fields_from_db, pool_recent_fields

    target = target_for_epoch(epoch)
    source = live / f'epoch{epoch}' / 'producer' / 'config.yaml'
    config = yaml.safe_load(source.read_text())
    if config['protein_selection']['nanobody']['target'] != target:
        raise ValueError('replay config does not match the epoch target')
    fields = fetch_recent_fields_from_db(db, target=target, before_epoch=epoch, count=10)
    if not fields:
        raise ValueError('same-target replay reference is not available yet')
    pooled = pool_recent_fields(fields)
    payload = {'target': target, 'epochs': [e for e, _ in fields],
               'rows': json.loads(pooled.to_json(orient='records', double_precision=15))}
    from tools.slave_runtime.slave_multipass_writeback import validate_context
    validate_context(epoch, config, payload)
    output.mkdir(parents=True, exist_ok=True)
    config_path, field_path = output/'config.yaml', output/'field.json'
    config_path.write_text(yaml.safe_dump(config))
    field_path.write_text(json.dumps(payload, allow_nan=False))
    return config_path, field_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('epoch', type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        prepare(args.epoch, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(f'REPLAY_CONTEXT_PENDING epoch={args.epoch} reason={type(error).__name__}')
        raise SystemExit(3)
