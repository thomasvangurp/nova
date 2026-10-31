#!/usr/bin/env python3
"""CPU-only replay of recorded candidates through the unchanged qualified child.

Run only under an independently bounded CPU supervisor. No model, structure,
GPU, admission ledger, or production state is created or modified here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanobody_opt import pr100_live_worker as worker, rules


def replay(cohort, archive, output, *, expected_rejected=None):
    raw = Path(cohort).read_bytes()
    value = json.loads(raw)
    rows = value['rows']
    if not rows or len({rules.sequence_hash(row['sequence']) for row in rows}) != len(rows):
        raise ValueError('replay requires distinct recorded candidates')
    archive = Path(archive).resolve()
    if worker.file_digest(archive) != value['archive_sha256']:
        raise ValueError('replay archive identity mismatch')
    runtime = value.get('novelty_runtime') or value['job']['novelty_runtime']
    started = time.monotonic()
    kept, archive_rows = worker.check_novelty(rows, archive, runtime=runtime)
    elapsed = time.monotonic() - started
    kept_hashes = {rules.sequence_hash(row['sequence']) for row in kept}
    removed = [rules.sequence_hash(row['sequence']) for row in rows
               if rules.sequence_hash(row['sequence']) not in kept_hashes]
    expected = json.loads(Path(expected_rejected).read_text()) if expected_rejected else None
    if expected is not None:
        if expected['archive_sha256'] != value['archive_sha256']:
            raise ValueError('expected rejection archive mismatch')
        if expected['final_novelty_kept_count'] != 0:
            raise ValueError('expected cohort was not wholly rejected')
        expected_hashes = {rules.sequence_hash(row['sequence']) for row in expected['rows']}
        if not expected_hashes <= set(removed):
            raise ValueError('recorded final novelty rejection was not reproduced')
    result = {'schema': 'pr100-early-novelty-replay-v1', 'status': 'complete',
              'cohort_sha256': hashlib.sha256(raw).hexdigest(), 'challenge': value['challenge'],
              'archive_sha256': value['archive_sha256'], 'runtime_sha256': runtime['identity_sha256'],
              'archive_rows': archive_rows, 'input_count': len(rows), 'kept_count': len(kept),
              'removed_count': len(removed), 'removed_sequence_hashes': removed,
              'kept': kept, 'qualified_child_wall_seconds': elapsed,
              'expected_rejected_count': len(expected_hashes) if expected is not None else None,
              'native_profiles_run': 0,
              'source_sha256': {name: worker.file_digest(ROOT / name) for name in (
                  'nanobody_opt/check_validator_novelty.py', 'nanobody_opt/pr100_novelty_runtime.py')}}
    worker.pr100_filter.atomic(Path(output), result)
    return {key: result[key] for key in ('status', 'input_count', 'kept_count', 'removed_count',
                                        'expected_rejected_count', 'qualified_child_wall_seconds')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--expected-rejected', type=Path)
    args = parser.parse_args()
    try:
        result = replay(args.cohort, args.archive, args.output,
                        expected_rejected=args.expected_rejected)
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_category': type(error).__name__}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
