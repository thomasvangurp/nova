"""Inventory existing sequences under PR100 eligibility; carry no old-target scores."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import sqlite3
import time

from nanobody_opt import rules
from nanobody_opt.pr100_filter import annotate, atomic, annotation_identity
from nanobody_opt.gates import check_sequence


def bucket_key(gene, length):
    return f'{gene}:{length}'


def memberships(annotation, config):
    if rules.annotation_reason(annotation, config['nanobody_requirements']):
        return []
    settings = config['germline_filtering']
    length = annotation['cdr3_length']
    if not settings['cdr3_length_range']['min'] <= length <= settings['cdr3_length_range']['max']:
        return []
    permitted = {g for genes in settings['v_neighborhood_categories'].values() for g in genes}
    return [bucket_key(g, length) for g in annotation['germline']['V']['genes'] if g in permitted]


def build(database, output, cache, *, workers=8, limit=None):
    config = rules.pinned_config(rules.ACTIVATION_EPOCH)
    identity = annotation_identity()
    settings = config['germline_filtering']
    buckets = {bucket_key(g, n): [] for genes in settings['v_neighborhood_categories'].values()
               for g in genes for n in range(settings['cdr3_length_range']['min'],
                                              settings['cdr3_length_range']['max'] + 1)}
    # Copy the read-only snapshot quickly, then release its transaction before
    # annotation. Never hold the producer's WAL snapshot for an annotation run.
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        rows = connection.execute('SELECT DISTINCT sequence FROM nodes').fetchall()
    if limit:
        # Alphabetical prefixes cluster families and are a biased pilot sample.
        rows = sorted(rows, key=lambda r: rules.sequence_hash(r[0]))[:limit]
    sequences = [r[0] for r in rows if not check_sequence(r[0], config['nanobody_requirements'])]
    started = time.monotonic(); last = started; rejected = Counter(); accepted = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (sequence, annotation) in enumerate(zip(sequences, pool.map(
                partial(annotate, cache_dir=cache), sequences, chunksize=16), strict=True), 1):
            keys = memberships(annotation, config)
            if keys:
                accepted += 1
                record = {'sequence': sequence, 'pr100_annotation': annotation,
                          'source': 'pr100_unscored_target_transfer',
                          'proposal_type': 'existing_sequence_new_target',
                          'sequence_gate_method': rules.PROFILE}
                for key in keys:
                    buckets[key].append(record)
            else:
                rejected[rules.annotation_reason(annotation, config['nanobody_requirements']) or
                         'outside_challenge_range'] += 1
            if time.monotonic() - last >= 30:
                atomic(Path(str(output) + '.progress.json'), {'annotated': i,
                       'total': len(sequences), 'accepted': accepted,
                       'elapsed_seconds': round(time.monotonic() - started, 1)})
                last = time.monotonic()
    receipt = {'profile': rules.PROFILE, 'target': rules.TARGET,
               'ruleset_sha256': rules.fingerprint(config), 'annotation_identity': identity,
               'input_count': len(rows), 'annotated_count': len(sequences), 'accepted_count': accepted,
               'qualification_stage': 'sequence_only', 'developability_verified': False,
               'bucket_counts': {k: len(v) for k, v in buckets.items()},
               'rejections': dict(rejected), 'target_scores_transferred': 0,
               'seconds': round(time.monotonic() - started, 1)}
    atomic(output, {**receipt, 'buckets': buckets})
    atomic(Path(str(output) + '.coverage.json'), receipt)
    return receipt


def select(bank, expected, limit):
    config = rules.pinned_config(expected['epoch'])
    if (bank.get('target') != rules.TARGET or bank.get('ruleset_sha256') != rules.fingerprint(config)
            or bank.get('target_scores_transferred') != 0):
        raise ValueError('PR100 bank profile mismatch')
    rows = bank['buckets'].get(bucket_key(expected['allowed_v'], expected['allowed_cdr3_length']), [])
    eligible = [r for r in rows if not rules.annotation_reason(r.get('pr100_annotation'),
                  config['nanobody_requirements'], sequence=r['sequence'], epoch_challenge=expected)]
    # No old-target rank is treated as new-target merit. Stable rotation gives
    # each challenge fresh breadth; the target ledger removes previous scores.
    return sorted(eligible, key=lambda r: rules.digest([expected['epoch'], r['sequence']]))[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be 1..32')
    print(json.dumps(build(args.database, args.output, args.cache, workers=args.workers, limit=args.limit)))


if __name__ == '__main__':
    main()
