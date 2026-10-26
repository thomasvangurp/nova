"""Research-only native parity probe. Never reads production caches or submits.

Population selection/comparison are dependency-free. Native execution lives in
the separate bounded runner. A passing probe is not validator qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

SCHEMA = 'pr100-native-reuse-probe-v1'
MODES = ('cli_a', 'cli_b', 'warm')
FIELDS = ('total_cdr_length', 'cdr3_length', 'cdr3_compactness',
          'surface_hydrophobic_patches', 'positive_charge_patches',
          'negative_charge_patches')
A3D_FIELDS = ('a3d_average', 'a3d_maximum', 'a3d_minimum', 'a3d_total',
              'a3d_positive_auc', 'residue_count', 'positive_residue_count')
REASONS = ('passed', 'psh_threshold', 'developability_red_flags',
           'aggrescan_unavailable_or_failed')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def freeze_population(reports, *, source_file_sha256=None):
    """Freeze 24 real identities; historical outcomes stratify, never certify.

    Cover every observed gate, each pilot length, and both sides of the PSH
    interval plus both A3D boundaries. Unique identities count once even when
    they were emitted by several arms. This is an intentionally enriched
    numerical stress set, NOT an acceptance-rate estimator.
    """
    rows = {}
    for source, report in sorted(reports.items()):
        for original in report['candidate_measurements']:
            row = {k: original[k] for k in ('sequence_hash', 'reason', 'psh',
                                          'a3d_average', 'a3d_maximum')}
            labels = original.get('buckets', original.get('arm_keys', []))
            row['cells'] = sorted({label.split('/')[0] for label in labels})
            row['source'] = source
            if row['sequence_hash'] in rows:
                raise ValueError('overlapping reports require explicit reconciliation')
            if row['reason'] not in REASONS or not row['cells']:
                raise ValueError('unclassified measured input')
            if any(not isinstance(row[k], (int, float)) or isinstance(row[k], bool)
                   or not math.isfinite(row[k])
                   for k in ('psh', 'a3d_average', 'a3d_maximum')):
                raise ValueError('nonfinite measurement is not a threshold fixture')
            rows[row['sequence_hash']] = row
    chosen = {}

    def take(pool, n, tag, key=lambda r: r['sequence_hash']):
        ordered = sorted(pool, key=lambda r: (key(r), r['sequence_hash']))
        if len(ordered) < n:
            raise ValueError('insufficient real population: ' + tag)
        for row in ordered[:n]:
            value = chosen.setdefault(row['sequence_hash'], dict(row, selection_tags=[]))
            value['selection_tags'].append(tag)

    for reason in REASONS:
        take([r for r in rows.values() if r['reason'] == reason], 3, reason)
    for length in (5, 16, 18):
        take([r for r in rows.values() if any(int(c.split(':')[1]) == length
              for c in r['cells'])], 2, 'length_' + str(length))
    for field, boundary in (('psh', 73.4), ('psh', 89.0),
                            ('a3d_average', -0.680027248662),
                            ('a3d_maximum', 1.58023066762925)):
        take(rows.values(), 2, 'near_' + field + '_' + str(boundary),
             key=lambda r: abs(r[field] - boundary))
    if len(chosen) > 24:
        raise ValueError('coverage selection exceeds the frozen 24-record cap')
    for key in sorted(rows):
        if len(chosen) == 24:
            break
        if key not in chosen:
            chosen[key] = dict(rows[key], selection_tags=['hash_order_fill'])
    if len(chosen) != 24:
        raise ValueError('population has fewer than 24 distinct records')
    records = sorted(chosen.values(), key=lambda r: digest([SCHEMA, r['sequence_hash']]))
    for ordinal, row in enumerate(records):
        row['ordinal'] = ordinal
        # Latin rotation balances load order without changing model settings.
        row['mode_order'] = list(MODES[ordinal % 3:] + MODES[:ordinal % 3])
    return {'schema': SCHEMA, 'records': records, 'population_sha256': digest(records),
            'scope': 'enriched_runtime_parity_not_yield_or_binding_quality',
            'source_canonical_json_sha256': {k: digest(v) for k, v in sorted(reports.items())},
            'source_file_sha256': source_file_sha256,
            'profile_count': 72, 'gpu_count': 0, 'cache_reads': False,
            'limits': {'wall_seconds': 1200, 'cpu_affinity_count': 8,
                       'rss_abort_threshold_bytes': 24 * 1024 ** 3,
                       'minimum_available_memory_bytes': 96 * 1024 ** 3,
                       'available_memory_abort_bytes': 64 * 1024 ** 3, 'workers': 4,
                       'profile_seconds': 300},
            'publication_enabled': False, 'runtime_qualified': False,
            'runtime_configuration_revision': 2, 'kernel_enforced_aggregate_memory_limit': False}


def compare_results(manifest, records):
    """Keep failures/missing results in the denominator; never hide gate flips."""
    expected = {(r['sequence_hash'], mode) for r in manifest['records'] for mode in MODES}
    found = {}
    for row in records:
        key = (row['sequence_hash'], row['mode'])
        if key not in expected or key in found:
            raise ValueError('unexpected or repeated measured observation')
        found[key] = row
    report = {'expected': len(expected), 'received': len(found),
              'missing': len(expected - found.keys()), 'pairs': [],
              'failure_counts': dict(Counter(r['status'] for r in records)),
              'qualification': False, 'native_equivalence_claim': False}
    for entry in manifest['records']:
        key = entry['sequence_hash']
        for a, b in (('cli_a', 'cli_b'), ('cli_a', 'warm'), ('cli_b', 'warm')):
            first, second = found.get((key, a)), found.get((key, b))
            pair = {'sequence_hash': key, 'comparison': [a, b], 'complete': False,
                    'statuses': [r['status'] if r else 'missing' for r in (first, second)]}
            pair['status_mismatch'] = pair['statuses'][0] != pair['statuses'][1]
            if first and second and first['status'] == second['status'] == 'ok':
                x, y = first['profile'], second['profile']
                diffs = {f: y[f] - x[f] for f in FIELDS}
                diffs.update({'aggrescan3d.'+f: y['aggrescan3d'][f] - x['aggrescan3d'][f]
                              for f in A3D_FIELDS})
                if not all(math.isfinite(v) for v in diffs.values()):
                    raise ValueError('nonfinite measured profile')
                pair.update(complete=True, numeric_deltas=diffs,
                            numeric_equal=all(type(x[f]) is type(y[f]) and x[f] == y[f] for f in FIELDS)
                                and all(type(x['aggrescan3d'][f]) is type(y['aggrescan3d'][f])
                                        and x['aggrescan3d'][f] == y['aggrescan3d'][f] for f in A3D_FIELDS),
                            gate_flip=first['eligible'] != second['eligible'],
                            a3d_gate_flip=x['aggrescan3d']['passed'] != y['aggrescan3d']['passed'],
                            native_gate_flip=x.get('passed') != y.get('passed'),
                            flag_mismatch=x.get('red_flags') != y.get('red_flags'),
                            reason_change=first['reason'] != second['reason'],
                            structure_equal=x['aggrescan3d']['input_structure_sha256'] ==
                                            y['aggrescan3d']['input_structure_sha256'],
                            normalized_structure_equal=x['aggrescan3d']['normalized_structure_sha256'] ==
                                                       y['aggrescan3d']['normalized_structure_sha256'],
                            wall_seconds=[first['wall_seconds'], second['wall_seconds']],
                            cpu_seconds=[first['cpu_seconds'], second['cpu_seconds']])
            report['pairs'].append(pair)
    report['complete_pairs'] = sum(p['complete'] for p in report['pairs'])
    report['gate_flips'] = sum(p.get('gate_flip', False) or p.get('a3d_gate_flip', False)
                               or p.get('native_gate_flip', False)
                               for p in report['pairs'])
    report['status_mismatches'] = sum(p['status_mismatch'] for p in report['pairs'])
    report['reason_mismatches'] = sum(p.get('reason_change', False) for p in report['pairs'])
    report['flag_mismatches'] = sum(p.get('flag_mismatch', False) for p in report['pairs'])
    for name, field in (('numeric_mismatches', 'numeric_equal'),
                        ('structure_mismatches', 'structure_equal'),
                        ('normalized_structure_mismatches', 'normalized_structure_equal')):
        report[name] = sum(p['complete'] and not p[field] for p in report['pairs'])
    complete = (len(found) == len(expected) and all(r['status'] == 'ok' for r in records)
                and report['complete_pairs'] == 3 * len(manifest['records']))
    parity = (complete and not report['gate_flips'] and
              all(p['normalized_structure_equal'] and p['numeric_equal'] and not p['reason_change']
                  and not p['flag_mismatch']
                  for p in report['pairs']))
    report['complete_data'] = complete
    report['stress_set_parity'] = parity
    report['decision'] = 'NO_GO_INCOMPLETE' if not complete else 'NO_GO_PARITY'
    if complete:
        times = {}
        for metric in ('wall_seconds', 'cpu_seconds'):
            if any(not isinstance(r[metric], (int, float)) or isinstance(r[metric], bool)
                   or not math.isfinite(r[metric]) or r[metric] <= 0 for r in records):
                raise ValueError('invalid measured timing')
            cli = sum(r[metric] for r in records if r['mode'] != 'warm') / 2
            warm = sum(r[metric] for r in records if r['mode'] == 'warm')
            times[metric + '_warm_to_cli_ratio'] = warm / cli
        report.update(times)
        report['timing_diagnostics'] = timing_diagnostics(records)
        report['speed_continuation_passed'] = times['wall_seconds_warm_to_cli_ratio'] <= .90
        report['cpu_continuation_passed'] = times['cpu_seconds_warm_to_cli_ratio'] <= 1.05
        if parity:
            report['decision'] = ('GO_TO_LARGER_EXPERIMENT_ONLY'
                if report['speed_continuation_passed'] and report['cpu_continuation_passed']
                else 'NO_GO_PERFORMANCE')
    # Even zero flips in this small enriched set cannot qualify a new runtime.
    return report


def timing_diagnostics(records):
    """Descriptive strata; primary continuation still includes initialization."""
    def summarize(rows):
        return {'profiles': len(rows),
                'summed_profile_wall_seconds': sum(r['wall_seconds'] for r in rows),
                'summed_profile_cpu_seconds': sum(r['cpu_seconds'] for r in rows),
                'summed_initialization_seconds': sum(r.get('initialization_seconds', 0) for r in rows)}
    init_keys = {r['sequence_hash'] for r in records
                 if r['mode'] == 'warm' and r.get('initialization_seconds', 0) > 0}
    return {'primary_includes_all_initialization': True,
            'by_mode': {m: summarize([r for r in records if r['mode'] == m]) for m in MODES},
            'initialization_bearing_identities': sorted(init_keys),
            'initialization_bearing': {m: summarize([r for r in records if r['mode'] == m
                  and r['sequence_hash'] in init_keys]) for m in MODES},
            'without_initialization_diagnostic_only': {m: summarize([r for r in records if r['mode'] == m
                  and r['sequence_hash'] not in init_keys]) for m in MODES},
            'by_position': {m: {str(p): summarize([r for r in records if r['mode'] == m
                  and r.get('mode_position') == p]) for p in (1, 2, 3)} for m in MODES}}
