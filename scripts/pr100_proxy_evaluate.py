#!/usr/bin/env python3
"""Offline proxy screening experiment. Never emits validator evidence or runs TNP.

Requires numpy/scipy/scikit-learn in an isolated analysis environment. Inputs are
local archived measurements. Unavailable/error observations are censored, not
negative labels. Rankings are evaluated within each challenge cell, and related
sequences stay together in held-out folds. Output contains hashes, not sequences.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {c: i for i, c in enumerate(AA)}
VALID_NEGATIVES = {'psh_threshold', 'developability_red_flags'}
KEEP_FRACTIONS = (0.05, 0.10, 0.20, 0.50, 0.75, 0.90, 1.0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw = gzip.decompress(raw)
    return json.loads(raw), digest(raw)


def outcome(reason):
    if reason == 'passed':
        return 1
    if reason in VALID_NEGATIVES:
        return 0
    return None


def join_records(measurements, sequences, *, cells_key, source):
    output, censored = [], Counter()
    for row in measurements:
        key = row['sequence_hash']
        if key not in sequences:
            raise ValueError('measurement has no matching sequence')
        sequence = sequences[key]
        if digest(sequence.encode()) != key:
            raise ValueError('sequence identity mismatch')
        if not sequence or set(sequence) - set(AA):
            raise ValueError('noncanonical input sequence')
        label = outcome(row['reason'])
        if label is None:
            censored[row['reason']] += 1
            continue
        cells = sorted(set(v.split('/')[0] for v in row[cells_key]))
        if not cells:
            raise ValueError('missing challenge cell')
        output.append({'hash': key, 'sequence': sequence, 'y': label,
                       'psh': row['psh'], 'cells': cells, 'cell': cells[0],
                       'source': source, 'reason': row['reason']})
    return output, dict(censored)


def load_dataset(readiness, backup):
    census, census_sha = read_json(readiness / 'full-developability-result.json')
    bank, bank_sha = read_json(backup / 'full-bank.json')
    if bank_sha != census['full_bank_sha256']:
        raise ValueError('census bank file identity mismatch')
    sequence_index = {}
    for bucket in bank['buckets'].values():
        for row in bucket:
            sequence = row['sequence']
            key = digest(sequence.encode())
            if key != row['pr100_annotation']['sequence_hash']:
                raise ValueError('annotation sequence identity mismatch')
            sequence_index[key] = sequence
    train, censored_train = join_records(census['candidate_measurements'], sequence_index,
                                         cells_key='buckets', source='census')
    pilot, pilot_sha = read_json(readiness / 'conditional-native-pilot-result.json')
    inputs, inputs_sha = read_json(backup / 'conditional-native-pilot-input.json.gz')
    rows = {digest(r['sequence'].encode()): r for r in inputs['rows']}
    for row in pilot['candidate_measurements']:
        if row['sequence_hash'] not in rows or sorted(row['arm_keys']) != sorted(rows[row['sequence_hash']]['arm_keys']):
            raise ValueError('pilot sequence/arm identity mismatch')
    test, censored_test = join_records(pilot['candidate_measurements'],
                                       {k: v['sequence'] for k, v in rows.items()},
                                       cells_key='arm_keys', source='conditional_pilot')
    train_hashes = {r['hash'] for r in train}
    overlap = [r['hash'] for r in test if r['hash'] in train_hashes]
    test = [r for r in test if r['hash'] not in train_hashes]
    metadata = {
        'census_report_sha256': census_sha, 'census_bank_sha256': bank_sha,
        'pilot_report_sha256': pilot_sha, 'pilot_input_sha256': inputs_sha,
        'pilot_declared_input_sha256': pilot['input_sha256'],
        'pilot_whole_file_digest_matches': inputs_sha == pilot['input_sha256'],
        'pilot_all_individual_sequence_hashes_and_arm_keys_verified': True,
        'census_annotation_runtime_qualified': census['annotation_runtime_qualified'],
        'pilot_annotation_runtime_qualified': pilot['annotation_runtime_qualified'],
        'census_censored': censored_train, 'pilot_censored': censored_test,
        'pilot_exact_overlap_removed': len(overlap),
        'label_scope': 'historical native screening outcome; not current end-to-end validator eligibility',
        'multi_cell_rows_use_lexicographically_first_cell': sum(len(r['cells']) > 1 for r in train + test),
    }
    return train, test, metadata


def features(rows):
    result = np.zeros((len(rows), 421), dtype=np.float64)
    for i, row in enumerate(rows):
        sequence = row['sequence']
        result[i, 0] = len(sequence)
        for char in sequence:
            result[i, 1 + AA_INDEX[char]] += 1 / len(sequence)
        for left, right in zip(sequence, sequence[1:]):
            result[i, 21 + AA_INDEX[left] * 20 + AA_INDEX[right]] += 1 / (len(sequence) - 1)
    return result


def aligned_for_grouping(sequence):
    # A coarse motif alignment ONLY for conservative leakage control, not scientific annotation.
    # Pairs without the motif are kept in their literal whole-sequence representation.
    i, j = sequence.rfind('C'), sequence.rfind('WG')
    return (sequence[:i+1], sequence[i+1:j], sequence[j:]) if 0 < i < j else (sequence, '', '')


def family_groups(rows, identity=0.90):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import pdist, squareform
    parts = [aligned_for_grouping(r['sequence']) for r in rows]
    widths = [max(len(p[i]) for p in parts) for i in range(3)]
    aligned = [''.join(part.ljust(width, '-') for part, width in zip(p, widths)) for p in parts]
    encoded = np.array([list(s.encode()) for s in aligned], dtype=np.uint8)
    mismatches = squareform(pdist(encoded, metric='hamming')) * encoded.shape[1]
    lengths = np.array([len(r['sequence']) for r in rows])
    adjacency = mismatches <= (1.0 - identity + 1e-9) * np.maximum.outer(lengths, lengths)
    _, groups = connected_components(csr_matrix(adjacency), directed=False)
    return groups


def recall_lower_bound(successes, total, alpha=0.05):
    """One-sided exact binomial lower bound; optimistic if examples are correlated."""
    if total == 0 or successes == 0:
        return 0.0
    from scipy.stats import beta
    return float(beta.ppf(alpha, successes, total - successes + 1))


def cell_budget_keep(rows, scores, fraction, *, abstain_cells=()):
    cells = defaultdict(list)
    for i, row in enumerate(rows):
        cells[row['cell']].append(i)
    selected = np.zeros(len(rows), dtype=bool)
    abstain_cells = set(abstain_cells)
    for cell, ids in cells.items():
        if cell in abstain_cells:
            selected[ids] = True
            continue
        # Ties get a deterministic hash tiebreak; no labels enter selection.
        ranked = sorted(ids, key=lambda i: (-float(scores[i]), rows[i]['hash']))
        count = min(len(ids), math.ceil(fraction * len(ids)))
        selected[ranked[:count]] = True
    return selected


def evaluate(rows, scores, *, abstain_cells=(), comparable_score_scale=True):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([r['y'] for r in rows])
    positives = int(y.sum())
    table = []
    for fraction in KEEP_FRACTIONS:
        keep = cell_budget_keep(rows, scores, fraction, abstain_cells=abstain_cells)
        tp, calls = int(y[keep].sum()), int(keep.sum())
        recall = tp / positives if positives else None
        table.append({'requested_calls_saved': 1-fraction, 'calls': calls,
                      'actual_calls_saved': 1-calls/len(rows), 'retained_passers': tp,
                      'total_passers': positives, 'lost_passers': positives-tp,
                      'recall': recall, 'precision': tp/calls if calls else None,
                      'optimistic_iid_recall_lower_95': recall_lower_bound(tp, positives),
                      'passers_per_tnp_call_relative_to_unfiltered': (recall/(calls/len(rows))) if positives else None})
    return {'n': len(rows), 'positives': positives,
            'average_precision': float(average_precision_score(y, scores)) if positives and comparable_score_scale else None,
            'roc_auc': float(roc_auc_score(y, scores)) if 0 < positives < len(rows) and comparable_score_scale else None,
            'ranking_metric_scope': 'one fitted model' if comparable_score_scale else 'not pooled across independently fitted folds; see folds',
            'budget_table': table}


def make_model(name):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == 'logistic_composition':
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=0))
    if name == 'extra_trees_dipeptides':
        return ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_leaf=4,
                                    class_weight='balanced', n_jobs=1, random_state=0)
    if name == 'hist_gradient_dipeptides':
        return HistGradientBoostingClassifier(max_iter=100, max_depth=3, min_samples_leaf=20,
                                               l2_regularization=10, class_weight='balanced', random_state=0)
    if name == 'ridge_psh':
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    raise ValueError(name)


def matrix_for(name, matrix):
    return matrix[:, :21] if name == 'logistic_composition' else matrix


def fitted_scores(name, train_x, train_rows, test_x):
    model = make_model(name)
    y = np.array([r['psh'] if name == 'ridge_psh' else r['y'] for r in train_rows])
    if name != 'ridge_psh' and len(set(y)) < 2:
        return np.full(len(test_x), float(y[0]) if len(y) else 0), {'one_class_training': True}
    started = time.perf_counter()
    model.fit(matrix_for(name, train_x), y)
    fitted = time.perf_counter()
    scores = -model.predict(matrix_for(name, test_x)) if name == 'ridge_psh' else model.predict_proba(matrix_for(name, test_x))[:, 1]
    return scores, {'fit_seconds': fitted-started, 'predict_seconds': time.perf_counter()-fitted}


def oof(name, rows, matrix, groups):
    from sklearn.model_selection import GroupKFold
    unique = len(set(groups))
    if unique < 2:
        return None, {'not_evaluable': 'fewer_than_two_groups'}
    scores = np.full(len(rows), np.nan)
    fold_ids = np.full(len(rows), -1, dtype=int)
    folds = []
    for fold, (train, test) in enumerate(GroupKFold(n_splits=min(5, unique)).split(matrix, groups=groups)):
        assert not set(groups[train]) & set(groups[test])
        fold_ids[test] = fold
        scores[test], timing = fitted_scores(name, matrix[train], [rows[i] for i in train], matrix[test])
        fold_metrics = evaluate([rows[i] for i in test], scores[test])
        folds.append({'train_n': len(train), 'test_n': len(test),
                      'train_passers': sum(rows[i]['y'] for i in train),
                      'test_passers': sum(rows[i]['y'] for i in test),
                      'average_precision': fold_metrics['average_precision'],
                      'roc_auc': fold_metrics['roc_auc'], **timing})
    assert np.isfinite(scores).all()
    return scores, {'folds': folds, 'fold_ids': fold_ids.tolist()}


def main(argv=None):
    import scipy
    import sklearn
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--readiness', type=Path, required=True)
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    train, test, metadata = load_dataset(args.readiness, args.backup)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); x = features(train); xt = features(test)
    feature_seconds = time.perf_counter()-started
    groups = family_groups(train + test)
    gt, ge = groups[:len(train)], groups[len(train):]
    train_cells = {r['cell'] for r in train}
    unseen_cells = sorted({r['cell'] for r in test} - train_cells)
    train_families = set(gt)
    external_unseen = [i for i, group in enumerate(ge) if group not in train_families]
    group_sizes = Counter(map(int, groups))
    positive_groups = Counter(int(g) for r, g in zip(train+test, groups) if r['y'])
    report = {'metadata': metadata, 'training_n':len(train), 'training_positives':sum(r['y'] for r in train),
              'software':{'python':platform.python_version(),'numpy':np.__version__,
                          'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
                          'script_sha256':digest(Path(__file__).read_bytes())},
              'external_n':len(test), 'external_positives':sum(r['y'] for r in test),
              'feature_extraction_seconds':feature_seconds, 'feature_count':x.shape[1],
              'features':'whole-sequence length, amino-acid fractions, adjacent-pair fractions',
              'family_grouping':'connected components at90% identity after coarse C/WG motif alignment; approximate ancestry control',
              'group_count':len(group_sizes),'largest_group_sizes':group_sizes.most_common(10),
              'positive_groups':dict(positive_groups), 'unseen_external_cells':unseen_cells,
              'training_positive_groups':dict(Counter(int(g) for r,g in zip(train,gt) if r['y'])),
              'training_cells':dict(Counter(r['cell'] for r in train)),
              'training_positive_cells':dict(Counter(r['cell'] for r in train if r['y'])),
              'external_unseen_family_n':len(external_unseen),
              'external_unseen_family_positives':sum(test[i]['y'] for i in external_unseen),
              'models':{},'production_qualified':False,'deployment_performed':False,
              'limitations':['older annotation runtimes were not qualified',
                             'pilot input whole-file digest differs; individual hashes and arm memberships match',
                             'single stochastic native outcomes, not stable pass probabilities',
                             'family grouping is a sequence-based approximation, not recorded lineage',
                             'binomial intervals assume independent positives and are optimistic for related sequences',
                             'model/budget comparisons are exploratory; intervals do not correct for policy selection',
                             'external call savings condition on labeled complete cases, excluding censored outcomes',
                             'budget curves are exploratory rankings; no calibrated failure probability is claimed',
                             'current worker drops individual failed profiles; future journals must record all outcomes']}
    predictions = []
    for name in ['logistic_composition','extra_trees_dipeptides','hist_gradient_dipeptides','ridge_psh']:
        result = {}
        for split in ['family_held_out', 'cell_held_out']:
            # Exclude ambiguous multi-cell members from the strict cell holdout.
            # Otherwise one identity could belong to both train and test cells.
            ids = np.array([i for i,r in enumerate(train) if split == 'family_held_out' or len(r['cells']) == 1])
            split_rows = [train[i] for i in ids]
            split_groups = gt[ids] if split == 'family_held_out' else np.array([r['cell'] for r in split_rows])
            scores, evidence = oof(name, split_rows, x[ids], split_groups)
            # Different folds have different fitted models. Never compare their raw
            # score scales when spending a batch budget; select within fold+cell.
            fold_ids = evidence.pop('fold_ids', [])
            evaluation_rows = [dict(r,cell=f'{fold}:{r["cell"]}') for r,fold in zip(split_rows,fold_ids)]
            result[split] = {**(evaluate(evaluation_rows,scores,comparable_score_scale=False) if scores is not None else {}),**evidence,
                             'multi_cell_rows_excluded':len(train)-len(split_rows),
                             'budget_scope':'within each held-out fold and challenge cell'}
        scores,timing=fitted_scores(name,x,train,xt)
        result['external_pilot']={**evaluate(test,scores),**timing}
        result['external_pilot_abstain_unseen_cells']=evaluate(test,scores,abstain_cells=unseen_cells)
        if external_unseen:
            result['external_unseen_families']=evaluate([test[i] for i in external_unseen],scores[external_unseen])
        report['models'][name]=result
        predictions.extend({'model':name,'sequence_hash':r['hash'],'observed_pass':r['y'],'cell':r['cell'],'score':float(s)} for r,s in zip(test,scores))
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({'model_completed':name,'external_positive_count':sum(r['y'] for r in test)}),flush=True)
    (args.output/'external_predictions.json').write_text(json.dumps(predictions,indent=2,allow_nan=False))
    print(json.dumps({'report':str(args.output/'report.json'),'production_qualified':False}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
