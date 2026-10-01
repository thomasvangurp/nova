#!/usr/bin/env python3
"""Prediction-before-update replay of saved NIM labels; no GPU or downloads.

This is a conditional, fixed-pool surrogate diagnostic, NOT a new search run,
canonical portfolio validation, unbiased test set, or a live-win claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

for repository in (Path(__file__).resolve().parents[1], Path('/root/nim19-test/oss-repo')):
    if (repository/'compound_miner').is_dir():
        sys.path.insert(0, str(repository))
        break

import numpy as np
from scipy.stats import spearmanr

from compound_miner.adaptive_strategy import OnlineResidualProxy
from compound_miner.proxy import TargetProxy
from compound_miner.types import Candidate, ScoredCandidate


def decode(row):
    components = row['components']
    if isinstance(components, str):
        components = json.loads(components)
    c = Candidate(row['name'], row['smiles'], row['reaction'], tuple(components),
                  row['inchikey'], row['heavy_atoms'], row['rotatable_bonds'])
    return ScoredCandidate(c, row['affinity_probability_binary'], row['affinity_pred_value'],
                           row['score'], row['scorer'], row['elapsed_seconds'])


def metrics(predictions, scores, shortlist=192):
    predicted = np.asarray([c.proxy_mean for c in predictions])
    truth = np.asarray([s.score for s in scores])
    names = np.asarray([c.name for c in predictions])
    order = np.lexsort((names, -predicted))[:shortlist]
    oracle = np.argsort(-truth)[:20]
    correlation = float(spearmanr(predicted, truth).statistic)
    return dict(spearman=correlation if np.isfinite(correlation) else None,
                actual_top20_captured=len(set(order) & set(oracle)),
                best20_nim_total_in_shortlist=float(np.sort(truth[order])[-20:].sum()),
                oracle20_nim_total=float(truth[oracle].sum()))


def run(root, rounds, output):
    started = time.monotonic()
    frozen = TargetProxy.load(root/'proxy.joblib')
    first = json.loads((root/'round-1-nim.json').read_text())
    expected = first['scorer']
    adaptive = OnlineResidualProxy(frozen, expected_scorer=expected, jobs=4)
    seen = set()
    records = []
    for iteration in range(1, rounds+1):
        path = root/f'round-{iteration}-nim.json'
        payload = json.loads(path.read_text())
        if payload['scorer'] != expected or payload['inference_controls'] != first['inference_controls']:
            raise ValueError('inconsistent_nim_recipe')
        scores = [decode(row) for row in payload['scores']]
        keys = {s.candidate.inchikey for s in scores}
        if len(keys) != len(scores) or seen & keys or payload['count'] != len(scores):
            raise ValueError('replayed_chemical_identity')
        candidates = [s.candidate for s in scores]
        # The current batch's labels are unavailable until AFTER both predictions.
        before = len(adaptive.observations)
        begin = time.monotonic()
        base_predictions = frozen.predict(candidates)
        adapted_predictions = adaptive.predict(candidates)
        prediction_seconds = time.monotonic()-begin
        row = dict(iteration=iteration, previous_labels_only=before,
                   input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   count=len(scores), frozen=metrics(base_predictions, scores),
                   adaptive=metrics(adapted_predictions, scores),
                   prediction_seconds=prediction_seconds)
        row['update'] = adaptive.observe(scores)
        records.append(row)
        seen.update(keys)
        print(json.dumps({'event':'adaptation_diagnostic', **row}), flush=True)
    result = dict(epoch=25176, reaction=3, rounds=records,
                  elapsed_seconds=time.monotonic()-started,
                  scorer=expected, proxy_sha256=hashlib.sha256((root/'proxy.joblib').read_bytes()).hexdigest(),
                  prediction_before_update=True, fixed_baseline_acquisition_pool=True,
                  retrospective_conditional_diagnostic=True,
                  canonical_validated=False, new_search_trajectory=False,
                  live_win_claim=False, off_chain=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    temporary.write_text(json.dumps(result, allow_nan=False, indent=2))
    temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--rounds', type=int, choices=[6, 12], default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.rounds, args.output)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'event':'adaptation_failed','error_class':type(error).__name__}), flush=True)
        sys.exit(1)
