#!/usr/bin/env python3
"""Measure P(binding probability > 0.7) in uniform reaction-3 combinatorial space.

Why this exists: every ranking and filtering idea tried so far fails because the
high-probability region is absent from what our search proposes AND from what it
scores. Across 7,827 search-directed exact scores exactly one molecule exceeded
probability 0.7; across 271 near-uniform coverage-lane molecules, none did. 271
draws only bound the rate at about 1.1%, which cannot distinguish 1% from the
0.013% our directed search achieves -- a two-order-of-magnitude uncertainty that
decides whether exploration-by-exact-scoring is viable at all.

This draws molecules uniformly at random from the full valid role space, applies
only construction validity, and exactly scores them with the unchanged
full-quality NIM recipe. The 0.5 historical novelty gate is *recorded per
molecule*, never used to filter, so both P(prob>0.7) and
P(prob>0.7 | submittable) can be computed afterwards.

Results are checkpointed continuously: a probe that dies at 60% still answers the
question with 60% of the samples.
"""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import threading
import time

BASE = Path('/root/nim19-test'); REPO = BASE / 'oss-repo'; RUN = BASE / 'uniform-probe-20260918'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(BASE / 'client'))
LABEL = 'nim19-mw-corrected-search-only'
TARGET_SCORES = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
BUDGET_SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 13500
LOCK = threading.Lock()
DEADLINE = None


def save(name, payload):
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / name; temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, allow_nan=False)); temporary.replace(path)


def remaining():
    return math.inf if DEADLINE is None else max(0.0, DEADLINE - time.monotonic())


def controls():
    return dict(recycling_steps=3, sampling_steps=100, diffusion_samples=1,
                sampling_steps_affinity=100, diffusion_samples_affinity=3,
                affinity_mw_correction=False, step_scale=1.638, max_parallel_samples=1,
                output_format='mmcif', write_full_pae=False, write_full_pde=False)


def main():
    global DEADLINE
    started = time.monotonic(); DEADLINE = started + BUDGET_SECONDS
    import requests
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from compound_miner.database import ReactionDatabase
    from compound_miner.targets import TARGETS
    from compound_miner.validity import HistoricalIndex
    from analyze_nim19_pair import mw_correct
    RDLogger.DisableLog('rdApp.*')

    database = ReactionDatabase()
    roles = database.reactions[3].roles
    # Same construction as the runner: the 0.5 threshold lives on the index.
    novelty = HistoricalIndex(RUN / 'novelty.csv', maximum_similarity=0.5)
    alignment = (BASE / 'client/P40261.a3m').read_text()
    rng = random.Random(68)

    save('probe.json', dict(stage='drawing', target=TARGET_SCORES, budget_seconds=BUDGET_SECONDS,
                            uniform_over_full_role_space=True, novelty_gate_recorded_not_applied=True,
                            role_sizes=[len(database.role_ids[r]) for r in roles],
                            off_chain=True, live_submission_authorized=False))

    # Uniform draws over the entire valid role space -- no history weighting, no
    # proxy, no parent mutation. This is the null distribution of the space.
    seen = set(); pool = []
    attempts = 0
    while len(pool) < TARGET_SCORES and attempts < TARGET_SCORES * 60:
        attempts += 1
        components = tuple(rng.choice(database.role_ids[r]) for r in roles)
        candidate = database.candidate(3, components)
        if candidate is None or candidate.inchikey in seen:
            continue
        seen.add(candidate.inchikey); pool.append(candidate)
    save('probe.json', dict(stage='drawn', drawn=len(pool), draw_attempts=attempts,
                            construction_rate=round(len(pool) / max(1, attempts), 5),
                            target=TARGET_SCORES, off_chain=True, live_submission_authorized=False))

    # Novelty is a recorded property here, never a filter.
    novel_flags = {}
    for candidate in pool:
        try:
            novel_flags[candidate.name] = bool(novelty.is_novel(candidate))
        except Exception:
            novel_flags[candidate.name] = None

    records = []; failures = 0
    settings = controls()

    def one(candidate):
        request = dict(complex_id='uniform_probe_' + candidate.name.replace(':', '_'),
                       polymers=[dict(id='A', molecule_type='protein', sequence=TARGETS['P40261'],
                                      msa={'uniref90': {'a3m': dict(format='a3m', alignment=alignment, rank=0)}})],
                       ligands=[dict(id='B', smiles=candidate.smiles, predict_affinity=True)],
                       **settings)
        session = requests.Session(); session.trust_env = False; begin = time.monotonic()
        for attempt in range(1, 4):
            try:
                left = remaining()
                if left <= 1:
                    raise RuntimeError('probe_deadline_exceeded')
                response = session.post('http://127.0.0.1:8000/biology/mit/boltz2/predict',
                                        json=request, timeout=min(180, left))
                if response.status_code != 200:
                    raise RuntimeError('nim_http_status')
                affinity = response.json()['affinities']['B']
                scalar = lambda x: float(x[0] if isinstance(x, list) else x)
                probability = scalar(affinity['affinity_probability_binary'])
                raw = scalar(affinity['affinity_pred_value'])
                weight = Descriptors.MolWt(Chem.MolFromSmiles(candidate.smiles))
                prediction = mw_correct(raw, weight)
                score = (probability - prediction) / candidate.heavy_atoms
                if not 0 <= probability <= 1 or not all(map(math.isfinite, [raw, prediction, score])):
                    raise RuntimeError('nim_nonfinite')
                return dict(name=candidate.name, smiles=candidate.smiles,
                            components=list(candidate.components), heavy_atoms=candidate.heavy_atoms,
                            rotatable_bonds=candidate.rotatable_bonds,
                            affinity_probability_binary=probability, raw_affinity_pred_value=raw,
                            affinity_pred_value=prediction, score=score, scorer=LABEL,
                            molecular_weight=weight, passes_novelty_gate=novel_flags.get(candidate.name),
                            elapsed_seconds=time.monotonic() - begin, attempts=attempt)
            except Exception as error:
                if remaining() <= 1:
                    raise RuntimeError('probe_deadline_exceeded') from None
                if attempt == 3:
                    raise RuntimeError(type(error).__name__) from None
                time.sleep(attempt)

    def checkpoint(final=False):
        probabilities = [r['affinity_probability_binary'] for r in records]
        summary = dict(stage='complete' if final else 'scoring', scored=len(records),
                       failures=failures, target=TARGET_SCORES,
                       elapsed_seconds=time.monotonic() - started,
                       uniform_over_full_role_space=True, scorer=LABEL,
                       inference_controls=settings, off_chain=True, live_submission_authorized=False)
        if probabilities:
            ordered = sorted(probabilities)
            pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
            for threshold in (0.5, 0.6, 0.7, 0.8):
                summary['count_prob_above_%.1f' % threshold] = sum(1 for p in probabilities if p > threshold)
            gated = [r for r in records if r['passes_novelty_gate']]
            summary.update(mean_probability=sum(probabilities) / len(probabilities),
                           median_probability=pick(0.5), p90_probability=pick(0.9),
                           p99_probability=pick(0.99), max_probability=max(probabilities),
                           mean_score=sum(r['score'] for r in records) / len(records),
                           max_score=max(r['score'] for r in records),
                           novelty_gate_pass_count=len(gated),
                           novelty_gate_pass_rate=round(len(gated) / len(records), 5))
        save('probe.json', summary)
        save('probe-records.json', dict(records=records, count=len(records)))
        return summary

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(one, candidate) for candidate in pool]
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception:
                failures += 1
            if len(records) % 32 == 0 or remaining() <= 1:
                with LOCK:
                    checkpoint()
            if remaining() <= 1:
                for pending in futures:
                    pending.cancel()
                break
    summary = checkpoint(final=True)
    print(json.dumps({'event': 'source', 'summary': {k: v for k, v in summary.items()
                                                     if k != 'inference_controls'}}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        save('probe.json', dict(stage='failed', error_class=type(error).__name__,
                                off_chain=True, live_submission_authorized=False))
        print(json.dumps({'event': 'source', 'summary': {'error_class': type(error).__name__}}))
        sys.exit(1)
