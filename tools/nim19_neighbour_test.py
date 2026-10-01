#!/usr/bin/env python3
"""Controlled test: does swapping role C for a CHEMICALLY SIMILAR reagent
preserve a strong molecule better than swapping it for a random one?

Retrospective evidence motivating this (17,812 pairs sharing roles A and B,
differing only in role C, from our own unbiased exact scores):

    similarity of the pair   |dscore|   P(neighbour top-decile | anchor top-decile)
      0.0 - 0.6              0.0180     0.0833   (0.83x base - worse than random)
      0.6 - 0.7              0.0151     0.1583   (1.58x)
      0.7 - 0.8              0.0119     0.2481   (2.48x)
      0.8 - 1.0              0.0101     0.3289   (3.29x)

That is observational: the pairs are whatever our search happened to generate.
This builds the analogues deliberately and scores them, with a matched random-C
control arm drawn from the same anchors, so the comparison is causal rather than
a property of what the search chose to propose.

Both arms use the identical anchor set and the same number of analogues per
anchor. Role C similarity is Tanimoto over the REAGENT structures, not the
assembled molecules. The 0.5 historical novelty gate is recorded, never applied.
"""
import concurrent.futures
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import threading
import time

BASE = Path('/root/nim19-test'); REPO = BASE / 'oss-repo'; RUN = BASE / 'neighbour-test-20260919'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(BASE / 'client'))
LABEL = 'nim19-mw-corrected-search-only'
ANCHORS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PER_ARM = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
LOCK = threading.Lock(); DEADLINE = None


def save(name, payload):
    RUN.mkdir(parents=True, exist_ok=True)
    p = RUN / name; t = p.with_suffix('.partial')
    t.write_text(json.dumps(payload, allow_nan=False)); t.replace(p)


def remaining():
    return math.inf if DEADLINE is None else max(0.0, DEADLINE - time.monotonic())


def controls():
    return dict(recycling_steps=3, sampling_steps=100, diffusion_samples=1,
                sampling_steps_affinity=100, diffusion_samples_affinity=3,
                affinity_mw_correction=False, step_scale=1.638, max_parallel_samples=1,
                output_format='mmcif', write_full_pae=False, write_full_pde=False)


def main():
    global DEADLINE
    started = time.monotonic(); DEADLINE = started + BUDGET
    import requests
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    from compound_miner.database import ReactionDatabase
    from compound_miner.targets import TARGETS
    from compound_miner.validity import HistoricalIndex
    from analyze_nim19_pair import mw_correct
    RDLogger.DisableLog('rdApp.*')

    database = ReactionDatabase(); roles = database.reactions[3].roles
    novelty = HistoricalIndex(RUN / 'novelty.csv', maximum_similarity=0.5)
    alignment = (BASE / 'client/P40261.a3m').read_text()
    rng = random.Random(68)

    # Anchors: our own strongest exactly-scored molecules, from every archive.
    scored = {}
    for name in ['epoch', 'strategy-trial-20260918', 'rowcolumn-trial-20260918',
                 'leader-12x384', 'schedule-comparison/12x192']:
        path = BASE / name / 'candidates.sqlite'
        if not path.is_file():
            continue
        with sqlite3.connect('file:%s?mode=ro' % path, uri=True) as connection:
            for key, components, score in connection.execute(
                    'SELECT name,components,score FROM candidates '
                    'WHERE reaction=3 AND score IS NOT NULL'):
                scored.setdefault(key, (tuple(json.loads(components)), score))
    ranked = sorted(scored.items(), key=lambda kv: -kv[1][1])[:ANCHORS]
    save('test.json', dict(stage='anchors', anchors=len(ranked), per_arm=PER_ARM,
                           anchor_scores=[round(v[1], 6) for _, v in ranked],
                           pool_scored=len(scored), off_chain=True,
                           live_submission_authorized=False))

    # Role-C reagent structures, for similarity ranking.
    with sqlite3.connect('file:%s?mode=ro' % (REPO / 'combinatorial_db/molecules.sqlite'), uri=True) as connection:
        smiles = {int(i): s for i, s in connection.execute('SELECT mol_id,smiles FROM molecules')}
    role_c = [int(x) for x in database.role_ids[roles[2]]]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    for identifier in role_c:
        molecule = Chem.MolFromSmiles(smiles.get(identifier, '')) if identifier in smiles else None
        if molecule is not None:
            fingerprints[identifier] = generator.GetFingerprint(molecule)

    jobs = []
    for key, (components, score) in ranked:
        anchor_c = components[2]
        if anchor_c not in fingerprints:
            continue
        others = [i for i in fingerprints if i != anchor_c]
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprints[anchor_c],
                                                        [fingerprints[i] for i in others])
        order = sorted(zip(others, similarity), key=lambda x: -x[1])
        picked = 0
        for identifier, value in order:                       # SIMILAR arm
            if picked >= PER_ARM:
                break
            candidate = database.candidate(3, (components[0], components[1], identifier))
            if candidate is None:
                continue
            jobs.append(dict(arm='similar', anchor=key, anchor_score=score,
                             similarity=float(value), candidate=candidate)); picked += 1
        pool = [i for i, _ in order]; rng.shuffle(pool); picked = 0
        lookup = dict(order)
        for identifier in pool:                               # RANDOM control arm
            if picked >= PER_ARM:
                break
            candidate = database.candidate(3, (components[0], components[1], identifier))
            if candidate is None:
                continue
            jobs.append(dict(arm='random', anchor=key, anchor_score=score,
                             similarity=float(lookup[identifier]), candidate=candidate)); picked += 1
    seen = set(); unique = []
    for job in jobs:
        if job['candidate'].inchikey in seen:
            continue
        seen.add(job['candidate'].inchikey); unique.append(job)
    jobs = unique
    save('test.json', dict(stage='built', jobs=len(jobs),
                           similar=sum(1 for j in jobs if j['arm'] == 'similar'),
                           random=sum(1 for j in jobs if j['arm'] == 'random'),
                           anchors=len(ranked), off_chain=True, live_submission_authorized=False))

    records = []; failures = 0; settings = controls()

    def one(job):
        candidate = job['candidate']
        request = dict(complex_id='neighbour_' + candidate.name.replace(':', '_'),
                       polymers=[dict(id='A', molecule_type='protein', sequence=TARGETS['P40261'],
                                      msa={'uniref90': {'a3m': dict(format='a3m', alignment=alignment, rank=0)}})],
                       ligands=[dict(id='B', smiles=candidate.smiles, predict_affinity=True)],
                       **settings)
        session = requests.Session(); session.trust_env = False
        for attempt in range(1, 4):
            try:
                left = remaining()
                if left <= 1:
                    raise RuntimeError('deadline')
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
                return dict(arm=job['arm'], anchor=job['anchor'], anchor_score=job['anchor_score'],
                            reagent_similarity=job['similarity'], name=candidate.name,
                            smiles=candidate.smiles, components=list(candidate.components),
                            heavy_atoms=candidate.heavy_atoms,
                            affinity_probability_binary=probability, affinity_pred_value=prediction,
                            score=score, delta_vs_anchor=score - job['anchor_score'],
                            passes_novelty_gate=bool(novelty.is_novel(candidate)), scorer=LABEL)
            except Exception as error:
                if remaining() <= 1:
                    raise RuntimeError('deadline') from None
                if attempt == 3:
                    raise RuntimeError(type(error).__name__) from None
                time.sleep(attempt)

    def summarise(final=False):
        out = dict(stage='complete' if final else 'scoring', scored=len(records),
                   failures=failures, total_jobs=len(jobs), anchors=len(ranked),
                   elapsed_seconds=time.monotonic() - started, scorer=LABEL,
                   off_chain=True, live_submission_authorized=False)
        for arm in ('similar', 'random'):
            rows = [r for r in records if r['arm'] == arm]
            if not rows:
                continue
            deltas = [r['delta_vs_anchor'] for r in rows]
            out[arm] = dict(n=len(rows),
                            mean_score=sum(r['score'] for r in rows) / len(rows),
                            max_score=max(r['score'] for r in rows),
                            mean_probability=sum(r['affinity_probability_binary'] for r in rows) / len(rows),
                            mean_delta_vs_anchor=sum(deltas) / len(deltas),
                            beat_anchor=sum(1 for d in deltas if d > 0),
                            beat_anchor_rate=round(sum(1 for d in deltas if d > 0) / len(rows), 4),
                            mean_reagent_similarity=sum(r['reagent_similarity'] for r in rows) / len(rows),
                            novelty_pass_rate=round(sum(1 for r in rows if r['passes_novelty_gate']) / len(rows), 4))
        save('test.json', out); save('test-records.json', dict(records=records, count=len(records)))
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(one, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception:
                failures += 1
            if len(records) % 24 == 0:
                with LOCK:
                    summarise()
            if remaining() <= 1:
                for pending in futures:
                    pending.cancel()
                break
    out = summarise(final=True)
    print(json.dumps({'event': 'source', 'summary': {k: v for k, v in out.items() if k != 'inference_controls'}}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        save('test.json', dict(stage='failed', error_class=type(error).__name__))
        print(json.dumps({'event': 'source', 'summary': {'error_class': type(error).__name__}}))
        sys.exit(1)
