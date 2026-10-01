#!/usr/bin/env python3
"""Warm-feedback integration smoke test; NOT a fresh competitive trial.

Uses saved epoch-25176 NIM labels, never the public leader's labels. Generates
new chemistry with contextual proposals, strict historical novelty, a separate
online residual head, real NIM scoring, and the native validator. Baseline
archives and source are not modified. The output cannot establish epoch fit.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

BASE = Path('/root/nim19-test')
REPO = BASE/'oss-repo'
RUN = BASE/'strategy-smoke-20260918'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE/'client'))
sys.path.insert(0, str(BASE/'strategy-client'))


def run():
    import pandas as pd
    from rdkit import RDLogger
    from compound_miner.adaptive_strategy import (
        FeedbackProposalPolicy, OnlineResidualProxy, refresh_unscored_archive,
    )
    from compound_miner.archive import CandidateArchive
    from compound_miner.experimental_search import SearchEngine
    from compound_miner.history import aggregate_observations
    from compound_miner.database import ReactionDatabase
    from compound_miner.proxy import TargetProxy
    from compound_miner.release import VALIDATOR_EXACT_SCORER
    from compound_miner.selection import select_submission
    from compound_miner.validity import HistoricalIndex
    from check_compound_with_upstream_validator import UpstreamCompoundValidator
    from analyze_compound_adaptation import decode
    from tools.score_boltz_batch import load_records
    started = time.monotonic()
    RDLogger.DisableLog('rdApp.*')
    if RUN.exists():
        raise RuntimeError('smoke_already_exists')
    RUN.mkdir()
    spec = importlib.util.spec_from_file_location('smoke_epoch', BASE/'client/nim19_epoch_runner.py')
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    r.RUN = RUN
    r.EPOCH_DEADLINE = started+240
    prior = BASE/'leader-12x384'
    scores = []
    recipe = None
    for n in range(1,13):
        payload = json.loads((prior/f'round-{n}-nim.json').read_text())
        if recipe is None:
            recipe = payload['inference_controls']
        if payload['inference_controls'] != recipe or payload['scorer'] != r.LABEL:
            raise ValueError('mixed_feedback_recipe')
        scores.extend(decode(row) for row in payload['scores'])
    seen = {s.candidate.inchikey for s in scores}
    if len(scores) != 4608 or len(seen) != 4608:
        raise ValueError('warm_feedback_contract')
    class SearchArchive(CandidateArchive):
        def top(self,reaction,limit=500,scored_only=True,scorer=None):
            return super().top(reaction,limit,scored_only,
                               r.LABEL if scorer==VALIDATOR_EXACT_SCORER else scorer)
    archive = SearchArchive(RUN/'candidates.sqlite')
    archive.add_candidates([s.candidate for s in scores])
    archive.add_scores(scores)
    proxy = OnlineResidualProxy(TargetProxy.load(prior/'proxy.joblib'), expected_scorer=r.LABEL)
    update = proxy.observe(scores)
    def factory(history,reaction,database,rng,exact_parents=None):
        return FeedbackProposalPolicy(history,reaction,database,rng,exact_parents,
                                      feedback=proxy.feedback_frame())
    db = ReactionDatabase()
    engine = SearchEngine(db,aggregate_observations(pd.read_csv(prior/'training.csv')),
                          proxy,archive,HistoricalIndex(prior/'novelty.csv',maximum_similarity=.5),
                          seed=68,proposal_policy_factory=factory)
    search = engine.run_round(3,proposals=5000,exact_shortlist=0,similarity_shortlist=5000,
                              construction_workers=4,similarity_workers=4)
    refresh = refresh_unscored_archive(archive,3,proxy)
    records = load_records(archive.path,3,32,excluded=seen)
    if len(records) < 8:
        raise RuntimeError('smoke_fresh_pool_underfilled')
    candidates = [db.candidate(3,tuple(json.loads(x['components']))) for x in records]
    if any(c is None or c.inchikey in seen for c in candidates):
        raise ValueError('smoke_candidate_identity')
    scored, nim_seconds = r.nim_score(candidates,1)
    archive.add_scores(scored)
    proxy.observe(scored)
    selected = select_submission(archive.top(3,5000,True,scorer=r.LABEL))
    validator = UpstreamCompoundValidator(prior/'upstream-validator',db.path,prior/'novelty.csv',
                                          target='P40261',reaction=3)
    proof = validator.validate([dict(uid=0,records=[s.as_dict() for s in selected])])
    portfolio = proof['portfolios'][0]
    if portfolio['valid_molecules'] != 20 or portfolio['atompair_entropy'] < .25:
        raise ValueError('smoke_native_validation_failed')
    result = dict(stage='complete',epoch=25176,warm_feedback_count=4608,
                  fresh_nim_count=len(scored),fresh_best_nim_score=max(s.score for s in scored),
                  search=search,refresh=refresh,initial_model_update=update,
                  nim_seconds=nim_seconds,elapsed_seconds=time.monotonic()-started,
                  native_validation=proof,scorer=r.LABEL,
                  canonical_rescored=False,new_fair_search_trial=False,
                  proves_epoch_fit=False,live_win_claim=False,off_chain=True)
    r.save('smoke-final.json',result)
    print(json.dumps({'event':'source','summary':dict(
        smoke_complete=True,fresh_nim_count=len(scored),nim_seconds=nim_seconds,
        elapsed_seconds=result['elapsed_seconds'],native_portfolios=proof['portfolios'],
        canonical_rescored=False,new_fair_search_trial=False)}),flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(json.dumps({'event':'source','summary':{
            'smoke_failed':True,'error_class':type(e).__name__}}),flush=True)
        sys.exit(1)
