#!/usr/bin/env python3
"""Coverage + row/column block trial: fresh archive, no warm-start labels.

Same contracts as the baseline epoch runner, with three deliberate differences:
a separate run directory so no prior NIM score row is reused, a warmup drawn
from pre-epoch public history instead of the leaderboard, and the adaptive
proposal/proxy lane enabled from the challenge file. NIM labels remain
search-only and never masquerade as canonical.
"""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

BASE=Path('/root/nim19-test');REPO=BASE/'oss-repo';RUN=BASE/'trial3-20260919'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(BASE/'client'))
LABEL='nim19-mw-corrected-search-only'
REDUCED_LABEL='nim19-r1-s20-as10-ad1-mw-corrected-screen-only'
FULL_CANDIDATES_PER_ROUND=384
FINAL_RESERVE_SECONDS=600
EPOCH_DEADLINE=None
LOCK=threading.Lock()

def save(name,payload):
    p=RUN/name;t=p.with_suffix('.partial');t.write_text(json.dumps(payload,allow_nan=False));t.replace(p)

def state(**fields):save('progress.json',dict(off_chain=True,live_submission_authorized=False,**fields))

def remaining_epoch_seconds():
    return math.inf if EPOCH_DEADLINE is None else max(0,EPOCH_DEADLINE-time.monotonic())

def planned_full_count(remaining,rounds_left,seconds_per_molecule=1.15,search_reserve=150,
                       maximum=FULL_CANDIDATES_PER_ROUND,include_current_search=True):
    searches=rounds_left if include_current_search else rounds_left-1
    available=remaining-FINAL_RESERVE_SECONDS-searches*search_reserve
    count=math.floor(available/(rounds_left*seconds_per_molecule)/8)*8
    if count<64:raise RuntimeError('epoch_budget_insufficient')
    return min(maximum,count)

def run_bounded(command,env,cwd,timeout):
    """Private process group: deadline cleanup never reaches the NIM server."""
    p=subprocess.Popen(command,env=env,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:out,err=p.communicate(timeout=max(1,timeout))
    except subprocess.TimeoutExpired:
        # Include descendant groups (the canonical scorer owns a nested group).
        table=subprocess.run(['ps','-eo','pid=,ppid=,pgid='],capture_output=True,text=True).stdout.splitlines()
        rows=[tuple(map(int,line.split())) for line in table if len(line.split())==3]
        owned={p.pid}
        while True:
            children={pid for pid,parent,pgid in rows if parent in owned}
            if children.issubset(owned):break
            owned.update(children)
        groups={pgid for pid,parent,pgid in rows if pid in owned and pgid in owned}|{p.pid}
        if os.getpgrp() in groups:raise RuntimeError('process_group_guard')
        for sig in [signal.SIGTERM,signal.SIGKILL]:
            for group in groups:
                try:os.killpg(group,sig)
                except ProcessLookupError:pass
        p.communicate()
        raise RuntimeError('epoch_deadline_exceeded') from None
    return subprocess.CompletedProcess(command,p.returncode,out,err)

def acquire(engine,archive,load_records,seen,iteration,started,limit=1024,max_passes=8,allow_partial=False):
    """Fill the fixed funnel with fresh valid chemistry, never relax novelty.

    The 0.5 historical gate can reject >99% of the proxy shortlist. Additional
    search passes are explicit acquisition work, not extra scored iterations.
    Bound the work and persist its cost instead of assuming a full batch.
    """
    passes=[];begin=time.monotonic();previous=0
    for attempt in range(1,max_passes+1):
        if remaining_epoch_seconds()<FINAL_RESERVE_SECONDS+150:raise RuntimeError('epoch_budget_insufficient')
        state(stage='search',iteration=iteration,acquisition_pass=attempt,
              elapsed_seconds=time.monotonic()-started)
        metrics=engine.run_round(3,proposals=250000,exact_shortlist=0,similarity_shortlist=50000,
                                 construction_workers=40,similarity_workers=32)
        if getattr(getattr(engine,'proxy',None),'refresh_acquisition_archive',False) is True:
            from compound_miner.adaptive_strategy import refresh_unscored_archive
            metrics.update(refresh_unscored_archive(archive,3,engine.proxy))
        candidates=load_records(archive.path,3,limit,excluded=seen)
        passes.append(dict(metrics,acquisition_pass=attempt,available_fresh=len(candidates)))
        summary=dict(elapsed_seconds=time.monotonic()-begin,passes=passes,
          acquisition_passes=len(passes),available_fresh=len(candidates),required=limit,
          novelty_threshold=0.5,
          proposed_tuples=sum(p['proposed_tuples'] for p in passes),
          constructed=sum(p['constructed'] for p in passes))
        save(f'round-{iteration}-search.json',summary)
        if len(candidates)==limit:return candidates,summary
        if len(candidates)<previous:raise RuntimeError('acquisition_count_regressed')
        previous=len(candidates)
    if allow_partial and len(candidates)>=64:
        summary['partial_acquisition']=True;save(f'round-{iteration}-search.json',summary)
        return candidates,summary
    raise RuntimeError('acquisition_refill_exhausted')

def environment():
    return dict(HOME=str(BASE/'oss-home'),PATH='/root/.pixi/envs/bz2/bin:/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',NOVA_MAX_SIMILARITY='0.5',
        NOVA_GPU_NOVELTY='1',NOVA_BOLTZ_KERNELS='1')

def canonical(records,output,reduced=False):
    inputs=output.with_name(output.stem+'-inputs.json');save(inputs.name,dict(scores=records))
    (RUN/'scratch'/output.stem).mkdir(parents=True,exist_ok=True)
    command=['/root/search_python.sh',str(REPO/'tools/score_boltz_batch.py'),
        '--archive',str(RUN/'candidates.sqlite'),'--records-json',str(inputs),'--preserve-record-order',
        '--reaction','3','--limit',str(len(records)),'--target','P40261','--msa',str(BASE/'client/P40261.a3m'),
        '--gpus','0,1,2,3,4,5,6,7','--shards-per-gpu','1','--output',str(output),
        '--scratch-root',str(RUN/'scratch'/output.stem)]
    if reduced:command+=['--recycling-steps','1','--sampling-steps','20','--recycling-steps-affinity','1',
                         '--sampling-steps-affinity','10','--diffusion-samples-affinity','1','--skip-affinity-structure']
    result=run_bounded(command,environment(),REPO,min(1800,remaining_epoch_seconds()))
    if result.returncode:
        labels=['CUDA out of memory','ModuleNotFoundError','ImportError','RuntimeError','TypeError','ValueError','FileNotFoundError']
        save(output.stem+'-failure.json',dict(exit_code=result.returncode,error_classes=[x for x in labels if x.encode() in result.stdout+result.stderr]))
        raise RuntimeError('canonical_scoring_failed')
    payload=json.loads(output.read_text())
    if payload['count']!=len(records) or set(x['name'] for x in payload['scores'])!=set(x['name'] for x in records):raise RuntimeError('canonical_count')
    return payload

def nim_controls(reduced=False):
    return dict(recycling_steps=1 if reduced else 3,sampling_steps=20 if reduced else 100,
          diffusion_samples=1,sampling_steps_affinity=10 if reduced else 100,
          diffusion_samples_affinity=1 if reduced else 3,affinity_mw_correction=False,
          step_scale=1.638,max_parallel_samples=1,output_format='mmcif',write_full_pae=False,write_full_pde=False)

def nim_score(candidates,round_number,reduced=False):
    import requests
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from compound_miner.targets import TARGETS
    from compound_miner.types import ScoredCandidate
    from analyze_nim19_pair import mw_correct
    alignment=(BASE/'client/P40261.a3m').read_text();started=time.monotonic();records=[];values=[]
    label=REDUCED_LABEL if reduced else LABEL
    phase='reduced_nim' if reduced else 'full_nim'
    artifact=f'round-{round_number}-reduced-nim' if reduced else f'round-{round_number}-nim'
    controls=nim_controls(reduced)
    state(stage=phase,iteration=round_number,completed=0,expected=len(candidates),phase_seconds=0)
    def one(candidate):
        request=dict(complex_id=phase+'_'+candidate.name.replace(':','_'),
          polymers=[dict(id='A',molecule_type='protein',sequence=TARGETS['P40261'],msa={'uniref90':{'a3m':dict(format='a3m',alignment=alignment,rank=0)}})],
          ligands=[dict(id='B',smiles=candidate.smiles,predict_affinity=True)],
          **controls)
        session=requests.Session();session.trust_env=False;begin=time.monotonic()
        for attempt in range(1,4):
            try:
                remaining=remaining_epoch_seconds()
                if remaining<=1:raise RuntimeError('epoch_deadline_exceeded')
                response=session.post('http://127.0.0.1:8000/biology/mit/boltz2/predict',json=request,timeout=min(180,remaining))
                if response.status_code!=200:raise RuntimeError('nim_http_status')
                result=response.json();aff=result['affinities']['B']
                scalar=lambda x:float(x[0] if isinstance(x,list) else x)
                probability=scalar(aff['affinity_probability_binary']);raw=scalar(aff['affinity_pred_value'])
                mw=Descriptors.MolWt(Chem.MolFromSmiles(candidate.smiles));prediction=mw_correct(raw,mw)
                score=(probability-prediction)/candidate.heavy_atoms
                if not 0<=probability<=1 or not all(map(math.isfinite,[raw,prediction,score])):raise RuntimeError('nim_nonfinite')
                if not result.get('structures'):raise RuntimeError('nim_no_structure')
                value=ScoredCandidate(candidate,probability,prediction,score,label,time.monotonic()-begin)
                record=dict(value.as_dict(),raw_affinity_pred_value=raw,molecular_weight=mw,attempts=attempt,
                            structure_present=True,inference_controls=controls,
                            request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest())
                return value,record
            except Exception as e:
                if remaining_epoch_seconds()<=1:raise RuntimeError('epoch_deadline_exceeded') from None
                if attempt==3:
                    with LOCK:save(artifact+'-failure.json',dict(candidate=candidate.name,attempt=attempt,error_class=type(e).__name__,phase=phase))
                    raise RuntimeError('nim_scoring_failed') from None
                time.sleep(attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(one,c) for c in candidates]
        for future in concurrent.futures.as_completed(futures):
            value,record=future.result();values.append(value);records.append(record)
            if len(records)%16==0 or len(records)==len(candidates):
                save(artifact+'.json',dict(scores=records,count=len(records),wall_seconds=time.monotonic()-started,
                     scorer=label,inference_controls=controls,canonical_equivalent=False))
                state(stage=phase,iteration=round_number,completed=len(records),expected=len(candidates),phase_seconds=time.monotonic()-started)
    return values,time.monotonic()-started

def replay_search_rng(engine,archive,source,completed,proposals=250000):
    """Restore proposal RNG without repeating chemistry construction or inference.

    In the pinned parallel SearchEngine path, only ProposalPolicy consumes this
    RNG. Each saved pass provides its tuple count; saved NIM receipts identify
    exactly which parents were available before the next pass.
    """
    from compound_miner.search import ProposalPolicy
    beginning=time.monotonic();available=set()
    scored=archive.top(3,100000,True,scorer=LABEL)
    by_name={s.candidate.name:s for s in scored}
    for iteration in range(1,completed+1):
        state(stage='resume_rng_replay',iteration=iteration,completed_rounds=completed)
        passes=json.loads((source/f'round-{iteration}-search.json').read_text())['passes']
        parents=[s for s in scored if s.candidate.name in available][:100]
        for receipt in passes:
            policy=ProposalPolicy(engine.history,3,engine.database,engine.rng,exact_parents=parents)
            unique=set();target=proposals*2
            for _ in range(target*2):
                if len(unique)>=target:break
                unique.add(policy.propose())
            if len(unique)!=int(receipt['proposed_tuples']):raise RuntimeError('resume_rng_receipt_mismatch')
        records=json.loads((source/f'round-{iteration}-nim.json').read_text())['scores']
        if len(records)!=384:raise RuntimeError('resume_score_receipt_mismatch')
        for record in records:
            existing=by_name.get(record['name'])
            if existing is None or not math.isclose(existing.score,record['score'],abs_tol=1e-14,rel_tol=0):
                raise RuntimeError('resume_score_receipt_mismatch')
            available.add(record['name'])
    if len(available)!=completed*384 or len(scored)!=len(available):raise RuntimeError('resume_score_receipt_mismatch')
    save('search-rng-state.json',dict(completed_rounds=completed,rng_state=engine.rng.getstate(),
        restored_by_proposal_only_replay=True,elapsed_seconds=time.monotonic()-beginning))
    return time.monotonic()-beginning

def run():
    global EPOCH_DEADLINE
    started=time.monotonic()
    conf=json.loads((RUN/'challenge.json').read_text())
    diagnostic=conf.get('offline_leader_diagnostic',False)
    resumed_rounds=conf.get('resume_completed_rounds',0)
    if type(diagnostic) is not bool or type(resumed_rounds) is not int:raise RuntimeError('challenge_contract')
    if diagnostic:
        if (conf.get('rounds'),conf.get('full_candidates_per_round'),resumed_rounds)!=(12,384,6):
            raise RuntimeError('challenge_contract')
        previous=json.loads((BASE/'epoch/final.json').read_text())
        if previous.get('stage')!='complete' or previous.get('completed_rounds')!=6:
            raise RuntimeError('resume_acceptance_missing')
        started-=previous['elapsed_seconds']
    elif resumed_rounds:raise RuntimeError('challenge_contract')
    budget_seconds=min(conf['budget_seconds'],7200 if diagnostic else 4320)
    EPOCH_DEADLINE=started+budget_seconds
    gates=json.loads((RUN/'pixi-gates.json').read_text())
    preflight=json.loads((RUN/'pixi-preflight.json').read_text())
    calibration=json.loads((RUN/'pixi-calibration.json').read_text())
    if not (gates.get('passed') and preflight.get('stage')=='complete' and preflight.get('failed')==0
            and calibration.get('eligible_for_offline_matched_rescore') and calibration.get('source_matches')):
        raise RuntimeError('pixi_acceptance_missing')
    import pandas as pd
    from rdkit import RDLogger,Chem
    from compound_miner.archive import CandidateArchive
    from compound_miner.database import ReactionDatabase
    from compound_miner.history import aggregate_observations
    from compound_miner.proxy import TargetProxy
    from compound_miner.release import VALIDATOR_EXACT_SCORER
    from compound_miner.search import SearchEngine
    from compound_miner.selection import select_submission
    from compound_miner.types import ScoredCandidate
    from compound_miner.validity import HistoricalIndex
    from check_compound_with_upstream_validator import UpstreamCompoundValidator
    RDLogger.DisableLog('rdApp.*')
    board=json.loads((RUN/'leaderboard.json').read_text())
    round_count=conf['rounds']
    if conf['reaction']!=3 or type(round_count) is not int or round_count not in (6,12) or not board['settled']:
        raise RuntimeError('challenge_contract')
    full_budget=conf.get('full_candidates_per_round',FULL_CANDIDATES_PER_ROUND)
    if type(full_budget) is not int or not 20<=full_budget<=8192:raise RuntimeError('full_budget_contract')
    fixed_batch=conf.get('comparison_fixed_batch',False)
    if type(fixed_batch) is not bool or (fixed_batch and (round_count,full_budget)!=(12,192) and not diagnostic):
        raise RuntimeError('full_budget_contract')
    adaptive_proxy=conf.get('adaptive_proxy',False)
    contextual_rxn3=conf.get('contextual_rxn3',False)
    if type(adaptive_proxy) is not bool or type(contextual_rxn3) is not bool:
        raise RuntimeError('adaptive_strategy_contract')
    rowcolumn=conf.get('rowcolumn_lane',False)
    coverage_fraction=conf.get('coverage_fraction',0.35)
    sweep_fraction=conf.get('sweep_fraction',0.30)
    block_count=conf.get('block_count',6)
    quota_fraction=conf.get('lane_quota_fraction',0.35)
    # Split the quota per lane. A single shared quota ordered by proxy_mean let
    # sweep rows take every reserved slot on the first trial: coverage went
    # 222 -> 3 -> 0 -> 9 -> 8 rows of the batch.
    coverage_quota_share=conf.get('coverage_quota_share',0.5)
    bank_proxy_path=conf.get('bank_proxy','')
    minimum_similarity=conf.get('sweep_minimum_similarity',0.5)
    # Molecules exactly scored in earlier runs against the same target and
    # scorer remain valid candidates: the target does not change and we never
    # submitted them, so they are still novel. Selecting the best 20 from the
    # pooled 7,827 scored molecules gave NIM sum20 1.5630 against 1.4365 for a
    # single run's archive -- a larger gain than any search change measured.
    prior_archives=conf.get('prior_archives',[])
    prior_limit=conf.get('prior_pool_limit',60)
    depth_per_block=conf.get('sweep_depth_per_block',4)
    if type(rowcolumn) is not bool or type(block_count) is not int:raise RuntimeError('rowcolumn_contract')
    for value in (coverage_fraction,sweep_fraction,quota_fraction):
        if type(value) is not float or not 0.0<=value<=1.0:raise RuntimeError('rowcolumn_contract')
    if type(coverage_quota_share) is not float or not 0.0<=coverage_quota_share<=1.0:
        raise RuntimeError('rowcolumn_contract')
    if type(bank_proxy_path) is not str:raise RuntimeError('rowcolumn_contract')
    if type(minimum_similarity) is not float or not 0.0<=minimum_similarity<=1.0:
        raise RuntimeError('rowcolumn_contract')
    if type(prior_archives) is not list or not all(type(x) is str for x in prior_archives):
        raise RuntimeError('prior_pool_contract')
    if type(prior_limit) is not int or not 0<=prior_limit<=500:raise RuntimeError('prior_pool_contract')
    if type(depth_per_block) is not int or not 1<=depth_per_block<=64:raise RuntimeError('rowcolumn_contract')
    if rowcolumn and not (contextual_rxn3 and adaptive_proxy):raise RuntimeError('rowcolumn_contract')
    if conf.get('experimental_search_module',False) is True:
        if not (adaptive_proxy or contextual_rxn3):raise RuntimeError('adaptive_strategy_contract')
        from compound_miner.experimental_search import SearchEngine
    if resumed_rounds and (adaptive_proxy or contextual_rxn3):
        raise RuntimeError('adaptive_resume_not_supported')
    save('strategy.json',dict(strategy='direct_full_nim',full_candidates_per_round=full_budget,
      rounds=round_count,reduced_pass_used=False,full_recipe=nim_controls(),novelty_threshold=0.5,
      independent_final_canonical_rescore=True,budget_seconds=budget_seconds,final_reserve_seconds=FINAL_RESERVE_SECONDS,
      adaptive_batch_size=not fixed_batch,startup_and_warmup_in_budget=True,
      adaptive_proxy=adaptive_proxy,contextual_rxn3=contextual_rxn3,
      feedback_labels_search_only=True,off_chain=True,live_submission_authorized=False))
    if diagnostic:
        save('diagnostic-contract.json',dict(offline_leader_diagnostic=True,resume_completed_rounds=6,
          budget_seconds=budget_seconds,baseline_active_elapsed_seconds=previous['elapsed_seconds'],
          first_six_scores_reused_not_rescored=True,not_a_single_epoch_fit_test=True,
          proposal_rng_reconstructed_from_saved_receipts=True))
    class SearchArchive(CandidateArchive):
        def top(self,reaction,limit=500,scored_only=True,scorer=None):
            # Explicit offline parent-source mapping, never a false exact label.
            return super().top(reaction,limit,scored_only,LABEL if scorer==VALIDATOR_EXACT_SCORER else scorer)
    archive=SearchArchive(RUN/'candidates.sqlite');database=ReactionDatabase()
    history=pd.read_csv(RUN/'training.csv')
    if int(history['epoch'].max())>conf['maximum_training_epoch']:raise RuntimeError('training_leakage')
    frozen_proxy=TargetProxy.load(RUN/'proxy.joblib')
    bank=None
    if bank_proxy_path:
        # Pre-trained on our own exact-scorer labels, chemistry features only.
        # Round one is otherwise spent choosing molecules with a ranker that has
        # no labels at all -- the weakest round of every run so far.
        from compound_miner.bank_proxy import BankProxy
        bank=BankProxy.load(RUN/bank_proxy_path)
        save('bank-proxy.json',dict(bank.metadata(),path=bank_proxy_path,
          replaces_frozen_acquisition_proxy=True))
    feedback=None;policy_options={}
    if adaptive_proxy or contextual_rxn3:
        from compound_miner.adaptive_strategy import OnlineResidualProxy,FeedbackProposalPolicy
        feedback=OnlineResidualProxy(bank if bank is not None else frozen_proxy,expected_scorer=LABEL,
          minimum_labels=32 if adaptive_proxy else 10000000)
        if contextual_rxn3:
            def policy_factory(history,reaction,database,rng,exact_parents=None):
                return FeedbackProposalPolicy(history,reaction,database,rng,exact_parents,
                                              feedback=feedback.feedback_frame())
            policy_options['proposal_policy_factory']=policy_factory
    sweep_state=None
    if rowcolumn:
        from compound_miner.rowcolumn_strategy import (ExplorationSweepPolicy,SweepState,
                                                        tanimoto_reagent_ranker)
        sweep_state=SweepState()
        # Role-C fingerprints are built once; swaps below the threshold retained
        # top-decile molecules at 0.83x the base rate, so they are dropped.
        ranker=tanimoto_reagent_ranker(database)
        save('sweep-ranker.json',dict(role_c_with_structures=ranker.coverage,
          minimum_similarity=minimum_similarity,block_axis='fix_AB_sweep_C'))
        def policy_factory(history,reaction,database,rng,exact_parents=None):
            return ExplorationSweepPolicy(history,reaction,database,rng,exact_parents,
              feedback=feedback.feedback_frame(),state=sweep_state,
              coverage_fraction=coverage_fraction,sweep_fraction=sweep_fraction,
              block_count=block_count,similarity_ranker=ranker,
              minimum_similarity=minimum_similarity,depth_per_block=depth_per_block)
        policy_options['proposal_policy_factory']=policy_factory
    acquisition=feedback if adaptive_proxy else (bank if bank is not None else frozen_proxy)
    engine=SearchEngine(database,aggregate_observations(history),acquisition,
                        archive,HistoricalIndex(RUN/'novelty.csv',maximum_similarity=0.5),seed=68,**policy_options)
    validator=UpstreamCompoundValidator(RUN/'upstream-validator',database.path,RUN/'novelty.csv',target='P40261',reaction=3)
    # Warmup is explicitly included in the simulated epoch budget.
    # Deployable warmup: pre-epoch public history only. Leader molecules stay out
    # of acquisition entirely and are touched only by the final matched audit.
    warm=[];warm_seen=set()
    for _,r in history[history['reaction']==3].sort_values('score',ascending=False).iterrows():
        c=database.candidate(3,(int(r['component_a']),int(r['component_b']),int(r['component_c'])))
        if c is None or c.name in warm_seen:continue
        warm_seen.add(c.name);warm.append(c)
        if len(warm)==8:break
    if len(warm)!=8:raise RuntimeError('warmup_source')
    save('warmup-source.json',dict(source='pre_epoch_training_history',count=len(warm),
      maximum_training_epoch=int(history['epoch'].max()),leader_molecules_used=False,
      names=[c.name for c in warm]))
    state(stage='warmup');_,warm_seconds=nim_score(warm,0)
    rounds=[];seen=set();selected_sets=[];seconds_per_molecule=1.15;search_reserve=150
    replay_seconds=0
    if resumed_rounds:
        source=BASE/'epoch'
        rounds=json.loads((source/'rounds.json').read_text())['rounds']
        if len(rounds)!=resumed_rounds or any(r['full_count']!=384 for r in rounds):
            raise RuntimeError('resume_acceptance_missing')
        selected_sets=[json.loads((source/f'round-{n}-selected.json').read_text())['records'] for n in range(1,resumed_rounds+1)]
        replay_seconds=replay_search_rng(engine,archive,source,resumed_rounds)
        seen={s.candidate.inchikey for s in archive.top(3,100000,True,scorer=LABEL)}
        save('rounds.json',dict(rounds=rounds))
    (RUN/'scratch').mkdir(exist_ok=True)
    for iteration in range(resumed_rounds+1,round_count+1):
        begin=time.monotonic();state(stage='search',iteration=iteration,elapsed_seconds=begin-started)
        # Reuse the established broad acquisition lane selection from the scorer.
        from tools.score_boltz_batch import load_records
        rounds_left=round_count+1-iteration
        planned=full_budget if fixed_batch else planned_full_count(remaining_epoch_seconds(),rounds_left,seconds_per_molecule,search_reserve,full_budget)
        candidates,metrics=acquire(engine,archive,load_records,seen,iteration,started,limit=planned,
                                   max_passes=3,allow_partial=not fixed_batch)
        search_reserve=max(150,metrics['elapsed_seconds']*1.15)
        save('search-rng-state.json',dict(completed_search_round=iteration,rng_state=engine.rng.getstate()))
        after_search=full_budget if fixed_batch else planned_full_count(remaining_epoch_seconds(),rounds_left,seconds_per_molecule,
                                       search_reserve,full_budget,include_current_search=False)
        if after_search<len(candidates):candidates=load_records(archive.path,3,after_search,excluded=seen)
        lane=None
        if sweep_state is not None:
            # Reserve part of the exact-score batch for the tagged lanes. The
            # quota displaces the weakest proxy picks so the total exact-score
            # budget is unchanged, and actual gate-passing coverage is recorded
            # rather than assumed.
            from compound_miner.rowcolumn_strategy import quota_records,merge_with_quota
            quota=int(round(quota_fraction*len(candidates)))
            coverage_quota=int(round(quota*coverage_quota_share))
            sweep_quota=quota-coverage_quota
            tagged={}
            # Exploration is ranked by tree disagreement, not predicted mean: the
            # mean is the statistic most biased against chemistry we have no
            # labels for, which is exactly what this lane exists to reach.
            for record in quota_records(archive.path,3,sweep_state.coverage_components,
                                        coverage_quota,excluded=seen,order_by='proxy_std'):
                tagged.setdefault(record['name'],record)
            for record in quota_records(archive.path,3,sweep_state.sweep_components,
                                        sweep_quota,excluded=seen,order_by='proxy_mean'):
                tagged.setdefault(record['name'],record)
            candidates,lane=merge_with_quota(candidates,list(tagged.values())[:quota])
            picked=[tuple(json.loads(r['components']) if isinstance(r['components'],str) else r['components'])
                    for r in candidates]
            lane.update(sweep_state.summary())
            lane.update(sweep_rows_in_batch=sum(1 for c in picked if c in sweep_state.sweep_components),
                        coverage_rows_in_batch=sum(1 for c in picked if c in sweep_state.coverage_components),
                        quota_target=quota,coverage_quota=coverage_quota,sweep_quota=sweep_quota)
            # Record which lane produced each scored molecule. Counts alone made
            # the lane-versus-lane comparison only approximate after the fact.
            tags=[]
            for record,components in zip(candidates,picked):
                if components in sweep_state.sweep_components:origin='sweep'
                elif components in sweep_state.coverage_components:origin='coverage'
                else:origin='baseline'
                tags.append(dict(name=record['name'],lane=origin))
            save(f'round-{iteration}-lane-tags.json',dict(tags=tags))
            save(f'round-{iteration}-lane.json',lane)
        proposed=[database.candidate(3,tuple(json.loads(r['components']) if isinstance(r['components'],str)
                                             else r['components'])) for r in candidates]
        if any(c is None for c in proposed):raise RuntimeError('candidate_identity')
        scored,nim_seconds=nim_score(proposed,iteration);archive.add_scores(scored)
        feedback_update=feedback.observe(scored) if feedback is not None else None
        if feedback_update is not None:save(f'round-{iteration}-feedback.json',feedback_update)
        seconds_per_molecule=max(1.15,nim_seconds/len(scored)*1.15)
        seen.update(s.candidate.inchikey for s in scored)
        pool=archive.top(3,5000,True,scorer=LABEL);selected=select_submission(pool)
        records=[s.as_dict() for s in selected]
        proof=validator.validate([dict(uid=0,records=records)])
        save(f'round-{iteration}-selected.json',dict(records=records,upstream_validation=proof,scorer=LABEL,canonical_rescored=False))
        selected_sets.append(records)
        rounds.append(dict(iteration=iteration,elapsed_seconds=time.monotonic()-begin,
              cumulative_seconds=time.monotonic()-started,search_seconds=metrics['elapsed_seconds'],
              acquisition_passes=metrics['acquisition_passes'],proposed_tuples=metrics['proposed_tuples'],
              strategy='direct_full_nim',reduced_pass_used=False,reduced_seconds=0,
              nim_seconds=nim_seconds,reduced_count=0,full_count=len(scored),full_recipe=nim_controls(),
              preferred_full_count=full_budget,planned_full_count=planned,adaptive_batch_size=not fixed_batch,
              nim_quoted_sum20=proof['portfolios'][0]['quoted_score'],entropy=proof['portfolios'][0]['atompair_entropy'],
              quoted_leader_score=board['winner_score'],canonical_rescored=False))
        if feedback_update is not None:rounds[-1]['feedback_update']=feedback_update
        if lane is not None:rounds[-1]['lane']=lane
        save('rounds.json',dict(rounds=rounds));state(stage='round_complete',iteration=iteration,rounds=rounds,
             elapsed_seconds=time.monotonic()-started,within_epoch_budget=time.monotonic()-started<=budget_seconds)
    search_complete_seconds=time.monotonic()-started
    state(stage='final_canonical',completed_rounds=round_count,rounds=rounds,elapsed_seconds=search_complete_seconds)
    pool=archive.top(3,60,True,scorer=LABEL);union={s.candidate.name:s.as_dict() for s in pool}
    for records in selected_sets:
        union.update({r['name']:r for r in records})
    # Carry forward the strongest molecules from earlier runs. They are rescored
    # in the same canonical pass as this run's, so nothing is compared across
    # scorers, and identity is re-derived from components rather than trusted.
    prior_names=set();prior_receipt=[]
    for relative in prior_archives:
        source=BASE/relative
        if not source.is_file():
            prior_receipt.append(dict(archive=relative,present=False,loaded=0));continue
        import sqlite3
        with sqlite3.connect('file:%s?mode=ro'%source,uri=True) as connection:
            rows=connection.execute(
              'SELECT name,components,score FROM candidates WHERE reaction=3 AND score IS NOT NULL '
              'AND scorer=? ORDER BY score DESC LIMIT ?',(LABEL,prior_limit)).fetchall()
        loaded=0
        for name,components,value in rows:
            if name in union:continue
            parts=tuple(json.loads(components))
            c=database.candidate(3,parts)
            if c is None or c.name!=name:raise RuntimeError('prior_pool_identity')
            union[name]=dict(c.as_dict(),score=value,scorer=LABEL)
            prior_names.add(name);loaded+=1
        prior_receipt.append(dict(archive=relative,present=True,rows=len(rows),loaded=loaded))
    save('prior-pool.json',dict(archives=prior_receipt,carried_forward=len(prior_names),
      limit_per_archive=prior_limit,rescored_in_this_canonical_pass=True))
    winner=next(r for r in board['rows'] if r['uid']==board['winner_uid']);leader_names=[]
    for row in winner['molecules']:
        parts=tuple(int(x) for x in row['name'].split(':')[2:]);c=database.candidate(3,parts)
        if c is None or Chem.MolToInchiKey(Chem.MolFromSmiles(row['smiles']))!=c.inchikey:raise RuntimeError('leader_identity')
        union[c.name]=dict(c.as_dict(),score=row.get('final_score',0));leader_names.append(c.name)
    final=canonical(list(union.values()),RUN/'final-canonical-scores.json')
    by_name={r['name']:r for r in final['scores']}
    own_names=(set(r['name'] for records in selected_sets for r in records)
               |set(s.candidate.name for s in pool)|prior_names)
    choices=[ScoredCandidate(database.candidate(3,tuple(r['components'])),r['affinity_probability_binary'],r['affinity_pred_value'],r['score'],r['scorer'],r['elapsed_seconds']) for name,r in by_name.items() if name in own_names]
    best=select_submission(choices);best_records=[s.as_dict() for s in best]
    native=validator.validate([dict(uid=0,records=best_records)])
    score=native['portfolios'][0]['quoted_score'];matched_leader=sum(by_name[n]['score'] for n in leader_names)
    comparison=validator.compare_leaderboard(board,[score],epoch=conf['epoch'])
    result=dict(stage='complete',epoch=conf['epoch'],reaction=3,completed_rounds=round_count,rounds=rounds,
       elapsed_seconds=time.monotonic()-started,budget_seconds=budget_seconds,
       within_epoch_budget=time.monotonic()-started<=budget_seconds,warmup_seconds=warm_seconds,
       startup_and_warmup_in_budget=True,adaptive_batch_size=not fixed_batch,
       search_complete_seconds=search_complete_seconds,independent_validation_seconds=time.monotonic()-started-search_complete_seconds,
       offline_leader_diagnostic=diagnostic,resume_completed_rounds=resumed_rounds,
       within_single_epoch_72_minutes=time.monotonic()-started<=4320,
       resume_rng_reconstruction_seconds=replay_seconds,
       acquisition_backend='stock NIM 1.9 direct full-quality',strategy='direct_full_nim',
       full_candidates_per_round=full_budget,reduced_pass_used=False,full_recipe=nim_controls(),
       adaptive_proxy=adaptive_proxy,contextual_rxn3=contextual_rxn3,
       feedback_labels_search_only=True,warmup_source='pre_epoch_training_history',
       rowcolumn_lane=rowcolumn,coverage_fraction=coverage_fraction,sweep_fraction=sweep_fraction,
       block_count=block_count,lane_quota_fraction=quota_fraction,
       sweep_minimum_similarity=minimum_similarity,sweep_axis='fix_AB_sweep_C',
       prior_pool_archives=prior_archives,prior_pool_carried_forward=len(prior_names),
       sweep_depth_per_block=depth_per_block,
       coverage_quota_share=coverage_quota_share,bank_proxy=bank.metadata() if bank is not None else None,
       lane_summary=sweep_state.summary() if sweep_state is not None else None,
       leader_used_only_in_final_audit=True,fresh_archive=True,warm_start_labels=0,
       canonical_sum20=score,quoted_leader_score=board['winner_score'],matched_rescored_leader_sum20=matched_leader,
       beats_quoted_leader=score>board['winner_score'],beats_matched_rescored_leader=score>matched_leader,
       final_records=best_records,upstream_validation=native,leaderboard_comparison=comparison,
       off_chain=True,live_submission_authorized=False,live_win_claim=False,
       reference_runtime='Locked Pixi Blackwell Torch 2.7.1 CUDA 12.8; same pinned Boltz source/checkpoints; separate seed calibration',
       pixi_manifest_sha256=gates['manifest_sha256'],pixi_lock_sha256=gates['lock_sha256'])
    save('final.json',result);state(stage='complete',completed_rounds=round_count,canonical_sum20=score,
       quoted_leader_score=board['winner_score'],matched_rescored_leader_sum20=matched_leader,
       elapsed_seconds=result['elapsed_seconds'],within_epoch_budget=result['within_epoch_budget'])

if __name__=='__main__':
    try:run()
    except Exception as e:
        allowed={'pixi_acceptance_missing','challenge_contract','full_budget_contract','training_leakage','acquisition_count_regressed',
          'acquisition_refill_exhausted','canonical_scoring_failed','canonical_count','candidate_identity',
          'nim_scoring_failed','leader_identity','epoch_budget_insufficient','epoch_deadline_exceeded'}
        allowed.update({'resume_acceptance_missing','resume_rng_receipt_mismatch','resume_score_receipt_mismatch'})
        allowed.update({'adaptive_strategy_contract','adaptive_resume_not_supported','warmup_source','rowcolumn_contract','prior_pool_contract','prior_pool_identity'})
        state(stage='failed',error_class=type(e).__name__,reason=str(e) if str(e) in allowed else None);sys.exit(1)
