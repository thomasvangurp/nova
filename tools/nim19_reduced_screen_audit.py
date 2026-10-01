#!/usr/bin/env python3
"""Small stock-NIM reduced recipe pilot; never changes the six-round search.

Uses public leaderboard molecules for diagnostics only. Source/weights, target,
MSA and MW adapter match the existing test. No raw responses or error logs.
This eight-molecule pilot cannot certify a 1024-to-768 acquisition policy.
"""
import concurrent.futures
import json
import math
from pathlib import Path
import sys
import time

BASE=Path('/root/nim19-test');RUN=BASE/'epoch'
sys.path.insert(0,str(BASE/'client'))
sys.path.insert(0,str(BASE/'oss-repo'))

def controls():
    return dict(recycling_steps=1,sampling_steps=20,diffusion_samples=1,
                sampling_steps_affinity=10,diffusion_samples_affinity=1,
                affinity_mw_correction=False,max_parallel_samples=1,step_scale=1.638,
                output_format='mmcif',write_full_pae=False,write_full_pde=False)

def main():
    import requests
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from compound_miner.database import ReactionDatabase
    from compound_miner.targets import TARGETS
    from nim19_epoch_runner import canonical,save
    from analyze_nim19_pair import agreement,mw_correct

    def stage():
        p=RUN/'progress.json'
        return json.loads(p.read_text()).get('stage') if p.exists() else 'unknown'
    before=stage()
    # Do not launch a competing audit while the main job is GPU scoring.
    if before!='search':
        save('reduced-nim-audit.json',dict(stage='deferred',reason='main_job_not_in_search'))
        return
    board=json.loads((RUN/'leaderboard.json').read_text())
    winner=next(r for r in board['rows'] if r['uid']==board['winner_uid'])
    database=ReactionDatabase()
    candidates=[database.candidate(3,tuple(map(int,r['name'].split(':')[2:])))
                for r in winner['molecules'][:8]]
    if len(candidates)!=8 or any(c is None for c in candidates):raise RuntimeError('pilot_identity')
    save('reduced-nim-audit.json',dict(stage='running',count=8,recipe=controls(),off_chain=True))
    alignment=(BASE/'client/P40261.a3m').read_text()
    def one(candidate):
        payload=dict(controls(),complex_id='reduced_pilot_'+candidate.name.replace(':','_'),
          polymers=[dict(id='A',molecule_type='protein',sequence=TARGETS['P40261'],
             msa={'uniref90':{'a3m':dict(format='a3m',alignment=alignment,rank=0)}})],
          ligands=[dict(id='B',smiles=candidate.smiles,predict_affinity=True)])
        session=requests.Session();session.trust_env=False;begin=time.monotonic()
        response=session.post('http://127.0.0.1:8000/biology/mit/boltz2/predict',json=payload,timeout=180)
        if response.status_code!=200:raise RuntimeError('nim_reduced_http')
        result=response.json();aff=result['affinities']['B']
        scalar=lambda v:float(v[0] if isinstance(v,list) else v)
        raw=scalar(aff['affinity_pred_value']);prob=scalar(aff['affinity_probability_binary'])
        prediction=mw_correct(raw,Descriptors.MolWt(Chem.MolFromSmiles(candidate.smiles)))
        score=(prob-prediction)/candidate.heavy_atoms
        if not result.get('structures') or not 0<=prob<=1 or not all(map(math.isfinite,[raw,prediction,score])):
            raise RuntimeError('nim_reduced_incomplete')
        allowed={'total_time_seconds','input_preparation_time_seconds','dataloader_setup_time_seconds',
          'model_inference_time_seconds','affinity_prediction_time_seconds','postprocessing_time_seconds',
          'response_construction_time_seconds'}
        metrics={k:float(v) for k,v in result.get('metrics',{}).items()
                 if k in allowed and type(v) in [int,float] and math.isfinite(v)}
        return dict(name=candidate.name,score=score,probability=prob,raw_affinity=raw,
                    corrected_affinity=prediction,wall_seconds=time.monotonic()-begin,metrics=metrics)
    waves=[];durations=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for repeat in range(2):
            begin=time.monotonic();waves.append(list(pool.map(one,candidates)));durations.append(time.monotonic()-begin)
    records=[dict(c.as_dict(),score=0) for c in candidates]
    reference=canonical(records,RUN/'reduced-pilot-oss.json',reduced=True)
    names=[c.name for c in candidates]
    by_name={r['name']:r for r in reference['scores']}
    comparisons={}
    for i,wave in enumerate(waves):
        values={r['name']:r['score'] for r in wave}
        comparisons[f'oss_reduced_vs_nim_reduced_{i}']=agreement(
            [by_name[n]['score'] for n in names],[values[n] for n in names])
    full_artifact=RUN/'nim-throughput-canary.json'
    if full_artifact.exists():
        artifact=json.loads(full_artifact.read_text());full={}
        for row in artifact.get('records',[]):
            full.setdefault(row['name'],row['raw_affinity'])
        if set(names).issubset(full):
            full_scores=[(full[c.name]['affinity_probability_binary']-
               mw_correct(full[c.name]['affinity_pred_value'],Descriptors.MolWt(Chem.MolFromSmiles(c.smiles))))/c.heavy_atoms
               for c in candidates]
            for i,wave in enumerate(waves):
                values={r['name']:r['score'] for r in wave}
                comparisons[f'nim_full_vs_nim_reduced_{i}']=agreement(full_scores,[values[n] for n in names])
            comparisons['nim_full_vs_oss_reduced']=agreement(full_scores,[by_name[n]['score'] for n in names])
    comparisons['nim_reduced_repeatability']=agreement([r['score'] for r in waves[0]],[r['score'] for r in waves[1]])
    save('reduced-nim-audit.json',dict(stage='complete',passed=True,count=8,recipe=controls(),
      stock_nim_unmodified=True,canonical_equivalent=False,policy_accepted=False,
      nim_wave_seconds=durations,oss_reduced_wall_seconds=reference['wall_seconds'],
      comparisons=comparisons,records=waves,main_job_stage_before=before,main_job_stage_after=stage(),
      timings_not_isolated=True,off_chain=True,live_submission_authorized=False,
      limitations=['Eight public leader molecules, not a representative acquisition pool',
        'Reused full-NIM canary predictions; not a paired noise-controlled full-quality rerun',
        'NIM affinity recycling remains stock default and skip-affinity-structure is unavailable',
        'Concurrent main search may affect wall times; no isolated speedup claim']))

if __name__=='__main__':
    try:main()
    except Exception as e:
        from nim19_epoch_runner import save
        save('reduced-nim-audit.json',dict(stage='failed',error_class=type(e).__name__,policy_accepted=False))
        sys.exit(1)
