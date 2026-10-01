#!/usr/bin/env python3
"""Fixed scientific portfolio and paired local NIM/OSS measurements (no auth)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import types
import urllib.request

BASE=Path('/root/nim19-test')
REPO=BASE/'oss-repo'
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(BASE/'client'))

def save(path,data):
    temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(data,allow_nan=False))
    temporary.replace(path)

def emit(**fields):
    print(json.dumps(dict(event='benchmark_summary',summary=fields)),flush=True)

def portfolio(count=20):
    import pandas as pd
    from rdkit import Chem,DataStructs,RDLogger
    from rdkit.Chem import Descriptors,rdFingerprintGenerator
    from compound_miner.validity import ValidityGate,HistoricalIndex
    from compound_miner.types import Candidate
    # Only the logger is stubbed. Official reaction execution is unchanged.
    logger=types.SimpleNamespace(error=lambda *a:None,warning=lambda *a:None,info=lambda *a:None)
    sys.modules['bittensor']=types.SimpleNamespace(logging=logger)
    from combinatorial_db.reactions import get_available_reactions,get_molecule_ids_for_role,get_smiles_from_reaction
    RDLogger.DisableLog('rdApp.*')
    history_path=BASE/'P40261-history.csv'
    if not history_path.exists():
        with urllib.request.urlopen('https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/P40261_molecules.csv',timeout=180) as response:
            history_path.write_bytes(response.read())
    history=HistoricalIndex(history_path)
    gate=ValidityGate(history)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    db=str(REPO/'combinatorial_db/molecules.sqlite')
    rng=random.Random(20260917)
    pool=[];seen=set()
    for reaction,a,b,c in get_available_reactions(db):
        if reaction not in range(1,6): continue
        roles=[a,b]+([c] if c else [])
        ids=[get_molecule_ids_for_role(db,role) for role in roles]
        accepted=[]
        for attempt in range(3000):
            components=tuple(rng.choice(values) for values in ids)
            name='rxn:'+':'.join(map(str,(reaction,*components)))
            smiles=get_smiles_from_reaction(name)
            if not smiles:continue
            mol=Chem.MolFromSmiles(smiles)
            if mol is None or len(Chem.GetMolFrags(mol))!=1:continue
            candidate=Candidate(name,smiles,reaction,components,Chem.MolToInchiKey(mol),mol.GetNumHeavyAtoms(),Descriptors.NumRotatableBonds(mol))
            if candidate.inchikey in seen or not gate.accepts(candidate,check_similarity=False):continue
            if not history.is_novel(candidate,maximum_similarity=0.5):continue
            seen.add(candidate.inchikey)
            accepted.append((candidate,generator.GetFingerprint(mol)))
            if len(accepted)>=80:break
        if not accepted:raise ValueError('no_legal_reaction_candidates')
        quota=count//5+(reaction<=count%5)
        selected=[accepted.pop(min(range(len(accepted)),key=lambda i:abs(accepted[i][0].heavy_atoms-25)))]
        while len(selected)<quota and accepted:
            i=max(range(len(accepted)),key=lambda i:min(1-DataStructs.TanimotoSimilarity(accepted[i][1],other[1]) for other in selected))
            selected.append(accepted.pop(i))
        pool.extend(candidate.as_dict() for candidate,fp in selected)
    if len(pool)!=count:raise ValueError('underfilled_portfolio')
    artifact=dict(records=pool,history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
                  history_rows=len(pd.read_csv(history_path)),max_similarity_to_historical=0.5,
                  qualification='reaction_chemistry_brenk_atoms_rotors_boltz_safety_and_frozen_history; not a live epoch set/entropy proof')
    save(BASE/'paired-portfolio.json',artifact)
    emit(stage='portfolio_complete',count=len(pool),heavy_atoms=[x['heavy_atoms'] for x in pool],reaction_counts={str(i):sum(x['reaction']==i for x in pool) for i in range(1,6)},history_sha256=artifact['history_sha256'])

def oss(mode,repeat,seed=68,seed_artifact=False):
    if mode=='optimized':
        sys.path.insert(0,str(BASE/'optimized-repo/external_tools/boltz/src'))
    import torch,yaml
    from boltz.main import predict
    from boltz.model.models.boltz2 import Boltz2
    import boltz.model.layers.triangular_mult as triangle_module
    from nim19_benchmark import P40261
    from tools.score_boltz_batch import record_id
    records=json.loads((BASE/'paired-portfolio.json').read_text())['records']
    label=f'paired-{mode}-seed-{seed}' if seed_artifact or seed!=68 else f'paired-{mode}-{repeat}'
    root=BASE/label
    if root.exists() and any(root.iterdir()):
        raise ValueError('existing_pass_directory')
    inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    for record in records:
        payload=dict(version=1,sequences=[{'protein':dict(id='A',sequence=P40261,msa=str(BASE/'client/P40261.a3m'))},{'ligand':dict(id='B',smiles=record['smiles'])}],properties=[{'affinity':dict(binder='B')}])
        (inputs/f"{record_id(record['smiles'])}_P40261.yaml").write_text(yaml.safe_dump(payload,sort_keys=False))
    timings=[];fused_calls=[0]
    original_kernel=triangle_module.kernel_triangular_mult
    def counted_kernel(*args,**kwargs):
        result=original_kernel(*args,**kwargs);fused_calls[0]+=1;return result
    triangle_module.kernel_triangular_mult=counted_kernel
    original=Boltz2.predict_step
    def timed(self,batch,*args,**kwargs):
        torch.cuda.synchronize();start=time.perf_counter()
        result=original(self,batch,*args,**kwargs)
        torch.cuda.synchronize()
        timings.append(dict(affinity=bool(self.affinity_prediction),wall_seconds=time.perf_counter()-start))
        if mode=='optimized':
            timings[-1]['prepare_enabled']=bool(self.structure_module.prepare_diffusion_invariants)
            timings[-1]['direct_fullchunk_enabled']=bool(self.structure_module.direct_diffusion_fullchunk)
        save(BASE/'paired-progress.json',dict(stage=mode,repeat=repeat,seed=seed,completed_predict_steps=len(timings),last_step_seconds=timings[-1]['wall_seconds']))
        return result
    Boltz2.predict_step=timed
    start=time.perf_counter()
    controls=dict(prepare_diffusion_invariants=True,direct_diffusion_fullchunk=True) if mode=='optimized' else {}
    predict(data=str(inputs),out_dir=str(root/'outputs'),cache=str(BASE/'oss-home/.boltz'),
            recycling_steps=3,recycling_steps_affinity=5,sampling_steps=100,diffusion_samples=1,
            sampling_steps_affinity=100,diffusion_samples_affinity=3,seed=seed,
            affinity_mw_correction=True,output_format='mmcif',override=False,num_workers=0,
            affinity_only_output=(mode=='optimized'),in_memory_affinity=(mode=='optimized'),
            compress_pre_affinity=(mode!='optimized'),
            skip_affinity_structure=False,skip_affinity_confidence=False,**controls)
    wall=time.perf_counter()-start
    results=[]
    keys=['affinity_probability_binary','affinity_pred_value','affinity_probability_binary1','affinity_pred_value1','affinity_probability_binary2','affinity_pred_value2','confidence_score']
    for record in records:
        path=root/'outputs/boltz_results_inputs/predictions'/f"{record_id(record['smiles'])}_P40261"
        metrics={}
        for file in path.glob('*.json'):
            if file.name.startswith(('affinity','confidence')):
                metrics.update({k:v for k,v in json.loads(file.read_text()).items() if k in keys and isinstance(v,(float,int)) and math.isfinite(v)})
        if not all(k in metrics for k in keys[:2]):raise ValueError('missing_prediction')
        score=(metrics[keys[0]]-metrics[keys[1]])/record['heavy_atoms']
        results.append(dict(name=record['name'],metrics=metrics,score=score))
    artifact=dict(mode=mode,repeat=repeat,seed=seed,wall_seconds=wall,records=results,predict_step_timings=timings,
                  portfolio_sha256=hashlib.sha256((BASE/'paired-portfolio.json').read_bytes()).hexdigest(),
                  torch_version=torch.__version__,cuda_version=torch.version.cuda,
                  max_vram_bytes=torch.cuda.max_memory_allocated(),
                  execution_controls=controls,
                  fused_triangular_calls=fused_calls[0],
                  model_source_sha256=hashlib.sha256(Path(sys.modules[Boltz2.__module__].__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(sys.modules['boltz.main'].__file__).read_bytes()).hexdigest())
    save(BASE/(label+'.json'),artifact)
    emit(stage='oss_complete',mode=mode,repeat=repeat,seed=seed,count=len(results),wall_seconds=wall,inference_seconds=sum(x['wall_seconds'] for x in timings),max_vram_bytes=artifact['max_vram_bytes'])

def nim(repeats):
    import requests
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from nim19_benchmark import P40261,run_request
    from tools.score_boltz_batch import record_id
    records=json.loads((BASE/'paired-portfolio.json').read_text())['records']
    alignment=(BASE/'client/P40261.a3m').read_text()
    session=requests.Session();session.trust_env=False
    results=[];path=BASE/'paired-nim.json'
    for repeat in range(repeats):
        for record in records:
            request=dict(complex_id=f"{record_id(record['smiles'])}_P40261",polymers=[dict(id='A',molecule_type='protein',sequence=P40261,msa={'uniref90':{'a3m':dict(format='a3m',alignment=alignment,rank=0)}})],
                         ligands=[dict(id='B',smiles=record['smiles'],predict_affinity=True)],recycling_steps=3,sampling_steps=100,diffusion_samples=1,
                         sampling_steps_affinity=100,diffusion_samples_affinity=3,affinity_mw_correction=True,step_scale=1.638,
                         output_format='mmcif',max_parallel_samples=1,write_full_pae=False,write_full_pde=False)
            value=run_request(session,'http://127.0.0.1:8000',dict(name=record['name'],request=request),repeat)
            value['molecular_weight']=Descriptors.MolWt(Chem.MolFromSmiles(record['smiles']))
            value['heavy_atoms']=record['heavy_atoms']
            if value.get('passed'):
                affinity=value['affinities']['B']
                value['score']=(affinity['affinity_probability_binary']-affinity['affinity_pred_value'])/record['heavy_atoms']
            results.append(value)
            save(path,dict(records=results,repeats=repeats))
            emit(stage='nim_request',count=len(results),repeat=repeat,passed=value['passed'],wall_seconds=value['wall_seconds'])
    emit(stage='nim_complete',count=len(results),passed=sum(x['passed'] for x in results))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['portfolio','reference','optimized','nim'])
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--seed',type=int,default=68)
    parser.add_argument('--seed-artifact',action='store_true')
    args=parser.parse_args()
    if args.operation=='portfolio':portfolio(args.count)
    elif args.operation=='nim':nim(args.repeats)
    else:oss(args.operation,args.repeat,args.seed,args.seed_artifact)

if __name__=='__main__':
    try:main()
    except Exception as error:
        emit(stage='failed',error_class=type(error).__name__);sys.exit(1)
