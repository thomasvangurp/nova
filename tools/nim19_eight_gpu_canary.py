#!/usr/bin/env python3
"""Eight simultaneous real-task API requests; stdlib, no client environment."""
import concurrent.futures
import argparse
import json
import math
from pathlib import Path
import sys
import subprocess
import time
import urllib.request

BASE=Path('/root/nim19-test'); RUN=BASE/'epoch'
OUTPUT='nim-eight-gpu-canary.json'
sys.path.insert(0,str(BASE/'oss-repo'))
from compound_miner.targets import TARGETS

def save(value):
    p=RUN/OUTPUT;tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(value));tmp.replace(p)

def main(waves=1):
    board=json.loads((RUN/'leaderboard.json').read_text())
    winner=next(r for r in board['rows'] if r['uid']==board['winner_uid'])
    alignment=(BASE/'client/P40261.a3m').read_text()
    started=time.monotonic();save(dict(stage='running',requests=8*waves))
    def one(pair):
        index,row=pair
        payload=dict(complex_id='eight_gpu_canary_'+str(int(started))+'_'+str(index),
            polymers=[dict(id='A',molecule_type='protein',sequence=TARGETS['P40261'],
                msa={'uniref90':{'a3m':dict(format='a3m',alignment=alignment,rank=0)}})],
            ligands=[dict(id='B',smiles=row['smiles'],predict_affinity=True)],
            recycling_steps=3,sampling_steps=100,diffusion_samples=1,
            sampling_steps_affinity=100,diffusion_samples_affinity=3,
            affinity_mw_correction=False,max_parallel_samples=1,step_scale=1.638,
            output_format='mmcif',write_full_pae=False,write_full_pde=False)
        request=urllib.request.Request('http://127.0.0.1:8000/biology/mit/boltz2/predict',
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        begin=time.monotonic()
        with opener.open(request,timeout=300) as response:
            assert response.status==200
            result=json.load(response)
        aff=result['affinities']['B']
        scalar=lambda x:float(x[0] if isinstance(x,list) else x)
        values={k:scalar(aff[k]) for k in ['affinity_probability_binary','affinity_pred_value']}
        assert result.get('structures') and all(math.isfinite(v) for v in values.values())
        return dict(index=index,name=row['name'],wall_seconds=time.monotonic()-begin,
            structure_present=True,raw_affinity=values)
    records=[];wave_times=[];peak_util=[0]*8;peak_memory=[0]*8
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for wave in range(waves):
            begin=time.monotonic()
            futures=[pool.submit(one,(wave*8+i,winner['molecules'][(wave*8+i)%20])) for i in range(8)]
            while not all(f.done() for f in futures):
                result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                if result.returncode==0:
                    for line in result.stdout.splitlines():
                        index,memory,util=map(int,line.split(','))
                        peak_util[index]=max(peak_util[index],util);peak_memory[index]=max(peak_memory[index],memory)
                time.sleep(0.5)
            records.extend(f.result() for f in futures)
            wave_times.append(time.monotonic()-begin)
    save(dict(stage='complete',passed=True,requests=8*waves,records=records,wave_seconds=wave_times,
        peak_gpu_utilization_percent=peak_util,peak_vram_mb=peak_memory,
        all_eight_gpus_observed_working=all(x>0 for x in peak_util),
        wall_seconds=time.monotonic()-started,recipe='r3/s100/d1/ar5-default/as100/ad3',
        mw_correction='disabled here; acquisition adapter applies exactly once',
        note='Eight simultaneous API requests per wave; driver utilization sampled while requests run.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--waves',type=int,choices=[1,3],default=1)
    parser.add_argument('--throughput-artifact',action='store_true');args=parser.parse_args()
    if args.throughput_artifact:OUTPUT='nim-throughput-canary.json'
    try:main(args.waves)
    except Exception as e:
        save(dict(stage='failed',error_class=type(e).__name__));sys.exit(1)
