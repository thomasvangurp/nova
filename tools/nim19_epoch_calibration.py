#!/usr/bin/env python3
"""Check the new Blackwell reference runtime before interpreting a replay."""
import json
import subprocess
import sys
from pathlib import Path
import time

BASE=Path('/root/nim19-test');RUN=BASE/'epoch'
sys.path.insert(0,str(BASE/'client'))
from nim19_epoch_runner import environment,save
from analyze_nim19_pair import agreement
PINNED_BOLTZ2_SHA256='8b626489353b4cc923721d86da01f4aa9d288566cbbdd00d919d7f6190179863'

def source_matches(current,original):
    return (current['source_sha256']==original['source_sha256']
            and current['model_source_sha256']==original.get('model_source_sha256',PINNED_BOLTZ2_SHA256))

def main():
    started=time.monotonic();save('pixi-calibration.json',dict(stage='running'))
    env=environment();env['CUDA_VISIBLE_DEVICES']='0'
    if '--assess-existing' not in sys.argv:
        r=subprocess.run(['/root/search_python.sh',str(BASE/'client/nim19_paired_benchmark.py'),
             'reference','--seed','68','--repeat','1'],env=env,
             stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1200)
        if r.returncode:
            labels=['CUDA out of memory','ModuleNotFoundError','ImportError','RuntimeError','TypeError',
                    'ValueError','FileNotFoundError','sm_120','no kernel image','invalid device function','not supported']
            save('pixi-calibration.json',dict(stage='failed',exit_code=r.returncode,
                 error_classes=[k for k in labels if k.encode() in r.stdout+r.stderr]));return 1
    original=json.loads((RUN/'paired-reference-0.json').read_text())
    current=json.loads((BASE/'paired-reference-1.json').read_text())
    by_name={r['name']:r for r in current['records']}
    if set(by_name)!=set(r['name'] for r in original['records']):raise RuntimeError('calibration_count')
    result=agreement([r['score'] for r in original['records']],[by_name[r['name']]['score'] for r in original['records']])
    passed=result['spearman']>=0.99 and result['winner_agrees'] and result['top_k_overlap']==5
    # Preserve the strict parity result. An offline matched-runtime experiment
    # can proceed without equivalence: both our set AND the leader are rescored.
    sources_match=source_matches(current,original)
    offline_eligible=result['spearman']>=0.95 and result['winner_agrees'] and result['top_k_overlap']==5 and current['fused_triangular_calls']>0
    save('pixi-calibration.json',dict(stage='complete',passed=passed,agreement=result,
         eligible_for_offline_matched_rescore=offline_eligible,
         source_matches=sources_match,fused_triangular_calls=current['fused_triangular_calls'],
         baseline_model_digest_origin='artifact' if 'model_source_sha256' in original else 'pinned original source revision; older artifact omitted field',
         inference_wall_seconds=current['wall_seconds'],reused_completed_inference='--assess-existing' in sys.argv,
         torch_version=current['torch_version'],cuda_version=current['cuda_version'],elapsed_s=time.monotonic()-started,
         limitation='One target/20 molecules; same source/recipe/seed, new Blackwell-compatible runtime. Not private validator revision certification.'))
    return 0 if offline_eligible and sources_match else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        save('pixi-calibration.json',dict(stage='failed',error_classes=[type(e).__name__]));sys.exit(1)
