#!/usr/bin/env python3
"""Sequential isolated comparison, after the first reference batch finishes."""
import json
from pathlib import Path
import subprocess
import time

BASE=Path('/root/nim19-test')
STATE=BASE/'pair-runner.json'

def save(**fields):
    temp=STATE.with_suffix('.partial')
    temp.write_text(json.dumps(fields));temp.replace(STATE)

def main():
    started=time.monotonic()
    save(stage='waiting_reference_0')
    deadline=started+1500
    while not (BASE/'paired-reference-0.json').is_file():
        if time.monotonic()>deadline:
            save(stage='failed',error_classes=['ReferenceDeadline']);return
        time.sleep(2)
    env=dict(HOME=str(BASE/'oss-home'),PATH=str(BASE/'oss-venv/bin')+':/usr/bin:/bin',LANG='C.UTF-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled')
    jobs=[('nim',['python3',str(BASE/'client/nim19_paired_benchmark.py'),'nim','--repeats','2']),
          ('reference_1',[str(BASE/'oss-venv/bin/python'),str(BASE/'client/nim19_paired_benchmark.py'),'reference','--repeat','1']),
          ('optimized_0',[str(BASE/'oss-venv/bin/python'),str(BASE/'client/nim19_paired_benchmark.py'),'optimized','--repeat','0'])]
    for stage,argv in jobs:
        save(stage=stage,elapsed_s=time.monotonic()-started)
        try:
            result=subprocess.run(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
        except subprocess.TimeoutExpired:
            save(stage='failed',failed_stage=stage,error_classes=['PassDeadline']);return
        if result.returncode:
            events=[]
            for line in result.stdout.splitlines():
                try:
                    x=json.loads(line)
                    if x.get('event')=='benchmark_summary':events.append(x.get('summary'))
                except Exception:pass
            labels=['ModuleNotFoundError','ImportError','TypeError','ValueError','AttributeError','RuntimeError','CUDA out of memory']
            classes=[x for x in labels if x.encode() in result.stdout+result.stderr]
            save(stage='failed',failed_stage=stage,exit_code=result.returncode,error_classes=classes,scientific_events=events);return
    save(stage='complete',elapsed_s=time.monotonic()-started)

if __name__=='__main__':
    try:main()
    except Exception:save(stage='failed',error_classes=['RunnerException'])
