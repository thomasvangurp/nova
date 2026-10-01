#!/usr/bin/env python3
"""Fresh full-quality OSS batches, changing only the scientific seed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

BASE = Path('/root/nim19-test')
STATE = BASE/'seed-runner.json'
SEEDS = [68, 69, 70, 71]
ASSETS = [
    ('boltz-2','boltz2_conf.ckpt',2286561469,'090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'),
    ('boltz-2','boltz2_aff.ckpt',2062139170,'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e'),
    ('boltz-2','mols.tar',1855662080,'39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7'),
    ('boltz-1','ccd.pkl',345859128,'2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4'),
]


def save(**fields):
    temp = STATE.with_suffix('.partial')
    temp.write_text(json.dumps(fields, allow_nan=False));temp.replace(STATE)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    started = time.monotonic()
    save(stage='preflight')
    result = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20)
    rows = result.stdout.decode().strip().splitlines()
    if result.returncode or len(rows)!=1:
        raise RuntimeError('gpu_preflight_failed')
    name, driver, memory = [x.strip() for x in rows[0].split(',')]
    if 'RTX 6000 Ada' not in name or int(memory)<48000 or tuple(map(int,driver.split('.')))<(590,44,1):
        raise RuntimeError('gpu_preflight_failed')
    assets = BASE/'models';assets.mkdir(exist_ok=True)
    for model, filename, size, digest in ASSETS:
        target = assets/filename
        if target.is_file() and target.stat().st_size==size and sha(target)==digest:continue
        save(stage='assets',file=filename,elapsed_s=time.monotonic()-started)
        temp = target.with_suffix('.download-partial')
        count=0;h=hashlib.sha256();last=time.monotonic()
        with urllib.request.urlopen('https://huggingface.co/boltz-community/'+model+'/resolve/main/'+filename,timeout=180) as response,temp.open('wb') as stream:
            while True:
                chunk=response.read(8*1024*1024)
                if not chunk:break
                stream.write(chunk);h.update(chunk);count+=len(chunk)
                if time.monotonic()-last>20:
                    save(stage='assets',file=filename,downloaded_bytes=count,total_bytes=size,elapsed_s=time.monotonic()-started)
                    last=time.monotonic()
        if count!=size or h.hexdigest()!=digest:raise RuntimeError('asset_digest_mismatch')
        temp.replace(target)
    save(stage='environment',elapsed_s=time.monotonic()-started)
    from nim19_pair_setup import main as setup
    if setup():
        save(stage='failed',error_classes=['EnvironmentSetup']);return
    env=dict(HOME=str(BASE/'oss-home'),PATH=str(BASE/'oss-venv/bin')+':/usr/bin:/bin',LANG='C.UTF-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled')
    for seed in SEEDS:
        save(stage='reference',seed=seed,elapsed_s=time.monotonic()-started)
        argv=[str(BASE/'oss-venv/bin/python'),str(BASE/'client/nim19_paired_benchmark.py'),'reference','--seed',str(seed),'--seed-artifact']
        try:result=subprocess.run(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1200)
        except subprocess.TimeoutExpired:
            save(stage='failed',seed=seed,error_classes=['PassDeadline']);return
        if result.returncode:
            labels=['ModuleNotFoundError','ImportError','TypeError','ValueError','AttributeError','RuntimeError','CUDA out of memory']
            classes=[x for x in labels if x.encode() in result.stdout+result.stderr]
            save(stage='failed',seed=seed,exit_code=result.returncode,error_classes=classes);return
        artifact=BASE/f'paired-reference-seed-{seed}.json'
        d=json.loads(artifact.read_text())
        if len(d['records'])!=20 or d['seed']!=seed:
            save(stage='failed',seed=seed,error_classes=['InvalidScientificArtifact']);return
    save(stage='complete',seeds=SEEDS,elapsed_s=time.monotonic()-started,
         gpu_name=name,driver_version=driver,
         asset_digests={filename:sha(assets/filename) for _,filename,_,_ in ASSETS})


if __name__=='__main__':
    try:main()
    except Exception as e:save(stage='failed',error_classes=[type(e).__name__])
