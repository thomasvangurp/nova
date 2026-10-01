#!/usr/bin/env python3
"""Public offline experiment setup; never emit package or HTTP error text."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

BASE=Path('/root/nim19-test')
REPO=BASE/'oss-repo'
RUN=BASE/'epoch'

def save(name,**fields):
    RUN.mkdir(exist_ok=True)
    path=RUN/name;temporary=path.with_suffix('.partial')
    temporary.write_text(json.dumps(fields,allow_nan=False));temporary.replace(path)

def install(kind):
    started=time.monotonic();prefix=BASE/('epoch-cpu' if kind=='cpu' else 'oss-venv')
    env=dict(HOME=str(BASE/'oss-home'),PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             PIP_CONFIG_FILE='/dev/null',PIP_DISABLE_PIP_VERSION_CHECK='1')
    (BASE/'oss-home').mkdir(exist_ok=True)
    commands=[['python3','-m','venv',str(prefix)]]
    if kind=='cpu':
        commands.append([str(prefix/'bin/python'),'-m','pip','install','numpy==1.26.4',
                         'pandas==2.2.3','rdkit==2024.9.4','scikit-learn==1.6.1','requests==2.32.3'])
    else:
        commands.extend([
            [str(prefix/'bin/python'),'-m','pip','install','torch==2.7.1','--index-url','https://download.pytorch.org/whl/cu128'],
            [str(prefix/'bin/python'),'-m','pip','install','-e',str(REPO/'external_tools/boltz')+'[cuda]',
             'torch==2.7.1','numpy==1.26.4','rdkit==2024.9.4','pandas==2.2.3',
             'cuequivariance==0.6.1','cuequivariance-torch==0.6.1','cuequivariance-ops-cu12==0.6.1',
             'cuequivariance-ops-torch-cu12==0.6.1','pytorch-lightning==2.5.5']])
    for i,command in enumerate(commands):
        save(kind+'-setup.json',stage='install',step=i,elapsed_s=time.monotonic()-started)
        result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1500)
        if result.returncode:
            known=['ResolutionImpossible','No matching distribution','No space left','ConnectionError','Timeout']
            save(kind+'-setup.json',stage='failed',step=i,exit_code=result.returncode,
                 error_classes=[k for k in known if k.encode() in result.stdout+result.stderr]);return 1
    if kind=='canonical':
        cache=BASE/'oss-home/.boltz';cache.mkdir(exist_ok=True)
        for name in ['boltz2_conf.ckpt','boltz2_aff.ckpt','ccd.pkl','mols.tar']:
            link=cache/name
            if not link.is_symlink() and not link.exists():link.symlink_to(BASE/'models'/name)
    save(kind+'-setup.json',stage='complete',elapsed_s=time.monotonic()-started)
    if kind=='cpu':
        r=subprocess.run([str(prefix/'bin/python'),__file__,'prepare'],env=env,
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
        return r.returncode
    return 0

def prepare():
    started=time.monotonic();sys.path.insert(0,str(REPO))
    import pandas as pd
    from compound_miner.history import _epoch_rows,aggregate_observations
    from compound_miner.proxy import TargetProxy
    sys.path.insert(0,str(BASE/'client'))
    from check_compound_with_upstream_validator import SOURCE_HASHES,UPSTREAM_COMMIT,PUBLIC_DATABASE_SHA256
    conf=json.loads((RUN/'challenge.json').read_text())
    save('preparation.json',stage='history',completed_epochs=0,total_epochs=len(conf['training_epochs']))
    rows=[];done=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(_epoch_rows,e) for e in conf['training_epochs']]
        for f in concurrent.futures.as_completed(futures):
            rows.extend(f.result());done+=1
            if done%25==0:save('preparation.json',stage='history',completed_epochs=done,total_epochs=len(futures))
    frame=pd.DataFrame.from_records(rows)
    if frame.empty or not set(frame['epoch']).issubset(set(conf['training_epochs'])):raise RuntimeError('training_cutoff')
    frame.to_csv(RUN/'training.csv',index=False)
    revision=conf['history_revision']
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision):raise RuntimeError('history_revision')
    url='https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/'+revision+'/P40261_molecules.csv'
    with urllib.request.urlopen(url,timeout=180) as response:data=response.read()
    (RUN/'novelty.csv').write_bytes(data)
    db=REPO/'combinatorial_db/molecules.sqlite';db.parent.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite',timeout=180) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=PUBLIC_DATABASE_SHA256:raise RuntimeError('database_digest')
    db.write_bytes(data)
    source=RUN/'upstream-validator'
    for name,digest in SOURCE_HASHES.items():
        with urllib.request.urlopen('https://raw.githubusercontent.com/metanova-labs/nova/'+UPSTREAM_COMMIT+'/'+name,timeout=30) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('upstream_source_digest')
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    save('preparation.json',stage='proxy',history_rows=len(frame),maximum_training_epoch=int(frame['epoch'].max()))
    training=aggregate_observations(frame)
    TargetProxy(trees=256,jobs=32).fit(training).save(RUN/'proxy.joblib')
    save('preparation.json',stage='complete',history_rows=len(frame),unique_training_rows=len(training),
         maximum_training_epoch=int(frame['epoch'].max()),novelty_sha256=hashlib.sha256((RUN/'novelty.csv').read_bytes()).hexdigest(),
         database_sha256=PUBLIC_DATABASE_SHA256,upstream_validator_commit=UPSTREAM_COMMIT,elapsed_s=time.monotonic()-started)
    return 0

if __name__=='__main__':
    kind=argparse.ArgumentParser();kind.add_argument('kind',choices=['cpu','canonical','prepare']);a=kind.parse_args()
    try:sys.exit(prepare() if a.kind=='prepare' else install(a.kind))
    except Exception as e:
        save(a.kind+'-setup.json',stage='failed',error_classes=[type(e).__name__]);sys.exit(1)
