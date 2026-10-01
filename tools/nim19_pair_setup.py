#!/usr/bin/env python3
"""Isolated public-source OSS setup. Never emit subprocess logs or exceptions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

BASE = Path('/root/nim19-test')
REPO = BASE / 'oss-repo'
STATUS = BASE / 'pair-setup.json'

def save(stage, **fields):
    state = dict(stage=stage, **fields)
    temporary = STATUS.with_suffix('.partial')
    temporary.write_text(json.dumps(state))
    temporary.replace(STATUS)

def main():
    started = time.monotonic()
    home = BASE/'oss-home'
    home.mkdir(exist_ok=True)
    env = dict(HOME=str(home), PATH='/usr/local/bin:/usr/bin:/bin', LANG='C.UTF-8',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE='/dev/null')
    prefix = BASE/'oss-venv'
    commands = [
        ['python3','-m','venv',str(prefix)],
        [str(prefix/'bin/python'),'-m','pip','install','torch==2.6.0',
         '--index-url','https://download.pytorch.org/whl/cu126'],
        [str(prefix/'bin/python'),'-m','pip','install','-c',str(REPO/'requirements/boltz-worker-constraints.txt'),
         '-e',str(REPO/'external_tools/boltz')+'[cuda]', 'pytorch-lightning==2.5.5',
         'rdkit==2024.9.4','pandas==2.2.3'],
    ]
    for stage, command in zip(['venv','torch','dependencies'],commands):
        save(stage,elapsed_s=time.monotonic()-started)
        result = subprocess.run(command, env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1500)
        if result.returncode:
            classes = [x for x in ['No space left','ResolutionImpossible','No matching distribution','ConnectionError','Timeout','externally-managed','ensurepip'] if x.encode() in result.stdout+result.stderr]
            save('failed',failed_stage=stage,exit_code=result.returncode,error_classes=classes)
            return 1
    save('public_database',elapsed_s=time.monotonic()-started)
    path=REPO/'combinatorial_db/molecules.sqlite'
    path.parent.mkdir(parents=True,exist_ok=True)
    expected='a84bd5216d38975b33670a137aaa20f2b3c2a29bc363db28d44f653d8ff2c617'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        with urllib.request.urlopen('https://huggingface.co/datasets/Metanova/Mol-Rxn-DB/resolve/main/molecules.sqlite?download=true',timeout=180) as response:
            data=response.read()
        if hashlib.sha256(data).hexdigest()!=expected:
            save('failed',error_classes=['DatabaseDigestMismatch']);return 1
        path.write_bytes(data)
    cache=home/'.boltz'
    cache.mkdir(exist_ok=True)
    for name in ['boltz2_conf.ckpt','boltz2_aff.ckpt','ccd.pkl','mols.tar']:
        link=cache/name
        if not link.exists(): link.symlink_to(BASE/'models'/name)
    save('complete',elapsed_s=time.monotonic()-started,database_sha256=expected)
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except Exception:
        save('failed',error_classes=['SetupException']);sys.exit(1)
