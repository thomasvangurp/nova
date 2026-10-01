#!/usr/bin/env python3
"""Guarded offline eight-GPU NIM experiment; credentials are runtime-only."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error

import nim19_test_control as base
from nim19_seed_control import api as provider_api

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/nim19_epoch/20260917'
MANIFEST=DEST/'instance.json'
LABEL='nova-nim19-epoch-offline-8gpu-20260917'
OFFER=49106941
ROUTE=DEST/'ssh-route.json'

def api(*args,**kwargs):
    for attempt in range(4):
        try:return provider_api(*args,**kwargs)
        except urllib.error.HTTPError as error:
            if error.code!=429 or attempt==3:raise RuntimeError('provider_api_unavailable') from None
            print(json.dumps(dict(event='source',summary={'provider_http_status':429,'retry_delay_seconds':30})),flush=True)
            time.sleep(30)

def guarded(host,saved):
    if host.get('id')!=saved['id'] or host.get('label')!=LABEL or host.get('num_gpus')!=8 or host['id']==51125470:
        raise RuntimeError('instance_guard_failed')
    return host

def instance(fresh=False):
    saved=json.loads(MANIFEST.read_text())
    if not fresh and ROUTE.is_file() and time.time()-ROUTE.stat().st_mtime<300:
        return guarded(json.loads(ROUTE.read_text()),saved)
    _,data=api('instances/?owner=me')
    host=next(h for h in data.get('instances',[]) if h.get('id')==saved['id'])
    guarded(host,saved)
    # Persist only public transport/state fields, never the raw account response,
    # secrets, registry configuration, environment, signed URLs or credentials.
    route={k:host.get(k) for k in ['id','label','num_gpus','gpu_name','cur_state','actual_status','public_ipaddr']}
    route['ports']={'22/tcp':[{'HostPort':p['HostPort']} for p in host.get('ports',{}).get('22/tcp',[])]}
    temp=ROUTE.with_suffix('.partial');temp.write_text(json.dumps(route));temp.replace(ROUTE)
    return route

def ssh(host):
    mapping=host.get('ports',{}).get('22/tcp') or []
    if not mapping:raise RuntimeError('direct_ssh_not_ready')
    return ['ssh','-F','/dev/null','-i',base.SSH_KEY,'-p',str(mapping[0]['HostPort']),
            '-o','ProxyCommand=none','-o','ProxyJump=none',
            '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
            '-o','BatchMode=yes','-o','ConnectTimeout=15',
            '-o','StrictHostKeyChecking=accept-new','-o','LogLevel=ERROR']

def remote(code,payload=None,timeout=60):
    host=instance()
    result=subprocess.run([*ssh(host),'root@'+host['public_ipaddr'],
                           'python3 -c '+shlex.quote(code)],
                          input=json.dumps(payload or {}).encode(),
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    for line in result.stdout.splitlines():
        try:event=json.loads(line)
        except (ValueError,UnicodeError):continue
        if isinstance(event,dict) and event.get('event') in base.SAFE_EVENTS:
            print(json.dumps({k:v for k,v in event.items() if k in base.SAFE_FIELDS}),flush=True)
    if result.returncode:print(json.dumps(dict(event='remote_failed',exit_code=result.returncode)),flush=True)
    return result.returncode

def create():
    if MANIFEST.exists():
        host=instance();print(json.dumps(dict(event='existing',id=host['id'])));return 0
    _,data=api('instances/?owner=me')
    matches=[h for h in data.get('instances',[]) if h.get('label')==LABEL]
    if matches:raise RuntimeError('existing_label_requires_review')
    query={'limit':100,'type':'on-demand','num_gpus':{'eq':8},'verified':{'eq':True},
           'rentable':{'eq':True},'rented':{'eq':False},'gpu_ram':{'gte':90000},
           'cpu_cores_effective':{'gte':96},'cpu_ram':{'gte':288000},'disk_space':{'gte':300}}
    _,data=api('bundles/',query,'POST')
    offer=next(o for o in data.get('offers',[]) if o.get('id')==OFFER)
    if not (offer.get('num_gpus')==8 and offer.get('gpu_name')=='RTX PRO 6000 WS'
            and offer.get('gpu_ram',0)>=90000 and offer.get('dph_total',999)<=10.5
            and tuple(map(int,offer['driver_version'].split('.')))>=(590,44,1)
            and offer.get('direct_port_count',0)>0):raise RuntimeError('offer_guard_failed')
    status,data=api('asks/'+str(OFFER)+'/',dict(image='vastai/base-image:cuda-13.2.1-auto',
                   label=LABEL,disk=300,runtype='ssh_direct',target_state='running',cancel_unavail=True),'PUT')
    if data.get('success') is not True or not isinstance(data.get('new_contract'),int):
        print(json.dumps(dict(event='create_failed',http_status=status)));return 1
    DEST.mkdir(parents=True,exist_ok=True)
    saved=dict(id=data['new_contract'],label=LABEL,offer_id=OFFER,gpu_count=8,
               created_unix=time.time(),quoted_dph=offer['dph_total'],off_chain=True,
               live_submission_authorized=False,lease_safety_seconds=10800)
    MANIFEST.write_text(json.dumps(saved,indent=2))
    print(json.dumps(dict(event='created',id=saved['id'],num_gpus=8,quoted_dph=saved['quoted_dph'])),flush=True)
    return 0

def stop():
    host=instance(fresh=True)
    if host.get('cur_state')=='stopped' and host.get('actual_status') in ['exited','stopped']:
        print(json.dumps(dict(event='stopped',id=host['id'])));return 0
    status,data=api('instances/'+str(host['id'])+'/',dict(state='stopped'),'PUT')
    print(json.dumps(dict(event='stop_requested',http_status=status,passed=data.get('success') is True)))
    return 0 if data.get('success') is True else 1

def watch():
    saved=json.loads(MANIFEST.read_text());deadline=saved['created_unix']+saved['lease_safety_seconds']
    while True:
        try:
            if time.time()>=deadline:
                if stop()==0:return 0
            else:
                host=instance()
                if host.get('cur_state')=='stopped':return 0
        except Exception as error:
            print(json.dumps(dict(event='source',summary={'lease_watch_retry':True,
              'error_class':type(error).__name__})),flush=True)
        delay=30 if time.time()>=deadline else min(30,max(1,deadline-time.time()))
        time.sleep(delay)

def collect():
    host=instance();target=DEST/'results';target.mkdir(exist_ok=True)
    common=['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms',
            '-e',shlex.join(ssh(host))]
    commands=[[*common,'--include=/*.json','--include=/*.csv','--include=/candidates.sqlite',
      '--include=/proxy.joblib','--include=/attempt-1-underfill/','--include=/attempt-2-mixed/',
      '--include=/attempt-3-two-lane/','--include=/attempt-4-large-budget/',
      '--include=/attempt-1-underfill/*.json','--include=/attempt-1-underfill/candidates.sqlite',
      '--include=/attempt-2-mixed/*.json','--include=/attempt-2-mixed/candidates.sqlite',
      '--include=/attempt-3-two-lane/*.json','--include=/attempt-3-two-lane/candidates.sqlite',
      '--include=/attempt-4-large-budget/*.json','--include=/attempt-4-large-budget/candidates.sqlite','--exclude=*',
      'root@'+host['public_ipaddr']+':/root/nim19-test/epoch/',str(target)+'/'],
      [*common,'--ignore-missing-args','root@'+host['public_ipaddr']+':/root/pixi.lock',str(DEST/'pixi.lock')],
      [*common,'--ignore-missing-args','root@'+host['public_ipaddr']+':/root/nim19-test/paired-reference-1.json',str(target)+'/']]
    for command in commands:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        if result.returncode:raise RuntimeError('science_collection_failed')
    print(json.dumps(dict(event='source',summary={'scientific_artifacts_collected':True})),flush=True)
    return 0

def monitor():
    # Existing lease watchdog remains independent. This monitor collects and
    # stops early on completion/failure, so finishing does not leave an idle GPU.
    previous=None
    code=base.DOCKER+r'''
from pathlib import Path
import math
b=Path('/root/nim19-test/epoch');result={}
for name in ['supervisor','progress']:
 p=b/(name+'.json')
 if p.is_file():
  d=json.loads(p.read_text())
  allowed={'waiting_for_pixi_gates','pixi_calibration','six_sequential_rounds','search','warmup','reduced','reduced_nim','full_nim','round_complete','final_canonical','complete','failed'}
  if d.get('stage') in allowed:result[name]=d['stage']
  for k in ['iteration','completed','expected','completed_rounds']:
   if type(d.get(k)) is int and 0<=d[k]<=10000:result[k]=d[k]
  for k in ['phase_seconds','elapsed_seconds']:
   if type(d.get(k)) in [int,float] and math.isfinite(d[k]) and d[k]>=0:result[k]=d[k]
  rounds=d.get('rounds',[])
  if isinstance(rounds,list) and rounds and isinstance(rounds[-1],dict):
   result['last_round']={k:v for k,v in rounds[-1].items() if k in
     ['iteration','full_count','nim_quoted_sum20','entropy','quoted_leader_score','elapsed_seconds','search_seconds','nim_seconds','cumulative_seconds']
     and type(v) in [int,float] and math.isfinite(v)}
emit('source',summary=result)
'''
    saved=json.loads(MANIFEST.read_text());deadline=saved['created_unix']+saved['lease_safety_seconds']-120
    while time.time()<deadline:
        host=instance()
        if host.get('cur_state')=='stopped':return 0
        try:
            result=subprocess.run([*ssh(host),'root@'+host['public_ipaddr'],
              'python3 -c '+shlex.quote(code)],input=b'{}',stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(event='source',summary={'ssh_poll_retry':True})),flush=True)
            time.sleep(15)
            continue
        status={}
        for line in result.stdout.splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('event')=='source':status=event.get('summary',{})
        signature=(status.get('supervisor'),status.get('progress'),status.get('iteration'),status.get('completed'))
        if signature!=previous:
            print(json.dumps(dict(event='source',summary=status)),flush=True);previous=signature
        if status.get('supervisor') in ['complete','failed']:
            try:collect()
            finally:stop()
            return 0
        time.sleep(15)
    # Preserve partial scientific evidence before the hard lease cutoff.
    try:collect()
    finally:stop()
    return 0

def preflight():
    code=base.NATIVE_PREFLIGHT.replace('len(gpus)==1','len(gpus)==8').replace(
        "gpus[0]['memory_mb']>=48000", "all(g['memory_mb']>=90000 and tuple(map(int,g['driver'].split('.'))) >= (590,44,1) for g in gpus)").replace('ram>=59','ram>=268').replace('free>=100','free>=200')
    return remote(code,timeout=600)

def bootstrap():
    # Two independent immutable asset paths; no shared partial writers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        extraction=pool.submit(remote,base.NATIVE_EXTRACT,dict(credential=base.keychain('NGC_API_KEY')),5400)
        assets=pool.submit(remote,base.ASSETS,None,7400)
        codes=[extraction.result(),assets.result()]
    return max(codes)

def setup():
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        h=instance()
        if (h.get('ports',{}).get('22/tcp') or []) and h.get('actual_status')=='running':
            code=preflight()
            if code:return code
            return bootstrap()
        print(json.dumps(dict(event='waiting_for_direct_ssh',elapsed_s=round(600-(deadline-time.monotonic())))),flush=True)
        time.sleep(15)
    raise RuntimeError('ssh_startup_deadline')

def deploy():
    host=instance()
    code="from pathlib import Path;b=Path('/root/nim19-test');[(b/n).mkdir(parents=True,exist_ok=True) for n in ['client','epoch','oss-repo/tools','oss-repo/compound_miner','oss-repo/external_tools/boltz']]"
    if remote(code):return 1
    fleet=Path('/Users/tvgdeena/nova-compound-fleet')
    modules=['__init__','archive','database','features','proxy','contextual','reservoir','search','scorer','types','selection','validity','history','release','upstream_config','fleet_contract','gpu_novelty','own_scores','provider']
    transfers=[
      ([ROOT/'tools'/n for n in ['nim19_native_runtime.py','nim19_epoch_setup.py','nim19_epoch_runner.py','nim19_epoch_calibration.py','nim19_paired_benchmark.py','nim19_benchmark.py','analyze_nim19_pair.py','nim19_pixi_gates.py','nim19_eight_gpu_canary.py','nim19_epoch_supervisor.py']],'/root/nim19-test/client/'),
      ([fleet/'tools/check_compound_with_upstream_validator.py',ROOT/'tools/score_boltz_batch.py'],'/root/nim19-test/client/'),
      ([ROOT/'tools/score_boltz_batch.py'],'/root/nim19-test/oss-repo/tools/'),
      ([fleet/'compound_miner'/(n+'.py') for n in modules]+[ROOT/'compound_miner/targets.py',ROOT/'compound_miner/adaptive_strategy.py'],'/root/nim19-test/oss-repo/compound_miner/'),
      ([ROOT/'external_tools/boltz'/n for n in ['src','pyproject.toml','README.md','LICENSE']],'/root/nim19-test/oss-repo/external_tools/boltz/'),
      ([DEST/n for n in ['challenge.json','leaderboard.json']],'/root/nim19-test/epoch/'),
      ([ROOT/'data/nim19_benchmark/20260917/paired/paired-reference-0.json'],'/root/nim19-test/epoch/'),
      ([ROOT/'data/nim19_benchmark/20260917/paired/paired-portfolio.json'],'/root/nim19-test/'),
      ([Path('/Users/tvgdeena/nova-nanobody-standalone/data/msa_files/P40261.a3m')],'/root/nim19-test/client/'),
    ]
    for files,target in transfers:
        result=subprocess.run(['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms',
             '--exclude=.git','--exclude=.env*','--exclude=__pycache__','--exclude=*.key','--exclude=*.pem',
             '--exclude=*credential*','--exclude=*secret*','--exclude=.npmrc','-e',shlex.join(ssh(host)),
             *map(str,files),'root@'+host['public_ipaddr']+':'+target],
             stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        print(json.dumps(dict(event='source',file_count=len(files),verified=result.returncode==0)),flush=True)
        if result.returncode:return 1
    return 0

def pixi():
    # Reuse prepared science; no NIM runtime patch, credential file transfer or
    # production playbook deployment. Only public helpers/source are synchronized.
    if deploy():return 1
    fleet=Path('/Users/tvgdeena/nova-compound-fleet')
    msa_root=Path('/Users/tvgdeena/nova-nanobody-standalone/data/msa_files')
    msas=sorted(msa_root.glob('*.a3m'))
    if len(msas)!=21:raise RuntimeError('msa_count')
    import hashlib
    (DEST/'msa-checksums.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in msas}))
    (DEST/'asset-checksums.json').write_text(json.dumps({
      'boltz2_conf.ckpt':'090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1',
      'boltz2_aff.ckpt':'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e',
      'mols.tar':'39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7',
      'ccd.pkl':'2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4'}))
    host=instance()
    if remote("from pathlib import Path;b=Path('/root/nim19-test/oss-repo');[(b/n).mkdir(parents=True,exist_ok=True) for n in ['data/msa_files','combinatorial_db']]"):return 1
    transfers=[
      ([fleet/'preflight.sh',fleet/'ansible/files/root/extract_ccd.sh'],'/root/nim19-test/client/'),
      ([fleet/'tools'/n for n in ['gate_audit.py','flag_audit.py','run_epoch_search.py','build_proxy_checkpoint.py','prepare_historical_replay.py']],'/root/nim19-test/oss-repo/tools/'),
      ([fleet/'combinatorial_db/reactions.py'],'/root/nim19-test/oss-repo/combinatorial_db/'),
      (msas,'/root/nim19-test/oss-repo/data/msa_files/'),
      ([DEST/'msa-checksums.json',DEST/'asset-checksums.json'],'/root/nim19-test/epoch/')]
    for files,target in transfers:
        result=subprocess.run(['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms',
          '-e',shlex.join(ssh(host)),*map(str,files),'root@'+host['public_ipaddr']+':'+target],
          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
        if result.returncode:raise RuntimeError('public_artifact_sync')
    return playbook()

def playbook():
    host=instance()
    with tempfile.TemporaryDirectory(prefix='nim19-offline-ansible-') as scratch:
        inventory=Path(scratch)/'inventory.json'
        inventory.write_text(json.dumps({'offline_nim':{'hosts':{'offline':{
          'ansible_host':host['public_ipaddr'],'ansible_port':int(host['ports']['22/tcp'][0]['HostPort']),
          'ansible_user':'root','ansible_ssh_private_key_file':base.SSH_KEY,
          'ansible_ssh_common_args':'-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR',
          'offline_instance_id':host['id'],'offline_gpu_count':8,'live_submission_authorized':False}}}}))
        inventory.chmod(0o600)
        print(json.dumps(dict(event='started',summary={'locked_pixi_compute_playbook':'running'})),flush=True)
        started=time.monotonic()
        result=subprocess.run(['/opt/homebrew/bin/ansible-playbook','-i',str(inventory),
          str(ROOT/'deploy/nim19_offline/compute.yml')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800,
          env=dict(PATH='/opt/homebrew/bin:/usr/bin:/bin',HOME='/Users/tvgdeena',
                   LANG='en_US.UTF-8',LC_ALL='en_US.UTF-8',
                   ANSIBLE_CONFIG=str(ROOT/'deploy/nim19_offline/ansible.cfg')))
        import re
        task_names=re.findall(rb'TASK \[([^\r\n]+)\]',result.stdout)
        # Task names are our static strings, never remote logs or authentication.
        summary=dict(stage='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,
          completed_task_headers=len(task_names),last_task=task_names[-1].decode() if task_names else None,
          elapsed_s=time.monotonic()-started,instance_id=host['id'],direct_ssh=True)
        (DEST/'pixi-playbook.json').write_text(json.dumps(summary))
        print(json.dumps(dict(event='source',summary=summary)),flush=True)
        return result.returncode

def prepare_clients():
    code=base.DOCKER+r'''
from pathlib import Path
b=Path('/root/nim19-test');env=dict(HOME=str(b/'oss-home'),PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8')
for kind in ['cpu','canonical']:
 p=b/'epoch'/(kind+'-setup.json')
 if p.exists():continue
 subprocess.Popen(['python3',str(b/'client/nim19_epoch_setup.py'),kind],env=env,
    stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
emit('started',summary={'cpu_and_canonical_setup':'started'})
'''
    return remote(code)

def progress():
    code=base.DOCKER+r'''
from pathlib import Path
b=Path('/root/nim19-test')
for name in ['cpu-setup.json','canonical-setup.json','preparation.json','calibration.json','pixi-gates.json','pixi-gates-failure.json','pixi-preflight.json','pixi-calibration.json','nim-eight-gpu-canary.json','supervisor.json','progress.json','final.json']:
 p=b/'epoch'/name
 if p.is_file():emit('source',file=name,summary=json.loads(p.read_text()))
for name in ['server-status.json']:
 p=b/name
 if p.is_file():emit('source',file=name,summary=json.loads(p.read_text()))
paths=list((b/'oci/blobs/sha256').glob('*')) if (b/'oci/blobs/sha256').exists() else []
emit('source',summary={'image_downloaded_bytes':sum(p.stat().st_size for p in paths if p.is_file()),'image_blob_files':len(paths),'image_unpacked':(b/'unpack-complete').is_file(),
 'model_downloaded_bytes':sum((b/'models'/n).stat().st_size for n in ['boltz2_conf.ckpt','boltz2_aff.ckpt','mols.tar','ccd.pkl'] if (b/'models'/n).is_file())})
'''
    return remote(code)

def boot():
    code=base.DOCKER+r'''
from pathlib import Path
if not Path('/root/nim19-test/unpack-complete').is_file():
 emit('operation_failed',error_class='ImageUnpackNotVerified');sys.exit(1)
for op in ['direct-probe','direct-start']:
 result=subprocess.run(['python3','/root/nim19-test/client/nim19_native_runtime.py',op,'--expected-gpus','8'],
  stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
 for line in result.stdout.splitlines():
  try:d=json.loads(line)
  except ValueError:continue
  if d.get('event') in ['native_runtime','started','operation_failed']:print(json.dumps(d),flush=True)
 if result.returncode:sys.exit(result.returncode)
'''
    return remote(code,timeout=400)

def calibration():
    code=base.DOCKER+r'''
from pathlib import Path
b=Path('/root/nim19-test')
if (b/'epoch/pixi-calibration.json').exists():emit('started',already_running=True);sys.exit(0)
ready=json.loads((b/'epoch/pixi-gates.json').read_text())
if not ready.get('passed'):emit('operation_failed',error_class='ReferenceNotReady');sys.exit(1)
env=dict(HOME=str(b/'oss-home'),PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8')
subprocess.Popen(['/root/search_python.sh',str(b/'client/nim19_epoch_calibration.py')],env=env,
 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
emit('started',summary={'reference_calibration':'started'})
'''
    return remote(code)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['create','status','preflight','bootstrap','setup','deploy','pixi','playbook','prepare-clients','progress','boot','calibration','collect','monitor','stop','watch'])
    operation=parser.parse_args().operation
    if operation=='create':return create()
    if operation=='stop':return stop()
    if operation=='watch':return watch()
    if operation=='collect':return collect()
    if operation=='monitor':return monitor()
    if operation=='preflight':return preflight()
    if operation=='bootstrap':return bootstrap()
    if operation=='setup':return setup()
    if operation=='deploy':return deploy()
    if operation=='pixi':return pixi()
    if operation=='playbook':return playbook()
    if operation=='prepare-clients':return prepare_clients()
    if operation=='progress':return progress()
    if operation=='boot':return boot()
    if operation=='calibration':return calibration()
    h=instance(fresh=True)
    print(json.dumps({k:h.get(k) for k in ['id','label','cur_state','actual_status','gpu_name','num_gpus']}))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:
        safe={'instance_guard_failed','direct_ssh_not_ready','existing_label_requires_review','offer_guard_failed'}
        reason=str(e) if str(e) in safe else None
        print(json.dumps(dict(event='operation_failed',error_class=type(e).__name__,reason=reason)),flush=True);sys.exit(1)
