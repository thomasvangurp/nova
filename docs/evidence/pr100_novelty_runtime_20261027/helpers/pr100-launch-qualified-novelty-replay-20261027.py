import pathlib,json,base64,subprocess
report=pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027')
prepared=json.loads((report/'qualification-preparation.json').read_text())
identities={h:v['value']['result']['identity_sha256'] for h,v in prepared.items()}
worker=pathlib.Path('/tmp/pr100-qualified-novelty-replay-worker-20261027.py').read_text()
remote=r'''import gzip,json,pathlib,base64,time,subprocess,os
p=json.loads(gzip.decompress(__import__('sys').stdin.buffer.read()));prefix=pathlib.Path('/root/pr100-novelty-runtime-20261027');base=prefix/'replay-25967-v2'
if base.exists():print(json.dumps({'status':'already_exists','directory':str(base)}));raise SystemExit(0)
base.mkdir()
for name,encoded in p['files'].items():
 assert name in ('input.json','old-archive.csv','new-archive.csv','worker.py');(base/name).write_bytes(base64.b64decode(encoded))
runtime={'python':str(prefix/'bin/novelty-python'),'path':str(prefix/'bin')+':/usr/local/bin:/usr/bin:/bin','python_path':str(prefix/'shim'),'identity_sha256':p['identity']}
(base/'qualification.json').write_text(json.dumps({'runtime':runtime}))
now=time.time();(base/'clock.json').write_text(json.dumps({'epoch':1,'epoch_length':100000,'block':100001,'updated_at_unix':now}))
env={'PATH':runtime['path'],'PYTHONPATH':str(prefix/'qualification-source')+':'+runtime['python_path'],'HOME':'/root','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
spec={'command':[runtime['python'],'-B',str(base/'worker.py')],'cwd':str(prefix/'qualification-source'),'environment':env,'wall_seconds':550,'deadline_unix':now+550,'receipt_path':str(base/'supervisor-receipt.json'),'cpu_slots':8,'clock_path':str(base/'clock.json'),'epoch':1,'epoch_length':100000,'reserve_blocks':0}
(base/'spec.json').write_text(json.dumps(spec));supervisor='/root/releases/22407a4ae6324b8c051ef89f738eb74fddaad978/scripts/pr100_live_supervisor.py'
child=subprocess.Popen([runtime['python'],'-B',supervisor,'--spec',str(base/'spec.json')],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'status':'launched','directory':str(base),'pid':child.pid,'runtime_sha256':p['identity'],'deadline_unix':now+550}))
'''
pi=r'''import pathlib,subprocess,json,base64,gzip,shlex
root=pathlib.Path('/home/thomasg/pr100-controller-state/25967/attempts/00006');request=json.loads((root/'novelty-rechecks/faa66bc72600e2fd4979754ab02c2637ef7d8725ee487fb5cd48cfa260b7eb2e/novelty-request.json').read_text())['request']
files={'input.json':request['files']['input.json'],'old-archive.csv':base64.b64encode((root/'archive.csv').read_bytes()).decode(),'new-archive.csv':request['files']['archive.csv'],'worker.py':WORKER}
results={}
for host,address,port,alias in [('master','211.72.13.201','40096','[ssh9.vast.ai]:10616'),('slave','ssh2.vast.ai','28358','[ssh2.vast.ai]:28358')]:
 payload=gzip.compress(json.dumps({'identity':IDENTITIES[host],'files':files}).encode(),compresslevel=6)
 try:
  p=subprocess.run(['ssh','-C','-i','/home/thomasg/.ssh/id_ed25519','-oBatchMode=yes','-oConnectTimeout=8','-oStrictHostKeyChecking=yes','-oHostKeyAlias='+alias,'-p',port,'root@'+address,shlex.join(['python3','-B','-c',REMOTE])],input=payload,capture_output=True,timeout=45)
  results[host]=json.loads(p.stdout) if p.returncode==0 else {'exit_code':p.returncode}
 except Exception as e:results[host]={'error_type':type(e).__name__}
print(json.dumps(results))
'''
script='IDENTITIES='+repr(identities)+'\nWORKER='+repr(base64.b64encode(worker.encode()).decode())+'\nREMOTE='+repr(remote)+'\n'+pi
p=subprocess.run(['ssh','-C','-oBatchMode=yes','-oConnectTimeout=8','thomasg@100.64.243.65','python3','-'],input=script,text=True,capture_output=True,timeout=110)
try:v=json.loads(p.stdout)
except Exception:v={'exit_code':p.returncode,'parse_error':True}
(report/'qualified-replay-launch.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
