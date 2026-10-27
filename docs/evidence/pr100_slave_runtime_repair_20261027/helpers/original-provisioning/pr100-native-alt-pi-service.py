import hashlib,json,os,pathlib,shlex,subprocess,time
base=pathlib.Path('/home/thomasg/pr100-native-alt-20261027');remote='/root/pr100-native-alt-staging-20261027';receipt=base/'provision-receipt.json';start=time.time()
opts=['-i','/home/thomasg/.ssh/id_ed25519','-oIdentitiesOnly=yes','-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh9.vast.ai]:10616','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=2']
def save(status,**fields):
 p=receipt.with_suffix('.next');p.write_text(json.dumps({'status':status,'updated_at_unix':time.time(),'elapsed_seconds':time.time()-start,**fields}));os.replace(p,receipt)
def ssh(command):return ['ssh','-T',*opts,'-p','40096','root@211.72.13.201',shlex.join(command)]
try:
 save('staging_helpers')
 files={n:(base/n).read_text() for n in ('manifest.py','pack.py','unpack.py','runner.py')}
 code='import json,pathlib,sys; b=pathlib.Path('+repr(remote)+'); b.mkdir(parents=True,exist_ok=True); d=json.load(sys.stdin); [(b/n).write_text(t) for n,t in d.items()]'
 p=subprocess.run(ssh(['python3','-c',code]),input=json.dumps(files).encode(),capture_output=True,timeout=50)
 if p.returncode:raise RuntimeError('helper_stage_failed')
 expected={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()}
 verify='import hashlib,pathlib; b=pathlib.Path('+repr(remote)+'); e='+repr(expected)+'; assert all(hashlib.sha256((b/n).read_bytes()).hexdigest()==h for n,h in e.items())'
 p=subprocess.run(ssh(['python3','-c',verify]),capture_output=True,timeout=20);assert p.returncode==0
 save('provisioning')
 p=subprocess.run(ssh(['timeout','--kill-after=8','1400','python3',remote+'/runner.py']),capture_output=True,timeout=1430)
 if p.returncode:raise RuntimeError('master_orchestration_failed')
 check='import json,pathlib; d=json.loads(pathlib.Path('+repr(remote+'/direct-receipt.json')+').read_text()); print(json.dumps({k:d[k] for k in ("status","manifest_sha256","files","error_type","elapsed_seconds") if k in d}))'
 p=subprocess.run(ssh(['python3','-c',check]),capture_output=True,timeout=20)
 result=json.loads(p.stdout)
 save('files_verified' if result['status']=='files_verified' else 'failed',remote=result)
except BaseException as error:save('failed',error_type=type(error).__name__)
