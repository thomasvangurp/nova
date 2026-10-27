import hashlib,json,os,pathlib,shlex,subprocess,time
base=pathlib.Path('/home/thomasg/pr100-native-alt-20261027');receipt=base/'qualification-status-e.json';remote='/root/pr100-profile/slave-native-alt-qualification-20261027e';started=time.time()
opts=['-i','/home/thomasg/.ssh/id_ed25519','-oIdentitiesOnly=yes','-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8']
def call(code,data=None,timeout=25):
 p=subprocess.run(['ssh','-T',*opts,'-p','4116','root@122.51.254.66',shlex.join(['python3','-c',code])],input=None if data is None else json.dumps(data).encode(),capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError('bounded_slave_transport')
 return json.loads(p.stdout) if p.stdout.strip() else {}
def save(status,**fields):
 p=receipt.with_suffix('.next');p.write_text(json.dumps({'status':status,'updated_at_unix':time.time(),'elapsed_seconds':time.time()-started,**fields}));os.replace(p,receipt)
try:
 save('waiting_for_verified_runtime')
 config={'source_sha':'51688c0a458f2aa401d7b1f569b2cf364f96aebc','release':'/root/releases/51688c0a458f2aa401d7b1f569b2cf364f96aebc','manifest':'/root/pr100-native-alt-staging-20261027/manifest.json','manifest_sha256':'637d20c23d4a8c21f9d09331afbc543b997017e6e361c6fd5a4c0031235eaa58','reference_manifest':'/root/pr100-native-runtime-20261027/reviewed-runtime-allowlist.json','unpack_receipt':'/root/pr100-native-alt-staging-20261027/unpack-receipt.json'}
 config['supplemental_manifest']='/root/pr100-native-alt-staging-20261027/mkl-supplement-family-e.json'
 config['supplemental_manifest_sha256']=json.loads((base/'mkl-family-receipt-e.json').read_text())['manifest_sha256']
 config['metadata_sha256']=json.loads((base/'metadata-receipt.json').read_text())['sha256']
 source=pathlib.Path('/home/thomasg/pr100-controller-state/25964/attempts/00001/native-input.json')
 if not source.exists():source=source.parent/'sequence/result.json'
 value=json.loads(source.read_text());rows=value['candidates'][:32];assert len(rows)==32
 incoming={'challenge':value['challenge'],'candidates':[{'sequence':r['sequence'],'sequence_hash':hashlib.sha256(r['sequence'].encode()).hexdigest()} for r in rows]}
 files={'qualification-worker.py':(base/'qualification-worker-e.py').read_text(),'qualification-guard.py':(base/'qualification-guard-e.py').read_text(),'qualification-config.json':json.dumps(config),'input.json':json.dumps(incoming)}
 deadline=time.monotonic()+1000
 while time.monotonic()<deadline:
  try:
   state=call('import json,pathlib; p=pathlib.Path('+repr(config['unpack_receipt'])+'); d=json.loads(p.read_text()) if p.exists() else {}; print(json.dumps({k:d[k] for k in ("status","files","bytes","error_type","weights") if k in d}))')
   save('waiting_for_verified_runtime',runtime_status=state)
   if state.get('status')=='failed':raise ValueError('runtime_provision_failed')
   if state.get('status')=='files_verified':break
  except (RuntimeError,subprocess.TimeoutExpired):pass
  time.sleep(20)
 else:raise TimeoutError('runtime_ready_deadline')
 stage='import json,pathlib,sys; b=pathlib.Path('+repr(remote)+'); b.mkdir(parents=True,exist_ok=True); d=json.load(sys.stdin); assert not (b/"qualification-receipt.json").exists(); [(b/n).write_text(t) for n,t in d.items()]'
 call(stage,files,timeout=50)
 launch='import json,pathlib,subprocess; b=pathlib.Path('+repr(remote)+'); p=subprocess.Popen(["python3","-B",str(b/"qualification-guard.py"),str(b)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); (b/"guard.pid").write_text(str(p.pid)); print(json.dumps({"pid":p.pid}))'
 save('qualification_running',**call(launch));end=time.monotonic()+380
 while time.monotonic()<end:
  try:
   result=call('import json,pathlib; p=pathlib.Path('+repr(remote+'/qualification-receipt.json')+'); print(p.read_text() if p.exists() else "{}")')
   if result:
    (base/'qualification-receipt-e.json').write_text(json.dumps(result,indent=2));save(result.get('status','not_qualified'),completed_profiles=result.get('completed_profiles',0),runtime_identity_sha256=result.get('runtime_identity_sha256'),cleanup=result.get('cleanup'),error_type=result.get('error_type'));break
  except (RuntimeError,subprocess.TimeoutExpired):pass
  time.sleep(10)
 else:raise TimeoutError('qualification_receipt_deadline')
except BaseException as error:save('failed',error_type=type(error).__name__)
