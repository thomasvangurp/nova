import concurrent.futures,hashlib,json,os,pathlib,shlex,subprocess,time
os.nice(10)
base=pathlib.Path('/root/pr100-native-alt-staging-20261027');source=pathlib.Path('/root/.pixi/envs/metanano/lib');prefix=pathlib.Path('/root/pr100-native-alt-20261027');report=base/'mkl-family-progress-e.json'
names=['libmkl_'+n+'.so.2' for n in ['core','def','mc3','avx2','avx512','intel_lp64','intel_ilp64','intel_thread','gnu_thread','sequential','vml_avx2','vml_avx512','vml_cmpt','vml_def','vml_mc3']]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
opts=['-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=2']
def call(code,data=None,timeout=45):
 p=subprocess.run(['ssh','-T',*opts,'-p','4116','root@122.51.254.66',shlex.join(['python3','-c',code])],input=None if data is None else json.dumps(data).encode(),capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError('slave_operation_failed')
 return json.loads(p.stdout) if p.stdout.strip() else {}
entries={str(source/n):{'target':str(prefix/'lib'/n),'bytes':(source/n).stat().st_size,'sha256':sha(source/n)} for n in names}
inspect=r'''import hashlib,json,pathlib,shutil,sys,os
entries=json.load(sys.stdin);out={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
for source,e in entries.items():
 name=pathlib.Path(source).name;t=pathlib.Path(e['target']);status='needs_copy'
 if t.exists():
  status='verified_existing' if sha(t)==e['sha256'] else 'conflict'
 else:
  for directory in ['/root/bgen/lib','/root/bgen/lib/python3.12/site-packages/torch/lib','/root/.pixi/envs/metanano/lib']:
   p=pathlib.Path(directory)/name
   if p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256']:
    scratch=t.with_name(t.name+'.reuse-e');shutil.copy2(p,scratch);os.replace(scratch,t);status='reused_exact_local';break
 out[name]=status
print(json.dumps(out))'''
started=time.time();inventory=call(inspect,entries,timeout=90);assert 'conflict' not in inventory.values()
report.write_text(json.dumps({'status':'copying','started_at_unix':started,'inventory':inventory}))
def one(item):
 src,e=item;name=pathlib.Path(src).name
 if inventory[name]!='needs_copy':return name,inventory[name]
 if name=='libmkl_core.so.2':
  end=time.monotonic()+320
  while time.monotonic()<end:
   result=call(inspect,{src:e},timeout=30)
   if result[name]=='verified_existing':return name,'verified_parallel_core_copy'
   if result[name]=='conflict':raise ValueError('core_hash_conflict')
   time.sleep(10)
  raise TimeoutError('parallel_core_copy_deadline')
 stage=base/('mkl-family-e-'+name)
 p=subprocess.run(['scp','-q','-C',*opts,'-P','4116',src,'root@122.51.254.66:'+str(stage)],capture_output=True,timeout=360)
 if p.returncode:raise RuntimeError('exact_library_copy_failed')
 verify='import hashlib,json,os,pathlib,sys; d=json.load(sys.stdin); s=pathlib.Path(d["stage"]); t=pathlib.Path(d["target"]); assert hashlib.sha256(s.read_bytes()).hexdigest()==d["sha256"]; assert not t.exists() or hashlib.sha256(t.read_bytes()).hexdigest()==d["sha256"]; os.replace(s,t); print(json.dumps({"verified":True}))'
 call(verify,{'stage':str(stage),**e});return name,'copied_exact_master'
statuses={};errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures={pool.submit(one,item):item[0] for item in entries.items()}
 for f in concurrent.futures.as_completed(futures):
  try:name,status=f.result();statuses[name]=status
  except BaseException as error:errors.append({'library':pathlib.Path(futures[f]).name,'error_type':type(error).__name__})
  report.write_text(json.dumps({'status':'copying','started_at_unix':started,'elapsed_seconds':time.time()-started,'inventory':inventory,'verified':statuses,'errors':errors}))
if errors:raise RuntimeError('family_copy_incomplete')
verified=call(inspect,entries,timeout=90);assert set(verified.values())=={'verified_existing'}
manifest={'schema':'pr100-exact-cpu-library-supplement-v1','purpose':'Exact master MKL CPU dynamic-dispatch closure; no distributed or GPU libraries','files':entries};raw=json.dumps(manifest,sort_keys=True,indent=2)+'\n';digest=hashlib.sha256(raw.encode()).hexdigest();mp=base/'mkl-supplement-family-e.json';mp.write_text(raw)
call('import hashlib,json,pathlib,sys; d=json.load(sys.stdin); p=pathlib.Path("/root/pr100-native-alt-staging-20261027/mkl-supplement-family-e.json"); raw=d["text"].encode(); assert hashlib.sha256(raw).hexdigest()==d["sha256"]; p.write_bytes(raw)',{'text':raw,'sha256':digest})
out={'status':'verified','manifest_sha256':digest,'manifest_path':str(mp),'files':len(entries),'bytes':sum(e['bytes'] for e in entries.values()),'elapsed_seconds':time.time()-started,'file_statuses':statuses,'library_hashes':{pathlib.Path(k).name:e['sha256'] for k,e in entries.items()}};report.write_text(json.dumps(out));print(json.dumps(out))
