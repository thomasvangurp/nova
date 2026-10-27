import json,pathlib,subprocess,shlex,sys
names=sys.argv[1:]
master=r'''import hashlib,json,os,pathlib,re,shlex,subprocess,sys,time
names=json.loads(sys.stdin.read());prefix=pathlib.Path('/root/pr100-native-alt-20261027');base=pathlib.Path('/root/pr100-native-alt-staging-20261027');mp=base/'mkl-supplement-e.json'
assert names and all(re.fullmatch(r'libmkl_[A-Za-z0-9_]+[.]so[.]2',n) for n in names)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
opts=['-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8']
def ssh(code,data=None):
 p=subprocess.run(['ssh','-T',*opts,'-p','4116','root@122.51.254.66',shlex.join(['python3','-c',code])],input=None if data is None else json.dumps(data).encode(),capture_output=True,timeout=30)
 if p.returncode:raise RuntimeError('slave_verify_failed')
 return json.loads(p.stdout) if p.stdout.strip() else {}
manifest=json.loads(mp.read_text()) if mp.exists() else {'schema':'pr100-exact-cpu-library-supplement-v1','files':{}}
for name in names:
 source=pathlib.Path('/root/.pixi/envs/metanano/lib')/name;assert source.is_file() and not source.is_symlink();target=prefix/'lib'/name;digest=sha(source);stage=base/('mkl-e-'+name)
 p=subprocess.run(['scp','-q','-C',*opts,'-P','4116',str(source),'root@122.51.254.66:'+str(stage)],capture_output=True,timeout=240)
 if p.returncode:raise RuntimeError('exact_library_copy_failed')
 code='import hashlib,json,os,pathlib,sys; d=json.load(sys.stdin); s=pathlib.Path(d["stage"]); t=pathlib.Path(d["target"]); assert hashlib.sha256(s.read_bytes()).hexdigest()==d["sha256"]; assert not t.exists() or hashlib.sha256(t.read_bytes()).hexdigest()==d["sha256"]; os.replace(s,t); print(json.dumps({"verified":True}))'
 ssh(code,{'stage':str(stage),'target':str(target),'sha256':digest})
 manifest['files'][str(source)]={'target':str(target),'bytes':source.stat().st_size,'sha256':digest}
raw=json.dumps(manifest,sort_keys=True,indent=2)+'\n';mp.write_text(raw);digest=hashlib.sha256(raw.encode()).hexdigest()
ssh('import hashlib,json,pathlib,sys; d=json.load(sys.stdin); p=pathlib.Path("/root/pr100-native-alt-staging-20261027/mkl-supplement-e.json"); raw=d["text"].encode(); assert hashlib.sha256(raw).hexdigest()==d["sha256"]; p.write_bytes(raw)',{'text':raw,'sha256':digest})
print(json.dumps({'status':'verified','manifest_sha256':digest,'libraries':{pathlib.Path(k).name:v for k,v in manifest['files'].items()}}))
'''
pi="import subprocess,json,shlex,pathlib\ncode="+repr(master)+"\np=subprocess.run(['ssh','-T','-i','/home/thomasg/.ssh/id_ed25519','-oIdentitiesOnly=yes','-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh9.vast.ai]:10616','-oConnectTimeout=8','-p','40096','root@211.72.13.201',shlex.join(['python3','-c',code])],input="+repr(json.dumps(names).encode())+",capture_output=True,timeout=290)\ntry:d=json.loads(p.stdout) if p.returncode==0 else None\nexcept Exception:d=None\nif d and d.get('status')=='verified':pathlib.Path('/home/thomasg/pr100-native-alt-20261027/mkl-supplement-receipt-e.json').write_text(json.dumps(d,indent=2)+'\\n')\nprint(json.dumps({'master_exit':p.returncode,'result':d,'error_categories':[k for k in ['TimeoutExpired','exact_library_copy_failed','slave_verify_failed','AssertionError','FileNotFoundError'] if k.encode() in p.stderr]}))\n"
p=subprocess.run(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10','thomasg@100.64.243.65','python3','-'],input=pi,text=True,capture_output=True,timeout=310)
try:d=json.loads(p.stdout) if p.returncode==0 else None
except Exception:d=None
out={'pi_exit':p.returncode,'result':d};pathlib.Path('/Users/tvgdeena/nova-reports/pr100-slave-native-alt-20261027/recovery-e/mkl-copy-'+names[-1]+'.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
