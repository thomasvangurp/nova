import pathlib,subprocess,json,base64
build=pathlib.Path('/tmp/pr100-build-novelty-assets-20261027.py').read_text()
pi=r'''import pathlib,subprocess,json,base64,shlex,hashlib
hosts={'master':('211.72.13.201','40096','[ssh9.vast.ai]:10616'),'slave':('122.51.254.66','4116','[ssh2.vast.ai]:28358')}
def ssh(host,code,data=None,python='python3',timeout=150):
 address,port,alias=hosts[host]
 return subprocess.run(['ssh','-i','/home/thomasg/.ssh/id_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-p',port,'root@'+address,shlex.join([python,'-B','-c',code])],input=data,capture_output=True,timeout=timeout)
r=ssh('slave',base64.b64decode(BUILD).decode(),python='/root/pr100-qualification-20261026/validator-annotation-runtime/bin/python')
if r.returncode:
 print(json.dumps({'stage':'build','exit_code':r.returncode,'error_types':[x for x in ['PackageNotFoundError','AssertionError','FileNotFoundError'] if x.encode() in r.stderr]}));raise SystemExit(0)
receipt=json.loads(r.stdout)
r=ssh('slave',"import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/root/pr100-novelty-runtime-staging-20261027/annotation-assets.tar.gz').read_bytes())",timeout=180)
assert r.returncode==0 and hashlib.sha256(r.stdout).hexdigest()==receipt['archive_sha256'];archive=r.stdout
root=pathlib.Path('/home/thomasg/pr100-novelty-runtime-20261027');root.mkdir(exist_ok=True);(root/'annotation-assets.tar.gz').write_bytes(archive)
install=r"""import sys,pathlib,subprocess,json,tarfile,io,hashlib
prefix=pathlib.Path('/root/pr100-novelty-runtime-20261027');raw=sys.stdin.buffer.read();expected=EXPECTED
assert hashlib.sha256(raw).hexdigest()==expected
if not (prefix/'bin/python').exists():
 p=subprocess.run([BASE,'-m','venv','--copies',str(prefix)],capture_output=True,timeout=60);assert p.returncode==0
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 manifest=json.load(tar.extractfile('assets-manifest.json'))
 for item in tar.getmembers():
  target=pathlib.Path(item.name);assert not target.is_absolute() and '..' not in target.parts and item.isfile()
  data=tar.extractfile(item).read()
  if item.name!='assets-manifest.json':assert hashlib.sha256(data).hexdigest()==manifest['files'][item.name]['sha256']
  p=prefix/target;p.parent.mkdir(parents=True,exist_ok=True);assert not p.is_symlink();p.write_bytes(data);p.chmod(item.mode)
for name,v in manifest['files'].items():assert hashlib.sha256((prefix/name).read_bytes()).hexdigest()==v['sha256']
print(json.dumps({'status':'installed','files':len(manifest['files']),'manifest_sha256':hashlib.sha256((prefix/'assets-manifest.json').read_bytes()).hexdigest()}))
"""
results={}
for host in ['slave','master']:
 base='/root/pr100-qualification-20261026/validator-annotation-runtime/bin/python' if host=='slave' else '/root/.pixi/envs/metanano/bin/python3.12'
 r=ssh(host,'EXPECTED='+repr(receipt['archive_sha256'])+'\nBASE='+repr(base)+'\n'+install,archive,timeout=180)
 results[host]=json.loads(r.stdout) if r.returncode==0 else {'exit_code':r.returncode,'error_types':[x for x in ['AssertionError','FileNotFoundError','TimeoutExpired'] if x.encode() in r.stderr]}
out={'build':receipt,'hosts':results};(root/'provision-receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','thomasg@100.64.243.65','python3','-'],input='BUILD='+repr(base64.b64encode(build.encode()).decode())+'\n'+pi,text=True,capture_output=True,timeout=590)
try:v=json.loads(p.stdout)
except Exception:v={'exit_code':p.returncode,'parse_error':True,'stderr_present':bool(p.stderr)}
out=pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027');out.mkdir(exist_ok=True);(out/'provision-receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
