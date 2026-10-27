import pathlib,ast,subprocess,json
text=pathlib.Path('/tmp/pr100-provision-novelty-runtime-20261027.py').read_text();tree=ast.parse(text);pi=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='pi' for x in n.targets)));tree=ast.parse(pi);install=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='install' for x in n.targets))).replace('assets-manifest.json','hmmer-assets-manifest.json')
pi=r'''import subprocess,json,shlex,hashlib,time,pathlib
hosts={'master':('211.72.13.201','40096','[ssh9.vast.ai]:10616'),'slave':('ssh2.vast.ai','28358','[ssh2.vast.ai]:28358')}
def call(host,code,data=None):
 address,port,alias=hosts[host]
 for attempt in range(3):
  p=subprocess.run(['ssh','-C','-i','/home/thomasg/.ssh/id_ed25519','-oBatchMode=yes','-oConnectTimeout=8','-oStrictHostKeyChecking=yes','-oHostKeyAlias='+alias,'-p',port,'root@'+address,shlex.join(['python3','-B','-c',code])],input=data,capture_output=True,timeout=50)
  if p.returncode!=255:return p
  time.sleep(1)
 return p
p=call('slave',"import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('/root/pr100-novelty-runtime-staging-20261027/hmmer-assets.tar.gz').read_bytes())")
assert p.returncode==0;raw=p.stdout;assert hashlib.sha256(raw).hexdigest()=='0278416e53b0a632b3eca2c58bbf800b49b03b9c97ca0288ef9b38d0a9391561'
root=pathlib.Path('/home/thomasg/pr100-novelty-runtime-20261027');root.mkdir(exist_ok=True);(root/'hmmer-assets.tar.gz').write_bytes(raw)
results={}
for host in ['master','slave']:
 code="EXPECTED='0278416e53b0a632b3eca2c58bbf800b49b03b9c97ca0288ef9b38d0a9391561'\nBASE='/root/.pixi/envs/metanano/bin/python3.12'\n"+INSTALL
 p=call(host,code,raw);results[host]=json.loads(p.stdout) if p.returncode==0 else {'exit_code':p.returncode}
(root/'hmmer-copy-receipt.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
'''
p=subprocess.run(['ssh','-C','-oBatchMode=yes','-oConnectTimeout=8','thomasg@100.64.243.65','python3','-'],input='INSTALL='+repr(install)+'\n'+pi,text=True,capture_output=True,timeout=210)
try:v=json.loads(p.stdout)
except Exception:v={'exit_code':p.returncode,'parse_error':True,'error_categories':[x for x in ['TimeoutExpired','AssertionError'] if x in p.stderr]}
pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027/hmmer-copy-receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
