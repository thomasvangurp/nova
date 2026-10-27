import pathlib,json,base64,sys,importlib,concurrent.futures
sys.path.insert(0,'/tmp');remote=importlib.import_module('pr100-novelty-remote')
local=pathlib.Path('/tmp/sn68-pr100-novelty-runtime-20261027')
files={name:base64.b64encode((local/name).read_bytes()).decode() for name in ['nanobody_opt/pr100_novelty_runtime.py','nanobody_opt/check_validator_novelty.py']}
code=r'''import pathlib,shutil,json,base64,hashlib,subprocess,os
prefix=pathlib.Path('/root/pr100-novelty-runtime-20261027');source=prefix/'qualification-source';source.mkdir(exist_ok=True)
release=pathlib.Path('/root/releases/22407a4ae6324b8c051ef89f738eb74fddaad978');stamp=json.loads((release/'build_stamp.json').read_text());assert stamp['git_sha']=='22407a4ae6324b8c051ef89f738eb74fddaad978'
for name,digest in stamp['files'].items():
 if not (name.startswith(('nanobody_opt/','utils/','NOVA-nanobody-filter/metanano/')) or name=='config/nanobody_pr100.json'):continue
 raw=(release/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
for name,encoded in FILES.items():(source/name).write_bytes(base64.b64decode(encoded))
probe="""import json,contextlib,io
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  from nanobody_opt import pr100_novelty_runtime as n
  from nanobody_opt import rules
  identity=n.runtime_identity()
 result={'status':'ready','identity_sha256':rules.digest(identity),'identity':identity}
except Exception as e:result={'status':'error','error_type':type(e).__name__,'module':getattr(e,'name',None)}
print(json.dumps(result))
"""
env={'PATH':str(prefix/'bin')+':/usr/local/bin:/usr/bin:/bin','PYTHONPATH':str(source)+':'+str(prefix/'shim'),'HOME':'/root','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','ROCR_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
p=subprocess.run([str(prefix/'bin/novelty-python'),'-B','-c',probe],cwd=source,env=env,capture_output=True,text=True,timeout=45)
try:result=json.loads(p.stdout)
except Exception:result={'status':'probe_failed','exit_code':p.returncode}
print(json.dumps({'source':str(source),'result':result}))
'''
results={}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 fs={pool.submit(remote.run,h,'FILES='+repr(files)+'\n'+code,timeout=90):h for h in ['master','slave']}
 for f in concurrent.futures.as_completed(fs):results[fs[f]]=f.result();print(json.dumps({fs[f]:results[fs[f]]}),flush=True)
pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027/qualification-preparation.json').write_text(json.dumps(results,indent=2)+'\n')
