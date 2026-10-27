import pathlib,sys,importlib,json,concurrent.futures
sys.path.insert(0,'/tmp');remote=importlib.import_module('pr100-novelty-remote')
code="""import pathlib,json
base=pathlib.Path('/root/pr100-novelty-runtime-20261027/replay-25967-v2');out={}
for name in ['result.json','supervisor-receipt.json']:
 p=base/name
 if not p.exists():out[name]={'present':False};continue
 v=json.loads(p.read_text())
 if name=='result.json':out[name]=v
 else:out[name]={k:v.get(k) for k in ['reason','exit_code','elapsed_seconds','cleanup','error_type']}
print(json.dumps(out))
"""
results={}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 fs={pool.submit(remote.run,h,code,timeout=30):h for h in ['master','slave']}
 for f in concurrent.futures.as_completed(fs):
  host=fs[f];results[host]=f.result();value=results[host].get('value',{});r=value.get('result.json',{});print(json.dumps({'host':host,'transport_exit':results[host].get('exit_code'),'result':{k:r.get(k) for k in ['status','error_type','runtime_sha256','input_count']},'snapshots':{k:{key:v.get(key) for key in ['archive_rows','archive_cdr_available','kept_count','elapsed_seconds']} for k,v in r.get('snapshots',{}).items()},'supervisor':value.get('supervisor-receipt.json')}),flush=True)
pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027/qualified-replay-receipts.json').write_text(json.dumps(results,indent=2)+'\n')
