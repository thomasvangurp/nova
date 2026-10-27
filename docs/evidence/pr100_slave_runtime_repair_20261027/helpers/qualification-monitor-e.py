import subprocess,json,shlex,pathlib,collections
remote='''import pathlib,json,collections
root=pathlib.Path('/root/pr100-profile/slave-native-alt-qualification-20261027e');out={}
for name,allowed in [('qualification-receipt.json',['schema','status','host','workers','authoritative','publication_enabled','application_source_sha','hard_cap_seconds','started_at_unix','release_manifest_files','affinity','cpu_affinity_verified','runtime','returncode','worker_output_bytes','worker_error_categories','runtime_identity','runtime_identity_sha256','reference_runtime_identity','runtime_differences','runtime_manifest_verified','completed_profiles','runtime_inconclusive_count','native_wall_seconds','python_version','openmm_platforms','error_type','missing_distribution','stage','cleanup','wall_seconds']),('measurements.json',['status','authoritative','publication_enabled','application_source_sha','stage','runtime_manifest_verified','runtime_identity','runtime_identity_sha256','reference_runtime_identity','runtime_differences','python_version','openmm_platforms','memory_available_bytes','challenge_sha256','cohort_hashes','native_wall_seconds','completed_profiles','runtime_inconclusive_count','sources_unchanged','error_type','missing_distribution','wall_seconds'])]:
 p=root/name
 if p.exists():
  d=json.loads(p.read_text());out[name]={k:d[k] for k in allowed if k in d}
  if name=='measurements.json':out[name]['records']=[{k:r[k] for k in ['sequence_hash','length','numeric_metrics_complete','provenance_complete','runtime_complete','observation'] if k in r} for r in d.get('records',[])]
print(json.dumps(out))
'''
pi="import subprocess,json,shlex,pathlib\ncode="+repr(remote)+"\np=subprocess.run(['ssh','-T','-i','/home/thomasg/.ssh/id_ed25519','-oIdentitiesOnly=yes','-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8','-p','4116','root@122.51.254.66',shlex.join(['python3','-c',code])],capture_output=True,timeout=30)\ntry:d=json.loads(p.stdout) if p.returncode==0 else None\nexcept Exception:d=None\ns=pathlib.Path('/home/thomasg/pr100-native-alt-20261027/qualification-status-e.json');v=json.loads(s.read_text()) if s.exists() else {};safestatus={k:v[k] for k in ['status','updated_at_unix','elapsed_seconds','completed_profiles','runtime_identity_sha256','cleanup','error_type'] if k in v}\nprint(json.dumps({'slave_exit':p.returncode,'reports':d,'pi_status':safestatus}))\n"
p=subprocess.run(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10','thomasg@100.64.243.65','python3','-'],input=pi,text=True,capture_output=True,timeout=45)
try:d=json.loads(p.stdout) if p.returncode==0 else None
except Exception:d=None
out={'pi_exit':p.returncode,'result':d};root=pathlib.Path('/Users/tvgdeena/nova-reports/pr100-slave-native-alt-20261027/recovery-e')
if d:
 for name,v in (d.get('reports') or {}).items():
  if name in ['qualification-receipt.json','measurements.json']:(root/name).write_text(json.dumps(v,indent=2)+'\n')
 summary={'pi_status':d.get('pi_status'),'slave_exit':d.get('slave_exit')}
 for name,v in (d.get('reports') or {}).items():
  summary[name]={k:v[k] for k in ['status','stage','completed_profiles','runtime_inconclusive_count','runtime_identity_sha256','error_type','missing_distribution','native_wall_seconds','wall_seconds','cleanup','worker_error_categories','returncode','sources_unchanged'] if k in v}
  if name=='measurements.json':summary[name]['record_count']=len(v.get('records',[]));summary[name]['numeric_complete_count']=sum(r.get('numeric_metrics_complete') is True for r in v.get('records',[]));summary[name]['provenance_complete_count']=sum(r.get('provenance_complete') is True for r in v.get('records',[]));summary[name]['outcome_counts']=dict(collections.Counter(r.get('observation',{}).get('outcome') for r in v.get('records',[])))
 print(json.dumps({'pi_exit':p.returncode,'result':summary}))
else:print(json.dumps({'pi_exit':p.returncode,'parsed':False}))
