"""Original CPU profiles in an explicitly different isolated runtime."""
import hashlib,importlib.metadata,json,math,os,re,sys,time
from pathlib import Path
from decimal import Decimal
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(root):
 config=json.loads((root/'qualification-config.json').read_text());started=time.monotonic()
 output={'status':'starting','authoritative':False,'publication_enabled':False,'application_source_sha':config['source_sha'],'records':[]}
 def save():
  p=root/'measurements.next';p.write_text(json.dumps(output,allow_nan=False));os.replace(p,root/'measurements.json')
 output['stage']='startup';save()
 try:
  assert set(os.sched_getaffinity(0))==set(range(32))
  assert all(os.environ.get(k)=='' for k in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
  prefix=Path('/root/pr100-native-alt-20261027');toolsdir=prefix/'bin'
  assert os.environ.get('OPENMM_PLUGIN_DIR')==str(prefix/'lib/plugins') and os.environ.get('LD_LIBRARY_PATH')==str(prefix/'lib')
  sys.path.insert(0,config['release'])
  output['stage']='application_imports';save()
  from nanobody_opt import pr100_live_worker as worker,pr100_filter,rules,pr100_native_journal as journal
  from nanobody_opt.pr100_resources import available_memory
  output['stage']='manifest_verification';save()
  manifest_path=Path(config['manifest']);manifest=json.loads(manifest_path.read_text());manifest_hash=digest(manifest_path)
  assert manifest_hash==config['manifest_sha256']
  reference=json.loads(Path(config['reference_manifest']).read_text());known={**reference['files'],**reference['external_scientific_assets']};verified={}
  for source,entry in manifest['files'].items():
   p=Path(entry['target']);h=digest(p)
   if p.parent==toolsdir and p.name in ('TNP','NanoBodyBuilder2'):
    first,_,rest=p.read_bytes().partition(b'\n');assert first==b'#!/root/pr100-native-alt-20261027/bin/python3.12'
    assert any(hashlib.sha256(('#!/root/.pixi/envs/metanano/bin/'+n+'\n').encode()+rest).hexdigest()==entry['sha256'] for n in ('python','python3','python3.12'))
   else:assert h==entry['sha256']
   verified[entry['target']]=h
  for name,dest in manifest['aliases'].items():assert Path(name).is_symlink() and Path(name).resolve()==Path(dest).resolve()
  weights={k:digest(v['target']) for k,v in manifest['weights'].items()};assert weights=={k:v['sha256'] for k,v in manifest['weights'].items()}
  metadata_path=Path('/root/pr100-native-alt-staging-20261027/metadata-supplement.json')
  metadata_hash=digest(metadata_path);assert metadata_hash==config['metadata_sha256']
  metadata=json.loads(metadata_path.read_text());assert metadata and all(digest(p)==v['sha256'] for p,v in metadata.items())
  supplemental_path=Path(config['supplemental_manifest']);supplemental_hash=digest(supplemental_path)
  assert supplemental_hash==config['supplemental_manifest_sha256']
  supplemental=json.loads(supplemental_path.read_text());supplemental_verified={}
  assert supplemental['schema']=='pr100-exact-cpu-library-supplement-v1' and supplemental['files']
  for source,entry in supplemental['files'].items():
   p=Path(entry['target']);assert p.parent==prefix/'lib' and p.name.startswith('libmkl_')
   assert digest(p)==entry['sha256'];supplemental_verified[str(p)]=entry['sha256']
  output['runtime_manifest_verified']=True
  output['stage']='package_versions';save()
  versions={}
  for n in ('torch','numpy','openmm','ImmuneBuilder','theraprofnano','biopython','abnumber','pdbfixer'):
   try:versions[n]=importlib.metadata.version(n)
   except importlib.metadata.PackageNotFoundError:output['missing_distribution']=n;raise
  assert versions=={'torch':'2.7.1+cu126','numpy':'1.26.4','openmm':'8.5.1','ImmuneBuilder':'1.2','theraprofnano':'0.0.1','biopython':'1.86','abnumber':'0.2.7','pdbfixer':'1.12.0'}
  output['stage']='torch_import';save()
  import torch
  output['stage']='openmm_import';save()
  import openmm
  assert not torch.cuda.is_available()
  platforms=[openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())];assert 'CPU' in platforms
  old='/root/.pixi/envs/metanano/lib/python3.12/site-packages/'
  sources={'builder':old+'ImmuneBuilder/NanoBodyBuilder2.py','refine':old+'ImmuneBuilder/refine.py','cdr':old+'theraprofnano/CDR_Profiler/CDR3_Conf_Assigner.py','surface':old+'theraprofnano/Hydrophobicity_and_Charge_Profiler/Hydrophobicity_and_Charge_Assigner.py'}
  tools={n:digest(toolsdir/n) for n in ('TNP','NanoBodyBuilder2','hmmscan','psa','mkdssp','aggrescan3d-rs','freesasa','native-python')}
  tools['native_python_wrapper']=tools.pop('native-python')
  reference_tools={n:known['/root/.pixi/envs/metanano/bin/'+n] for n in ('TNP','NanoBodyBuilder2','hmmscan','psa','mkdssp')}
  reference_tools.update({n:known['/root/pr100-qualification-20261026/NOVA-nanobody-filter/.tools/bin/'+n] for n in ('aggrescan3d-rs','freesasa')})
  actual={'manifest_sha256':manifest_hash,'scientific_sources':{k:verified[manifest['files'][v]['target']] for k,v in sources.items()},'model_weights':weights,'tools':tools,'python_binary_sha256':digest(sys.executable),'versions':versions,'torch_libraries':{n:digest(Path(torch.__file__).parent/'lib'/n) for n in ('libtorch_cpu.so','libtorch_python.so','libc10.so')}}
  actual['metadata_manifest_sha256']=metadata_hash
  actual['supplemental_manifest_sha256']=supplemental_hash
  actual['supplemental_cpu_libraries']=supplemental_verified
  output.update(runtime_identity=actual,runtime_identity_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest(),reference_runtime_identity={'manifest_sha256':manifest['reference_manifest_sha256'],'scientific_sources':{k:known[v] for k,v in sources.items()},'model_weights':weights,'tools':reference_tools,'metadata_manifest_sha256':metadata_hash,'supplemental_cpu_libraries':supplemental_verified},runtime_differences=manifest['runtime_differences']+['Native wrapper pins alternate CPU plugin and shared-library directories','MKL CPU libraries are copied byte-for-byte from the master and bound by a supplemental manifest'],python_version=list(sys.version_info[:3]),openmm_platforms=platforms)
  memory=available_memory();assert memory['cgroup_resolution']=='resolved_visible_ancestors' and memory['available_bytes']>=53*1024**3
  output['memory_available_bytes']=memory['available_bytes'];incoming=json.loads((root/'input.json').read_text());rows=incoming['candidates']
  assert len(rows)==32 and len({r['sequence'] for r in rows})==32 and all(rules.sequence_hash(r['sequence'])==r['sequence_hash'] for r in rows)
  requirements=rules.pinned_config(incoming['challenge']['epoch'])['nanobody_requirements']
  pr100_filter.helper('aggrescan3d').tool_paths=lambda:(toolsdir/'aggrescan3d-rs',toolsdir/'freesasa')
  output.update(challenge_sha256=rules.digest(incoming['challenge']),cohort_hashes=[r['sequence_hash'] for r in rows],status='profiling');save()
  output['stage']='native_profiles';save()
  values,_,elapsed=worker.measure_batch([r['sequence'] for r in rows],32);assert len(values)==32
  for row,value in zip(rows,values):
   observed=journal.observation(value,requirements,row['sequence']);a3d=value.get('aggrescan3d') or {}
   numeric=all(type(value.get(k)) in (int,float) and math.isfinite(value[k]) for k in journal.METRICS)
   numeric=numeric and all(Decimal(str(a3d.get(k))).is_finite() for k in journal.A3D_METRICS)
   provenance=(a3d.get('status')=='ok' and a3d.get('sequence_hash')==row['sequence_hash'] and a3d.get('residue_count')==len(row['sequence']) and a3d.get('ph_scale')=='default-scale' and a3d.get('surface_method')=='FreeSASA-2.0.1-NACCESS-LR100' and a3d.get('mode')=='static-no-minimization' and Decimal(str(a3d.get('radius_angstrom')))==10 and all(re.fullmatch('[0-9a-f]{64}',a3d.get(k,'')) for k in ('input_structure_sha256','normalized_structure_sha256')))
   output['records'].append({'sequence_hash':row['sequence_hash'],'length':len(row['sequence']),'numeric_metrics_complete':numeric,'provenance_complete':bool(provenance),'runtime_complete':bool(numeric and provenance and not value.get('error') and not observed['execution_censored']),'observation':observed})
  output['native_wall_seconds']=elapsed;output['completed_profiles']=sum(r['runtime_complete'] for r in output['records']);output['runtime_inconclusive_count']=32-output['completed_profiles']
  output['sources_unchanged']=all(digest(p)==h for p,h in supplemental_verified.items()) and all(digest(p)==h for p,h in verified.items()) and all(digest(v['target'])==weights[k] for k,v in manifest['weights'].items())
  output['status']='passed' if output['completed_profiles']==32 and output['sources_unchanged'] else 'not_qualified'
 except BaseException as error:output.update(status='not_qualified',error_type=type(error).__name__)
 finally:output['wall_seconds']=time.monotonic()-started;save()
 return output
if __name__=='__main__':sys.exit(0 if main(Path(sys.argv[1]))['status']=='passed' else 1)
