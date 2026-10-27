"""Dedicated bounded owned-tree qualification; never a publication job."""
import hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
def main(root):
 started=time.monotonic();config=json.loads((root/'qualification-config.json').read_text());release=Path(config['release'])
 spec=importlib.util.spec_from_file_location('_qualification_control',release/'scripts/pr100_native_probe.py')
 control=importlib.util.module_from_spec(spec);sys.modules[spec.name]=control;spec.loader.exec_module(control);control.dedicated_controller()
 receipt={'schema':'pr100-native-runtime-qualification-v1','status':'starting','host':'slave','workers':32,'authoritative':False,'publication_enabled':False,'application_source_sha':config['source_sha'],'hard_cap_seconds':360,'started_at_unix':time.time()};child=None
 def alarm(*_):raise TimeoutError('qualification_deadline')
 for number in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):signal.signal(number,alarm)
 signal.alarm(350)
 try:
  raw=Path(config['manifest']).read_bytes();assert hashlib.sha256(raw).hexdigest()==config['manifest_sha256']
  manifest=json.loads(raw);bundle=json.loads(Path(config['unpack_receipt']).read_text())
  assert bundle['status']=='files_verified' and bundle['files']==len(manifest['files']) and len(bundle['weights'])==4 and bundle['manifest_sha256']==config['manifest_sha256']
  stamp=json.loads((release/'build_stamp.json').read_text());assert stamp['git_sha']==config['source_sha'] and stamp['validation']['passed'] and stamp['validation']['skipped']==0
  assert all(hashlib.sha256((release/n).read_bytes()).hexdigest()==h for n,h in stamp['files'].items())
  receipt['release_manifest_files']=len(stamp['files']);cores=set(range(32));assert cores<=set(os.sched_getaffinity(0))
  identities=set()
  for cpu in cores:
   top=Path('/sys/devices/system/cpu/cpu'+str(cpu)+'/topology');identities.add(((top/'physical_package_id').read_text(),(top/'core_id').read_text()))
  assert len(identities)==32;os.sched_setaffinity(0,cores);receipt.update(affinity=sorted(cores),cpu_affinity_verified=True)
  prefix=Path('/root/pr100-native-alt-20261027');wrapper=prefix/'bin/native-python'
  wrapper_bytes=('#!/bin/sh\nexport OPENMM_PLUGIN_DIR='+str(prefix/'lib/plugins')+'\nexport LD_LIBRARY_PATH='+str(prefix/'lib')+'\nexport OPENMM_CPU_THREADS=1\nexport MKL_NUM_THREADS=1\nexport HIP_VISIBLE_DEVICES=\nexport ROCR_VISIBLE_DEVICES=\nexec '+str(prefix/'bin/python3.12')+' "$@"\n').encode()
  if wrapper.exists():assert wrapper.read_bytes()==wrapper_bytes
  else:wrapper.write_bytes(wrapper_bytes);wrapper.chmod(0o755)
  runtime={'native_python':str(wrapper),'native_path':str(prefix/'bin')+':/usr/local/bin:/usr/bin:/bin','native_python_path':str(prefix/'shim'),'native_tools':str(prefix/'bin')};receipt['runtime']=runtime
  environment={'PATH':runtime['native_path'],'PYTHONPATH':str(release)+':'+runtime['native_python_path'],'HOME':'/root','USER':'root','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  child=subprocess.Popen([str(wrapper),'-B',str(root/'qualification-worker.py'),str(root)],env=environment,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  stdout,stderr=child.communicate(timeout=max(1,345-(time.monotonic()-started)));receipt['returncode']=child.returncode
  receipt['worker_output_bytes']={'stdout':len(stdout),'stderr':len(stderr)}
  receipt['worker_error_categories']=[label for label in ('SyntaxError','ModuleNotFoundError','ImportError','Permission denied','No such file or directory',"can't open file",'unrecognized arguments','error while loading shared libraries','MKL','OpenBLAS','Intel','libmkl','libiomp','Fatal','could not load','Cannot load') if label.encode() in stdout+stderr]
  result=json.loads((root/'measurements.json').read_text())
  for key in ('runtime_identity','runtime_identity_sha256','reference_runtime_identity','runtime_differences','runtime_manifest_verified','completed_profiles','runtime_inconclusive_count','native_wall_seconds','python_version','openmm_platforms','error_type','missing_distribution','stage'):
   if key in result:receipt[key]=result[key]
  receipt['status']='passed' if child.returncode==0 and result['status']=='passed' else 'not_qualified'
 except BaseException as error:receipt.update(status='not_qualified',error_type=type(error).__name__)
 finally:
  signal.alarm(0);receipt['cleanup']=control.stop_descendants([child] if child else [],timeout=8)
  if receipt['cleanup']['remaining_descendants']!=0 or not receipt['cleanup']['fixed_point_reached']:receipt['status']='not_qualified'
  receipt['wall_seconds']=time.monotonic()-started;(root/'qualification-receipt.json').write_text(json.dumps(receipt))
 return receipt
if __name__=='__main__':
 code=1
 try:code=0 if main(Path(sys.argv[1]))['status']=='passed' else 1
 finally:os._exit(code)
