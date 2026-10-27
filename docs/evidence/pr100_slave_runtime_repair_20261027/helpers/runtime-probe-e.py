import subprocess,json,shlex,pathlib,time
remote='''import subprocess,json,re,os
prefix='/root/pr100-native-alt-20261027';release='/root/releases/51688c0a458f2aa401d7b1f569b2cf364f96aebc'
environment={'PATH':prefix+'/bin:/usr/local/bin:/usr/bin:/bin','PYTHONPATH':release+':'+prefix+'/shim','HOME':'/root','USER':'root','LANG':'C.UTF-8','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
code='import torch; print("PR100_IMPORT_OK",flush=True); a=torch.ones((8,8)); b=a@a; c=torch.linalg.svd(a); print("PR100_CPU_MATH_OK",flush=True); import openmm; assert "CPU" in [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]; print("PR100_OPENMM_OK",flush=True)'
p=subprocess.run([prefix+'/bin/native-python','-B','-c',code],env=environment,cwd=prefix,capture_output=True,timeout=30,preexec_fn=lambda:os.sched_setaffinity(0,set(range(32))))
raw=p.stdout+p.stderr;names=sorted(set(m.decode() for m in re.findall(rb'libmkl_[A-Za-z0-9_.-]+[.]so(?:[.][0-9]+)*',raw)))
print(json.dumps({'returncode':p.returncode,'output_bytes':{'stdout':len(p.stdout),'stderr':len(p.stderr)},'missing_library_basenames':names,'torch_import_ok':b'PR100_IMPORT_OK' in raw,'cpu_math_ok':b'PR100_CPU_MATH_OK' in raw,'openmm_ok':b'PR100_OPENMM_OK' in raw,'categories':[s for s in ['MKL','Cannot load','No such file','error while loading shared libraries','ImportError','ModuleNotFoundError'] if s.encode() in raw]}))
'''
pi="import subprocess,json,shlex\ncode="+repr(remote)+"\np=subprocess.run(['ssh','-T','-i','/home/thomasg/.ssh/id_ed25519','-oIdentitiesOnly=yes','-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8','-p','4116','root@122.51.254.66',shlex.join(['python3','-c',code])],capture_output=True,timeout=45)\ntry:d=json.loads(p.stdout) if p.returncode==0 else None\nexcept Exception:d=None\nprint(json.dumps({'slave_transport_exit':p.returncode,'probe':d}))\n"
p=subprocess.run(['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10','thomasg@100.64.243.65','python3','-'],input=pi,text=True,capture_output=True,timeout=55)
try:d=json.loads(p.stdout) if p.returncode==0 else None
except Exception:d=None
out={'pi_transport_exit':p.returncode,'result':d};root=pathlib.Path('/Users/tvgdeena/nova-reports/pr100-slave-native-alt-20261027/recovery-e');(root/('mkl-probe-'+str(time.time_ns())+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
