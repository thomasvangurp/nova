import hashlib,json,os,pathlib,shlex,subprocess,time
base=pathlib.Path('/root/pr100-native-alt-staging-20261027');receipt=base/'direct-receipt.json';started=time.time();children=[]
def save(status,**fields):
 p=receipt.with_suffix('.next');p.write_text(json.dumps({'status':status,'updated_at_unix':time.time(),'elapsed_seconds':time.time()-started,**fields}));os.replace(p,receipt)
opts=['-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8','-oServerAliveInterval=15','-oServerAliveCountMax=2']
def ssh(command):return ['ssh','-T',*opts,'-p','4116','root@122.51.254.66',shlex.join(command)]
try:
 save('creating_manifest')
 p=subprocess.run(['python3',str(base/'manifest.py'),str(base/'manifest.json')],capture_output=True,timeout=120)
 if p.returncode:raise RuntimeError('manifest_failed')
 manifest=json.loads((base/'manifest.json').read_text());assert len(manifest['weights'])==4
 p=subprocess.run(ssh(['mkdir','-p',str(base)]),capture_output=True,timeout=20);assert p.returncode==0
 for name in ('manifest.json','unpack.py'):
  p=subprocess.run(['scp','-q',*opts,'-P','4116',str(base/name),'root@122.51.254.66:'+str(base/name)],capture_output=True,timeout=90)
  if p.returncode:raise RuntimeError('helper_copy_failed')
 expected={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in ('manifest.json','unpack.py')}
 verify='import hashlib,pathlib; b=pathlib.Path('+repr(str(base))+'); e='+repr(expected)+'; assert all(hashlib.sha256((b/n).read_bytes()).hexdigest()==h for n,h in e.items())'
 p=subprocess.run(ssh(['python3','-c',verify]),capture_output=True,timeout=20);assert p.returncode==0
 save('streaming',manifest_sha256=expected['manifest.json'],files=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values()))
 source=subprocess.Popen(['timeout','--kill-after=5','1250','python3',str(base/'pack.py'),str(base/'manifest.json'),str(base/'pack-receipt.json')],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL);children.append(source)
 dest=subprocess.Popen(ssh(['timeout','--kill-after=5','1250','python3',str(base/'unpack.py'),str(base/'manifest.json'),str(base/'unpack-receipt.json')]),stdin=source.stdout,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);children.append(dest);source.stdout.close()
 dest.wait(timeout=1290);source.wait(timeout=10)
 if source.returncode or dest.returncode:raise RuntimeError('provision_stream_failed')
 save('files_verified',manifest_sha256=expected['manifest.json'],files=len(manifest['files']))
except BaseException as error:
 save('failed',error_type=type(error).__name__)
finally:
 for p in children:
  if p.poll() is None:p.kill()
 for p in children:
  try:p.wait(timeout=8)
  except subprocess.TimeoutExpired:pass
