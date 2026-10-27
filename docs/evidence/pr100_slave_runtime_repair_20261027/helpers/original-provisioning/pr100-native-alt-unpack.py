import concurrent.futures,hashlib,json,os,pathlib,subprocess,sys,tarfile,tempfile,time,urllib.request
manifest_path=pathlib.Path(sys.argv[1]);manifest=json.loads(manifest_path.read_text());receipt=pathlib.Path(sys.argv[2]);prefix=pathlib.Path(manifest['prefix']);started=time.time();deadline=time.monotonic()+1200
count=0;total=0;seen=set();weight_status={}
def save(status,**extra):
 value={'status':status,'files':count,'bytes':total,'weights':weight_status,'scientific_qualification':False,
    'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'elapsed_seconds':time.time()-started,**extra}
 tmp=receipt.with_suffix('.next');tmp.write_text(json.dumps(value));os.replace(tmp,receipt)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def parents(p):
 for parent in p.parents:
  if parent.is_symlink():raise ValueError('symlink_parent')
def weight(index,entry):
 p=pathlib.Path(entry['target']);parents(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  if digest(p)!=entry['sha256']:raise ValueError('existing_weight_mismatch')
  return index,{'status':'verified','bytes':p.stat().st_size}
 temp=p.with_name('.'+p.name+'.download');attempt=0
 while time.monotonic()<deadline and attempt<8:
  attempt+=1
  try:
   offset=temp.stat().st_size if temp.exists() else 0
   headers={'User-Agent':'pr100-native-runtime/1'}
   if offset:headers['Range']='bytes='+str(offset)+'-'
   with urllib.request.urlopen(urllib.request.Request(entry['url'],headers=headers),timeout=20) as response:
    mode='ab' if offset and response.status==206 and response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-') else 'wb'
    with temp.open(mode) as f:
     while part:=response.read(256*1024):
      if time.monotonic()>deadline:raise TimeoutError('weight_deadline')
      f.write(part)
   if digest(temp)!=entry['sha256']:raise ValueError('weight_hash_mismatch')
   os.replace(temp,p);return index,{'status':'verified','bytes':p.stat().st_size,'attempts':attempt}
  except (OSError,ValueError,TimeoutError):
   if time.monotonic()>=deadline:break
   time.sleep(min(3,max(0,deadline-time.monotonic())))
 raise RuntimeError('weight_download_failed')
pool=None
try:
 assert str(prefix)=='/root/pr100-native-alt-20261027'
 if not (prefix/'pyvenv.cfg').exists():
  assert not prefix.exists()
  p=subprocess.run([manifest['base_python'],'-m','venv','--system-site-packages','--without-pip',str(prefix)],capture_output=True,timeout=40)
  assert p.returncode==0
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
 futures=[pool.submit(weight,k,v) for k,v in manifest['weights'].items()]
 save('receiving')
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
  for member in archive:
   source='/'+member.name
   if not member.isfile() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts or source not in manifest['files'] or source in seen:raise ValueError('unexpected_member')
   entry=manifest['files'][source];p=pathlib.Path(entry['target'])
   assert p.is_relative_to(prefix) or str(p)=='/root/pr100-qualification-20261026/build-tools/.pixi/envs/default/lib/libgcc_s.so.1'
   parents(p);p.parent.mkdir(parents=True,exist_ok=True)
   raw=archive.extractfile(member).read()
   if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('file_hash_mismatch')
   if p.name in ('TNP','NanoBodyBuilder2') and p.parent==prefix/'bin':
    first,sep,rest=raw.partition(b'\n');assert first.startswith(b'#!') and b'/root/.pixi/envs/metanano/bin/python' in first
    raw=b'#!'+str(prefix/'bin/python3.12').encode()+sep+rest
   target_hash=hashlib.sha256(raw).hexdigest()
   if p.exists() or p.is_symlink():
    if p.is_symlink() or digest(p)!=target_hash:raise ValueError('existing_target_mismatch')
   else:
    fd,name=tempfile.mkstemp(prefix='.native-alt-',dir=p.parent)
    try:
     with os.fdopen(fd,'wb') as f:f.write(raw)
     os.chmod(name,member.mode&0o777);os.replace(name,p)
    finally:
     if os.path.exists(name):os.unlink(name)
   seen.add(source);count+=1;total+=len(raw)
   if count%50==0:save('receiving')
 if seen!=set(manifest['files']):raise ValueError('incomplete_bundle')
 for name,target in manifest['aliases'].items():
  p=pathlib.Path(name);d=pathlib.Path(target);assert p.is_relative_to(prefix) and d.is_relative_to(prefix)
  parents(p);p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists() or p.is_symlink():
   if not p.is_symlink() or p.resolve()!=d.resolve():raise ValueError('alias_mismatch')
  else:p.symlink_to(os.path.relpath(d,p.parent))
 wrapper=prefix/'bin/native-python'
 wrapper_bytes=('#!/bin/sh\nexport OPENMM_PLUGIN_DIR='+str(prefix/'lib/plugins')+'\nexport LD_LIBRARY_PATH='+str(prefix/'lib')+'\nexec '+str(prefix/'bin/python3.12')+' "$@"\n').encode()
 if wrapper.exists():
  if wrapper.read_bytes()!=wrapper_bytes:raise ValueError('existing_wrapper_mismatch')
 else:wrapper.write_bytes(wrapper_bytes);wrapper.chmod(0o755)
 save('waiting_for_weights')
 for future in concurrent.futures.as_completed(futures,timeout=max(1,deadline-time.monotonic())):
  index,value=future.result();weight_status[index]=value;save('waiting_for_weights')
 assert len(weight_status)==4
 save('files_verified',aliases=len(manifest['aliases']))
except BaseException as error:
 save('failed',error_type=type(error).__name__)
 raise SystemExit(1)
finally:
 if pool:pool.shutdown(wait=True,cancel_futures=True)
