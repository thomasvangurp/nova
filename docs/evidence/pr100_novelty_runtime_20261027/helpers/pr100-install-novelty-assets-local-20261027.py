import sys,pathlib,subprocess,json,tarfile,io,hashlib
prefix=pathlib.Path('/root/pr100-novelty-runtime-20261027');raw=pathlib.Path('/root/pr100-novelty-runtime-staging-20261027/annotation-assets.tar.gz').read_bytes();expected=EXPECTED
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
