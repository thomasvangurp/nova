import pathlib,hashlib,json,tarfile,io,re
source=pathlib.Path('/root/pr100-qualification-20261026/hmmer-libs');stage=pathlib.Path('/root/pr100-novelty-runtime-staging-20261027');files={}
for p in source.iterdir():
 if p.is_file() and re.fullmatch(r'lib[A-Za-z0-9_.+-]+[.]so(?:[.][0-9]+)*',p.name):files['hmmer-libs/'+p.name]=p.read_bytes()
assert files and 'hmmer-libs/libmpi.so.40' in files
files['bin/novelty-python']=b'#!/bin/sh\nset -eu\nruntime_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\nexport LD_LIBRARY_PATH="$runtime_root/hmmer-libs"\nexec "$runtime_root/bin/python" "$@"\n'
manifest={'schema':'pr100-public-novelty-assets-v1','files':{n:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for n,v in files.items()}}
raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();files['hmmer-assets-manifest.json']=raw
archive=stage/'hmmer-assets.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
 for n,v in sorted(files.items()):
  info=tarfile.TarInfo(n);info.size=len(v);info.mode=0o755 if n=='bin/novelty-python' else 0o644;tar.addfile(info,io.BytesIO(v))
(stage/'hmmer-assets-manifest.json').write_bytes(raw)
print(json.dumps({'files':len(manifest['files']),'libraries':sorted(pathlib.Path(n).name for n in manifest['files'] if n.startswith('hmmer-libs/')),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(raw).hexdigest()}))
