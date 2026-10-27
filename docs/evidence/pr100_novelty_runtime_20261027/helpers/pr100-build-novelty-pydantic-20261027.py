"""Run only in the qualified slave annotation interpreter; export public assets."""
import hashlib,importlib.metadata as m,json,pathlib,tarfile,re,io
out=pathlib.Path('/root/pr100-novelty-runtime-staging-20261027');out.mkdir(exist_ok=True)
names=['pydantic','pydantic_core','annotated-types','typing_extensions','typing-inspection']
files={};packages={}
for name in names:
 d=m.distribution(name);packages[name]=d.version;base=pathlib.Path(d.locate_file('')).resolve()
 assert str(base).endswith('/lib/python3.12/site-packages')
 for rel in d.files or []:
  p=pathlib.Path(d.locate_file(rel));relative=pathlib.Path(str(rel))
  if '..' in relative.parts or relative.is_absolute() or '__pycache__' in relative.parts or p.suffix=='.pyc':continue
  if not p.is_file():continue
  assert p.resolve().is_relative_to(base)
  assert not any(x.lower() in {'.env','.ssh','.npmrc','credentials','auth','secrets'} or x.endswith(('.pem','.key')) for x in relative.parts)
  target='lib/python3.12/site-packages/'+relative.as_posix()
  files[target]=p
manifest={'schema':'pr100-public-novelty-assets-v1','packages':packages,'files':{}}
archive=out/'pydantic-assets.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
 for target,p in sorted(files.items()):
  raw=p.read_bytes();manifest['files'][target]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  info=tarfile.TarInfo(target);info.size=len(raw);info.mode=0o755 if target=='bin/hmmscan' else 0o644;tar.addfile(info,io.BytesIO(raw))
 raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode();info=tarfile.TarInfo('pydantic-assets-manifest.json');info.size=len(raw);info.mode=0o644;tar.addfile(info,io.BytesIO(raw))
(out/'pydantic-assets-manifest.json').write_bytes(raw)
print(json.dumps({'status':'built','packages':packages,'files':len(files),'asset_bytes':sum(v['bytes'] for v in manifest['files'].values()),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(raw).hexdigest()}))
