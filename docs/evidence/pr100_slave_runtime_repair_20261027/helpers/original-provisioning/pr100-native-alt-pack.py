import gzip,hashlib,io,json,os,pathlib,sys,tarfile,time
manifest=json.loads(pathlib.Path(sys.argv[1]).read_text());receipt=pathlib.Path(sys.argv[2]);start=time.time();count=0;total=0
os.sched_setaffinity(0,{56});os.nice(10)
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed:
 with tarfile.open(fileobj=compressed,mode='w|') as archive:
  for name,entry in manifest['files'].items():
   p=pathlib.Path(name);raw=p.read_bytes()
   if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('source_hash_changed')
   info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);info.mode=p.stat().st_mode&0o777
   archive.addfile(info,io.BytesIO(raw));count+=1;total+=len(raw)
   if count%50==0:receipt.write_text(json.dumps({'status':'packing','files':count,'bytes':total,'elapsed_seconds':time.time()-start}))
receipt.write_text(json.dumps({'status':'packed','files':count,'bytes':total,'elapsed_seconds':time.time()-start}))
