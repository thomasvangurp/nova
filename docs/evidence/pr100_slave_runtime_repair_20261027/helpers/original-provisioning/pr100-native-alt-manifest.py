import hashlib,json,os,pathlib,re,subprocess,sys
base=pathlib.Path('/root/.pixi/envs/metanano')
source=pathlib.Path('/root/pr100-native-runtime-20261027/reviewed-runtime-allowlist.json')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='75b34070e241a95ee8155b2cb617119d3eaa82dfa252daa4fde206b4c5f84470'
original=json.loads(source.read_text());known={**original['files'],**original['external_scientific_assets']}
destination='/root/pr100-native-alt-20261027';chosen=set()
for name in known:
    if '/site-packages/' in name:
        package=name.split('/site-packages/',1)[1].split('/')[0]
        if package.lower().startswith(('torch','triton','pyside','shiboken','nvidia')):continue
        if '/ImmuneBuilder/trained_model/' in name:continue
        chosen.add(name)
    elif '/share/' in name:chosen.add(name)
    elif '/lib/' in name and 'OpenMM' in pathlib.Path(name).name and not any(x in name for x in ('CUDA','Cuda','OpenCL','HIP')):chosen.add(name)
for name in ('TNP','NanoBodyBuilder2','hmmscan','psa','mkdssp'):
    p=str(base/'bin'/name);assert p in known;chosen.add(p)
chosen.update(original['external_scientific_assets'])
queue=list(chosen);examined=set()
while queue:
    name=queue.pop()
    if name in examined:continue
    examined.add(name);p=pathlib.Path(name)
    with p.open('rb') as f:elf=f.read(4)==b'\x7fELF'
    if not elf:continue
    result=subprocess.run(['ldd',name],capture_output=True,text=True,timeout=10)
    for line in result.stdout.splitlines():
        if 'not found' in line:raise RuntimeError('source_library_unresolved')
        for match in re.findall(r'(/[^\s()]+)',line):
            q=pathlib.Path(match).resolve()
            if str(q).startswith(str(base)+'/') or str(q) in original['external_scientific_assets']:
                assert str(q) in known
                if str(q) not in chosen:chosen.add(str(q));queue.append(str(q))
targets={}
for name in sorted(chosen):
    if name.startswith(str(base)+'/'):target=destination+name[len(str(base)):]
    elif name.endswith('/bittensor.py'):target=destination+'/shim/bittensor.py'
    elif name.endswith('/aggrescan3d-rs'):target=destination+'/bin/aggrescan3d-rs'
    elif name.endswith('/freesasa'):target=destination+'/bin/freesasa'
    else:target=name
    targets[name]={'target':target,'sha256':known[name],'bytes':pathlib.Path(name).stat().st_size}
aliases={}
for name,target in original.get('symlinks',{}).items():
    if target in targets and name.startswith(str(base)+'/'):
        aliases[destination+name[len(str(base)):]]=targets[target]['target']
weights={str(i):{'url':'https://zenodo.org/record/7258553/files/nanobody_model_'+str(i)+'?download=1',
    'target':destination+'/lib/python3.12/site-packages/ImmuneBuilder/trained_model/nanobody_model_'+str(i),
    'sha256':known[str(base/'lib/python3.12/site-packages/ImmuneBuilder/trained_model'/('nanobody_model_'+str(i)))]}
    for i in range(1,5)}
value={'schema':'pr100-native-alt-runtime-v1','reference_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'prefix':destination,'base_python':'/root/bgen/bin/python','files':targets,'aliases':aliases,'weights':weights,
    'runtime_differences':['Python base reused from slave bgen','Torch2.7.1+cu126 reused from slave bgen','TNP and NBB2 entrypoint shebangs relocated'],
    'scientific_qualification':False}
out=pathlib.Path(sys.argv[1]);out.write_text(json.dumps(value,sort_keys=True))
print(json.dumps({'status':'manifest_ready','files':len(targets),'bytes':sum(v['bytes'] for v in targets.values()),
    'aliases':len(aliases),'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
