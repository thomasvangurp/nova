import pathlib,ast,json,base64,sys,importlib
sys.path.insert(0,'/tmp');connector=importlib.import_module('pr100-novelty-remote')
s=pathlib.Path('/tmp/pr100-launch-qualified-novelty-replay-20261027.py').read_text();tree=ast.parse(s);code=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='remote' for x in n.targets)))
code=code.replace("p=json.loads(gzip.decompress(__import__('sys').stdin.buffer.read()));", "p=PAYLOAD;")
old="for name,encoded in p['files'].items():\n assert name in ('input.json','old-archive.csv','new-archive.csv','worker.py');(base/name).write_bytes(base64.b64decode(encoded))"
new="""import hashlib
previous=prefix/'replay-25967-v1'
for name,expected in {'old-archive.csv':'2e426028e1bb34b0d14ba17759c539fc91f7a825c9b7e5fe4f9ec5c3618ba260','new-archive.csv':'faa66bc72600e2fd4979754ab02c2637ef7d8725ee487fb5cd48cfa260b7eb2e'}.items():
 raw=(previous/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected;(base/name).write_bytes(raw)
(base/'input.json').write_bytes((previous/'input.json').read_bytes())
(base/'worker.py').write_bytes(base64.b64decode(p['worker']))"""
assert old in code;code=code.replace(old,new)
report=pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027');prepared=json.loads((report/'qualification-preparation.json').read_text())
payload={'identity':prepared['slave']['value']['result']['identity_sha256'],'worker':base64.b64encode(pathlib.Path('/tmp/pr100-qualified-novelty-replay-worker-20261027.py').read_bytes()).decode()}
v=connector.run('slave','PAYLOAD='+repr(payload)+'\n'+code,timeout=35);(report/'qualified-replay-reused-slave-launch.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
