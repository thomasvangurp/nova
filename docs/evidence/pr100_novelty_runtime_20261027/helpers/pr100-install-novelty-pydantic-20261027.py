import pathlib,sys,importlib,json
sys.path.insert(0,'/tmp');remote=importlib.import_module('pr100-novelty-remote')
install=pathlib.Path('/tmp/pr100-install-novelty-assets-local-20261027.py').read_text().replace('annotation-assets.tar.gz','pydantic-assets.tar.gz').replace('assets-manifest.json','pydantic-assets-manifest.json')
copy="""import subprocess,json
p=subprocess.run(['scp','-q','-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-P','28358','/root/pr100-novelty-runtime-staging-20261027/pydantic-assets.tar.gz','root@ssh2.vast.ai:/root/pr100-novelty-runtime-staging-20261027/pydantic-assets.tar.gz'],capture_output=True,timeout=80)
print(json.dumps({'copy_exit':p.returncode}))
"""
results={'copy':remote.run('master',copy,timeout=100)}
print(json.dumps(results),flush=True)
for host in ['slave','master']:
 base='/root/pr100-qualification-20261026/validator-annotation-runtime/bin/python' if host=='slave' else '/root/.pixi/envs/metanano/bin/python3.12'
 code="EXPECTED='5b8a4a656f087a4f43425686237ab37b56ef8296041809884fe89d8221cefdf2'\nBASE="+repr(base)+'\n'+install
 results[host]=remote.run(host,code,timeout=70);print(json.dumps({host:results[host]}),flush=True)
pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027/pydantic-install.json').write_text(json.dumps(results,indent=2)+'\n')
