import sys,importlib,json,pathlib
sys.path.insert(0,'/tmp');remote=importlib.import_module('pr100-novelty-remote')
install=pathlib.Path('/tmp/pr100-install-novelty-assets-local-20261027.py').read_text()
slave="import pathlib,hashlib\nEXPECTED=hashlib.sha256(pathlib.Path('/root/pr100-novelty-runtime-staging-20261027/annotation-assets.tar.gz').read_bytes()).hexdigest()\nBASE='/root/pr100-qualification-20261026/validator-annotation-runtime/bin/python'\n"+install
print(json.dumps({'host':'slave','result':remote.run('slave',slave,timeout=100)}),flush=True)
master=r'''import subprocess,pathlib,shlex,hashlib,json
base=pathlib.Path('/root/pr100-novelty-runtime-staging-20261027');base.mkdir(exist_ok=True)
opts=['-oBatchMode=yes','-oForwardAgent=no','-oStrictHostKeyChecking=yes','-oHostKeyAlias=[ssh2.vast.ai]:28358','-oConnectTimeout=8']
code="import pathlib,hashlib,json;print(json.dumps({'sha256':hashlib.sha256(pathlib.Path('/root/pr100-novelty-runtime-staging-20261027/annotation-assets.tar.gz').read_bytes()).hexdigest()}))"
p=subprocess.run(['ssh','-T',*opts,'-p','28358','root@ssh2.vast.ai',shlex.join(['python3','-c',code])],capture_output=True,timeout=30);assert p.returncode==0;EXPECTED=json.loads(p.stdout)['sha256']
path=base/'annotation-assets.tar.gz'
if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED:
 p=subprocess.run(['scp','-q',*opts,'-P','28358','root@ssh2.vast.ai:/root/pr100-novelty-runtime-staging-20261027/annotation-assets.tar.gz',str(path)],capture_output=True,timeout=240);assert p.returncode==0
BASE='/root/.pixi/envs/metanano/bin/python3.12'
'''+install
v=remote.run('master',master,timeout=300);print(json.dumps({'host':'master','result':v}),flush=True)
pathlib.Path('/Users/tvgdeena/nova-reports/pr100-novelty-runtime-20261027/resume-provision-receipt.json').write_text(json.dumps(v,indent=2)+'\n')
