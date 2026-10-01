#!/usr/bin/env python3
"""One isolated, guarded seed-noise test. Runtime credentials stay in memory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import urllib.request

import nim19_test_control as control

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'data/nim19_benchmark/20260917/normal-seeds'
MANIFEST = DEST / 'instance.json'
LABEL = 'nova-boltz2-seed-noise-test-20260917'
OFFER_ID = 40396208
FILES = ['seed-runner.json', 'seed-runner-bootstrap-failure.json', 'paired-progress.json', 'pair-setup.json'] + [
    f'paired-reference-seed-{seed}.json' for seed in [68, 69, 70, 71]
]


def api(path, body=None, method='GET'):
    request = urllib.request.Request(
        'https://console.vast.ai/api/v0/' + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={'Authorization': 'Bearer ' + control.keychain('vast.ai'),
                 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.status, json.load(response)


def configure():
    saved = json.loads(MANIFEST.read_text())
    if saved['label'] != LABEL or not isinstance(saved['id'], int):
        raise RuntimeError('wrong_manifest')
    control.INSTANCE_ID = saved['id']
    control.LABEL = LABEL
    return control.instance()


def create():
    if MANIFEST.exists():
        configure()
        print(json.dumps({'event': 'instance_exists'}))
        return 0
    _, existing = api('instances/?owner=me')
    matches = [h for h in existing.get('instances', []) if h.get('label') == LABEL]
    if len(matches) > 1:
        raise RuntimeError('ambiguous_instance')
    if matches:
        h = matches[0]
        if h.get('num_gpus') != 1 or h.get('gpu_name') != 'RTX 6000Ada':
            raise RuntimeError('wrong_existing_instance')
        instance_id = h['id']
    else:
        query = {'limit': 100, 'type': 'on-demand', 'num_gpus': {'eq': 1},
                 'gpu_ram': {'gte': 47000}, 'cpu_ram': {'gte': 64000},
                 'cpu_cores_effective': {'gte': 12},
                 'verified': {'eq': True}, 'rentable': {'eq': True},
                 'rented': {'eq': False}}
        _, data = api('bundles/', query, 'POST')
        o = next(x for x in data['offers'] if x['id'] == OFFER_ID)
        driver = o.get('driver_version', '')
        if not re.fullmatch(r'[0-9.]+', driver):
            raise RuntimeError('unknown_driver')
        if not (o.get('num_gpus') == 1 and o.get('gpu_name') == 'RTX 6000Ada'
                and o.get('gpu_ram', 0) >= 48000 and o.get('dph_total', 999) <= .75
                and o.get('cpu_cores_effective', 0) >= 12
                and o.get('cpu_ram', 0) >= 64000
                and tuple(map(int, driver.split('.'))) >= (590, 44, 1)):
            raise RuntimeError('offer_guard_failed')
        status, data = api('asks/' + str(OFFER_ID) + '/', {
            'image': 'vastai/base-image:cuda-13.2.1-auto', 'label': LABEL,
            'disk': 100, 'runtype': 'ssh_direct', 'target_state': 'running',
            'cancel_unavail': True,
        }, 'PUT')
        if data.get('success') is not True or not isinstance(data.get('new_contract'), int):
            print(json.dumps({'event': 'create_failed', 'http_status': status}))
            return 1
        instance_id = data['new_contract']
    DEST.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({'id': instance_id, 'label': LABEL,
                                   'offer_id': OFFER_ID, 'gpu_count': 1}))
    configure()
    print(json.dumps({'event': 'created', 'id': instance_id, 'label': LABEL}))
    return 0


def ssh(host):
    port = str(host['ports']['22/tcp'][0]['HostPort'])
    return ['ssh', '-i', control.SSH_KEY, '-p', port, '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'LogLevel=ERROR']


def deploy():
    host = configure()
    code = "from pathlib import Path;b=Path('/root/nim19-test');[(b/n).mkdir(parents=True,exist_ok=True) for n in ['client','oss-repo/tools','oss-repo/compound_miner','oss-repo/requirements','oss-repo/external_tools/boltz']]"
    if control.remote(code, timeout=45):
        return 1
    transfers = [
        ([REPO/'tools'/name for name in ['nim19_paired_benchmark.py', 'nim19_benchmark.py',
           'nim19_pair_setup.py', 'nim19_seed_runner.py']], '/root/nim19-test/client/'),
        ([REPO/'tools/score_boltz_batch.py'], '/root/nim19-test/oss-repo/tools/'),
        ([REPO/'compound_miner/targets.py', REPO/'compound_miner/__init__.py', REPO/'compound_miner/types.py'], '/root/nim19-test/oss-repo/compound_miner/'),
        ([REPO/'requirements/boltz-worker-constraints.txt'], '/root/nim19-test/oss-repo/requirements/'),
        ([REPO/'external_tools/boltz/pyproject.toml', REPO/'external_tools/boltz/LICENSE', REPO/'external_tools/boltz/README.md'], '/root/nim19-test/oss-repo/external_tools/boltz/'),
        ([REPO/'external_tools/boltz/src'], '/root/nim19-test/oss-repo/external_tools/boltz/'),
        ([REPO/'data/nim19_benchmark/20260917/paired/paired-portfolio.json'], '/root/nim19-test/'),
        ([Path('/Users/tvgdeena/nova-nanobody-standalone/data/msa_files/P40261.a3m')], '/root/nim19-test/client/'),
    ]
    for files, target in transfers:
        # Only public source/scientific inputs. Never transfer wallet/config roots.
        result = subprocess.run([
            'rsync', '-rlt', '--checksum', '--no-owner', '--no-group', '--no-perms',
            '--exclude=.git', '--exclude=.env*', '--exclude=__pycache__',
            '--exclude=*.pem', '--exclude=*.key', '--exclude=*credential*',
            '--exclude=*secret*', '--exclude=.npmrc', '-e', shlex.join(ssh(host)),
            *map(str, files), 'root@'+host['public_ipaddr']+':'+target,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        print(json.dumps({'event': 'source', 'file_count': len(files),
                          'verified': result.returncode == 0}), flush=True)
        if result.returncode:
            return 1
    return 0


def start():
    configure()
    code = control.DOCKER + r'''
from pathlib import Path
b=Path('/root/nim19-test')
state=b/'seed-runner.json'
if state.exists():
    if json.loads(state.read_text()).get('stage')!='failed':
        emit('source',summary={'stage':'already_started'});sys.exit(0)
    if any((b/f'paired-reference-seed-{s}.json').exists() for s in [68,69,70,71]):
        emit('operation_failed',error_class='PartialSuiteNeedsReview');sys.exit(1)
    previous=b/'seed-runner-bootstrap-failure.json'
    if previous.exists():
        emit('operation_failed',error_class='RepeatedBootstrapFailure');sys.exit(1)
    state.replace(previous)
env=dict(HOME='/root/nim19-test/oss-home',PATH='/usr/local/bin:/usr/bin:/bin',LANG='C.UTF-8')
(b/'oss-home').mkdir(exist_ok=True)
subprocess.Popen(['python3',str(b/'client/nim19_seed_runner.py')],env=env,
                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
emit('source',summary={'stage':'seed_runner_started'})
'''
    return control.remote(code, timeout=60)


def progress():
    configure()
    code = control.DOCKER + r'''
from pathlib import Path
b=Path('/root/nim19-test')
for n in ['seed-runner.json','pair-setup.json','paired-progress.json']:
    if (b/n).is_file():emit('source',file=n,summary=json.loads((b/n).read_text()))
for seed in [68,69,70,71]:
    n=f'paired-reference-seed-{seed}.json'
    if (b/n).is_file():
        d=json.loads((b/n).read_text());emit('source',file=n,summary={'seed':seed,'count':len(d['records']),'wall_seconds':d['wall_seconds']})
'''
    return control.remote(code, timeout=60)


def fetch():
    host = configure()
    argv = ssh(host)
    code = "import json,hashlib;from pathlib import Path;b=Path('/root/nim19-test');items=[];state=json.loads((b/'seed-runner.json').read_text());names="+repr(FILES)+"\nif state.get('stage') not in ['complete','failed']:names=[n for n in names if n.startswith('paired-reference-seed-')]\nfor n in names:\n if (b/n).is_file():\n  d=(b/n).read_bytes();items.append(dict(file=n,bytes=len(d),sha256=hashlib.sha256(d).hexdigest()))\nprint(json.dumps(items))"
    result = subprocess.run([*argv, 'root@'+host['public_ipaddr'], 'python3 -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    if result.returncode:
        raise RuntimeError('manifest_failed')
    manifest = json.loads(result.stdout)
    for item in manifest:
        name = item['file']
        if name not in FILES or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise RuntimeError('unsafe_manifest')
        target = DEST/name
        result = subprocess.run(['rsync','-rlt','--checksum','--no-owner','--no-group','--no-perms','-e',shlex.join(argv),'root@'+host['public_ipaddr']+':/root/nim19-test/'+name,str(target)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        verified = result.returncode == 0 and hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'] and target.stat().st_size == item['bytes']
        print(json.dumps({'event':'source',**item,'verified':verified}),flush=True)
        if not verified:
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['create','status','deploy','start','progress','fetch','stop'])
    operation = parser.parse_args().operation
    if operation == 'create':return create()
    if operation == 'deploy':return deploy()
    if operation == 'start':return start()
    if operation == 'progress':return progress()
    if operation == 'fetch':return fetch()
    host = configure()
    if operation == 'stop':return control.stop_test_instance()
    print(json.dumps({k:host.get(k) for k in ['id','label','actual_status','cur_state','gpu_name','num_gpus']}))
    return 0


if __name__ == '__main__':
    try:sys.exit(main())
    except Exception as e:
        print(json.dumps({'event':'operation_failed','error_class':type(e).__name__}));sys.exit(1)
