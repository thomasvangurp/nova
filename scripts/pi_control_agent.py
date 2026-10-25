#!/usr/bin/env python3
"""Pi-owned, read-only consumer qualification; independent of submission jobs."""
from __future__ import annotations
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pi_shadow_receipt import collect
from scripts.promotion_gate import require, sha

VAST = '/opt/nova-nanobody/deploy/slot_host/.pixi/envs/default/bin/vastai'

FETCH = r'''
import base64,json,pathlib,re,time
root=pathlib.Path('/root')
path=root/'artifacts/control_qualification_request.json'
if not path.is_file():
 print(json.dumps({'status':'idle'})); raise SystemExit(0)
request=json.loads(path.read_text())
assert request['schema']=='sn68-control-pi-request-v1'
assert re.fullmatch('[0-9a-f]{40}',request['candidate_sha'])
assert re.fullmatch('[0-9a-f]{32}',request['nonce'])
if request['expires_at']<=time.time():
 print(json.dumps({'status':'idle'})); raise SystemExit(0)
assert request['created_at']<=time.time()
previous=root/'artifacts/control_pi_receipt.json'
if previous.is_file():
 receipt=json.loads(previous.read_text())
 if receipt.get('request_nonce')==request['nonce'] and receipt.get('candidate_sha')==request['candidate_sha']:
  print(json.dumps({'status':'idle'})); raise SystemExit(0)
artifact=root/'releases'/request['candidate_sha']/'pi_trials'/(request['nonce']+'.json')
assert artifact.is_file() and not artifact.is_symlink()
print(json.dumps({'status':'requested','request':request,'artifact':base64.b64encode(artifact.read_bytes()).decode()}))
'''

DELIVER = r'''
import json,pathlib,re,sys,time
root=pathlib.Path('/root')
receipt=json.load(sys.stdin)
request=json.loads((root/'artifacts/control_qualification_request.json').read_text())
assert receipt['candidate_sha']==request['candidate_sha'] and receipt['request_nonce']==request['nonce']
assert receipt['artifact_sha256']==request['artifact_sha256']
assert receipt['schema']=='sn68-pi-shadow-v1' and receipt['read_only'] is True and receipt['submission_attempts']==0
assert 0<=time.time()-receipt['verified_at']<900 and time.time()<request['expires_at']
assert re.fullmatch('[0-9a-f]{64}',receipt['consumer_sha256'])
output=root/'artifacts/control_pi_receipt.json'; pending=output.with_suffix('.next')
pending.write_text(json.dumps(receipt,indent=2)+'\n'); pending.chmod(0o600); pending.replace(output)
print(json.dumps({'status':'delivered'}))
'''


def endpoint(vast=VAST):
    result = subprocess.run([vast, 'show', 'instances', '--raw'], capture_output=True, text=True, timeout=40)
    require(result.returncode == 0, 'producer_inventory_unavailable')
    rows = json.loads(result.stdout)
    if isinstance(rows, dict): rows = rows.get('instances', [])
    matches = [r for r in rows if r.get('label') == 'nova-nanobody-pixi' and r.get('actual_status') == 'running']
    require(len(matches) == 1, 'producer_inventory_ambiguous')
    host, port = matches[0]['ssh_host'], int(matches[0]['ssh_port'])
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host) and 1 <= port <= 65535,
            'invalid_producer_endpoint')
    return host, port


def remote(host, port, script, payload=None):
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
        '-o', 'StrictHostKeyChecking=accept-new', '-p', str(port), 'root@' + host,
        'python3 -c ' + shlex.quote(script)], input=payload, capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, 'qualification_transport_failed')
    return json.loads(result.stdout)


def tick(*, vast=VAST, consumer_root=Path('/opt/nova-submit')):
    host, port = endpoint(vast)
    message = remote(host, port, FETCH)
    if message['status'] == 'idle':
        return {'status': 'idle', 'checked_at': time.time()}
    request = message['request']
    require(request['schema'] == 'sn68-control-pi-request-v1'
            and sha(request['candidate_sha'], 40) and sha(request['nonce'], 32)
            and request['created_at'] <= time.time() < request['expires_at'], 'invalid_qualification_request')
    raw = base64.b64decode(message['artifact'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == request['artifact_sha256'], 'qualification_artifact_mismatch')
    artifact = json.loads(raw)
    uids = [int(row['uid']) for row in artifact['submissions']]
    with tempfile.TemporaryDirectory(prefix='nova-control-proof-') as temporary:
        trial = Path(temporary) / 'trial.json'; trial.write_bytes(raw); trial.chmod(0o600)
        receipt = collect(trial, consumer_root, revision=request['candidate_sha'], target=request['target'],
                          epoch=request['epoch'], uids=uids)
    receipt['request_nonce'] = request['nonce']
    require(remote(host, port, DELIVER, json.dumps(receipt))['status'] == 'delivered', 'receipt_not_delivered')
    return {'status': 'verified', 'revision': request['candidate_sha'], 'epoch': request['epoch'],
            'accepted_entries': receipt['accepted_entries'], 'submission_attempts': 0, 'checked_at': time.time()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--vast', default=VAST)
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with (args.state / 'agent.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = tick(vast=args.vast)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        result = {'status': 'qualification_unavailable', 'checked_at': time.time()}
    pending = args.state / 'health.next'
    pending.write_text(json.dumps(result, indent=2) + '\n'); pending.replace(args.state / 'health.json')
    print(json.dumps(result))
    return 0 if result['status'] in {'idle', 'verified'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
