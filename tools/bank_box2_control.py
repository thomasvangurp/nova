#!/usr/bin/env python3
"""SECOND dedicated bank box. Credentials runtime-only.

Identical role to bank_box_control, separate label/manifest/guard so the two can
never be confused. Rented because the NIM box has been stuck on
`resources_unavailable` for hours with no ETA; a second L40S halves the
wall-clock to 150,000 molecules at the same cost per molecule. Same GPU as the
first box, whose throughput is measured at 2.834 s/molecule.

Separate from the NIM epoch box in every respect -- its own label, its own
manifest, its own guard -- because the two must never be confused. This box runs
the vendored OSS Boltz worker, not NIM: bank building is not time-bound, so the
2.1x throughput penalty (2.12 s/molecule measured against NIM's 1.01) buys us
out of provisioning and running the NIM container entirely.

Rationale for the box itself: best-20 portfolio score grows as
0.0818*ln(bank_size) + 0.8344, i.e. +0.057 per doubling, so reaching the top
operators' 1.7603 needs roughly 82,000 exactly-scored molecules. At 4x L40S
list price that is the cheapest per-molecule option that does not require NIM.
L40S Boltz throughput is UNMEASURED; the first run benchmarks it before any
long commitment.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

import nim19_test_control as base
from nim19_seed_control import api as provider_api

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/bank_box2'
MANIFEST = DEST / 'instance.json'
ROUTE = DEST / 'ssh-route.json'
LABEL = 'nova-compound-bank2-l40s-20260919'
OFFER = 48802920
GPUS = 4
#: Production, the NIM epoch box, and every OTHER box in the fleet. The
#: guard above already requires the id, label and GPU count to match this
#: module's own manifest, so this is the second line: a manifest that has
#: been edited, swapped or restored from another box's backup still cannot
#: aim a stop or a destroy at a sibling's disk. tests/test_bank_sync_boxes.py
#: holds every module to naming the others.
FORBIDDEN = {51125470, 51335750, 51598530, 51676703}


def api(*a, **k):
    for attempt in range(4):
        try:
            return provider_api(*a, **k)
        except Exception as error:
            if attempt == 3:
                raise RuntimeError('provider_api_unavailable') from None
            time.sleep(20)


def guarded(host, saved):
    if (host.get('id') != saved['id'] or host.get('label') != LABEL
            or host.get('num_gpus') != GPUS or host['id'] in FORBIDDEN):
        raise RuntimeError('instance_guard_failed')
    return host


def instance(fresh=False):
    saved = json.loads(MANIFEST.read_text())
    if not fresh and ROUTE.is_file() and time.time() - ROUTE.stat().st_mtime < 300:
        return guarded(json.loads(ROUTE.read_text()), saved)
    _, data = api('instances/?owner=me')
    host = next(h for h in data.get('instances', []) if h.get('id') == saved['id'])
    guarded(host, saved)
    route = {k: host.get(k) for k in ['id', 'label', 'num_gpus', 'gpu_name',
                                      'cur_state', 'actual_status', 'public_ipaddr']}
    route['ports'] = {'22/tcp': [{'HostPort': p['HostPort']}
                                 for p in host.get('ports', {}).get('22/tcp', [])]}
    temp = ROUTE.with_suffix('.partial'); temp.write_text(json.dumps(route)); temp.replace(ROUTE)
    return route


def ssh(host):
    mapping = host.get('ports', {}).get('22/tcp') or []
    if not mapping:
        raise RuntimeError('direct_ssh_not_ready')
    return ['ssh', '-F', '/dev/null', '-i', base.SSH_KEY, '-p', str(mapping[0]['HostPort']),
            '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'StrictHostKeyChecking=accept-new', '-o', 'LogLevel=ERROR']


def create():
    if MANIFEST.exists():
        print(json.dumps(dict(event='existing', id=instance()['id']))); return 0
    _, data = api('instances/?owner=me')
    if [h for h in data.get('instances', []) if h.get('label') == LABEL]:
        raise RuntimeError('existing_label_requires_review')
    query = {'limit': 100, 'type': 'on-demand', 'num_gpus': {'eq': GPUS},
             'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
             'gpu_ram': {'gte': 45000}, 'disk_space': {'gte': 200}}
    _, data = api('bundles/', query, 'POST')
    offer = next((o for o in data.get('offers', []) if o.get('id') == OFFER), None)
    if offer is None:
        print(json.dumps(dict(event='offer_unavailable', offer=OFFER))); return 1
    if not (offer.get('num_gpus') == GPUS and offer.get('gpu_ram', 0) >= 45000
            and offer.get('dph_total', 999) <= 3.6 and offer.get('direct_port_count', 0) > 0):
        raise RuntimeError('offer_guard_failed')
    status, data = api('asks/' + str(OFFER) + '/',
                       dict(image='vastai/base-image:cuda-13.2.1-auto', label=LABEL,
                            disk=250, runtype='ssh_direct', target_state='running',
                            cancel_unavail=True), 'PUT')
    if data.get('success') is not True or not isinstance(data.get('new_contract'), int):
        print(json.dumps(dict(event='create_failed', http_status=status,
                              error=data.get('error'), msg=str(data.get('msg'))[:120]))); return 1
    DEST.mkdir(parents=True, exist_ok=True)
    saved = dict(id=data['new_contract'], label=LABEL, offer_id=OFFER, gpu_count=GPUS,
                 gpu_name=offer.get('gpu_name'), created_unix=time.time(),
                 quoted_dph=offer['dph_total'], purpose='offline molecule bank building',
                 scorer='oss_boltz_reference', off_chain=True,
                 live_submission_authorized=False, lease_safety_seconds=7200)
    MANIFEST.write_text(json.dumps(saved, indent=2))
    print(json.dumps(dict(event='created', id=saved['id'], num_gpus=GPUS,
                          gpu_name=saved['gpu_name'], quoted_dph=saved['quoted_dph'])))
    return 0


def status():
    host = instance(fresh=True)
    print(json.dumps({k: host.get(k) for k in
                      ['id', 'label', 'num_gpus', 'gpu_name', 'cur_state', 'actual_status']}))
    print(json.dumps(dict(ssh_mapped=bool(host.get('ports', {}).get('22/tcp')))))
    return 0


def stop():
    host = instance(fresh=True)
    if host.get('cur_state') == 'stopped':
        print(json.dumps(dict(event='stopped', id=host['id']))); return 0
    s, d = api('instances/' + str(host['id']) + '/', dict(state='stopped'), 'PUT')
    print(json.dumps(dict(event='stop_requested', http_status=s, passed=d.get('success') is True)))
    return 0 if d.get('success') is True else 1


def destroy():
    """Only ever on explicit instruction: this deletes the disk."""
    host = instance(fresh=True)
    s, d = api('instances/' + str(host['id']) + '/', None, 'DELETE')
    print(json.dumps(dict(event='destroy_requested', http_status=s,
                          passed=d.get('success') is True)))
    return 0 if d.get('success') is True else 1


def watch():
    saved = json.loads(MANIFEST.read_text())
    deadline = saved['created_unix'] + saved['lease_safety_seconds']
    while True:
        try:
            if time.time() >= deadline:
                if stop() == 0:
                    return 0
            elif instance(fresh=True).get('cur_state') == 'stopped':
                return 0
        except Exception as error:
            print(json.dumps(dict(event='source', summary={'watch_retry': type(error).__name__})), flush=True)
        time.sleep(30 if time.time() >= deadline else min(30, max(1, deadline - time.time())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['create', 'status', 'stop', 'destroy', 'watch'])
    a = p.parse_args()
    return dict(create=create, status=status, stop=stop, destroy=destroy, watch=watch)[a.operation]()


if __name__ == '__main__':
    sys.exit(main())
