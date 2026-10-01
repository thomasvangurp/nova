#!/usr/bin/env python3
"""The dedicated chain-writing box: one machine commits every uid we hold.

Separate from both bank boxes and from production in every respect -- its own
label, its own manifest, its own guard -- because this is the only box that
signs. Credentials are runtime-only and never enter an argument or a log.

Why it exists: `provider.result_path` keys on the reaction alone and the fleet
allocator binds ONE release to the live uid+hotkey roster, so every uid has to
be committed from the machine holding that roster. Splitting the fleet across
machines is how two writers end up racing for one uid, which this repository
has already paid for once (the chain writer overwriting uid 19).

Why 4 GPUs for a box that only signs: `ansible/live.yml` asserts
`gpu_count >= 4`, and the pre-submission rescore has to run here. Rescore
coverage decays as the bank grows -- 109 of the current top-132 -- so the
three-pass mean has to be refreshed against the epoch's own bank, not shipped
in from a box that rescored yesterday. 300 molecules x 3 passes is about 900
Boltz runs, roughly a quarter of an epoch on four Ampere cards.

Why RTX 3090 and not the cheaper Turing/Volta offers: Boltz's fused kernels
need Ampere, and bf16 does not exist below it.

THE LEASE IS NOT SILENTLY EXTENDED. `watch` stops the instance at the manifest
deadline. Extending is an explicit operation with an explicit horizon.

WALLETS ARE NOT PLACED BY THIS SCRIPT. It never touches key material; the
operator stages hotkeys and the playbook asserts no coldkey is present.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nim19_test_control as base
from nim19_seed_control import api as provider_api

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/submit_box'
MANIFEST = DEST / 'instance.json'
ROUTE = DEST / 'ssh-route.json'
LABEL = 'nova-compound-submit-20260919'
GPUS = 4
MAXIMUM_DPH = 1.30
#: Ampere or newer. Boltz's fused kernels do not exist below it and neither
#: does bf16, so the cheap Turing and Volta four-card offers are not options
#: however good the price looks. Mining cards are excluded outright.
ALLOWED_GPUS = ('RTX 3090', 'RTX 3090 Ti', 'RTX 4090', 'RTX 5090', 'RTX A5000',
                'RTX A6000', 'RTX 6000Ada', 'RTX PRO 6000', 'L40S', 'A100', 'H100')
#: Taiwanese hosts failed to reach Docker Hub on two prior rentals.
EXCLUDED_REGIONS = ('Taiwan',)
#: This box misses an epoch when it is down, so reliability outranks price.
MINIMUM_RELIABILITY = 0.995
#: Production, the NIM epoch box, and every OTHER box in the fleet. The
#: guard above already requires the id, label and GPU count to match this
#: module's own manifest, so this is the second line: a manifest that has
#: been edited, swapped or restored from another box's backup still cannot
#: aim a stop or a destroy at a sibling's disk. tests/test_bank_sync_boxes.py
#: holds every module to naming the others.
FORBIDDEN = {51125470, 51335750, 51598530, 51620065}
DEFAULT_LEASE_SECONDS = 86400


def api(*a, **k):
    for attempt in range(4):
        try:
            return provider_api(*a, **k)
        except Exception:
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
    temp = ROUTE.with_suffix('.partial')
    temp.write_text(json.dumps(route))
    temp.replace(ROUTE)
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


def _acceptable(offer):
    """Every property this box is being chosen for, as one predicate.

    Offer ids on this provider rotate every few minutes, so pinning the id
    found by a survey does not survive to the moment of spending: the first
    attempt here refused with `offer_unavailable` on an id that had existed
    ninety seconds earlier. The policy is pinned instead of the id, and the
    manifest records which offer it actually took.
    """
    name = str(offer.get('gpu_name') or '')
    region = str(offer.get('geolocation') or '')
    return (offer.get('num_gpus') == GPUS
            and any(name.startswith(allowed) for allowed in ALLOWED_GPUS)
            and not any(bad in region for bad in EXCLUDED_REGIONS)
            and offer.get('gpu_ram', 0) >= 23000
            and offer.get('disk_space', 0) >= 220
            and offer.get('dph_total', 999) <= MAXIMUM_DPH
            and offer.get('reliability2', 0) >= MINIMUM_RELIABILITY
            and offer.get('direct_port_count', 0) >= 2
            and offer.get('cpu_cores_effective', 0) >= 8 * GPUS)


SURVEY_FIELDS = ['id', 'gpu_name', 'num_gpus', 'gpu_ram', 'dph_total', 'disk_space',
                 'reliability2', 'direct_port_count', 'inet_down',
                 'cpu_cores_effective', 'geolocation', 'cuda_max_good']


def _shortlist():
    """Acceptable offers, most reliable first, price breaking the tie."""
    _, data = api('bundles/', {'limit': 400, 'type': 'on-demand',
                               'num_gpus': {'eq': GPUS}, 'verified': {'eq': True},
                               'rentable': {'eq': True}, 'rented': {'eq': False},
                               'gpu_ram': {'gte': 23000}, 'disk_space': {'gte': 220}},
                  'POST')
    offers = [o for o in data.get('offers', []) if _acceptable(o)]
    offers.sort(key=lambda o: (-round(o.get('reliability2', 0), 4), o['dph_total']))
    return offers


def survey():
    """What create() would take, without spending anything."""
    offers = _shortlist()
    print(json.dumps(dict(event='survey', acceptable=len(offers))))
    for offer in offers[:10]:
        print(json.dumps({k: offer.get(k) for k in SURVEY_FIELDS}))
    return 0 if offers else 1


def create():
    if MANIFEST.exists():
        print(json.dumps(dict(event='existing', id=instance()['id'])))
        return 0
    _, data = api('instances/?owner=me')
    if [h for h in data.get('instances', []) if h.get('label') == LABEL]:
        raise RuntimeError('existing_label_requires_review')
    for offer in _shortlist():
        if not _acceptable(offer):           # re-checked at the moment of spending
            continue
        status, data = api('asks/' + str(offer['id']) + '/',
                           dict(image='vastai/base-image:cuda-13.2.1-auto', label=LABEL,
                                disk=250, runtype='ssh_direct', target_state='running',
                                cancel_unavail=True), 'PUT')
        if data.get('success') is not True or not isinstance(data.get('new_contract'), int):
            print(json.dumps(dict(event='offer_declined', offer=offer['id'],
                                  http_status=status,
                                  msg=str(data.get('msg'))[:120])))
            continue
        DEST.mkdir(parents=True, exist_ok=True)
        saved = dict(id=data['new_contract'], label=LABEL, offer_id=offer['id'],
                     gpu_count=GPUS, gpu_name=offer.get('gpu_name'),
                     geolocation=offer.get('geolocation'),
                     reliability=offer.get('reliability2'), created_unix=time.time(),
                     quoted_dph=offer['dph_total'],
                     purpose='chain writer for every uid; pre-submission rescore',
                     scorer='oss_boltz_reference', off_chain=False,
                     live_submission_authorized=True,
                     wallets_staged_by_operator=False,
                     lease_safety_seconds=DEFAULT_LEASE_SECONDS)
        MANIFEST.write_text(json.dumps(saved, indent=2))
        print(json.dumps(dict(event='created', id=saved['id'], offer=offer['id'],
                              num_gpus=GPUS, gpu_name=saved['gpu_name'],
                              geolocation=saved['geolocation'],
                              reliability=saved['reliability'],
                              quoted_dph=round(saved['quoted_dph'], 4),
                              stops_at_utc=time.strftime(
                                  '%Y-%m-%dT%H:%M:%SZ',
                                  time.gmtime(saved['created_unix'] + DEFAULT_LEASE_SECONDS)))))
        return 0
    print(json.dumps(dict(event='no_acceptable_offer')))
    return 1


#: The fields ansible/inventory/vast_nanobody.py reads, and nothing else. The
#: provider's instance document also carries `jupyter_token` and ssh key
#: material; writing the raw document to disk so the inventory could read it
#: would put both in a file and then in `ansible-inventory` output. The
#: document is rebuilt from an allowlist instead.
INVENTORY_FIELDS = ('id', 'label', 'actual_status', 'num_gpus', 'gpu_name',
                    'machine_id', 'public_ipaddr')


def inventory(destination=None):
    """Write a minimal instances document for NOVA_VAST_INSTANCES_FILE.

    The compound fleet's dynamic inventory resolves a label through the vastai
    CLI, which is authenticated to the account holding production. This box is
    in a different account, so the label is invisible there. Rather than move
    an API key between accounts, this emits the one document the inventory
    already knows how to read.
    """
    _, data = api('instances/?owner=me')
    saved = json.loads(MANIFEST.read_text())
    host = next((h for h in data.get('instances', []) if h.get('id') == saved['id']), None)
    if host is None:
        print(json.dumps(dict(event='instance_not_found', id=saved['id'])), file=sys.stderr)
        return 1
    guarded(host, saved)
    document = {field: host.get(field) for field in INVENTORY_FIELDS}
    document['ports'] = {'22/tcp': [{'HostPort': entry['HostPort']}
                                    for entry in (host.get('ports') or {}).get('22/tcp', [])
                                    if entry.get('HostPort')]}
    path = Path(destination) if destination else (DEST / 'instances.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps({'instances': [document]}, indent=1))
    temporary.replace(path)
    print(json.dumps(dict(event='inventory_written', path=str(path),
                          label=document['label'], id=document['id'],
                          actual_status=document['actual_status'],
                          ssh_port=(document['ports']['22/tcp'] or [{}])[0].get('HostPort'))))
    return 0


def status():
    host = instance(fresh=True)
    saved = json.loads(MANIFEST.read_text())
    print(json.dumps({k: host.get(k) for k in
                      ['id', 'label', 'num_gpus', 'gpu_name', 'cur_state', 'actual_status']}))
    remaining = saved['created_unix'] + saved['lease_safety_seconds'] - time.time()
    print(json.dumps(dict(ssh_mapped=bool(host.get('ports', {}).get('22/tcp')),
                          wallets_staged=saved.get('wallets_staged_by_operator'),
                          lease_hours_remaining=round(remaining / 3600, 2))))
    return 0


def stop():
    host = instance(fresh=True)
    if host.get('cur_state') == 'stopped':
        print(json.dumps(dict(event='stopped', id=host['id'])))
        return 0
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


def extend(hours):
    """Move the stop deadline, on an explicit horizon and never implicitly.

    A chain writer that keeps itself alive is a chain writer that bills
    unattended, so the number is required and the new deadline is printed.
    """
    if hours is None:
        print(json.dumps(dict(event='extend_requires_hours')), file=sys.stderr)
        return 2
    saved = json.loads(MANIFEST.read_text())
    saved['lease_safety_seconds'] = int(
        max(0.0, time.time() - saved['created_unix']) + float(hours) * 3600)
    MANIFEST.write_text(json.dumps(saved, indent=2))
    print(json.dumps(dict(event='extended', id=saved['id'], added_hours=float(hours),
                          stops_at_utc=time.strftime(
                              '%Y-%m-%dT%H:%M:%SZ',
                              time.gmtime(saved['created_unix'] + saved['lease_safety_seconds'])))))
    return 0


def watch():
    saved = json.loads(MANIFEST.read_text())
    while True:
        saved = json.loads(MANIFEST.read_text())   # re-read: extend() moves it
        deadline = saved['created_unix'] + saved['lease_safety_seconds']
        try:
            if time.time() >= deadline:
                if stop() == 0:
                    return 0
            elif instance(fresh=True).get('cur_state') == 'stopped':
                return 0
        except Exception as error:
            print(json.dumps(dict(event='source', summary={'watch_retry': type(error).__name__})),
                  flush=True)
        time.sleep(30 if time.time() >= deadline else min(30, max(1, deadline - time.time())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation',
                   choices=['survey', 'create', 'status', 'inventory', 'stop', 'destroy',
                            'extend', 'watch'])
    p.add_argument('--out', default=None,
                   help='inventory only: where to write the instances document')
    p.add_argument('--hours', type=float, default=None,
                   help='extend only: hours from now to move the stop deadline to')
    a = p.parse_args()
    if a.operation == 'extend':
        return extend(a.hours)
    if a.operation == 'inventory':
        return inventory(a.out)
    return dict(survey=survey, create=create, status=status, stop=stop,
                destroy=destroy, watch=watch)[a.operation]()


if __name__ == '__main__':
    sys.exit(main())
