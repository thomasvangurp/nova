#!/usr/bin/env python3
"""Provision and run a fleet of bank-building boxes for ONE reaction.

Why a new module rather than another copy of bank_box_control: that one pins a
single OFFER id and a single LABEL, so it can only ever hold one box. A second
reaction needs several boxes at once, and each must be individually
addressable -- a stop or a destroy that can reach a sibling's disk is how a
bank gets deleted.

Every box here is off-chain. They hold no wallet, sign nothing, and the
manifest records that, so a box built for banking can never be mistaken for a
submitter.

The expected value that justifies this: deepening reaction 3 from 58.7k to
2x buys +2.7 points of epochs won, while the SAME 58.7k molecules spent on a
new reaction buys +12.0. Reaction 5 is the target because its field is thin --
10 uids score, the same as reaction 3, against 15-19 elsewhere.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from nim19_seed_control import api as provider_api      # noqa: E402

REACTION = 5
#: -nim, because the first five boxes under -bank were RTX 3090s that
#: cannot run the NIM image; they are stopped, not destroyed, so their
#: labels are still taken.
PREFIX = 'nova-rxn5-nim'
#: Buy GPU-HOURS, not boxes. Surveyed 2026-09-21: the cheapest four-card
#: offer is $0.165/GPU-hr while two-card offers start at $0.125, and pinning
#: num_gpus==4 left just 2 acceptable offers out of 55. Banking parallelises
#: across boxes as easily as across cards, so the card count per box is not
#: worth paying for.
GPU_CHOICES = (2, 4, 8)
MAXIMUM_DPH_PER_GPU = 0.70
#: NIM's boltz2 image needs ~48 GB. docs/NIM19_BENCHMARK_20260917.md ran it on
#: an RTX 6000 Ada with 49,140 MB and states plainly that a low between-request
#: VRAM reading "does not prove that unsupported 24 GB cards can run the
#: workload safely". A 23 GB floor let RTX 3090s through and five were rented
#: before this was caught. Turing (Q RTX 8000) and CMP mining cards clear the
#: memory bar but not the fused-kernel or policy bars, so they are named out.
MINIMUM_VRAM_MB = 46000
ALLOWED_GPUS = ('RTX A6000', 'RTX 6000Ada', 'RTX PRO 6000', 'L40', 'L40S',
                'A40', 'A100', 'A100X', 'H100', 'H200')
EXCLUDED_REGIONS = ('Taiwan',)
#: Banking is restartable and off-chain, so a box that drops costs hours, not
#: an epoch. Price matters more here than it does for the submitter.
MINIMUM_RELIABILITY = 0.97
#: Production, the submitter, the Pi's peers, and every box this module did
#: not create. Enumerated so an edited or restored manifest still cannot aim a
#: destroy at a sibling.
FORBIDDEN = {51125470, 51335750, 51598530, 51620065, 51676703}
DEST = ROOT / 'data/rxn5_bank'


def api(*a, **k):
    for attempt in range(4):
        try:
            return provider_api(*a, **k)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def manifests():
    return sorted(DEST.glob('instance-*.json'))


def held_ids():
    out = set()
    for path in manifests():
        try:
            out.add(int(json.loads(path.read_text())['id']))
        except Exception:
            continue
    return out


def per_gpu(offer):
    return offer.get('dph_total', 999) / max(1, offer.get('num_gpus', 1))


def acceptable(offer):
    return (offer.get('num_gpus') in GPU_CHOICES
            and offer.get('gpu_name') in ALLOWED_GPUS
            and offer.get('gpu_ram', 0) >= MINIMUM_VRAM_MB
            and offer.get('disk_space', 0) >= 220
            and per_gpu(offer) <= MAXIMUM_DPH_PER_GPU
            and offer.get('reliability2', 0) >= MINIMUM_RELIABILITY
            and offer.get('direct_port_count', 0) > 0
            and offer.get('geolocation') not in EXCLUDED_REGIONS)


def offers():
    found = []
    for count in GPU_CHOICES:
        _, data = api('bundles/', {'limit': 300, 'type': 'on-demand',
                                   'num_gpus': {'eq': count},
                                   'verified': {'eq': True},
                                   'rentable': {'eq': True}, 'rented': {'eq': False},
                                   'gpu_ram': {'gte': MINIMUM_VRAM_MB},
                                   'disk_space': {'gte': 220}}, 'POST')
        found += [o for o in data.get('offers', []) if acceptable(o)]
    # Cheapest first, but never below the reliability floor: banking is a
    # multi-hour job and a host that dies at hour six costs the whole run.
    found.sort(key=lambda o: (per_gpu(o), -o.get('reliability2', 0)))
    return found


def survey(arguments):
    rows = offers()[:arguments.count * 3]
    print(json.dumps(dict(event='survey', acceptable=len(offers()),
                          cheapest=[dict(id=o['id'], gpu=o.get('gpu_name'),
                                         n=o.get('num_gpus'),
                                         dph=round(o['dph_total'], 3),
                                         per_gpu=round(per_gpu(o), 3),
                                         rel=round(o.get('reliability2', 0), 3),
                                         where=o.get('geolocation'))
                                    for o in rows[:8]],
                          plan_gpus=sum(o['num_gpus'] for o in rows[:arguments.count]),
                          would_spend_per_hour=round(
                              sum(o['dph_total'] for o in rows[:arguments.count]), 3)),
                     indent=1))
    return 0


def create(arguments):
    existing = manifests()
    if len(existing) >= arguments.count:
        print(json.dumps(dict(event='already_held', boxes=len(existing))))
        return 0
    _, data = api('instances/?owner=me')
    live = {h.get('label') for h in data.get('instances', [])}
    pool = [o for o in offers() if o['id'] not in FORBIDDEN]
    DEST.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(len(existing), arguments.count):
        label = f'{PREFIX}-{index + 1}'
        if label in live:
            print(json.dumps(dict(event='label_in_use', label=label)))
            continue
        if not pool:
            print(json.dumps(dict(event='offers_exhausted', made=len(made))))
            break
        offer = pool.pop(0)
        status, reply = api('asks/' + str(offer['id']) + '/',
                            dict(image='vastai/base-image:cuda-13.2.1-auto',
                                 label=label, disk=250, runtype='ssh_direct',
                                 target_state='running', cancel_unavail=True), 'PUT')
        if reply.get('success') is not True or not isinstance(reply.get('new_contract'), int):
            print(json.dumps(dict(event='create_failed', label=label,
                                  http_status=status,
                                  msg=str(reply.get('msg'))[:120])))
            continue
        saved = dict(id=reply['new_contract'], label=label, offer_id=offer['id'],
                     gpu_count=offer['num_gpus'], gpu_name=offer.get('gpu_name'),
                     quoted_dph=offer['dph_total'], created_unix=time.time(),
                     reaction=REACTION, purpose='offline molecule bank building',
                     scorer='nim', off_chain=True,
                     live_submission_authorized=False,
                     lease_safety_seconds=7200)
        (DEST / f'instance-{index + 1}.json').write_text(json.dumps(saved, indent=2))
        made.append(saved)
        print(json.dumps(dict(event='created', label=label, id=saved['id'],
                              gpu_name=saved['gpu_name'],
                              dph=round(saved['quoted_dph'], 4))))
    print(json.dumps(dict(event='create_complete', made=len(made),
                          total=len(manifests()),
                          spend_per_hour=round(sum(
                              json.loads(p.read_text())['quoted_dph']
                              for p in manifests()), 3))))
    return 0


def status(arguments):
    _, data = api('instances/?owner=me')
    live = {h['id']: h for h in data.get('instances', [])}
    rows = []
    for path in manifests():
        held = json.loads(path.read_text())
        host = live.get(held['id'], {})
        rows.append(dict(label=held['label'], id=held['id'],
                         gpu=held.get('gpu_name'),
                         state=host.get('actual_status') or host.get('cur_state'),
                         dph=round(held.get('quoted_dph', 0), 4),
                         hours=round((time.time() - held['created_unix']) / 3600, 2)))
    print(json.dumps(dict(event='status', boxes=rows,
                          spend_per_hour=round(sum(r['dph'] for r in rows), 3)),
                     indent=1))
    return 0


def stop(arguments):
    for path in manifests():
        held = json.loads(path.read_text())
        if held['id'] in FORBIDDEN:
            raise RuntimeError(f"refusing to stop a forbidden id: {held['id']}")
        api('instances/' + str(held['id']) + '/',
            dict(state='stopped'), 'PUT')
        print(json.dumps(dict(event='stopped', label=held['label'], id=held['id'])))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('operation', choices=['survey', 'create', 'status', 'stop'])
    parser.add_argument('--count', type=int, default=5,
                        help='how many boxes the fleet should hold')
    arguments = parser.parse_args(argv)
    return dict(survey=survey, create=create, status=status, stop=stop)[
        arguments.operation](arguments)


if __name__ == '__main__':
    raise SystemExit(main())
