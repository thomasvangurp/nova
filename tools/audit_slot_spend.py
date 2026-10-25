#!/usr/bin/env python3
"""Reconcile alleged slot burns only against exact failed on-chain calls.

No signing, transaction submission, key creation, or staking occurs here.
--apply appends corrections and requires the slot buyer to be stopped first.
Original purchase records remain intact; transaction fees are not refunded.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt import slot_manager as sm


def decoded(value):
    return getattr(value, 'value', value)


def error_name(substrate, event):
    attributes = event.get('attributes', {})
    dispatch = attributes.get('dispatch_error') if isinstance(attributes, dict) else None
    module = dispatch.get('Module') if isinstance(dispatch, dict) else None
    if isinstance(module, dict) and 'index' in module and 'error' in module:
        raw = module['error']
        index = (int(raw[2:4], 16) if isinstance(raw, str) and raw.startswith('0x')
                 else raw[0] if isinstance(raw, list) and raw else None)
        if index is not None:
            metadata = decoded(substrate.metadata.get_module_error(module['index'], index))
            name = metadata.get('name') if isinstance(metadata, dict) else None
            if isinstance(name, str) and re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,79}', name):
                return name
    return 'unclassified_chain_error'


def registration_calls(substrate, block):
    block_hash = substrate.get_block_hash(block)
    receipts = []
    for item in substrate.get_events(block_hash=block_hash):
        row = decoded(item)
        event = row.get('event', row)
        if (event.get('module_id') == 'System'
                and event.get('event_id') in {'ExtrinsicFailed', 'ExtrinsicSuccess'}
                and type(row.get('extrinsic_idx')) is int):
            failed = event['event_id'] == 'ExtrinsicFailed'
            receipts.append((row['extrinsic_idx'], failed,
                             error_name(substrate, event) if failed else None))
    if not receipts:
        return []
    body = substrate.get_block(block_hash=block_hash)
    result = []
    for index, failed, error in receipts:
        value = decoded(body['extrinsics'][index])
        call = value.get('call', {})
        if (not isinstance(call, dict) or call.get('call_module') != 'SubtensorModule'
                or call.get('call_function') != 'burned_register'):
            continue
        arguments = {arg['name']: arg['value'] for arg in call.get('call_args', [])
                     if isinstance(arg, dict) and 'name' in arg and 'value' in arg}
        result.append({'block': block, 'block_hash': block_hash, 'extrinsic_index': index,
            'extrinsic_hash': value['extrinsic_hash'], 'signer': value.get('address'),
            'hotkey': arguments.get('hotkey'), 'netuid': arguments.get('netuid'),
            'failed': failed, 'error': error})
    return result


def audit(subtensor, wallet_factory, *, wallet_name, records, netuid=68, now=None):
    now = time.time() if now is None else now
    coldkey = str(wallet_factory(name=wallet_name).coldkeypub.ss58_address)
    effective = {sm.purchase_record_id(row) for row in sm.effective_purchase_records(records)}
    cache, proofs, unresolved, seen_receipts = {}, [], 0, set()
    attempts = 0
    for index, row in enumerate(records):
        if (row.get('action') != sm.BUY or row.get('executed') is not True
                or row.get('netuid') != netuid or row.get('coldkey_ss58') != coldkey
                or not 0 <= now - float(row.get('time', 0)) <= 86400
                or sm.purchase_record_id(row) not in effective):
            continue
        attempts += 1
        following = next((entry for entry in records[index + 1:]
                          if type(entry.get('block')) is int and entry['block'] > row['block']), None)
        if following is None or not 0 < following['block'] - row['block'] <= 120:
            unresolved += 1
            continue
        address = str(wallet_factory(name=wallet_name, hotkey=row['hotkey']).hotkey.ss58_address)
        matching = []
        for block in range(row['block'], following['block'] + 1):
            if block not in cache:
                if len(cache) >= 1000:
                    raise ValueError('audit_block_limit')
                cache[block] = registration_calls(subtensor.substrate, block)
            matching.extend(call for call in cache[block]
                            if call['signer'] == coldkey and call['hotkey'] == address
                            and call['netuid'] == netuid)
        # A retry that succeeded in the same window makes a failed receipt
        # insufficient to release a charge. Preserve every ambiguous charge.
        if len(matching) != 1 or not matching[0]['failed']:
            unresolved += 1
            continue
        call = matching[0]
        receipt_id = (call['block_hash'], call['extrinsic_index'])
        if receipt_id in seen_receipts:
            unresolved += 1
            continue
        seen_receipts.add(receipt_id)
        proof = {'schema': 'sn68-registration-chain-receipt-v1',
            'outcome': 'System.ExtrinsicFailed', 'call_module': 'SubtensorModule',
            'call_function': 'burned_register', 'netuid': netuid,
            'signer_matches': True, 'hotkey_matches': True,
            'block': call['block'], 'block_hash': call['block_hash'],
            'extrinsic_index': call['extrinsic_index'], 'extrinsic_hash': call['extrinsic_hash'],
            'observed_through_block': following['block'], 'error_code': call['error']}
        correction = {'kind': 'slot_spend_correction', 'at': now,
                      'purchase_record_id': sm.purchase_record_id(row), 'proof': proof}
        # Exercise the same conservative validator that the buyer will use.
        if sm.effective_purchase_records([row, correction]):
            raise ValueError('unusable_chain_receipt')
        proofs.append(correction)
    before = sm.spend_in_window(sm.executed_spends(records), now)
    after = sm.spend_in_window(sm.executed_spends([*records, *proofs]), now)
    return {'schema': 'sn68-slot-spend-audit-v1', 'created_at': now, 'netuid': netuid,
        'attempts_checked': attempts, 'failed_registrations_proven': len(proofs),
        'unresolved_attempts': unresolved, 'scanned_blocks': len(cache),
        'recorded_burn_before_tao': before, 'recorded_burn_after_tao': after,
        'corrected_fictitious_burn_tao': before - after, 'corrections': proofs,
        'errors': dict(collections.Counter(row['proof']['error_code'] for row in proofs)),
        'transactions_submitted': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision-log', type=Path, required=True)
    parser.add_argument('--wallet', default='mining_ops')
    parser.add_argument('--network', default='finney')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('audit_output_exists')
        if args.apply:
            for unit in ['nova-slot-manager.timer', 'nova-slot-manager.service']:
                state = subprocess.run(['systemctl', 'is-active', unit], capture_output=True,
                                       text=True, timeout=15)
                if state.returncode != 3 or state.stdout.strip() not in {'inactive', 'failed'}:
                    raise ValueError('stop_buyer_before_reconciling')
        with args.decision_log.with_suffix('.audit.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            records = sm.read_records(args.decision_log)
            # SDK diagnostics and signed/public chain payloads never reach logs.
            with io.StringIO() as private, contextlib.redirect_stdout(private), contextlib.redirect_stderr(private):
                bt = sm._import_bittensor()
                subtensor = (getattr(bt, 'Subtensor', None) or bt.subtensor)(network=args.network)
                result = audit(subtensor, getattr(bt, 'Wallet', None) or bt.wallet,
                               wallet_name=args.wallet, records=records)
            raw = json.dumps(result, indent=2) + '\n'
            with args.output.open('x') as output:
                args.output.chmod(0o600)
                output.write(raw)
            if args.apply:
                evidence_hash = hashlib.sha256(raw.encode()).hexdigest()
                for row in result['corrections']:
                    sm.append_record(args.decision_log, {**row, 'audit_sha256': evidence_hash})
            print(json.dumps({k: v for k, v in result.items() if k != 'corrections'}
                             | {'applied': args.apply}))
        return 0
    except Exception:
        print(json.dumps({'status': 'slot_spend_audit_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
