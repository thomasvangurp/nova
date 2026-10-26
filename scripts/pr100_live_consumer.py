"""Restore current-rule nanobody consumption without changing installed signing.

The live Pi has independent molecule/deadline fixes. Patch its exact source,
retain those behaviors, and validate PR100 evidence before UID allocation.
This module only prepares bytes; it never installs or submits anything.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELD_CONSUMER_SHA256 = 'c25d21e2fa8bf62393e4db67f04a47041a274a139eb525ae0542cae5d15a2a3f'
LIVE_MARKER = '# PR100_CURRENT_RULE_LIVE_V1'


def live_sequences(path, epoch, network, netuid, allow_stale, crew):
    """Reject a bad nanobody artifact without discarding the molecule half."""
    try:
        from nanobody_opt import rules
        if epoch < rules.ACTIVATION_EPOCH:
            return sequences_by_uid(_load(path), epoch, allow_stale=allow_stale, crew=crew)
        artifact = _load(path)
        import bittensor as bt
        attestation = attest_epoch(bt.Subtensor(network=network), epoch, netuid)
        print(json.dumps(dict(event='pr100_chain_attestation', **attestation)), flush=True)
        return sequences_by_uid(artifact, epoch, allow_stale=False, crew=crew,
            epoch_start_block_hash=attestation['epoch_start_block_hash'],
            epoch_length=attestation['epoch_length'])
    except Exception as error:
        if epoch < 25955:
            raise
        print(json.dumps(dict(event='pr100_nanobody_rejected',
            error_type=type(error).__name__,
            action='retain independently valid molecule submissions')), flush=True)
        return {}


def _function(text, name):
    found = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise ValueError('required source function missing or ambiguous')
    return ast.get_source_segment(text, found[0])


def patch_live_consumer(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != HELD_CONSUMER_SHA256:
        raise ValueError('installed held consumer changed')
    text = source.decode()
    original = _function(text, 'sequences_by_uid')
    old = '''    # PR100_QUALIFICATION_HOLD_V1
    if epoch >= 25955:
        print(json.dumps({"event": "pr100_qualification_hold", "epoch": epoch,
                          "action": "withhold unqualified nanobodies"}), flush=True)
        return {}
'''
    new = '''    # PR100_CURRENT_RULE_LIVE_V1
    from nanobody_opt import rules
    if epoch >= rules.ACTIVATION_EPOCH:
        artifact = rules.verified_fleet(artifact, epoch, epoch_start_block_hash, epoch_length)
        allow_stale = False
        if not artifact["sequence_pool"]:
            return {}
'''
    signature = 'crew=None) -> dict[int, str]:'
    if original.count(old) != 1 or original.count(signature) != 1:
        raise ValueError('unexpected held allocator')
    allocator = original.replace(old, new, 1).replace(signature,
        'crew=None, epoch_start_block_hash=None, epoch_length=None) -> dict[int, str]:', 1)
    tracked = (ROOT / 'deploy/submit_host/pi_submit.py').read_text()
    helper = _function(Path(__file__).read_text(), 'live_sequences')
    text = text.replace(original, _function(tracked, 'attest_epoch') + '\n\n\n' + helper + '\n\n\n' + allocator, 1)
    main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id == 'sequences']
    if len(calls) != 1:
        raise ValueError('unexpected installed allocation route')
    before = ast.get_source_segment(text, calls[0])
    after = '''sequences = live_sequences(Path(arguments.nanobody), arguments.epoch,
                               arguments.network, arguments.netuid,
                               arguments.allow_stale_nanobody, crew)'''
    text = text.replace(before, after, 1)
    # The PR100 local artifact is delivered by the Pi-owned live collector.
    # Keep the legacy Vast pull disabled so an old producer cannot overwrite it.
    compile(text, '<pr100-live-consumer>', 'exec')
    return text.encode()
