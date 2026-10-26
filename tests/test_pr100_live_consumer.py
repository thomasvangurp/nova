"""Control tests only: synthetic checkpoint checks are not native proof."""
import ast
import builtins
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from scripts.pr100_live_consumer import HELD_CONSUMER_SHA256, patch_live_consumer, patch_live_loop

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/fixtures/pr100/pi_consumer_held.py'


def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef)}


def load_candidate():
    source = patch_live_consumer(SOURCE.read_bytes())
    module = types.ModuleType('_live_pi_test')
    module.__file__ = str(ROOT / 'deploy/submit_host/pi_submit.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def test_live_patch_preserves_all_other_installed_functions():
    source = SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == HELD_CONSUMER_SHA256
    candidate = patch_live_consumer(source)
    before, after = functions(source), functions(candidate)
    for name in before.keys() - {'main', 'sequences_by_uid'}:
        assert before[name] == after[name], name
    assert set(after) - set(before) == {'attest_epoch', 'live_sequences'}
    assert b'PR100_QUALIFICATION_HOLD_V1' not in candidate
    assert b'PR100_CURRENT_RULE_LIVE_V1' in candidate
    with pytest.raises(ValueError):
        patch_live_consumer(candidate)
    with pytest.raises(ValueError):
        patch_live_consumer(source + b'\n')


def test_live_loop_patch_only_registers_new_stage_and_producer(tmp_path):
    source = (ROOT / 'tests/fixtures/pr100/pi_loop_deadline_fixed.py').read_bytes()
    candidate = patch_live_loop(source)
    assert functions(candidate) == functions(source)
    module = types.ModuleType('_live_loop_test')
    module.__file__ = str(ROOT / 'tools/pi_epoch_loop.py')
    exec(compile(candidate, module.__file__, 'exec'), module.__dict__)
    path = tmp_path / 'fleet.json'
    artifact = {'artifact_kind': 'nanobody-fleet', 'publication_stage': 'live_trial',
                'status': 'live_trial', 'producer': {'name': 'pr100_live_collector'},
                'submissions': [{'publication_stage': 'live_trial'}]}
    path.write_text(json.dumps(artifact))
    assert module.artifact_provenance(path)[0] is True
    artifact['producer']['name'] = 'unrecognized'
    path.write_text(json.dumps(artifact))
    assert module.artifact_provenance(path)[0] is False
    with pytest.raises(ValueError): patch_live_loop(candidate)


def test_live_allocation_uses_independent_chain_context(monkeypatch, tmp_path):
    module = load_candidate()
    artifact = tmp_path / 'artifact.json'
    artifact.write_text(json.dumps({'epoch': 25956}))
    context = {'epoch_start_block_hash': '0x' + 'a' * 64, 'epoch_length': 361}
    chain = object()
    monkeypatch.setitem(sys.modules, 'bittensor', types.SimpleNamespace(Subtensor=lambda **kw: chain))
    monkeypatch.setattr(module, 'attest_epoch', lambda c, e, n: context if c is chain and e == 25956 and n == 68 else None)
    calls = []
    def allocation(payload, epoch, **kw):
        calls.append((payload, epoch, kw))
        return {4: 'MEASURED'}
    monkeypatch.setattr(module, 'sequences_by_uid', allocation)
    assert module.live_sequences(artifact, 25956, 'finney', 68, True, [(4, 'owner')]) == {4: 'MEASURED'}
    assert calls[0][2] == {'allow_stale': False, 'crew': [(4, 'owner')], **context}


@pytest.mark.parametrize('failure', ['missing', 'malformed', 'chain', 'evidence', 'import'])
def test_live_failure_retains_independent_molecule_path(monkeypatch, tmp_path, failure):
    module = load_candidate()
    path = tmp_path / 'artifact.json'
    if failure != 'missing':
        path.write_text('{' if failure == 'malformed' else '{}')
    monkeypatch.setitem(sys.modules, 'bittensor', types.SimpleNamespace(Subtensor=lambda **kw: object()))
    def attest(*args):
        if failure == 'chain':
            raise RuntimeError('RPC unavailable')
        return {'epoch_start_block_hash': '0x' + 'a' * 64, 'epoch_length': 361}
    monkeypatch.setattr(module, 'attest_epoch', attest)
    monkeypatch.setattr(module, 'sequences_by_uid', lambda *a, **k: (_ for _ in ()).throw(ValueError('bad evidence')))
    if failure == 'import':
        original = builtins.__import__
        def broken(name, *args, **kwargs):
            if name == 'nanobody_opt':
                raise ImportError('rules unavailable')
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, '__import__', broken)
    assert module.live_sequences(path, 25956, 'finney', 68, True, []) == {}
    assert module.commit_batch([], None, None) == (0, 0, [])
