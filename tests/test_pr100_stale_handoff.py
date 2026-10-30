"""Replay recorded epoch-25956 evidence through the actual submitter modules.

Set NOVA_PI_MIRROR to include the separately maintained Pi code. The mirror
omits JSON configs, so those tests use this repo's pinned public rule config.
Only chain I/O and the final commit boundary are replaced; eligibility,
checkpoint validation, allocation, plaintext validation and main are real.
The 25957 chain draw is controlled test data, not a live-chain assertion.
"""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from tests.test_pr100_live_handoff_contract import build, replay


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['repo', 'pi'])
def runtime(request, replay, monkeypatch):
    artifact = build(replay)
    artifact['updated_at_unix'] = time.time() - 7200
    root = ROOT
    if request.param == 'pi':
        if not os.environ.get('NOVA_PI_MIRROR'):
            pytest.skip('set NOVA_PI_MIRROR to exercise the diverged Pi files')
        root = Path(os.environ['NOVA_PI_MIRROR'])

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        if '.' in name:
            package, attribute = name.rsplit('.', 1)
            monkeypatch.setattr(importlib.import_module(package), attribute, module, raising=False)
        return module

    # Reload the actual rules, annotation identity and checkpoint validator for
    # each runtime; do not accidentally test Pi allocation with repo gates.
    active_rules = load('nanobody_opt.rules', root / 'nanobody_opt/rules.py')
    monkeypatch.setattr(active_rules, 'CONFIG_DIR', ROOT / 'config')
    load('nanobody_opt.pr100_filter', root / 'nanobody_opt/pr100_filter.py')
    load('nanobody_opt.publication_gates', root / 'nanobody_opt/publication_gates.py')
    provider = load('compound_miner.nanobody_provider', root / 'compound_miner/nanobody_provider.py')
    # The code-only mirror also omits YAML; redirect just the config read.
    load_metrics = provider.load_boltzgen_metrics
    monkeypatch.setattr(provider, 'load_boltzgen_metrics',
                        lambda path: load_metrics(str(ROOT / 'config/boltzgen_config.yaml')))
    submit_path = root / ('pi_submit.py' if request.param == 'pi'
                          else 'deploy/submit_host/pi_submit.py')
    # pi_submit adds its own directories to sys.path; restore those after each test.
    monkeypatch.setattr(sys, 'path', list(sys.path))
    load('nanobody_opt.commit_submission', root / 'nanobody_opt/commit_submission.py')
    submit = load('_stale_submitter', submit_path)
    monkeypatch.setattr(submit, 'current_archive_digest', lambda target: artifact['archive_sha256'])
    return SimpleNamespace(name=request.param, rules=active_rules, submit=submit,
                           artifact=artifact, previous=replay.challenge,
                           epoch=25957, epoch_length=361)


def current_hash(runtime, kind='match'):
    previous = runtime.previous
    for i in range(10000):
        value = '0x' + hashlib.sha256(f'stale-replay-{i}'.encode()).hexdigest()
        current = runtime.rules.challenge(runtime.rules.pinned_config(runtime.epoch),
                                          runtime.epoch, value, runtime.epoch_length)
        same_v = current['allowed_v'] == previous['allowed_v']
        same_length = current['allowed_cdr3_length'] == previous['allowed_cdr3_length']
        if ((kind == 'match' and same_v and same_length)
                or (kind == 'wrong_v' and not same_v and same_length)
                or (kind == 'wrong_length' and same_v and not same_length)):
            return value
    raise AssertionError('test draw not found')


def chain_replay(runtime, kind='match'):
    calls = []
    value = current_hash(runtime, kind)

    def determine_block_hash(block):
        calls.append(block)
        if block == runtime.epoch * runtime.epoch_length:
            return value
        assert block == runtime.previous['epoch_start_block']
        return runtime.previous['epoch_start_block_hash']

    chain = SimpleNamespace(get_current_block=lambda: runtime.epoch * runtime.epoch_length + 10,
                            tempo=lambda netuid: runtime.epoch_length - 1,
                            determine_block_hash=determine_block_hash)
    return chain, calls, value


def test_epoch_25956_reaches_submission_in_epoch_25957(runtime, monkeypatch, tmp_path):
    chain, calls, value = chain_replay(runtime)
    monkeypatch.setitem(sys.modules, 'bittensor', SimpleNamespace(Subtensor=lambda **kw: chain))
    artifact_path = tmp_path / 'fleet.json'
    artifact_path.write_text(json.dumps(runtime.artifact))
    crew_path = tmp_path / 'crew.json'
    crew_path.write_text(json.dumps([(1, 'fixture-owner-1'), (2, 'fixture-owner-2')]))
    committed = []

    def commit_boundary(rows, arguments, environment):
        committed.extend(rows)
        return len(rows), 0, []

    monkeypatch.setattr(runtime.submit, 'commit_batch', commit_boundary)
    assert runtime.submit.main([
        '--epoch', str(runtime.epoch), '--reaction', '1', '--nanobody', str(artifact_path),
        '--crew', str(crew_path), '--fleet', str(tmp_path / 'no-molecules.json'),
    ]) == 0
    assert len(committed) == 2
    assert all(row['has_sequence'] for row in committed)
    assert {row['plaintext'].split('|')[1] for row in committed} == set(runtime.artifact['sequence_pool'])
    assert calls == [runtime.epoch * runtime.epoch_length, runtime.previous['epoch_start_block']]
    print(f'{runtime.name}: epoch 25956 -> 25957; age > 3600s; '
          '2 validated sequences reached commit_batch; historical hash checked')


@pytest.mark.parametrize('kind,reason', [('wrong_v', 'wrong_epoch_v_gene'),
                                       ('wrong_length', 'wrong_epoch_cdr3_length')])
def test_previous_eligibility_does_not_establish_current_eligibility(runtime, kind, reason):
    chain, calls, value = chain_replay(runtime, kind)
    accepted = runtime.rules.verified_fleet(
        runtime.artifact, runtime.epoch, value, runtime.epoch_length,
        get_block_hash=chain.determine_block_hash)
    assert accepted['sequence_pool'] == []
    assert accepted['submissions'] == []
    assert accepted['sequence_pool_checkpoints'] == {}
    if runtime.name == 'repo':
        assert accepted['handoff_diagnostics']['reason_counts'] == {reason: 2}
    assert runtime.submit.sequences_by_uid(
        runtime.artifact, runtime.epoch, crew=[(1, 'owner')],
        epoch_start_block_hash=value, epoch_length=runtime.epoch_length,
        get_block_hash=chain.determine_block_hash) == {}


@pytest.mark.parametrize('fault', ['missing', 'mismatch', 'unavailable', 'invalid', 'tempo'])
def test_historical_challenge_requires_independent_attestation(runtime, fault):
    chain, calls, value = chain_replay(runtime)
    lookup = chain.determine_block_hash
    artifact = copy.deepcopy(runtime.artifact)
    if fault == 'missing':
        lookup = None
    elif fault == 'mismatch':
        # The carried challenge/evidence remain internally valid but the chain
        # reports a different block hash: the artifact cannot attest itself.
        lookup = lambda block: '0x' + '0' * 64
    elif fault == 'unavailable':
        def lookup(block):
            raise RuntimeError('RPC unavailable')
    elif fault == 'invalid':
        lookup = lambda block: None
    else:
        artifact['challenge']['epoch_length'] += 1
    with pytest.raises(ValueError):
        runtime.rules.verified_fleet(artifact, runtime.epoch, value, runtime.epoch_length,
                                     get_block_hash=lookup)
    if fault == 'tempo':
        assert calls == []


def test_untrusted_marker_cannot_bypass_current_artifact_freshness(runtime):
    artifact = copy.deepcopy(runtime.artifact)
    artifact['handoff_diagnostics'] = {'stale_epoch_downgrade': True}
    with pytest.raises(ValueError, match='updated_at_unix is stale'):
        runtime.submit.sequences_by_uid(
            artifact, runtime.previous['epoch'], crew=[(1, 'owner')],
            epoch_start_block_hash=runtime.previous['epoch_start_block_hash'],
            epoch_length=runtime.epoch_length)


@pytest.mark.parametrize('digest', [None, 'f' * 64])
def test_stale_fleet_requires_independent_current_archive_identity(runtime, digest):
    chain, _, value = chain_replay(runtime)
    accepted = runtime.rules.verified_fleet(runtime.artifact, runtime.epoch, value,
        runtime.epoch_length, get_block_hash=chain.determine_block_hash,
        current_archive_sha256=digest)
    assert accepted['sequence_pool'] == []
    assert accepted['handoff_diagnostics']['reason_counts'] == {'current_novelty_evidence_required': 2}


@pytest.mark.parametrize('fault', ['accelerated_developability', 'empty_confirmation'])
def test_actual_submitter_rules_refuse_unqualified_evidence(runtime, fault):
    checkpoint = next(iter(runtime.artifact['sequence_pool_checkpoints'].values()))
    evidence = copy.deepcopy(checkpoint['pr100_evidence'])
    if fault == 'accelerated_developability':
        evidence['developability'].update(authoritative=False, evidence_backend='cuda_refine')
    else:
        evidence['native_screen_confirmation'] = {'minimum': 3, 'measurements': []}
    with pytest.raises(ValueError):
        runtime.rules.validate_evidence(evidence, checkpoint['sequences'][0], runtime.previous,
            runtime.rules.pinned_config(runtime.previous['epoch']))


def test_target_override_cannot_relabel_old_scientific_assets(runtime, monkeypatch):
    monkeypatch.setattr(runtime.rules, 'TARGET', 'UNQUALIFIED')
    with pytest.raises(ValueError, match='target override'):
        runtime.rules.pinned_config(runtime.epoch)
