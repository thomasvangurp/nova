"""Adversarial lifecycle contracts: retry ownership, binding, and epoch boundaries."""
from __future__ import annotations

import base64
import copy
import json
import time

import pytest

from nanobody_opt import pr100_epoch_continuation as continuation
from nanobody_opt import pr100_live_controller as controller

EPOCH, LENGTH = 25963, 361

class Chain:
    def __init__(self):
        self.block = EPOCH * LENGTH + 10
    def get_current_block(self):
        return self.block
    def tempo(self, _netuid):
        return LENGTH - 1
    def determine_block_hash(self, _block):
        return '0xlifecycle'

def challenge():
    return {'epoch': EPOCH, 'epoch_length': LENGTH,
            'epoch_start_block_hash': '0xlifecycle', 'target': 'inert-target',
            'allowed_v': 'inert-v', 'allowed_cdr3_length': 12}

def config(tmp_path):
    host = {'address': 'host.invalid', 'port': 2222, 'release': '/release',
            'annotation_python': '/python', 'native_python': '/python',
            'scoring_python': '/python', 'python_path': '/extra', 'path': '/bin',
            'references': '/bank.json', 'scorer': 'registered', 'scorer_settings': '/settings.json'}
    return {'source_sha': 'a' * 40, 'state_root': str(tmp_path / 'state'),
            'remote_root': '/remote', 'gpu': [0], '_attempt': 1, '_continuation': True,
            'worker': {'native_tools': '/tools', 'repeats': 3},
            'hosts': {'master': dict(host), 'slave': dict(host)}}

def terminal(*, remaining=0, fixed=True):
    return {'running': False, 'receipt': {'reason': 'completed' if not remaining else 'cleanup_failed',
            'exit_code': 0, 'cleanup': {'remaining_descendants': remaining, 'fixed_point_reached': fixed}},
            'files': {'result.json': base64.b64encode(b'{"status":"complete","candidates":[]}').decode()}}

def invoke(tmp_path, cfg, chain, transport, *, expected=None, files=None):
    local = tmp_path / 'attempt'
    local.mkdir(exist_ok=True)
    return controller.run_phase(cfg, expected or challenge(), 'native',
        {'input.json': b'{"inert":1}'} if files is None else files,
        time.time() + 2000, chain, local, transport=transport, sleep=lambda _seconds: None)

def test_ambiguous_start_retains_exact_intent_and_refreshes_clock(tmp_path):
    cfg, chain, dispatched = config(tmp_path), Chain(), []
    def interrupted(_cfg, host, request):
        intent = json.loads((tmp_path / 'attempt/native-request.json').read_text())
        assert intent['request']['files'] == request['files']
        assert intent['host'] == host
        dispatched.append(copy.deepcopy(request))
        raise RuntimeError('inert ambiguous transport')
    with pytest.raises(RuntimeError):
        invoke(tmp_path, cfg, chain, interrupted)
    chain.block += 3
    def recovered(_cfg, _host, request):
        dispatched.append(copy.deepcopy(request))
        return terminal()
    invoke(tmp_path, cfg, chain, recovered)
    assert dispatched[1]['directory'] == dispatched[0]['directory']
    assert dispatched[1]['spec'] == dispatched[0]['spec']
    assert dispatched[1]['files'] == dispatched[0]['files']
    assert dispatched[1]['clock']['block'] == dispatched[0]['clock']['block'] + 3

@pytest.mark.parametrize('change', ['input', 'challenge', 'source', 'settings'])
@pytest.mark.parametrize('already_collected', [False, True])
def test_retry_refuses_changed_phase_identity_before_any_remote_call(tmp_path, change, already_collected):
    cfg, chain, expected = config(tmp_path), Chain(), challenge()
    def initial(*_args):
        if already_collected:
            return terminal()
        raise RuntimeError('inert ambiguous transport')
    if already_collected:
        invoke(tmp_path, cfg, chain, initial)
    else:
        with pytest.raises(RuntimeError):
            invoke(tmp_path, cfg, chain, initial)
    files = {'input.json': b'{"inert":1}'}
    if change == 'input':
        files['input.json'] = b'{"inert":2}'
    elif change == 'challenge':
        expected['epoch_start_block_hash'] = '0xother'
    elif change == 'source':
        cfg['source_sha'] = 'b' * 40
    else:
        cfg['worker']['repeats'] = 2
    def forbidden(*_args):
        pytest.fail('changed phase identity reached remote transport')
    with pytest.raises(ValueError):
        invoke(tmp_path, cfg, chain, forbidden, expected=expected, files=files)

@pytest.mark.parametrize('remaining,fixed', [(1, False), (0, False)])
def test_completed_scientific_result_does_not_release_uncertain_cleanup(tmp_path, remaining, fixed):
    cfg, chain = config(tmp_path), Chain()
    for _attempt in range(2):
        with pytest.raises(RuntimeError):
            invoke(tmp_path, cfg, chain, lambda *_: terminal(remaining=remaining, fixed=fixed))

def test_cached_result_cannot_cross_epoch_boundary(tmp_path):
    cfg, chain = config(tmp_path), Chain()
    invoke(tmp_path, cfg, chain, lambda *_: terminal())
    chain.block = (EPOCH + 1) * LENGTH
    def forbidden(*_args):
        pytest.fail('old epoch reached transport')
    with pytest.raises(TimeoutError):
        invoke(tmp_path, cfg, chain, forbidden)

def test_missing_legacy_receipt_does_not_imply_no_prior_remote_owner(tmp_path):
    local = tmp_path / 'legacy'
    (local / 'sequence').mkdir(parents=True)
    (local / 'sequence/result.json').write_text(json.dumps({
        'challenge': challenge(), 'status': 'complete', 'candidates': [], 'proposals': 600}))
    (local / 'sequence-supervisor.json').write_text(json.dumps(terminal()['receipt']))
    state = continuation.migrate(local, challenge(), config(tmp_path))
    assert state.get('legacy_uncertain_phase') in ('native', 'score', 'sequence')

def test_new_empty_epoch_has_no_invented_legacy_ownership(tmp_path):
    local = tmp_path / 'new'
    local.mkdir()
    state = continuation.migrate(local, challenge(), config(tmp_path))
    assert not state.get('legacy_uncertain_phase')
    assert state['active'] is None

def test_transport_retry_gets_fresh_chain_clock_not_original_sample(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cfg, chain = config(tmp_path), Chain()
    clocks, sent = [], []
    def refresh():
        chain.block += 1
        value = controller.clock_state(chain)
        clocks.append(value)
        return value
    def run(_command, *, input, **_kwargs):
        sent.append(json.loads(input))
        return SimpleNamespace(returncode=255 if len(sent) == 1 else 0,
            stdout=b'{"running":true}\n', stderr=b'')
    monkeypatch.setattr(controller.subprocess, 'run', run)
    result = controller.remote({**cfg, '_refresh_clock': refresh}, 'master', {
        'operation': 'poll', 'directory': '/remote/inert',
        'clock': controller.clock_state(chain), 'transport_deadline_unix': time.time() + 30},
        sleep=lambda _seconds: None)
    assert result['running'] is True
    assert len(sent) == 2
    assert sent[0]['clock'] == clocks[0] and sent[1]['clock'] == clocks[1]
    assert sent[1]['clock']['block'] > sent[0]['clock']['block']


def test_epoch_resume_does_not_change_active_worker_binding_after_sequence_progress(tmp_path, monkeypatch):
    """Generated hashes change durable state while the same native intent is active."""
    cfg, chain, expected = config(tmp_path), Chain(), challenge()
    cfg.pop('_attempt')
    sequence = 'A' * 100
    sequence_hash = continuation.rules.sequence_hash(sequence)
    monkeypatch.setattr(continuation.rules, 'pinned_config', lambda _epoch: {})
    monkeypatch.setattr(continuation.rules, 'challenge', lambda *_args: expected)
    monkeypatch.setattr(controller, 'fetch_archive', lambda _target: b'sequence\n')
    monkeypatch.setattr(controller, 'start_native_journal_collection', lambda *_args: False)
    original = controller.run_phase
    native_calls = []

    def transport(_cfg, _host, request):
        phase = json.loads(base64.b64decode(request['files']['job.json']))['phase']
        if phase == 'sequence':
            value = {'challenge': expected, 'status': 'complete',
                     'generated_sequence_hashes': [sequence_hash],
                     'candidates': [{'sequence': sequence}]}
        else:
            assert phase == 'native'
            native_calls.append(request['directory'])
            if len(native_calls) == 1:
                raise RuntimeError('inert ambiguous native transport')
            value = {'challenge': expected, 'status': 'complete', 'novelty_checked': True,
                     'archive_sha256': 'c' * 64, 'attempted_sequence_hashes': [sequence_hash],
                     'candidates': []}
        response = terminal()
        response['files']['result.json'] = base64.b64encode(json.dumps(value).encode()).decode()
        return response

    def phase(*args, **kwargs):
        return original(*args, **kwargs, transport=transport, sleep=lambda _seconds: None)

    monkeypatch.setattr(controller, 'run_phase', phase)
    first = continuation.run_epoch(cfg, chain, controller.clock_state(chain))
    assert first['status'] == 'recoverable_failure' and first['stage'] == 'native'
    second = continuation.run_epoch(cfg, chain, controller.clock_state(chain))
    assert second['status'] == 'no_passing_candidate'
    assert len(native_calls) == 2 and native_calls[0] == native_calls[1]


def test_completed_legacy_sequence_does_not_hide_an_uncollected_running_native(tmp_path, monkeypatch):
    cfg, chain, expected = config(tmp_path), Chain(), challenge()
    local = tmp_path / 'state' / str(EPOCH)
    (local / 'sequence').mkdir(parents=True)
    (local / 'sequence/result.json').write_text(json.dumps({
        'challenge': expected, 'status': 'complete', 'candidates': [], 'proposals': 600}))
    (local / 'sequence-supervisor.json').write_text(json.dumps(terminal()['receipt']))
    monkeypatch.setattr(continuation.rules, 'pinned_config', lambda _epoch: {})
    monkeypatch.setattr(continuation.rules, 'challenge', lambda *_args: expected)
    monkeypatch.setattr(controller, 'fetch_archive', lambda _target: b'sequence\n')
    queried = []

    def remote(_cfg, _host, request):
        phase = request['directory'].rsplit('/', 1)[-1]
        queried.append(phase)
        if phase == 'native':
            return {'running': True}
        return terminal()

    def forbidden(*_args, **_kwargs):
        pytest.fail('new phase launched while legacy native ownership was unexamined')

    monkeypatch.setattr(controller, 'remote', remote)
    monkeypatch.setattr(controller, 'run_phase', forbidden)
    outcome = continuation.run_epoch(cfg, chain, controller.clock_state(chain))
    assert outcome['status'] == 'recovering_legacy_ownership'
    assert 'native' in queried


def test_remote_poll_distinguishes_absent_job_from_crash_during_launch(tmp_path):
    """spec.json precedes Popen; missing launch.json is not proof no child exists."""
    import subprocess
    import sys
    directory = tmp_path / 'remote'
    request = {'operation': 'poll', 'directory': str(directory),
               'clock': controller.clock_state(Chain())}

    def poll():
        reply = subprocess.run([sys.executable, '-c', controller.TRANSPORT],
            input=json.dumps(request), capture_output=True, text=True, timeout=5, check=True)
        return json.loads(reply.stdout)

    absent = poll()
    assert absent['running'] is False
    assert not directory.exists()
    directory.mkdir()
    (directory / 'spec.json').write_text('{"inert_launch_intent":true}')
    uncertain = poll()
    assert uncertain.get('running') is True or not continuation.cleanup_verified(uncertain.get('receipt', {}))


def test_native_admission_refuses_exact_timeout_without_any_setup_headroom(tmp_path, monkeypatch):
    """A300s worker budget cannot admit a300s call after any positive setup time."""
    cfg, chain = config(tmp_path), Chain()
    cfg['worker']['candidate_budget_seconds'] = 300
    cfg['novelty_reserve_seconds'] = 300
    now = 1000.0
    monkeypatch.setattr(controller.time, 'time', lambda: now)
    stop_at = (now + controller.PUBLISH_RESERVE_SECONDS + controller.PHASE_FLOORS['score']
               + cfg['worker']['candidate_budget_seconds'] + cfg['novelty_reserve_seconds'])
    with pytest.raises(TimeoutError):
        controller.phase_request(cfg, challenge(), 'native', {}, stop_at,
                                 controller.clock_state(chain))
