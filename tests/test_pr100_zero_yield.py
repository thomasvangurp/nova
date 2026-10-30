"""Zero yield is a continuation outcome; malformed evidence remains a failure."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobody_opt import pr100_epoch_continuation as continuation
from nanobody_opt import pr100_live_controller as controller, rules

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def legacy(tmp_path, monkeypatch):
    epoch, length, block_hash = 26018, 361, '0x' + 'a' * 64
    challenge = rules.challenge(rules.pinned_config(epoch), epoch, block_hash, length)
    class Chain:
        def get_current_block(self): return epoch * length + 10
        def tempo(self, _): return length - 1
        def determine_block_hash(self, _): return block_hash
    config = {'state_root': str(tmp_path / 'state'), 'remote_root': '/remote',
              'source_sha': 'a' * 40, 'publication_path': str(tmp_path / 'published.json'),
              'worker': {'native_tools': '/tools', 'proposals': 600, 'repeats': 3}}
    write(Path(config['publication_path']), {'epoch': epoch - 1, 'sequence_pool': ['AAAA']})
    monkeypatch.setattr(controller, 'fetch_archive', lambda _: b'sequence\n')
    monkeypatch.setattr(controller, 'start_native_journal_collection', lambda *a: None)
    return SimpleNamespace(config=config, chain=Chain(), local=tmp_path / 'state' / str(epoch),
        replay=SimpleNamespace(challenge=challenge), rows=[{'sequence': 'AAAA'}, {'sequence': 'CCCC'}])


def pending(x, rows, *, native=False):
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state.update(native_rows=rows if native else [], score_rows={},
                 native_attempted_hashes=[], score_attempted_hashes=[],
                 pending_native=[] if native else rows)
    write(x.local / 'continuation.json', state)


def empty_scores(challenge, sequences, *, admitted):
    hashes = [rules.sequence_hash(s) for s in sequences]
    attempted = hashes if admitted else []
    return {'result.json': json.dumps({
        'challenge': challenge, 'epoch': challenge['epoch'], 'status': 'partial',
        'input_count': len(sequences), 'scored_candidates': 0, 'candidates': [],
        'attempted_sequence_hashes': attempted,
        'deferred_sequence_hashes': [] if admitted else hashes}).encode(),
        'score/report.json': json.dumps({
            'schema': 'pr100-live-score-v1', 'status': 'partial', 'groups': [],
            'scored_candidates': 0, 'registered_scorer_sha256': None,
            'attempted_sequence_hashes': attempted}).encode()}


@pytest.mark.parametrize('multi_host', [False, True])
def test_completed_zero_native_candidates_continue_without_scoring(legacy, monkeypatch, multi_host):
    x = legacy
    rows = [{'sequence': 'A' * (100 + i)} for i in range(61)]
    pending(x, rows)
    if multi_host:
        x.config['native_hosts'] = ['master', 'slave']
    calls = []

    def phase(cfg, challenge, name, files, stop, chain, local, **kwargs):
        calls.append(name)
        assert name == 'native'
        incoming = json.loads(files['input.json'])
        hashes = [rules.sequence_hash(r['sequence']) for r in incoming['candidates']]
        write(local / 'native-supervisor.json', {'reason': 'completed', 'exit_code': 0,
            'cleanup': {'remaining_descendants': 0, 'fixed_point_reached': True}})
        value = {**incoming, 'status': 'complete', 'native_classification_complete': True,
            'novelty_checked': False, 'input_count': len(hashes), 'candidate_count': 0,
            'candidates': [], 'attempted_sequence_hashes': hashes,
            'fresh_admitted_sequence_hashes': hashes,
            'terminal_native_outcomes': {h: 'aggrescan_threshold' for h in hashes}}
        return {'result.json': json.dumps(value).encode(),
            'native-confirmations.json': json.dumps({'challenge': challenge, 'records': [
                {'sequence_hash': h, 'terminal_reason': 'aggrescan_threshold'} for h in hashes]}).encode()}

    monkeypatch.setattr(controller, 'run_phase', phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'no_passing_candidate'
    assert calls == ['native'] * (2 if multi_host else 1)
    saved = continuation.read(x.local / 'continuation.json')
    assert len(saved['native_attempted_hashes']) == 61
    assert saved['active'] is None and saved['pending_confirmation'] == []
    assert continuation.mode_for(saved) == 'sequence'


@pytest.mark.parametrize('admitted', [False, True])
def test_zero_score_report_retains_admission_and_continues(legacy, monkeypatch, admitted):
    x = legacy
    pending(x, x.rows, native=True)
    before = Path(x.config['publication_path']).read_bytes()

    def phase(cfg, challenge, name, files, stop, chain, local, **kwargs):
        assert name == 'score'
        return empty_scores(challenge, json.loads(files['input.json'])['sequences'], admitted=admitted)

    monkeypatch.setattr(controller, 'run_phase', phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'no_scored_candidate'
    saved = continuation.read(x.local / 'continuation.json')
    assert saved['score_rows'] == {} and saved['active'] is None
    assert set(saved['score_attempted_hashes']) == (
        {rules.sequence_hash(r['sequence']) for r in x.rows} if admitted else set())
    assert continuation.mode_for(saved) == 'sequence'
    if not admitted:
        assert saved['score_retry_after_unix'] > saved.get('updated_at_unix', 0)
        assert saved['score_retry_after_unix'] <= (outcome['publication_cutoff_unix']
            - continuation.required_seconds('score', x.config))
        assert continuation.mode_for(saved, now=saved['score_retry_after_unix']) == 'score'
    assert Path(x.config['publication_path']).read_bytes() == before


@pytest.mark.parametrize('bad', ['missing_report', 'wrong_challenge', 'unknown_admission', 'missing_partition'])
def test_invalid_zero_score_report_still_fails_closed(legacy, monkeypatch, bad):
    x = legacy
    pending(x, x.rows, native=True)
    files = empty_scores(x.replay.challenge, [r['sequence'] for r in x.rows], admitted=False)
    result = json.loads(files['result.json'])
    if bad == 'missing_report':
        files.pop('score/report.json')
    elif bad == 'wrong_challenge':
        result['challenge'] = {}
    elif bad == 'unknown_admission':
        result['attempted_sequence_hashes'] = ['f' * 64]
    else:
        result['deferred_sequence_hashes'] = []
    files['result.json'] = json.dumps(result).encode()
    monkeypatch.setattr(controller, 'run_phase', lambda *a, **kw: files)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'recoverable_failure'
    assert outcome['error_code'] == 'score_no_group'
    assert outcome['error_location'].startswith('pr100_live_controller.py:')
    assert continuation.read(x.local / 'continuation.json')['score_rows'] == {}


@pytest.mark.parametrize('message,code', [
    ('saved phase input or source identity mismatch', 'phase_binding_mismatch'),
    ('native dispatch routing or settings changed', 'native_dispatch_binding_mismatch'),
    ('private-runtime-message-must-not-be-logged', 'unclassified')])
def test_failure_receipt_identifies_safe_reason_without_raw_message(legacy, monkeypatch, message, code):
    x = legacy
    pending(x, x.rows)
    def fail(*args, **kwargs):
        raise ValueError(message)
    monkeypatch.setattr(controller, 'run_phase', fail)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'recoverable_failure' and outcome['stage'] == 'native'
    assert outcome['error_code'] == code
    assert outcome['error_location'].startswith('pr100_epoch_continuation.py:')
    assert message not in json.dumps(outcome)


def test_scoring_resource_backoff_preserves_candidates_and_allows_native_work(legacy):
    x = legacy
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state.update(native_rows=x.rows, score_retry_after_unix=200, pending_native=[{'sequence': 'DDDD'}])
    assert continuation.mode_for(state, now=100) == 'native'
    assert continuation.mode_for(state, now=200) == 'score'
    assert state['native_rows'] == x.rows


def test_backoff_does_not_launch_scoring_after_a_native_phase(legacy, monkeypatch):
    x = legacy
    pending(x, x.rows, native=True)
    state = continuation.read(x.local / 'continuation.json')
    state.update(active={'number': 1, 'phase': 'score', 'proposal_offset': 0},
                 score_retry_after_unix=10**12)
    write(x.local / 'continuation.json', state)
    monkeypatch.setattr(controller, 'run_phase', lambda *a, **k: pytest.fail('must honor persisted retry time'))
    controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    saved = continuation.read(x.local / 'continuation.json')
    assert saved['active'] is None and saved['native_rows'] == x.rows
    assert saved['score_attempted_hashes'] == []


def test_duplicate_refill_advances_past_every_attempted_proposal(legacy, monkeypatch):
    x = legacy
    def phase(config, challenge, name, files, stop, chain, local, **kwargs):
        assert name == 'sequence'
        return {'result.json': json.dumps({'status': 'complete', 'challenge': challenge,
            'candidates': [], 'generated_sequence_hashes': [], 'next_proposal_offset': 1203}).encode()}
    monkeypatch.setattr(controller, 'run_phase', phase)
    controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    saved = continuation.read(x.local / 'continuation.json')
    assert saved['proposal_offset'] == 1203
