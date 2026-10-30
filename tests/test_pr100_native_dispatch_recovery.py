"""A late peer must add evidence without replaying an already scored cohort."""
import base64
import json
import pytest

from nanobody_opt import pr100_epoch_continuation as continuation
from nanobody_opt import pr100_live_controller as controller
from nanobody_opt import pr100_native_dispatch as dispatch
from nanobody_opt import rules
from tests.test_pr100_live_handoff_contract import replay
from tests.test_pr100_epoch_continuation import legacy, score_files, write
from tests.test_pr100_live_controller import config as route_config


def test_both_host_dispatch_preserves_assigned_complete_native_histories(tmp_path):
    rows = [{'sequence': 'A' * 60}, {'sequence': 'C' * 60}]
    challenge = {'epoch': 26020}
    incoming = {'challenge': challenge, 'epoch': 26020, 'target': 'test',
                'archive_sha256': 'a' * 64, 'candidates': rows}
    def phase(config, current, name, payload, stop, chain, directory, **kwargs):
        local = json.loads(payload['input.json'])
        histories = [{'sequence_hash': rules.sequence_hash(row['sequence']),
                      'execution_host': config['_native_host'],
                      'measurements': [{'passed': False}], 'request_sha256s': ['b' * 64]}
                     for row in local['candidates']]
        write(directory / 'native-supervisor.json', {'cleanup': {
            'fixed_point_reached': True, 'remaining_descendants': 0}})
        value = {**local, 'candidates': [], 'status': 'complete', 'novelty_checked': True,
                 'native_classification_complete': True, 'native_complete_histories': histories}
        return {'result.json': json.dumps(value).encode()}
    config = {'worker': {'workers': 1}, 'native_hosts': ['master', 'slave'], 'source_sha': 'c' * 40}
    result = dispatch.run(config, challenge, {'input.json': json.dumps(incoming).encode()},
                          100, object(), tmp_path, run_phase=phase)
    histories = json.loads(result['result.json'])['native_complete_histories']
    assert {row['sequence_hash'] for row in histories} == {rules.sequence_hash(row['sequence']) for row in rows}
    assert all(row['measurements'] == [{'passed': False}] for row in histories)


def test_foreign_native_history_cannot_enter_dispatch_merge():
    rows = [{'sequence': 'A' * 60}]
    incoming = {'challenge': {'epoch': 26020}, 'epoch': 26020, 'target': 'test', 'archive_sha256': 'a' * 64}
    value = {**incoming, 'candidates': [], 'native_complete_histories': [
        {'sequence_hash': rules.sequence_hash('C' * 60), 'measurements': []}]}
    detail = {}
    result, _ = dispatch.host_artifacts({'result.json': json.dumps(value).encode()}, incoming,
                                        rows, {}, False, detail)
    assert result['status'] == 'failed'
    assert detail['artifact_error'] == 'unusable_native_result'


def confirmation(row, count=3):
    return {**row, 'native_resume': {
        'identity': {'resume_eligible': True}, 'measurements': [{}] * count,
        'required_cpu_observations': 3}}


def pending_native_state(x):
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state.update(native_rows=[], score_rows={}, native_attempted_hashes=[],
                 score_attempted_hashes=[], pending_native=x.rows)
    write(x.local / 'continuation.json', state)


@pytest.mark.parametrize('peer', ['complete', 'failed', 'incomplete', 'unresolved', 'no_novelty'])
def test_novelty_rejection_finishes_only_after_every_native_host_is_complete(legacy, monkeypatch, peer):
    x = legacy
    x.config.update(native_hosts=['master', 'slave'],
                    native_worker_overrides={'master': {'workers': 1}, 'slave': {'workers': 1}})
    pending_native_state(x)

    def phase(cfg, challenge, name, files, stop, chain, local, **kwargs):
        assert name == 'native'
        incoming = json.loads(files['input.json'])
        is_peer = cfg['_native_host'] == 'slave'
        unresolved = is_peer and peer == 'unresolved'
        write(local / 'native-supervisor.json', {'cleanup': {
            'fixed_point_reached': not unresolved, 'remaining_descendants': None if unresolved else 0}})
        value = {**incoming, 'status': 'failed' if is_peer and peer == 'failed' else 'complete',
                 'native_classification_complete': not (is_peer and peer in ('failed', 'incomplete')),
                 'novelty_checked': not (is_peer and peer == 'no_novelty'), 'candidates': [],
                 'attempted_sequence_hashes': [rules.sequence_hash(r['sequence']) for r in incoming['candidates']]}
        return {'result.json': json.dumps(value).encode(),
                'native-confirmations.json': json.dumps({'challenge': challenge,
                    'records': [confirmation(r) for r in incoming['candidates']]}).encode()}

    monkeypatch.setattr(controller, 'run_phase', phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'no_passing_candidate'
    saved = continuation.read(x.local / 'continuation.json')
    merged = continuation.read(x.local / 'attempts/00001/native/result.json')
    assert merged['native_classification_complete'] is (peer == 'complete')
    if peer == 'complete':
        assert saved['pending_confirmation'] == []
        assert continuation.mode_for(saved) == 'sequence'
    else:
        assert {r['sequence'] for r in saved['pending_confirmation']} == {r['sequence'] for r in x.rows}
        assert continuation.mode_for(saved) == 'native'
    if peer == 'unresolved':
        assert saved['active']['number'] == 1 and saved['active']['phase'] == 'native'
        assert saved['active']['native_unresolved'] is True
    else:
        assert saved['active'] is None


@pytest.mark.parametrize('uncertainty', [None, 'partial_confirmation', 'missing_classification',
    'missing_archive', 'different_archive', 'different_challenge', 'different_epoch', 'different_target'])
def test_checkpoint_retirement_requires_complete_current_archive_bound_classification(legacy, monkeypatch, uncertainty):
    x = legacy
    pending_native_state(x)

    def phase(cfg, challenge, name, files, stop, chain, local, **kwargs):
        assert name == 'native'
        incoming = json.loads(files['input.json'])
        value = {**incoming, 'status': 'complete', 'native_classification_complete': True,
                 'novelty_checked': True, 'candidates': [],
                 'attempted_sequence_hashes': [rules.sequence_hash(r['sequence']) for r in incoming['candidates']]}
        if uncertainty == 'missing_classification':
            value.pop('native_classification_complete')
        elif uncertainty == 'missing_archive':
            value.pop('archive_sha256')
        elif uncertainty == 'different_archive':
            value['archive_sha256'] = 'f' * 64
        elif uncertainty == 'different_challenge':
            value['challenge'] = {**challenge, 'epoch': challenge['epoch'] + 1}
        elif uncertainty == 'different_epoch':
            value['epoch'] += 1
        elif uncertainty == 'different_target':
            value['target'] = 'other'
        records = [confirmation(r, 2 if uncertainty == 'partial_confirmation' else 3)
                   for r in incoming['candidates']]
        return {'result.json': json.dumps(value).encode(),
                'native-confirmations.json': json.dumps({'challenge': challenge, 'records': records}).encode()}

    monkeypatch.setattr(controller, 'run_phase', phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'no_passing_candidate'
    saved = continuation.read(x.local / 'continuation.json')
    assert len(saved['pending_confirmation']) == (0 if uncertainty is None else len(x.rows))
    assert continuation.mode_for(saved) == ('sequence' if uncertainty is None else 'native')


def test_clean_score_failure_cannot_release_an_unresolved_native_peer(legacy, monkeypatch):
    x = legacy
    routes = route_config(x.local)
    x.config.update(hosts=routes['hosts'], gpu=[0], native_hosts=['master', 'slave'])
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state.update(native_rows=[], score_rows={}, native_attempted_hashes=[],
                 score_attempted_hashes=[], pending_native=x.rows)
    write(x.local / 'continuation.json', state)

    def native(cfg, challenge, payload, stop, chain, local, **kwargs):
        write(local / 'native-supervisor.json', {'reason': 'ownership_unresolved', 'cleanup': {
            'fixed_point_reached': False, 'remaining_descendants': None}})
        value = {**x.replay.validity, 'candidates': x.rows[:1],
                 'attempted_sequence_hashes': [rules.sequence_hash(x.rows[0]['sequence'])],
                 'unresolved_native_hosts': ['slave']}
        return {'result.json': json.dumps(value).encode()}

    def phase(cfg, challenge, name, files, stop, chain, local, **kwargs):
        assert name == 'score'
        write(local / 'score-supervisor.json', {'reason': 'nonzero_exit', 'cleanup': {
            'fixed_point_reached': True, 'remaining_descendants': 0}})
        write(local / 'score/result.json', {'attempted_sequence_hashes': []})
        raise RuntimeError('inert clean score failure')

    monkeypatch.setattr(dispatch, 'run', native)
    monkeypatch.setattr(controller, 'run_phase', phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert outcome['status'] == 'recoverable_failure'
    saved = continuation.read(x.local / 'continuation.json')
    assert saved['active'] is not None
    assert saved['active']['number'] == 1 and saved['active']['native_unresolved'] is True


def test_peer_recovery_scores_each_new_cohort_once_and_publishes_cumulative_union(legacy, monkeypatch):
    x = legacy
    routes = route_config(x.local)
    x.config.update(hosts=routes['hosts'], gpu=[0], native_hosts=['master', 'slave'],
                    native_worker_overrides={'master': {'workers': 1}, 'slave': {'workers': 1}})
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state.update(native_rows=[], score_rows={}, native_attempted_hashes=[],
                 score_attempted_hashes=[], pending_native=x.rows)
    write(x.local / 'continuation.json', state)
    native_calls, scored_hashes, score_directories = [], [], []

    def native(cfg, challenge, payload, stop, chain, local, **kwargs):
        native_calls.append(cfg['_attempt'])
        recovered = len(native_calls) == 2
        rows = x.rows if recovered else x.rows[:1]
        identity = cfg['hosts']['master']['novelty_runtime']['identity_sha256']
        rows = [{**row, 'novelty': {**row['novelty'], 'runtime_sha256': identity}} for row in rows]
        write(local / 'native-supervisor.json', {'reason': 'completed', 'cleanup': {
            'fixed_point_reached': recovered, 'remaining_descendants': 0 if recovered else None}})
        value = {**x.replay.validity, 'candidates': rows, 'novelty_runtime_sha256': identity,
                 'attempted_sequence_hashes': [rules.sequence_hash(r['sequence']) for r in rows],
                 'unresolved_native_hosts': [] if recovered else ['slave']}
        return {'result.json': json.dumps(value).encode()}

    original_phase = controller.run_phase

    def transport(cfg, host, request):
        incoming = json.loads(base64.b64decode(request['files']['input.json']))
        sequences = incoming['sequences']
        scored_hashes.extend(rules.sequence_hash(s) for s in sequences)
        score_directories.append(request['directory'])
        files = score_files(x.replay, [r for r in x.replay.data['rows'] if r['sequence'] in sequences])
        files['result.json'] = json.dumps({'attempted_sequence_hashes': [
            rules.sequence_hash(s) for s in sequences]}).encode()
        return {'running': False, 'receipt': {'reason': 'completed', 'cleanup': {
            'remaining_descendants': 0, 'fixed_point_reached': True}},
            'files': {name: base64.b64encode(raw).decode() for name, raw in files.items()}}

    def phase(cfg, challenge, name, files, stop, chain, local, **kwargs):
        assert name == 'score'
        return original_phase(cfg, challenge, name, files, stop, chain, local,
                              transport=transport, sleep=lambda _: None, **kwargs)

    monkeypatch.setattr(dispatch, 'run', native)
    monkeypatch.setattr(controller, 'run_phase', phase)
    first = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    second = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    assert first['status'] == second['status'] == 'published'
    assert native_calls == [1, 1]
    expected = {rules.sequence_hash(r['sequence']) for r in x.rows}
    assert len(scored_hashes) == len(expected) == 2 and set(scored_hashes) == expected
    assert len(set(score_directories)) == 2
    fleet = json.loads(__import__('pathlib').Path(x.config['publication_path']).read_text())
    assert set(fleet['sequence_pool']) == {r['sequence'] for r in x.rows}
