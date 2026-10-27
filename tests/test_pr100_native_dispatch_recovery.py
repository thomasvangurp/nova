"""A late peer must add evidence without replaying an already scored cohort."""
import base64
import json

from nanobody_opt import pr100_epoch_continuation as continuation
from nanobody_opt import pr100_live_controller as controller
from nanobody_opt import pr100_native_dispatch as dispatch
from nanobody_opt import rules
from tests.test_pr100_live_handoff_contract import replay
from tests.test_pr100_epoch_continuation import legacy, score_files, write
from tests.test_pr100_live_controller import config as route_config


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
        write(local / 'native-supervisor.json', {'reason': 'completed', 'cleanup': {
            'fixed_point_reached': recovered, 'remaining_descendants': 0 if recovered else None}})
        value = {**x.replay.validity, 'candidates': rows,
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
