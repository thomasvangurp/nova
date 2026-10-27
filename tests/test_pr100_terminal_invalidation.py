"""A completed negative must revoke earlier evidence; an unavailable tool must not."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobody_opt import rules, pr100_live_worker as worker, pr100_filter
from nanobody_opt import pr100_epoch_continuation as continuation, pr100_live_controller as controller
from tests.test_pr100_epoch_continuation import legacy, replay, write
from tests.test_pr100_native_screen_integration import CHALLENGE, profile


def failed_profile(sequence):
    value = profile(sequence, passed=False)
    value['aggrescan3d'].update(passed=False, a3d_average='-0.600000000000000001')
    return value


@pytest.mark.parametrize('damage', [None, 'missing', 'nan', 'infinite', 'sequence', 'structure',
    'method', 'radius', 'residue_count', 'status', 'error', 'timeout', 'gpu', 'unknown_backend', 'no_cap_failure'])
def test_only_complete_authoritative_aggregation_failure_becomes_terminal(damage):
    sequence = 'AAAA'; value = failed_profile(sequence)
    if damage == 'missing': value.pop('cdr3_compactness')
    elif damage == 'nan': value['aggrescan3d']['a3d_average'] = 'NaN'
    elif damage == 'infinite': value['aggrescan3d']['a3d_maximum'] = 'Infinity'
    elif damage == 'sequence': value['aggrescan3d']['sequence_hash'] = 'c'*64
    elif damage == 'structure': value['aggrescan3d']['input_structure_sha256'] = 'invalid'
    elif damage == 'method': value['aggrescan3d']['surface_method'] = 'different'
    elif damage == 'radius': value['aggrescan3d']['radius_angstrom'] = 12
    elif damage == 'residue_count': value['aggrescan3d']['residue_count'] = 10
    elif damage == 'status': value['aggrescan3d']['status'] = 'error'
    elif damage == 'error': value['error'] = True
    elif damage == 'timeout': value['error_kind'] = 'timeout'
    elif damage == 'gpu': value.update(evidence_backend='cuda_refine', authoritative=False)
    elif damage == 'unknown_backend': value.update(evidence_backend='unattested', authoritative=True)
    elif damage == 'no_cap_failure': value['aggrescan3d']['a3d_average'] = -1
    before = copy.deepcopy(value)
    reason = worker.terminal_rejection_reason([profile(sequence), value],
        {'psh_strict_rejection_threshold': 89}, sequence, 'aggrescan_unavailable_or_failed')
    assert reason == ('aggrescan_threshold' if damage is None else 'aggrescan_unavailable_or_failed')
    assert value == before


def run_confirmation(directory, monkeypatch, row, challenge, value, interleave=False):
    identity = {'resume_eligible': True, 'challenge_sha256': rules.digest(challenge)}
    monkeypatch.setattr(worker, 'native_resume_identity', lambda *_a: identity)
    monkeypatch.setattr(pr100_filter, 'helper', lambda _: SimpleNamespace(tool_paths=None))
    sequence = row['sequence']
    row = {**row, 'native_resume': {'schema': 'pr100-native-confirmation-checkpoint-v1',
        'sequence_hash': rules.sequence_hash(sequence), 'identity': identity,
        'measurements': [profile(sequence), profile(sequence)], 'required_cpu_observations': 3}}
    write(directory / 'input.json', {'challenge': challenge, 'target': challenge['target'],
        'epoch': challenge['epoch'], 'archive_sha256': 'a'*64, 'archive_checked_at_unix': 1,
        'candidates': [row]})
    calls = []
    def measure(sequences, *_a, **_kw):
        calls.extend(sequences)
        return [copy.deepcopy(value) for _ in sequences], 0., 1.
    result = worker.native_phase({'challenge': challenge, 'native_tools': '/tools',
        'workers': 1, 'repeats': 3, 'allow_marginal': True,
        'native_interleave_confirmations': interleave}, directory, measure=measure)
    assert calls == [sequence]
    assert result['fresh_evaluations'] == 0 and result['repeat_evaluations'] == 1
    return row, result, json.loads((directory / 'native-confirmations.json').read_text())


@pytest.mark.parametrize('interleave', [False, True])
def test_failed_confirmation_persists_same_canonical_reason_before_and_after_classification(
        tmp_path, monkeypatch, interleave):
    sequence = 'AAAA'; failed = failed_profile(sequence)
    assert rules.developability_reason(failed, {'psh_strict_rejection_threshold': 89},
                                       sequence=sequence) == 'aggrescan_unavailable_or_failed'
    _, result, checkpoint = run_confirmation(tmp_path, monkeypatch,
        {'sequence': sequence, 'pr100_annotation': {}}, CHALLENGE, failed, interleave)
    key = rules.sequence_hash(sequence)
    assert result['terminal_native_outcomes'] == {key: 'aggrescan_threshold'}
    assert checkpoint['records'] == [{'sequence_hash': key, 'terminal_reason': 'aggrescan_threshold'}]
    assert result['candidates'] == [] and result['deferred_confirmations'] == []
    assert result['measurement_outcomes'] == {'passed': 0, 'rejected': 1, 'inconclusive': 0}


def test_callback_persists_rejection_even_if_phase_stops_before_final_classification(tmp_path, monkeypatch):
    sequence = 'AAAA'; failed = failed_profile(sequence)
    run_confirmation(tmp_path / 'setup', monkeypatch,
        {'sequence': sequence, 'pr100_annotation': {}}, CHALLENGE, failed)
    directory = tmp_path / 'interrupted'; directory.mkdir()
    (directory / 'input.json').write_bytes((tmp_path / 'setup/input.json').read_bytes())
    def interrupted(sequences, *_a, on_result, **_kw):
        assert sequences == [sequence]
        on_result(sequence, failed)
        checkpoint = json.loads((directory / 'native-confirmations.json').read_text())
        assert checkpoint['records'][0]['terminal_reason'] == 'aggrescan_threshold'
        raise InterruptedError('inert phase cutoff after completed callback')
    monkeypatch.setattr(worker, 'measure_batch', interrupted)
    with pytest.raises(InterruptedError):
        worker.native_phase({'challenge': CHALLENGE, 'native_tools': '/tools',
            'workers': 1, 'repeats': 3}, directory)
    unfinished = json.loads((directory / 'result.json').read_text())
    assert unfinished['native_classification_complete'] is False
    assert json.loads((directory / 'native-confirmations.json').read_text())['records'] == [
        {'sequence_hash': rules.sequence_hash(sequence), 'terminal_reason': 'aggrescan_threshold'}]


@pytest.mark.parametrize('failure', ['measured', 'timeout', 'unavailable', 'provenance'])
@pytest.mark.parametrize('checkpoint_only', [False, True])
def test_continuation_revokes_prior_publication_only_after_proven_cpu_negative(
        legacy, monkeypatch, tmp_path, failure, checkpoint_only):
    x = legacy; sequence = x.rows[0]['sequence']; key = rules.sequence_hash(sequence)
    failed = failed_profile(sequence)
    if failure == 'timeout': failed = {'error': True, 'error_kind': 'timeout'}
    elif failure == 'unavailable': failed['aggrescan3d']['status'] = 'error'
    elif failure == 'provenance': failed['aggrescan3d']['sequence_hash'] = 'c'*64
    row, result, progress = run_confirmation(tmp_path / 'measured', monkeypatch,
        x.rows[0], x.replay.challenge, failed)
    state = continuation.migrate(x.local, x.replay.challenge, x.config)
    state['pending_confirmation'] = [row]
    write(x.local / 'continuation.json', state)
    if checkpoint_only: result['terminal_native_outcomes'] = {}
    def phase(_config, challenge, name, _files, *_a, **_kw):
        assert name == 'native'  # Previously completed scoring must never run again.
        return {'result.json': json.dumps({**x.replay.validity, **result}).encode(),
                'native-confirmations.json': json.dumps(progress).encode()}
    monkeypatch.setattr(controller, 'run_phase', phase)
    outcome = controller.run_epoch(x.config, x.chain, controller.clock_state(x.chain))
    saved = json.loads((x.local / 'continuation.json').read_text())
    publication = json.loads(Path(x.config['publication_path']).read_text())
    if failure == 'measured':
        assert key in saved['invalidated_native_hashes']
        assert saved['native_rows'] == [] and publication['sequence_pool'] == []
        assert outcome['status'] == 'withdrawn'
    else:
        assert key not in saved.get('invalidated_native_hashes', [])
        assert len(saved['native_rows']) == 1 and publication['sequence_pool'] == [sequence]
        assert saved['pending_confirmation'][0]['native_resume']['measurements'] == [profile(sequence)]*2
