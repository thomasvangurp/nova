"""Pure transport regressions, using synthetic eligibility/score evidence."""
import json

import pytest

from compound_miner import nanobody_provider
from nanobody_opt import rules
from tests.test_pr100 import EPOCH, HASH, SEQ, live_trial_checkpoint
from tests.test_pr100_live_consumer import load_candidate


def fleet(checkpoint=None):
    return {'epoch': EPOCH, 'target': rules.TARGET, 'sequence_pool': [SEQ],
            'archive_sha256': 'c' * 64, 'submissions': [],
            'sequence_pool_checkpoints': {SEQ: checkpoint or live_trial_checkpoint()}}


def verify(payload):
    return rules.verified_fleet(payload, EPOCH, HASH, 361)


def test_one_observation_novelty_bound_checkpoint_still_passes(monkeypatch):
    # No annotation/model/archive execution is needed to validate carried data.
    def forbidden(*args, **kwargs):
        raise AssertionError('archive search attempted on the consumer')
    import nanobody_opt.check_validator_novelty as search
    monkeypatch.setattr(search, '_search_one', forbidden)
    result = verify(fleet())
    assert result['sequence_pool'] == [SEQ]
    assert result['handoff_diagnostics'] == {
        'status': 'accepted', 'checked_count': 1, 'accepted_count': 1, 'rejected_count': 0,
        'usable_sequence_count': 1, 'reason_counts': {}, 'rejections': []}
    assert result['sequence_pool_checkpoints'][SEQ]['exact']['observation_count'] == 1


@pytest.mark.parametrize('field,value,code', [
    ('validator_novel', False, 'novelty_not_passed'),
    ('validator_novel', 1, 'novelty_not_passed'),
    ('exact_historical', True, 'novelty_not_passed'),
    ('sequence_hash', 'd' * 64, 'novelty_sequence_mismatch'),
    ('target', 'P20809', 'novelty_target_mismatch'),
    ('archive_sha256', 'not-an-archive', 'novelty_archive_invalid'),
    ('archive_sha256', 'd' * 64, 'novelty_archive_mismatch'),
])
def test_novelty_faults_are_visible_without_sequence_text(field, value, code):
    payload = fleet()
    payload['sequence_pool_checkpoints'][SEQ]['validation']['layer_1']['novelty'][field] = value
    result = verify(payload)
    assert result['sequence_pool'] == []
    diagnostics = result['handoff_diagnostics']
    assert diagnostics['status'] == 'all_rejected'
    assert diagnostics['checked_count'] == diagnostics['rejected_count'] == 1
    assert diagnostics['accepted_count'] == diagnostics['usable_sequence_count'] == 0
    assert diagnostics['reason_counts'] == {code: 1}
    assert diagnostics['rejections'] == [{'sequence_hash': rules.sequence_hash(SEQ), 'reason_code': code}]
    assert SEQ not in json.dumps(diagnostics)


@pytest.mark.parametrize('flag,code', [
    ('historically_unique', 'layer_1_historically_unique_failed'),
    ('duplicate_gate_passed', 'layer_1_duplicate_gate_failed'),
])
@pytest.mark.parametrize('value', [False, None, 1])
def test_existing_layer_one_flags_are_checked(flag, code, value):
    cp = live_trial_checkpoint()
    cp['validation']['layer_1'][flag] = value
    assert verify(fleet(cp))['handoff_diagnostics']['reason_counts'] == {code: 1}


def test_missing_novelty_and_missing_fleet_archive_are_rejected():
    payload = fleet()
    payload['sequence_pool_checkpoints'][SEQ]['validation']['layer_1'].pop('novelty')
    assert verify(payload)['handoff_diagnostics']['reason_counts'] == {'novelty_missing': 1}
    payload = fleet(); payload.pop('archive_sha256')
    assert verify(payload)['handoff_diagnostics']['reason_counts'] == {'novelty_archive_invalid': 1}


def test_missing_checkpoint_is_counted_with_partial_fleet():
    payload = fleet(); payload['sequence_pool'].append('A' * 120)
    result = verify(payload)
    assert result['sequence_pool'] == [SEQ]
    summary = result['handoff_diagnostics']
    assert summary['status'] == 'partial'
    assert (summary['checked_count'], summary['accepted_count'], summary['rejected_count']) == (2, 1, 1)
    assert summary['reason_counts'] == {'checkpoint_missing': 1}


def test_real_eligibility_failure_has_safe_specific_code():
    cp = live_trial_checkpoint()
    cp['pr100_evidence']['developability']['surface_hydrophobic_patches'] = 90.
    assert verify(fleet(cp))['handoff_diagnostics']['reason_counts'] == {'psh_threshold': 1}


def test_arbitrary_exception_details_never_enter_diagnostics(monkeypatch):
    marker = 'private diagnostic text that must not leave the validator'
    def reject(*args, **kwargs):
        raise nanobody_provider.NanobodyCheckpointError(marker, reason_code=marker)
    monkeypatch.setattr(nanobody_provider, 'validate_nanobody_checkpoint', reject)
    diagnostics = verify(fleet())['handoff_diagnostics']
    assert diagnostics['reason_counts'] == {'checkpoint_invalid': 1}
    assert marker not in json.dumps(diagnostics)
    assert SEQ not in json.dumps(diagnostics)


def test_consumer_logs_all_rejected_summary_and_keeps_molecule_independence(capsys):
    consumer = load_candidate()
    payload = fleet()
    payload['sequence_pool_checkpoints'][SEQ]['validation']['layer_1'].pop('novelty')
    assert consumer.sequences_by_uid(payload, EPOCH, crew=[], epoch_start_block_hash=HASH,
                                     epoch_length=361) == {}
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    log = next(row for row in output if row.get('event') == 'pr100_fleet_verification')
    assert log['status'] == 'all_rejected' and log['accepted_count'] == 0
    assert log['reason_counts'] == {'novelty_missing': 1}
    assert SEQ not in json.dumps(log)
    assert consumer.commit_batch([], None, None) == (0, 0, [])
