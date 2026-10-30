"""Measured historical pools, with a labelled authentication stub for allocation.

These snapshots test ranking and the real Pi allocator. They do not reconstruct
native receipts or make unsubmitted candidates' predicted fitness into outcomes.
"""
import hashlib
import json
from pathlib import Path

import pytest

from deploy.submit_host import pi_submit
from nanobody_opt import rules
from nanobody_opt import pr100_fitness_reference as reference
from nanobody_opt.pr100_live_trial_export import METRICS

FIXTURE = Path(__file__).parent / 'fixtures/pr100/fitness_selection'


def load(epoch):
    data = json.loads((FIXTURE / f'actual_candidates_{epoch}.json').read_text())
    raw = (FIXTURE / f'field_{epoch}.json').read_bytes()
    field = reference.prepare(raw, expected_sha256=data['field_sha256'],
        target=data['target'], epoch=epoch, metrics=METRICS, **data['scoring_identity'])
    return data, field


def ordered(data, field=None):
    return sorted(data['rows'], key=lambda r: (
        r['selection']['score'] if field is None else field.score(r['metrics']),
        r['sequence_hash']))


def allocate(monkeypatch, data, rows):
    # Eligibility was already recorded in the immutable snapshot. This stub is
    # solely the authentication boundary, so the real downstream allocator can
    # be tested without pretending the fixture carries complete native receipts.
    monkeypatch.setattr(rules, 'verified_fleet', lambda artifact, *a, **kw: artifact)
    artifact = {'epoch': data['epoch'], 'sequence_pool': [r['sequence'] for r in rows],
                'submissions': [], 'handoff_diagnostics': {}}
    selected = pi_submit.sequences_by_uid(artifact, data['epoch'],
                                         crew=[(i, 'public-test-slot') for i in range(13)])
    return [rules.sequence_hash(sequence) for sequence in selected.values()]


@pytest.mark.parametrize('epoch,field_count', [(26023, 93), (26026, 159)])
def test_actual_publication_order_and_eligibility_metadata_reproduce(epoch, field_count):
    data, field = load(epoch)
    assert field.row_count == field_count
    assert max(field.reference_epochs) == epoch - 1
    assert all(e < epoch for e in field.reference_epochs)
    assert [r['sequence_hash'] for r in ordered(data)] == data['published_pool_hashes']
    for row in data['rows']:
        assert hashlib.sha256(row['sequence'].encode()).hexdigest() == row['sequence_hash']
        assert row['layer2_passed'] is True
        assert row['novelty']['validator_novel'] is True
        assert row['novelty']['exact_historical'] is False
        observed_own_rank = sum(1 + sum(
            v > row['metrics'][key] if direction == 'max' else v < row['metrics'][key]
            for v in {other['metrics'][key] for other in data['rows']})
            for key, direction in METRICS.items())
        assert observed_own_rank == row['selection']['score']


def test_epoch26026_real_pi_allocation_reproduces_every_observed_submission(monkeypatch):
    data, _ = load(26026)
    selected = allocate(monkeypatch, data, ordered(data))
    assert len(selected) == 13
    assert set(selected) == {r['sequence_hash'] for r in data['actual_public_observations']}


def test_epoch26026_first_near_copy_is_stronger_on_fixed_field_without_losing_native_tier(monkeypatch):
    data, field = load(26026)
    old_first = ordered(data)[0]
    new_first = ordered(data, field)[0]
    assert old_first['sequence_hash'].startswith('4faae0fc')
    assert new_first['sequence_hash'].startswith('68d7c2fa')
    assert old_first['source'] == new_first['source'] == 'edit_generation'
    assert old_first['parent_sha256'] == new_first['parent_sha256']
    assert old_first['tier'] == new_first['tier'] == 'A'
    assert old_first['repeats'] == new_first['repeats'] == 3
    assert new_first['psh_boundary_margin'] > old_first['psh_boundary_margin']
    assert pi_submit._identity(old_first['sequence'], new_first['sequence']) >= .95
    assert field.score(old_first['metrics']) == 528
    assert field.score(new_first['metrics']) == 414
    assert old_first['metrics']['design_iiptm'] == .27492
    assert new_first['metrics']['design_iiptm'] == .38372
    old_pool = allocate(monkeypatch, data, ordered(data))
    new_pool = allocate(monkeypatch, data, ordered(data, field))
    assert old_first['sequence_hash'] in old_pool and new_first['sequence_hash'] not in old_pool
    assert new_first['sequence_hash'] in new_pool and old_first['sequence_hash'] not in new_pool
    by_hash = {r['sequence_hash']: r for r in data['rows']}
    assert sum(field.score(by_hash[h]['metrics']) for h in old_pool) == 8727
    assert sum(field.score(by_hash[h]['metrics']) for h in new_pool) == 8463
    assert sum(by_hash[h]['tier'] == 'A' for h in old_pool) == 7
    assert sum(by_hash[h]['tier'] == 'A' for h in new_pool) == 7
    assert sum(by_hash[h]['psh_boundary_margin'] for h in new_pool) > sum(
        by_hash[h]['psh_boundary_margin'] for h in old_pool)
    # This is a prediction-only counterfactual, not a fabricated public result.
    assert new_first['sequence_hash'] not in {
        r['sequence_hash'] for r in data['actual_public_observations']}


def test_epoch26023_both_rankings_keep_the_two_best_public_edits(monkeypatch):
    data, field = load(26023)
    expected = ['ddaa2324', 'c4677e7f']
    assert [h[:8] for h in allocate(monkeypatch, data, ordered(data))[:2]] == expected
    assert [h[:8] for h in allocate(monkeypatch, data, ordered(data, field))[:2]] == expected
    public = {r['sequence_hash'][:8]: r for r in data['actual_public_observations']}
    assert public['ddaa2324']['metrics']['design_iiptm'] == .59922
    assert public['c4677e7f']['metrics']['design_iiptm'] == .49687
    # No assumption that the other five simulated slots produced API outcomes.
    assert len(data['actual_public_observations']) == 8


def test_new_pool_reuses_no_stale_epoch26019_field():
    data, _ = load(26026)
    raw = (FIXTURE / 'field_26019.json').read_bytes()
    with pytest.raises(ValueError, match='stale'):
        reference.prepare(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
            target=data['target'], epoch=26026, metrics=METRICS, **data['scoring_identity'])
