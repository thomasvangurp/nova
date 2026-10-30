"""Real measured ranking regressions; exporter eligibility is a labelled stub boundary."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from nanobody_opt import rules
from nanobody_opt import pr100_fitness_reference as reference
from nanobody_opt.pr100_live_trial_export import build, METRICS

FIXTURE = Path(__file__).parent / 'fixtures/pr100/fitness_selection'
DATA = json.loads((FIXTURE / 'real_candidates_26019.json').read_text())
RAW = (FIXTURE / 'field_26019.json').read_bytes()
SETTINGS = {'test_scoring_protocol': 'explicit_export_boundary_fixture'}


def prepare(raw=RAW, **changes):
    args = {'expected_sha256': hashlib.sha256(raw).hexdigest(), 'target': DATA['target'],
        'epoch': DATA['epoch'], 'metrics': METRICS, 'scorer_sha256': 'a' * 64,
        'settings_sha256': rules.digest(SETTINGS), 'settings_file_sha256': 'b' * 64}
    args.update(changes)
    return reference.prepare(raw, **args)


def published(): return [row for row in DATA['baseline'] if row['published']]


def own_score(row, cohort):
    return sum(1 + sum(v > row['metrics'][key] if direction == 'max' else v < row['metrics'][key]
                       for v in {other['metrics'][key] for other in cohort})
               for key, direction in METRICS.items())


def test_actual_five_submission_order_reversals_against_frozen306_field():
    field = prepare(); rows = published()
    assert field.source_sha256 == DATA['field_sha256']
    assert [r['sequence_hash'][:8] for r in sorted(rows, key=lambda r: own_score(r, rows))] == [
        'b9a16b61', 'c60c8779', 'f14a3943', 'eb5e443a', 'c9dab9a5']
    assert [(r['sequence_hash'][:8], field.score(r['metrics'])) for r in
            sorted(rows, key=lambda r: field.score(r['metrics']))] == [
        ('c60c8779', 749), ('b9a16b61', 795), ('c9dab9a5', 811), ('eb5e443a', 865), ('f14a3943', 878)]


def test_all_real_experimental_and_heldout_scores_reproduce_hash_bound_report():
    field = prepare()
    for row in DATA['experimental_initial']:
        assert field.score(row['metrics']) == row['single_pass_dense10_rank_sum']
    for row in DATA['experimental_heldout']:
        scores = [field.score(draw['metrics']) for draw in row['draws']]
        assert scores == [draw['dense10_rank_sum'] for draw in row['draws']]
        assert sum(scores) / len(scores) == pytest.approx(row['mean_dense10_rank_sum'])


def test_adding_measured_other_candidates_cannot_change_existing_fixed_scores():
    field = prepare(); before = {r['sequence_hash']: field.selection(r['metrics']) for r in published()}
    combined = published() + DATA['experimental_initial']
    after = {r['sequence_hash']: field.selection(r['metrics']) for r in combined}
    assert all(after[key] == value for key, value in before.items())
    assert own_score(published()[0], published()) != own_score(published()[0], combined)


@pytest.mark.parametrize('fault', ['target', 'current_epoch', 'future_epoch', 'duplicates', 'stale',
    'empty', 'missing_metric', 'bool', 'nan', 'infinity', 'hash', 'oversize'])
def test_unusable_reference_cannot_silently_change_ranking(fault):
    value = json.loads(RAW); changes = {}
    if fault == 'target': value['target'] = 'WRONG'
    elif fault == 'current_epoch': value['epochs'][0] = DATA['epoch']
    elif fault == 'future_epoch': value['epochs'][0] = DATA['epoch'] + 1
    elif fault == 'duplicates': value['epochs'][0] = value['epochs'][1]
    elif fault == 'stale': changes['epoch'] = 26030
    elif fault == 'empty': value['rows'] = []
    elif fault == 'missing_metric': value['rows'][0].pop('design_iiptm')
    elif fault == 'bool': value['rows'][0]['design_iiptm'] = True
    elif fault == 'nan': value['rows'][0]['design_iiptm'] = float('nan')
    elif fault == 'infinity': value['rows'][0]['design_iiptm'] = float('inf')
    elif fault == 'hash': changes['expected_sha256'] = '0' * 64
    raw = json.dumps(value).encode()
    if fault == 'oversize': raw = b' ' * (8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError): prepare(raw, **changes)


@pytest.fixture
def exported_inputs(monkeypatch):
    # This fixture tests transport/ranking only. It never claims that synthetic
    # sequence strings or these stubbed annotations are publishable evidence.
    monkeypatch.setattr(rules, 'validate_evidence', lambda *args: None)
    challenge = rules.challenge(rules.pinned_config(DATA['epoch']), DATA['epoch'], '0x' + 'a' * 64, 361)
    validity = {'challenge': challenge, 'target': DATA['target'], 'archive_sha256': 'c' * 64,
                'archive_checked_at_unix': 1.0, 'candidates': []}
    measured = {}
    ids = {}
    for index, row in enumerate(published()):
        sequence = 'A' * 119 + 'CDEFG'[index]; ids[row['sequence_hash'][:8]] = sequence
        validity['candidates'].append({'sequence': sequence, 'layer2_passed': True,
            'pr100_evidence': {'synthetic_eligibility_test_boundary': True},
            'novelty': {'validator_novel': True, 'exact_historical': False,
                        'coarse_candidates': 7, 'maximum_identity': 0.81}})
        measured[sequence] = {DATA['target']: row['metrics']}
    scored = {'target': DATA['target'], 'per_nanobody_components': {'0': measured}}
    return validity, scored, ids


def export(inputs, field):
    validity, scored, _ = inputs
    return build(validity, scored, SETTINGS, scorer_sha256='a' * 64,
                 settings_file_sha256='b' * 64, now=2.0, reference_field=field)


def test_opt_in_export_retains_every_valid_partial_member_and_honest_labels(exported_inputs):
    old = export(exported_inputs, None); new = export(exported_inputs, prepare())
    assert set(new['sequence_pool']) == set(old['sequence_pool'])
    ids = exported_inputs[2]
    assert old['sequence_pool'][0] == ids['b9a16b61'] and new['sequence_pool'][0] == ids['c60c8779']
    for cp in new['sequence_pool_checkpoints'].values():
        assert cp['selection']['reference_count'] == 306
        assert cp['selection']['reference_sha256'] == DATA['field_sha256']
        assert cp['selection']['reference_latest_epoch_age'] == 1
        assert cp['selection']['multipass_measured'] is False
        assert cp['exact']['observation_count'] == 1
        assert cp['validation']['boltzgen']['validator_matched'] is False
        assert cp['validation']['boltzgen']['provenance']['scorer_sha256'] == 'a' * 64


@pytest.mark.parametrize('fault', ['native_fail', 'novelty_fail', 'historical', 'unscored', 'native_evidence_rejected'])
def test_reference_ranking_cannot_bypass_existing_publication_evidence(exported_inputs, monkeypatch, fault):
    validity, scored, _ = exported_inputs; row = validity['candidates'][0]
    if fault == 'native_fail': row['layer2_passed'] = False
    elif fault == 'novelty_fail': row['novelty']['validator_novel'] = False
    elif fault == 'historical': row['novelty']['exact_historical'] = True
    elif fault == 'unscored': scored['per_nanobody_components']['0'].pop(row['sequence'])
    else:
        def rejected(*args): raise ValueError('real gate refuses evidence')
        monkeypatch.setattr(rules, 'validate_evidence', rejected)
    if fault == 'unscored':
        assert row['sequence'] not in export(exported_inputs, prepare())['sequence_pool']
    else:
        with pytest.raises(ValueError): export(exported_inputs, prepare())


@pytest.mark.parametrize('field', [lambda: prepare(epoch=26020),
    lambda: prepare(scorer_sha256='d'*64), lambda: prepare(settings_sha256='e'*64),
    lambda: prepare(settings_file_sha256='f'*64)])
def test_field_is_bound_to_current_challenge_and_exact_measured_scorer(exported_inputs, field):
    with pytest.raises(ValueError, match='reference field'): export(exported_inputs, field())
