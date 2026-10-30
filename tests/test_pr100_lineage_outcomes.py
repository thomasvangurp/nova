"""Lineage and measured-outcome analysis on real recorded public fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nanobody_opt import pr100_lineage_outcomes as lineage
from nanobody_opt import pr100_parent_bank as bank
from nanobody_opt import rules

FIXTURE = Path(__file__).parent / 'fixtures/pr100/parent_feed'
PUBLIC = json.loads((FIXTURE / 'public_board_26028.json').read_text())
TARGET = 'Q9NZQ7'


def public_rows():
    return bank.public_epoch(copy.deepcopy(PUBLIC), TARGET, 26028, 26030)['rows']


def test_metric_directions_match_the_pinned_scorer_config():
    directions = lineage.metric_directions()
    assert directions['design_iiptm'] == 'max'
    assert directions['liability_score'] == 'min'
    assert directions['interaction_pae'] == 'min'
    assert set(directions) >= {'design_iiptm', 'liability_score', 'interaction_pae'}


def test_cell_uses_the_requested_cell_then_the_challenge_cell():
    challenge = {'allowed_v': 'IGHV3-74', 'allowed_cdr3_length': 10}
    edit = {'requested_v': 'IGHV3-74', 'requested_cdr3_length': 10}
    assert lineage.cell_of(edit, challenge)['cell'] == 'IGHV3-74:10'
    assert lineage.cell_of({}, challenge)['cell'] == 'IGHV3-74:10'
    assert lineage.cell_of({}, {}).get('cell') is None
    assert lineage.cell_of({}, {}).get('unknown') == 'cell_unknown'


def test_edit_kind_distinguishes_each_lane():
    assert lineage.edit_kind({'source': 'edit_generation',
                              'proposal_lane': 'lineage_cdr_single_substitution',
                              'edit_policy': 'chemistry_conservative_v3'}) == {
        'kind': 'lineage_cdr_single_substitution',
        'policy': 'chemistry_conservative_v3', 'source': 'edit_generation'}
    assert lineage.edit_kind({'source': 'fresh_baseline_proposal',
                              'reference_stratum': 'unseeded_control'})['kind'] == 'unseeded_control'
    assert lineage.edit_kind({'source': 'existing_native_reference'})['kind'] == 'existing_native_reference'
    assert lineage.edit_kind({}).get('unknown') == 'source_unknown'


def test_parent_origin_resolves_real_owner_epoch_and_coldkey_from_public_observations():
    rows = public_rows()
    index = lineage.bank_index(rows)
    parent = rows[0]
    child = {'source': 'edit_generation', 'parent_sequence': parent['sequence'],
             'proposal_lane': 'framework_single_substitution'}
    origin = lineage.parent_origin(child, index)
    assert origin['resolved'] is True
    assert origin['parent_hash'] == parent['sequence_hash']
    assert origin['kind'] == 'edit_parent'
    assert origin['parent_epochs'] == [26028]
    assert origin['parent_coldkeys'] == [parent['coldkey']]
    assert origin['parent_uids'] == [parent['uid']]


def test_parent_origin_exports_honest_unknowns_when_parent_is_absent():
    child = {'source': 'edit_generation', 'parent_sequence': 'ACDEFGHIKLMNPQRSTVWY' * 4}
    origin = lineage.parent_origin(child, lineage.bank_index(public_rows()))
    assert origin['resolved'] is False
    assert origin['parent_hash'] == rules.sequence_hash(child['parent_sequence'])
    assert origin['unknown_reason'] == 'parent_not_in_bank'
    no_bank = lineage.parent_origin(child, None)
    assert no_bank['unknown_reason'] == 'parent_bank_unavailable'
    assert lineage.parent_origin({'source': 'unseeded_control'}, {}) == {
        'resolved': False, 'parent_hash': None, 'kind': None,
        'unknown_reason': 'no_parent_identity'}


def test_selected_lineage_combines_cell_edit_kind_and_parent_origin_with_unknowns():
    rows = public_rows()
    index = lineage.bank_index(rows)
    parent = rows[0]
    challenge = {'allowed_v': 'IGHV3-74', 'allowed_cdr3_length': 10}
    child = {'source': 'edit_generation', 'parent_sequence': parent['sequence'],
             'proposal_lane': 'lineage_haplotype', 'edit_policy': 'balanced_v2',
             'requested_v': 'IGHV3-74', 'requested_cdr3_length': 10,
             'sequence': 'ACDEFGHIKLMNPQRSTVWY' * 5}
    value = lineage.selected_lineage(child, index, challenge)
    assert value['cell']['cell'] == 'IGHV3-74:10'
    assert value['edit_kind']['kind'] == 'lineage_haplotype'
    assert value['parent_origin']['resolved'] is True
    assert value['unknowns'] == []
    unknown = lineage.selected_lineage({'source': 'edit_generation',
                                        'parent_sequence': 'W' * 120}, index, challenge)
    assert 'parent_not_in_bank' in unknown['unknowns']


def _recorded_epoch(root, epoch, *, scores, rows, status='published', identity=None,
                    cell=('IGHV3-74', 10), challenge=None, published=None):
    directory = root / str(epoch)
    directory.mkdir(parents=True, exist_ok=True)
    if challenge is None and cell is not None:
        challenge = {'epoch': epoch, 'target': TARGET, 'epoch_length': 361,
                     'allowed_v': cell[0], 'allowed_cdr3_length': cell[1]}
    state = {'schema': 'pr100-epoch-continuation-v1', 'challenge': challenge,
             'score_rows': scores, 'scoring_identity': identity,
             'native_rows': rows, 'pending_native': [], 'generated_hashes': [],
             'native_attempted_hashes': [], 'score_attempted_hashes': [],
             'next_attempt': 1, 'proposal_offset': 0}
    (directory / 'continuation.json').write_text(json.dumps(state))
    (directory / 'receipt.json').write_text(json.dumps({'epoch': epoch, 'status': status, 'attempt': 1}))
    if published is not None:
        members = [{'sequence_hash': rules.sequence_hash(s), 'tier_rank': 0, 'psh_mean': None}
                   for s in published]
        (directory / 'published-quality.json').write_text(json.dumps(
            {'epoch': epoch, 'quality': {'tier_a': 0, 'count': len(members), 'members': members}}))


def _seq(tag, length=120):
    return (('ACDEFGHIKLMNPQRSTVWY' * 7)[:length - len(tag)] + tag).upper()


IDENTITY = {'scorer_sha256': 'a' * 64, 'settings_sha256': 'b' * 64,
            'settings_file_sha256': 'c' * 64, 'target': TARGET}


def test_completed_epochs_is_the_fixed_strictly_past_window_not_receipt_filtered(tmp_path):
    # Only a few epochs have any recorded state; the window must not skip to
    # older epochs or depend on receipt status.
    for epoch in (26021, 26022, 26029, 26030):
        _recorded_epoch(tmp_path, epoch, scores={}, rows=[], status='published')
    assert lineage.completed_epochs(tmp_path, 26031, 10) == list(range(26021, 26031))
    assert lineage.completed_epochs(tmp_path, 26031, 3) == [26028, 26029, 26030]
    # Never leaks the current or future epoch.
    window = lineage.completed_epochs(tmp_path, 26030, 10)
    assert window == list(range(26020, 26030))
    assert all(epoch < 26030 for epoch in window)
    # Clamped at the PR100 activation epoch.
    assert lineage.completed_epochs(tmp_path, 25960, 10) == list(range(25955, 25960))


def test_top_bottom5_orders_within_one_protocol_and_isolates_other_fields(tmp_path):
    for i in range(26021, 26031):
        rows, scores = [], {}
        for k in range(3):
            sequence = _seq(f'{i}:{k}')
            value = 0.1 * (i - 26021) + 0.01 * k
            rows.append({'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                         'source': 'fresh_baseline_proposal',
                         'reference_stratum': 'unseeded_control'})
            scores[sequence] = {TARGET: {'design_iiptm': value}}
        _recorded_epoch(tmp_path, i, scores=scores, rows=rows, identity=IDENTITY)
    before = 26031
    outcomes = lineage.scored_outcomes(tmp_path, lineage.completed_epochs(tmp_path, before, 10),
                                       None, 'design_iiptm')
    ranking = lineage.top_bottom5(outcomes, metric_field='design_iiptm', direction='max')
    assert ranking['primary_protocol'] == rules.digest(IDENTITY)
    assert len(ranking['top']) == 5 and len(ranking['bottom']) == 5
    top_values = [o['measured']['design_iiptm'] for o in ranking['top']]
    bottom_values = [o['measured']['design_iiptm'] for o in ranking['bottom']]
    assert top_values == sorted(top_values, reverse=True)
    assert bottom_values == sorted(bottom_values)
    assert min(top_values) >= max(bottom_values)
    assert ranking['comparable_count'] == 30


def test_top_bottom5_never_mixes_different_scoring_identities(tmp_path):
    for i in range(26021, 26025):
        sequence = _seq(str(i))
        value = 0.1 * (i - 26021)
        other = dict(IDENTITY, scorer_sha256='d' * 64 if i == 26024 else IDENTITY['scorer_sha256'])
        _recorded_epoch(tmp_path, i,
            scores={sequence: {TARGET: {'design_iiptm': value}}}, rows=[{
                'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                'source': 'fresh_baseline_proposal', 'reference_stratum': 'unseeded_control'}],
            identity=other)
    before = 26025
    outcomes = lineage.scored_outcomes(tmp_path, lineage.completed_epochs(tmp_path, before, 10),
                                       None, 'design_iiptm')
    ranking = lineage.top_bottom5(outcomes, metric_field='design_iiptm', direction='max')
    assert ranking['primary_protocol'] == rules.digest(dict(IDENTITY, scorer_sha256='d' * 64))
    assert ranking['comparable_count'] == 1
    assert len(ranking['non_comparable']) == 3


def test_scored_outcomes_uses_each_epochs_own_cell_and_unknown_challenge(tmp_path):
    for i, cell in ((26021, ('IGHV3-74', 10)), (26022, ('IGHV3-66', 12))):
        sequence = _seq(str(i))
        _recorded_epoch(tmp_path, i,
            scores={sequence: {TARGET: {'design_iiptm': 0.5}}},
            rows=[{'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                   'source': 'fresh_baseline_proposal',
                   'reference_stratum': 'unseeded_control'}],
            identity=IDENTITY, cell=cell)
    missing = _seq('MISSING')
    directory = tmp_path / '26023'
    directory.mkdir(parents=True)
    (directory / 'continuation.json').write_text(json.dumps({
        'schema': 'x', 'score_rows': {missing: {TARGET: {'design_iiptm': 0.5}}},
        'scoring_identity': IDENTITY, 'native_rows': [
            {'sequence': missing, 'sequence_hash': rules.sequence_hash(missing),
             'source': 'fresh_baseline_proposal', 'reference_stratum': 'unseeded_control'}],
        'pending_native': []}))
    (directory / 'receipt.json').write_text(json.dumps({'epoch': 26023, 'status': 'published'}))
    outcomes = lineage.scored_outcomes(tmp_path, [26021, 26022, 26023], None, 'design_iiptm')
    by_epoch = {o['epoch']: o for o in outcomes}
    assert by_epoch[26021]['lineage']['cell']['cell'] == 'IGHV3-74:10'
    assert by_epoch[26022]['lineage']['cell']['cell'] == 'IGHV3-66:12'
    assert 'cell_unknown' in by_epoch[26023]['lineage']['unknowns']
    assert 'historical_challenge_unknown' in by_epoch[26023]['unknowns']
    assert 'scoring_identity_target_mismatch' in by_epoch[26023]['unknowns']
    assert by_epoch[26023]['target'] is None
    assert by_epoch[26023]['protocol'] is None


def test_incomplete_or_mismatched_scoring_identity_is_unknown(tmp_path):
    sequence = _seq('X')
    _recorded_epoch(tmp_path, 26021,
        scores={sequence: {TARGET: {'design_iiptm': 0.5}}},
        rows=[{'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
               'source': 'fresh_baseline_proposal', 'reference_stratum': 'unseeded_control'}],
        identity={'scorer_sha256': 'a' * 64})
    outcomes = lineage.scored_outcomes(tmp_path, [26021], None, 'design_iiptm')
    assert outcomes[0]['protocol'] is None
    assert 'scoring_identity_incomplete' in outcomes[0]['unknowns']
    _recorded_epoch(tmp_path, 26022,
        scores={sequence: {TARGET: {'design_iiptm': 0.5}}},
        rows=[{'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
               'source': 'fresh_baseline_proposal', 'reference_stratum': 'unseeded_control'}],
        identity=dict(IDENTITY, target='OTHER'))
    outcomes = lineage.scored_outcomes(tmp_path, [26022], None, 'design_iiptm')
    assert 'scoring_identity_target_mismatch' in outcomes[0]['unknowns']


def test_published_flag_distinguishes_scored_from_published_without_inventing_chain(tmp_path):
    published_seq, unpublished = _seq('PUB'), _seq('UNPUB')
    rows = [{'sequence': s, 'sequence_hash': rules.sequence_hash(s),
             'source': 'fresh_baseline_proposal', 'reference_stratum': 'unseeded_control'}
            for s in (published_seq, unpublished)]
    _recorded_epoch(tmp_path, 26021,
        scores={published_seq: {TARGET: {'design_iiptm': 0.6}},
                unpublished: {TARGET: {'design_iiptm': 0.4}}},
        rows=rows, identity=IDENTITY, published=[published_seq])
    outcomes = {o['sequence_hash']: o
                for o in lineage.scored_outcomes(tmp_path, [26021], None, 'design_iiptm')}
    assert outcomes[rules.sequence_hash(published_seq)]['published'] is True
    assert outcomes[rules.sequence_hash(unpublished)]['published'] is False
    assert all(o['validator_outcome'] is None for o in outcomes.values())
    value = lineage.build(tmp_path, None, {'epoch': 26022, 'target': TARGET})
    assert value['populations'] == {'locally_scored': 2, 'published': 1,
                                    'published_unknown': 0, 'validator_scored': 0}


def test_published_evidence_survives_a_later_recoverable_failure(tmp_path):
    sequence = _seq('SURVIVE')
    _recorded_epoch(tmp_path, 26029,
        scores={sequence: {TARGET: {'design_iiptm': 0.5}}},
        rows=[{'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
               'source': 'fresh_baseline_proposal', 'reference_stratum': 'unseeded_control'}],
        identity=IDENTITY, status='published', published=[sequence])
    # A later attempt in the same epoch failed; the earlier publication remains.
    (tmp_path / '26029' / 'receipt.json').write_text(json.dumps(
        {'epoch': 26029, 'status': 'recoverable_failure', 'attempt': 3}))
    before = 26030
    assert 26029 in lineage.completed_epochs(tmp_path, before, 10)
    value = lineage.build(tmp_path, None, {'epoch': before, 'target': TARGET})
    assert value['epoch_evidence']['26029']['receipt_status'] == 'recoverable_failure'
    assert value['epoch_evidence']['26029']['published_count'] == 1
    assert value['outcome_count'] == 1
    assert value['top'][0]['published'] is True
    assert value['populations']['published'] == 1


def test_missing_epochs_are_exposed_not_backfilled(tmp_path):
    value = lineage.build(tmp_path, None, {'epoch': 26030, 'target': TARGET})
    assert value['completed_epochs'] == list(range(26020, 26030))
    assert value['outcome_count'] == 0
    for epoch in value['completed_epochs']:
        assert value['epoch_evidence'][str(epoch)]['state_recorded'] is False
        assert value['epoch_evidence'][str(epoch)]['receipt_status'] is None
        assert value['epoch_evidence'][str(epoch)]['published_count'] is None


def test_build_exports_lineage_and_honest_unknowns_for_unmeasured_sequences(tmp_path):
    index = lineage.bank_index(public_rows())
    parent = public_rows()[0]
    parent_sequence = parent['sequence']
    child = _seq('CHILD', len(parent_sequence))
    for i in range(26021, 26026):
        measured = _seq(f'M{i}', len(parent_sequence))
        rows = [
            {'sequence': measured, 'sequence_hash': rules.sequence_hash(measured),
             'source': 'edit_generation', 'parent_sequence': parent_sequence,
             'proposal_lane': 'lineage_haplotype', 'edit_policy': 'balanced_v2',
             'requested_v': 'IGHV3-74', 'requested_cdr3_length': 10},
            {'sequence': child, 'sequence_hash': rules.sequence_hash(child),
             'source': 'edit_generation', 'parent_sequence': parent_sequence,
             'proposal_lane': 'lineage_cdr_single_substitution',
             'edit_policy': 'balanced_v2', 'requested_v': 'IGHV3-74',
             'requested_cdr3_length': 10},
        ]
        _recorded_epoch(tmp_path, i,
            scores={measured: {TARGET: {'design_iiptm': 0.1 * (i - 26020)}},
                    child: {TARGET: {'design_ptm': 0.5}}},
            rows=rows, identity=IDENTITY)
    before = 26026
    value = lineage.build(tmp_path, public_rows(), {'epoch': before, 'target': TARGET},
                          metric_field='design_iiptm')
    assert value['schema'] == lineage.SCHEMA
    assert value['completed_epochs'] == list(range(26016, 26026))
    assert value['outcome_count'] == 10
    assert len(value['top']) == 5 and value['bottom'] == []
    top = value['top'][0]
    assert top['lineage']['parent_origin']['resolved'] is True
    assert top['lineage']['parent_origin']['parent_coldkeys'] == [parent['coldkey']]
    assert top['lineage']['cell']['cell'] == 'IGHV3-74:10'
    # The unmeasured child appears once as an honest unknown, never a fake score.
    unknowns = [o for o in value['non_comparable'] if 'metric_not_measured' in o['unknowns']]
    assert any(o['lineage']['edit_kind']['kind'] == 'lineage_cdr_single_substitution'
               for o in unknowns)


def test_record_writes_a_durable_artifact_and_survives_a_missing_bank(tmp_path):
    parent = public_rows()[0]['sequence']
    for i in range(26021, 26024):
        sequence = _seq(str(i), len(parent))
        _recorded_epoch(tmp_path, i,
            scores={sequence: {TARGET: {'design_iiptm': 0.1 * (i - 26020)}}},
            rows=[{'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                   'source': 'edit_generation', 'parent_sequence': parent,
                   'proposal_lane': 'lineage_haplotype', 'edit_policy': 'balanced_v2',
                   'requested_v': 'IGHV3-74', 'requested_cdr3_length': 10}],
            identity=IDENTITY)
    from test_pr100_live_controller import config as controller_config
    config = controller_config(tmp_path)
    challenge = {'epoch': 26024, 'target': TARGET, 'allowed_v': 'IGHV3-74',
                 'allowed_cdr3_length': 10}
    summary = lineage.record(str(tmp_path), challenge, config)
    assert summary['status'] == 'complete'
    value = json.loads((tmp_path / 'lineage-outcomes.json').read_text())
    assert value['schema'] == lineage.SCHEMA
    assert value['status'] == 'complete'
    assert value['outcome_count'] == 3
    # No parent bank is configured, so every edit parent resolves its hash but
    # exports an honest "bank unavailable" unknown instead of invented lineage.
    assert all(o['lineage']['parent_origin']['parent_hash'] == rules.sequence_hash(parent)
               and o['lineage']['parent_origin']['unknown_reason'] == 'parent_bank_unavailable'
               for o in value['top'])


def test_record_never_raises_on_disk_or_parse_failure(tmp_path, monkeypatch):
    from test_pr100_live_controller import config as controller_config
    config = controller_config(tmp_path)
    challenge = {'epoch': 26024, 'target': TARGET}

    def boom(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(lineage, 'build', boom)
    assert lineage.record(str(tmp_path), challenge, config)['status'] == 'unavailable'
    value = json.loads((tmp_path / 'lineage-outcomes.json').read_text())
    assert value['status'] == 'unavailable' and value['error_type'] == 'OSError'


def test_controller_lineage_hook_never_fails_the_epoch(monkeypatch):
    from nanobody_opt import pr100_epoch_continuation as continuation

    def boom(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr('nanobody_opt.pr100_lineage_outcomes.record', boom)
    result = continuation.record_lineage({'state_root': '/nonexistent'},
                                         {'epoch': 26030, 'target': TARGET})
    assert result == {'status': 'unavailable'}


def test_load_bank_rows_binds_the_qualified_bank_and_refuses_drift(tmp_path):
    from test_pr100_live_controller import config as controller_config
    config = controller_config(tmp_path)
    config['hosts']['slave'].update(reference_bank_sha256='a' * 64)
    config['parent_bank_feed'] = {'enabled': True, 'cpu_affinity': [64, 65, 66, 67],
                                  'remote_directory': str(tmp_path / 'remote-feed')}
    cfg = bank.settings(config)
    directory = Path(cfg['state_directory']) / TARGET
    directory.mkdir(parents=True)
    rows = public_rows()
    value = {'schema': bank.SCHEMA, 'target': TARGET, 'before_epoch': 26030,
             'source_sha': config['source_sha'], 'runtime_sha256': 'e' * 64,
             'annotation_identity': 'f' * 64, 'native_qualified': False,
             'novelty_qualified': False, 'rows': rows}
    digest = bank.sha(bank.raw(value))
    bank.immutable(directory / ('bank-' + digest + '.json'), bank.raw(value))
    bank.atomic(directory / 'current.json', {'schema': bank.POINTER, 'target': TARGET,
        'bank_sha256': digest, 'qualification_sha256': '0' * 64})
    challenge = {'epoch': 26030, 'target': TARGET}
    assert lineage.load_bank_rows(config, challenge) == rows
    # A disabled feed or a changed bank byte must not silently reuse provenance.
    config['parent_bank_feed']['enabled'] = False
    assert lineage.load_bank_rows(config, challenge) is None
    config['parent_bank_feed']['enabled'] = True
    (directory / ('bank-' + digest + '.json')).write_bytes(b'corrupted')
    assert lineage.load_bank_rows(config, challenge) is None
