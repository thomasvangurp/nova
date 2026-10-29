"""Missing-cell proposals only: inert annotation, no native/model/scoring calls."""
import copy
import json
import sys
import types

import pytest

from nanobody_opt import pr100_fresh_budget as budget
from nanobody_opt import pr100_live_worker as worker
from nanobody_opt import pr100_proposal_pilot as proposal
from nanobody_opt import pr100_reference_donors as donors
from nanobody_opt import rules


# Cell counts from the frozen 78-row bank, SHA256
# 58525a3a3d21b25ccc446efdf5ce437929895d53286e851c5dfc3c34f4ea4f36.
# Identities here are synthetic; no historical sequences or scores are needed.
BANK_CELLS = {'IGHV3-23:16': 6, 'IGHV3-23:18': 68,
              'IGHV3-23:5': 2, 'IGHV3-7:16': 2}
MISSING_CELLS = [(25969, 'IGHV3-23', 17), (25970, 'IGHV3-48', 17),
                 (25971, 'IGHV3-43', 21)]


def panel():
    j = {p: ('A' if p >= 115 else '-') for p in range(1, 129)}
    return {'V': {g + '*01': {p: 'A' for p in range(1, 129)} for g in proposal.GENES},
            'J': {'IGHJ4*01': dict(j), 'IGHJ6*01': dict(j)}}


@pytest.fixture
def bank():
    rows = []
    for cell, count in BANK_CELLS.items():
        gene, length = cell.split(':')
        for index in range(count):
            sequence = proposal.propose(panel(), gene, int(length), seed=198,
                                        ordinal=index)['sequence']
            rows.append({'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                         'cells': [cell], 'source': 'full_native_census',
                         'old_target_binding_score': index})
    assert len({r['sequence_hash'] for r in rows}) == 78
    return rows


def challenge(epoch=25969, gene='IGHV3-23', length=17):
    return {'epoch': epoch, 'target': rules.TARGET,
            'allowed_v': gene, 'allowed_cdr3_length': length}


@pytest.mark.parametrize('epoch,gene,length', MISSING_CELLS)
def test_actual_missing_cells_choose_bounded_ranked_donors(bank, epoch, gene, length):
    exact, chosen, evidence = donors.select(bank, challenge(epoch, gene, length), 16,
                                           fallback=True)
    assert exact == [] and len(chosen) == 16
    assert evidence['bank_cells'] == BANK_CELLS
    assert evidence['exact_cell_available'] == 0 and evidence['fallback_used']
    expected = 'same_v_nearby_length' if epoch == 25969 else 'other_v_nearby_length'
    assert evidence['selected_by_stratum'] == {expected: 16}
    if epoch == 25969:
        assert [d['reference_length_distance'] for _, d in chosen] == [1]*16
    elif epoch == 25970:
        assert {d['reference_length_distance'] for _, d in chosen} == {1}
    else:
        assert {d['reference_cell'] for _, d in chosen} == {'IGHV3-23:18'}
    reversed_bank = list(reversed(copy.deepcopy(bank)))
    for row in reversed_bank:
        row['old_target_binding_score'] = -999999
    assert donors.select(reversed_bank, challenge(epoch, gene, length), 16,
                         fallback=True)[2] == evidence


def test_same_v_precedes_other_v_then_fills_remaining_slots(bank):
    exact, chosen, evidence = donors.select(bank, challenge(gene='IGHV3-7', length=17),
                                           4, fallback=True)
    assert exact == []
    assert [d['reference_stratum'] for _, d in chosen] == (
        ['same_v_nearby_length']*2 + ['other_v_nearby_length']*2)
    assert evidence['selected_by_stratum'] == {'same_v_nearby_length': 2,
                                               'other_v_nearby_length': 2}


def test_exact_cell_selection_keeps_original_order_and_rows(bank):
    c = challenge(length=18)
    original = sorted((r for r in bank if 'IGHV3-23:18' in r['cells']),
                      key=lambda r: rules.digest([c['epoch'], r['sequence_hash']]))[:16]
    for enabled in (False, True):
        exact, chosen, coverage = donors.select(bank, c, 16, fallback=enabled)
        assert exact == original and [r for r, _ in chosen] == original
        assert not coverage['fallback_used']
        assert all(d['reference_stratum'] == 'exact_cell' for _, d in chosen)


def test_default_off_and_empty_bank_do_not_invent_donors(bank):
    assert donors.enabled({}) is False
    assert donors.select(bank, challenge(), 16)[1] == []
    assert donors.select([], challenge(), 16, fallback=True)[1] == []


@pytest.mark.parametrize('value', [None, 0, 1, 'true', {}, []])
def test_opt_in_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match='boolean'):
        donors.enabled({'missing_cell_donor_fallback': value})


@pytest.mark.parametrize('change', [{'sequence_hash': 'f'*64}, {'cells': []},
                                    {'cells': ['IGHV3-23:99']}])
def test_fallback_refuses_bad_identity_or_cell(bank, change):
    bank[0].update(change)
    with pytest.raises(ValueError, match='donor'):
        donors.select(bank, challenge(), 16, fallback=True)


@pytest.fixture
def sequence_runner(monkeypatch, tmp_path, bank):
    from nanobody_opt import gates
    calls = []
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(
        Chain=lambda sequence, **kw: types.SimpleNamespace(cdr3_seq=sequence[104:-11])))
    monkeypatch.setattr(worker.rules, 'pinned_config', lambda epoch: {'nanobody_requirements': {}})
    monkeypatch.setattr(worker.pr100_filter, 'validate_annotation_runtime', lambda: None)
    monkeypatch.setattr(worker.pr100_filter, 'helper', lambda name:
                        types.SimpleNamespace(_anarci_panel=lambda species: (panel(), None)))
    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached',
                        lambda sequence: calls.append(sequence) or {'fixture_annotation': True})
    monkeypatch.setattr(worker.pr100_filter, 'annotation_identity', lambda: 'a'*64)
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: '')
    monkeypatch.setattr(gates, 'check_sequence', lambda *a: False)
    def forbidden(*a, **kw):
        raise AssertionError('no scientific profiling or novelty in this proposal test')
    monkeypatch.setattr(worker.pr100_filter, 'develop', forbidden)
    monkeypatch.setattr(worker, 'check_novelty', forbidden)
    reference_path = tmp_path / 'bank.json'
    reference_path.write_text(json.dumps({'rows': bank}))
    runs = []
    def run(c=None, **overrides):
        directory = tmp_path / str(len(runs))
        directory.mkdir()
        (directory / 'archive.csv').write_text('sequence\nAAA\n')
        job = {'challenge': c or challenge(), 'references': str(reference_path),
               'archive_sha256': worker.file_digest(directory / 'archive.csv'),
               'archive_fetched_at': 1, 'proposals': 40, **overrides}
        runs.append(job)
        return worker.sequence_phase(job, directory)
    run.annotation_calls = calls
    return run


@pytest.mark.parametrize('epoch,gene,length', MISSING_CELLS)
def test_current_cell_fresh_unique_half_control_with_full_source_accounting(
        sequence_runner, bank, epoch, gene, length):
    result = sequence_runner(challenge(epoch, gene, length),
                             missing_cell_donor_fallback=True, proposals=600)
    candidates = result['candidates']
    assert len(candidates) == len(result['generated_sequence_hashes']) == 600
    assert result['existing_references'] == 0 and not result['novelty_checked']
    assert result['proposal_source_counts']['unseeded_control'] == 300
    assert result['proposal_source_counts'] == result['eligible_proposal_source_counts']
    assert len(sequence_runner.annotation_calls) == 600
    assert {r['sequence_hash'] for r in candidates}.isdisjoint(r['sequence_hash'] for r in bank)
    assert all(r['source'] == 'fresh_baseline_proposal' for r in candidates)
    assert all(r['requested_v'] == gene and r['requested_cdr3_length'] == length for r in candidates)
    assert all(r['v_reference_allele'] == gene+'*01' and len(r['sequence']) == 104+length+11
               for r in candidates)
    assert all(not r['acceptance_qualified'] and not r['target_scored'] for r in candidates)
    assert sum(r['reference_hash'] is not None for r in candidates) == 300
    assert all('old_target_binding_score' not in r for r in candidates)
    repeat = sequence_runner(challenge(epoch, gene, length),
                             missing_cell_donor_fallback=True, proposals=600)
    assert repeat['generated_sequence_hashes'] == result['generated_sequence_hashes']


def test_default_off_missing_cell_and_enabled_exact_cell_preserve_baseline(sequence_runner):
    missing = sequence_runner()
    assert missing['proposal_source_counts'] == {'unseeded_control': 40}
    exact = sequence_runner(challenge(length=18))
    enabled = sequence_runner(challenge(length=18), missing_cell_donor_fallback=True)
    assert exact['candidates'] == enabled['candidates']
    assert exact['generated_sequence_hashes'] == enabled['generated_sequence_hashes']
    assert exact['existing_references'] == 16


def test_accidental_parent_reuse_is_dropped_without_retry_or_budget_growth(
        sequence_runner, bank, monkeypatch):
    original, ordinals = proposal.propose, []
    def collision(*args, **kwargs):
        ordinals.append(kwargs['ordinal'])
        row = original(*args, **kwargs)
        if kwargs['ordinal'] == 0:
            row['sequence'] = bank[0]['sequence']
        return row
    monkeypatch.setattr(proposal, 'propose', collision)
    result = sequence_runner(missing_cell_donor_fallback=True)
    assert ordinals == list(range(40)) and result['next_proposal_offset'] == 40
    assert result['donor_parent_reuse_blocked'] == 1 and len(result['candidates']) == 39
    assert bank[0]['sequence_hash'] not in {r['sequence_hash'] for r in result['candidates']}


def test_seeded_proposals_cannot_bypass_existing_annotation_gate(sequence_runner, monkeypatch):
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: 'germline_mismatch')
    result = sequence_runner(missing_cell_donor_fallback=True)
    assert result['annotation_reasons'] == {'germline_mismatch': 40}
    assert len(sequence_runner.annotation_calls) == 40 and result['candidates'] == []
    assert sum(result['proposal_source_counts'].values()) == 40
    assert result['eligible_proposal_source_counts'] == {}


def test_resume_exclusions_and_shared_fresh_cap_remain_unchanged(sequence_runner):
    first = sequence_runner(missing_cell_donor_fallback=True)
    excluded = first['generated_sequence_hashes']
    rerun = sequence_runner(missing_cell_donor_fallback=True, excluded_sequence_hashes=excluded)
    assert rerun['candidates'] == [] and rerun['excluded_generated'] == 40
    second = sequence_runner(missing_cell_donor_fallback=True, proposal_offset=40,
                              excluded_sequence_hashes=excluded)
    assert second['next_proposal_offset'] == 80 and len(second['candidates']) == 40
    assert set(second['generated_sequence_hashes']).isdisjoint(excluded)
    state = {}
    ledger = budget.ensure(state, {'epoch_fresh_native_limit': 4})
    selected = budget.reserve(ledger, 1, first['candidates'])
    assert len(selected) == 4 and budget.summary(ledger)['remaining'] == 0
    assert budget.reserve(ledger, 1, first['candidates']) == selected
    assert budget.reserve(ledger, 2, second['candidates']) == []
