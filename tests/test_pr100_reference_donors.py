"""Missing-cell proposals only: inert annotation, no native/model/scoring calls."""
from collections import Counter
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
from pr100_germline import real_panel


# Cell counts from a synthetic bank. The L>=12 template emits 8 unique CDR3s
# per cell (4 anchors x 2 head pos-4 draws, deterministic body), so IGHV3-23:18
# is capped at 8 rows rather than the historical 68. Identities are synthetic;
# no historical sequences or scores are needed.
BANK_CELLS = {'IGHV3-23:16': 6, 'IGHV3-23:18': 8,
              'IGHV3-23:5': 2, 'IGHV3-7:16': 2}
MISSING_CELLS = [(25969, 'IGHV3-23', 11), (25970, 'IGHV3-48', 17),
                 (25971, 'IGHV3-43', 21)]


def panel():
    return real_panel()


@pytest.fixture
def bank():
    rows = []
    for cell, count in BANK_CELLS.items():
        gene, length = cell.split(':')
        seen = set()
        ordinal = 0
        while len(seen) < count:
            sequence = proposal.propose(panel(), gene, int(length), seed=198,
                                        ordinal=ordinal)['sequence']
            ordinal += 1
            if sequence in seen:
                continue
            seen.add(sequence)
            rows.append({'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                         'cells': [cell], 'source': 'full_native_census',
                         'old_target_binding_score': ordinal - 1})
    assert len({r['sequence_hash'] for r in rows}) == len(rows)
    return rows


def challenge(epoch=25969, gene='IGHV3-23', length=11):
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
        assert [d['reference_length_distance'] for _, d in chosen] == [5]*6 + [6]*2 + [7]*8
    elif epoch == 25970:
        assert {d['reference_length_distance'] for _, d in chosen} == {1}
    else:
        assert {d['reference_cell'] for _, d in chosen} == {'IGHV3-23:18', 'IGHV3-23:16',
                                                           'IGHV3-7:16'}
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


# ---------------------------------------------------------------------------
# v2 port #3: developability flag + opt-in CDR3-family ceiling on donors.
# ---------------------------------------------------------------------------
def _developability(total=30.0, cdr3=14.0, compact=1.0):
    """Minimal in-range (or caller-overridden) developability profile.

    ``cdr_developability_in_range`` reads only these three red-region fields;
    the remaining native metrics are irrelevant to the donor flag.
    """
    return {'total_cdr_length': total, 'cdr3_length': cdr3, 'cdr3_compactness': compact}


def test_developability_flag_attached_and_applied(bank):
    # The synthetic bank has no profiles, so every donor fails closed (False).
    # Give one row an in-range profile: it must rank first and carry the flag.
    rows = copy.deepcopy(bank)
    rows[0]['developability'] = _developability()
    exact, chosen, evidence = donors.select(rows, challenge(), 16, fallback=True)
    flags = [d['cdr_developability_in_range'] for _, d in chosen]
    assert flags.count(True) == 1 and flags[0] is True
    assert chosen[0][0]['sequence_hash'] == rows[0]['sequence_hash']
    assert evidence['developability_in_range_selected'] == 1
    assert evidence['developability_unknown_selected'] == 15
    assert evidence['selected_donors'][0]['cdr_developability_in_range'] is True


def test_developability_flag_fails_closed_without_profile(bank):
    exact, chosen, evidence = donors.select(bank, challenge(), 16, fallback=True)
    assert len(chosen) == 16  # deprioritisation never changes the count
    assert all(not d['cdr_developability_in_range'] for _, d in chosen)
    assert evidence['developability_in_range_selected'] == 0
    assert evidence['developability_unknown_selected'] == 16


@pytest.mark.parametrize('in_range_cell', ['IGHV3-7:16', 'IGHV3-23:18'])
def test_v_match_and_length_distance_precede_developability(bank, in_range_cell):
    preferred = copy.deepcopy(next(r for r in bank if r['cells'] == ['IGHV3-23:16']))
    in_range = copy.deepcopy(next(r for r in bank if r['cells'] == [in_range_cell]))
    in_range['developability'] = _developability()
    for rows in ([in_range, preferred], [preferred, in_range]):
        _, chosen, _ = donors.select(rows, challenge(), 1, fallback=True)
        assert chosen[0][0]['sequence_hash'] == preferred['sequence_hash']
        assert chosen[0][1]['cdr_developability_in_range'] is False


def test_out_of_range_donor_is_dropped_when_in_range_fills_slots(bank):
    # Sixteen in-range donors fill every slot; the two out-of-range donors are
    # deprioritised out of the selection without shrinking the count.
    rows = copy.deepcopy(bank)
    for row in rows[:16]:
        row['developability'] = _developability()
    exact, chosen, evidence = donors.select(rows, challenge(), 16, fallback=True)
    assert len(chosen) == 16
    assert all(d['cdr_developability_in_range'] for _, d in chosen)
    assert evidence['developability_in_range_selected'] == 16
    assert evidence['developability_out_of_range_selected'] == 0


def test_exact_cell_donors_carry_the_developability_flag(bank):
    exact, chosen, evidence = donors.select(bank, challenge(length=18), 16,
                                           fallback=False)
    assert exact and all('cdr_developability_in_range' in d for _, d in chosen)
    assert all(not d['cdr_developability_in_range'] for _, d in chosen)
    assert evidence['developability_unknown_selected'] == len(exact)


def test_family_cap_is_opt_in_and_preserves_default_selection(bank):
    # No family_cap: the default path keeps the exact 16-donor, distance-ordered
    # selection even though the synthetic bank collapses under the 0.15 family
    # threshold (its length-16 CDR3s differ by one residue).
    exact, chosen, evidence = donors.select(bank, challenge(), 16, fallback=True)
    assert len(chosen) == 16
    assert [d['reference_length_distance'] for _, d in chosen] == [5]*6 + [6]*2 + [7]*8


def test_family_cap_bounds_selection_to_one_per_clonotype(bank):
    # Label each bank cell as one clonotype family and cap at one: the ceiling
    # admits a single donor per family, never a hard dedup of the whole bank.
    # The labels are deliberately far apart under Levenshtein so the 0.15 family
    # predicate does not itself collapse them (cell labels differ by one char).
    labels = {cell: 'FAM' + str(i) * 3 for i, cell in enumerate(sorted(BANK_CELLS))}
    labelled = copy.deepcopy(bank)
    for row in labelled:
        row['family'] = labels[next(iter(row['cells']))]
    cdr3_of = lambda r: r['family']
    exact, chosen, evidence = donors.select(labelled, challenge(), 16, fallback=True,
                                            family_cap=1, cdr3_of=cdr3_of)
    assert evidence['family_cap'] == 1 and evidence['family_cap_trimmed'] is True
    assert len(chosen) == len(BANK_CELLS)
    assert evidence['family_cap_families_selected'] == len(BANK_CELLS)
    assert max(Counter(r['family'] for r, _ in chosen).values()) == 1


@pytest.mark.parametrize('family_cap', [0, -1, True, 1.5])
def test_family_cap_rejects_invalid_values(bank, family_cap):
    with pytest.raises(ValueError, match='family_cap'):
        donors.select(bank, challenge(), 16, fallback=True,
                      family_cap=family_cap, cdr3_of=lambda r: r['sequence'])


def test_family_cap_requires_a_cdr3_extractor(bank):
    with pytest.raises(ValueError, match='cdr3_of'):
        donors.select(bank, challenge(), 16, fallback=True, family_cap=2)


@pytest.fixture
def sequence_runner(monkeypatch, tmp_path, bank):
    from nanobody_opt import gates
    calls = []
    # The real IGHV3 germline has 96 non-gap V positions (IMGT 1-104), then the
    # CDR3 (loop + 3-residue tail), then the 11-residue FR4.
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(
        Chain=lambda sequence, **kw: types.SimpleNamespace(cdr3_seq=sequence[96:-11])))
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
    assert len(result['generated_sequence_hashes']) == 600
    assert result['fresh_unique_proposals'] == 600 and not result['proposal_exhausted']
    assert result['existing_references'] == 0 and not result['novelty_checked']
    assert result['proposal_source_counts']['unseeded_control'] == 300
    assert len(sequence_runner.annotation_calls) == len(result['generated_sequence_hashes'])
    assert {r['sequence_hash'] for r in candidates}.isdisjoint(r['sequence_hash'] for r in bank)
    assert all(r['source'] == 'fresh_baseline_proposal' for r in candidates)
    assert all(r['requested_v'] == gene and r['requested_cdr3_length'] == length for r in candidates)
    assert all(r['v_reference_allele'] == gene+'*01' and len(r['sequence']) == 96+length+11
               for r in candidates)
    assert all(not r['acceptance_qualified'] and not r['target_scored'] for r in candidates)
    assert all('old_target_binding_score' not in r for r in candidates)
    # The cheap hydro screen trims some flat/grafted candidates; survivors are a
    # nonempty subset and the eligible source counts still sum to the survivors.
    assert 0 < len(candidates) <= 600
    assert sum(result['eligible_proposal_source_counts'].values()) == len(candidates)
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
    assert exact['existing_references'] == 8


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
    assert ordinals == list(range(result['proposal_attempts']))
    assert result['next_proposal_offset'] == result['proposal_attempts'] >= 41
    assert result['donor_parent_reuse_blocked'] == 1 and 0 < len(result['candidates']) <= 40
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
    assert rerun['candidates'] and rerun['excluded_generated'] >= 20
    assert set(rerun['generated_sequence_hashes']).isdisjoint(excluded)
    second = sequence_runner(missing_cell_donor_fallback=True, proposal_offset=40,
                              excluded_sequence_hashes=excluded)
    assert second['next_proposal_offset'] >= 80 and 0 < len(second['candidates']) <= 40
    assert set(second['generated_sequence_hashes']).isdisjoint(excluded)
    state = {}
    ledger = budget.ensure(state, {'epoch_fresh_native_limit': 4})
    selected = budget.reserve(ledger, 1, first['candidates'])
    assert len(selected) == 4 and budget.summary(ledger)['remaining'] == 0
    assert budget.reserve(ledger, 1, first['candidates']) == selected
    assert budget.reserve(ledger, 2, second['candidates']) == []


def test_merit_orders_exact_parents_and_ignores_cross_target_raw_scores(bank):
    rows = copy.deepcopy([r for r in bank if r['cells'] == ['IGHV3-23:18']][:3])
    c = challenge(length=18)
    for i, row in enumerate(rows):
        row['quality_evidence'] = [{'target': c['target'], 'epoch': c['epoch'] - 1,
            'kind': 'public_validator_percentile', 'value': [0.8, 0.1, 0.5][i],
            'passes': 1, 'field_size': 30}]
    exact, _, _ = donors.select(rows, c, 3)
    assert exact == [rows[1], rows[2], rows[0]]
    rows[0]['old_target_binding_score'] = -1e9
    rows[0]['quality_evidence'][0]['target'] = 'old-target'
    exact, _, _ = donors.select(rows, c, 3)
    assert exact[-1] == rows[0]


def test_missing_and_failed_developability_are_distinct(bank):
    rows = copy.deepcopy([r for r in bank if r['cells'] == ['IGHV3-23:18']][:3])
    rows[0]['developability'] = _developability(cdr3=30)
    rows[1]['developability'] = _developability()
    _, chosen, evidence = donors.select(rows, challenge(length=18), 3)
    assert [d['cdr_developability_status'] for _, d in chosen] == [
        'in_range', 'unknown', 'out_of_range']
    assert evidence['developability_unknown_selected'] == 1
    assert evidence['developability_out_of_range_selected'] == 1


def test_exact_parent_identity_is_validated_too(bank):
    rows = copy.deepcopy(bank)
    rows[0]['sequence_hash'] = 'f' * 64
    with pytest.raises(ValueError, match='identity'):
        donors.select(rows, challenge(length=16), 3)


def test_bad_or_unmatched_merit_does_not_become_measurement(bank):
    row = copy.deepcopy(bank[0]); c = challenge()
    for value in [float('nan'), float('inf'), -1, 2]:
        row['quality_evidence'] = [{'target': c['target'], 'epoch': c['epoch'] - 1,
            'kind': 'public_validator_percentile', 'value': value, 'passes': 1, 'field_size': 10}]
        assert donors.merit(row, c)[1] == 'unknown'
    row['quality_evidence'] = [{'target': c['target'], 'epoch': c['epoch'] - 1,
        'kind': 'matched_multipass_mean', 'value': 900, 'passes': 20,
        'field_sha256': 'a' * 64, 'protocol': 'matched-v1'}]
    assert donors.merit(row, c)[1] == 'unknown'
    assert donors.merit(row, {**c, 'merit_field_sha256': 'a' * 64,
                             'merit_protocol': 'matched-v1'})[1] == 'matched_multipass_mean'


def test_duplicate_generator_exhaustion_is_bounded(sequence_runner, bank, monkeypatch):
    def exhausted(*args, **kwargs):
        return {'sequence': bank[0]['sequence']}
    monkeypatch.setattr(proposal, 'propose', exhausted)
    result = sequence_runner(missing_cell_donor_fallback=True, proposals=40)
    assert result['proposal_exhausted'] and result['fresh_unique_proposals'] == 0
    assert result['proposal_attempts'] <= 600
    assert result['candidates'] == []


def test_sequence_phase_verifies_optional_bank_identity_before_work(sequence_runner):
    first = sequence_runner()
    pinned = first['reference_coverage']['bank_sha256']
    assert sequence_runner(reference_bank_sha256=pinned)['reference_coverage']['bank_sha256'] == pinned
    for identity in ('f' * 64, 'malformed', True):
        with pytest.raises(ValueError, match='bank identity'):
            sequence_runner(reference_bank_sha256=identity)


def test_actual_generation_journal_keeps_rejected_candidates_and_template_links(sequence_runner, monkeypatch, tmp_path):
    from nanobody_opt import pr100_history as history
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: 'fixture_rejected')
    result = sequence_runner(missing_cell_donor_fallback=True)
    assert result['candidates'] == []
    assert result['generation_rows'] and result['lineage_references']
    assert result['history_capture_status'] == 'sidecar_saved'
    db = history.open_store(tmp_path / 'history.sqlite')
    try:
        report = history.import_generation(db, json.dumps(result).encode(), locator='recorded-test/result.json')
        assert report['rows'] == len(result['generation_rows'])
        assert report['recorded_relationships'] > 0 and report['unresolved_references'] == 0
        assert {r[0] for r in db.execute('SELECT proposal_type FROM lineage_edges')} == {'recorded_cdr3_template'}
        assert db.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0
    finally:
        db.close()
