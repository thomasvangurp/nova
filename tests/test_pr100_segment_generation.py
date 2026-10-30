"""Real numbered historical donors; proposal provenance and live gate contracts."""
import copy
import json
from pathlib import Path
import sys
import types

import pytest

from nanobody_opt import pr100_segment_generation as s, pr100_history as h
from nanobody_opt import pr100_live_worker as worker, pr100_fresh_budget as budget, rules
from pr100_germline import real_panel

CHALLENGE = {'epoch': 26030, 'target': 'Q9NZQ7', 'allowed_v': 'IGHV3-74', 'allowed_cdr3_length': 10}


@pytest.fixture
def bank():
    path = Path(__file__).parent / 'fixtures/pr100/segment_donors_26030.json'
    data = json.loads(path.read_bytes())
    assert data['before_epoch'] == CHALLENGE['epoch']
    assert all(e['epoch'] < CHALLENGE['epoch'] for r in data['rows'] for e in r['quality_evidence'])
    return data['rows']


@pytest.fixture
def prepared(bank):
    return [s.prepare(row, row['numbered']) for row in bank]


def test_default_off_and_strict_bounded_settings():
    assert not s.settings({})['enabled']
    for value in [None, True, {'enabled': 1}, {'limit': 0}, {'beam_width': 513},
                  {'max_donors': 65}, {'unknown': True}, {'limit': True}]:
        with pytest.raises(ValueError): s.settings({'segment_generation': value})


def test_enabled_default_limit_2400_with_hard_4800_ceiling_and_smaller_kept():
    assert s.settings({}) == {'enabled': False, 'limit': 2400, 'max_donors': 32,
                              'beam_width': 128, 'max_per_cdr3': 8}
    assert s.settings({'segment_generation': {'enabled': True}})['limit'] == 2400
    assert s.settings({'segment_generation': {'enabled': True, 'limit': 16}})['limit'] == 16
    assert s.settings({'segment_generation': {'enabled': True, 'limit': 4800}})['limit'] == 4800
    with pytest.raises(ValueError): s.settings({'segment_generation': {'enabled': True, 'limit': 4801}})


def test_real_numbering_regions_round_trip_and_preserve_distinct_junctions(prepared):
    assert {len(d['segments']['cdr3']['sequence']) for d in prepared} == {7, 8, 14}
    assert all(''.join(d['segments'][k]['sequence'] for k in s.REGIONS) == d['sequence'] for d in prepared)
    assert len({d['segments']['fr4']['sequence'] for d in prepared}) > 1
    for donor in prepared:
        for name, low, high in zip(s.REGIONS, (1, 27, 39, 56, 66, 105, 118), s.ENDS):
            part = donor['segments'][name]
            assert low <= part['imgt_start'][0] <= part['imgt_end'][0] <= high


@pytest.mark.parametrize('fault', ['missing', 'residue', 'duplicate_position', 'canonical_cysteine', 'outside_domain'])
def test_malformed_numbering_is_not_replaced_by_heuristic_boundaries(bank, fault):
    row = copy.deepcopy(bank[0]); numbered = row['numbered']
    if fault == 'missing': numbered.pop()
    elif fault == 'residue': numbered[5]['aa'] = 'X'
    elif fault == 'duplicate_position': numbered[1].update(numbered[0])
    elif fault == 'canonical_cysteine':
        i = next(i for i, p in enumerate(numbered) if p['number'] == 23)
        numbered[i]['aa'] = 'A'; row['sequence'] = row['sequence'][:i] + 'A' + row['sequence'][i+1:]
        row['sequence_hash'] = rules.sequence_hash(row['sequence'])
    else: numbered[-1]['number'] = 129
    with pytest.raises(ValueError): s.prepare(row, numbered)


def test_real_missing_cell_produces_multi_donor_children_and_no_native_claims(prepared):
    assert all('IGHV3-74:10' not in d['cells'] for d in prepared)
    stats = {}; rows = s.generate(prepared, CHALLENGE, limit=32, stats=stats)
    assert len(rows) == len({row['sequence_hash'] for row in rows}) == 16
    assert {row['segment_layout'] for row in rows} == {'linked_v', 'region'}
    references = {r['sequence_hash']: r['sequence'] for r in prepared}
    original_loops = {d['segments']['cdr3']['sequence'] for d in prepared}
    for row in rows:
        assert s.reconstruct(row, references) == row['sequence']
        assert not row['acceptance_qualified'] and not row['target_scored']
        assert len({p['source_hash'] for p in row['segment_provenance']}) >= 2
        loop = ''.join(row['sequence'][p['child_start']:p['child_end']]
                       for p in row['segment_provenance'] if p['region'].startswith('cdr3'))
        assert len(loop) == 10 and all(s.cdr_similarity(loop, old) < .85 for old in original_loops)
        assert row['segment_splice']['after_imgt'] in (109, 110, 111, 112)
    assert stats['beam_expansions'] < 1000 and stats['emitted'] == 16
    assert stats['emitted_distinct_cdr3s'] == 2


def test_loop_cap_bounds_redundancy_without_inflating_the_proposal_quota(prepared):
    stats = {}
    rows = s.generate(prepared, CHALLENGE, limit=128, max_per_cdr3=2, stats=stats)
    assert len(rows) == 4
    # The cap is global across rotated rounds: only two distinct loops exist,
    # so the search reports exhaustion instead of a filled quota.
    assert stats['emitted'] == 4 and stats['emitted_distinct_cdr3s'] == 2
    assert stats['filled'] is False and stats['exhausted'] is True
    assert stats['cdr3_cap_drops'] > 0


def test_beam_width_eight_real_fixture_reports_actual_capacity_not_two_beam(prepared):
    # The audited single-pass search capped real output at 2*beam_width. The
    # real three-donor fixture has only two length-compatible loops and fewer
    # region options than the window width, so its pools cannot rotate: with
    # beam_width=8 the honest ceiling stays <= 2*beam_width. The refactor must
    # report that as exhaustion, never as a filled quota.
    stats = {}
    rows = s.generate(prepared, CHALLENGE, limit=108, beam_width=8, max_per_cdr3=64, stats=stats)
    assert len(rows) == stats['emitted'] <= 2 * 8
    assert stats['filled'] is False and stats['exhausted'] is True
    assert stats['rounds_run'] < s.MAX_ROUNDS
    assert stats['emitted_distinct_cdr3s'] == 2


def test_donor_order_does_not_change_search_or_provenance(prepared):
    assert s.generate(prepared, CHALLENGE, limit=16) == s.generate(list(reversed(prepared)), CHALLENGE, limit=16)


def test_duplicate_resume_exclusions_do_not_create_new_native_budget(prepared):
    first = s.generate(prepared, CHALLENGE, limit=8)
    seen = {r['sequence_hash'] for r in first}
    again = s.generate(prepared, CHALLENGE, limit=8, excluded=seen, offset=1)
    assert again and seen.isdisjoint(r['sequence_hash'] for r in again)
    ledger = budget.ensure({}, {'epoch_fresh_native_limit': 3})
    assert len(budget.reserve(ledger, 1, first)) == 3
    assert budget.reserve(ledger, 2, again) == []


def test_no_compatible_new_loop_is_explicit_empty_result(prepared):
    stats = {}
    assert s.generate(prepared[:1], CHALLENGE, stats=stats) == []
    assert stats['emitted'] == stats['novel_loop_options'] == 0


def test_conserved_cdr_rule_is_checked_in_addition_to_the_85_percent_loop_rule():
    donor = {'segments': {name: {'sequence': seq} for name, seq in
                         [('cdr1', 'A' * 10), ('cdr2', 'G' * 10), ('cdr3', 'R' * 10)]}}
    parts = [{'region': name, 'sequence': seq} for name, seq in
             [('cdr1', 'A' * 10), ('cdr2', 'G' * 10), ('cdr3_left', 'QQ'), ('cdr3_right', 'R' * 8)]]
    assert s.cdr_similarity('QQ' + 'R' * 8, 'R' * 10) == .8
    assert s.source_cdr_conflict(parts, [donor])
    parts[0]['sequence'] = 'SS' + 'A' * 8
    assert not s.source_cdr_conflict(parts, [donor])


def test_prepared_segment_tampering_cannot_generate_a_candidate(prepared):
    prepared[0]['segments']['fr1']['sequence'] = 'A'
    with pytest.raises(ValueError, match='segment drift'): s.generate(prepared, CHALLENGE)


def test_component_pools_can_use_wrong_cell_but_not_future_archive_rows(bank):
    future = copy.deepcopy(bank[0]); future['first_archive_epoch'] = 26030
    selected = s.choose_donors([future] + bank[1:], CHALLENGE, 8)
    assert future['sequence_hash'] not in {d['sequence_hash'] for d in selected}
    assert selected and all('IGHV3-74:10' not in d['cells'] for d in selected)


def test_partial_or_tampered_provenance_is_not_accepted(prepared):
    row = s.generate(prepared, CHALLENGE, limit=1)[0]
    refs = {d['sequence_hash']: d['sequence'] for d in prepared}
    for fault in ('child_gap', 'source_interval', 'unknown_source', 'sequence'):
        bad = copy.deepcopy(row)
        if fault == 'child_gap': bad['segment_provenance'][1]['child_start'] += 1
        elif fault == 'source_interval': bad['segment_provenance'][0]['source_end'] -= 1
        elif fault == 'unknown_source': bad['segment_provenance'][0]['source_hash'] = 'f' * 64
        else: bad['sequence'] = 'A' + bad['sequence'][1:]
        with pytest.raises(ValueError): s.reconstruct(bad, refs)


def test_history_captures_all_segment_donors_as_contributions_not_full_parents(prepared, tmp_path):
    row = s.generate(prepared, CHALLENGE, limit=1)[0]
    journal = h.generation_records([row], prepared)
    hashes = {p['source_hash'] for p in row['segment_provenance']}
    assert {r['sequence_hash'] for r in journal['lineage_references']} == hashes
    store = h.open_store(tmp_path / 'q9.sqlite')
    raw = h.canonical({'challenge': CHALLENGE, **journal}).encode()
    for _ in range(2): h.import_generation(store, raw, locator='recorded/sequence/result.json')
    edges = store.execute('SELECT proposal_type,mutations_json,metadata_json FROM lineage_edges').fetchall()
    assert len(edges) == len(hashes)
    for kind, mutations, metadata in edges:
        assert kind == 'recorded_segment_donor' and mutations == '[]'
        evidence = json.loads(metadata)
        assert evidence['inferred'] is False and evidence['copied_segments']
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0
    store.close()


def test_missing_history_donor_does_not_invent_partial_relationships(prepared, tmp_path):
    row = s.generate(prepared, CHALLENGE, limit=1)[0]
    store = h.open_store(tmp_path / 'q9.sqlite')
    report = h.import_generation(store, h.canonical({'challenge': CHALLENGE, 'generation_rows': [row]}).encode(),
                                 locator='missing-sources.json')
    assert report['unresolved_references'] >= 2
    assert store.execute('SELECT COUNT(*) FROM lineage_edges').fetchone()[0] == 0
    store.close()


def test_outcome_analytics_preserve_multiple_donors_without_naming_a_single_parent(prepared):
    from nanobody_opt import pr100_lineage_outcomes as outcomes
    row = s.generate(prepared, CHALLENGE, limit=1)[0]
    origin = outcomes.parent_origin(row, outcomes.bank_index(prepared))
    assert origin['resolved'] and origin['kind'] == 'multi_donor_segments'
    assert origin['parent_hash'] is None and len(origin['contributors']) >= 2
    assert all(c['kind'] == 'segment_donor' and c['copied_segments'] for c in origin['contributors'])
    assert outcomes.edit_kind(row)['kind'] == row['proposal_lane']


@pytest.fixture
def run_worker(bank, monkeypatch, tmp_path):
    from nanobody_opt import gates
    lookup = {row['sequence']: row for row in bank}
    class Chain:
        def __init__(self, seq, **kwargs):
            self.row = lookup[seq]
            self.cdr3_seq = ''.join(x['aa'] for x in self.row['numbered'] if 105 <= x['number'] <= 117)
        def __iter__(self):
            return iter((types.SimpleNamespace(number=x['number'], letter=x['letter']), x['aa'])
                        for x in self.row['numbered'])
    monkeypatch.setitem(sys.modules, 'abnumber', types.SimpleNamespace(Chain=Chain))
    monkeypatch.setattr(worker.pr100_filter, 'validate_annotation_runtime', lambda: None)
    monkeypatch.setattr(worker.pr100_filter, 'helper', lambda name:
                        types.SimpleNamespace(_anarci_panel=lambda species: (real_panel(), None)))
    monkeypatch.setattr(worker.pr100_filter, '_annotate_uncached', lambda seq: {})
    monkeypatch.setattr(worker.pr100_filter, 'annotation_identity', lambda: 'a' * 64)
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: '')
    monkeypatch.setattr(gates, 'check_sequence', lambda *a: None)
    from nanobody_opt import hydrophobicity
    monkeypatch.setattr(hydrophobicity, 'check', lambda sequence: (None, True))
    path = tmp_path / 'bank.json'; path.write_text(json.dumps({'rows': bank}))
    (tmp_path / 'archive.csv').write_text('sequence\nAAA\n')
    def run(**overrides):
        return worker.sequence_phase({'challenge': CHALLENGE, 'references': str(path),
            'archive_sha256': worker.file_digest(tmp_path / 'archive.csv'), 'archive_fetched_at': 1,
            'proposals': 12, 'missing_cell_donor_fallback': True, **overrides}, tmp_path)
    return run


def test_live_worker_keeps_baseline_and_interleaves_experimental_native_admission(run_worker):
    off = run_worker()
    on = run_worker(segment_generation={'enabled': True, 'limit': 16})
    assert off['fresh_unique_proposals'] == on['fresh_unique_proposals'] == 12
    assert on['segment_generation_stats']['emitted'] == on['eligible_segment_generation'] == 16
    sources = [r['source'] for r in on['candidates'][:5]]
    assert sources == [s.SOURCE] * 4 + ['fresh_baseline_proposal']
    assert {r['sequence_hash'] for r in off['candidates']} <= {r['sequence_hash'] for r in on['candidates']}
    assert on['generation_capture_complete'] and len(on['generation_rows']) == 28


def test_live_recombination_cannot_bypass_child_annotation(run_worker, monkeypatch):
    monkeypatch.setattr(worker.rules, 'annotation_reason', lambda *a, **kw: 'wrong_epoch_v_gene')
    result = run_worker(segment_generation={'enabled': True, 'limit': 8})
    assert result['segment_generation_stats']['emitted'] == 8
    assert result['candidates'] == [] and result['annotation_reasons'] == {'wrong_epoch_v_gene': 20}


def _src(source, n):
    return {'source': source, 'n': n}


def test_interleave_four_to_one_alternating_edit_and_baseline_controls():
    rows = ([_src(s.SOURCE, i) for i in range(8)]
            + [_src('edit_generation', i) for i in range(3)]
            + [_src('fresh_baseline_proposal', i) for i in range(3)])
    ordered = worker.interleave_segment_controls(rows, segment_source=s.SOURCE)
    expected = ([s.SOURCE] * 4 + ['edit_generation']
                + [s.SOURCE] * 4 + ['fresh_baseline_proposal']
                + ['edit_generation', 'fresh_baseline_proposal'] * 2)
    assert [r['source'] for r in ordered] == expected
    # Each arm's internal order is preserved exactly.
    assert [r['n'] for r in ordered if r['source'] == s.SOURCE] == list(range(8))
    assert [r['n'] for r in ordered if r['source'] == 'edit_generation'] == list(range(3))
    assert [r['n'] for r in ordered if r['source'] == 'fresh_baseline_proposal'] == list(range(3))


def test_interleave_work_conserving_when_an_arm_is_exhausted():
    # Edit arm absent: every control slot falls through to baseline, in order.
    rows = [_src(s.SOURCE, i) for i in range(2)] + [_src('fresh_baseline_proposal', i) for i in range(6)]
    ordered = worker.interleave_segment_controls(rows, segment_source=s.SOURCE)
    assert [r['source'] for r in ordered] == [s.SOURCE] * 2 + ['fresh_baseline_proposal'] * 6
    # Segment arm absent: edit and baseline controls still alternate.
    rows = [_src('edit_generation', i) for i in range(2)] + [_src('fresh_baseline_proposal', i) for i in range(2)]
    ordered = worker.interleave_segment_controls(rows, segment_source=s.SOURCE)
    assert [r['source'] for r in ordered] == ['edit_generation', 'fresh_baseline_proposal'] * 2


def test_live_worker_source_exhaustion_releases_slots_to_baseline(run_worker):
    result = run_worker(segment_generation={'enabled': True, 'limit': 4})
    assert result['segment_generation_stats']['emitted'] == result['eligible_segment_generation'] == 4
    assert [r['source'] for r in result['candidates']] == (
        [s.SOURCE] * 4 + ['fresh_baseline_proposal'] * 12)


def test_live_worker_default_2400_limit_reports_exhaustion_not_filled(run_worker):
    result = run_worker(segment_generation={'enabled': True})
    assert result['segment_generation_settings']['limit'] == 2400
    # Two length-compatible loops * default per-CDR3 cap 8 => 16 actual children.
    assert result['segment_generation_stats']['emitted'] == 16
    assert result['segment_generation_stats']['filled'] is False
    assert result['segment_generation_stats']['exhausted'] is True
