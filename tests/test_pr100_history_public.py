"""Public coverage must expose missing epochs and never invent replay draws."""
import json
from pathlib import Path

import pytest

from nanobody_opt import pr100_history as h
from nanobody_opt import pr100_history_public as p


@pytest.fixture
def store(tmp_path):
    conn = h.open_store(tmp_path / 'q9.sqlite')
    yield conn
    conn.close()


def board(score=42):
    return {'success': True, 'data': [{'uid': 54, 'coldkey': 'public-miner',
            'nanobodies': [{'protein_name': h.TARGET, 'sequence': 'ACDEFGHIK',
                           'final_score': score, 'design_iiptm': 0.5}]}]}


def ingest(store, value, epoch=25955):
    return p.import_board(store, json.dumps(value).encode(), epoch=epoch, before_epoch=26032)


def test_published_result_retry_is_idempotent_and_not_raw_replay(store):
    first = ingest(store, board())
    assert ingest(store, board()) == first
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 1
    row = store.execute('SELECT epoch,observation_count,provenance_json FROM observations').fetchone()
    assert row[:2] == (25955, 1)
    provenance = json.loads(row[2])
    assert provenance['raw_draw_count'] is None and provenance['scorer_protocol'] is None
    assert provenance['evidence_kind'] == 'published_validator_result'


def test_corrected_board_retains_audit_history_and_replaces_active_pointer(store):
    old = ingest(store, board(42))
    new = ingest(store, board(40))
    assert old['source_id'] != new['source_id']
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 2
    assert store.execute('SELECT source_id FROM history_public_epochs').fetchone()[0] == new['source_id']
    assert store.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] == 1


@pytest.mark.parametrize('fault', ['truncated', 'paginated', 'malformed_sequence', 'missing_score',
                                  'duplicate', 'nan', 'wrong_success'])
def test_invalid_or_incomplete_board_cannot_be_counted_as_available(store, fault):
    value = board()
    nb = value['data'][0]['nanobodies'][0]
    if fault == 'truncated': value['meta'] = {'total': 2}
    elif fault == 'paginated': value['meta'] = {'next_cursor': 'another-page'}
    elif fault == 'malformed_sequence': nb['sequence'] = 'INVALIDX'
    elif fault == 'missing_score': del nb['final_score']
    elif fault == 'duplicate': value['data'][0]['nanobodies'].append(dict(nb))
    elif fault == 'nan': nb['design_iiptm'] = float('nan')
    elif fault == 'wrong_success': value['success'] = False
    with pytest.raises(ValueError): ingest(store, value)
    assert store.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] == 0
    assert store.execute('SELECT COUNT(*) FROM history_public_epochs').fetchone()[0] == 0


def test_empty_missing_and_unfetched_epochs_are_distinct_gaps(store):
    ingest(store, board(), epoch=25955)
    ingest(store, {'success': True, 'data': []}, epoch=25956)
    p.unavailable(store, epoch=25957, before_epoch=25959, http_status=404)
    result = p.coverage(store, before_epoch=25959)
    assert result['epochs_expected'] == 4 and result['epochs_available'] == 1
    assert [row['status'] for row in result['gaps']] == ['empty_unverified', 'unavailable', 'unfetched']
    assert not result['public_api_interval_reconciled']


def test_failed_refresh_keeps_previous_evidence_but_revokes_complete_claim(store):
    previous = ingest(store, board())
    p.unavailable(store, epoch=25955, before_epoch=25956, error_type='secretish_untrusted_error')
    row = p.coverage(store, before_epoch=25956)['epochs'][0]
    assert row['source_id'] == previous['source_id'] and row['rows'] == 1
    assert row['status'] == 'unavailable' and row['error_type'] == 'OtherError'
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 1


@pytest.mark.parametrize('epoch', [25954, 26032, None, True])
def test_future_or_unknown_or_old_era_epoch_rejected(store, epoch):
    with pytest.raises(ValueError): ingest(store, board(), epoch)
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0


def test_interval_complete_still_does_not_attest_replay_or_competitor_parents(store):
    ingest(store, board())
    report = p.coverage(store, before_epoch=25956)
    assert report['public_api_interval_reconciled']
    assert not report['raw_multipass_coverage_proven']
    assert not report['competitor_parentage_proven']


def test_real_recorded_public_board_all_target_rows_preserved(store):
    path = Path(__file__).parent / 'fixtures/pr100/parent_feed/public_board_26028.json'
    value = json.loads(path.read_bytes())
    expected = sum(nb.get('protein_name') == h.TARGET
                   for member in value['data'] for nb in member.get('nanobodies', []))
    report = ingest(store, value, epoch=26028)
    assert report['rows'] == expected == 20
    assert store.execute('SELECT COUNT(*) FROM history_public_observations').fetchone()[0] == expected


def test_existing_staged_history_gets_additive_coverage_tables(tmp_path):
    path = tmp_path / 'q9.sqlite'
    conn = h.open_store(path)
    h.node(conn, 'ACDE')
    conn.commit()
    conn.execute('DROP TABLE history_public_epochs')
    conn.execute('DROP TABLE history_public_observations')
    conn.commit(); conn.close()
    conn = h.open_store(path)
    assert conn.execute('SELECT sequence FROM nodes').fetchall() == [('ACDE',)]
    assert p.coverage(conn, before_epoch=25956)['epochs'][0]['status'] == 'unfetched'
    conn.close()
