"""Archive revisions recover historical availability without API dependencies."""
import hashlib

import pytest

from nanobody_opt import pr100_archive_history as a, pr100_history as h

BASE, ONE, TWO = 'a' * 40, 'b' * 40, 'c' * 40


@pytest.fixture
def store(tmp_path):
    conn = h.open_store(tmp_path / 'q9.sqlite')
    a.schema(conn)
    yield conn
    conn.close()


def csv(*pairs):
    result = ['sequence,sequence_hash,design_iiptm']
    result += [f'{seq},{hashlib.sha256(seq.encode()).hexdigest()},{score}' for seq, score in pairs]
    return ('\n'.join(result) + '\n').encode()


OLD = csv(('ACDE', .4), ('ACDF', .5))
NEW = csv(('ACDE', .4), ('ACDF', .5), ('ACDG', .6))


def metadata(epoch=25956, count=1, revision=ONE):
    return {'id': revision, 'title': f'Added {count} nanobody sequences from epoch {epoch} to Q9NZQ7_nanobodies.csv'}


def test_complete_csv_chain_recovers_epochs_without_leaderboards_or_fake_replay(store):
    a.import_baseline(store, OLD, revision=BASE)
    report = a.import_delta(store, OLD, NEW, metadata=metadata(), parent_revision=BASE, before_epoch=26031)
    assert report['added'] == 1
    assert store.execute('SELECT first_archive_epoch,available_before_epoch,baseline '
                         'FROM history_archive_availability ORDER BY node_id').fetchall() == [
        (None, 25955, 1), (None, 25955, 1), (25956, 25957, 0)]
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0
    assert store.execute('SELECT COUNT(*) FROM lineage_edges').fetchone()[0] == 0
    coverage = a.reconcile(store, NEW)
    assert coverage['all_csv_sequences_accounted_for'] and coverage['epoch_attributed_additions'] == 1
    assert coverage['baseline_sequences'] == 2


def test_csv_chain_retry_is_idempotent_and_uses_existing_node_identity(store):
    nid = h.node(store, 'ACDG'); store.commit()
    for _ in range(2):
        a.import_baseline(store, OLD, revision=BASE)
        a.import_delta(store, OLD, NEW, metadata=metadata(), parent_revision=BASE, before_epoch=26031)
    assert store.execute("SELECT id FROM nodes WHERE sequence='ACDG'").fetchone()[0] == nid
    assert store.execute('SELECT COUNT(*) FROM history_archive_availability').fetchone()[0] == 3
    assert store.execute('SELECT COUNT(*) FROM history_archive_revisions').fetchone()[0] == 2


@pytest.mark.parametrize('fault', ['wrong_count', 'removed', 'changed_metrics', 'wrong_parent_hash',
                                  'unknown_parent', 'future_epoch', 'wrong_title', 'backdated_epoch'])
def test_unreconciled_revision_never_creates_false_epoch_links(store, fault):
    a.import_baseline(store, OLD, revision=BASE)
    old, new, meta, parent, before = OLD, NEW, metadata(), BASE, 26031
    if fault == 'wrong_count': meta = metadata(count=2)
    elif fault == 'removed': new = csv(('ACDE', .4), ('ACDG', .6))
    elif fault == 'changed_metrics': new = csv(('ACDE', .9), ('ACDF', .5), ('ACDG', .6))
    elif fault == 'wrong_parent_hash': old = b'sequence\nACDE\n'
    elif fault == 'unknown_parent': parent = TWO
    elif fault == 'future_epoch': before = 25956
    elif fault == 'wrong_title': meta['title'] = 'no epoch proof'
    elif fault == 'backdated_epoch': meta = metadata(epoch=25954)
    with pytest.raises(ValueError):
        a.import_delta(store, old, new, metadata=meta, parent_revision=parent, before_epoch=before)
    assert store.execute('SELECT COUNT(*) FROM history_archive_availability').fetchone()[0] == 2
    assert store.execute('SELECT COUNT(*) FROM history_archive_revisions').fetchone()[0] == 1


def test_reconcile_exposes_missing_and_unexplained_sequences(store):
    a.import_baseline(store, OLD, revision=BASE)
    report = a.reconcile(store, csv(('ACDE', .4), ('ACDG', .6)))
    assert report['missing_sequences'] == report['unexplained_extra_sequences'] == 1
    assert not report['all_csv_sequences_accounted_for']


@pytest.mark.parametrize('bad', [b'sequence,design_iiptm\nACDE,NaN\n',
                               b'sequence,design_iiptm\nACDE,\n',
                               b'sequence\nACDE\nACDE\n',
                               b'sequence,sequence_hash\nACDE,wrong\n'])
def test_bad_csv_cannot_seed_an_availability_history(store, bad):
    with pytest.raises(ValueError): a.import_baseline(store, bad, revision=BASE)
    assert store.execute('SELECT COUNT(*) FROM history_archive_availability').fetchone()[0] == 0
