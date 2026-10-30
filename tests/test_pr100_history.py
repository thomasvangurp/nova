"""History keeps source truth separate from qualification and inferred ancestry."""
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from nanobody_opt import pr100_history as h


@pytest.fixture
def store(tmp_path):
    conn = h.open_store(tmp_path / 'q9.sqlite')
    yield conn
    conn.close()


def raw(value):
    return json.dumps(value).encode()


def artifact(rows, epoch=26030):
    return raw({'challenge': {'target': h.TARGET, 'epoch': epoch}, 'candidates': rows})


def test_refuses_unmarked_legacy_database_without_writing(tmp_path):
    path = tmp_path / 'legacy.sqlite'
    c = sqlite3.connect(path); c.execute('CREATE TABLE observations(x)'); c.close()
    before = path.read_bytes()
    with pytest.raises(ValueError, match='unmarked_history_store'):
        h.open_store(path)
    assert path.read_bytes() == before


def test_archive_exact_membership_idempotent_and_no_invented_passes(store):
    value = b'sequence,design_iiptm\nACDE,0.5\nACDF,0.6\nACDE,0.5\n'
    first = h.import_archive(store, value, locator='Q9NZQ7_nanobodies.csv')
    second = h.import_archive(store, value, locator='Q9NZQ7_nanobodies.csv')
    assert first == second and first['unexplained_missing'] == 0
    assert first['input_rows'] == 3 and first['stored_sequences'] == 2
    assert store.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] == 2
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0
    assert store.execute('SELECT protocol,raw_observation_count FROM history_archive_metrics').fetchall() == [(None, None)] * 2
    assert store.execute('SELECT epoch FROM history_sources').fetchone()[0] is None


def test_archive_rejects_hash_corruption_and_accounts_invalid_rows(store):
    value = b'sequence,sequence_hash\nACDE,wrong\nAXDE,\n'
    report = h.import_archive(store, value, locator='snapshot.csv')
    assert report['stored_sequences'] == 0
    assert [x['reason'] for x in report['rejected_rows']] == [
        'archive_sequence_hash_mismatch', 'invalid_sequence']
    assert len(report['rejected_rows']) == report['input_rows']


def test_corrected_archive_versions_are_not_merged_into_fake_raw_draws(store):
    h.import_archive(store, b'sequence,design_iiptm\nACDE,0.5\n', locator='snapshot.csv')
    h.import_archive(store, b'sequence,design_iiptm\nACDE,0.7\n', locator='snapshot.csv')
    assert store.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] == 1
    assert store.execute('SELECT COUNT(*) FROM history_sources').fetchone()[0] == 2
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0


def test_explicit_indel_parent_is_preserved_and_ingestion_is_idempotent(store):
    data = artifact([{'sequence': 'ACDE', 'parent_sequence': 'ACGDE',
                      'source': 'edit_generation', 'edit_policy': 'recorded_policy'}])
    h.import_generation(store, data, locator='26030/sequence/result.json', before_epoch=26031)
    h.import_generation(store, data, locator='26030/sequence/result.json', before_epoch=26031)
    rows = store.execute('SELECT epoch,proposal_type,mutations_json,metadata_json FROM lineage_edges').fetchall()
    assert len(rows) == 1 and rows[0][:2] == (26030, 'recorded_direct_parent')
    assert json.loads(rows[0][2]) == ['del@3:G']
    assert json.loads(rows[0][3])['inferred'] is False
    assert store.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 0


def test_baseline_reference_is_cdr3_template_not_asserted_direct_parent(store):
    parent = 'ACGDE'; sha = h.digest(parent.encode())
    data = artifact([{'sequence': 'ACDE', 'reference_hash': sha, 'source': 'fresh_baseline_proposal'}])
    h.import_generation(store, data, locator='sequence.json', references=[
        {'sequence': parent, 'sequence_hash': sha}])
    assert store.execute('SELECT proposal_type FROM lineage_edges').fetchone()[0] == 'recorded_cdr3_template'


def test_two_recorded_artifacts_are_two_proofs_of_one_relationship(store):
    data = artifact([{'sequence': 'ACDE', 'parent_sequence': 'ACGDE', 'source': 'edit_generation'}])
    for locator in ('attempt1/result.json', 'attempt2/result.json'):
        h.import_generation(store, data, locator=locator)
    assert store.execute('SELECT COUNT(*) FROM lineage_edges').fetchone()[0] == 1
    assert store.execute('SELECT COUNT(*) FROM history_lineage_evidence').fetchone()[0] == 2


def test_missing_reference_sequence_stays_unresolved_without_guessing(store):
    data = artifact([{'sequence': 'ACDE', 'reference_hash': 'a' * 64, 'source': 'fresh_baseline_proposal'}])
    report = h.import_generation(store, data, locator='sequence.json')
    assert report['unresolved_references'] == 1
    assert store.execute('SELECT COUNT(*) FROM lineage_edges').fetchone()[0] == 0
    assert store.execute('SELECT reason FROM history_unresolved').fetchone()[0] == 'reference_sequence_unresolved'


@pytest.mark.parametrize('fault', ['target', 'future', 'epoch_unknown', 'hash'])
def test_bad_generation_context_writes_no_lineage(store, fault):
    value = json.loads(artifact([{'sequence': 'ACDE', 'source': 'edit_generation', 'parent_sequence': 'ACGDE'}]))
    if fault == 'target': value['challenge']['target'] = 'P20809'
    elif fault == 'future': value['challenge']['epoch'] = 26031
    elif fault == 'epoch_unknown': value['challenge']['epoch'] = None
    else: value['candidates'][0]['sequence_hash'] = 'a' * 64
    with pytest.raises(ValueError):
        h.import_generation(store, raw(value), locator='record.json', before_epoch=26031)
    assert store.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] == 0


def test_legacy_target_copy_preserves_raw_ids_unknown_epochs_and_source(tmp_path, store):
    path = tmp_path / 'legacy.sqlite'
    src = sqlite3.connect(path)
    src.executescript(Path(h.__file__).with_name('local_schema.sql').read_text())
    for ident, target, seq in [(10, h.TARGET, 'ACDE'), (20, 'P20809', 'ACDF')]:
        src.execute('INSERT INTO nodes VALUES(?,?,?,?,?)', (ident, target, seq, len(seq), b''))
        src.execute('INSERT INTO observations(observation_id,node_id,epoch,source,score_kind,score,metrics_json,provenance_json) VALUES(?,?,?,?,?,?,?,?)',
                    ('obs' + str(ident), ident, None, 'local_exact_raw', 'unqualified', 1., '{}', '{}'))
    src.commit(); src.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = h.import_legacy(store, path)
    assert report == {'target': h.TARGET, 'nodes': 1, 'observations': 1, 'unknown_epoch_observations': 1}
    assert store.execute('SELECT id,target FROM nodes').fetchall() == [(10, h.TARGET)]
    assert store.execute('SELECT observation_id,epoch FROM observations').fetchall() == [('obs10', None)]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_real_public_fixture_can_be_ingested_without_sequence_qualification_claims(store):
    # Recorded public board supplies real sequences; this exercises history only.
    path = Path(__file__).parent / 'fixtures/pr100/parent_feed/public_board_26028.json'
    data = json.loads(path.read_bytes())
    rows = [{'sequence': nb['sequence'], 'source': 'public_observation'}
            for member in data['data'] for nb in member.get('nanobodies', [])
            if nb.get('protein_name') == h.TARGET]
    report = h.import_generation(store, artifact(rows, 26028), locator='recorded-public-26028')
    assert report['rows'] == 20
    assert store.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] == len({r['sequence'] for r in rows})
    assert store.execute('SELECT COUNT(*) FROM lineage_edges').fetchone()[0] == 0
