"""Consolidate + join of the durable native developability profile (observations schema)."""
import hashlib
import json
import sqlite3

import pytest

from nanobody_opt import pr100_developability_store as store

TARGET = 'Q9NZQ7'
SEQ_A = 'EVQLLESGGGLVQPGGSLRLSCAASGPTESSGAMSWVRQAPGKEREWVSAIRRSGGSKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCGFFDYWGQGTLVTVSS'
SEQ_B = 'QVQLQESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDYWGQGTLVTVSS'


def _dev(**overrides):
    value = {
        'passed': True,
        'flags': {'L': 'green', 'L3': 'green', 'C': 'green',
                  'PSH': 'green', 'PPC': 'green', 'PNC': 'green'},
        'total_cdr_length': 30.0, 'cdr3_length': 12.0, 'cdr3_compactness': 1.0,
        'surface_hydrophobic_patches': 40.0, 'positive_charge_patches': 2.0,
        'negative_charge_patches': 1.0,
    }
    value.update(overrides)
    return value


def _cache_key(sequence):
    return hashlib.sha256((store.PROFILER_VERSION + '\n' + sequence).encode('ascii')).hexdigest()


def _schema(db):
    conn = sqlite3.connect(str(db))
    conn.executescript(
        'CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT, length INTEGER, residue_vector BLOB);'
        'CREATE TABLE observations (observation_id TEXT PRIMARY KEY, node_id INTEGER NOT NULL, epoch INTEGER,'
        ' source TEXT NOT NULL, score_kind TEXT NOT NULL, score REAL NOT NULL,'
        ' observation_count INTEGER NOT NULL, metrics_json TEXT NOT NULL, pose_json TEXT,'
        ' provenance_json TEXT NOT NULL, replicated_at REAL, repl_token INTEGER NOT NULL);'
        'CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT, replicated_at REAL);')
    conn.commit()
    return conn


def _seed_node(db, sequence, node_id=1, target=TARGET):
    conn = sqlite3.connect(str(db))
    conn.execute('INSERT OR IGNORE INTO nodes (id, target, sequence, length) VALUES (?, ?, ?, ?)',
                 (node_id, target, sequence, len(sequence)))
    conn.commit(); conn.close()


def _write_cache(cache, sequence, developability):
    (cache / (_cache_key(sequence) + '.json')).write_text(
        json.dumps({'sequence': sequence, 'developability': developability}))


def test_consolidate_and_attach_round_trip(tmp_path):
    cache = tmp_path / 'layer2'; cache.mkdir()
    db = tmp_path / 'q9.sqlite'
    _schema(db); _seed_node(db, SEQ_A, 1); _seed_node(db, SEQ_B, 2)
    _write_cache(cache, SEQ_A, _dev())
    _write_cache(cache, SEQ_B, _dev(surface_hydrophobic_patches=95.0, passed=False))

    accepted, rejected, matched = store.consolidate(cache, db, target=TARGET)
    assert (accepted, rejected, matched) == (2, 0, 2)

    rows = [{'sequence': SEQ_A, 'sequence_hash': store._content_hash(SEQ_A)},
            {'sequence': SEQ_B, 'sequence_hash': store._content_hash(SEQ_B)},
            {'sequence': 'A' * 120, 'sequence_hash': store._content_hash('A' * 120)}]
    store.attach(rows, db, target=TARGET)
    assert rows[0]['developability']['surface_hydrophobic_patches'] == 40.0
    assert rows[0]['developability']['passed'] is True
    assert rows[1]['developability']['passed'] is False
    assert rows[1]['developability']['surface_hydrophobic_patches'] == 95.0
    assert 'developability' not in rows[2]


def test_rejects_partial_or_error_profiles(tmp_path):
    cache = tmp_path / 'layer2'; cache.mkdir()
    db = tmp_path / 'q9.sqlite'
    _schema(db); _seed_node(db, SEQ_A, 1)
    _write_cache(cache, SEQ_A, _dev(surface_hydrophobic_patches=None))  # missing metric

    accepted, rejected, _ = store.consolidate(cache, db, target=TARGET)
    assert accepted == 0 and rejected == 1


def test_attach_is_fail_loud_on_missing_version(tmp_path):
    db = tmp_path / 'q9.sqlite'
    _schema(db); _seed_node(db, SEQ_A, 1)
    with pytest.raises(ValueError, match='version mismatch'):
        store.attach([{'sequence': SEQ_A, 'sequence_hash': store._content_hash(SEQ_A)}],
                     db, target=TARGET)


def test_attach_never_touches_quality_evidence(tmp_path):
    db = tmp_path / 'q9.sqlite'
    _schema(db); _seed_node(db, SEQ_A, 1)
    # empty store (no score_kind rows) -> fail loud, not silent
    conn = sqlite3.connect(str(db))
    conn.execute('INSERT INTO metadata (key, value) VALUES (?, ?)', (store.VERSION_KEY, store.PROFILER_VERSION))
    conn.commit(); conn.close()
    rows = [{'sequence': SEQ_A, 'sequence_hash': store._content_hash(SEQ_A),
             'quality_evidence': [{'kind': 'public_validator_percentile'}]}]
    with pytest.raises(ValueError, match='empty'):
        store.attach(rows, db, target=TARGET)
    assert rows[0]['quality_evidence'][0]['kind'] == 'public_validator_percentile'
