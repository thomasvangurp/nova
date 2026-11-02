"""Durable developability profile in a dedicated ``node_features`` table.

The profile is computed by the native profiler and cached on the master as
``<cache_root>/<sha256(VERSION + "\\n" + sequence)>.json``. It is stored in a
dedicated ``node_features`` table (NOT the shared ``observations`` table), so no
shared reader (rank-sum, multipass, existence/dedup checks) can ever see it.

- ``consolidate`` ingests cache files into ``node_features`` (idempotent).
- ``attach`` joins the profile onto parent-bank rows read-only, fail-loud.
- ``migrate`` moves any legacy F rows out of ``observations`` (lossless archive
  + delete + a trigger that blocks re-insertion).

Invariants: a known profile is never replaced by NULL; ``quality_evidence`` is
never touched; the join is read-only on the slave.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

PROFILER_VERSION = 'tnp-2026-07-redregion-v1'
SCORE_KIND = PROFILER_VERSION
SOURCE = 'tnp_profiler'
VERSION_KEY = 'tnp.version'
METRIC_KEYS = ('total_cdr_length', 'cdr3_length', 'cdr3_compactness',
               'surface_hydrophobic_patches', 'positive_charge_patches',
               'negative_charge_patches')
FLAG_KEYS = ('L', 'L3', 'C', 'PSH', 'PPC', 'PNC')

_NODE_FEATURES_DDL = """
CREATE TABLE IF NOT EXISTS node_features (
    node_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    ingested_at REAL,
    PRIMARY KEY (node_id, kind)
);
"""
# A trigger blocks legacy F rows from re-entering the shared observations table.
_BLOCK_OBSERVATIONS_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS block_f_observations
BEFORE INSERT ON observations
WHEN NEW.score_kind = '{version}'
BEGIN
    SELECT RAISE(IGNORE);
END;
"""


def _cache_key(sequence):
    return hashlib.sha256((PROFILER_VERSION + '\n' + sequence).encode('ascii')).hexdigest()


def _content_hash(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def _number(value):
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def _parse_profile(developability):
    if not isinstance(developability, dict):
        return None
    passed = developability.get('passed')
    flags = developability.get('flags')
    if type(passed) is not bool or not isinstance(flags, dict):
        return None
    if set(flags) != set(FLAG_KEYS):
        return None
    metrics = {key: _number(developability.get(key)) for key in METRIC_KEYS}
    if any(metrics[key] is None for key in METRIC_KEYS):
        return None
    return passed, flags, metrics


def ensure_schema(connection):
    connection.executescript(_NODE_FEATURES_DDL)
    connection.execute(_BLOCK_OBSERVATIONS_TRIGGER.format(version=PROFILER_VERSION))
    connection.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS ux_node_features_node_kind '
        'ON node_features(node_id, kind)')


def consolidate(cache_root, db_path, *, target):
    """Ingest cache files into ``node_features`` for every ``target`` node."""
    cache_root = Path(cache_root)
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.execute('PRAGMA busy_timeout=30000')
    ensure_schema(connection)
    nodes = connection.execute(
        'SELECT id, sequence FROM nodes WHERE target = ?', (target,)).fetchall()
    accepted = rejected = 0
    for node_id, sequence in nodes:
        path = cache_root / (_cache_key(sequence) + '.json')
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            rejected += 1
            continue
        parsed = _parse_profile(body.get('developability'))
        if parsed is None:
            rejected += 1
            continue
        passed, flags, metrics = parsed
        profile = {'passed': passed, 'flags': flags, **metrics}
        connection.execute(
            'INSERT INTO node_features (node_id, kind, source, profile_json, content_sha256, cache_key, ingested_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?) '
            'ON CONFLICT(node_id, kind) DO NOTHING',
            (node_id, PROFILER_VERSION, SOURCE, json.dumps(profile, sort_keys=True),
             _content_hash(sequence), _cache_key(sequence), None))
        accepted += 1
    connection.execute('INSERT INTO metadata (key, value) VALUES (?, ?) '
                       'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                       (VERSION_KEY, PROFILER_VERSION))
    connection.commit()
    connection.close()
    return accepted, rejected, len(nodes)


def attach(rows, db_path, *, target, min_coverage=None):
    """Join ``node_features`` onto parent-bank rows, read-only and fail-loud."""
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=30)
    try:
        version = connection.execute(
            'SELECT value FROM metadata WHERE key = ?', (VERSION_KEY,)).fetchone()
        if version is None or version[0] != PROFILER_VERSION:
            raise ValueError('developability store version mismatch')
        count = connection.execute(
            'SELECT COUNT(*) FROM node_features WHERE kind = ?', (PROFILER_VERSION,)).fetchone()[0]
        if count == 0:
            raise ValueError('developability store is empty (node_features has no rows)')
        sequences = [row['sequence'] for row in rows]
        profiles = {}
        seq_to_node = {}
        for start in range(0, len(sequences), 500):
            chunk = sequences[start:start + 500]
            placeholders = ','.join('?' for _ in chunk)
            for node_id, sequence in connection.execute(
                    f'SELECT id, sequence FROM nodes WHERE target = ? AND sequence IN ({placeholders})',
                    (target, *chunk)):
                seq_to_node[_content_hash(sequence)] = node_id
        node_ids = list(seq_to_node.values())
        node_to_profile = {}
        for start in range(0, len(node_ids), 500):
            chunk = node_ids[start:start + 500]
            placeholders = ','.join('?' for _ in chunk)
            for node_id, profile_json in connection.execute(
                    f'SELECT node_id, profile_json FROM node_features '
                    f'WHERE kind = ? AND node_id IN ({placeholders})',
                    (PROFILER_VERSION, *chunk)):
                node_to_profile[node_id] = json.loads(profile_json)
        for content_hash, node_id in seq_to_node.items():
            profile = node_to_profile.get(node_id)
            if profile is not None:
                profiles[content_hash] = profile
    finally:
        connection.close()
    matched = 0
    for row in rows:
        profile = profiles.get(_content_hash(row['sequence']))
        if profile is not None:
            if not isinstance(row.get('developability'), dict):
                row['developability'] = profile
            matched += 1
    if min_coverage is not None and rows:
        coverage = matched / len(rows)
        if coverage < min_coverage:
            raise ValueError(
                f'developability join coverage {coverage:.3f} below floor {min_coverage}')
    return {'matched': matched, 'total': len(rows)}


def migrate(db_path, *, target, expect_rows=None):
    """Move legacy F rows from ``observations`` into ``node_features`` (lossless)."""
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.execute('PRAGMA busy_timeout=30000')
    connection.execute('BEGIN IMMEDIATE')
    try:
        ensure_schema(connection)
        rows = connection.execute(
            'SELECT o.observation_id, o.node_id, o.metrics_json, o.provenance_json, o.source '
            'FROM observations o WHERE o.score_kind = ? ORDER BY o.node_id',
            (PROFILER_VERSION,)).fetchall()
        if expect_rows is not None and len(rows) != expect_rows:
            raise ValueError(f'migrate expected {expect_rows} rows, found {len(rows)}')
        if not rows:
            connection.rollback()
            connection.close()
            return {'moved': 0, 'note': 'nothing to migrate'}
        connection.execute(
            'CREATE TABLE IF NOT EXISTS f_observations_archive AS '
            'SELECT * FROM observations WHERE 0')
        connection.execute(
            'INSERT OR IGNORE INTO f_observations_archive SELECT * FROM observations '
            'WHERE score_kind = ?', (PROFILER_VERSION,))
        moved = 0
        for observation_id, node_id, metrics_json, provenance_json, source in rows:
            metrics = json.loads(metrics_json) if metrics_json else {}
            provenance = json.loads(provenance_json) if provenance_json else {}
            profile = {'passed': bool(provenance.get('passed')),
                       'flags': provenance.get('flags', {}), **metrics}
            connection.execute(
                'INSERT OR IGNORE INTO node_features '
                '(node_id, kind, source, profile_json, content_sha256, cache_key, ingested_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (node_id, PROFILER_VERSION, source or SOURCE, json.dumps(profile, sort_keys=True),
                 provenance.get('content_sha256') or '', provenance.get('cache_key') or '', None))
            moved += 1
        deleted = connection.execute(
            'DELETE FROM observations WHERE score_kind = ?', (PROFILER_VERSION,)).rowcount
        if deleted != len(rows):
            raise ValueError(f'migrate deleted {deleted} != read {len(rows)}')
        connection.execute('INSERT INTO metadata (key, value) VALUES (?, ?) '
                           'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                           (VERSION_KEY, PROFILER_VERSION))
        connection.commit()
        return {'moved': moved, 'archived': len(rows)}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
