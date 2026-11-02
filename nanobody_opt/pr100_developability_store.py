"""Join the native developability profile (raw PSH) into the parent bank via the Q9-scoped DB.

The profile is computed by the native profiler and cached on the master as
``<cache_root>/<sha256(VERSION + "\\n" + sequence)>.json``. The durable store is
the Q9-scoped ``nodes``/``observations`` SQLite DB (schema-identical to the
historical multi-target bank, filtered to the Q9 target). This module:

- ``consolidate`` ingests the cache files into ``observations`` as rows with
  ``score_kind`` = the profiler version, ``metrics_json`` = the six metrics,
  and ``provenance_json`` = ``{passed, flags, content_sha256, cache_key}``.
- ``attach`` joins those rows onto parent-bank rows by ``sequence`` -> ``nodes``
  -> ``observations``, read-only, so the selector reads the real profile
  instead of the leaderboard ``developability_rank`` proxy.

Invariants: a known profile is never replaced by NULL; ``quality_evidence`` is
never touched; the join is read-only on the slave; the store is fail-loud when
configured (a missing/empty store is an error, not a silent fallback).
"""
from __future__ import annotations

import hashlib
import json
import math
import secrets
import sqlite3
from pathlib import Path

# Mirrors nanobody_opt.filter_layer2.DEVELOPABILITY_CACHE_VERSION; the two must
# stay in lockstep so a version bump invalidates both the cache and this store.
PROFILER_VERSION = 'tnp-2026-07-redregion-v1'
SCORE_KIND = PROFILER_VERSION
SOURCE = 'tnp_profiler'
VERSION_KEY = 'tnp.version'
METRIC_KEYS = ('total_cdr_length', 'cdr3_length', 'cdr3_compactness',
               'surface_hydrophobic_patches', 'positive_charge_patches',
               'negative_charge_patches')
# The profiler's per-metric red/green verdicts use short flag names, not the
# metric keys (see tnp_daemon.py: L, L3, C, PSH, PPC, PNC).
FLAG_KEYS = ('L', 'L3', 'C', 'PSH', 'PPC', 'PNC')


def _cache_key(sequence):
    return hashlib.sha256((PROFILER_VERSION + '\n' + sequence).encode('ascii')).hexdigest()


def _content_hash(sequence):
    return hashlib.sha256(sequence.encode('ascii')).hexdigest()


def _number(value):
    if type(value) in (int, float) and math.isfinite(value):
        return value
    return None


def _parse_profile(developability):
    """Validate one cached dict into (passed, flags, metrics) or None."""
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


def consolidate(cache_root, db_path, *, target):
    """Ingest cache files into ``observations`` for every ``target`` node.

    Iterates the existing nodes (not the files), so a cache file without a node
    is ignored and an already-measured profile is never recomputed. The profile
    is deterministic per (content, version), so the upsert is ``DO NOTHING``.
    Returns ``(accepted, rejected, matched_nodes)``.
    """
    cache_root = Path(cache_root)
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.execute('PRAGMA busy_timeout=30000')
    nodes = connection.execute(
        'SELECT id, sequence FROM nodes WHERE target = ?', (target,)).fetchall()
    accepted = rejected = 0
    rows = []
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
        rows.append((f'{PROFILER_VERSION}:{node_id}', node_id, SOURCE, SCORE_KIND,
                     1.0 if passed else 0.0,
                     json.dumps(metrics, sort_keys=True),
                     json.dumps({'passed': passed, 'flags': flags,
                                 'content_sha256': _content_hash(sequence),
                                 'cache_key': _cache_key(sequence)}, sort_keys=True),
                     secrets.randbits(63)))
        accepted += 1
    connection.executemany(
        'INSERT INTO observations '
        '(observation_id, node_id, source, score_kind, score, observation_count, metrics_json, provenance_json, repl_token) '
        'VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?) '
        'ON CONFLICT(observation_id) DO NOTHING',
        rows)
    connection.execute(
        'INSERT INTO metadata (key, value) VALUES (?, ?) '
        'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
        (VERSION_KEY, PROFILER_VERSION))
    connection.commit()
    connection.close()
    return accepted, rejected, len(nodes)


def _load_profiles(connection, target, sequences):
    """Return ``{content_sha256: profile}`` for bank sequences present in the store."""
    profiles = {}
    if not sequences:
        return profiles
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
        for node_id, metrics_json, provenance_json in connection.execute(
                f'SELECT node_id, metrics_json, provenance_json FROM observations '
                f'WHERE score_kind = ? AND node_id IN ({placeholders})',
                (SCORE_KIND, *chunk)):
            metrics = json.loads(metrics_json) if metrics_json else {}
            provenance = json.loads(provenance_json) if provenance_json else {}
            profile = {'passed': bool(provenance.get('passed')),
                       'flags': provenance.get('flags', {}), **metrics}
            node_to_profile[node_id] = profile
    for content_hash, node_id in seq_to_node.items():
        profile = node_to_profile.get(node_id)
        if profile is not None:
            profiles[content_hash] = profile
    return profiles


def attach(rows, db_path, *, target):
    """Join the durable profile onto parent-bank rows, read-only and fail-loud.

    Raises on a missing store, a version mismatch, or a store whose
    ``score_kind`` has zero rows (misconfigured). Never touches
    ``quality_evidence``; rows without a match are left for the coarse-rank
    fallback.
    """
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=30)
    try:
        version = connection.execute(
            'SELECT value FROM metadata WHERE key = ?', (VERSION_KEY,)).fetchone()
        if version is None or version[0] != PROFILER_VERSION:
            raise ValueError('developability store version mismatch')
        count = connection.execute(
            'SELECT COUNT(*) FROM observations WHERE score_kind = ?', (SCORE_KIND,)).fetchone()[0]
        if count == 0:
            raise ValueError('developability store is empty (score_kind has no rows)')
        sequences = [row['sequence'] for row in rows]
        profiles = _load_profiles(connection, target, sequences)
    finally:
        connection.close()
    for row in rows:
        profile = profiles.get(_content_hash(row['sequence']))
        if profile is not None and not isinstance(row.get('developability'), dict):
            row['developability'] = profile
    return rows
