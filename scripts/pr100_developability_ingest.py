#!/usr/bin/env python3
"""Scheduled developability ingest + Q9-scoped snapshot, dedicated node_features table.

Runs on the master box. The profile lives in ``node_features`` (never the shared
``observations`` table). Subcommands:

  init     -- ensure node_features + the block trigger + the unique index.
  migrate  -- move any legacy F rows out of observations (lossless archive).
  run      -- ingest cache files -> node_features, build the scoped snapshot.
  install  -- verify + atomically replace the live snapshot.
  go       -- migrate (if needed) + run + install in one idempotent command.
  status   -- report coverage and gates.

Never modifies an existing profile; a known F is never replaced by NULL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
import time
from pathlib import Path

VERSION_RE = re.compile(r'[A-Za-z0-9._-]{1,64}')
TARGET = 'Q9NZQ7'
METRIC_KEYS = ('total_cdr_length', 'cdr3_length', 'cdr3_compactness',
               'surface_hydrophobic_patches', 'positive_charge_patches',
               'negative_charge_patches')
FLAG_KEYS = ('L', 'L3', 'C', 'PSH', 'PPC', 'PNC')


def _cache_key(version, content):
    return hashlib.sha256((version + '\n' + content).encode('ascii')).hexdigest()


def _content_hash(content):
    return hashlib.sha256(content.encode('ascii')).hexdigest()


def _number(value):
    if type(value) in (int, float) and value == value and value not in (float('inf'), float('-inf')):
        return value
    return None


def _parse_profile(developability):
    if not isinstance(developability, dict):
        return None
    passed = developability.get('passed')
    flags = developability.get('flags')
    if type(passed) is not bool or not isinstance(flags, dict) or set(flags) != set(FLAG_KEYS):
        return None
    metrics = {k: _number(developability.get(k)) for k in METRIC_KEYS}
    if any(metrics[k] is None for k in METRIC_KEYS):
        return None
    return passed, flags, metrics


def _ensure_schema(con, version):
    con.executescript(
        'CREATE TABLE IF NOT EXISTS node_features ('
        ' node_id INTEGER NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL,'
        ' profile_json TEXT NOT NULL, content_sha256 TEXT NOT NULL, cache_key TEXT NOT NULL,'
        ' ingested_at REAL, PRIMARY KEY (node_id, kind));')
    con.execute(
        f"CREATE TRIGGER IF NOT EXISTS block_f_observations BEFORE INSERT ON observations "
        f"WHEN NEW.score_kind = '{version}' BEGIN SELECT RAISE(IGNORE); END;")
    con.execute('CREATE UNIQUE INDEX IF NOT EXISTS ux_node_features_node_kind '
                'ON node_features(node_id, kind)')


def _ingest(cache_root, db_path, version, target, source='tnp_profiler'):
    cache_root = Path(cache_root)
    files = list(cache_root.glob('*.json'))
    now = time.time()
    arrivals_24h = sum(1 for f in files if now - f.stat().st_mtime < 86400)
    newest_age = min((now - f.stat().st_mtime for f in files), default=0.0)
    con = sqlite3.connect(str(db_path), timeout=30)
    con.execute('PRAGMA busy_timeout=30000')
    _ensure_schema(con, version)
    nodes = con.execute('SELECT id, sequence FROM nodes WHERE target = ?', (target,)).fetchall()
    inserted = skipped = rejected = 0
    for node_id, sequence in nodes:
        path = cache_root / (_cache_key(version, sequence) + '.json')
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
        cur = con.execute('SELECT 1 FROM node_features WHERE node_id = ? AND kind = ?',
                          (node_id, version))
        if cur.fetchone():
            skipped += 1
            continue
        profile = {'passed': passed, 'flags': flags, **metrics}
        con.execute(
            'INSERT INTO node_features (node_id, kind, source, profile_json, content_sha256, cache_key, ingested_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (node_id, version, source, json.dumps(profile, sort_keys=True),
             _content_hash(sequence), _cache_key(version, sequence), now))
        inserted += 1
    con.execute('INSERT INTO metadata (key, value) VALUES (?, ?) '
                'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                ('tnp.version', version))
    con.commit()
    con.close()
    return inserted, skipped, rejected, arrivals_24h, newest_age


def _build_snapshot(db_path, out_path, version, target):
    src = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    src.execute('BEGIN')
    nodes = src.execute(
        'SELECT id, target, sequence, length, residue_vector FROM nodes WHERE target = ?',
        (target,)).fetchall()
    feats = src.execute(
        'SELECT node_id, kind, source, profile_json, content_sha256, cache_key, ingested_at '
        'FROM node_features WHERE kind = ? AND node_id IN (SELECT id FROM nodes WHERE target = ?)',
        (version, target)).fetchall()
    meta = src.execute("SELECT key, value, replicated_at FROM metadata WHERE key = 'tnp.version'").fetchall()
    src.execute('COMMIT')
    src.close()

    tmp = out_path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    dst = sqlite3.connect(tmp)
    dst.executescript(
        'CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT, length INTEGER, residue_vector BLOB);'
        'CREATE TABLE node_features (node_id INTEGER NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL,'
        ' profile_json TEXT NOT NULL, content_sha256 TEXT NOT NULL, cache_key TEXT NOT NULL, ingested_at REAL,'
        ' PRIMARY KEY (node_id, kind));'
        'CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT, replicated_at REAL);')
    dst.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)', nodes)
    dst.executemany('INSERT INTO node_features VALUES (?,?,?,?,?,?,?)', feats)
    dst.executemany('INSERT INTO metadata VALUES (?,?,?)', meta)
    dst.commit()
    dst.execute('PRAGMA journal_mode=DELETE')
    dst.execute('PRAGMA user_version = 1')
    dst.commit()
    dst.close()
    os.replace(tmp, out_path)
    return len(nodes), len(feats)


def _verify(snapshot, version, target, prev=None, allow_shrink=False):
    for side in ('-wal', '-shm', '-journal'):
        if os.path.exists(snapshot + side):
            raise SystemExit(f'{snapshot}{side} exists: not a standalone snapshot')
    con = sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True)
    n_nodes = con.execute('SELECT COUNT(*) FROM nodes WHERE target = ?', (target,)).fetchone()[0]
    n_feats = con.execute('SELECT COUNT(*) FROM node_features WHERE kind = ?',
                          (version,)).fetchone()[0]
    ver = con.execute("SELECT value FROM metadata WHERE key = 'tnp.version'").fetchone()
    jm = con.execute('PRAGMA journal_mode').fetchone()[0]
    has_obs = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='observations'").fetchone()[0]
    con.close()
    if n_feats == 0:
        raise SystemExit('snapshot has zero node_features rows (empty join)')
    if ver is None or ver[0] != version:
        raise SystemExit('snapshot version mismatch')
    if jm.lower() != 'delete':
        raise SystemExit(f'snapshot journal_mode={jm}, expected delete')
    if has_obs:
        raise SystemExit('snapshot must not carry the observations table')
    if prev is not None and os.path.exists(prev) and not allow_shrink:
        prev_con = sqlite3.connect(f'file:{prev}?mode=ro', uri=True)
        prev_n = prev_con.execute('SELECT COUNT(*) FROM node_features WHERE kind = ?',
                                  (version,)).fetchone()[0]
        prev_con.close()
        if n_feats < prev_n:
            raise SystemExit(f'snapshot shrank ({prev_n} -> {n_feats}); use --allow-shrink deliberately')
    return {'n_nodes': n_nodes, 'n_feats': n_feats}


def _migrate(db_path, version, expect_rows=None):
    con = sqlite3.connect(str(db_path), timeout=30)
    con.execute('PRAGMA busy_timeout=30000')
    con.execute('BEGIN IMMEDIATE')
    try:
        _ensure_schema(con, version)
        rows = con.execute(
            'SELECT observation_id, node_id, metrics_json, provenance_json, source '
            'FROM observations WHERE score_kind = ? ORDER BY node_id', (version,)).fetchall()
        if expect_rows is not None and len(rows) != expect_rows:
            raise ValueError(f'migrate expected {expect_rows} rows, found {len(rows)}')
        if not rows:
            con.rollback(); con.close()
            return 0
        con.execute('CREATE TABLE IF NOT EXISTS f_observations_archive AS '
                    'SELECT * FROM observations WHERE 0')
        con.execute('INSERT OR IGNORE INTO f_observations_archive SELECT * FROM observations '
                    'WHERE score_kind = ?', (version,))
        moved = 0
        for observation_id, node_id, metrics_json, provenance_json, source in rows:
            metrics = json.loads(metrics_json) if metrics_json else {}
            provenance = json.loads(provenance_json) if provenance_json else {}
            profile = {'passed': bool(provenance.get('passed')),
                       'flags': provenance.get('flags', {}), **metrics}
            con.execute(
                'INSERT OR IGNORE INTO node_features '
                '(node_id, kind, source, profile_json, content_sha256, cache_key, ingested_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (node_id, version, source or 'tnp_profiler', json.dumps(profile, sort_keys=True),
                 provenance.get('content_sha256') or '', provenance.get('cache_key') or '', None))
            moved += 1
        deleted = con.execute('DELETE FROM observations WHERE score_kind = ?', (version,)).rowcount
        if deleted != len(rows):
            raise ValueError(f'migrate deleted {deleted} != read {len(rows)}')
        con.commit()
        return moved
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def cmd_init(a):
    con = sqlite3.connect(a.db)
    _ensure_schema(con, a.version)
    con.commit(); con.close()
    print('SCHEMA_OK version=%s' % a.version)


def cmd_migrate(a):
    moved = _migrate(a.db, a.version, expect_rows=a.expect_rows)
    print('MIGRATED moved=%d version=%s' % (moved, a.version))


def cmd_run(a):
    inserted, skipped, rejected, arrivals, newest = _ingest(
        a.cache_root, a.db, a.version, a.target)
    n_nodes, n_feats = _build_snapshot(a.db, a.out, a.version, a.target)
    status = {'ts': time.time(), 'version': a.version, 'target': a.target,
              'ingest': {'inserted': inserted, 'skipped': skipped, 'rejected': rejected,
                         'cache_arrivals_24h': arrivals, 'cache_newest_age_s': round(newest, 1)},
              'snapshot': {'n_nodes': n_nodes, 'n_feats': n_feats}}
    tmp = a.out + '.status.json.tmp'
    with open(tmp, 'w') as h:
        json.dump(status, h, sort_keys=True); h.flush(); os.fsync(h.fileno())
    os.replace(tmp, a.out + '.status.json')
    print('RUN inserted=%d skipped=%d rejected=%d snapshot_feats=%d arrivals_24h=%d' %
          (inserted, skipped, rejected, n_feats, arrivals))
    if rejected > max(5, 0.01 * (inserted + skipped + rejected)):
        sys.exit(3)


def cmd_install(a):
    dest_dir = os.path.dirname(os.path.abspath(a.dest))
    if os.path.dirname(os.path.abspath(a.snapshot)) != dest_dir:
        raise SystemExit('snapshot must live in the dest directory (same-fs atomic rename)')
    st = _verify(a.snapshot, a.version, a.target, prev=a.dest, allow_shrink=a.allow_shrink)
    os.chmod(a.snapshot, 0o444)
    os.replace(a.snapshot, a.dest)
    print('INSTALLED dest=%s n_feats=%d' % (a.dest, st['n_feats']))


def cmd_go(a):
    moved = _migrate(a.db, a.version, expect_rows=a.expect_migrate_rows)
    inserted, skipped, rejected, arrivals, newest = _ingest(
        a.cache_root, a.db, a.version, a.target)
    n_nodes, n_feats = _build_snapshot(a.db, a.out, a.version, a.target)
    if a.install_dest:
        st = _verify(a.out, a.version, a.target, prev=a.install_dest, allow_shrink=a.allow_shrink)
        if os.path.dirname(os.path.abspath(a.out)) != os.path.dirname(os.path.abspath(a.install_dest)):
            raise SystemExit('out and install-dest must share a directory')
        os.chmod(a.out, 0o444)
        os.replace(a.out, a.install_dest)
    print('GO migrated=%d inserted=%d skipped=%d rejected=%d snapshot_feats=%d arrivals_24h=%d' %
          (moved, inserted, skipped, rejected, n_feats, arrivals))


def cmd_status(a):
    con = sqlite3.connect(f'file:{a.db}?mode=ro', uri=True)
    try:
        n_nodes = con.execute('SELECT COUNT(*) FROM nodes WHERE target = ?', (a.target,)).fetchone()[0]
        n_feats = con.execute('SELECT COUNT(*) FROM node_features WHERE kind = ? AND node_id IN '
                              '(SELECT id FROM nodes WHERE target = ?)', (a.version, a.target)).fetchone()[0]
        n_obs = con.execute('SELECT COUNT(*) FROM observations WHERE score_kind = ?',
                            (a.version,)).fetchone()[0] if con.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='observations'").fetchone()[0] else 0
        print('STATUS nodes=%d feats=%d obs_left=%d coverage=%.3f' %
              (n_nodes, n_feats, n_obs, (n_feats / n_nodes) if n_nodes else 0.0))
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    def add(p, name, fn):
        s = sp.add_parser(name); s.set_defaults(func=fn); return s
    s = add(sp, 'init', cmd_init); s.add_argument('--db', required=True); s.add_argument('--version', required=True)
    s = add(sp, 'migrate', cmd_migrate); s.add_argument('--db', required=True); s.add_argument('--version', required=True); s.add_argument('--expect-rows', type=int)
    s = add(sp, 'run', cmd_run)
    for f in ('--db', '--cache-root', '--target', '--version', '--out'):
        s.add_argument(f, required=True)
    s = add(sp, 'install', cmd_install)
    for f in ('--snapshot', '--dest', '--version', '--target'):
        s.add_argument(f, required=True)
    s.add_argument('--allow-shrink', action='store_true')
    s = add(sp, 'go', cmd_go)
    for f in ('--db', '--cache-root', '--target', '--version', '--out'):
        s.add_argument(f, required=True)
    s.add_argument('--install-dest'); s.add_argument('--expect-migrate-rows', type=int); s.add_argument('--allow-shrink', action='store_true')
    s = add(sp, 'status', cmd_status); s.add_argument('--db', required=True); s.add_argument('--version', required=True); s.add_argument('--target', required=True)
    a = ap.parse_args()
    if not VERSION_RE.fullmatch(a.version):
        raise SystemExit('unsafe version string')
    a.func(a)


if __name__ == '__main__':
    main()
