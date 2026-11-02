#!/usr/bin/env python3
"""Scheduled developability ingest + Q9-scoped snapshot, stdlib-only.

Runs on the master box: (re)ingest the profiler's JSON cache into
``observations``, then emit a target-scoped snapshot (``nodes`` + F
``observations`` + ``metadata``) that the slave pulls read-only.

Three subcommands:
  init     -- build the idempotency index, failing loudly on duplicates.
  run      -- consolidate + build the scoped snapshot + write a status manifest.
  install  -- verify an incoming snapshot and atomically replace the live copy.

Never modifies an existing F row; a known F is never replaced by NULL. The
snapshot is written in one read transaction and installed by same-fs os.replace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

VERSION_RE = re.compile(r'[A-Za-z0-9._-]{1,64}')
TARGET = 'Q9NZQ7'


def _atomic_write_json(path, value):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _cache_key(version, content):
    return hashlib.sha256((version + '\n' + content).encode('ascii')).hexdigest()


def _number(value):
    if type(value) in (int, float) and value == value and value not in (float('inf'), float('-inf')):
        return value
    return None


def _parse_profile(developability):
    flags_ok = {'L', 'L3', 'C', 'PSH', 'PPC', 'PNC'}
    if not isinstance(developability, dict):
        return None
    passed = developability.get('passed')
    flags = developability.get('flags')
    if type(passed) is not bool or not isinstance(flags, dict) or set(flags) != flags_ok:
        return None
    metric_keys = ('total_cdr_length', 'cdr3_length', 'cdr3_compactness',
                   'surface_hydrophobic_patches', 'positive_charge_patches',
                   'negative_charge_patches')
    metrics = {k: _number(developability.get(k)) for k in metric_keys}
    if any(metrics[k] is None for k in metric_keys):
        return None
    return passed, flags, metrics


def _ingest(cache_root, db_path, version, source='tnp_profiler'):
    """Return (inserted, conflict_skipped, rejected, cache_arrivals_24h, newest_age)."""
    cache_root = Path(cache_root)
    files = list(cache_root.glob('*.json'))
    now = time.time()
    arrivals_24h = sum(1 for f in files if now - f.stat().st_mtime < 86400)
    newest_age = min((now - f.stat().st_mtime for f in files), default=0.0)

    con = sqlite3.connect(str(db_path), timeout=30)
    con.execute('PRAGMA busy_timeout=30000')
    nodes = con.execute('SELECT id, sequence FROM nodes WHERE target = ?', (TARGET,)).fetchall()

    inserted = conflict_skipped = rejected = 0
    metric_keys = ('total_cdr_length', 'cdr3_length', 'cdr3_compactness',
                   'surface_hydrophobic_patches', 'positive_charge_patches',
                   'negative_charge_patches')
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
        metrics_json = json.dumps(metrics, sort_keys=True)
        provenance = json.dumps({'passed': passed, 'flags': flags,
                                 'content_sha256': hashlib.sha256(sequence.encode('ascii')).hexdigest(),
                                 'cache_key': _cache_key(version, sequence)}, sort_keys=True)
        cur = con.execute(
            'SELECT 1 FROM observations WHERE observation_id = ?',
            (f'{version}:{node_id}',))
        if cur.fetchone():
            conflict_skipped += 1
            continue
        import secrets
        con.execute(
            'INSERT INTO observations (observation_id, node_id, source, score_kind, score, '
            'observation_count, metrics_json, provenance_json, repl_token) '
            'VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)',
            (f'{version}:{node_id}', node_id, source, version, 1.0 if passed else 0.0,
             metrics_json, provenance, secrets.randbits(63)))
        inserted += 1
    con.execute('INSERT INTO metadata (key, value) VALUES (?, ?) '
                'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                ('tnp.version', version))
    con.commit()
    con.close()
    return inserted, conflict_skipped, rejected, arrivals_24h, newest_age


def _build_snapshot(db_path, out_path, version):
    """Emit a WAL-free scoped snapshot (nodes + F obs + metadata) in one transaction."""
    src = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    src.execute('BEGIN')
    nodes = src.execute(
        'SELECT id, target, sequence, length, residue_vector FROM nodes WHERE target = ?',
        (TARGET,)).fetchall()
    obs = src.execute(
        'SELECT observation_id, node_id, epoch, source, score_kind, score, observation_count, '
        'metrics_json, pose_json, provenance_json, replicated_at, repl_token '
        'FROM observations WHERE score_kind = ? AND node_id IN (SELECT id FROM nodes WHERE target = ?)',
        (version, TARGET)).fetchall()
    meta = src.execute('SELECT key, value, replicated_at FROM metadata WHERE key = ?',
                       ('tnp.version',)).fetchall()
    src.execute('COMMIT')
    src.close()

    tmp = out_path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    dst = sqlite3.connect(tmp)
    dst.executescript(
        'CREATE TABLE nodes (id INTEGER PRIMARY KEY, target TEXT, sequence TEXT, length INTEGER, residue_vector BLOB);'
        'CREATE TABLE observations (observation_id TEXT PRIMARY KEY, node_id INTEGER NOT NULL, epoch INTEGER,'
        ' source TEXT NOT NULL, score_kind TEXT NOT NULL, score REAL NOT NULL,'
        ' observation_count INTEGER NOT NULL, metrics_json TEXT NOT NULL, pose_json TEXT,'
        ' provenance_json TEXT NOT NULL, replicated_at REAL, repl_token INTEGER NOT NULL);'
        'CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT, replicated_at REAL);')
    dst.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)', nodes)
    dst.executemany('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', obs)
    dst.executemany('INSERT INTO metadata VALUES (?,?,?)', meta)
    dst.commit()
    dst.execute('PRAGMA journal_mode=DELETE')
    dst.execute('PRAGMA user_version = 1')
    dst.commit()
    dst.close()
    os.replace(tmp, out_path)
    return len(nodes), len(obs)


def _verify(snapshot, version, *, prev=None, allow_shrink=False):
    for side in ('-wal', '-shm', '-journal'):
        if os.path.exists(snapshot + side):
            raise SystemExit(f'{snapshot}{side} exists: not a standalone snapshot')
    con = sqlite3.connect(f'file:{snapshot}?mode=ro', uri=True)
    n_nodes = con.execute('SELECT COUNT(*) FROM nodes WHERE target = ?', (TARGET,)).fetchone()[0]
    n_obs = con.execute('SELECT COUNT(*) FROM observations WHERE score_kind = ?',
                        (version,)).fetchone()[0]
    ver = con.execute("SELECT value FROM metadata WHERE key = 'tnp.version'").fetchone()
    jm = con.execute('PRAGMA journal_mode').fetchone()[0]
    con.close()
    if n_obs == 0:
        raise SystemExit('snapshot has zero F observations (empty join)')
    if ver is None or ver[0] != version:
        raise SystemExit('snapshot version mismatch')
    if jm.lower() != 'delete':
        raise SystemExit(f'snapshot journal_mode={jm}, expected delete')
    if prev is not None and os.path.exists(prev) and not allow_shrink:
        prev_con = sqlite3.connect(f'file:{prev}?mode=ro', uri=True)
        prev_n = prev_con.execute('SELECT COUNT(*) FROM observations WHERE score_kind = ?',
                                  (version,)).fetchone()[0]
        prev_con.close()
        if n_obs < prev_n:
            raise SystemExit(f'snapshot shrank ({prev_n} -> {n_obs}); use --allow-shrink deliberately')
    return {'n_nodes': n_nodes, 'n_obs': n_obs}


def cmd_init(args):
    con = sqlite3.connect(args.db)
    # version is validated against VERSION_RE (alnum + ._-), so inlining is safe;
    # SQLite forbids bind parameters in a partial-index WHERE clause.
    con.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS idx_obs_scorekind_node '
                f'ON observations(score_kind, node_id) '
                f"WHERE score_kind = '{args.version}'")
    # Partial index already forbids duplicates at build time; also report existing.
    dup = con.execute(
        'SELECT COUNT(*) FROM (SELECT node_id FROM observations WHERE score_kind = ? '
        'GROUP BY node_id HAVING COUNT(*) > 1)', (args.version,)).fetchone()[0]
    con.close()
    if dup:
        raise SystemExit(f'{dup} duplicate node rows already exist for {args.version}')
    print('INDEX_OK version=%s' % args.version)


def cmd_run(args):
    inserted, skipped, rejected, arrivals, newest = _ingest(
        args.cache_root, args.db, args.version)
    n_nodes, n_obs = _build_snapshot(args.db, args.out, args.version)
    status = {'ts': time.time(), 'version': args.version, 'target': TARGET,
              'ingest': {'inserted': inserted, 'conflict_skipped': skipped, 'rejected': rejected,
                         'cache_arrivals_24h': arrivals, 'cache_newest_age_s': round(newest, 1)},
              'snapshot': {'n_nodes': n_nodes, 'n_obs': n_obs}}
    _atomic_write_json(args.out + '.status.json', status)
    print('RUN inserted=%d skipped=%d rejected=%d snapshot_obs=%d arrivals_24h=%d' %
          (inserted, skipped, rejected, n_obs, arrivals))
    if rejected > max(5, 0.01 * (inserted + skipped + rejected)):
        sys.exit(3)


def cmd_install(args):
    dest_dir = os.path.dirname(os.path.abspath(args.dest))
    if os.path.dirname(os.path.abspath(args.incoming)) != dest_dir:
        raise SystemExit('incoming must live in the dest directory (same-fs atomic rename)')
    st = _verify(args.incoming, args.version, prev=args.dest, allow_shrink=args.allow_shrink)
    os.chmod(args.incoming, 0o444)
    os.replace(args.incoming, args.dest)
    print('INSTALLED dest=%s n_obs=%d' % (args.dest, st['n_obs']))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('init', 'run', 'install'):
        p = sp.add_parser(name)
        p.add_argument('--version', required=True)
        if name in ('init', 'run'):
            p.add_argument('--db', required=True)
        if name in ('run', 'install'):
            p.add_argument('--target', default=TARGET)
        if name == 'run':
            p.add_argument('--cache-root', required=True)
            p.add_argument('--out', required=True)
        if name == 'install':
            p.add_argument('--incoming', required=True)
            p.add_argument('--dest', required=True)
            p.add_argument('--allow-shrink', action='store_true')
    args = ap.parse_args()
    if not VERSION_RE.fullmatch(args.version):
        raise SystemExit('unsafe version string')
    {'init': cmd_init, 'run': cmd_run, 'install': cmd_install}[args.cmd](args)


if __name__ == '__main__':
    main()
