"""Target-scoped SQLite history; archived metrics never become replay evidence.

This module writes only an explicitly initialized Q9NZQ7 history store. The
legacy database is opened read-only. Known generation relationships and archive
membership are recorded separately from inferred similarity relationships.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import time

TARGET = 'Q9NZQ7'
ERA_START = 25955
AA = 'ACDEFGHIKLMNPQRSTVWY'
HEX = re.compile('[0-9a-f]{64}')
MAX_BYTES = 32 * 1024 * 1024
GENERATION_FIELDS = ('sequence', 'sequence_hash', 'source', 'parent_sequence',
                     'reference_hash', 'reference_stratum', 'proposal_lane',
                     'edit_policy', 'mutations', 'requested_v', 'requested_cdr3_length')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sequence(value):
    if not isinstance(value, str):
        raise ValueError('sequence_not_text')
    result = value.strip().upper()
    if not 1 <= len(result) <= 500 or any(c not in AA for c in result):
        raise ValueError('invalid_sequence')
    return result


def open_store(path):
    """Refuse an existing unmarked database, including the legacy production DB."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('history_symlink')
    path = path.absolute()
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        check = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)
        try:
            marker = check.execute('SELECT target,era_start FROM history_store').fetchone()
            if marker != (TARGET, ERA_START):
                raise ValueError('wrong_history_store')
        except sqlite3.Error as error:
            raise ValueError('unmarked_history_store') from error
        finally:
            check.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=1)
    if not exists:
        conn.executescript(Path(__file__).with_name('local_schema.sql').read_text())
        conn.executescript('''
            CREATE TABLE history_store(target TEXT PRIMARY KEY,era_start INTEGER NOT NULL);
            CREATE TABLE history_sources(
                source_id TEXT PRIMARY KEY,kind TEXT NOT NULL,sha256 TEXT NOT NULL,
                locator TEXT NOT NULL,epoch INTEGER,recorded_at REAL NOT NULL,
                expected_rows INTEGER NOT NULL,report_json TEXT NOT NULL);
            CREATE TABLE history_membership(
                source_id TEXT NOT NULL,node_id INTEGER NOT NULL,
                PRIMARY KEY(source_id,node_id));
            CREATE TABLE history_archive_metrics(
                source_id TEXT NOT NULL,node_id INTEGER NOT NULL,metrics_json TEXT NOT NULL,
                protocol TEXT,raw_observation_count INTEGER,
                PRIMARY KEY(source_id,node_id));
            CREATE TABLE history_unresolved(
                source_id TEXT NOT NULL,node_id INTEGER NOT NULL,reason TEXT NOT NULL,
                detail_json TEXT NOT NULL,PRIMARY KEY(source_id,node_id,reason));
            CREATE TABLE history_lineage_evidence(
                source_id TEXT NOT NULL,parent_node_id INTEGER NOT NULL,
                child_node_id INTEGER NOT NULL,epoch INTEGER NOT NULL,kind TEXT NOT NULL,
                PRIMARY KEY(source_id,parent_node_id,child_node_id,epoch,kind));
        ''')
        conn.execute('INSERT INTO history_store VALUES(?,?)', (TARGET, ERA_START))
        conn.commit()
    conn.execute('PRAGMA busy_timeout=1000')
    # Additive migrations also run for previously staged, marked history stores.
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS history_public_epochs(
            epoch INTEGER PRIMARY KEY,status TEXT NOT NULL,source_id TEXT,
            checked_at REAL NOT NULL,row_count INTEGER NOT NULL,
            http_status INTEGER,error_type TEXT);
        CREATE TABLE IF NOT EXISTS history_public_observations(
            source_id TEXT NOT NULL,observation_id TEXT NOT NULL,
            PRIMARY KEY(source_id,observation_id));
    ''')
    return conn


def edit_descriptors(left, right):
    """A deterministic valid edit description, without a minimality claim."""
    if len(left) == len(right):
        return [f'{a}{i + 1}{b}' for i, (a, b) in enumerate(zip(left, right)) if a != b]
    result = []
    for tag, i, end, j, stop in difflib.SequenceMatcher(
            a=left, b=right, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if tag in ('replace', 'delete'):
            result.append(f'del@{i + 1}:{left[i:end]}')
        if tag in ('replace', 'insert'):
            result.append(f'ins@{i + 1}:{right[j:stop]}')
    return result


def generation_records(rows, donors):
    """Small generation journal, including rejected candidates but no native blobs."""
    projected = [{k: row[k] for k in GENERATION_FIELDS if k in row} for row in rows]
    used = {row.get('reference_hash') for row in projected if row.get('reference_hash')}
    references = [{'sequence': row['sequence'], 'sequence_hash': row['sequence_hash']}
                  for row in donors if row.get('sequence_hash') in used]
    return {'generation_rows': projected, 'lineage_references': references,
            'generation_capture_complete': True,
            'generation_capture_scope': 'emitted_unique_candidates_before_annotation'}


def node(conn, value):
    seq = sequence(value)
    conn.execute('INSERT OR IGNORE INTO nodes(target,sequence,length,residue_vector) '
                 'VALUES(?,?,?,?)', (TARGET, seq, len(seq), bytes(AA.index(c) for c in seq)))
    return conn.execute('SELECT id FROM nodes WHERE target=? AND sequence=?',
                        (TARGET, seq)).fetchone()[0]


def source(conn, kind, raw, locator, epoch, count, report):
    if len(raw) > MAX_BYTES or not isinstance(locator, str) or not locator:
        raise ValueError('invalid_source')
    if epoch is not None and (type(epoch) is not int or epoch < ERA_START):
        raise ValueError('invalid_source_epoch')
    sha = digest(raw)
    ident = digest(canonical([TARGET, kind, sha, locator, epoch]).encode())
    conn.execute('INSERT OR IGNORE INTO history_sources VALUES(?,?,?,?,?,?,?,?)',
                 (ident, kind, sha, locator, epoch, time.time(), count, canonical(report)))
    return ident


def import_archive(conn, raw, *, locator):
    """Import every canonical CSV sequence without inventing epochs or raw passes."""
    if len(raw) > MAX_BYTES:
        raise ValueError('archive_size_limit')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not reader.fieldnames or 'sequence' not in reader.fieldnames:
        raise ValueError('archive_sequence_column_missing')
    rows = list(reader)
    accepted, rejected = {}, []
    for index, row in enumerate(rows):
        try:
            seq = sequence(row['sequence'])
            supplied = row.get('sequence_hash')
            if supplied and supplied != digest(seq.encode()):
                raise ValueError('archive_sequence_hash_mismatch')
        except ValueError as error:
            rejected.append({'row': index + 2, 'reason': str(error)})
            continue
        metrics = {}
        for key, value in row.items():
            if key in ('sequence', 'sequence_hash') or value in ('', None):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                metrics[key] = number
        # Preserve corrected/different metric records under distinct source hashes.
        if seq in accepted and accepted[seq] != metrics:
            raise ValueError('conflicting_archive_duplicate')
        accepted[seq] = metrics
    report = {'input_rows': len(rows), 'unique_sequences': len(accepted),
              'rejected_rows': rejected, 'epoch_known': False,
              'protocol_known': False, 'raw_observation_count_known': False}
    with conn:
        ident = source(conn, 'target_archive_csv', raw, locator, None, len(rows), report)
        for seq, metrics in accepted.items():
            nid = node(conn, seq)
            conn.execute('INSERT OR IGNORE INTO history_membership VALUES(?,?)', (ident, nid))
            conn.execute('INSERT OR IGNORE INTO history_archive_metrics VALUES(?,?,?,NULL,NULL)',
                         (ident, nid, canonical(metrics)))
    stored = conn.execute('SELECT COUNT(*) FROM history_membership WHERE source_id=?',
                           (ident,)).fetchone()[0]
    return {**report, 'source_id': ident, 'sha256': digest(raw), 'stored_sequences': stored,
            'unexplained_missing': len(accepted) - stored}


def import_generation(conn, raw, *, locator, references=None, before_epoch=None):
    """Recover explicit parents/templates from a recorded PR100 generation artifact.

    A baseline's reference supplies its CDR3 template; it is not relabelled as a
    full direct parent. Missing reference sequences remain unresolved evidence.
    """
    if len(raw) > MAX_BYTES:
        raise ValueError('generation_size_limit')
    data = json.loads(raw)
    challenge = data.get('challenge', {})
    epoch = challenge.get('epoch')
    if (challenge.get('target') != TARGET or type(epoch) is not int or epoch < ERA_START
            or before_epoch is not None and epoch >= before_epoch):
        raise ValueError('generation_challenge_mismatch')
    rows = data.get('generation_rows', data.get('candidates'))
    if not isinstance(rows, list):
        raise ValueError('generation_rows_missing')
    refs = {}
    for row in references if references is not None else data.get('lineage_references', []):
        seq = sequence(row['sequence'])
        sha = digest(seq.encode())
        if row.get('sequence_hash', sha) != sha:
            raise ValueError('reference_hash_mismatch')
        refs[sha] = seq
    prepared = []
    for row in rows:
        child = sequence(row['sequence'])
        if row.get('sequence_hash', digest(child.encode())) != digest(child.encode()):
            raise ValueError('child_hash_mismatch')
        parent, kind, missing = None, None, None
        if row.get('source') == 'edit_generation' and row.get('parent_sequence'):
            parent, kind = sequence(row['parent_sequence']), 'recorded_direct_parent'
        elif row.get('source') == 'fresh_baseline_proposal' and row.get('reference_hash'):
            ref = row['reference_hash']
            if not isinstance(ref, str) or HEX.fullmatch(ref) is None:
                raise ValueError('invalid_reference_hash')
            parent, kind = refs.get(ref), 'recorded_cdr3_template'
            if parent is None:
                missing = {'reason': 'reference_sequence_unresolved', 'reference_hash': ref}
        prepared.append((row, child, parent, kind, missing))
    relationships = unresolved = 0
    with conn:
        ident = source(conn, 'pr100_generation_artifact', raw, locator, epoch, len(rows),
                       {'challenge': challenge, 'recorded_rows': len(rows),
                        'generation_capture_complete': data.get('generation_capture_complete'),
                        'generation_capture_scope': data.get('generation_capture_scope'),
                        'origin_artifact_sha256': data.get('origin_artifact_sha256')})
        for row, child, parent, kind, missing in prepared:
            cid = node(conn, child)
            conn.execute('INSERT OR IGNORE INTO history_membership VALUES(?,?)', (ident, cid))
            if missing:
                conn.execute('INSERT OR IGNORE INTO history_unresolved VALUES(?,?,?,?)',
                             (ident, cid, missing['reason'], canonical(missing)))
                unresolved += 1
            if parent is None or parent == child:
                continue
            pid = node(conn, parent)
            metadata = {'relationship_kind': kind, 'inferred': False, 'source_id': ident,
                        'artifact_sha256': digest(raw), 'parent_hash': digest(parent.encode()),
                        'child_hash': digest(child.encode()), 'edit_policy': row.get('edit_policy'),
                        'proposal_lane': row.get('proposal_lane'), 'challenge': challenge}
            conn.execute('INSERT OR IGNORE INTO lineage_edges('
                         'parent_node_id,child_node_id,epoch,source,proposal_type,mutations_json,metadata_json) '
                         'VALUES(?,?,?,?,?,?,?)',
                         (pid, cid, epoch, 'pr100_recorded:' + kind, kind,
                          canonical(row.get('mutations') or edit_descriptors(parent, child)),
                          canonical(metadata)))
            conn.execute('INSERT OR IGNORE INTO history_lineage_evidence VALUES(?,?,?,?,?)',
                         (ident, pid, cid, epoch, kind))
            relationships += 1
    return {'source_id': ident, 'epoch': epoch, 'rows': len(rows),
            'recorded_relationships': relationships, 'unresolved_references': unresolved}


def import_legacy(conn, legacy_path):
    """Copy Q9 nodes/observations into the isolated store, preserving legacy IDs."""
    source_path = Path(legacy_path).resolve()
    src = sqlite3.connect(source_path.as_uri() + '?mode=ro', uri=True, timeout=1)
    src.execute('PRAGMA query_only=ON')
    if src.execute('PRAGMA journal_mode').fetchone()[0] != 'wal':
        src.close()
        raise ValueError('legacy_read_snapshot_requires_wal')
    deadline = time.monotonic() + 30
    src.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
    src.execute('BEGIN')
    try:
        # Import into an empty node store only: preserved legacy IDs must not collide.
        if conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]:
            raise ValueError('legacy_import_requires_empty_store')
        with conn:
            for row in src.execute('SELECT id,target,sequence,length,residue_vector FROM nodes '
                                   'WHERE target=?', (TARGET,)):
                conn.execute('INSERT INTO nodes VALUES(?,?,?,?,?)', row)
            allowed = {r[0] for r in conn.execute('SELECT id FROM nodes')}
            for table in ('observations', 'observation_artifacts', 'lineage_edges',
                          'exact_runs', 'exact_run_observations'):
                columns = [r[1] for r in src.execute('PRAGMA table_info(' + table + ')')]
                if not columns:
                    continue
                if table == 'observations':
                    sql = 'SELECT o.* FROM observations o JOIN nodes n ON n.id=o.node_id WHERE n.target=?'
                elif table == 'observation_artifacts':
                    sql = ('SELECT a.* FROM observation_artifacts a JOIN observations o '
                           'ON o.observation_id=a.observation_id JOIN nodes n ON n.id=o.node_id WHERE n.target=?')
                elif table == 'lineage_edges':
                    sql = ('SELECT l.* FROM lineage_edges l JOIN nodes n ON n.id=l.child_node_id '
                           'JOIN nodes p ON p.id=l.parent_node_id WHERE n.target=? AND p.target=\'' + TARGET + '\'')
                else:
                    sql = 'SELECT * FROM ' + table + ' WHERE target=?'
                destination = {r[1] for r in conn.execute('PRAGMA table_info(' + table + ')')}
                selected = [i for i, name in enumerate(columns) if name in destination]
                names = ','.join(columns[i] for i in selected)
                insert = 'INSERT OR IGNORE INTO ' + table + '(' + names + ') VALUES(' + ','.join('?' for _ in selected) + ')'
                for row in src.execute(sql, (TARGET,)):
                    conn.execute(insert, tuple(row[i] for i in selected))
            report = {'target': TARGET, 'nodes': len(allowed),
                      'observations': conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
                      'unknown_epoch_observations': conn.execute('SELECT COUNT(*) FROM observations '
                                                               'WHERE epoch IS NULL').fetchone()[0]}
            conn.execute('INSERT INTO metadata(key,value) VALUES(?,?)',
                         ('q9_legacy_import', canonical(report)))
        return report
    finally:
        src.close()


def inventory(conn):
    """Counts are evidence of progress, never a claim of public-era completeness."""
    return {'target': TARGET,
            'nodes': conn.execute('SELECT COUNT(*) FROM nodes WHERE target=?', (TARGET,)).fetchone()[0],
            'other_target_nodes': conn.execute('SELECT COUNT(*) FROM nodes WHERE target<>?', (TARGET,)).fetchone()[0],
            'observations': conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0],
            'unknown_epoch_observations': conn.execute('SELECT COUNT(*) FROM observations WHERE epoch IS NULL').fetchone()[0],
            'recorded_relationships': conn.execute('SELECT COUNT(*) FROM lineage_edges').fetchone()[0],
            'unresolved_relationships': conn.execute('SELECT COUNT(*) FROM history_unresolved').fetchone()[0],
            'source_snapshots': conn.execute('SELECT COUNT(*) FROM history_sources').fetchone()[0],
            'orphan_observations': conn.execute('SELECT COUNT(*) FROM observations o LEFT JOIN nodes n '
                                               'ON o.node_id=n.id WHERE n.id IS NULL').fetchone()[0],
            'canonical_csv_coverage': archive_coverage(conn),
            'public_era_completeness_proven': False}


def archive_coverage(conn):
    """Sequence completeness is established by the CSV, independently of API gaps.

    This verifies a specific downloaded snapshot. Missing miner/epoch/replay
    metadata never makes an imported CSV sequence disappear from the store.
    """
    row = conn.execute("SELECT source_id,sha256,expected_rows,report_json FROM history_sources "
                       "WHERE kind='target_archive_csv' ORDER BY recorded_at DESC,source_id DESC LIMIT 1").fetchone()
    if row is None:
        return {'status': 'not_imported', 'all_snapshot_sequences_stored': False}
    ident, sha, expected_rows, report_json = row
    report = json.loads(report_json)
    stored = conn.execute('SELECT COUNT(*) FROM history_membership m JOIN nodes n '
                          'ON n.id=m.node_id WHERE m.source_id=? AND n.target=?',
                          (ident, TARGET)).fetchone()[0]
    rejected = len(report['rejected_rows'])
    unique = report['unique_sequences']
    complete = not rejected and stored == unique
    return {'status': 'matched' if complete else 'incomplete', 'source_id': ident,
            'sha256': sha, 'input_rows': expected_rows, 'unique_sequences': unique,
            'stored_sequences': stored, 'rejected_rows': rejected,
            'missing_sequences': unique - stored, 'all_snapshot_sequences_stored': complete,
            'epoch_metadata_required_for_sequence_coverage': False,
            'raw_replay_metadata_required_for_sequence_coverage': False}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('legacy', 'archive', 'generation', 'inventory'))
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--before-epoch', type=int)
    args = parser.parse_args(argv)
    if args.operation != 'inventory' and args.input is None:
        parser.error('--input is required for this operation')
    conn = None
    try:
        if args.operation == 'inventory' and not args.store.is_file():
            raise ValueError('history_store_missing')
        conn = open_store(args.store)
        if args.operation == 'legacy':
            report = import_legacy(conn, args.input)
        elif args.operation in ('archive', 'generation'):
            with args.input.open('rb') as stream:
                content = stream.read(MAX_BYTES + 1)
            if args.operation == 'archive':
                report = import_archive(conn, content, locator=str(args.input))
            else:
                report = import_generation(conn, content, locator=str(args.input),
                                           before_epoch=args.before_epoch)
        else:
            report = inventory(conn)
        print(canonical({'status': 'complete', 'operation': args.operation, 'report': report}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        # Never emit raw database, transport, or arbitrary path/error contents.
        print(canonical({'status': 'failed', 'operation': args.operation,
                         'error_type': type(error).__name__}))
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == '__main__':
    raise SystemExit(main())
