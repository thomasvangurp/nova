"""Recover archive availability from verified, append-only CSV commit deltas.

Epoch labels describe archive additions, not private ancestry or replay draws.
The baseline is available before activation; its original submission epoch is
unknown. Every subsequent snapshot must reconcile exactly before it is imported.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re

from nanobody_opt import pr100_history as h

COMMIT = re.compile('[0-9a-f]{40}')
TITLE = re.compile(r'Added (\d+) nanobody sequences from epoch (\d+) to Q9NZQ7_nanobodies\.csv')


def rows(raw):
    if len(raw) > h.MAX_BYTES:
        raise ValueError('archive_size_limit')
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not reader.fieldnames or 'sequence' not in reader.fieldnames:
        raise ValueError('archive_sequence_column_missing')
    result = {}
    for row in reader:
        seq = h.sequence(row['sequence'])
        if row.get('sequence_hash') and row['sequence_hash'] != h.digest(seq.encode()):
            raise ValueError('archive_sequence_hash_mismatch')
        if seq in result:
            raise ValueError('duplicate_archive_sequence')
        metrics = {}
        for key, value in row.items():
            if key in ('sequence', 'sequence_hash'):
                continue
            try:
                number = float(value)
            except (ValueError, TypeError):
                raise ValueError('archive_metric_missing') from None
            if not math.isfinite(number):
                raise ValueError('archive_metric_not_finite')
            metrics[key] = number
        result[seq] = metrics
    return result


def commit_metadata(metadata):
    revision = metadata.get('id')
    match = TITLE.fullmatch(metadata.get('title', ''))
    if not isinstance(revision, str) or COMMIT.fullmatch(revision) is None or match is None:
        raise ValueError('invalid_archive_commit_metadata')
    count, epoch = map(int, match.groups())
    if not h.ERA_START <= epoch or count < 1:
        raise ValueError('invalid_archive_commit_epoch')
    return revision, count, epoch


def schema(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS history_archive_revisions(
            revision TEXT PRIMARY KEY,parent_revision TEXT,epoch INTEGER,
            source_id TEXT NOT NULL,previous_sha256 TEXT,sha256 TEXT NOT NULL,
            expected_additions INTEGER NOT NULL,actual_additions INTEGER NOT NULL,
            snapshot_rows INTEGER NOT NULL,metadata_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS history_archive_availability(
            node_id INTEGER PRIMARY KEY,first_archive_epoch INTEGER,
            available_before_epoch INTEGER NOT NULL,revision TEXT NOT NULL,
            source_id TEXT NOT NULL,baseline INTEGER NOT NULL);
    ''')


def import_baseline(conn, raw, *, revision):
    if not isinstance(revision, str) or COMMIT.fullmatch(revision) is None:
        raise ValueError('invalid_baseline_revision')
    data = rows(raw)
    if conn.execute('SELECT COUNT(*) FROM history_archive_revisions').fetchone()[0]:
        previous = conn.execute('SELECT sha256 FROM history_archive_revisions WHERE revision=?',
                                (revision,)).fetchone()
        if previous is None or previous[0] != h.digest(raw):
            raise ValueError('baseline_already_initialized')
        return {'baseline_rows': len(data), 'revision': revision, 'sha256': h.digest(raw)}
    with conn:
        ident = h.source(conn, 'target_archive_baseline', raw, 'huggingface-commit:' + revision,
                         None, len(data), {'available_before_epoch': h.ERA_START})
        conn.execute('INSERT INTO history_archive_revisions VALUES(?,NULL,NULL,?,NULL,?,?,?,?,?)',
                     (revision, ident, h.digest(raw), len(data), len(data), len(data), '{}'))
        for seq in data:
            nid = h.node(conn, seq)
            conn.execute('INSERT INTO history_archive_availability VALUES(?,NULL,?,?,?,1)',
                         (nid, h.ERA_START, revision, ident))
    return {'baseline_rows': len(data), 'revision': revision, 'sha256': h.digest(raw)}


def import_delta(conn, previous_raw, current_raw, *, metadata, parent_revision, before_epoch):
    revision, expected, epoch = commit_metadata(metadata)
    if type(before_epoch) is not int or epoch >= before_epoch:
        raise ValueError('archive_revision_not_completed')
    parent = conn.execute('SELECT sha256,epoch FROM history_archive_revisions WHERE revision=?',
                          (parent_revision,)).fetchone()
    if parent is None or parent[0] != h.digest(previous_raw):
        raise ValueError('archive_revision_parent_not_verified')
    if parent[1] is not None and parent[1] >= epoch:
        raise ValueError('archive_revision_epoch_order')
    old, new = rows(previous_raw), rows(current_raw)
    added = set(new) - set(old)
    if set(old) - set(new):
        raise ValueError('archive_revision_removed_sequences')
    if any(old[seq] != new[seq] for seq in old):
        raise ValueError('archive_revision_changed_metrics')
    if len(added) != expected:
        raise ValueError('archive_revision_addition_count_mismatch')
    existing = conn.execute('SELECT parent_revision,sha256 FROM history_archive_revisions WHERE revision=?',
                            (revision,)).fetchone()
    if existing is not None and existing != (parent_revision, h.digest(current_raw)):
        raise ValueError('archive_revision_changed')
    with conn:
        ident = h.source(conn, 'target_archive_csv_revision', current_raw,
                         'huggingface-commit:' + revision, epoch, len(added),
                         {'previous_sha256': h.digest(previous_raw), 'metadata': metadata})
        conn.execute('INSERT OR IGNORE INTO history_archive_revisions VALUES(?,?,?,?,?,?,?,?,?,?)',
                     (revision, parent_revision, epoch, ident, h.digest(previous_raw),
                      h.digest(current_raw), expected, len(added), len(new), h.canonical(metadata)))
        for seq in sorted(added):
            nid = h.node(conn, seq)
            conn.execute('INSERT OR IGNORE INTO history_archive_availability VALUES(?,?,?,?,?,0)',
                         (nid, epoch, epoch + 1, revision, ident))
            conn.execute('INSERT OR IGNORE INTO history_membership VALUES(?,?)', (ident, nid))
    return {'revision': revision, 'epoch': epoch, 'added': len(added),
            'snapshot_rows': len(new), 'sha256': h.digest(current_raw)}


def reconcile(conn, current_raw):
    current = rows(current_raw)
    indexed = {seq for seq, in conn.execute('SELECT n.sequence FROM history_archive_availability a '
                                           'JOIN nodes n ON n.id=a.node_id WHERE n.target=?', (h.TARGET,))}
    baseline = conn.execute('SELECT COUNT(*) FROM history_archive_availability WHERE baseline=1').fetchone()[0]
    additions = conn.execute('SELECT COUNT(*) FROM history_archive_availability WHERE baseline=0').fetchone()[0]
    missing, extra = set(current) - indexed, indexed - set(current)
    return {'csv_sequences': len(current), 'baseline_sequences': baseline,
            'epoch_attributed_additions': additions, 'missing_sequences': len(missing),
            'unexplained_extra_sequences': len(extra), 'csv_sha256': h.digest(current_raw),
            'all_csv_sequences_accounted_for': not missing and not extra,
            'competitor_private_parentage_proven': False,
            'raw_replay_draw_count_known': False}
