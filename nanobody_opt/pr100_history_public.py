"""Strict public-epoch reconciliation for the isolated Q9 history database.

Public results are recorded as published observations, not raw replay draws.
Missing/empty/truncated boards remain coverage gaps. Corrected boards supersede
the active epoch pointer while their earlier evidence remains in the ledger.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request

from nanobody_opt import pr100_history as h

METRICS = ('design_iiptm', 'design_ptm', 'design_to_target_iptm',
           'min_design_to_target_pae', 'interaction_pae', 'plip_hbonds_refolded',
           'plip_saltbridge_refolded', 'delta_sasa_refolded', 'liability_score',
           'liability_num_violations', 'confidence_rank', 'physical_interaction_rank',
           'developability_rank', 'final_score')
MAX_ROWS = 10000


def check_epoch(epoch, before_epoch):
    if (type(epoch) is not int or type(before_epoch) is not int
            or not h.ERA_START <= epoch < before_epoch):
        raise ValueError('public_epoch_outside_completed_era')


def public_url(epoch):
    return ('https://compound-api.metanova-labs.ai/api/competitions/leaderboard/'
            f'{epoch}/nanobodies?limit=10000')


def normalize(raw):
    if len(raw) > h.MAX_BYTES:
        raise ValueError('public_size_limit')
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get('success') is not True
            or not isinstance(data.get('data'), list)):
        raise ValueError('invalid_public_board')
    members = data['data']
    if len(members) >= MAX_ROWS:
        raise ValueError('possibly_truncated_public_board')
    meta = data.get('meta')
    if meta is not None:
        if not isinstance(meta, dict):
            raise ValueError('invalid_public_meta')
        if any(meta.get(key) for key in ('has_more', 'next', 'next_cursor')):
            raise ValueError('paginated_public_board')
        for key in ('total', 'total_count'):
            if key in meta and (type(meta[key]) is not int or not 0 <= meta[key] <= len(members)):
                raise ValueError('truncated_public_board')
    rows, identities = [], set()
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get('nanobodies', []), list):
            raise ValueError('invalid_public_member')
        for nb in member.get('nanobodies', []):
            if not isinstance(nb, dict):
                raise ValueError('invalid_public_nanobody')
            if nb.get('protein_name') != h.TARGET:
                continue
            seq = h.sequence(nb.get('sequence'))
            uid, owner = member.get('uid'), member.get('coldkey')
            if type(uid) is not int or uid < 0 or not isinstance(owner, str) or not owner:
                raise ValueError('public_owner_identity_missing')
            metrics = {}
            for key in METRICS:
                value = nb.get(key)
                if value is None:
                    continue
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError('invalid_public_metric')
                metrics[key] = float(value)
            if 'final_score' not in metrics:
                raise ValueError('public_score_missing')
            identity = (uid, owner, seq)
            if identity in identities:
                raise ValueError('duplicate_public_entry')
            identities.add(identity)
            rows.append({'sequence': seq, 'uid': uid, 'coldkey': owner, 'metrics': metrics})
    if len(rows) >= MAX_ROWS:
        raise ValueError('possibly_truncated_target_board')
    return rows


def import_board(conn, raw, *, epoch, before_epoch, checked_at=None):
    check_epoch(epoch, before_epoch)
    rows = normalize(raw)  # A malformed row prevents a false complete-epoch claim.
    checked_at = time.time() if checked_at is None else checked_at
    status = 'available' if rows else 'empty_unverified'
    report = {'epoch': epoch, 'status': status, 'rows': len(rows),
              'raw_replay_protocol_known': False, 'raw_replay_draw_count_known': False}
    with conn:
        ident = h.source(conn, 'public_validator_board', raw, public_url(epoch), epoch,
                         len(rows), report)
        for row in rows:
            nid = h.node(conn, row['sequence'])
            observation_id = 'q9-public:' + h.digest(h.canonical([
                ident, row['uid'], row['coldkey'], row['sequence']]).encode())
            provenance = {'source_id': ident, 'source_sha256': h.digest(raw),
                          'epoch': epoch, 'uid': row['uid'], 'coldkey': row['coldkey'],
                          'evidence_kind': 'published_validator_result',
                          'scorer_protocol': None, 'raw_draw_count': None}
            conn.execute('INSERT OR IGNORE INTO observations('
                         'observation_id,node_id,epoch,source,score_kind,score,'
                         'observation_count,metrics_json,provenance_json) '
                         'VALUES(?,?,?,?,?,?,?,?,?)',
                         (observation_id, nid, epoch, 'q9_public_validator_board',
                          'published_final_score', row['metrics']['final_score'], 1,
                          h.canonical(row['metrics']), h.canonical(provenance)))
            conn.execute('INSERT OR IGNORE INTO history_membership VALUES(?,?)', (ident, nid))
            conn.execute('INSERT OR IGNORE INTO history_public_observations VALUES(?,?)',
                         (ident, observation_id))
        conn.execute('INSERT OR REPLACE INTO history_public_epochs VALUES(?,?,?,?,?,?,?)',
                     (epoch, status, ident, checked_at, len(rows), 200, None))
    return {**report, 'source_id': ident, 'sha256': h.digest(raw)}


def unavailable(conn, *, epoch, before_epoch, http_status=None, error_type=None,
                checked_at=None):
    check_epoch(epoch, before_epoch)
    allowed_errors = {'TimeoutError', 'URLError', 'JSONDecodeError', 'ValueError',
                      'OSError', 'HTTPError', 'ConnectionError', 'OtherError'}
    if error_type not in allowed_errors | {None}:
        error_type = 'OtherError'
    if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
        raise ValueError('invalid_http_status')
    # A later fetch failure must not delete already recovered scientific data.
    with conn:
        conn.execute('INSERT INTO history_public_epochs VALUES(?,?,?,?,?,?,?) '
                     'ON CONFLICT(epoch) DO UPDATE SET status=excluded.status,'
                     'checked_at=excluded.checked_at,http_status=excluded.http_status,'
                     'error_type=excluded.error_type',
                     (epoch, 'unavailable', None, time.time() if checked_at is None else checked_at,
                      0, http_status, error_type))
    return {'epoch': epoch, 'status': 'unavailable', 'http_status': http_status,
            'error_type': error_type}


def fetch(epoch, before_epoch):
    check_epoch(epoch, before_epoch)
    try:
        request = urllib.request.Request(public_url(epoch), headers={'User-Agent': 'nova-q9-history/1'})
        with urllib.request.urlopen(request, timeout=12) as response:
            raw = response.read(h.MAX_BYTES + 1)
        normalize(raw)
        return {'epoch': epoch, 'raw': raw}
    except urllib.error.HTTPError as error:
        return {'epoch': epoch, 'http_status': error.code, 'error_type': 'HTTPError'}
    except Exception as error:
        # No raw exception/response text escapes the transport boundary.
        return {'epoch': epoch, 'error_type': type(error).__name__}


def coverage(conn, *, before_epoch):
    check_epoch(h.ERA_START, before_epoch)
    fetched = {row[0]: row[1:] for row in conn.execute(
        'SELECT epoch,status,source_id,row_count,http_status,error_type '
        'FROM history_public_epochs WHERE epoch>=? AND epoch<?', (h.ERA_START, before_epoch))}
    epochs = []
    for epoch in range(h.ERA_START, before_epoch):
        status, ident, count, http, error = fetched.get(epoch, ('unfetched', None, 0, None, None))
        epochs.append({'epoch': epoch, 'status': status, 'source_id': ident,
                       'rows': count, 'http_status': http, 'error_type': error})
    gaps = [row for row in epochs if row['status'] != 'available']
    return {'target': h.TARGET, 'era_start': h.ERA_START, 'before_epoch': before_epoch,
            'epochs_expected': len(epochs), 'epochs_available': len(epochs) - len(gaps),
            'epochs': epochs, 'gaps': gaps,
            'public_api_interval_reconciled': not gaps,
            # API/CSV equality does not prove private competitor ancestry or raw replay coverage.
            'raw_multipass_coverage_proven': False, 'competitor_parentage_proven': False}
