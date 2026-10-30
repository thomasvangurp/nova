"""Immutable annotation-only parent banks and bounded public refresh state.

Neither parent merit nor a parent's native profile qualifies any child. The
live child's annotation, native, novelty and measured scoring gates still run.
"""
from __future__ import annotations

from bisect import bisect_left
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

from nanobody_opt import rules

SCHEMA = 'pr100-recurring-parent-bank-v1'
POINTER = 'pr100-qualified-parent-pointer-v1'
PUBLIC = 'pr100-public-parent-epoch-v1'
MAX_BYTES = 32 * 1024 * 1024
MAX_ROWS = 10000
METRICS = ('design_iiptm', 'design_ptm', 'design_to_target_iptm',
           'min_design_to_target_pae', 'interaction_pae', 'plip_hbonds_refolded',
           'plip_saltbridge_refolded', 'delta_sasa_refolded', 'liability_score',
           'liability_num_violations', 'confidence_rank', 'physical_interaction_rank',
           'developability_rank', 'final_score')


def require(value, message):
    if not value:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def hex64(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def read(path):
    path = Path(path)
    require(not path.is_symlink(), 'bank_symlink')
    with path.open('rb') as stream:
        value = stream.read(MAX_BYTES + 1)
    require(len(value) <= MAX_BYTES, 'bank_size_limit')
    return value


def immutable(path, value):
    """Existing content-addressed files are verified, never overwritten."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(value); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(read(path) == value, 'immutable_bank_changed')
    finally:
        os.unlink(temporary)


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with tmp.open('wb') as stream:
            stream.write(raw(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def settings(config):
    supplied = config.get('parent_bank_feed', {})
    require(isinstance(supplied, dict) and type(supplied.get('enabled', False)) is bool,
            'invalid_parent_feed')
    if not supplied.get('enabled', False):
        return None
    allowed = {'enabled', 'state_directory', 'remote_directory', 'cpu_affinity', 'workers',
               'wall_seconds', 'max_epoch_requests', 'max_pointer_age_epochs', 'max_retained_age_epochs'}
    require(not set(supplied) - allowed, 'unknown_parent_feed_setting')
    result = {'state_directory': str(Path(config['state_root']) / 'parent-feed'),
              'remote_directory': config['remote_root'] + '/parent-feed',
              'workers': 4, 'wall_seconds': 180, 'max_epoch_requests': 8,
              'max_pointer_age_epochs': 3, 'max_retained_age_epochs': 24, **supplied}
    require(all(isinstance(result[k], str) and Path(result[k]).is_absolute()
                and '..' not in Path(result[k]).parts for k in ('state_directory', 'remote_directory')),
            'parent_feed_paths_must_be_absolute')
    for key, low, high in (('workers', 1, 8), ('wall_seconds', 30, 240),
                           ('max_epoch_requests', 1, 12), ('max_pointer_age_epochs', 1, 24),
                           ('max_retained_age_epochs', 1, 72)):
        require(type(result[key]) is int and low <= result[key] <= high, 'invalid_parent_feed_bound')
    cpus = result.get('cpu_affinity')
    require(isinstance(cpus, list) and result['workers'] <= len(cpus) <= 8
            and all(type(v) is int and v >= 0 for v in cpus) and len(set(cpus)) == len(cpus),
            'parent_feed_requires_dedicated_cpus')
    require(result['max_retained_age_epochs'] >= result['max_pointer_age_epochs'],
            'invalid_parent_retention_age')
    require(hex64(config['hosts']['slave'].get('reference_bank_sha256')),
            'configured_bank_requires_hash')
    from nanobody_opt.pr100_novelty_runtime import validate_config
    validate_config(config['hosts']['slave'].get('novelty_runtime'))
    return result


def public_epoch(payload, target, epoch, before_epoch):
    """Normalize one complete public board; percentile fields never cross epochs."""
    require(type(epoch) is int and epoch < before_epoch, 'future_public_epoch')
    require(isinstance(payload, dict) and payload.get('success') is True
            and isinstance(payload.get('data'), list), 'invalid_public_board')
    require(len(payload['data']) < MAX_ROWS, 'possibly_truncated_public_board')
    meta = payload.get('meta')
    if isinstance(meta, dict):
        require(not meta.get('has_more') and not meta.get('next') and not meta.get('next_cursor'),
                'paginated_public_board')
        for key in ('total', 'total_count'):
            if key in meta:
                require(type(meta[key]) is int and meta[key] <= len(payload['data']),
                        'truncated_public_board')
    rows = []
    for item in payload['data']:
        require(isinstance(item, dict) and isinstance(item.get('nanobodies', []), list),
                'invalid_public_member')
        for nb in item.get('nanobodies', []):
            if not isinstance(nb, dict) or nb.get('protein_name') != target:
                continue
            sequence = str(nb.get('sequence', '')).strip().upper()
            score = nb.get('final_score')
            if (re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{60,200}', sequence) is None
                    or type(score) not in (int, float) or not math.isfinite(score)):
                continue
            rows.append({'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence),
                'metrics': {k: float(nb[k]) for k in METRICS
                    if type(nb.get(k)) in (int, float) and math.isfinite(nb[k])},
                'uid': item.get('uid') if type(item.get('uid')) is int else None,
                'coldkey': item.get('coldkey') if re.fullmatch(
                    '[1-9A-HJ-NP-Za-km-z]{48}', str(item.get('coldkey'))) else None})
    require(len(rows) < MAX_ROWS, 'possibly_truncated_target_board')
    rows.sort(key=lambda r: (r['sequence_hash'], r['uid'] if r['uid'] is not None else -1))
    field = sorted(r['metrics']['final_score'] for r in rows)
    field_sha = sha(raw(rows))
    for row in rows:
        score = row['metrics']['final_score']
        row['quality_evidence'] = [] if len(field) < 2 else [{
            'kind': 'public_validator_percentile', 'target': target, 'epoch': epoch,
            'value': bisect_left(field, score) / (len(field) - 1), 'field_size': len(field),
            'field_sha256': field_sha, 'raw_score': score, 'passes': 1,
            'source_url': f'https://compound-api.metanova-labs.ai/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000',
            'metrics': row['metrics']}]
        row['source'] = 'public_validator_parent_feed'
        row['public_observations'] = [{'target': target, 'epoch': epoch,
            'uid': row['uid'], 'coldkey': row['coldkey'], 'field_sha256': field_sha,
            'metrics': row['metrics']}]
    return {'schema': PUBLIC, 'target': target, 'epoch': epoch,
            'field_sha256': field_sha, 'rows': rows}


def due_epochs(ledger, before_epoch, now, limit=8):
    """Revisit recent/delayed boards, then oldest missing or due backfill work."""
    start = rules.ACTIVATION_EPOCH
    def due(epoch):
        item = ledger.get(str(epoch), {})
        return now >= item.get('retry_after_unix', 0)
    recent = [e for e in range(before_epoch - 1, max(start - 1, before_epoch - 4), -1) if due(e)]
    older = [e for e in range(start, before_epoch - 3) if due(e)]
    older.sort(key=lambda e: (ledger.get(str(e), {}).get('last_checked_unix', 0), e))
    return (recent[:max(1, limit // 2)] + older + recent[max(1, limit // 2):])[:limit]


def merge_rows(old, public, *, target, before_epoch):
    """Preserve prior provenance; updated same-epoch percentiles keep audit history."""
    from tools.de_novo_bank import validate
    validate(old)
    merged = {r['sequence_hash']: copy.deepcopy(r) for r in old}
    require(len(merged) == len(old), 'duplicate_old_parent')
    for row in public:
        require(isinstance(row, dict) and rules.sequence_hash(row['sequence']) == row['sequence_hash'],
                'public_parent_identity')
        saved = merged.setdefault(row['sequence_hash'], {
            'sequence': row['sequence'], 'sequence_hash': row['sequence_hash'],
            'source': 'public_validator_parent_feed', 'native_qualified': False})
        evidence = saved.setdefault('quality_evidence', [])
        lineage = saved.setdefault('public_observations', [])
        for observation in row.get('public_observations', []):
            require(observation.get('target') == target and type(observation.get('epoch')) is int
                    and observation['epoch'] < before_epoch
                    and (observation.get('uid') is None or type(observation['uid']) is int)
                    and (observation.get('coldkey') is None or re.fullmatch(
                        '[1-9A-HJ-NP-Za-km-z]{48}', str(observation['coldkey'])) is not None)
                    and hex64(observation.get('field_sha256')), 'invalid_parent_public_lineage')
            if observation not in lineage:
                lineage.append(copy.deepcopy(observation))
        for item in row.get('quality_evidence', []):
            require(item.get('target') == target and type(item.get('epoch')) is int
                and item['epoch'] < before_epoch and item.get('kind') == 'public_validator_percentile'
                and item.get('passes') == 1 and type(item.get('field_size')) is int
                and item['field_size'] > 1 and type(item.get('value')) in (int, float)
                and math.isfinite(item['value']) and 0 <= item['value'] <= 1
                and hex64(item.get('field_sha256')), 'invalid_public_parent_merit')
            key = (item['kind'], item['target'], item['epoch'])
            replaced = [v for v in evidence if (v.get('kind'), v.get('target'), v.get('epoch')) == key
                        and v != item]
            for value in replaced:
                history = saved.setdefault('quality_evidence_history', [])
                if value not in history:
                    history.append(copy.deepcopy(value))
                evidence.remove(value)
            if item not in evidence:
                evidence.append(copy.deepcopy(item))
    require(len(merged) <= MAX_ROWS, 'parent_bank_capacity')
    return list(merged.values())


def expected_identity(config, target):
    from nanobody_opt.pr100_filter import annotation_identity
    runtime = config['hosts']['slave']['novelty_runtime']
    return {'source_sha': config['source_sha'], 'target': target,
            'runtime_sha256': runtime['identity_sha256'], 'annotation_identity': annotation_identity()}


def validate_pointer(pointer, config, challenge, *, fresh=True, reannotation_input=False):
    cfg = settings(config); root = Path(cfg['state_directory']) / challenge['target']
    require(isinstance(pointer, dict) and pointer.get('schema') == POINTER, 'invalid_bank_pointer')
    if reannotation_input:
        # Old bytes may be INPUT to a fresh full annotation run, never an
        # executable worker snapshot under a different scientific identity.
        require(pointer.get('target') == challenge['target']
            and isinstance(pointer.get('source_sha'), str)
            and re.fullmatch('[0-9a-f]{40}', pointer['source_sha']) is not None
            and hex64(pointer.get('runtime_sha256')) and hex64(pointer.get('annotation_identity')),
            'invalid_historical_bank_identity')
    else:
        for key, expected in expected_identity(config, challenge['target']).items():
            require(pointer.get(key) == expected, 'bank_pointer_identity_drift')
    epoch = pointer.get('before_epoch')
    require(type(epoch) is int and 0 <= epoch <= challenge['epoch'], 'future_bank_pointer')
    age_limit = cfg['max_pointer_age_epochs'] if fresh else cfg['max_retained_age_epochs']
    if not reannotation_input:
        require(challenge['epoch'] - epoch <= age_limit, 'stale_bank_pointer')
    for name in ('bank_sha256', 'qualification_sha256'):
        require(hex64(pointer.get(name)), 'invalid_bank_pointer_digest')
    bank_path = root / ('bank-' + pointer['bank_sha256'] + '.json')
    receipt_path = root / ('qualification-' + pointer['qualification_sha256'] + '.json')
    bank_raw, receipt_raw = read(bank_path), read(receipt_path)
    require(sha(bank_raw) == pointer['bank_sha256'] and sha(receipt_raw) == pointer['qualification_sha256'],
            'qualified_bank_bytes_changed')
    bank, receipt = json.loads(bank_raw), json.loads(receipt_raw)
    require(receipt.get('status') == 'complete' and receipt.get('bank_sha256') == pointer['bank_sha256']
            and receipt.get('cleanup', {}).get('remaining_descendants') == 0
            and receipt.get('cleanup', {}).get('fixed_point_reached') is True,
            'bank_cleanup_not_verified')
    at = receipt.get('completed_at_unix')
    require(type(at) in (int, float) and math.isfinite(at) and 0 < at <= time.time() + 30
            and pointer.get('qualified_at_unix') == at, 'bank_qualification_time')
    for key in (*expected_identity(config, challenge['target']), 'before_epoch'):
        require(bank.get(key) == pointer[key] and receipt.get(key) == pointer[key], 'bank_receipt_identity')
    require(bank.get('schema') == SCHEMA and bank.get('native_qualified') is False
            and bank.get('novelty_qualified') is False and len(bank.get('rows', [])) > 0
            and len(bank['rows']) == receipt.get('annotated_rows'), 'invalid_annotated_bank')
    expected_remote = cfg['remote_directory'] + '/banks/bank-' + pointer['bank_sha256'] + '.json'
    require(pointer.get('remote_bank_path') == expected_remote, 'bank_remote_path_changed')
    return pointer


def reannotation_base(config, challenge):
    """Retain accumulated coverage across source changes, without old approval."""
    cfg = settings(config); root = Path(cfg['state_directory']) / challenge['target']
    for name in ('current.json', 'accepted.json'):
        try:
            pointer = validate_pointer(json.loads(read(root / name)), config, challenge,
                                       fresh=False, reannotation_input=True)
            return pointer['remote_bank_path'], pointer['bank_sha256']
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return config['hosts']['slave']['references'], config['hosts']['slave']['reference_bank_sha256']


def snapshot(config, challenge):
    """Called once for a NEW sequence attempt, never for an existing worker."""
    cfg = settings(config)
    if cfg is None:
        return None
    root = Path(cfg['state_directory']) / challenge['target']
    fallback = {'references': config['hosts']['slave']['references'],
                'reference_bank_sha256': config['hosts']['slave'].get('reference_bank_sha256'),
                'status': 'configured_fallback'}
    reason = None
    for name, fresh in (('current.json', True), ('accepted.json', False)):
        try:
            pointer = validate_pointer(json.loads(read(root / name)), config, challenge, fresh=fresh)
            if fresh:
                atomic(root / 'accepted.json', pointer)
            result = {'references': pointer['remote_bank_path'],
                'reference_bank_sha256': pointer['bank_sha256'], 'qualification_sha256': pointer['qualification_sha256'],
                'source_sha': pointer['source_sha'], 'runtime_sha256': pointer['runtime_sha256'],
                'before_epoch': pointer['before_epoch'], 'status': 'fresh' if fresh else 'retained_qualified',
                'pointer_sha256': sha(raw(pointer))}
            break
        except (OSError, ValueError, KeyError, TypeError) as error:
            reason = str(error) if isinstance(error, ValueError) else type(error).__name__
    else:
        result = fallback
    result['refresh_issue'] = reason
    try:
        atomic(root / 'snapshot-status.json', {'epoch': challenge['epoch'], 'at_unix': time.time(),
                                             **result})
    except OSError:
        pass  # An unavailable diagnostic disk cannot strand an otherwise valid epoch.
    return result
