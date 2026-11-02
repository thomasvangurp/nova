"""Durable lineage and measured-outcome analysis for PR100 selected sequences.

This is a read-only, non-gating analysis layer. It reuses the existing parent
feed's ``public_observations`` (owner epoch/coldkey provenance) and the edit
lane's emitted lineage (``parent_sequence`` / ``proposal_lane`` /
``edit_policy``) to say, per selected sequence, where it came from and what
cell it was selected for, then tracks the top/bottom measured outcomes over the
last ten completed epochs under one honest metric field/protocol.

Each historical epoch is read from its own recorded ``challenge`` (gene/CDR3
cell and target), never from the current challenge, so a previous epoch's cell
is never relabelled. Outcomes are ``locally_scored``: the metric is our own
scorer's measurement under a named scoring identity, explicitly NOT the
leaderboard/on-chain final score. Whether a sequence was published is read from
the epoch's published-quality record; an on-chain validator result is never
invented.

Nothing here changes a gate, a budget, or a publication decision. Unknowns are
exported explicitly: a missing parent, a missing owner coldkey/epoch, a missing
cell, a missing historical challenge, a non-measured metric and a non-comparable
scoring identity are all labelled rather than guessed or silently mixed.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

from nanobody_opt import rules
from nanobody_opt import pr100_filter

SCHEMA = 'pr100-lineage-outcomes-v1'
DEFAULT_METRIC_FIELD = 'design_iiptm'
DEFAULT_EPOCHS = 10

#: Fields a scoring identity must carry before its measurements are comparable.
SCORING_IDENTITY_FIELDS = ('scorer_sha256', 'settings_sha256',
                           'settings_file_sha256', 'target')

_GENE = re.compile(r'IGHV3-\d+')


def metric_directions() -> dict[str, str]:
    """Scorer metric field -> optimization direction, from the pinned config.

    Loaded lazily so a drift between this analysis and the scorer's own config
    is caught instead of silently mis-ranking a field.
    """
    from config.config_loader import load_boltzgen_metrics
    root = Path(__file__).resolve().parents[1]
    return load_boltzgen_metrics(str(root / 'config/boltzgen_config.yaml'))


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def cell_of(row, challenge):
    """Gene/CDR3-length cell a selected sequence was generated for.

    The worker records ``requested_v``/``requested_cdr3_length`` on edit rows;
    every other lane targets the challenge cell, so that is the honest fallback.
    """
    gene = row.get('requested_v')
    length = row.get('requested_cdr3_length')
    if not (isinstance(gene, str) and _GENE.fullmatch(gene)):
        gene = challenge.get('allowed_v')
    if not (type(length) is int and 5 <= length <= 23):
        length = challenge.get('allowed_cdr3_length')
    if isinstance(gene, str) and _GENE.fullmatch(gene) and type(length) is int and 5 <= length <= 23:
        return {'gene': gene, 'cdr3_length': length, 'cell': f'{gene}:{length}'}
    return {'gene': gene, 'cdr3_length': length, 'cell': None, 'unknown': 'cell_unknown'}


def edit_kind(row):
    """The proposal/edit lane that produced a selected sequence."""
    source = row.get('source')
    if source == 'segment_recombination':
        return {'kind': row.get('proposal_lane') or 'segment_lane_unknown',
                'policy': row.get('proposal_type'), 'source': source}
    if source == 'leaderboard_graft':
        return {'kind': row.get('proposal_lane') or 'graft_lane_unknown',
                'policy': row.get('proposal_type'), 'source': source}
    if source == 'edit_generation':
        return {'kind': row.get('proposal_lane') or 'edit_generation_lane_unknown',
                'policy': row.get('edit_policy'), 'source': source}
    if source == 'fresh_baseline_proposal':
        return {'kind': row.get('reference_stratum') or 'fresh_baseline_proposal',
                'policy': None, 'source': source}
    if source == 'existing_native_reference':
        return {'kind': 'existing_native_reference', 'policy': None, 'source': source}
    return {'kind': source or 'source_unknown', 'policy': None, 'source': source or None,
            'unknown': 'source_unknown'}


def bank_index(bank_rows):
    """Identity-checked sequence_hash -> bank row lookup table."""
    if not isinstance(bank_rows, list):
        raise ValueError('invalid_bank_rows')
    index = {}
    for row in bank_rows:
        if not isinstance(row, dict) or not isinstance(row.get('sequence'), str):
            raise ValueError('invalid_bank_row')
        digest = row.get('sequence_hash')
        if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or rules.sequence_hash(row['sequence']) != digest):
            raise ValueError('invalid_bank_row_identity')
        index[digest] = row
    return index


def parent_origin(row, bank_index):
    """Resolve a selected sequence's parent provenance from the qualified bank.

    Reuses the feed's ``public_observations`` (owner epoch/coldkey/uid). Missing
    provenance is exported as an explicit unknown, never invented.
    """
    source = row.get('source')
    if source == 'segment_recombination':
        parts = row.get('segment_provenance', [])
        hashes = sorted({p['source_hash'] for p in parts if isinstance(p, dict) and p.get('source_hash')})
        contributors = []
        for sha in hashes:
            origin = parent_origin({'reference_hash': sha}, bank_index)
            origin['kind'] = 'segment_donor'
            origin['copied_segments'] = [p for p in parts if p.get('source_hash') == sha]
            contributors.append(origin)
        return {'resolved': len(contributors) >= 2 and all(c['resolved'] for c in contributors),
                'kind': 'multi_donor_segments', 'parent_hash': None,
                'contributors': contributors, 'segment_layout': row.get('segment_layout')}
    if source == 'leaderboard_graft':
        parts = row.get('segment_provenance', [])
        hashes = sorted({p['source_hash'] for p in parts
                         if isinstance(p, dict) and p.get('source_hash')})
        contributors = []
        for sha in hashes:
            origin = parent_origin({'reference_hash': sha}, bank_index)
            origin['kind'] = 'graft_donor'
            origin['copied_segments'] = [p for p in parts if p.get('source_hash') == sha]
            contributors.append(origin)
        return {'resolved': len(contributors) >= 1 and all(c['resolved'] for c in contributors),
                'kind': 'multi_donor_segments', 'parent_hash': None,
                'contributors': contributors, 'segment_layout': row.get('segment_layout'),
                'graft_mode': row.get('graft_mode')}
    parent_sequence = row.get('parent_sequence')
    if isinstance(parent_sequence, str) and parent_sequence:
        parent_hash = rules.sequence_hash(parent_sequence)
        kind = 'edit_parent'
    elif isinstance(row.get('reference_hash'), str) and row['reference_hash']:
        parent_hash = row['reference_hash']
        kind = 'baseline_reference'
    elif source == 'existing_native_reference' and isinstance(row.get('sequence'), str):
        parent_hash = rules.sequence_hash(row['sequence'])
        kind = 'self_reference'
    else:
        return {'resolved': False, 'parent_hash': None, 'kind': None,
                'unknown_reason': 'no_parent_identity'}
    bank_row = None if bank_index is None else bank_index.get(parent_hash)
    origin = {'resolved': bank_row is not None, 'parent_hash': parent_hash, 'kind': kind}
    if isinstance(parent_sequence, str) and parent_sequence:
        origin['parent_sequence'] = parent_sequence
    if bank_row is None:
        origin['unknown_reason'] = ('parent_not_in_bank' if bank_index is not None
                                    else 'parent_bank_unavailable')
        return origin
    observations = [o for o in bank_row.get('public_observations', []) if isinstance(o, dict)]
    epochs = sorted({o['epoch'] for o in observations if type(o.get('epoch')) is int})
    coldkeys = sorted({o['coldkey'] for o in observations
                       if isinstance(o.get('coldkey'), str) and o['coldkey']})
    uids = sorted({o['uid'] for o in observations if type(o.get('uid')) is int})
    cells = list(bank_row.get('cells', []))
    origin.update(parent_epochs=epochs, parent_coldkeys=coldkeys, parent_uids=uids,
                  parent_cells=cells)
    unknowns = []
    if not epochs:
        unknowns.append('parent_epoch_unknown')
    if not coldkeys:
        unknowns.append('parent_coldkey_unknown')
    if unknowns:
        origin['unknown_reason'] = ','.join(unknowns)
    return origin


def selected_lineage(row, bank_index, challenge):
    """Explicit lineage for one selected sequence plus its honest unknowns."""
    if not isinstance(row, dict):
        row = {}
    sequence = row.get('sequence')
    sequence_hash = (rules.sequence_hash(sequence) if isinstance(sequence, str) and sequence
                     else row.get('sequence_hash'))
    value = {'sequence': sequence, 'sequence_hash': sequence_hash,
             'cell': cell_of(row, challenge), 'edit_kind': edit_kind(row),
             'parent_origin': parent_origin(row, bank_index)}
    unknowns = []
    if value['cell'].get('cell') is None:
        unknowns.append('cell_unknown')
    if value['edit_kind'].get('unknown'):
        unknowns.append('edit_kind_unknown')
    if value['parent_origin'].get('unknown_reason'):
        unknowns.append(value['parent_origin']['unknown_reason'])
    value['unknowns'] = unknowns
    return value


def read_json(state_root, *parts):
    try:
        return json.loads((Path(state_root).joinpath(*parts)).read_text())
    except (OSError, ValueError):
        return None


def read_receipt(state_root, epoch):
    return read_json(state_root, str(epoch), 'receipt.json')


def read_epoch_state(state_root, epoch):
    return read_json(state_root, str(epoch), 'continuation.json')


def completed_epochs(state_root, before_epoch, count=DEFAULT_EPOCHS):
    """The ``count`` strictly-past chain epochs, never skipping to older ones.

    An epoch is finished once the chain clock passes ``before_epoch``, not when
    its last attempt happened to be terminal. A later ``recoverable_failure``
    attempt must not erase an earlier publication from this window, and missing
    epochs are reported explicitly rather than silently replaced with older
    epochs. The window is clamped at ``rules.ACTIVATION_EPOCH``.
    """
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    start = max(rules.ACTIVATION_EPOCH, before_epoch - count)
    return list(range(start, before_epoch))


def epoch_evidence(state_root, epochs):
    """Per-window-epoch recorded evidence, so missing/failed is never silent."""
    evidence = {}
    for epoch in epochs:
        receipt = read_receipt(state_root, epoch)
        state = read_epoch_state(state_root, epoch)
        published = published_members(state_root, epoch)
        evidence[str(epoch)] = {
            'receipt_status': receipt.get('status') if isinstance(receipt, dict) else None,
            'state_recorded': state is not None,
            'published_recorded': published is not None,
            'published_count': len(published) if published is not None else None}
    return evidence


def historical_challenge(state, epoch):
    """The epoch's own recorded challenge, or ``None`` when unverifiable."""
    challenge = state.get('challenge') if isinstance(state, dict) else None
    if (isinstance(challenge, dict) and challenge.get('epoch') == epoch
            and isinstance(challenge.get('target'), str) and challenge['target']):
        return challenge
    return None


def scoring_protocol(identity, target):
    """Comparable scoring identity, or ``(None, unknown_reason)`` when unusable."""
    if not isinstance(identity, dict):
        return None, 'scoring_identity_incomplete'
    for key in SCORING_IDENTITY_FIELDS:
        if not isinstance(identity.get(key), str) or not identity[key]:
            return None, 'scoring_identity_incomplete'
    if not isinstance(target, str) or not target or identity['target'] != target:
        return None, 'scoring_identity_target_mismatch'
    return rules.digest(identity), None


def published_members(state_root, epoch):
    """Sequence hashes published this epoch, or ``None`` when unrecorded."""
    record = read_json(state_root, str(epoch), 'published-quality.json')
    if not isinstance(record, dict):
        return None
    members = record.get('quality', {}).get('members')
    if not isinstance(members, list):
        return None
    return {m['sequence_hash'] for m in members
            if isinstance(m, dict) and isinstance(m.get('sequence_hash'), str)}


def scored_outcomes(state_root, epochs, bank_index, metric_field):
    """One outcome record per scored sequence, using each epoch's own challenge."""
    outcomes = []
    for epoch in epochs:
        state = read_epoch_state(state_root, epoch)
        if not isinstance(state, dict):
            continue
        epoch_challenge = historical_challenge(state, epoch)
        target = epoch_challenge['target'] if epoch_challenge is not None else None
        protocol, protocol_unknown = scoring_protocol(state.get('scoring_identity'), target)
        score_rows = state.get('score_rows', {})
        if not isinstance(score_rows, dict):
            score_rows = {}
        lineage_rows = {}
        for row in state.get('pending_native', []) + state.get('native_rows', []):
            if isinstance(row, dict) and isinstance(row.get('sequence'), str):
                lineage_rows[row['sequence']] = row
        published = published_members(state_root, epoch)
        for sequence, per_target in score_rows.items():
            metrics = (per_target or {}).get(target) if isinstance(per_target, dict) else None
            value = metrics.get(metric_field) if isinstance(metrics, dict) else None
            row = lineage_rows.get(sequence)
            if row is not None:
                lineage = selected_lineage(row, bank_index, epoch_challenge or {})
            else:
                lineage = {'sequence': sequence,
                           'sequence_hash': rules.sequence_hash(sequence),
                           'cell': cell_of({}, epoch_challenge or {}),
                           'edit_kind': edit_kind({}),
                           'parent_origin': {'resolved': False, 'parent_hash': None,
                                             'unknown_reason': 'lineage_row_not_in_state'},
                           'unknowns': ['lineage_row_not_in_state']}
            unknowns = list(lineage.get('unknowns', []))
            if epoch_challenge is None:
                unknowns.append('historical_challenge_unknown')
            if value is None or not _finite(value):
                unknowns.append('metric_not_measured')
            if protocol_unknown:
                unknowns.append(protocol_unknown)
            sequence_hash = lineage['sequence_hash']
            published_flag = (sequence_hash in published if published is not None else None)
            if published is None:
                unknowns.append('published_unknown')
            outcomes.append({'epoch': epoch, 'target': target, 'sequence': sequence,
                             'sequence_hash': sequence_hash,
                             'protocol': protocol, 'scoring_identity': state.get('scoring_identity'),
                             'measured': {metric_field: value} if _finite(value) else None,
                             'published': published_flag,
                             # On-chain result is never read or inferred here.
                             'validator_outcome': None,
                             'lineage': lineage, 'unknowns': unknowns})
    return outcomes


def _protocol_summary(outcomes):
    groups = {}
    for outcome in outcomes:
        protocol = outcome.get('protocol')
        if protocol is None:
            continue
        entry = groups.setdefault(protocol, {'protocol': protocol, 'epochs': [],
                                             'count': 0,
                                             'scoring_identity': outcome.get('scoring_identity')})
        entry['count'] += 1
        if outcome['epoch'] not in entry['epochs']:
            entry['epochs'].append(outcome['epoch'])
    return sorted(groups.values(), key=lambda entry: max(entry['epochs']), reverse=True)


def top_bottom5(outcomes, *, metric_field, direction, count=5):
    """Best/worst ``count`` measured outcomes under a single honest protocol.

    Outcomes from a different scoring identity (or without a finite measurement)
    are never mixed in; they are returned as ``non_comparable``.
    """
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    comparable = [o for o in outcomes if o.get('measured') and o.get('protocol')]
    by_protocol = {}
    for outcome in comparable:
        by_protocol.setdefault(outcome['protocol'], []).append(outcome)
    primary = (max(by_protocol, key=lambda p: max(o['epoch'] for o in by_protocol[p]))
               if by_protocol else None)
    pool = sorted(by_protocol.get(primary, []),
                  key=lambda o: o['measured'][metric_field], reverse=(direction == 'max'))
    non_comparable = [o for o in outcomes
                      if o.get('measured') is None or o.get('protocol') is None
                      or o['protocol'] != primary]
    bottom = list(reversed(pool[-count:])) if len(pool) > count else []
    return {'top': pool[:count], 'bottom': bottom, 'primary_protocol': primary,
            'comparable_count': len(pool), 'non_comparable': non_comparable}


def populations(outcomes):
    published = sum(1 for o in outcomes if o.get('published') is True)
    published_unknown = sum(1 for o in outcomes if o.get('published') is None)
    return {'locally_scored': len(outcomes), 'published': published,
            'published_unknown': published_unknown,
            'validator_scored': 0}


def build(state_root, bank_rows, challenge, *, metric_field=DEFAULT_METRIC_FIELD,
          direction=None, count=DEFAULT_EPOCHS):
    """Durable lineage/outcome record over the last ``count`` completed epochs."""
    directions = metric_directions()
    if metric_field not in directions:
        raise ValueError('unknown_metric_field')
    expected = directions[metric_field]
    if direction is None:
        direction = expected
    if direction != expected:
        raise ValueError('metric_direction_mismatch')
    index = bank_index(bank_rows) if bank_rows is not None else None
    epochs = completed_epochs(state_root, challenge['epoch'], count)
    outcomes = scored_outcomes(state_root, epochs, index, metric_field)
    ranking = top_bottom5(outcomes, metric_field=metric_field, direction=direction, count=5)
    return {'schema': SCHEMA, 'target': challenge['target'],
            'before_epoch': challenge['epoch'], 'metric_field': metric_field,
            'metric_direction': direction, 'metric_source': 'our_scorer',
            'validator_outcome_source': 'not_available_locally',
            'completed_epochs': epochs, 'epoch_evidence': epoch_evidence(state_root, epochs),
            'outcome_count': len(outcomes),
            'populations': populations(outcomes),
            'protocols': _protocol_summary(outcomes),
            'top': ranking['top'], 'bottom': ranking['bottom'],
            'comparable_count': ranking['comparable_count'],
            'non_comparable': ranking['non_comparable'],
            'limitations': ['our_scorer metric, not the leaderboard final_score',
                            'top/bottom are compared only within one scoring identity',
                            'validator/on-chain outcomes are not read locally and are never inferred']}


def load_bank_rows(config, challenge):
    """Read the locally-qualified parent bank rows, or ``None`` when unavailable."""
    from nanobody_opt import pr100_parent_bank as bank
    try:
        cfg = bank.settings(config)
        if cfg is None:
            return None
        root = Path(cfg['state_directory']) / challenge['target']
        pointer = json.loads(bank.read(root / 'current.json'))
        if (not isinstance(pointer, dict) or not bank.hex64(pointer.get('bank_sha256'))
                or pointer.get('target') != challenge['target']):
            return None
        path = root / ('bank-' + pointer['bank_sha256'] + '.json')
        raw = bank.read(path)
        if bank.sha(raw) != pointer['bank_sha256']:
            return None
        value = json.loads(raw)
    except Exception:
        return None
    rows = value.get('rows') if isinstance(value, dict) else None
    return rows if isinstance(rows, list) else None


def record(state_root, challenge, config, *, metric_field=DEFAULT_METRIC_FIELD,
           count=DEFAULT_EPOCHS):
    """Controller wiring: build and persist the durable analysis, never raising.

    Optional analytics must not abort publication or the next epoch, so every
    failure path (bank read, parse, build, disk write) returns a status instead
    of raising.
    """
    root = Path(state_root)
    try:
        bank_rows = load_bank_rows(config, challenge)
        value = build(root, bank_rows, challenge, metric_field=metric_field, count=count)
        value.update(status='complete', generated_at_unix=time.time())
        pr100_filter.atomic(root / 'lineage-outcomes.json', value)
        return {'status': 'complete', 'outcome_count': value.get('outcome_count'),
                'completed_epochs': value.get('completed_epochs', []),
                'top_count': len(value.get('top', [])),
                'bottom_count': len(value.get('bottom', []))}
    except Exception as error:
        unavailable = {'schema': SCHEMA, 'status': 'unavailable',
                       'target': challenge.get('target'), 'before_epoch': challenge.get('epoch'),
                       'error_type': type(error).__name__, 'generated_at_unix': time.time()}
        try:
            pr100_filter.atomic(root / 'lineage-outcomes.json', unavailable)
        except Exception:
            pass  # The analytics disk is optional; the epoch must proceed regardless.
        return {'status': 'unavailable', 'outcome_count': 0, 'completed_epochs': []}
