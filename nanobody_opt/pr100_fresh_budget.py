"""Durable shared admission capacity, separate from completions and confirmations."""
from __future__ import annotations
from nanobody_opt import rules

DEFAULT_LIMIT = 4800
SCHEMA = 'pr100-fresh-native-budget-v1'


def validate_limit(value, *, allow_zero=False):
    if type(value) is not int or not (0 if allow_zero else 1) <= value <= DEFAULT_LIMIT:
        raise ValueError('fresh native limit must be an integer within the 4800-candidate ceiling')
    return value


def ensure(state, config):
    limit = validate_limit(config.get('epoch_fresh_native_limit', DEFAULT_LIMIT))
    budget = state.get('fresh_native_budget')
    if budget is None:
        # Legacy hashes include interrupted calls. Never invent a refund.
        legacy = set(state.get('native_attempted_hashes', []))
        legacy.update(rules.sequence_hash(r['sequence']) for r in
                      state.get('native_rows', []) + state.get('pending_confirmation', []))
        budget = {'schema': SCHEMA, 'limit': limit, 'admitted_hashes': [],
                  'legacy_charged_hashes': sorted(legacy), 'reservations': {}}
        state['fresh_native_budget'] = budget
    if budget.get('schema') != SCHEMA:
        raise ValueError('invalid fresh native budget schema')
    budget['limit'] = min(validate_limit(budget['limit']), limit)
    return budget


def charged(budget):
    value = set(budget['admitted_hashes']) | set(budget['legacy_charged_hashes'])
    for reservation in budget['reservations'].values():
        value.update(reservation['held_hashes'])
    return value


def summary(budget):
    admitted = set(budget['admitted_hashes'])
    legacy = set(budget['legacy_charged_hashes']) - admitted
    used = charged(budget)
    return {'limit': budget['limit'], 'admitted_unique': len(admitted),
            'legacy_charged_unique': len(legacy), 'reserved_uncertain_unique': len(used - admitted - legacy),
            'capacity_used': len(used), 'remaining': max(0, budget['limit'] - len(used)),
            'legacy_uncertainty': bool(legacy)}


def reserve(budget, attempt, rows):
    """Persist this result before remote start; recovery reuses its exact plan."""
    key = str(attempt)
    indexed = {rules.sequence_hash(r['sequence']): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('duplicate fresh admission input')
    existing = budget['reservations'].get(key)
    if existing is not None:
        if any(h not in indexed for h in existing['input_hashes']):
            raise ValueError('saved fresh admission plan input changed')
        return [indexed[h] for h in existing['input_hashes']]
    used = charged(budget); remaining = max(0, budget['limit'] - len(used))
    selected, fresh = [], []
    for digest, row in indexed.items():
        if row.get('native_resume') is not None:
            selected.append(digest)
        elif digest not in used and len(fresh) < remaining:
            selected.append(digest); fresh.append(digest)
    budget['reservations'][key] = {'input_hashes': selected, 'fresh_hashes': fresh,
                                   'held_hashes': list(fresh)}
    return [indexed[h] for h in selected]


def reconcile(budget, attempt, result, *, cleanup_verified):
    plan = budget['reservations'].get(str(attempt))
    if plan is None:
        return
    fresh = set(plan['fresh_hashes'])
    explicit = 'fresh_admitted_sequence_hashes' in result
    if explicit and set(result['fresh_admitted_sequence_hashes']) - fresh:
        raise ValueError('fresh admission outside persisted epoch reservation')
    admitted = set(result.get('fresh_admitted_sequence_hashes',
                               result.get('attempted_sequence_hashes', []))) & fresh
    target = 'admitted_hashes' if explicit else 'legacy_charged_hashes'
    budget[target] = sorted(set(budget[target]) | admitted)
    uncertain = set(result.get('fresh_admission_uncertain_sequence_hashes', [])) & fresh
    budget['legacy_charged_hashes'] = sorted(set(budget['legacy_charged_hashes']) | uncertain)
    # Dispatcher supplies only cleanup-verified per-host non-start proofs.
    never = set(result.get('verified_fresh_never_started_sequence_hashes', [])) & fresh
    if cleanup_verified:
        never.update(result.get('fresh_never_started_sequence_hashes', []))
        never.update(rules.sequence_hash(r['sequence']) for r in result.get('deferred_candidates', [])
                     if r.get('native_resume') is None)
        never &= fresh
    if never & (admitted | set(budget['admitted_hashes']) | set(budget['legacy_charged_hashes'])):
        raise ValueError('fresh admission and never-started evidence conflict')
    plan['held_hashes'] = sorted(set(plan['held_hashes']) - admitted - uncertain - never)


def planned_fresh_count(budget, attempt):
    return len(budget['reservations'][str(attempt)]['fresh_hashes'])


def reserve_saved_input(budget, attempt, rows):
    selected = reserve(budget, attempt, rows)
    if [r['sequence'] for r in selected] != [r['sequence'] for r in rows]:
        # An old saved payload can describe a previously launched job. It
        # cannot be truncated/relaunched or silently refunded during migration.
        unknown = {rules.sequence_hash(r['sequence']) for r in rows if r.get('native_resume') is None}
        budget['legacy_charged_hashes'] = sorted(set(budget['legacy_charged_hashes']) | unknown)
        raise ValueError('saved native input exceeds or conflicts with fresh reservation')
