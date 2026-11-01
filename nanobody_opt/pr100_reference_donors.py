"""Select compatible proposal parents by measured merit, preserving uncertainty.

Parent scores guide search only: children still need current annotation, native
and novelty evidence. Raw rank-sums from different target fields are never mixed.
"""
from __future__ import annotations

from collections import Counter
import re
import math

from nanobody_opt import rules
from nanobody_opt import pr100_parent_gates as parent_gates
from nanobody_opt.pr100_proposal_pilot import GENES

REFERENCE_STRATA = frozenset({'exact_cell', 'same_v_nearby_length',
                             'other_v_nearby_length', 'unseeded_control'})


def enabled(job):
    value = job.get('missing_cell_donor_fallback', False)
    if type(value) is not bool:
        raise ValueError('missing_cell_donor_fallback must be a boolean')
    return value


def _donor_cdr_developability_in_range(row):
    """Fail-closed CDR-only red-region flag for one native-pass donor row.

    The native-pass bank carries the measured developability profile under
    ``developability`` (or ``pr100_evidence.developability``). A row without a
    usable profile fails closed to out-of-range, so an unmeasured donor cannot
    silently out-rank a measured in-range donor at the same stratum/distance.
    """
    profile = row.get('developability') if isinstance(row, dict) else None
    if not isinstance(profile, dict):
        evidence = row.get('pr100_evidence')
        profile = evidence.get('developability') if isinstance(evidence, dict) else None
    return parent_gates.cdr_developability_in_range(profile)


def _cap_families(ordered, *, family_cap, cdr3_of):
    """Keep at most ``family_cap`` donors per first-seen CDR3 family.

    ``ordered`` is the already-ranked donor list. Distinct families are never
    merged; within a family the ranked order is preserved. The returned list is
    the bounded selection pool (it may be shorter than its input).
    """
    capped = []
    counts = {}
    for rank, row, descriptor in ordered:
        cdr3 = (cdr3_of(row) or '').strip()
        key = next((k for k in counts if parent_gates.same_cdr3_family(k, cdr3)), cdr3)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= family_cap:
            capped.append((rank, row, descriptor))
    return capped


def developability_status(row):
    profile = row.get('developability')
    if not isinstance(profile, dict):
        evidence = row.get('pr100_evidence')
        profile = evidence.get('developability') if isinstance(evidence, dict) else None
    if isinstance(profile, dict):
        try:
            values = [profile[key] for key in parent_gates.CDR_DEVELOPABILITY_RED_REGION]
            if all(type(value) in (float, int) and math.isfinite(value) for value in values):
                return ('in_range' if parent_gates.cdr_developability_in_range(profile)
                        else 'out_of_range')
        except (KeyError, TypeError):
            pass
    # No usable TNP developability profile (the parent-feed is annotation-only):
    # fall back to the leaderboard developability_rank, a PSH proxy. Lower rank
    # = better developability (lower PSH). Treat the top half of the epoch
    # field as in-range.
    for value in row.get('quality_evidence', []):
        if not isinstance(value, dict):
            continue
        metrics = value.get('metrics')
        rank = metrics.get('developability_rank') if isinstance(metrics, dict) else None
        field_size = value.get('field_size')
        if (type(rank) in (int, float) and math.isfinite(rank)
                and type(field_size) is int and field_size > 1):
            return 'in_range' if rank <= field_size / 2 else 'out_of_range'
    return 'unknown'


def merit(row, challenge):
    """Use comparable field-normalized public evidence, never unlabeled scores.

    Single-validator percentiles are a search prior, explicitly not multipass
    qualification. A matched-field replay mean has precedence only when the
    challenge explicitly identifies that replay field and protocol.
    """
    accepted = []
    values = row.get('quality_evidence', [])
    if not isinstance(values, list):
        return (2, 0.0, 0), 'unknown'
    for value in values:
        if (not isinstance(value, dict) or value.get('target') != challenge['target']
                or type(value.get('epoch')) is not int or value['epoch'] >= challenge['epoch']
                or type(value.get('value')) not in (float, int)
                or not math.isfinite(value['value'])):
            continue
        if (value.get('kind') == 'matched_multipass_mean'
                and type(value.get('passes')) is int and value['passes'] >= 10
                and challenge.get('merit_field_sha256')
                and value.get('field_sha256') == challenge['merit_field_sha256']
                and value.get('protocol') == challenge.get('merit_protocol')
                and value.get('protocol')):
            accepted.append((0, value))
        elif (value.get('kind') == 'public_validator_percentile'
                and 0 <= value['value'] <= 1
                and type(value.get('field_size')) is int and value['field_size'] > 1
                and value.get('passes') == 1):
            accepted.append((1, value))
    if not accepted:
        return (2, 0.0, 0), 'unknown'
    best_class = min(kind for kind, _ in accepted)
    evidence = max((v for kind, v in accepted if kind == best_class), key=lambda v: v['epoch'])
    return (best_class, evidence['value'], -evidence['epoch']), evidence['kind']


def select(bank, challenge, slots, *, fallback=False, family_cap=None, cdr3_of=None):
    """Rank exact-cell parents or, when enabled, compatible missing-cell donors.

    Compatibility comes first; at equal compatibility measured in-range CDRs
    precede unknown and known-failing CDRs, then comparable merit breaks ties.
    All identity and cell checks apply to exact-cell parents as well as fallback.
    """
    if type(slots) is not int or slots < 1:
        raise ValueError('slots must be a positive integer')
    if family_cap is not None:
        if type(family_cap) is not int or family_cap < 1:
            raise ValueError('family_cap must be a positive integer when set')
        if not callable(cdr3_of):
            raise ValueError('cdr3_of must be a callable row -> CDR3 string when family_cap is set')
    gene, length = challenge['allowed_v'], challenge['allowed_cdr3_length']
    cell = f'{gene}:{length}'
    found = {}
    for row in bank:
        digest = row.get('sequence_hash')
        if (not isinstance(row.get('sequence'), str) or not isinstance(digest, str)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or rules.sequence_hash(row['sequence']) != digest):
            raise ValueError('invalid native-pass donor identity')
        cells = []
        for value in row.get('cells', []):
            match = re.fullmatch(r'(IGHV3-\d+):(\d+)', value) if isinstance(value, str) else None
            if not match or match[1] not in GENES or not 5 <= int(match[2]) <= 23:
                raise ValueError('invalid native-pass donor cell')
            cells.append((match[1] != gene, abs(int(match[2]) - length), value))
        if not cells:
            raise ValueError('native-pass donor has no cell')
        other_v, distance, source_cell = min(cells)
        stratum = ('exact_cell' if source_cell == cell else
                   'other_v_nearby_length' if other_v else 'same_v_nearby_length')
        status = developability_status(row)
        merit_rank, merit_kind = merit(row, challenge)
        descriptor = {'reference_stratum': stratum, 'reference_cell': source_cell,
                      'reference_length_distance': distance,
                      'cdr_developability_in_range': status == 'in_range',
                      'cdr_developability_status': status, 'merit_kind': merit_kind}
        rank = (other_v, distance, {'in_range': 0, 'unknown': 1, 'out_of_range': 2}[status],
                *merit_rank, rules.digest([challenge['epoch'], digest]))
        if digest in found and found[digest] != (rank, row, descriptor):
            raise ValueError('conflicting native-pass donor cells or evidence')
        found[digest] = (rank, row, descriptor)
    ranked = sorted(found.values(), key=lambda item: item[0])
    exact_pool = [item for item in ranked if item[2]['reference_stratum'] == 'exact_cell']
    pool = exact_pool if exact_pool else ranked if fallback else []
    selected = (_cap_families(pool, family_cap=family_cap, cdr3_of=cdr3_of)
                if family_cap is not None else pool)
    chosen = [(row, descriptor) for _, row, descriptor in selected[:slots]]
    exact = [row for row, descriptor in chosen if descriptor['reference_stratum'] == 'exact_cell']
    status_counts = Counter(d['cdr_developability_status'] for _, d in chosen)
    evidence = {'requested_cell': cell, 'bank_rows': len(bank),
                'bank_cells': dict(sorted(Counter(c for r in bank for c in r['cells']).items())),
                'exact_cell_available': len(exact_pool), 'exact_selected': len(exact),
                'fallback_enabled': fallback, 'fallback_used': bool(chosen and not exact_pool),
                'available_by_stratum': dict(Counter(d['reference_stratum'] for _, _, d in pool)),
                'selected_by_stratum': dict(Counter(d['reference_stratum'] for _, d in chosen)),
                'selected_donors': [{'reference_hash': r['sequence_hash'], **d} for r, d in chosen],
                'developability_in_range_selected': status_counts['in_range'],
                'developability_out_of_range_selected': status_counts['out_of_range'],
                'developability_unknown_selected': status_counts['unknown']}
    if family_cap is not None:
        evidence.update(family_cap=family_cap,
                        family_cap_families_selected=parent_gates.family_count(
                            [r for r, _ in chosen], cdr3_of=cdr3_of),
                        family_cap_trimmed=len(pool) != len(selected))
    return exact, chosen, evidence
