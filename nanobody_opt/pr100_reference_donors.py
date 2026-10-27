"""Bounded, score-free donor choice for an explicitly enabled missing cell.

The input is the controller's frozen native-pass reference bank. A donor is
only a proposal input, never current-cell validation or binding evidence.
"""
from __future__ import annotations

from collections import Counter
import re

from nanobody_opt import rules
from nanobody_opt.pr100_proposal_pilot import GENES

REFERENCE_STRATA = frozenset({'exact_cell', 'same_v_nearby_length',
                             'other_v_nearby_length', 'unseeded_control'})


def enabled(job):
    value = job.get('missing_cell_donor_fallback', False)
    if type(value) is not bool:
        raise ValueError('missing_cell_donor_fallback must be a boolean')
    return value


def select(bank, challenge, slots, *, fallback=False):
    """Keep the old exact-cell order; fill missing cells from ranked donors.

    Same V takes precedence over length distance, then another V is allowed.
    The epoch/hash tie break is deterministic and ignores all score fields.
    The existing reference-slot bound also bounds donor diversity.
    """
    gene, length = challenge['allowed_v'], challenge['allowed_cdr3_length']
    cell = f'{gene}:{length}'
    exact = sorted((r for r in bank if cell in r['cells']),
                   key=lambda r: rules.digest([challenge['epoch'], r['sequence_hash']]))[:slots]
    evidence = {'requested_cell': cell, 'bank_rows': len(bank),
                'bank_cells': dict(sorted(Counter(c for r in bank for c in r['cells']).items())),
                'exact_cell_available': sum(cell in r['cells'] for r in bank),
                'exact_selected': len(exact), 'fallback_enabled': fallback,
                'fallback_used': False, 'available_by_stratum': {},
                'selected_by_stratum': {}, 'selected_donors': []}
    if exact or not fallback:
        return exact, [(r, {'reference_stratum': 'exact_cell', 'reference_cell': cell,
                           'reference_length_distance': 0}) for r in exact], evidence
    found = {}
    for row in bank:
        digest = row.get('sequence_hash')
        if (not isinstance(row.get('sequence'), str) or not isinstance(digest, str)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or rules.sequence_hash(row['sequence']) != digest):
            raise ValueError('invalid native-pass donor identity')
        cells = []
        for value in row['cells']:
            match = re.fullmatch(r'(IGHV3-\d+):(\d+)', value) if isinstance(value, str) else None
            if not match or match[1] not in GENES or not 5 <= int(match[2]) <= 23:
                raise ValueError('invalid native-pass donor cell')
            cells.append((match[1] != gene, abs(int(match[2]) - length), value))
        if not cells:
            raise ValueError('native-pass donor has no cell')
        other_v, distance, source_cell = min(cells)
        stratum = 'other_v_nearby_length' if other_v else 'same_v_nearby_length'
        descriptor = {'reference_stratum': stratum, 'reference_cell': source_cell,
                      'reference_length_distance': distance}
        rank = (other_v, distance, rules.digest([challenge['epoch'], digest]))
        # A repeated bank identity is one donor. Its metadata must agree.
        if digest in found and found[digest][2] != descriptor:
            raise ValueError('conflicting native-pass donor cells')
        found[digest] = (rank, row, descriptor)
    ordered = sorted(found.values(), key=lambda item: item[0])
    chosen = [(row, descriptor) for _, row, descriptor in ordered[:slots]]
    evidence.update(fallback_used=bool(chosen),
                    available_by_stratum=dict(Counter(d['reference_stratum'] for _, _, d in ordered)),
                    selected_by_stratum=dict(Counter(d['reference_stratum'] for _, d in chosen)),
                    selected_donors=[{'reference_hash': r['sequence_hash'], **d} for r, d in chosen])
    return exact, chosen, evidence
