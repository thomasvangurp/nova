"""Bounded, score-free donor choice for an explicitly enabled missing cell.

The input is the controller's frozen native-pass reference bank. A donor is
only a proposal input, never current-cell validation or binding evidence.

Two v2 parent gates are wired here (v2 port #3):

* ``cdr_developability_in_range`` (from ``pr100_parent_gates``) is evaluated
  for every donor from its measured developability profile and attached to the
  donor descriptor as ``cdr_developability_in_range``. It is a *deprioritising*
  gate, never a silent drop: an out-of-range donor sorts after every in-range
  donor at the same stratum/length-distance, so the ``slots`` count is
  unchanged. A donor whose profile is missing or malformed fails closed
  (out-of-range), matching the v2 selector.

* ``same_cdr3_family`` powers an *opt-in* per-clonotype ceiling
  (``family_cap``), disabled by default. It is off by default because the
  synthetic native-pass bank's CDR3s differ by one residue and collapse to a
  single family under the 0.15 threshold, so a hard dedup would regress
  selection-count behaviour. Callers opt in with ``family_cap`` and a
  ``cdr3_of`` extractor.
"""
from __future__ import annotations

from collections import Counter
import re

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


def select(bank, challenge, slots, *, fallback=False, family_cap=None, cdr3_of=None):
    """Keep the old exact-cell order; fill missing cells from ranked donors.

    Same V takes precedence over length distance, then another V is allowed.
    The epoch/hash tie break is deterministic and ignores all score fields.
    The existing reference-slot bound also bounds donor diversity.

    ``family_cap`` optionally bounds the per-clonotype donor count (opt-in).
    When set it must be a positive integer and ``cdr3_of`` must be a callable
    that extracts the CDR3 string from a donor row.
    """
    if family_cap is not None:
        if isinstance(family_cap, bool) or not isinstance(family_cap, int) or family_cap < 1:
            raise ValueError('family_cap must be a positive integer when set')
        if not callable(cdr3_of):
            raise ValueError('cdr3_of must be a callable row -> CDR3 string when family_cap is set')
    gene, length = challenge['allowed_v'], challenge['allowed_cdr3_length']
    cell = f'{gene}:{length}'
    exact = sorted((r for r in bank if cell in r['cells']),
                   key=lambda r: rules.digest([challenge['epoch'], r['sequence_hash']]))[:slots]
    evidence = {'requested_cell': cell, 'bank_rows': len(bank),
                'bank_cells': dict(sorted(Counter(c for r in bank for c in r['cells']).items())),
                'exact_cell_available': sum(cell in r['cells'] for r in bank),
                'exact_selected': len(exact), 'fallback_enabled': fallback,
                'fallback_used': False, 'available_by_stratum': {},
                'selected_by_stratum': {}, 'selected_donors': [],
                'developability_in_range_selected': 0,
                'developability_out_of_range_selected': 0}
    if exact or not fallback:
        chosen_exact = [(r, {'reference_stratum': 'exact_cell', 'reference_cell': cell,
                             'reference_length_distance': 0,
                             'cdr_developability_in_range':
                                 _donor_cdr_developability_in_range(r)}) for r in exact]
        evidence.update(
            developability_in_range_selected=sum(d['cdr_developability_in_range']
                                                 for _, d in chosen_exact),
            developability_out_of_range_selected=sum(not d['cdr_developability_in_range']
                                                     for _, d in chosen_exact))
        return exact, chosen_exact, evidence
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
        in_range = _donor_cdr_developability_in_range(row)
        descriptor = {'reference_stratum': stratum, 'reference_cell': source_cell,
                      'reference_length_distance': distance,
                      'cdr_developability_in_range': in_range}
        # Developability deprioritises (fail-closed), never drops: in-range
        # donors sort ahead of out-of-range donors at equal stratum/distance,
        # and the total selection count is unchanged.
        rank = (other_v, distance, not in_range, rules.digest([challenge['epoch'], digest]))
        # A repeated bank identity is one donor. Its metadata must agree.
        if digest in found and found[digest][2] != descriptor:
            raise ValueError('conflicting native-pass donor cells')
        found[digest] = (rank, row, descriptor)
    ranked = sorted(found.values(), key=lambda item: item[0])
    available_by_stratum = dict(Counter(d['reference_stratum'] for _, _, d in ranked))
    selected = (_cap_families(ranked, family_cap=family_cap, cdr3_of=cdr3_of)
                if family_cap is not None else ranked)
    chosen = [(row, descriptor) for _, row, descriptor in selected[:slots]]
    evidence.update(fallback_used=bool(chosen),
                    available_by_stratum=available_by_stratum,
                    selected_by_stratum=dict(Counter(d['reference_stratum'] for _, d in chosen)),
                    selected_donors=[{'reference_hash': r['sequence_hash'], **d} for r, d in chosen],
                    developability_in_range_selected=sum(d['cdr_developability_in_range']
                                                         for _, d in chosen),
                    developability_out_of_range_selected=sum(not d['cdr_developability_in_range']
                                                             for _, d in chosen))
    if family_cap is not None:
        evidence.update(family_cap=family_cap,
                        family_cap_families_selected=parent_gates.family_count(
                            [r for r, _ in chosen], cdr3_of=cdr3_of),
                        family_cap_trimmed=len(ranked) != len(selected))
    return exact, chosen, evidence
