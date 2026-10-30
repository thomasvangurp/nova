"""Bounded, proposal-only multi-donor recombination on verified IMGT numbering.

Whole-region and linked-region searches share a CDR3 splice library. Its only
internal cuts are the numbered central junctions 109/110, 110/111, 111/112 and
112/113; raw sequence offsets are never crossover decisions. Every emitted
residue has an exact donor interval. Ranking is an explicit search prior, not
a measured fitness prediction. All children still require ordinary live gates.

Capacity search is a bounded, deterministic rotation of the immutable search
core (numbering, splice library, region pools and known joins are built once
per call); each round only re-windows those pools. The per-CDR3 cap is global
across rounds, and the reported counters always reflect actual emissions.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math

from nanobody_opt import rules
from nanobody_opt.pr100_reference_donors import merit

SOURCE = 'segment_recombination'
PROTOCOL = 'imgt-multi-donor-v1'
REGIONS = ('fr1', 'cdr1', 'fr2', 'cdr2', 'fr3', 'cdr3', 'fr4')
ENDS = (26, 38, 55, 65, 104, 117, 128)
AA = frozenset('ACDEFGHIKLMNPQRSTVWY')
#: Hard ceiling on rotated search rounds; a bounded fallback, not a target.
MAX_ROUNDS = 64
#: Region-pool window width, unchanged from the audited single-pass search.
OPTION_WIDTH = 8


def settings(job):
    supplied = job.get('segment_generation', {})
    if (not isinstance(supplied, dict) or type(supplied.get('enabled', False)) is not bool
            or set(supplied) - {'enabled', 'limit', 'max_donors', 'beam_width', 'max_per_cdr3'}):
        raise ValueError('invalid segment_generation settings')
    result = {'enabled': False, 'limit': 2400, 'max_donors': 32, 'beam_width': 128,
              'max_per_cdr3': 8, **supplied}
    for key, low, high in (('limit', 1, 4800), ('max_donors', 2, 64), ('beam_width', 8, 512),
                           ('max_per_cdr3', 1, 64)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError('invalid segment_generation bound')
    return result


def _quality(row, challenge):
    rank, _ = merit(row, challenge)
    evidence_epoch = -rank[2]
    recent = evidence_epoch >= challenge['epoch'] - 10
    return (not recent, *rank, row['sequence_hash'])


def choose_donors(bank, challenge, maximum):
    """Independent V, loop-length and quality pools; no exact-cell prerequisite."""
    if type(maximum) is not int or not 2 <= maximum <= 64:
        raise ValueError('invalid donor bound')
    unique = {}
    for row in bank:
        seq, sha = row.get('sequence'), row.get('sequence_hash')
        if not isinstance(seq, str) or not seq or set(seq) - AA or rules.sequence_hash(seq) != sha:
            raise ValueError('segment donor sequence identity')
        first = row.get('first_archive_epoch')
        if first is not None and (type(first) is not int or first >= challenge['epoch']):
            continue
        if sha in unique and unique[sha] != row:
            raise ValueError('conflicting segment donor')
        unique[sha] = row
    rows = list(unique.values())
    gene = challenge['allowed_v'] + ':'
    def v_mismatch(row):
        return not any(cell.startswith(gene) for cell in row.get('cells', []))
    def distance(row):
        lengths = [int(cell.rsplit(':', 1)[1]) for cell in row.get('cells', [])]
        return min((abs(n - challenge['allowed_cdr3_length']) for n in lengths), default=100)
    ordered = [sorted(rows, key=lambda r: (v_mismatch(r), _quality(r, challenge))),
               sorted(rows, key=lambda r: (distance(r), _quality(r, challenge))),
               sorted(rows, key=lambda r: _quality(r, challenge))]
    selected, seen = [], set()
    for triple in itertools.zip_longest(*ordered):
        for row in triple:
            if row is not None and row['sequence_hash'] not in seen:
                selected.append(row); seen.add(row['sequence_hash'])
                if len(selected) == maximum:
                    return selected
    return selected


def prepare(row, numbered):
    """Validate an exact numbering round-trip; never infer boundaries by motifs.

    ``numbered`` is projected from a qualified ``abnumber.Chain`` iteration.
    Numbering itself is obtained by the worker, not accepted from bank claims.
    """
    seq = row['sequence']
    if rules.sequence_hash(seq) != row['sequence_hash'] or not seq or set(seq) - AA:
        raise ValueError('segment donor sequence identity')
    if len(numbered) != len(seq):
        raise ValueError('numbering does not cover exact donor')
    parts = {name: [] for name in REGIONS}
    coordinates, seen, previous = {}, set(), 0
    for index, item in enumerate(numbered):
        n, letter, aa = item['number'], item['letter'], item['aa']
        if (type(n) is not int or not previous <= n <= 128 or n < 1
                or not isinstance(letter, str) or len(letter) > 3
                or letter and (not letter.isalpha() or not letter.isupper())
                or (n, letter) in seen or aa != seq[index]):
            raise ValueError('invalid IMGT donor numbering')
        previous = n; seen.add((n, letter)); coordinates[(n, letter)] = index
        region = next(name for name, end in zip(REGIONS, ENDS) if n <= end)
        parts[region].append((index, n, letter, aa))
    if any(not value for value in parts.values()):
        raise ValueError('missing IMGT donor region')
    if any(coordinates.get((n, '')) is None or seq[coordinates[(n, '')]] != 'C' for n in (23, 104)):
        raise ValueError('donor canonical cysteine unavailable')
    segments = {}
    for name, values in parts.items():
        start, end = values[0][0], values[-1][0] + 1
        if end - start != len(values):
            raise ValueError('noncontiguous IMGT region')
        segments[name] = {'source_hash': row['sequence_hash'], 'source_start': start,
                          'source_end': end, 'region': name, 'sequence': seq[start:end],
                          'imgt_start': [values[0][1], values[0][2]],
                          'imgt_end': [values[-1][1], values[-1][2]]}
    return {**row, 'segments': segments, 'numbered': numbered}


def _piece(donor, start, end, region):
    values = donor['numbered'][start:end]
    return {'source_hash': donor['sequence_hash'], 'source_start': start, 'source_end': end,
            'region': region, 'sequence': donor['sequence'][start:end],
            'imgt_start': [values[0]['number'], values[0]['letter']],
            'imgt_end': [values[-1]['number'], values[-1]['letter']]}


def cdr_similarity(left, right):
    """The pinned search engine's positional CDR ratio, including length delta."""
    size = max(len(left), len(right))
    return (sum(a == b for a, b in zip(left, right)) / size) if size else 1.0


def _window(options, width, offset):
    """Keep strong priors and rotate exploration across continuation attempts."""
    if len(options) <= width:
        return options
    keep = width // 2
    rest = options[keep:]
    start = offset % len(rest)
    return options[:keep] + (rest + rest)[start:start + width - keep]


def _loop_pool(donors, challenge, stats):
    """Full, unwindowed CDR3 splice library, built once per generate call."""
    known = {d['segments']['cdr3']['sequence'] for d in donors}
    found = {}
    ordered = sorted(donors, key=lambda d: _quality(d, challenge))
    for li, left in enumerate(ordered):
        ls = left['segments']['cdr3']
        for ri, right in enumerate(ordered):
            if left['sequence_hash'] == right['sequence_hash']:
                continue
            rs = right['segments']['cdr3']
            for cut in (109, 110, 111, 112):
                l_end = next((i for i in range(ls['source_start'], ls['source_end'])
                              if left['numbered'][i]['number'] > cut), ls['source_end'])
                r_start = next((i for i in range(rs['source_start'], rs['source_end'])
                                if right['numbered'][i]['number'] > cut), rs['source_end'])
                if l_end == ls['source_start'] or r_start == rs['source_end']:
                    continue
                parts = [_piece(left, ls['source_start'], l_end, 'cdr3_left'),
                         _piece(right, r_start, rs['source_end'], 'cdr3_right')]
                loop = ''.join(p['sequence'] for p in parts)
                if len(loop) != challenge['allowed_cdr3_length']:
                    continue
                stats['length_compatible_loop_splices'] += 1
                if loop in known or any(cdr_similarity(loop, original) >= .85 for original in known):
                    stats['known_cdr3_conflicts'] += 1
                    continue
                tail = right['segments']['fr4']
                key = loop + tail['sequence']
                option = {'parts': parts + [tail], 'rank': li + ri,
                          'splice': {'after_imgt': cut, 'left_hash': left['sequence_hash'],
                                     'right_hash': right['sequence_hash']}}
                if key not in found or option['rank'] < found[key]['rank']:
                    found[key] = option
    options = sorted(found.values(), key=lambda o: (o['rank'], rules.digest(o)))
    stats['novel_loop_options'] = len(options)
    return options


def _option_pool(donors, names, challenge):
    """Full, unwindowed region pool, built once per generate call."""
    gene = challenge['allowed_v'] + ':'
    ordered = sorted(donors, key=lambda d: (
        not any(c.startswith(gene) for c in d.get('cells', [])), _quality(d, challenge)))
    unique = {}
    for rank, donor in enumerate(ordered):
        parts = [donor['segments'][n] for n in names]
        seq = ''.join(p['sequence'] for p in parts)
        unique.setdefault(seq, {'parts': parts, 'rank': rank})
    return list(unique.values())


def _join_key(left, right):
    return (left['region'], right['region'], left['sequence'][-3:], right['sequence'][:3])


def source_cdr_conflict(parts, donors):
    """Cheap source-library projection of the pinned conserved-CDR rule.

    This is not an archive-wide novelty verdict: the fully annotated child
    still goes through the existing complete validator novelty search.
    """
    cdrs = {name: ''.join(p['sequence'] for p in parts
                        if p['region'] == name or name == 'cdr3' and p['region'].startswith('cdr3'))
            for name in ('cdr1', 'cdr2', 'cdr3')}
    return any(cdr_similarity(cdrs['cdr3'], d['segments']['cdr3']['sequence']) >= .80
               and cdr_similarity(cdrs['cdr1'], d['segments']['cdr1']['sequence']) >= .90
               and cdr_similarity(cdrs['cdr2'], d['segments']['cdr2']['sequence']) >= .90
               for d in donors)


def reconstruct(row, references):
    """Exact residue-level provenance check, shared with history ingestion."""
    parts = row.get('segment_provenance')
    if not isinstance(parts, list) or not 2 <= len(parts) <= 16:
        raise ValueError('invalid segment provenance')
    assembled, cursor, sources = [], 0, set()
    for part in parts:
        sha = part['source_hash']; source = references.get(sha)
        a, b = part['source_start'], part['source_end']
        c, d = part['child_start'], part['child_end']
        if (source is None or rules.sequence_hash(source) != sha or type(a) is not int
                or type(b) is not int or not 0 <= a < b <= len(source)
                or type(c) is not int or type(d) is not int or c != cursor or d - c != b - a):
            raise ValueError('segment provenance interval mismatch')
        assembled.append(source[a:b]); sources.add(sha); cursor = d
    if len(sources) < 2 or ''.join(assembled) != row['sequence']:
        raise ValueError('segment provenance reconstruction mismatch')
    return ''.join(assembled)


def _coverage_rounds(pools):
    """Rounds until every rotating window has cycled; one when nothing rotates.

    Each pool advances by its own width-tied stride, so its rotation period is
    ``len(rest) // gcd(stride, len(rest))``. The maximum across pools bounds the
    rounds needed before a zero-yield round proves the library is exhausted.
    """
    periods = []
    for pool, width, stride in pools:
        if len(pool) <= width:
            continue
        rest = len(pool) - width // 2
        periods.append(rest // math.gcd(stride, rest))
    return max(periods, default=1)


def _search_round(framework_pools, loop_window, donors, known_joins, challenge, stats,
                  beam_width, round_offset):
    """One deterministic beam pass over windowed pools; no gate or native calls."""
    pools = list(framework_pools) + [loop_window]
    beam = [{'parts': [], 'rank': 0, 'unsupported_joins': 0}]
    for index, pool in enumerate(pools):
        is_loops = index == len(pools) - 1
        next_beam = {}
        for previous in beam:
            for option in pool:
                stats['beam_expansions'] += 1
                parts = previous['parts'] + option['parts']
                if is_loops and source_cdr_conflict(parts, donors):
                    stats['source_conserved_cdr_conflicts'] += 1
                    continue
                seq = ''.join(p['sequence'] for p in parts)
                # Source interval hashes are scientific lineage, never fitness evidence.
                joins = sum(_join_key(a, b) not in known_joins
                            for a, b in zip(parts, parts[1:])
                            if not a['region'].startswith('cdr3') and not b['region'].startswith('cdr3'))
                candidate = {'parts': parts, 'rank': previous['rank'] + option['rank'],
                             'unsupported_joins': joins,
                             'splice': option.get('splice', previous.get('splice'))}
                key = (joins, candidate['rank'],
                       rules.digest([challenge['epoch'], round_offset, seq]))
                if seq not in next_beam or key < next_beam[seq][0]:
                    next_beam[seq] = key, candidate
        ordered = [value for _, value in sorted(next_beam.values(), key=lambda x: x[0])]
        if is_loops:
            # Preserve loop diversity before the last beam truncation; a
            # strong framework prior must not fill it with one clonotype.
            by_loop = {}
            for value in ordered:
                loop = ''.join(p['sequence'] for p in value['parts']
                               if p['region'].startswith('cdr3'))
                by_loop.setdefault(loop, []).append(value)
            ordered = [value for group in itertools.zip_longest(*by_loop.values())
                       for value in group if value is not None]
        beam = ordered[:beam_width]
    return beam


def generate(donors, challenge, *, limit=128, beam_width=128, max_per_cdr3=8,
             offset=0, excluded=(), stats=None):
    """Bounded rotated-round capacity search; no target scoring/native gate claims."""
    bounds = settings({'segment_generation': {'enabled': True, 'limit': limit,
                                              'beam_width': beam_width,
                                              'max_per_cdr3': max_per_cdr3}})
    if type(offset) is not int or offset < 0 or len(donors) > 64:
        raise ValueError('invalid segment generation input bound')
    counters = Counter()
    donors = sorted(donors, key=lambda d: d['sequence_hash'])
    if len({d['sequence_hash'] for d in donors}) != len(donors):
        raise ValueError('duplicate prepared segment donor')
    for donor in donors:
        if prepare(donor, donor['numbered'])['segments'] != donor['segments']:
            raise ValueError('prepared donor segment drift')
    counters['prepared_donors'] = len(donors)
    excluded = set(excluded) | {d['sequence_hash'] for d in donors}
    known_joins = {_join_key(d['segments'][a], d['segments'][b]) for d in donors
                   for a, b in zip(REGIONS, REGIONS[1:])}
    references = {d['sequence_hash']: d['sequence'] for d in donors}
    layouts = {'linked_v': [('fr1', 'cdr1', 'fr2'), ('cdr2', 'fr3')],
               'region': [(name,) for name in REGIONS[:5]]}

    # Immutable search core: numbering was validated above, and the loop
    # library, region pools and known joins are built exactly once. Rotated
    # rounds only re-window these cached lists; nothing is re-prepared or
    # re-spliced per round.
    loop_pool = _loop_pool(donors, challenge, counters)
    framework = {layout: [_option_pool(donors, names, challenge) for names in groups]
                 for layout, groups in layouts.items()}
    loop_width = min(beam_width, 64)
    loop_stride = max(1, loop_width - loop_width // 2)
    option_stride = max(1, OPTION_WIDTH - OPTION_WIDTH // 2)
    coverage_rounds = _coverage_rounds([(loop_pool, loop_width, loop_stride)]
                                       + [(pool, OPTION_WIDTH, option_stride)
                                          for pools in framework.values() for pool in pools])

    result, seen, loop_counts = [], set(), Counter()
    rounds, empty_streak, filled, exhausted = 0, 0, False, False
    for r in range(MAX_ROUNDS):
        rounds = r + 1
        loop_offset = offset + r * loop_stride
        option_offset = offset + r * option_stride
        streams = []
        for layout, groups in layouts.items():
            windows = [_window(pool, OPTION_WIDTH, option_offset) for pool in framework[layout]]
            candidates = _search_round(windows, _window(loop_pool, loop_width, loop_offset),
                                       donors, known_joins, challenge, counters,
                                       beam_width, loop_offset)
            records = []
            for candidate in candidates:
                seq = ''.join(p['sequence'] for p in candidate['parts'])
                sha = rules.sequence_hash(seq)
                if sha in excluded:
                    counters['exact_or_resume_collisions'] += 1
                    continue
                provenance, cursor = [], 0
                for part in candidate['parts']:
                    end = cursor + len(part['sequence'])
                    provenance.append({k: v for k, v in part.items() if k != 'sequence'} |
                                      {'child_start': cursor, 'child_end': end})
                    cursor = end
                row = {'sequence': seq, 'sequence_hash': sha, 'source': SOURCE,
                       'proposal_type': PROTOCOL, 'proposal_lane': 'segment_' + layout,
                       'segment_layout': layout, 'segment_provenance': provenance,
                       'segment_splice': candidate['splice'],
                       'segment_search_prior': {'donor_rank_sum': candidate['rank'],
                                                'unsupported_joins': candidate['unsupported_joins']},
                       'requested_v': challenge['allowed_v'],
                       'requested_cdr3_length': challenge['allowed_cdr3_length'],
                       'acceptance_qualified': False, 'target_scored': False}
                reconstruct(row, references)
                records.append(row)
            streams.append(records)
        produced = 0
        for pair in itertools.zip_longest(*streams):
            for row in pair:
                if row is None or row['sequence_hash'] in seen:
                    continue
                loop = ''.join(row['sequence'][p['child_start']:p['child_end']]
                               for p in row['segment_provenance'] if p['region'].startswith('cdr3'))
                if loop_counts[loop] >= max_per_cdr3:
                    counters['cdr3_cap_drops'] += 1
                    continue
                result.append(row); seen.add(row['sequence_hash'])
                loop_counts[loop] += 1; produced += 1
                if len(result) == bounds['limit']:
                    break
            if len(result) == bounds['limit']:
                break
        if len(result) == bounds['limit']:
            filled = True
            break
        if produced == 0:
            empty_streak += 1
            if empty_streak >= coverage_rounds:
                exhausted = True
                break
        else:
            empty_streak = 0
    else:
        # The rotated window has cycled through every bounded round without
        # filling the requested quota: report exhaustion instead of a filled
        # claim. Emitted counters below still reflect exactly what was produced.
        exhausted = True
    counters['emitted'] = len(result)
    counters['emitted_distinct_cdr3s'] = len(loop_counts)
    counters['rounds_run'] = rounds
    counters['filled'] = filled
    counters['exhausted'] = exhausted
    if stats is not None:
        stats.update(counters)
    return result
