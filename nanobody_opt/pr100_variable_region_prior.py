"""Data-driven variable-region prior (``variable-region-prior-v1``).

A periodic scoring job ranks full top-scored nanobody variable regions on fast
boltzgen. This module distils those sequences into per-region fragment pools
(FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4) plus whole-sequence "haplotypes" (the seven
regions of one winning sequence as a single co-occurring unit), grouped by
germline ``allowed_v`` and CDR3 length bucket. The recombiner then assembles a
whole variable region whose every residue interval traces to a top-scored
source, either by reusing one winning haplotype or by independently drawing
each region from its weighted pool.

Region extraction reuses ``pr100_segment_generation``'s IMGT splitting: the
``REGIONS``/``ENDS`` boundaries and ``prepare``, which already validates the
canonical cysteines (23/104) and the exact numbering round-trip. No region is
ever inferred from sequence motifs.

The validator's historical-novelty gate still applies downstream; it is not
re-implemented here. With no usable prior (missing, wrong germline, or no
exact-length CDR3 fragment) the recombiner delegates to the existing
``pr100_segment_generation`` lane (germline-matched donors).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from nanobody_opt import pr100_graft_generation as graft
from nanobody_opt import pr100_segment_generation as segments
from nanobody_opt import rules

SCHEMA = 'variable-region-prior-v1'
SOURCE = 'top_fast_boltzgen_full_sequences'
PROTOCOL = 'imgt-variable-region-prior-v1'
REGIONS = segments.REGIONS
ENDS = segments.ENDS
AA = segments.AA

DEFAULT_TOP_REGIONS = 32
DEFAULT_TOP_HAPLOTYPES = 32
DEFAULT_LIMIT = 64
DEFAULT_WHOLE_HAPLOTYPE_PROBABILITY = 0.7
#: Hard ceiling on candidate draws per call; a bounded fallback, not a target.
_ATTEMPT_MULTIPLIER = 8


def project_numbering(sequence):
    """``numbered`` list for ``sequence`` via a qualified ``abnumber.Chain``.

    Yields one ``{'number', 'letter', 'aa'}`` dict per position, the same shape
    ``prepare`` consumes. Numbering is never accepted from bank claims.
    """
    from abnumber import Chain
    chain = Chain(sequence, scheme='imgt', cdr_definition='imgt', use_anarcii=False)
    return [{'number': pos.number, 'letter': pos.letter, 'aa': aa}
            for pos, aa in chain]


def region_substrings(sequence, numbered):
    """``{region: substring}`` for the seven IMGT regions of ``sequence``.

    ``numbered`` is projected from a qualified ``abnumber.Chain`` iteration.
    ``prepare`` validates the canonical cysteines and the exact numbering
    round-trip; boundaries are never inferred from motifs.
    """
    row = {'sequence': sequence, 'sequence_hash': rules.sequence_hash(sequence)}
    prepared = segments.prepare(row, numbered)
    return {name: prepared['segments'][name]['sequence'] for name in REGIONS}


def _weight(score):
    """A non-negative finite weight, or ``None`` for an unusable score."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) \
            or not math.isfinite(score):
        return None
    return max(float(score), 0.0)


def _germline_from_cells(cells):
    """The V-gene part of ``cells`` like ``['IGHV3-74:7']``."""
    if not isinstance(cells, (list, tuple)):
        return None
    for cell in cells:
        if isinstance(cell, str) and ':' in cell:
            return cell.rsplit(':', 1)[0]
    return None


def _resolve_entries(sequences, scores, numbered, allowed_v):
    """Normalise ``(sequences, scores[, numbered[, allowed_v]])`` into entries.

    Each element of ``sequences`` may be a plain string (then ``numbered`` must
    supply its numbering) or a bank-style row dict with ``sequence``,
    ``numbered`` and optionally ``cells``/``allowed_v``. When ``scores`` is
    ``None`` a row's own ``score`` field is used.
    """
    if not isinstance(sequences, (list, tuple)):
        raise ValueError('sequences must be a list')
    sequences = list(sequences)
    if scores is None:
        scores = [None] * len(sequences)
    else:
        scores = list(scores)
        if len(scores) != len(sequences):
            raise ValueError('sequences and scores must have the same length')
    if numbered is None:
        numbered = [None] * len(sequences)
    else:
        numbered = list(numbered)
        if len(numbered) != len(sequences):
            raise ValueError('sequences and numbered must have the same length')
    if isinstance(allowed_v, (list, tuple)):
        allowed_v = list(allowed_v)
        if len(allowed_v) != len(sequences):
            raise ValueError('sequences and allowed_v must have the same length')
    else:
        allowed_v = [allowed_v] * len(sequences)
    entries = []
    for seq, score, num, v in zip(sequences, scores, numbered, allowed_v):
        row = seq if isinstance(seq, dict) else None
        if row is not None:
            sequence = row.get('sequence')
            if num is None:
                num = row.get('numbered')
            if score is None:
                score = row.get('score')
            if v is None:
                v = row.get('allowed_v') or _germline_from_cells(row.get('cells'))
        else:
            sequence = seq
        entries.append({'sequence': sequence, 'score': score,
                        'numbered': num, 'allowed_v': v})
    return entries


def _representative_source(sources):
    """Highest-weight source hash, ties broken lexicographically."""
    return max(sources, key=lambda sha: (sources[sha], sha))


def _ordered_fragments(weights, limit):
    """Positive-weight fragments by (weight desc, lexicographic), best ``limit``."""
    positive = [frag for frag in weights if weights[frag]['weight'] > 0]
    positive.sort(key=lambda frag: (-weights[frag]['weight'], frag))
    return positive[:limit]


def _build_group(germline, bucket, regions, haplotypes, top_regions, top_haplotypes):
    out_regions = {}
    for name in REGIONS:
        out_regions[name] = [
            {'sequence': frag, 'weight': regions[name][frag]['weight'],
             'support': len(regions[name][frag]['sources']),
             'source_hash': _representative_source(regions[name][frag]['sources'])}
            for frag in _ordered_fragments(regions[name], top_regions)]
    ordered_hap = sorted((h for h in haplotypes.values() if h['weight'] > 0),
                         key=lambda h: (-h['weight'], h['sequence']))
    out_hap = [{'sequence': h['sequence'], 'sequence_hash': h['sequence_hash'],
                'weight': h['weight'], 'support': len(h['sources']),
                'regions': h['regions']} for h in ordered_hap[:top_haplotypes]]
    return {'germline': germline, 'bucket': bucket,
            'regions': out_regions, 'haplotypes': out_hap}


def derive_variable_region_prior(sequences, scores, *, numbered=None, allowed_v=None,
                                 top_n=None, top_regions=DEFAULT_TOP_REGIONS,
                                 top_haplotypes=DEFAULT_TOP_HAPLOTYPES,
                                 derived_at_unix=None):
    """Build a ``variable-region-prior-v1`` document from scored sequences.

    ``sequences[i]`` has score ``scores[i]`` (higher is better; pass
    ``1 / final_score`` when the bank stores lower-is-better ``final_score``).
    Each sequence is split into its seven IMGT regions via ``prepare``; entries
    with invalid residues, a non-finite/boolean score, no germline, or an
    invalid/exactness-failing numbering are skipped. Per (germline, CDR3 length
    bucket) group the top-``top_regions`` distinct fragments per region and the
    top-``top_haplotypes`` whole-sequence region sets are kept, weighted by
    score and ordered by weight desc then lexicographically. Groups without
    data keep empty lists so the recombiner falls back to the segment lane.
    """
    for value in (top_regions, top_haplotypes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('invalid top fragment/haplotype bound')
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int)
                              or top_n < 1):
        raise ValueError('invalid top_n bound')
    entries = _resolve_entries(sequences, scores, numbered, allowed_v)
    valid, skipped, numbering_failed = [], 0, 0
    for entry in entries:
        sequence = entry['sequence']
        weight = _weight(entry['score'])
        germline = entry['allowed_v']
        if (weight is None or not isinstance(sequence, str) or not sequence
                or set(sequence) - AA or not isinstance(germline, str) or not germline):
            skipped += 1
            continue
        num = entry['numbered']
        if num is None:
            try:
                num = project_numbering(sequence)
            except Exception:
                numbering_failed += 1
                continue
        try:
            substrings = region_substrings(sequence, num)
        except ValueError:
            numbering_failed += 1
            continue
        valid.append((sequence, weight, germline, substrings))
    valid.sort(key=lambda v: (-v[1], v[0]))
    if top_n is not None:
        valid = valid[:top_n]

    grouped = {}
    for sequence, weight, germline, substrings in valid:
        bucket = graft.length_bucket(len(substrings['cdr3']))
        key = f'{germline}:{bucket}'
        group = grouped.setdefault(key, {'germline': germline, 'bucket': bucket,
                                         'regions': {name: {} for name in REGIONS},
                                         'haplotypes': {}})
        sha = rules.sequence_hash(sequence)
        for name in REGIONS:
            frag = substrings[name]
            slot = group['regions'][name].setdefault(
                frag, {'weight': 0.0, 'sources': {}})
            slot['weight'] += weight
            slot['sources'][sha] = slot['sources'].get(sha, 0.0) + weight
        haplotype = group['haplotypes'].setdefault(
            sequence, {'weight': 0.0, 'sources': {}, 'sequence': sequence,
                       'sequence_hash': sha, 'regions': dict(substrings)})
        haplotype['weight'] += weight
        haplotype['sources'][sha] = haplotype['sources'].get(sha, 0.0) + weight

    by_group = {}
    for key in sorted(grouped):
        group = grouped[key]
        total = sum(slot['weight'] for name in REGIONS
                    for slot in group['regions'][name].values())
        if total <= 0:  # no positive signal: weight every distinct piece uniformly
            for name in REGIONS:
                for slot in group['regions'][name].values():
                    slot['weight'] = 1.0
            for haplotype in group['haplotypes'].values():
                haplotype['weight'] = 1.0
        by_group[key] = _build_group(group['germline'], group['bucket'],
                                     group['regions'], group['haplotypes'],
                                     top_regions, top_haplotypes)
    return {'schema': SCHEMA, 'source': SOURCE,
            'derived_at_unix': int(time.time() if derived_at_unix is None
                                   else derived_at_unix),
            'count': len(valid), 'skipped': skipped,
            'numbering_failed': numbering_failed,
            'top_regions': top_regions, 'top_haplotypes': top_haplotypes,
            'by_group': by_group}


def _valid_fragment(value):
    return (isinstance(value, dict) and isinstance(value.get('sequence'), str)
            and value['sequence'] and set(value['sequence']) <= AA
            and _weight(value.get('weight')) is not None
            and isinstance(value.get('support'), int)
            and isinstance(value.get('source_hash'), str))


def load_variable_region_prior(path):
    """The validated prior at ``path``, or ``None`` if missing/invalid/empty."""
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(document, dict) or document.get('schema') != SCHEMA
            or not isinstance(document.get('by_group'), dict)):
        return None
    populated = False
    for group in document['by_group'].values():
        if (not isinstance(group, dict) or not isinstance(group.get('germline'), str)
                or not isinstance(group.get('bucket'), str)
                or not isinstance(group.get('regions'), dict)
                or set(group['regions']) != set(REGIONS)
                or not isinstance(group.get('haplotypes'), list)):
            return None
        for name in REGIONS:
            fragments = group['regions'][name]
            if not isinstance(fragments, list) or any(not _valid_fragment(f)
                                                      for f in fragments):
                return None
            populated = populated or bool(fragments)
        for haplotype in group['haplotypes']:
            if (not isinstance(haplotype, dict)
                    or not isinstance(haplotype.get('sequence'), str)
                    or not haplotype['sequence'] or set(haplotype['sequence']) - AA
                    or _weight(haplotype.get('weight')) is None
                    or not isinstance(haplotype.get('support'), int)
                    or not isinstance(haplotype.get('sequence_hash'), str)
                    or not isinstance(haplotype.get('regions'), dict)
                    or set(haplotype['regions']) != set(REGIONS)
                    or any(not isinstance(v, str) or set(v) - AA
                           for v in haplotype['regions'].values())):
                return None
            populated = True
    return document if populated else None


def write_variable_region_prior(prior, path):
    """Atomically write ``prior`` as JSON to ``path``."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(prior, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def _group_for(prior, challenge):
    if not isinstance(prior, dict) or prior.get('schema') != SCHEMA:
        return None
    bucket = graft.length_bucket(challenge['allowed_cdr3_length'])
    return prior.get('by_group', {}).get(f"{challenge['allowed_v']}:{bucket}")


def _emit_row(sequence, sha, mode, provenance, challenge):
    return {'sequence': sequence, 'sequence_hash': sha, 'source': SOURCE,
            'proposal_type': PROTOCOL, 'proposal_lane': 'variable_region_prior',
            'variable_region_assembly': mode,
            'variable_region_provenance': provenance,
            'variable_region_sources': sorted({p['source_hash'] for p in provenance}),
            'requested_v': challenge['allowed_v'],
            'requested_cdr3_length': challenge['allowed_cdr3_length'],
            'acceptance_qualified': False, 'target_scored': False}


def recombine_variable_region(prior, challenge, *, seed=0, limit=DEFAULT_LIMIT,
                              whole_haplotype_probability=DEFAULT_WHOLE_HAPLOTYPE_PROBABILITY,
                              donors=(), excluded=(), stats=None):
    """Assemble whole variable regions from a derived prior, seeded and bounded.

    Each child's seven regions are drawn from the winning pools, weighted by
    score. With probability ``whole_haplotype_probability`` a single top
    sequence's full seven-region set is reused; otherwise every region is drawn
    independently. Only CDR3 fragments of the challenge's exact
    ``allowed_cdr3_length`` are used. Children are deduplicated against
    ``excluded`` and each other, and every emitted region traces to a
    top-scored source.

    When the prior is missing, the schema is wrong, no group matches the
    challenge's ``allowed_v`` + CDR3 length bucket, or no exact-length CDR3
    fragment exists, this delegates to ``pr100_segment_generation.generate``
    with the supplied germline-matched ``donors``.
    """
    if (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4800
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or not isinstance(whole_haplotype_probability, (int, float))
            or isinstance(whole_haplotype_probability, bool)
            or not 0.0 <= whole_haplotype_probability <= 1.0):
        raise ValueError('invalid variable-region recombination bound')

    group = _group_for(prior, challenge)
    length = challenge['allowed_cdr3_length']
    region_options = None
    if group is not None:
        pools = {name: [(f, f['weight']) for f in group['regions'][name]]
                 for name in REGIONS}
        pools['cdr3'] = [(f, w) for f, w in pools['cdr3']
                         if len(f['sequence']) == length]
        haplotype_options = [(h, h['weight']) for h in group['haplotypes']
                             if len(h['regions']['cdr3']) == length]
        if pools['cdr3']:
            region_options = pools

    if region_options is None:
        if stats is not None:
            stats['variable_region_fallback'] = True
        return segments.generate(donors, challenge, limit=limit,
                                 excluded=excluded, stats=stats)

    rng = random.Random(seed)
    excluded = set(excluded)
    result, seen = [], set()
    whole_count = independent_count = collisions = attempts = 0
    max_attempts = max(limit * _ATTEMPT_MULTIPLIER, 64)
    while len(result) < limit and attempts < max_attempts:
        attempts += 1
        if haplotype_options and rng.random() < whole_haplotype_probability:
            haplotype = rng.choices([h for h, _ in haplotype_options],
                                    weights=[w for _, w in haplotype_options], k=1)[0]
            substrings = {name: haplotype['regions'][name] for name in REGIONS}
            provenance = [{'region': name, 'sequence': substrings[name],
                           'source_hash': haplotype['sequence_hash']}
                          for name in REGIONS]
            mode = 'whole_haplotype'
            whole_count += 1
        else:
            substrings, provenance = {}, []
            for name in REGIONS:
                fragment = rng.choices([f for f, _ in region_options[name]],
                                       weights=[w for _, w in region_options[name]],
                                       k=1)[0]
                substrings[name] = fragment['sequence']
                provenance.append({'region': name, 'sequence': fragment['sequence'],
                                   'source_hash': fragment['source_hash']})
            mode = 'independent_draw'
            independent_count += 1
        sequence = ''.join(substrings[name] for name in REGIONS)
        sha = rules.sequence_hash(sequence)
        if sha in excluded or sha in seen:
            collisions += 1
            continue
        seen.add(sha)
        result.append(_emit_row(sequence, sha, mode, provenance, challenge))

    counters = Counter(emitted=len(result), attempts=attempts,
                       whole_haplotype_children=whole_count,
                       independent_children=independent_count,
                       duplicate_or_excluded_collisions=collisions,
                       variable_region_fallback=False)
    if stats is not None:
        stats.update(counters)
    return result
