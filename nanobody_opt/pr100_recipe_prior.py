"""Data-driven de-novo CDR3 recipe prior (``recipe-prior-v1``).

A periodic scoring job ranks candidate CDR3s on multipass fast boltzgen. This
module distils the top-scored ones into per-length-bucket anchors (first two
residues), tails (last three residues) and cores (recurrent k-mers between
them), writes them to ``recipe_prior.json`` and loads them back for
``pr100_graft_generation`` (``de_novo_cdr3(..., prior=...)``). The static
constants in the generator remain the fallback whenever no prior exists.

CLI::

    python -m nanobody_opt.pr100_recipe_prior derive \
        --input scored_cdr3s.jsonl --output recipe_prior.json [--top-n 50]

where each input line is ``{"cdr3": "ARFQPRYFDY", "score": 0.87}``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from nanobody_opt import pr100_graft_generation as graft

SCHEMA = 'recipe-prior-v1'
SOURCE = 'top_fast_boltzgen_cdr3s'
BUCKETS = tuple(name for name, _low, _high in graft.RECIPE_BUCKETS)
_MIN_LENGTH, _MAX_LENGTH = graft.RECIPE_BUCKETS[0][1], graft.RECIPE_BUCKETS[-1][2]
_ANCHOR_LEN, _TAIL_LEN = 2, 3
_KMER_SIZES = range(3, 7)
_KINDS = ('anchors', 'cores', 'tails')


def _weight(score):
    """A non-negative finite weight, or ``None`` for an unusable score."""
    if isinstance(score, bool) or not isinstance(score, (int, float)) \
            or not math.isfinite(score):
        return None
    return max(float(score), 0.0)


def _ranked(weights, limit):
    """Keys by (weight desc, longer first, lexicographic), best ``limit``."""
    ordered = sorted(weights, key=lambda k: (-weights[k], -len(k), k))
    return ordered[:limit]


def _derive_bucket(items, top_anchors, top_cores, top_tails):
    """Motifs from ``[(cdr3, weight)]`` of a single length bucket."""
    if items and not any(w > 0 for _c, w in items):
        items = [(c, 1.0) for c, _w in items]  # no signal: weight uniformly
    anchors, tails = defaultdict(float), defaultdict(float)
    kmer_weight, kmer_support = defaultdict(float), defaultdict(set)
    for index, (cdr3, weight) in enumerate(items):
        anchors[cdr3[:_ANCHOR_LEN]] += weight
        tails[cdr3[-_TAIL_LEN:]] += weight
        middle = cdr3[_ANCHOR_LEN:-_TAIL_LEN]
        for size in _KMER_SIZES:
            for start in range(len(middle) - size + 1):
                kmer = middle[start:start + size]
                if index not in kmer_support[kmer]:  # once per CDR3
                    kmer_support[kmer].add(index)
                    kmer_weight[kmer] += weight
    # Anchors and tails that are themselves liability motifs (e.g. FDP) would only
    # ever be generated to be rejected, so they never enter the prior.
    anchors = {k: w for k, w in anchors.items() if not graft.liability_count(k)}
    tails = {k: w for k, w in tails.items() if not graft.liability_count(k)}
    min_support = min(2, len(items))
    recurrent = {k: w for k, w in kmer_weight.items()
                 if len(kmer_support[k]) >= min_support and not graft.liability_count(k)}
    # A k-mer contained in a longer k-mer with identical support is redundant.
    cores = {k: w for k, w in recurrent.items()
             if not any(len(o) > len(k) and k in o and kmer_support[o] == kmer_support[k]
                        for o in recurrent)}
    return {'anchors': _ranked(anchors, top_anchors),
            'cores': _ranked(cores, top_cores),
            'tails': _ranked(tails, top_tails)}


def derive_recipe_prior(cdr3s, scores, *, top_anchors=4, top_cores=4, top_tails=3,
                        derived_at_unix=None):
    """Build a ``recipe-prior-v1`` document from scored CDR3s.

    ``cdr3s[i]`` has score ``scores[i]`` (higher is better). CDR3s of invalid
    residues, lengths outside 5-23, or with a non-finite/boolean score are
    skipped; negative scores weigh 0 (if a whole bucket weighs 0 it is weighted
    uniformly). Per bucket the anchors are the top first-two residues, the tails
    the top last-three residues and the cores the top recurrent 3-6-mers of the
    middle (not already implied by a longer k-mer with the same support), all
    weighted by score, ordered by score desc then lexicographically. Anchors,
    tails and cores that are liability motifs are excluded. Buckets without data
    keep empty lists, so the generator falls back to its static pool for them.
    """
    cdr3s, scores = list(cdr3s), list(scores)
    if len(cdr3s) != len(scores):
        raise ValueError('cdr3s and scores must have the same length')
    grouped, count = defaultdict(list), 0
    for cdr3, score in zip(cdr3s, scores):
        weight = _weight(score)
        if (weight is None or not isinstance(cdr3, str)
                or not _MIN_LENGTH <= len(cdr3) <= _MAX_LENGTH
                or not set(cdr3) <= graft._AMINO_ACIDS):
            continue
        grouped[graft.length_bucket(len(cdr3))].append((cdr3, weight))
        count += 1
    return {'schema': SCHEMA, 'source': SOURCE,
            'derived_at_unix': int(time.time() if derived_at_unix is None
                                   else derived_at_unix),
            'count': count,
            'by_length': {name: _derive_bucket(grouped.get(name, []), top_anchors,
                                               top_cores, top_tails)
                          for name in BUCKETS}}


def _valid_motifs(values, low, high):
    return (isinstance(values, list)
            and all(isinstance(v, str) and low <= len(v) <= high
                    and set(v) <= graft._AMINO_ACIDS for v in values))


def load_recipe_prior(path):
    """The validated prior at ``path``, or ``None`` if missing/invalid/empty.

    Strict: any schema violation (wrong ``schema``, unknown bucket, non-list
    motifs, anchors not 2 residues, tails not 3, cores not 1-12, non-amino-acid
    characters) invalidates the whole prior. A prior with no motif in any
    bucket is empty and also yields ``None``.
    """
    try:
        document = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(document, dict) or document.get('schema') != SCHEMA
            or not isinstance(document.get('by_length'), dict)):
        return None
    limits = {'anchors': (_ANCHOR_LEN, _ANCHOR_LEN), 'tails': (_TAIL_LEN, _TAIL_LEN),
              'cores': (1, graft._PRIOR_MAX_CORE_LEN)}
    populated = False
    for bucket, entry in document['by_length'].items():
        if bucket not in BUCKETS or not isinstance(entry, dict):
            return None
        for kind in _KINDS:
            values = entry.get(kind, [])
            if not _valid_motifs(values, *limits[kind]):
                return None
            populated = populated or bool(values)
    return document if populated else None


def write_recipe_prior(prior, path):
    """Atomically write ``prior`` as JSON to ``path``."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(prior, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def read_scored_jsonl(path, top_n=None):
    """``(cdr3s, scores)`` from a JSONL of ``{cdr3, score}``, best ``top_n`` first."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and isinstance(row.get('cdr3'), str) \
                and _weight(row.get('score')) is not None:
            rows.append((row['cdr3'], float(row['score'])))
    rows.sort(key=lambda r: (-r[1], r[0]))
    if top_n is not None:
        rows = rows[:top_n]
    return [r[0] for r in rows], [r[1] for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pr100_recipe_prior')
    sub = parser.add_subparsers(dest='command', required=True)
    derive = sub.add_parser('derive', help='derive recipe_prior.json from scored CDR3s')
    derive.add_argument('--input', required=True, help='JSONL of {cdr3, score}')
    derive.add_argument('--output', required=True, help='recipe_prior.json to write')
    derive.add_argument('--top-n', type=int, default=50,
                        help='use only the N best-scored CDR3s (default 50)')
    args = parser.parse_args(argv)
    cdr3s, scores = read_scored_jsonl(args.input, args.top_n)
    prior = derive_recipe_prior(cdr3s, scores)
    if not any(prior['by_length'][b][k] for b in BUCKETS for k in _KINDS):
        print('derived prior is empty; not writing', file=sys.stderr)
        return 1
    write_recipe_prior(prior, args.output)
    print(f"wrote {args.output} from {prior['count']} CDR3s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
