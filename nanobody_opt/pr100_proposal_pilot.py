"""CPU-only experimental proposals; no acceptance or binding-quality claim.

This is a deliberately simple, reproducible baseline for the new challenge,
not a production generator. Every emitted sequence still needs annotation,
native developability, archive novelty and target scoring.

CDR3 sampling is length-split:

* L >= 12 uses the positional recipe derived from the beat-5EHS analysis and
  the actually-submitted beater sequences::

      CDR3 = anchor(2) + head(5) + body(L-10) + tail(3)

  anchor ``S-K``/``S-R``/``A-K``, head ``Y-P-Y-[R/H]-F`` (three aromatics,
  positive R or relocated H at position 4), body drawn from the charged,
  balanced beater pool ``D/K/T/R/S/E`` (each K/R paired with D/E), tail the
  fixed IGHJ4-style ``FDY``. Zero L/I/V/M/W anywhere in the loop.
* L < 12 falls back to the flat ``AMINO_ACID_WEIGHTS`` sampling (b188a10).
  The positional template has no body below L=10 and its truncated head is the
  rival's LOSING L=7 regime, so short loops keep the flat draw for diversity.

The framework applies the full charged CDR2 motif ``DISSQGDQTD`` to all nine
genes (the salt-bridge network that buries the CDR3 aromatics and keeps surface
PSH low), FR1 ``E->D`` (1) + ``V->Q`` (2), and the FR2 hallmarks ``G->E`` (49)
+ ``L->R`` (50) with germline K (48). The 5EHS FR1 Q/K/T swaps and FR2 ``K->Q``
are dropped to keep the full CDR2 motif within the >= 82% V-identity gate.
"""
from __future__ import annotations

import hashlib
import math
import random

AMINO_ACIDS = 'ADEFGHIKLMNPQRSTVWY'  # Do not introduce an extra cysteine.

# Flat CDR3 residue weights (b188a10). Used ONLY for short loops (L < 12),
# where the positional template has no body and its truncated head matches the
# rival's losing L=7 regime. Zero weight removes N and the aliphatics
# L/I/V/M/W from the draw; A/E/T keep a small positive floor.
AMINO_ACID_WEIGHTS = {
    'F': 3.0, 'Y': 3.0, 'D': 3.0, 'R': 3.0,
    'H': 2.0, 'P': 2.0,
    'S': 1.6, 'K': 1.4, 'Q': 1.2, 'G': 1.0,
    'A': 0.6, 'E': 0.2, 'T': 0.2,
    'N': 0.0, 'L': 0.0, 'I': 0.0, 'V': 0.0, 'M': 0.0, 'W': 0.0,
}

# CDR3 positional recipe constants for L >= 12 (beat-5EHS head swap).
CDR3_ANCHOR_FIRST = 'SA'       # anchor first residue: S preferred, A allowed
CDR3_ANCHOR_SECOND = 'KR'      # anchor second residue: positive K/R
CDR3_HEAD = 'YPYRF'            # Y-P-Y-[R/H]-F: the submitted beater head
# Body is the beater DKTRDSE pattern with the extra D dropped and the D/E spaced
# (D-K-T-R-E-S): each positive K/R is paired with a negative D/E, and never more
# than one negative per cycle, so the charged body buries the CDR3 aromatics
# without clustering negative charge patches.
CDR3_BODY_PATTERN = 'DKTRES'
CDR3_TAIL = 'FDY'              # fixed IGHJ4-style junction tail (J identity)

# Full charged/polar CDR2 motif (first 10 of the 17-residue CDR2 window).
# Applied to all nine genes: the charged D/Q motif is the salt-bridge network
# that buries the CDR3 aromatics and keeps surface PSH low.
CDR2_MOTIF = 'DISSQGDQTD'

# Below this CDR3 length the positional template is not used.
CDR3_TEMPLATE_MIN_LENGTH = 12

STRATA = ('same_v_same_length', 'same_v_other_length', 'other_v', 'no_parent')
GENES = ('IGHV3-23', 'IGHV3-66', 'IGHV3-74', 'IGHV3-7', 'IGHV3-11',
         'IGHV3-43', 'IGHV3-48', 'IGHV3-53', 'IGHV3-64')
CELLS = (('IGHV3-23', 16), ('IGHV3-23', 18), ('IGHV3-7', 16), ('IGHV3-23', 5))
PROTOCOL = 'pr100-conditional-germline-loop-v1'


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest(), 'big')


def _weighted_choice(rng, *, exclude=None):
    """Draw one amino acid with the hydrophilicity bias, optionally excluding one."""
    population = [aa for aa in AMINO_ACIDS if aa != exclude]
    weights = [AMINO_ACID_WEIGHTS[aa] for aa in population]
    return rng.choices(population, weights=weights, k=1)[0]


def select_references(rows, gene, length, stratum, *, limit=32):
    """Stable, score-free reference selection; old-target merit is not reused."""
    if stratum not in STRATA:
        raise ValueError('unknown reference stratum')
    if stratum == 'no_parent':
        return []
    found = {}
    for row in rows:
        a = row['pr100_annotation']
        same = gene in a['germline']['V']['genes']
        match = (same and a['cdr3_length'] == length if stratum == STRATA[0] else
                 same and a['cdr3_length'] != length if stratum == STRATA[1] else not same)
        if match:
            found[hashlib.sha256(row['sequence'].encode()).hexdigest()] = row
    return [found[k] for k in sorted(found)[:limit]]


def scaffold(panels, gene):
    v = sorted(k for k in panels['V'] if k.split('*')[0] == gene)
    j4 = sorted(k for k in panels['J'] if k.split('*')[0] == 'IGHJ4')
    j6 = sorted(k for k in panels['J'] if k.split('*')[0] == 'IGHJ6')
    if not v or not j4 or not j6:
        raise ValueError('required human germline reference unavailable')
    # The HMM panel uses 128 aligned states. Runtime annotation is required to
    # check the resulting IMGT numbering; this assembly is not that evidence.
    prefix = {p: aa for p, aa in panels['V'][v[0]].items() if p <= 104 and aa != '-'}
    if not {23, 49, 50, 104} <= prefix.keys():
        raise ValueError('incomplete framework reference')
    prefix.update({23: 'C', 49: 'E', 50: 'R', 104: 'C'})
    # 5EHS uses a hybrid J: the CDR3 junction tail is IGHJ4's 'FDY' (IMGT
    # 115-117) while the FR4 suffix is IGHJ6's 'WGQGTTVTVSS' (IMGT >=118). The
    # full J region therefore matches IGHJ4 best (13/14), so the reported
    # reference allele is IGHJ4, not IGHJ6.
    tail = ''.join(panels['J'][j4[0]][p] for p in range(115, 118))
    suffix = ''.join(aa for p, aa in panels['J'][j6[0]].items() if p >= 118 and aa != '-')
    if len(tail) != 3 or '-' in tail or not suffix:
        raise ValueError('incomplete joining reference')
    return prefix, tail, suffix, v[0], j4[0]


def _apply_framework_edits(prefix, rng):
    """FR1 E->D (1) + V->Q (2); FR2 hallmarks G->E (49), L->R (50).

    The 5EHS FR1 Q/K/T swaps at 5/9/13/20 and the FR2 K->Q (48) are dropped so
    the full charged CDR2 motif fits the >= 82% V-identity budget. FR2 keeps
    germline K (``GKERE``).
    """
    prefix[1] = 'D'
    prefix[2] = 'Q'


def _cdr2_window(seq):
    """Locate the 17-residue CDR2 window, mirroring regions.find_regions."""
    c1 = seq.find('C', 18, 28)
    if c1 < 0:
        return None
    w = seq.find('WVRQ', c1)
    if w < 0:
        w = seq.find('W', c1 + 8)
    if w < 0 or w + 31 > len(seq):
        return None
    window = seq[w + 14:w + 31]
    if len(window) != 17:
        return None
    return w + 14, window


def _apply_cdr2_edit(base, replacement):
    """Overwrite the first 10 residues of the 17-residue CDR2 window."""
    found = _cdr2_window(base)
    if found is None:
        return base
    start, window = found
    return base[:start] + replacement + window[10:] + base[start + 17:]


def v_identity(panels, gene, prefix, loop):
    """Numeric V percent identity = matches / non-gap reference positions.

    Mirrors utils/germline_identity.py: the query at each non-gap reference
    position is the assembled prefix (IMGT 1-104) plus the first three CDR3
    residues (IMGT 105-107, which the V germline still covers). Missing query
    positions are mismatches.
    """
    alleles = sorted(k for k in panels['V'] if k.split('*')[0] == gene)
    if not alleles:
        raise ValueError('required human germline reference unavailable')
    ref = panels['V'][alleles[0]]
    nongap = [p for p in sorted(ref) if ref[p] != '-']
    matches = 0
    for p in nongap:
        if p <= 104:
            q = prefix.get(p, '-')
        elif p == 105 and len(loop) >= 1:
            q = loop[0]
        elif p == 106 and len(loop) >= 2:
            q = loop[1]
        elif p == 107 and len(loop) >= 3:
            q = loop[2]
        else:
            q = '-'
        if q == ref[p]:
            matches += 1
    return matches, len(nongap), matches / len(nongap)


def j_identity(panels, tail, suffix):
    """Numeric J percent identity = max matches / non-gap positions over J alleles."""
    region = tail + suffix
    best = 0.0
    for allele, ref in panels['J'].items():
        nongap = [p for p in sorted(ref) if ref[p] != '-']
        if not nongap:
            continue
        matches = 0
        for p in nongap:
            idx = p - 115
            q = region[idx] if 0 <= idx < len(region) else '-'
            if q == ref[p]:
                matches += 1
        best = max(best, matches / len(nongap))
    return best


def _build_cdr3_loop(rng, loop_len):
    """anchor(2) + head(5) + body(loop_len - 7); positional template for L >= 12.

    ``loop_len`` is the requested CDR3 length minus the fixed 3-residue tail,
    so ``loop_len >= 9`` here: the head is always full and the body always
    exists (requested length > 10). The head is the submitted beater head
    ``Y-P-Y-[R/H]-F`` (three aromatics buried by the charged body); the body is
    the beater-pattern ``D-K-T-R-E-S`` cycled to the requested length (each
    positive K/R paired with a negative D/E, spaced so negatives never cluster).
    """
    head_len = 5
    body_len = loop_len - 2 - head_len
    second = rng.choice(CDR3_ANCHOR_SECOND)  # K or R
    template = list(CDR3_HEAD)  # 'YPYRF': Y-P-Y-[R/H]-F
    template[3] = rng.choice('RH')  # R (charge) or H (relocated His)
    head = ''.join(template)
    body = ''.join(CDR3_BODY_PATTERN[i % len(CDR3_BODY_PATTERN)]
                   for i in range(body_len))
    # Head + tail carry 5 aromatics; an A anchor adds a sixth. Only offer A when
    # 6/length stays <= 0.44 (length >= 14); shorter loops force S.
    length = loop_len + 3
    first = rng.choice(CDR3_ANCHOR_FIRST) if 6 / length <= 0.44 else 'S'
    return first + second + head + body


def _graft_reference(rng, reference_cdr3, n):
    """Trim/pad/mutate a donor CDR3 to the requested loop length (b188a10)."""
    loop = list(reference_cdr3[:-3])
    while len(loop) > n:
        loop.pop(rng.randrange(len(loop)))
    while len(loop) < n:
        loop.insert(rng.randrange(len(loop)+1), _weighted_choice(rng))
    for p in rng.sample(range(n), max(1, math.ceil(n/4))):
        loop[p] = _weighted_choice(rng, exclude=loop[p])
    return loop


def propose(panels, gene, length, *, seed, ordinal, reference_cdr3=None):
    if gene not in GENES or not 5 <= length <= 23:
        raise ValueError('pilot cell outside declared scope')
    rng = random.Random(_seed(PROTOCOL, gene, length, seed, ordinal))
    prefix, tail, suffix, v, j = scaffold(panels, gene)
    _apply_framework_edits(prefix, rng)
    base = _apply_cdr2_edit(''.join(prefix[p] for p in sorted(prefix)), CDR2_MOTIF)
    ordered = sorted(prefix)
    prefix = {p: c for p, c in zip(ordered, base)}
    n = length - len(tail)
    if reference_cdr3 is None:
        if length >= CDR3_TEMPLATE_MIN_LENGTH:
            loop = _build_cdr3_loop(rng, n)
        else:
            loop = [_weighted_choice(rng) for _ in range(n)]
    else:
        if not isinstance(reference_cdr3, str) or len(reference_cdr3) < 3:
            raise ValueError('reference CDR3 unavailable')
        # Graft the donor's CDR3 chemistry (trim/pad/mutate), never erase it.
        loop = _graft_reference(rng, reference_cdr3, n)
    loop = list(loop)
    _, _, fraction = v_identity(panels, gene, prefix, loop)
    if fraction < 0.82:
        raise ValueError('emitted V identity below the 82% gate')
    sequence = base + ''.join(loop) + tail + suffix
    return {'sequence': sequence, 'proposal_type': PROTOCOL,
            'requested_v': gene, 'requested_cdr3_length': length,
            'v_reference_allele': v, 'j_reference_allele': j,
            'seed': seed, 'ordinal': ordinal, 'acceptance_qualified': False,
            'target_scored': False}


def dry_run(rows, panels, *, extract_cdr3, per_arm=256, seed=1000):
    """Return proposals and counts for the fixed 16 arms; performs no GPU work."""
    if not 1 <= per_arm <= 256:
        raise ValueError('pilot proposal cap must be 1..256')
    arms = []
    for gene, length in CELLS:
        for stratum in STRATA:
            available = select_references(rows, gene, length, stratum, limit=None)
            refs = available[:32]
            arm = {'cell': [gene, length], 'stratum': stratum,
                   'available_reference_count': len(available),
                   'selected_reference_count': len(refs), 'gpu_calls': 0,
                   'status': 'RUNNABLE', 'proposals': [], 'attempts': 0, 'emitted': 0}
            if stratum != 'no_parent' and not refs:
                arm.update(status='UNRUNNABLE', reason='no_reference_in_declared_stratum')
                arms.append(arm)
                continue
            loops = [extract_cdr3(r['sequence']) for r in refs]
            seen = set()
            for ordinal in range(per_arm*10):
                row = propose(panels, gene, length, seed=seed, ordinal=ordinal,
                              reference_cdr3=loops[ordinal % len(loops)] if loops else None)
                arm['attempts'] += 1
                if row['sequence'] in seen:
                    continue
                seen.add(row['sequence'])
                row['reference_hash'] = (hashlib.sha256(refs[ordinal % len(refs)]['sequence'].encode()).hexdigest()
                                         if refs else None)
                arm['proposals'].append(row)
                if len(arm['proposals']) == per_arm:
                    break
            arm['emitted'] = len(arm['proposals'])
            arms.append(arm)
    return arms
