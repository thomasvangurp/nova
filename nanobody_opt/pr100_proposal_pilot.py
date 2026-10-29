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
* L < 12 uses a short-regime template modeled on the rival's winning ``RAFDY``
  junction: head ``R[A/H]`` + fixed ``FDY`` tail, extended with the polar
  ``T/S/H/G`` pool for longer loops. This replaces the flat
  ``AMINO_ACID_WEIGHTS`` draw, whose flat heads (``HDFDY``, ``RSFDY``, ...)
  collapsed interface confidence ~2x below the rival's ``RAFDY``/``RHFDY``.

The framework applies the full charged CDR2 motif ``DISSQGDQTD`` to all nine
genes ONLY in the long regime (L >= 12), FR1 ``E->D`` (1) + ``V->Q`` (2), and
the FR2 hallmarks ``G->E`` (49) + ``L->R`` (50) with germline K (48). In the
short regime (L < 12) the germline CDR2 is kept per gene instead of the charged
motif, matching the rival's short-regime winning scaffold, AND the winners'
FR1 swaps V5/G9/V13/K20 are restored while FR2 keeps the germline K48
(``GKERE``). The restored framework is short-regime only: with the charged
long-regime CDR2 it would drop 5/9 genes below the >= 82% V-identity gate.
"""
from __future__ import annotations

import hashlib
import math
import random

AMINO_ACIDS = 'ADEFGHIKLMNPQRSTVWY'  # Do not introduce an extra cysteine.

# Flat CDR3 residue weights (b188a10). Used ONLY by the reference-graft path
# (_graft_reference) when a donor CDR3 is supplied, not by the short template.
# Zero weight removes N and the aliphatics L/I/V/M/W from the draw; A/E/T keep
# a small positive floor.
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
# Applied to all nine genes in the LONG-CDR3 regime only: the charged D/Q motif
# is the salt-bridge network that buries the CDR3 aromatics and keeps surface
# PSH low. In the short-CDR3 regime (CDR3 length < CDR3_TEMPLATE_MIN_LENGTH) the
# rival's winning CDR2 is the germline loop (IYSEGST/IYSDGST derive from
# IGHV3-66/53's germline IYSGGST), so the charged motif is NOT applied there.
CDR2_MOTIF = 'DISSQGDQTD'

# Below this CDR3 length the positional template and charged CDR2 are not used.
CDR3_TEMPLATE_MIN_LENGTH = 12

# Short-regime (L < 12) framework residues, IMGT position -> residue, carried by
# the 5EHS winners: V5/G9/V13/K20 in FR1, with germline K48 kept in FR2
# (``GKERE``). Applied on top of the shared D1/Q2 + E49/R50 hallmarks.
SHORT_REGIME_FRAMEWORK = {5: 'V', 9: 'G', 13: 'V', 20: 'K', 48: 'K'}

# Germline CDR2 (IMGT 56-65, gaps removed) per V gene, read from the ANARCI
# 128-state reference panel in tests/pr100_germline.py. This is the reference
# for the short-CDR3 regime; the proposal simply keeps the germline CDR2 rather
# than overwriting it, but the explicit table pins the per-gene truth for tests.
GERMLINE_CDR2 = {
    'IGHV3-23': 'ISGSGGST',
    'IGHV3-66': 'IYSGGST',
    'IGHV3-74': 'INSDGSST',
    'IGHV3-7': 'IKQDGSEK',
    'IGHV3-11': 'ISSSGSTI',
    'IGHV3-43': 'ISWDGGST',
    'IGHV3-48': 'ISSSSSTI',
    'IGHV3-53': 'IYSGGST',
    'IGHV3-64': 'ISSNGGST',
}

# Short-CDR3 head recipe (L < 12), modeled on the rival's RAFDY winner. The
# fixed FDY tail (IGHJ4 IMGT 115-117) is appended by the caller, so the head
# length n = CDR3 length - 3:
#   L=4 -> 'R', L=5 -> 'RA'/'RH', L=6 -> 'R[A/H]X', ...
# R is the always-present first residue (the RAFDY junction positive charge);
# the second position favors A (RAFDY, iiptm 0.50) with H as the alternate
# (RHFDY, iiptm 0.44); longer heads extend with the polar T/S/H/G pool that
# mirrors the rival's winning TSH length-6 head. No L/I/V/M/W/N anywhere.
SHORT_CDR3_HEAD_FIRST = 'R'
SHORT_CDR3_HEAD_SECOND = 'AH'
SHORT_CDR3_SECOND_WEIGHTS = (3.0, 1.0)          # favor A (RAFDY)
SHORT_CDR3_EXTENSION = 'TSHG'
SHORT_CDR3_EXTENSION_WEIGHTS = (3.0, 3.0, 2.0, 1.0)

# The three byte-exact short-regime 5EHS leaderboard winners. The scaffold must
# stay as close to these as the similarity gate allows but must never emit a
# byte-identical sequence (the dedup guard in propose() enforces this).
RIVAL_5EHS_SEQUENCES = frozenset({
    'EQQLVESGGGLVKPGRSLKLSCAASGFTVSSYSMSWFRQAPGKEREEVSVIYSEGSTYYAESVKGRFTISRDNSKNTLYLQMNSLKAEDTAVYYCRAFDYWGQGTTVTVSS',
    'DQQLVESGGGLVQPGRSLKLSCAASGFTVSSYSMSWFRQASGKEREEVSVIYSDGSTYYAESVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCRHFDYWGQGTTVTVSS',
    'DQQLVESGTGLVQPGSSLRLSCAASGFTVSSYSMSWFRQAPGKEREEVSVIYSDGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCTSHFDYWGQGTTVTVSS',
})

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


def _apply_framework_edits(prefix, rng, *, short_regime=False):
    """FR1 E->D (1) + V->Q (2); FR2 hallmarks G->E (49), L->R (50).

    In the SHORT-CDR3 regime (L < 12) the winners' framework is restored on
    top of those: ``SHORT_REGIME_FRAMEWORK`` sets V5/G9/V13/K20 and keeps the
    germline K48, giving the ``GKERE`` FR2 motif. This is the framework the
    rival's winning ``RAFDY`` scaffold carries and what drives ``design_iiptm``
    0.50; with the germline CDR2 it stays above the >= 82% V-identity gate for
    all nine genes (see ``test_short_regime_full_framework_v_identity_meets_gate``).

    The full framework is NOT applied in the long regime (L >= 12): the
    charged ``DISSQGDQTD`` CDR2 motif already consumes most of the >= 82%
    V-identity budget and the extra five swaps would drop 5/9 genes to ~81%.
    The long regime keeps the pruned framework (``GKERE``, germline K 48).
    """
    prefix[1] = 'D'
    prefix[2] = 'Q'
    if short_regime:
        prefix.update(SHORT_REGIME_FRAMEWORK)


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


def _build_short_cdr3_loop(rng, loop_len):
    """R[A/H] + polar extension: the RAFDY-family short-regime CDR3 head.

    ``loop_len`` is the requested CDR3 length minus the fixed 3-residue tail,
    so ``1 <= loop_len <= 8`` here. The head always opens with the positive R
    (the rival's RAFDY junction signature), the second position favors A
    (RAFDY, the strongest winner at iiptm 0.50) with H as the alternate (RHFDY,
    iiptm 0.44), and any remaining positions are filled from the polar T/S/H/G
    pool that mirrors the rival's winning TSH length-6 head. The caller appends
    the fixed FDY tail, so the head never contains L/I/V/M/W/N and stays a
    small, deterministic, high-confidence pool rather than an open flat draw.
    """
    if not 1 <= loop_len <= 8:
        raise ValueError('short CDR3 loop outside the 1..8 head window')
    loop = [SHORT_CDR3_HEAD_FIRST]
    if loop_len >= 2:
        loop.append(rng.choices(SHORT_CDR3_HEAD_SECOND,
                                weights=SHORT_CDR3_SECOND_WEIGHTS, k=1)[0])
    if loop_len >= 3:
        loop.extend(rng.choices(SHORT_CDR3_EXTENSION,
                                weights=SHORT_CDR3_EXTENSION_WEIGHTS, k=loop_len - 2))
    return loop


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
    _apply_framework_edits(prefix, rng,
                           short_regime=length < CDR3_TEMPLATE_MIN_LENGTH)
    ordered = sorted(prefix)
    base = ''.join(prefix[p] for p in ordered)
    if length >= CDR3_TEMPLATE_MIN_LENGTH:
        base = _apply_cdr2_edit(base, CDR2_MOTIF)
    prefix = {p: c for p, c in zip(ordered, base)}
    n = length - len(tail)
    if reference_cdr3 is None:
        if length >= CDR3_TEMPLATE_MIN_LENGTH:
            loop = _build_cdr3_loop(rng, n)
        else:
            loop = _build_short_cdr3_loop(rng, n)
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
    if sequence in RIVAL_5EHS_SEQUENCES:
        raise ValueError('emitted a byte-identical 5EHS leader sequence')
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
            seen = set(RIVAL_5EHS_SEQUENCES)
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
