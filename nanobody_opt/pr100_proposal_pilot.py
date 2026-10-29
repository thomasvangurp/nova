"""CPU-only experimental proposals; no acceptance or binding-quality claim.

This is a deliberately simple, reproducible baseline for the new challenge,
not a production generator. Every emitted sequence still needs annotation,
native developability, archive novelty and target scoring.

The CDR3 sampling below is no longer a flat residue-weight table. It is the
positional recipe measured from the dominant Q9NZQ7 rival (uid 8, coldkey
5EHS..., leaderboard epochs 25984-25997, 12 winning CDR3s):

    CDR3 = anchor(2) + head(5) + body(L-10) + tail(3)

The anchor is ``A[RKQ]`` or ``S[RKQ]``, the head is ``YPHRF`` (P at position
2, R at position 4, F at position 5, with H/Y alternating at positions 1 and
3), the body is a G/S/Q/K/P/R/D/E hydrophilic linker (no aromatic, no
aliphatic -- this is the stretch that grows with the loop), and the tail is
the fixed IGHJ4-style junction tail ``FDY``. The one length-independent
invariant is ZERO L/I/V/M/W anywhere in the CDR3.
"""
from __future__ import annotations

import hashlib
import random

# CDR3 positional recipe constants (all measured, not guessed -- see module
# docstring). ``CDR3_BODY`` is exactly the residue set observed in the rival's
# body stretch; it deliberately excludes every aromatic and every aliphatic.
CDR3_ANCHOR_FIRST = 'AS'       # anchor first residue: A or S
CDR3_ANCHOR_SECOND = 'RKQ'     # anchor second residue: basic/polar side chain
CDR3_HEAD = 'YPHRF'            # head: P@2, R@4, F@5 fixed; H/Y alternate @1,@3
CDR3_BODY = 'GSQKPRDE'         # body: hydrophilic linker, NO aromatic/aliphatic
# The head ends in F; a G as the first body residue plus a later G would form
# a false ``FGxG`` FR4 anchor for regions.find_regions. The rival's measured
# bodies never open with G, so the first body residue excludes it.
CDR3_BODY_FIRST = 'SQKPRDE'
CDR3_TAIL = 'FDY'              # fixed IGHJ4-style junction tail

# Charged/polar CDR2 motif. This is the first 10 residues of the rival's
# 17-residue CDR2 window; the remaining 7 stay germline. Together they put the
# CDR2 hydrophobic fraction at 4/17 ~= 0.235 (field median 0.353), which is the
# surface-hydrophobic-patches (PSH) lever.
CDR2_MOTIF = 'DISSQGDQTD'

# FR1 edits measured in framework.py (5EHS vs the epoch's allowed_v germline).
# IMGT positions (the HMM panel numbering scaffold() reads): 2 is the mandatory
# V->Q; 5/9/13/20 are the small set of Q/K/T swaps (framework.py FR1 positions
# 5/9/12/19, which are IMGT 5/9/13/20 because of the IMGT-10 gap). Weights are
# the observed target-residue counts across the rival's 12 entries.
_FR1_TARGETS = {
    5: (('Q', 'E'), (5, 3)),
    9: (('T', 'S'), (4, 3)),
    13: (('K', 'Q'), (5, 1)),
    20: (('K', 'Q', 'T'), (2, 3, 1)),
}

STRATA = ('same_v_same_length', 'same_v_other_length', 'other_v', 'no_parent')
GENES = ('IGHV3-23', 'IGHV3-66', 'IGHV3-74', 'IGHV3-7', 'IGHV3-11',
         'IGHV3-43', 'IGHV3-48', 'IGHV3-53', 'IGHV3-64')
CELLS = (('IGHV3-23', 16), ('IGHV3-23', 18), ('IGHV3-7', 16), ('IGHV3-23', 5))
PROTOCOL = 'pr100-conditional-germline-loop-v1'


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest(), 'big')


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
    # 5EHS uses a hybrid J: FR4 matches IGHJ6 ('WGQGTTVTVSS') while the CDR3
    # junction tail is the IGHJ4-style 'FDY' (IGHJ4 IMGT 115-117). Read the
    # suffix from IGHJ6 and the tail from IGHJ4 rather than one J allele, so the
    # emitted FR4 and tail reproduce the rival exactly.
    tail = ''.join(panels['J'][j4[0]][p] for p in range(115, 118))
    suffix = ''.join(aa for p, aa in panels['J'][j6[0]].items() if p >= 118 and aa != '-')
    if len(tail) != 3 or '-' in tail or not suffix:
        raise ValueError('incomplete joining reference')
    return prefix, tail, suffix, v[0], j6[0]


def _apply_framework_edits(prefix, rng):
    """FR1 V->Q plus the measured Q/K/T swaps, and the FR2 ``GKGLE -> GQERE``.

    FR2 ``G->E`` (IMGT 49) and ``L->R`` (IMGT 50) are already applied by
    scaffold()'s hallmark update; only ``K->Q`` (IMGT 48) completes the motif.
    """
    prefix[2] = 'Q'
    for pos, (population, weights) in _FR1_TARGETS.items():
        prefix[pos] = rng.choices(population, weights=weights, k=1)[0]
    if 48 in prefix:
        prefix[48] = 'Q'


def _apply_cdr2_motif(base):
    """Overwrite the first 10 residues of the 17-residue CDR2 window.

    Anchors mirror ``regions.find_regions`` so the window here is the same one
    ``hydrophobicity.score`` will read. Synthetic panels without the FR2
    tryptophan anchor leave the window untouched (the motif is a recipe detail
    that only applies to VHH-shaped assemblies).
    """
    c1 = base.find('C', 18, 28)
    if c1 < 0:
        return base
    w = base.find('WVRQ', c1)
    if w < 0:
        w = base.find('W', c1 + 8)
    if w < 0 or w + 31 > len(base):
        return base
    window = base[w + 14:w + 31]
    if len(window) != 17:
        return base
    return base[:w + 14] + CDR2_MOTIF + window[10:] + base[w + 31:]


def _build_cdr3_loop(rng, loop_len):
    """anchor(2) + head(min(5, loop_len-2)) + body(rest); tail is appended later.

    ``loop_len`` is the requested CDR3 length minus the fixed 3-residue tail.
    The body only exists when loop_len - 2 > 5 (i.e. requested length > 10).
    The anchor first residue is chosen so the finished CDR3 (with the FDY tail)
    stays under the 0.44 hydrophobic ceiling: A contributes exactly one more
    hydrophobic residue than S, so it is only offered when that still fits.
    """
    head_len = min(5, max(0, loop_len - 2))
    body_len = max(0, loop_len - 2 - head_len)
    second = rng.choice(CDR3_ANCHOR_SECOND)
    if head_len == 1:
        # A single Y here would put a 6-residue CDR3 at 0.5 hydrophobic.
        head = 'H'
    elif head_len >= 2:
        template = list(CDR3_HEAD)  # 'YPHRF': P@2, R@4, F@5 fixed
        hy = rng.choice(('H', 'Y'))
        template[0] = hy
        template[2] = 'Y' if hy == 'H' else 'H'
        head = ''.join(template[:head_len])
    else:
        head = ''
    body = ('' if body_len == 0
            else rng.choice(CDR3_BODY_FIRST) +
            ''.join(rng.choice(CDR3_BODY) for _ in range(body_len - 1)))
    core = head + body + CDR3_TAIL
    hydro = sum(1 for aa in core if aa in 'FLIVMWYA')
    first = (rng.choice(CDR3_ANCHOR_FIRST) if (hydro + 1) / (2 + len(core)) <= 0.44
             else 'S')
    return first + second + head + body


def propose(panels, gene, length, *, seed, ordinal, reference_cdr3=None):
    if gene not in GENES or not 5 <= length <= 23:
        raise ValueError('pilot cell outside declared scope')
    rng = random.Random(_seed(PROTOCOL, gene, length, seed, ordinal))
    prefix, tail, suffix, v, j = scaffold(panels, gene)
    _apply_framework_edits(prefix, rng)
    base = _apply_cdr2_motif(''.join(prefix[p] for p in sorted(prefix)))
    if reference_cdr3 is None:
        loop_rng = rng
    else:
        if not isinstance(reference_cdr3, str) or len(reference_cdr3) < 3:
            raise ValueError('reference CDR3 unavailable')
        # The reference only steers the draw: the emitted loop is still the
        # positional template, never a flat random graft of donor residues.
        loop_rng = random.Random(_seed(PROTOCOL, gene, length, seed, ordinal,
                                       reference_cdr3))
    loop = _build_cdr3_loop(loop_rng, length - len(tail))
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
