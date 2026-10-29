"""CPU-only experimental proposals; no acceptance or binding-quality claim.

This is a deliberately simple, reproducible baseline for the new challenge,
not a production generator. Every emitted sequence still needs annotation,
native developability, archive novelty and target scoring.
"""
from __future__ import annotations

import hashlib
import math
import random

AMINO_ACIDS = 'ADEFGHIKLMNPQRSTVWY'  # Do not introduce an extra cysteine.

# CDR3 residue weights replicating the measured winning recipe of the dominant
# Q9NZQ7 rival (uid 8, coldkey 5EHS..., leaderboard epochs 25979-25997, 16
# winning CDR3s / 201 residues). Their recipe concentrates on aromatic F/Y
# stacking contacts and charged D/R salt bridges, plus H/P, keeps S/K/Q/G
# moderate, uses A/E/T only sparsely, and never uses N or the aliphatics
# L/I/V/M/W. Measured fractions: Y .159, R .134, F .129, D .124, P .090,
# S .080, K .060, H .055, Q .050, A .040, G .040, E .025, T .015, and 0 for
# N/L/I/V/M/W. H and P are lifted above their raw frequency because histidine
# adds aromatic stacking plus H-bonds and proline breaks the loop for paratope
# exposure. N and the aliphatics get a weight of exactly zero, so
# ``random.choices`` never returns them; A/E/T keep a small positive floor (A
# slightly larger: it is the only non-F/Y residue in the hydrophobic set, so it
# holds the expected CDR3 hydrophobic fraction near 0.30). The weights are
# length-agnostic -- the same table is applied at whatever CDR3 length the
# epoch demands.
# Expected CDR3 hydrophobic fraction (FLIVMWYA weight share) is
# (3+3+0.6)/22.2 ~= 0.30 -- under the rival's ~33% surface hydrophobicity and
# safely below the 0.44 sequence pre-screen ceiling, which trims any overshoot.
AMINO_ACID_WEIGHTS = {
    # Favour: aromatic F/Y stacking contacts and charged D/R salt bridges.
    'F': 3.0, 'Y': 3.0, 'D': 3.0, 'R': 3.0,
    # H/P: contact chemistry, lifted above their raw frequency.
    'H': 2.0, 'P': 2.0,
    # Moderate: solubility and framework compatibility (S/K/Q/G).
    'S': 1.6, 'K': 1.4, 'Q': 1.2, 'G': 1.0,
    # Tiny: A/E/T, used sparsely by the rival. A carries the largest share of
    # the trio because it is the only remaining hydrophobic residue besides
    # F/Y and keeps the expected CDR3 hydrophobic fraction near 0.30.
    'A': 0.6, 'E': 0.2, 'T': 0.2,
    # Zero: N and the aliphatics L/I/V/M/W, absent from the rival's winning
    # CDR3s. A zero weight removes them from the draw entirely.
    'N': 0.0, 'L': 0.0, 'I': 0.0, 'V': 0.0, 'M': 0.0, 'W': 0.0,
}
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
    j = sorted(k for k in panels['J'] if k.split('*')[0] == 'IGHJ4')
    if not v or not j:
        raise ValueError('required human germline reference unavailable')
    # The HMM panel uses 128 aligned states. Runtime annotation is required to
    # check the resulting IMGT numbering; this assembly is not that evidence.
    prefix = {p: aa for p, aa in panels['V'][v[0]].items() if p <= 104 and aa != '-'}
    if not {23, 49, 50, 104} <= prefix.keys():
        raise ValueError('incomplete framework reference')
    prefix.update({23: 'C', 49: 'E', 50: 'R', 104: 'C'})
    tail = ''.join(panels['J'][j[0]][p] for p in range(115, 118))
    suffix = ''.join(aa for p, aa in panels['J'][j[0]].items() if p >= 118 and aa != '-')
    if len(tail) != 3 or '-' in tail or not suffix:
        raise ValueError('incomplete joining reference')
    return prefix, tail, suffix, v[0], j[0]


def propose(panels, gene, length, *, seed, ordinal, reference_cdr3=None):
    if gene not in GENES or not 5 <= length <= 23:
        raise ValueError('pilot cell outside declared scope')
    rng = random.Random(_seed(PROTOCOL, gene, length, seed, ordinal))
    prefix, tail, suffix, v, j = scaffold(panels, gene)
    positions = [p for p in sorted(prefix) if 27 <= p <= 38 or 56 <= p <= 65]
    if len(positions) < 6:
        raise ValueError('insufficient mutable loop positions')
    # Six CDR1/2 substitutions provide diversity even for a five-residue CDR3.
    # Their effect on V assignment, stability and target quality is measured,
    # never inferred from this construction.
    for p in rng.sample(positions, 6):
        prefix[p] = _weighted_choice(rng, exclude=prefix[p])
    n = length - len(tail)
    if reference_cdr3 is None:
        loop = [_weighted_choice(rng) for _ in range(n)]
    else:
        if not isinstance(reference_cdr3, str) or len(reference_cdr3) < 3:
            raise ValueError('reference CDR3 unavailable')
        loop = list(reference_cdr3[:-3])
        while len(loop) > n:
            loop.pop(rng.randrange(len(loop)))
        while len(loop) < n:
            loop.insert(rng.randrange(len(loop)+1), _weighted_choice(rng))
        for p in rng.sample(range(n), max(1, math.ceil(n/4))):
            loop[p] = _weighted_choice(rng, exclude=loop[p])
    sequence = ''.join(prefix[p] for p in sorted(prefix)) + ''.join(loop) + tail + suffix
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
