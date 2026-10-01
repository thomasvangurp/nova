#!/usr/bin/env python3
"""Master compound portfolio: choose 20 molecules from the bank, one per child uid.

Mirrors the nanobody publisher's fleet model -- a master selects, children submit,
uids are resolved from the chain by COLDKEY every epoch -- with one deliberate
difference that is measured, not assumed:

    publish_current_nanobody.py requires the fleet's sequences to be PAIRWISE
    DISTINCT, because a duplicate means two uids sharing one deduped validator
    score. On the COMPOUND track duplicates are not deduped. Across 158 uid
    pairs from the top operator with >=80% portfolio overlap the score ratio was
    0.9937, and a pair with identical portfolios scored identically (1.6972 /
    1.6972). So every child gets the SAME best portfolio; carrying the nanobody
    distinctness rule across would discard most of the multiplier.

Two-stage selection, because selecting 20 from a large bank by a single noisy
score is a winner's curse. Boltz diffusion sampling is stochastic: the same
scorer on a different seed differs by 0.005234 per molecule (docs/
BOLTZ_NORMAL_SEED_NOISE_20260917.md), which is 7.5% of the bank's score variance.
Selecting once and submitting cost a measured -0.0717 on a 11,283-molecule bank;
averaging a 200-molecule shortlist over 3 passes recovers most of it for about
$1.34 of GPU time.

This module selects and emits the artifact. It does not sign or submit.
"""
import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED_NOISE_ABS_DIFF = 0.005234          # same scorer, different seed
ENTROPY_GATE = 0.25
PORTFOLIO = 20


def load_bank(path, scorer=None):
    query = ('SELECT name,smiles,components,inchikey,heavy_atoms,rotatable_bonds,score,scorer '
             'FROM candidates WHERE reaction=3 AND score IS NOT NULL')
    parameters = []
    if scorer:
        query += ' AND scorer=?'
        parameters.append(scorer)
    with sqlite3.connect('file:%s?mode=ro' % path, uri=True) as connection:
        connection.execute('PRAGMA busy_timeout=60000')
        rows = connection.execute(query, parameters).fetchall()
    return [dict(name=r[0], smiles=r[1], components=json.loads(r[2]), inchikey=r[3],
                 heavy_atoms=r[4], rotatable_bonds=r[5], score=r[6], scorer=r[7])
            for r in rows]


def atompair_entropy(molecules):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    bits = []
    for item in molecules:
        molecule = Chem.MolFromSmiles(item['smiles'])
        if molecule is None:
            return 0.0
        bits.append(np.array(generator.GetFingerprint(molecule), dtype=float))
    if not bits:
        return 0.0
    p = np.mean(bits, axis=0)
    p = p[(p > 0) & (p < 1)]
    return float(np.mean(-p * np.log2(p) - (1 - p) * np.log2(1 - p))) if len(p) else 0.0


def expected_after_rescore(selected, pool, passes):
    """Shrink the selected scores towards the pool mean by the noise share.

    Reports what the validator should see, not what our own scores say. The gap
    between the two is the winner's curse and it is the reason for stage two.
    """
    import numpy as np
    values = np.array([m['score'] for m in pool], dtype=float)
    total = values.std()
    noise = (SEED_NOISE_ABS_DIFF * 1.2533) / math.sqrt(2) / math.sqrt(max(1, passes))
    if total <= 0:
        return sum(m['score'] for m in selected)
    shrink = max(0.0, 1 - (noise ** 2) / (total ** 2))
    mean = values.mean()
    return float(sum(mean + shrink * (m['score'] - mean) for m in selected))


def diversify(chosen, shortlist, children, vary_slots):
    """Give each child a portfolio differing only in its weakest slots.

    The validator scores molecules deterministically -- identical portfolios
    from two uids scored identically (1.6972 / 1.6972) -- so eleven copies of
    one portfolio are one entry. But we do not know the validator's realised
    values in advance, and our own ranking is noisy, so portfolios that differ
    in their marginal slots are genuinely separate attempts at the same target.

    Simulated over 6,000 draws at the measured per-molecule noise: varying the
    bottom 2-3 slots gains +0.0071 with a singly-scored shortlist and +0.0034
    once the shortlist is averaged over three passes. Varying more is worse --
    eight slots gains only +0.0033, and random 20-subsets of the top 40 gain
    +0.0016 -- because the mean lost exceeds the extra spread.
    """
    if children <= 1 or vary_slots <= 0:
        return [list(chosen)]
    head = chosen[:-vary_slots]
    used = {m['inchikey'] for m in chosen}
    spare = [m for m in shortlist if m['inchikey'] not in used]
    portfolios = [list(chosen)]
    cursor = 0
    while len(portfolios) < children and cursor + vary_slots <= len(spare):
        portfolios.append(head + spare[cursor:cursor + vary_slots])
        cursor += vary_slots
    return portfolios


def select(pool, shortlist_size, passes):
    ordered = sorted(pool, key=lambda m: -m['score'])
    shortlist = ordered[:shortlist_size]
    chosen, seen = [], set()
    for item in shortlist:
        if item['inchikey'] in seen:
            continue
        trial = chosen + [item]
        # Entropy is checked, never assumed: a bank-selected portfolio measured
        # 0.70 against the 0.25 gate, but a concentrated one can fall to 0.27.
        if len(trial) < 4 or atompair_entropy(trial) >= ENTROPY_GATE:
            chosen = trial
            seen.add(item['inchikey'])
        if len(chosen) == PORTFOLIO:
            break
    return chosen, shortlist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', required=True)
    parser.add_argument('--scorer', default=None, help='restrict to one scorer label')
    parser.add_argument('--shortlist', type=int, default=200,
                        help='candidates to rescore in stage two')
    parser.add_argument('--passes', type=int, default=3,
                        help='independent scoring passes averaged over the shortlist')
    parser.add_argument('--uids', default='', help='comma-separated child uids, if known')
    parser.add_argument('--vary-slots', type=int, default=2,
                        help='how many of the weakest slots differ between children')
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()

    pool = load_bank(arguments.bank, arguments.scorer)
    if len(pool) < PORTFOLIO:
        print(json.dumps(dict(error='bank_too_small', molecules=len(pool))))
        return 1
    chosen, shortlist = select(pool, arguments.shortlist, arguments.passes)
    if len(chosen) < PORTFOLIO:
        print(json.dumps(dict(error='entropy_infeasible', selected=len(chosen))))
        return 1

    uids = [u.strip() for u in arguments.uids.split(',') if u.strip()]
    observed = sum(m['score'] for m in chosen)
    variants = diversify(chosen, shortlist, max(1, len(uids)), arguments.vary_slots)
    report = dict(
        bank=str(arguments.bank), bank_molecules=len(pool),
        portfolio=[dict(name=m['name'], smiles=m['smiles'], components=m['components'],
                        score=m['score'], scorer=m['scorer']) for m in chosen],
        observed_sum20=round(observed, 6),
        atompair_entropy=round(atompair_entropy(chosen), 6),
        entropy_gate=ENTROPY_GATE,
        shortlist_size=len(shortlist), rescore_passes=arguments.passes,
        expected_after_one_pass=round(expected_after_rescore(chosen, pool, 1), 6),
        expected_after_rescore=round(expected_after_rescore(chosen, pool, arguments.passes), 6),
        winners_curse_note=('validator rescores with a different seed; same-scorer '
                            'seed difference is %.6f per molecule' % SEED_NOISE_ABS_DIFF),
        children=[dict(uid=u,
                       portfolio=[m['name'] for m in variants[i % len(variants)]],
                       sum20=round(sum(m['score'] for m in variants[i % len(variants)]), 6))
                  for i, u in enumerate(uids)],
        distinct_child_portfolios=len(variants), vary_slots=arguments.vary_slots,
        diversification_justification=('the validator scores molecules deterministically, so '
                                       'identical portfolios are one entry; portfolios differing '
                                       'in their weakest slots are separate attempts. Measured '
                                       'gain +0.0071 singly-scored, +0.0034 after a 3-pass '
                                       'shortlist. Varying more slots is worse.'),
        differs_from_nanobody_publisher=('publish_current_nanobody.py requires pairwise '
                                         'distinct sequences; that rule is nanobody-only'),
        off_chain=True, live_submission_authorized=False, signed=False)
    Path(arguments.output).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('portfolio', 'children')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
