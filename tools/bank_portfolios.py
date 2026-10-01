#!/usr/bin/env python3
"""Turn the bank into one distinct 20-molecule portfolio per uid we hold.

Emits the JSON that publish_current_nanobody --molecules-file consumes:
``{"<uid>": "name1,name2,...", ...}``.

Four things here are load-bearing, and each of them has a cost attached.

*Selection runs on RESCORED means, never single-pass scores.* Measured on a
200-molecule 3-pass rescore: the top 20 by single-pass score declare 1.6493 and
rescore to 1.5158 -- a 0.1335 winner's curse -- while selecting on the rescored
mean gives 1.5975. That +0.0817 is worth more than doubling the bank.

*Both entropy fingerprints must pass.* utils/molecules.compute_maccs_entropy in
this repo uses MACCS keys, but the live validator has been observed gating on
AtomPair, and this repo's config lags the live one elsewhere (it still says
min_entropy 0.1 where live is 0.25). An epoch was already lost to an entropy of
0.2462. Satisfying both costs almost nothing, so both are required.

*Submitted molecules burn their NEIGHBOURHOOD, not just themselves.* The
validator rejects anything within 0.5 Tanimoto of the historical archive, and
this bank is built by sweeping one role at >=0.5 similarity -- so every
submission invalidates its own analogues. Excluding only exact InChIKeys would
re-pick a near-twin next epoch and have it silently dropped.

*The portfolios differ.* The validator dedupes by molecule, so N uids carrying
one portfolio are worth exactly one ticket. Varying the tail is worth +0.0071
single-scored, +0.0034 after three passes -- small, but free.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

BASE = Path('/root/bank')
MINIMUM_ENTROPY = 0.25
NOVELTY = 0.5


def entropies(smiles):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    import math

    def shannon(fps, size):
        if not fps:
            return 0.0
        total = 0.0
        for bit in range(size):
            on = sum(1 for fp in fps if fp[bit])
            p = on / len(fps)
            if 0.0 < p < 1.0:
                total -= p * math.log2(p) + (1 - p) * math.log2(1 - p)
        return total / size

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    maccs = [MACCSkeys.GenMACCSKeys(m) for m in mols]
    pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    pairs = [pair_gen.GetFingerprint(m) for m in mols]
    return shannon(maccs, 167), shannon(pairs, 2048)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default=str(BASE / 'bank.sqlite'))
    parser.add_argument('--reaction', type=int, default=3)
    parser.add_argument('--uids', required=True,
                        help='comma-separated uids discovered FROM THE CHAIN')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--shortlist', type=int, default=200)
    parser.add_argument('--vary-tail', type=int, default=3,
                        help='slots varied between uids; 0 makes every uid identical')
    parser.add_argument('--submitted', default=str(BASE / 'submitted.json'),
                        help='names we have already committed, and whose '
                             'neighbourhoods are therefore burned')
    parser.add_argument('--novelty-csv', default=str(BASE / 'novelty.csv'))
    parser.add_argument('--output', default=str(BASE / 'molecules.json'))
    parser.add_argument('--require-rescored', action='store_true', default=True)
    arguments = parser.parse_args()

    uids = [int(u) for u in arguments.uids.split(',') if u.strip()]
    if not uids:
        print('no uids given; refusing to emit an empty fleet', file=sys.stderr)
        return 1

    connection = sqlite3.connect(arguments.bank)
    tables = {r[0] for r in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if 'rescored' not in tables:
        print('bank has no `rescored` table: run bank_rescore.py first. '
              'Selecting on single-pass scores costs 0.0817.', file=sys.stderr)
        return 1

    submitted_path = Path(arguments.submitted)
    submitted = set(json.loads(submitted_path.read_text())) \
        if submitted_path.is_file() else set()

    rows = list(connection.execute(
        'SELECT c.name, c.smiles, c.inchikey, r.mean_score, r.passes '
        'FROM rescored r JOIN candidates c ON c.name = r.name '
        'WHERE c.reaction = ? ORDER BY r.mean_score DESC', (arguments.reaction,)))
    rows = [r for r in rows if r[0] not in submitted]
    if len(rows) < arguments.count:
        print(f'only {len(rows)} rescored candidates for rxn {arguments.reaction}; '
              f'need {arguments.count}', file=sys.stderr)
        return 1

    sys.path.insert(0, str(BASE))
    from compound_miner.validity import HistoricalIndex, ValidityGate
    from compound_miner.types import Candidate

    history = HistoricalIndex(arguments.novelty_csv)
    gate = ValidityGate()

    def usable(name, smiles, inchikey):
        candidate = Candidate(name, smiles, arguments.reaction, (), inchikey, 0, 0, 0.0, 0.0)
        try:
            if not gate.accepts(candidate, check_similarity=False):
                return False
            return history.is_novel(candidate, NOVELTY)
        except Exception:
            return False

    pool = [r for r in rows if usable(r[0], r[1], r[2])][:arguments.shortlist]
    if len(pool) < arguments.count + arguments.vary_tail * len(uids):
        print(f'pool of {len(pool)} is too small for {len(uids)} distinct '
              f'portfolios', file=sys.stderr)
        return 1

    head = pool[:arguments.count - arguments.vary_tail]
    tail_pool = pool[arguments.count - arguments.vary_tail:]

    fleet, cursor = {}, 0
    for uid in uids:
        tail = tail_pool[cursor:cursor + arguments.vary_tail]
        cursor += arguments.vary_tail
        chosen = head + tail
        if len(chosen) != arguments.count:
            print(f'uid {uid}: built {len(chosen)} molecules, need '
                  f'{arguments.count}', file=sys.stderr)
            return 1
        names = [c[0] for c in chosen]
        if len(set(names)) != arguments.count:
            print(f'uid {uid}: duplicate molecules', file=sys.stderr)
            return 1
        maccs, pair = entropies([c[1] for c in chosen])
        if maccs < MINIMUM_ENTROPY or pair < MINIMUM_ENTROPY:
            print(f'uid {uid}: entropy maccs={maccs:.4f} atompair={pair:.4f} '
                  f'below {MINIMUM_ENTROPY}; the validator would skip it',
                  file=sys.stderr)
            return 1
        fleet[str(uid)] = ','.join(names)
        print(f'uid {uid}: sum={sum(c[3] for c in chosen):.4f} '
              f'maccs={maccs:.4f} atompair={pair:.4f}')

    Path(arguments.output).write_text(json.dumps(fleet, indent=1))
    print(f'wrote {arguments.output} for {len(fleet)} uids')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
