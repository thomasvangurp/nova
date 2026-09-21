#!/usr/bin/env python3
"""Second-stage rescore: average a shortlist over independent seeds.

Selecting the best 20 from a large bank by a single noisy score is a winner's
curse. Boltz diffusion sampling is stochastic -- the same scorer on a different
seed differs by 0.005234 per molecule (docs/BOLTZ_NORMAL_SEED_NOISE_20260917.md)
-- so the molecules that look best are disproportionately those whose sampling
ran high, and the validator's own rescore claws it back.

Measured on a 15,666-molecule bank: selecting on one pass gives an expected
validator sum20 of 1.5705, while selecting on the latent truth would give
1.6306. That 0.06 gap is worth roughly a doubling of the bank (~$250, ~40
hours). Averaging k passes cuts the measurement noise by sqrt(k) exactly where
selection happens, and a 200-molecule shortlist at 3 passes costs about $1.34.

No cleverer estimator recovers this. Blending the chemistry proxy in as a prior
measured WORSE (-0.0127), because the proxy's error (0.00981) is twice the
measurement noise (0.00464). More measurement is the only lever.

Writes a `rescored` table; it never overwrites the original single-pass scores.
"""
import argparse
import json
import sqlite3
import statistics
import subprocess
import time
from pathlib import Path

BASE = Path('/root/bank')
VENV = BASE / 'venv'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default=str(BASE / 'bank.sqlite'))
    parser.add_argument('--reaction', type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help='must match the bank being rescored. Hardcoding it made\n'
                             '                        this silently rescore rxn 3 records against\n'
                             '                        another reaction\'s bank.')
    parser.add_argument('--shortlist', type=int, default=200)
    parser.add_argument('--overfetch', type=int, default=4,
                        help='pull this many times the shortlist before '
                             'filtering, so the shortlist is still full after '
                             'the validator gate removes what cannot ship.')
    parser.add_argument('--archive', default=str(BASE / 'P40261_molecules.csv'),
                        help='Metanova/Submission-Archive CSV. Rescoring an '
                             'unshippable molecule is pure waste: 63%% of the '
                             'last rescore went on molecules the validator '
                             'refuses.')
    parser.add_argument('--no-archive', dest='archive', action='store_const',
                        const='', help='rescore without pre-filtering.')
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--seeds', default='', help='comma-separated; defaults to 68,69,70...')
    parser.add_argument('--gpus', default='0,1,2,3')
    arguments = parser.parse_args()

    seeds = ([int(s) for s in arguments.seeds.split(',') if s.strip()]
             or [68 + i for i in range(arguments.passes)])
    connection = sqlite3.connect(arguments.bank, timeout=180)
    connection.execute('PRAGMA busy_timeout=180000')
    connection.execute(
        'CREATE TABLE IF NOT EXISTS rescored(name TEXT PRIMARY KEY, passes INT, '
        'mean_score REAL, sd_score REAL, seeds TEXT, single_pass_score REAL)')
    connection.commit()
    # Filter BEFORE spending the GPU, not after.
    #
    # The shortlist used to be the top N by single-pass score, and 247 of the
    # 394 we rescored turned out to be unshippable -- 63% of the compute, spent
    # on molecules the validator refuses. Worse, the filter is biased against
    # exactly the molecules that rank highest, so a score-ordered shortlist is
    # the WORST place to draw from. Over-fetch, drop what cannot ship, then
    # take the shortlist from what remains.
    candidates = connection.execute(
        'SELECT name,smiles,components,inchikey,heavy_atoms,rotatable_bonds,score '
        'FROM candidates WHERE score IS NOT NULL AND reaction = ? ORDER BY score DESC '
        'LIMIT ?', (arguments.reaction, arguments.shortlist * arguments.overfetch)
    ).fetchall()
    if arguments.archive:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from bank_release import (historical_fingerprints, refused_by_validator,
                                  validator_helpers)
        archive = Path(arguments.archive)
        if not archive.is_file():
            print(f'archive not found: {archive}; pass --no-archive to rescore '
                  f'without it (the validator will still apply the gate)',
                  file=sys.stderr)
            return 1
        history = historical_fingerprints(archive, Path(str(archive) + '.fps'))
        flagged = refused_by_validator([r[1] for r in candidates], history,
                                       validator_helpers())
        kept = [r for i, r in enumerate(candidates) if i not in flagged]
        print(json.dumps(dict(stage='prefilter', considered=len(candidates),
                              refused=len(flagged), kept=len(kept))), flush=True)
        candidates = kept
    rows = candidates[:arguments.shortlist]
    records = [dict(name=r[0], smiles=r[1], reaction=arguments.reaction,
                    components=json.loads(r[2]),
                    inchikey=r[3], heavy_atoms=r[4], rotatable_bonds=r[5]) for r in rows]
    single = {r[0]: r[6] for r in rows}
    inputs = BASE / 'rescore-records.json'
    inputs.write_text(json.dumps(dict(scores=records)))
    (BASE / 'scratch').mkdir(exist_ok=True)

    gathered = {}
    started = time.monotonic()
    for index, seed in enumerate(seeds):
        output = BASE / ('rescore-%d.json' % seed)
        command = [str(VENV / 'bin/python'), str(BASE / 'tools/score_boltz_batch.py'),
                   '--archive', str(BASE / 'empty.sqlite'), '--records-json', str(inputs),
                   '--preserve-record-order', '--reaction', str(arguments.reaction),
                   '--limit', str(len(records)),
                   '--target', 'P40261', '--msa', str(BASE / 'P40261.a3m'),
                   '--gpus', arguments.gpus, '--shards-per-gpu', '1', '--seed', str(seed),
                   '--output', str(output), '--scratch-root', str(BASE / 'scratch')]
        environment = dict(HOME='/root', PATH=str(VENV / 'bin') + ':/usr/local/bin:/usr/bin:/bin',
                           LANG='C.UTF-8', PYTHONPATH=str(BASE), OMP_NUM_THREADS='1')
        result = subprocess.run(command, env=environment, cwd=str(BASE),
                                capture_output=True, text=True, timeout=14400)
        if result.returncode or not output.is_file():
            print(json.dumps(dict(stage='failed', seed=seed, exit_code=result.returncode,
                                  tail=(result.stderr or result.stdout)[-300:])))
            return 1
        for record in json.loads(output.read_text())['scores']:
            gathered.setdefault(record['name'], []).append(record['score'])
        print(json.dumps(dict(stage='pass_complete', seed=seed, index=index + 1,
                              of=len(seeds), elapsed=round(time.monotonic() - started, 1))),
              flush=True)

    written = []
    for name, values in gathered.items():
        if not values:
            continue
        written.append((name, len(values), statistics.fmean(values),
                        statistics.pstdev(values) if len(values) > 1 else 0.0,
                        ','.join(str(s) for s in seeds), single.get(name)))
    connection.executemany(
        'INSERT OR REPLACE INTO rescored VALUES (?,?,?,?,?,?)', written)
    connection.commit()
    top_single = sorted(single.items(), key=lambda kv: -kv[1])[:20]
    by_mean = sorted(written, key=lambda r: -r[2])[:20]
    moved = len(set(n for n, _ in top_single) ^ set(r[0] for r in by_mean)) // 2
    print(json.dumps(dict(stage='complete', shortlist=len(records), passes=len(seeds),
                          rescored=len(written),
                          single_pass_sum20=round(sum(v for _, v in top_single), 6),
                          averaged_sum20=round(sum(r[2] for r in by_mean), 6),
                          molecules_changed_in_top20=moved,
                          mean_sd_across_passes=round(
                              statistics.fmean([r[3] for r in written]), 6),
                          elapsed=round(time.monotonic() - started, 1))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
