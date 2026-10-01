#!/usr/bin/env python3
"""Continuous molecule-bank builder. Runs on the dedicated L40S box.

Why a bank at all: best-20 portfolio score grows with the log of the bank size,
    best20 = 0.0818*ln(size) + 0.8344    (+0.057 per doubling)
measured by subsampling our own 7,827 exactly-scored molecules. Nothing else we
tried moved the number by more than hundredths. Molecules stay valid across
epochs -- the target never changes and unsubmitted molecules stay novel -- so the
bank compounds while per-epoch search does not.

How candidates are chosen, both measured on real molecules:
  * Fix (A,B) and sweep role C by descending Tanimoto to the anchor's own C.
    Similar swaps retained top-decile molecules at 3.29x the base rate; swaps
    below ~0.6 similarity retained at 0.83x -- worse than random -- so the list
    is truncated, not merely ordered. Varying C retains 4.58x against 2.85x for
    B and 1.11x for A, so C is the axis.
  * Rank the resulting analogues with the label-bank proxy, which beat blind
    nearest-K by 14% and random-C by 49% on 1,120 scored analogues.

Molecules that fail validity or the 0.5 historical novelty gate are dropped
before scoring: they can never be submitted, and scoring costs real money.

Everything is checkpointed. A run that dies at 60% keeps its 60%.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

BASE = Path('/root/bank')
VENV = BASE / 'venv'
sys.path.insert(0, str(BASE))
LABEL = 'oss-boltz-reference-bank'


def save(name, payload):
    path = BASE / name
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(path)


def open_bank(path):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS candidates(name TEXT PRIMARY KEY,smiles TEXT,reaction INT,"
        "components TEXT,inchikey TEXT,heavy_atoms INT,rotatable_bonds INT,"
        "affinity_probability_binary REAL,affinity_pred_value REAL,score REAL,scorer TEXT)")
    connection.commit()
    return connection


def load_validator_gate(archive: str):
    """(history fingerprints, helpers) for the validator's refusal rules.

    None when the archive is absent, so a builder on a box without it still
    runs -- it just proposes molecules the release will later drop.
    """
    if not archive:
        return None
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bank_release import (historical_fingerprints, refused_by_validator,
                              validator_helpers)
    globals()["refused_by_validator"] = refused_by_validator
    path = Path(archive)
    if not path.is_file():
        print(json.dumps(dict(stage='validator_gate_absent', archive=str(path))),
              flush=True)
        return None
    return (historical_fingerprints(path, Path(str(path) + '.fps')),
            validator_helpers())


def score_with_nim(batch, workers=8):
    """Score a batch against a local NIM server, same recipe as the epoch runner.

    Identical inference controls to the OSS path so the two boxes' labels are
    interchangeable: measured agreement between them is spearman 0.867 with a
    mean absolute difference of 0.0065 per molecule.
    """
    import concurrent.futures
    import requests
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from compound_miner.targets import TARGETS
    from analyze_nim19_pair import mw_correct
    alignment = (BASE / 'P40261.a3m').read_text()
    controls = dict(recycling_steps=3, sampling_steps=100, diffusion_samples=1,
                    sampling_steps_affinity=100, diffusion_samples_affinity=3,
                    affinity_mw_correction=False, step_scale=1.638,
                    max_parallel_samples=1, output_format='mmcif',
                    write_full_pae=False, write_full_pde=False)

    def one(candidate):
        request = dict(complex_id='bank_' + candidate.name.replace(':', '_'),
                       polymers=[dict(id='A', molecule_type='protein',
                                      sequence=TARGETS['P40261'],
                                      msa={'uniref90': {'a3m': dict(format='a3m',
                                           alignment=alignment, rank=0)}})],
                       ligands=[dict(id='B', smiles=candidate.smiles, predict_affinity=True)],
                       **controls)
        session = requests.Session(); session.trust_env = False
        for attempt in range(1, 4):
            try:
                response = session.post('http://127.0.0.1:8000/biology/mit/boltz2/predict',
                                        json=request, timeout=180)
                if response.status_code != 200:
                    raise RuntimeError('nim_http_status')
                affinity = response.json()['affinities']['B']
                scalar = lambda x: float(x[0] if isinstance(x, list) else x)
                probability = scalar(affinity['affinity_probability_binary'])
                raw = scalar(affinity['affinity_pred_value'])
                weight = Descriptors.MolWt(Chem.MolFromSmiles(candidate.smiles))
                prediction = mw_correct(raw, weight)
                value = (probability - prediction) / candidate.heavy_atoms
                if not 0 <= probability <= 1 or not all(map(math.isfinite,
                                                            [raw, prediction, value])):
                    raise RuntimeError('nim_nonfinite')
                return dict(name=candidate.name,
                            affinity_probability_binary=probability,
                            affinity_pred_value=prediction, score=value)
            except Exception:
                if attempt == 3:
                    return None
                time.sleep(attempt)

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(one, batch):
            if record is not None:
                out.append(record)
    return out


#: How strongly a prefix's own mean is pulled towards the bank's. A prefix seen
#: once tells us almost nothing, and its single score is mostly seed noise --
#: the per-molecule seed sd is 0.00297 and selecting a top-20 on single passes
#: costs 0.1335 against a three-pass mean. Shrinking by five pseudo-observations
#: keeps a lucky depth-1 draw from being mistaken for a good neighbourhood.
PREFIX_SHRINKAGE = 5.0


def expected_gain_weights(connection, reaction, prefixes, cut):
    """P(a fresh draw from each prefix beats the current top-20 cut).

    The weighting this replaces was `best_score(prefix) - floor`, linear above
    the pool's worst. Linear does not concentrate when the low-mean prefixes
    outnumber the high-mean ones by a hundred to one: measured on the live
    reaction-3 bank at 45,731 molecules, only 1.4% of anchor weight reached the
    fifteen prefixes that actually held the top 20, and 67% of the bank sat in
    13,044 prefixes at median depth 1 whose best score ever was 0.0539 against
    a cut of 0.0802 -- draws that could not enter the top 20 at all.

    Weighting by the chance of beating the cut moved that to 24% in the top
    fifteen and raised P(draw beats the cut) from 0.000111 to 0.002550: one new
    top-20 entrant per ~390 molecules instead of per ~9,000.

    It also fixes a second problem for free. Using the prefix's SHRUNK MEAN
    rather than its maximum means a neighbourhood that produced one lucky 0.09
    and nine draws at 0.03 is judged on the 0.03, so the budget stops chasing
    seed noise.

    Returns None when the bank cannot yet support the estimate -- a cold start
    has no within-prefix spread to measure -- and the caller keeps the linear
    weighting for those rounds.
    """
    import math
    import statistics

    groups = {}
    for components, score in connection.execute(
            'SELECT components,score FROM candidates '
            'WHERE score IS NOT NULL AND reaction = ?', (reaction,)):
        groups.setdefault(tuple(json.loads(components))[:-1], []).append(score)
    deep = [scores for scores in groups.values() if len(scores) >= 10]
    if len(deep) < 20:
        return None
    spread = math.sqrt(statistics.mean(statistics.pvariance(s) for s in deep))
    if not spread > 0.0:
        return None
    grand = statistics.mean(s for scores in groups.values() for s in scores)
    root_two = math.sqrt(2.0)
    weights = []
    for prefix in prefixes:
        scores = groups.get(prefix)
        if not scores:
            shrunk = grand
        else:
            count = len(scores)
            shrunk = ((count * statistics.mean(scores) + PREFIX_SHRINKAGE * grand)
                      / (count + PREFIX_SHRINKAGE))
        weights.append(max(1e-9, 0.5 * math.erfc((cut - shrunk) / spread / root_two)))
    return weights



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-total', type=int, default=82000)
    parser.add_argument('--batch', type=int, default=400)
    parser.add_argument('--anchors', type=int, default=120)
    parser.add_argument('--pool-multiple', type=float, default=7.0,
                        help='generate this many times the batch before filtering. '
                             'Measured pass rate through validity + the 0.5 novelty '
                             'gate is ~14%%, so 3x yielded only 170 of a 400 batch.')
    parser.add_argument('--anchor-pool', type=int, default=25000,
                        help='breadth of the pool anchors are sampled from')
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--max-consecutive-failures', type=int, default=8,
                        help='give up only after this many batches fail in a row')
    parser.add_argument('--hot-depth', type=int, default=48,
                        help='sweep depth for a freshly discovered high scorer')
    parser.add_argument('--hot-percentile', type=float, default=0.995,
                        help='a new molecule above this bank percentile becomes a hot lead')
    parser.add_argument('--coverage-quota', type=float, default=0.18,
                        help='share of the batch RESERVED for coverage draws. '
                             'Coverage loses the shared proxy ranking to sweeps, '
                             'so without a reserve its share drifts with sweep '
                             'supply. 0.18 matches the share it won historically.')
    parser.add_argument('--coverage-oversample', type=int, default=3,
                        help='draw this many coverage candidates per reserved '
                             'slot and score only the best-predicted. Measured on '
                             '1,315 held-out coverage draws the proxy ranks them '
                             'at +0.423 Spearman -- better than it ranks sweep '
                             'molecules (+0.281), because coverage spans a wide '
                             'quality range while analogues of a good anchor sit '
                             'in a narrow band. Keeping the best tenth lifts the '
                             'coverage mean from 0.01724 to 0.03063. Generation is '
                             'CPU work; the GPU scoring cost is unchanged.')
    parser.add_argument("--archive", default=str(BASE / "P40261_molecules.csv"),
                        help="Metanova/Submission-Archive CSV. Molecules the "
                             "validator would refuse are dropped BEFORE they "
                             "are scored; 23%% of the current bank cannot be "
                             "submitted and it is worst at the top.")
    parser.add_argument("--no-archive", dest="archive", action="store_const",
                        const="", help="propose without the validator gate.")

    parser.add_argument('--reaction', type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help='which reaction to bank. The validator draws one '
                             'uniformly per epoch (seed %% 5), so a bank is only '
                             'usable in the ~1 epoch in 5 that draws its reaction.')
    parser.add_argument('--coverage-max-draws', type=int, default=25000,
                        help='hard ceiling on uniform draws per round. Drawing '
                             'and gating is CPU work while the batch waits on '
                             'the GPU, so this bounds how long a round can '
                             'spend looking before it commits to scoring.')
    parser.add_argument('--minimum-similarity', type=float, default=0.5,
                        help='floor for role-C swap similarity to the anchor')
    parser.add_argument('--minimum-novelty', type=float, default=0.5,
                        help='historical Tanimoto gate, matching the validator')
    parser.add_argument('--bank', default=None,
                        help='defaults to bank.sqlite under the working root, '
                             'so --base alone is enough to keep two reactions '
                             'in separate banks')
    parser.add_argument('--seed', type=int, default=68,
                        help='RNG seed; give each box a different one so their '
                             'anchor sampling and coverage draws do not overlap')
    parser.add_argument('--scorer', choices=['oss', 'nim'], default='nim',
                        help="'nim' posts to a local NIM server instead of running "
                             "the OSS Boltz worker; use it on the NIM box so both "
                             "boxes fill the bank in parallel")
    parser.add_argument('--sweep-roles', default='last', choices=['last', 'all'],
                        help="which role the sweep substitutes. `last` holds "
                             "components[:-1] fixed and varies the final "
                             "reagent -- the only axis this tool has ever "
                             "searched. `all` also varies each earlier role "
                             "with the others held fixed. The epoch-25226 "
                             "winner submitted rxn:3:61524:*:98582, varying "
                             "the MIDDLE role of three: a portfolio `last` "
                             "cannot generate at all.")
    parser.add_argument('--anchor-weighting', default='expected-gain',
                        choices=['expected-gain', 'linear'],
                        help="how the sweep's anchor budget is spread over "
                             "neighbourhoods. `linear` is the original "
                             "best-score-above-a-floor rule, kept so the change "
                             "can be turned off on a box without editing code; "
                             "it puts 1.4%% of the budget where the top 20 "
                             "actually lives. `expected-gain` weights by the "
                             "chance of beating the current cut, from each "
                             "prefix's shrunk mean.")
    parser.add_argument('--python', default=None,
                        help='interpreter that runs score_boltz_batch.py. The '
                             'default is <base>/venv/bin/python, which is the '
                             'hand-built venv the first two bank boxes carry. A '
                             'box provisioned by ansible/provision.yml already '
                             'has the locked bz2 environment and reaches it '
                             'through /root/search_python.sh, which also sets '
                             'BOLTZ_CACHE and the vendored-boltz PYTHONPATH -- '
                             'copying a 7.3 GB venv to duplicate it is not '
                             'worth the transfer.')
    parser.add_argument('--base', default=None,
                        help='override the working root, so several reactions '
                             'can be banked side by side without their state '
                             'files colliding')
    arguments = parser.parse_args()
    if arguments.base:
        # BASE is read by save(), the proxy/alignment/novelty loaders and the
        # scorer hand-off, so it has to be rebound before any of them run.
        # Assigning through globals() rather than a `global` statement, because
        # BASE is already read above to build the --bank default, and a `global`
        # declaration after that first read is a SyntaxError.
        globals()['BASE'] = Path(arguments.base)
        globals()['VENV'] = Path(arguments.base) / 'venv'
    if not arguments.bank:
        arguments.bank = str(BASE / 'bank.sqlite')

    # OSS is not an option for compound banking. Measured 2026-09-21 on an
    # RTX 6000 Ada: OSS 127 molecules/GPU-hour against NIM's 514 -- a 4x gap,
    # which on a 58,754-molecule bank is 46 hours versus 11, and ~$190 of
    # rental. A slow scorer is not a cheaper scorer, it is the same work for
    # four times the money.
    if arguments.scorer != 'nim':
        print('refusing to build a compound bank with the OSS scorer: it is '
              '4x slower than NIM on identical hardware (127 vs 514 '
              'molecules/GPU-hour). Bring up the local NIM server with '
              'ansible/nim_native.yml and use --scorer nim.', file=sys.stderr)
        return 2

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from compound_miner.bank_proxy import BankProxy
    from compound_miner.database import ReactionDatabase
    from compound_miner.rowcolumn_strategy import tanimoto_reagent_ranker
    from compound_miner.validity import HistoricalIndex, ValidityGate

    reaction = arguments.reaction
    bank_path = Path(arguments.bank)
    seeded = 0
    if not bank_path.exists():
        seed = BASE / 'seed-bank.sqlite'
        if seed.is_file():
            # Copy only this reaction's rows. The seed bank holds rxn 3 labels;
            # byte-copying it into a rxn 1 bank would make every anchor, every
            # `seen` inchikey and the whole scaling series refer to chemistry
            # that cannot be built from rxn 1's reagents.
            connection = open_bank(bank_path)
            source = sqlite3.connect(str(seed))
            rows = source.execute(
                'SELECT name,smiles,reaction,components,inchikey,heavy_atoms,'
                'rotatable_bonds,affinity_probability_binary,affinity_pred_value,'
                'score,scorer FROM candidates WHERE reaction=?', (reaction,)).fetchall()
            source.close()
            connection.executemany(
                'INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
            connection.commit()
            seeded = len(rows)
            connection.close()
    connection = open_bank(bank_path)
    validator_gate = load_validator_gate(arguments.archive)
    save('builder.json', dict(stage='starting', reaction=reaction, seeded=seeded,
                              bank=str(bank_path)))

    database = ReactionDatabase()
    roles = database.reactions[reaction].roles
    # The sweep holds every role but the last fixed and substitutes the last.
    # rxn 3 and 5 have three roles, rxn 1, 2 and 4 have two, so the "pair" that
    # a sweep exhausts is components[:-1] -- a pair for a 3-role reaction, a
    # single reagent for a 2-role one. Naming it `prefix` keeps the same lane
    # working for both instead of assuming (A, B, C).
    sweep_role = roles[-1]
    proxy = BankProxy.load(BASE / 'bank-proxy-v2.joblib')
    # Two deliberate details. main's HistoricalIndex takes no maximum_similarity
    # (the deployed fleet copy does), and main's ValidityGate.accepts() calls
    # is_novel() with its 0.7 DEFAULT rather than the validator's 0.5. Applying
    # accepts() unchanged would admit molecules the real gate rejects, i.e. spend
    # money scoring chemistry we could never submit. So the chemical filters and
    # the novelty threshold are applied separately and explicitly.
    history = HistoricalIndex(BASE / 'novelty.csv')
    gate = ValidityGate(history)
    novelty_threshold = arguments.minimum_novelty
    # The ranker fingerprints ONE role's reagents, and its defaults are rxn 3's
    # last role. Left at those defaults it returns [] for every other reaction,
    # because the reference id is absent from its fingerprint table -- which
    # silently disables the sweep lane rather than raising.
    # One ranker per swept role. Each fingerprints that role's whole reagent
    # set, so they are built only for the roles actually in use -- `last` pays
    # for exactly one, as before.
    swept_roles = (tuple(range(len(roles))) if arguments.sweep_roles == 'all'
                   else (len(roles) - 1,))
    rankers = {index: tanimoto_reagent_ranker(database, reaction=reaction,
                                              role_index=index)
               for index in swept_roles}
    ranker = rankers[len(roles) - 1]
    rng = random.Random(arguments.seed)

    # Inlined rather than imported: the box's layout puts this file under
    # /root/bank/tools, and a failed import here would kill the builder at
    # startup for no good reason. Must match score_boltz_batch.record_id.
    def _record_id(smiles, seed=68):
        digest = hashlib.sha256(smiles.encode()).digest()
        return (int.from_bytes(digest[:8], 'little') ^ seed) % (2 ** 31 - 1)
    blacklist_path = BASE / 'blacklist.json'
    blacklist = set(json.loads(blacklist_path.read_text())) if blacklist_path.is_file() else set()
    # Exhausted pairs must survive a restart. Held only in memory, every restart
    # re-tried pairs whose entire similarity neighbourhood was already scored,
    # which is why batches collapsed to 11 molecules after the last restart.
    exhausted_path = BASE / 'exhausted-pairs.json'
    failures = 0
    started = time.monotonic()
    rounds = 0
    exhausted_pairs = set(tuple(x) for x in json.loads(exhausted_path.read_text())) \
        if exhausted_path.is_file() else set()
    # Hot leads: molecules just discovered that rank near the top of the bank.
    # P(neighbour in top decile | anchor in top decile) is 4.58x the base rate,
    # so a fresh high find is the most valuable anchor we will ever hold -- but
    # dropped into a 4,000-molecule weighted pool it would only be sampled a few
    # percent of the time per round. Sweep it immediately and at full depth
    # instead, then retire it to the ordinary pool.
    hot = []
    coverage_survival = 0.058
    last_sweep_kept = 0
    while True:
        total = connection.execute(
            'SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL').fetchone()[0]
        if total >= arguments.target_total:
            save('builder.json', dict(stage='target_reached', total=total,
                                      elapsed=round(time.monotonic() - started, 1)))
            return 0
        rounds += 1
        round_started = time.monotonic()
        seen = {r[0] for r in connection.execute('SELECT inchikey FROM candidates')}

        # Anchor rotation. Taking the top-N by score alone locked the sweep onto
        # a handful of (A,B) pairs: at round 58 the top 120 molecules held only
        # 50 distinct pairs, so the ~800 reachable tuples were consumed and the
        # batch collapsed from 400 to 10. Sample from a much wider pool instead,
        # weighted towards quality but not confined to it, retiring pairs that
        # stop yielding.
        pool_rows = connection.execute(
            'SELECT components,score FROM candidates WHERE score IS NOT NULL '
            'ORDER BY score DESC LIMIT ?', (arguments.anchor_pool,)).fetchall()
        by_pair = {}
        for components, score in pool_rows:
            parts = tuple(json.loads(components))
            by_pair.setdefault(parts[:-1], (parts, score))
        pairs = sorted(by_pair.values(), key=lambda kv: -kv[1])
        hot_anchors = [h for h in hot if h[0][:-1] not in exhausted_pairs]
        hot = []
        fresh = [x for x in pairs if x[0][:-1] not in exhausted_pairs]
        if not fresh:
            exhausted_pairs.clear()
            fresh = pairs
        if not fresh:
            # A bank with no scored molecule yet has nothing to sweep FROM: the
            # sweep proposes analogues of what already scored well. Round one of
            # a new reaction therefore runs on the coverage lane alone, and the
            # sweep starts once there is something to anchor on. This only ever
            # happens cold, which is why it went unnoticed -- the first two
            # boxes were seeded from an existing bank, so `fresh` was never
            # empty and `fresh[-1]` never raised.
            anchors = []
            weighting = 'none'
        else:
            floor = fresh[-1][1]
            weights = [max(1e-6, s - floor + 1e-3) for _, s in fresh]
            weighting = 'linear'
            if arguments.anchor_weighting == 'expected-gain':
                cut_row = connection.execute(
                    'SELECT score FROM candidates WHERE score IS NOT NULL '
                    'AND reaction = ? ORDER BY score DESC LIMIT 1 OFFSET 19',
                    (reaction,)).fetchone()
                if cut_row is not None:
                    gain = expected_gain_weights(
                        connection, reaction,
                        [tuple(components[:-1]) for components, _ in fresh],
                        float(cut_row[0]))
                    if gain is not None:
                        weights = gain
                        weighting = 'expected-gain'
            chosen = set()
            guard = 0
            while (len(chosen) < min(arguments.anchors, len(fresh))
                   and guard < arguments.anchors * 50):
                guard += 1
                chosen.add(rng.choices(range(len(fresh)), weights=weights, k=1)[0])
            anchors = hot_anchors + [fresh[i] for i in sorted(chosen)][
                :max(0, arguments.anchors - len(hot_anchors))]
        proposals = {}
        for components, _ in anchors:
            # A hot lead is swept deeper than a routine anchor.
            depth = (arguments.hot_depth
                     if any(components == h[0] for h in hot_anchors) else arguments.depth)
            for index in swept_roles:
                swept = components[index]
                alphabet = [int(x) for x in database.role_ids[roles[index]]]
                for identifier in rankers[index](swept, alphabet,
                                                 arguments.minimum_similarity)[:depth]:
                    substituted = (tuple(components[:index]) + (identifier,)
                                   + tuple(components[index + 1:]))
                    candidate = database.candidate(reaction, substituted)
                    if candidate is None or candidate.inchikey in seen:
                        continue
                    proposals[candidate.inchikey] = candidate
        # A uniform lane keeps reagents absent from the bank reachable, and is
        # the ONLY source of fresh (A,B) pairs -- a sweep holds (A,B) fixed, so
        # without coverage the bank caps out near 100k.
        #
        # It used to share one pool and one proxy ranking with the sweep. Two
        # things went wrong with that. Coverage must be sized against the BATCH
        # we want to fill, not against the sweep's output: sizing it as a
        # fraction of `proposals` meant that when sweeps dried up -- precisely
        # when fresh pairs are needed -- coverage dried up too and batches
        # collapsed to 11-38 molecules. And competing head-to-head with sweep
        # analogues let coverage's batch share drift with sweep supply rather
        # than with how many new pairs we actually need.
        #
        # So coverage now gets a reserved quota, filled from a deliberately
        # over-generated pool. Uniform draws clear the 0.5 novelty gate about
        # 82% of the time and construction is CPU-bound, so generating ten
        # candidates per reserved slot costs no GPU: only the best-predicted
        # tenth is scored.
        coverage_slots = max(1, int(arguments.batch * arguments.coverage_quota))
        # Draw against SURVIVORS, not draws. Measured in production: uniform
        # draws clear validity+novelty only 5.8% of the time, not the ~82% an
        # earlier comment here claimed. Drawing `slots * oversample` therefore
        # left every survivor needed to fill the batch, so the proxy had nothing
        # to choose between and the over-generation was inert. Track the
        # observed rate and draw enough that selection actually bites.
        # Size against the slots coverage will actually be asked to fill. When
        # the sweep is healthy that is just the quota; when it collapses,
        # coverage has to carry the whole batch and must draw accordingly.
        needed = (max(coverage_slots, arguments.batch - last_sweep_kept)
                  * max(1, arguments.coverage_oversample))
        wanted = min(arguments.coverage_max_draws,
                     int(needed / max(coverage_survival, 0.005)))
        coverage = {}
        attempts = 0
        while wanted > 0 and attempts < max(wanted, 1) * 80:
            attempts += 1
            candidate = database.candidate(reaction, tuple(
                rng.choice(database.role_ids[role]) for role in roles))
            if candidate is None or candidate.inchikey in seen \
                    or candidate.inchikey in proposals or candidate.inchikey in coverage:
                continue
            coverage[candidate.inchikey] = candidate
            wanted -= 1

        # Retire anchors that yielded nothing so the next round anchors
        # elsewhere. The key is the neighbourhood the LAST role sweeps, which is
        # what `fresh` filters on; an anchor is only retired when no role
        # produced anything, because role 0 still having ground left is a reason
        # to come back even if role 2 is mined out.
        for components, _ in anchors:
            pair = tuple(components[:-1])
            if not any(tuple(c.components[:-1]) == pair for c in proposals.values()):
                exhausted_pairs.add(pair)
                exhausted_path.write_text(json.dumps([list(x) for x in sorted(exhausted_pairs)]))
        values = list(proposals.values()) + list(coverage.values())
        if not values:
            save('builder.json', dict(stage='exhausted', total=total, round=rounds))
            return 0
        # accepts() applies the chemical filters AND the 0.5 historical novelty
        # gate. Molecules failing it can never be submitted, so scoring them
        # would be money spent on chemistry we could not use.
        def survives(candidate):
            return (gate.accepts(candidate, check_similarity=False)
                    and history.is_novel(candidate, novelty_threshold))
        kept = [c for c in values if survives(c)]
        # Lane accounting. The two lanes survive the validity+novelty gate at
        # very different rates -- a sweep analogue inherits an anchor that
        # already passed, a uniform draw has to pass on its own -- so a single
        # `kept` number cannot tell a starved sweep from an over-generated
        # coverage pool, and the two call for opposite fixes.
        kept_cover = sum(1 for c in kept if c.inchikey in coverage)
        if len(coverage) >= 200:
            observed = kept_cover / len(coverage)
            # EWMA: the rate drifts as the historical archive grows and as the
            # bank absorbs the easy chemistry, so track it rather than pin it.
            coverage_survival = 0.7 * coverage_survival + 0.3 * max(observed, 0.002)
        last_sweep_kept = len(kept) - kept_cover
        if not kept:
            save('builder.json', dict(stage='no_valid_candidates', round=rounds, total=total,
                                      proposed=len(values)))
            return 1
        # Rank the two lanes separately and compose the batch, so the coverage
        # quota buys the best of its over-generated pool instead of whatever
        # happens to out-rank a sweep analogue. Each lane spills into the other
        # when it is short, which keeps the batch full when sweeps dry up.
        ranked = sorted(proxy.predict(kept), key=lambda c: -c.proxy_mean)
        cover_keys = set(coverage)
        cover_ranked = [c for c in ranked if c.inchikey in cover_keys]
        sweep_ranked = [c for c in ranked if c.inchikey not in cover_keys]
        take_cover = min(coverage_slots, len(cover_ranked))
        take_sweep = min(arguments.batch - take_cover, len(sweep_ranked))
        take_cover = min(len(cover_ranked), arguments.batch - take_sweep)
        batch = sweep_ranked[:take_sweep] + cover_ranked[:take_cover]
        batch.sort(key=lambda c: -c.proxy_mean)

        batch = [c for c in batch if str(_record_id(c.smiles)) not in blacklist]
        # Never PROPOSE what the validator will refuse.
        #
        # Measured on the 58,754-molecule rxn-3 bank: 23% of it can never be
        # submitted, and the damage is concentrated exactly where we select --
        # only 35% of the top 500 survives, against 83% of the tail, because
        # high scores come from the scaffolds everyone has already submitted.
        # Every refused molecule here is GPU time spent on something that can
        # never ship, and it also crowds the shortlist the rescore draws from.
        if validator_gate is not None:
            # NOT `history`: that name already holds the HistoricalIndex bound
            # above, and rebinding it here turned the novelty gate into a list
            # for every subsequent round -- the builder died on
            # "'list' object has no attribute 'is_novel'" at the start of
            # round 2, after banking exactly one batch.
            gate_history, gate_helpers = validator_gate
            refused = refused_by_validator([c.smiles for c in batch],
                                           gate_history, gate_helpers)
            if refused:
                reasons = {}
                for why in refused.values():
                    key = str(why).split()[0]
                    reasons[key] = reasons.get(key, 0) + 1
                batch = [c for i, c in enumerate(batch) if i not in refused]
                save('builder.json', dict(stage='validator_gate',
                                          refused=len(refused), kept=len(batch),
                                          reasons=reasons))
        if not batch:
            save('builder.json', dict(stage='batch_empty_after_blacklist', round=rounds,
                                      total=total, blacklist_size=len(blacklist)))
            continue
        # `reaction` here was the literal 3, left from when this tool only built
        # one bank. score_boltz_batch filters its input by reaction, so a
        # reaction-2 batch labelled 3 matched nothing and the scorer exited with
        # "archive shortlist is empty" -- the builder then counted it as a batch
        # failure and gave up after three, having scored nothing at all.
        records = [dict(name=c.name, smiles=c.smiles, reaction=reaction,
                        components=list(c.components), inchikey=c.inchikey,
                        heavy_atoms=c.heavy_atoms, rotatable_bonds=c.rotatable_bonds)
                   for c in batch]
        inputs = BASE / 'batch-records.json'
        inputs.write_text(json.dumps(dict(scores=records)))
        output = BASE / 'batch-scores.json'
        if output.exists():
            output.unlink()
        (BASE / 'scratch').mkdir(exist_ok=True)
        save('builder.json', dict(stage='scoring', round=rounds, total=total,
                                  batch=len(records), proposed=len(values),
                                  kept=len(kept), anchor_pairs=len(anchors),
                                  search_seconds=round(time.monotonic() - round_started, 1),
                                  coverage_survival=round(coverage_survival, 4),
                                  sweep_proposed=len(proposals),
                                  cover_proposed=len(coverage),
                                  sweep_kept=len(kept) - kept_cover,
                                  cover_kept=kept_cover,
                                  batch_cover=sum(1 for c in batch
                                                  if c.inchikey in coverage),
                                  weighting=weighting,
                                  swept_roles=len(swept_roles),
                                  exhausted_pairs=len(exhausted_pairs),
                                  elapsed=round(time.monotonic() - started, 1)))
        # search_python.sh exports its own PYTHONPATH for the vendored boltz, and
        # a PYTHONPATH passed here would replace it rather than add to it, so the
        # bank's own path is appended instead of assigned.
        interpreter = arguments.python or str(VENV / 'bin/python')
        python_path = str(BASE)
        if arguments.python:
            python_path = str(BASE) + ':/root/external_tools/boltz/src'
        environment = dict(HOME='/root', PATH=str(VENV / 'bin') + ':/usr/local/bin:/usr/bin:/bin',
                           LANG='C.UTF-8', PYTHONPATH=python_path, OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0,1,2,3',
                           BOLTZ_CACHE='/root/.boltz',
                           CUDA_CACHE_MAXSIZE='4294967296')
        command = [interpreter, str(BASE / 'tools/score_boltz_batch.py'),
                   '--archive', str(BASE / 'empty.sqlite'), '--records-json', str(inputs),
                   '--preserve-record-order', '--reaction', str(reaction),
                   '--limit', str(len(records)),
                   '--target', 'P40261', '--msa', str(BASE / 'P40261.a3m'),
                   '--gpus', '0,1,2,3', '--shards-per-gpu', '1',
                   '--output', str(output), '--scratch-root', str(BASE / 'scratch')]
        if arguments.scorer == 'nim':
            scored_records = score_with_nim(batch)
            output.write_text(json.dumps(dict(scores=scored_records,
                                              count=len(scored_records))))
            result = type('R', (), dict(returncode=0, stdout='', stderr=''))()
        else:
            result = subprocess.run(command, env=environment, cwd=str(BASE),
                                    capture_output=True, text=True, timeout=14400)
        if result.returncode or not output.is_file():
            # A single molecule Boltz cannot emit a prediction for used to kill
            # the whole run: both boxes died on record 844044928 and sat idle for
            # hours. Blacklist the offender, skip the batch, and carry on --
            # never exit on one bad molecule.
            text = (result.stderr or result.stdout or '')
            import re
            bad = set(re.findall(r'predictions/(\d+)_', text))
            if bad:
                blacklist.update(bad)
                blacklist_path.write_text(json.dumps(sorted(blacklist)))
            failures += 1
            save('builder.json', dict(stage='batch_skipped', round=rounds, total=total,
                                      exit_code=result.returncode,
                                      blacklisted=sorted(bad), blacklist_size=len(blacklist),
                                      consecutive_failures=failures,
                                      tail=text[-300:]))
            # Also to the log. builder.json holds one record, so the final
            # `failed` write overwrote every tail that led to it: a cold start
            # that never scores a molecule left `reason=too_many_consecutive_-
            # failures` and no cause at all. The log is append-only.
            print(json.dumps(dict(event='batch_failed', round=rounds,
                                  exit_code=result.returncode,
                                  consecutive_failures=failures,
                                  tail=text[-1200:])), flush=True)
            if failures >= arguments.max_consecutive_failures:
                save('builder.json', dict(stage='failed', reason='too_many_consecutive_failures',
                                          total=total, blacklist_size=len(blacklist)))
                return 1
            continue
        failures = 0
        scored = json.loads(output.read_text())['scores']
        by_name = {c.name: c for c in batch}
        rows = []
        for record in scored:
            candidate = by_name.get(record['name'])
            if candidate is None:
                continue
            rows.append((candidate.name, candidate.smiles, reaction, json.dumps(list(candidate.components)),
                         candidate.inchikey, candidate.heavy_atoms, candidate.rotatable_bonds,
                         record.get('affinity_probability_binary'), record.get('affinity_pred_value'),
                         record.get('score'),
                         'nim19-mw-corrected-search-only' if arguments.scorer == 'nim' else LABEL))
        connection.executemany(
            'INSERT OR IGNORE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?)', rows)
        connection.commit()
        total = connection.execute(
            'SELECT COUNT(*) FROM candidates WHERE score IS NOT NULL').fetchone()[0]
        # Promote this round's exceptional finds. The threshold is a bank
        # percentile, so it tightens automatically as the bank improves.
        cut = connection.execute(
            'SELECT score FROM candidates WHERE score IS NOT NULL ORDER BY score DESC '
            'LIMIT 1 OFFSET ?', (max(0, int(total * (1 - arguments.hot_percentile))),)).fetchone()
        threshold = cut[0] if cut else None
        if threshold is not None:
            for record in scored:
                candidate = by_name.get(record['name'])
                if candidate is not None and (record.get('score') or -9) >= threshold:
                    hot.append((tuple(candidate.components), record['score']))
        best = connection.execute(
            'SELECT score FROM candidates WHERE score IS NOT NULL ORDER BY score DESC LIMIT 20').fetchall()
        save('builder.json', dict(stage='round_complete', round=rounds, total=total,
                                  added=len(rows), best20_sum=round(sum(x[0] for x in best), 6),
                                  target=arguments.target_total,
                                  elapsed=round(time.monotonic() - started, 1),
                                  seconds_per_molecule=round(
                                      (time.monotonic() - started) / max(1, total - 7827), 3)
                                  if total > 7827 else None))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        save('builder.json', dict(stage='failed', error_class=type(error).__name__,
                                  reason=str(error)[:300]))
        raise
