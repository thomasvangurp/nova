#!/usr/bin/env python3
"""Emit artifacts/result_rxn{N}.json for one epoch, from the molecule bank.

The compound miner on deploy/compound-nanobody-live already reads this file,
re-checks novelty against the live archive, repairs stale picks from spares and
commits `molecules|nanobodies` with the nanobody as the slave section. None of
that is rebuilt here. The only thing missing was a producer that feeds it from a
pre-built bank instead of a per-epoch search.

Three constraints are not negotiable.

*Only rows from the `rescored` table are eligible.* The bank stores two scorer
labels, and neither is the validator-exact string -- but the OSS rows were
produced by score_boltz_batch at its defaults, which emit exactly
`boltz-2.2.1-r3-s100-d1-ar5-as100-ad3-skip0-skipconf0`, while the NIM rows were
not. Rather than relabel anything, this takes only molecules that bank_rescore
re-measured with the exact scorer. That is honest for both origins and it is
also where the quality is: selecting on a 3-pass mean instead of a single pass
is worth +0.0817, more than doubling the bank.

*The payload carries spares.* `_still_novel` repairs a submission whose picks
have gone stale, but only from molecules present in the file. Our own previous
submissions are what make them stale -- the archive gate rejects anything within
0.5 Tanimoto, and this bank is built by sweeping one role at >=0.5 similarity,
so every epoch we submit burns its own neighbourhood.

*The epoch is stamped.* validate_release_payload compares the manifest's epoch
against the live challenge, so this must be regenerated for the epoch it is
submitted in; a file left over from a previous epoch is refused, not silently
reused.

*The pool is sized for the fleet, not for one uid.* Splitting the bank across
uids is not this program's job: compound_miner.fleet.allocate_compound_fleet_-
from_release already binds one release to the live uid+hotkey roster, giving
every member a shared core of the best 12 and its own tail of 8 plus 4 spares.
All this has to do is carry enough validator-exact rows for that allocation --
12 + k*(8+4), which is 132 for ten uids, against the 60 a single uid needs.
Carrying fewer is not a smaller fleet, it is `insufficient_round_fleet_pool`
and no compound submission at all.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

BASE = Path('/root/bank')
VALIDATOR_EXACT_SCORER = "boltz-2.2.1-r3-s100-d1-ar5-as100-ad3-skip0-skipconf0"
RELEASE_SCHEMA_VERSION = 1
MINIMUM_ENTROPY = 0.25
MOLECULE_CONFORMER_ACTIVATION = 25955


# Mirrors compound_miner.fleet_contract.required_exact_pool_size. It is mirrored
# rather than imported because this runs on the bank box, which holds the bank
# and not the miner tree; tests assert the two agree wherever both are present.
FLEET_SHARED_CORE = 12
FLEET_SPARES_PER_UID = 4


# Read from the validator, not inferred: neurons/validator/molecule_validity.py
# fingerprints every historical submission with Morgan radius=2 fpSize=2048 and
# refuses any molecule whose Tanimoto to ANY of them reaches
# config['max_similarity_to_historical'], which config/config.yaml sets to 0.5.
# The history is the public HF dataset Metanova/Submission-Archive, file
# "{target}_molecules.csv" -- 102,022 rows for P40261.
HISTORY_DATASET = ("https://huggingface.co/datasets/Metanova/Submission-Archive"
                   "/resolve/main/P40261_molecules.csv")
MAX_SIMILARITY_TO_HISTORICAL = 0.5
HISTORY_RADIUS = 2
HISTORY_BITS = 2048


def refresh_archive(archive: Path, url: str = HISTORY_DATASET,
                    timeout: int = 150) -> dict:
    """Re-download the submission archive, keeping the cached copy on failure.

    The archive grows every epoch, because it is everyone's submissions. A
    frozen copy silently loosens our filter over time: a molecule that sits at
    0.49 today goes over 0.5 the moment someone else submits near it, and we
    are back to a whole-fleet zero with no error anywhere.

    Never fatal. A network blip must cost us a slightly stale filter, not the
    epoch -- so every failure path returns the cached file.
    """
    import shutil
    import urllib.request

    previous = archive.stat().st_size if archive.is_file() else 0
    scratch = archive.with_suffix(archive.suffix + '.download')
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with scratch.open('wb') as handle:
                shutil.copyfileobj(response, handle)
        # Validate BEFORE replacing: a truncated or error-page download that
        # overwrites a good archive would disable the gate rather than fail.
        import pandas as pd
        frame = pd.read_csv(scratch, usecols=['SMILES'])
        rows = len(frame)
        if rows < 1000 or scratch.stat().st_size < previous * 0.9:
            scratch.unlink(missing_ok=True)
            return dict(refreshed=False, why='downloaded archive looks truncated',
                        rows=rows, kept_bytes=previous)
        scratch.replace(archive)
        return dict(refreshed=True, rows=rows, bytes=archive.stat().st_size,
                    grew_by=archive.stat().st_size - previous)
    except Exception as error:
        scratch.unlink(missing_ok=True)
        return dict(refreshed=False, why=f'{type(error).__name__}: {error}'[:120],
                    using_cached=archive.is_file(), kept_bytes=previous)


def historical_fingerprints(archive: Path, cache: Path | None = None):
    """Morgan fingerprints for every molecule ever submitted for this target.

    Cached because it is ~40s for 102k rows and the release is cut inside the
    commit window, where 40s is a third of the producer slack.
    """
    import pickle
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    import pandas as pd

    stamp = f"{archive.stat().st_size}:{int(archive.stat().st_mtime)}"
    if cache and cache.is_file():
        try:
            held = pickle.loads(cache.read_bytes())
            if held.get("stamp") == stamp:
                return held["fingerprints"]
        except Exception:
            pass
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=HISTORY_RADIUS, fpSize=HISTORY_BITS)
    frame = pd.read_csv(archive, usecols=["SMILES"])
    molecules = [Chem.MolFromSmiles(v) for v in frame["SMILES"].dropna().astype(str)]
    molecules = [m for m in molecules if m is not None]
    fingerprints = list(generator.GetFingerprints(molecules, numThreads=0))
    if cache:
        try:
            cache.write_bytes(pickle.dumps(
                dict(stamp=stamp, fingerprints=fingerprints)))
        except OSError:
            pass
    return fingerprints


VALIDATOR_TREE = Path("/opt/nova-submit/nova_validator")


def validator_helpers(tree: Path = VALIDATOR_TREE):
    """The validator's OWN per-molecule checks, or None if its tree is absent.

    Imported rather than reimplemented. Every rule I re-derived from logs this
    week drifted from the real one -- the similarity threshold alone was read
    as 0.7, then 0.6, and is 0.5 -- so the rules that decide what we ship come
    from their code.
    """
    if not (tree / "neurons/validator/molecule_validity.py").is_file():
        return None
    # Exact source extraction avoids an already-imported, stale `utils`
    # package and does not execute that module's dotenv/application startup.
    spec = importlib.util.spec_from_file_location('_bank_molecule_gate', Path(__file__).with_name('molecule_gate.py'))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    helpers = gate.load_helpers(tree / 'utils/molecules.py')
    import yaml
    config = yaml.safe_load((tree / "config/config.yaml").read_text())
    helpers['requirements'] = config['molecule_requirements']
    return helpers


def refused_by_validator(smiles_list, fingerprints, helpers=None):
    """Every reason the validator would refuse a molecule, not just similarity.

    Applied at SELECTION, not only as a pre-flight. A pre-flight that fails
    closed protects us from shipping junk but costs the whole epoch; filtering
    here means selection never picks the molecule in the first place. Measured
    on this bank: 1 of 394 rescored molecules fails BRENK, so the pre-flight
    tripwire was one unlucky draw away from an empty epoch.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, FilterCatalog, rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=HISTORY_RADIUS, fpSize=HISTORY_BITS)
    brenk = None
    if helpers:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        brenk = FilterCatalog.FilterCatalog(params)
        wanted = helpers["requirements"]

    flagged = {}
    for index, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            flagged[index] = "unparseable"
            continue
        if helpers:
            if helpers["heavy"](smiles) < wanted["min_heavy_atoms"]:
                flagged[index] = "heavy_atoms"; continue
            if brenk.HasMatch(molecule):
                flagged[index] = "brenk"; continue
            if helpers["banned"](molecule, wanted["banned_atom_types"]):
                flagged[index] = "banned_atom"; continue
            bonds = int(Descriptors.NumRotatableBonds(molecule))
            if bonds < wanted["min_rotatable_bonds"] or bonds > wanted["max_rotatable_bonds"]:
                flagged[index] = "rotatable_bonds"; continue
            safe, _why = helpers["boltz"](smiles)
            if not safe:
                flagged[index] = "boltz_unsafe"; continue
        worst = max(DataStructs.BulkTanimotoSimilarity(
            generator.GetFingerprint(molecule), fingerprints), default=0.0)
        if worst >= MAX_SIMILARITY_TO_HISTORICAL:
            flagged[index] = f"similarity {worst:.3f}"
    return flagged


def too_close_to_history(smiles_list, fingerprints):
    """Indices of molecules the validator would refuse, and how close they are.

    One is enough to lose everything: molecule_validity breaks out of the loop
    on the first failure and then skips the whole submission because
    len(valid_smiles) != num_molecules. At 25248 a single molecule at 0.811
    took out all ten uids, because the fleet allocator puts a shared core in
    every one of them.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=HISTORY_RADIUS, fpSize=HISTORY_BITS)
    flagged = {}
    for index, smiles in enumerate(smiles_list):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            flagged[index] = 1.0
            continue
        worst = max(DataStructs.BulkTanimotoSimilarity(
            generator.GetFingerprint(molecule), fingerprints), default=0.0)
        if worst >= MAX_SIMILARITY_TO_HISTORICAL:
            flagged[index] = worst
    return flagged


def required_pool_size(fleet: int, count: int, spares: int) -> int:
    """Validator-exact rows a release must carry for a fleet of `fleet` uids."""
    if fleet <= 1:
        return count + spares
    tail = count - FLEET_SHARED_CORE
    return FLEET_SHARED_CORE + fleet * (tail + FLEET_SPARES_PER_UID)


def atompair_entropy(smiles_list) -> float:
    """AtomPair 2048-bit mean per-bit entropy -- the metric the network gates on.

    Deliberately the same statistic as compound_miner.provider._entropy. The
    MACCS variant in utils/molecules.py is a different number and gating on it
    would pass submissions the network then skips.
    """
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg
    generator = rfg.GetAtomPairGenerator(fpSize=2048)
    counts = np.zeros(2048)
    valid = 0
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            counts += np.array(generator.GetFingerprint(molecule), dtype=np.float64)
            valid += 1
    if not valid:
        return 0.0
    p = counts / valid
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = np.where((p > 0) & (p < 1),
                        -p * np.log2(p) - (1 - p) * np.log2(1 - p), 0.0)
    return float(np.mean(bits))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default=str(BASE / 'bank.sqlite'))
    parser.add_argument('--reaction', type=int, default=3)
    parser.add_argument('--target', default='P40261')
    parser.add_argument('--archive', default=str(BASE / 'P40261_molecules.csv'),
                        help='Metanova/Submission-Archive CSV for this target. '
                             'The validator refuses any molecule within '
                             '0.5 Tanimoto of it, and one such molecule voids '
                             'the entire submission.')
    parser.add_argument('--refresh-archive', action='store_true', default=True,
                        help='re-download the archive before filtering. It '
                             'grows every epoch, and a frozen copy loosens the '
                             'gate silently.')
    parser.add_argument('--no-refresh-archive', dest='refresh_archive',
                        action='store_false',
                        help='use the cached archive as-is (tests, offline).')
    parser.add_argument('--no-archive', dest='archive', action='store_const',
                        const='', help='skip the historical diversity check. '
                                       'The validator does not skip it.')
    parser.add_argument('--epoch', type=int, required=True,
                        help='the epoch this will be submitted in; the manifest '
                             'is checked against the live challenge')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--spares', type=int, default=40,
                        help='extra molecules for _still_novel to repair from')
    parser.add_argument('--baseline', type=float, default=0.0)
    parser.add_argument('--fleet', type=int, default=1,
                        help='how many uids the publisher will allocate to; '
                             'sizes the carried pool, one file either way')
    parser.add_argument('--output', default=str(BASE / 'result_rxn{reaction}.json'))
    parser.add_argument('--complete', default=None,
                        help='atomic completion marker beside the release. The '
                             'publisher authenticates the release through this '
                             'file -- target, epoch, reaction, the resolved '
                             'result path and its sha256 must all agree -- and '
                             'treats a release with no marker as "the search is '
                             'not finished", so a bank-fed box must write one '
                             'or it will never commit molecules.')
    arguments = parser.parse_args()
    if arguments.fleet < 1:
        print('--fleet must be at least 1', file=sys.stderr)
        return 1

    connection = sqlite3.connect(arguments.bank)
    if 'rescored' not in {r[0] for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}:
        print('bank has no `rescored` table; run bank_rescore.py first',
              file=sys.stderr)
        return 1

    rows = list(connection.execute(
        'SELECT c.name, c.smiles, c.inchikey, c.heavy_atoms, c.rotatable_bonds, '
        '       r.mean_score, r.passes, c.components, '
        '       c.affinity_probability_binary, c.affinity_pred_value '
        'FROM rescored r JOIN candidates c ON c.name = r.name '
        'WHERE c.reaction = ? ORDER BY r.mean_score DESC', (arguments.reaction,)))
    helpers, gate_evidence = None, None
    if arguments.epoch >= MOLECULE_CONFORMER_ACTIVATION:
        try:
            helpers = validator_helpers()
            if helpers is None or 'molecule_gate_identity' not in helpers:
                raise ValueError('bound molecule gate unavailable')
        except Exception as error:
            print(json.dumps(dict(event='molecule_gate_unavailable', error_class=type(error).__name__,
                                  action='release_refused')), file=sys.stderr)
            return 1
    # Drop what the validator would refuse BEFORE choosing, not after. At 25248
    # every one of our ten uids was skipped for a single molecule at 0.811, and
    # 44 of the 56 we sent were over the line.
    if arguments.archive:
        archive = Path(arguments.archive)
        if not archive.is_file():
            print(f'historical archive not found: {archive}. Download '
                  f'{HISTORY_DATASET} or pass --no-archive to skip the check '
                  f'(the validator will not skip it).', file=sys.stderr)
            return 1
        if arguments.refresh_archive:
            status = refresh_archive(archive)
            print(json.dumps(dict(event='archive_refresh', **status)), file=sys.stderr)
            if not status.get('refreshed') and not archive.is_file():
                print('no archive and refresh failed; refusing to select',
                      file=sys.stderr)
                return 1
        history = historical_fingerprints(archive, Path(str(archive) + '.fps'))
        if helpers is None:
            helpers = validator_helpers()
        if helpers is None:
            print(json.dumps(dict(event="validator_tree_absent",
                                  why="similarity only; the pre-flight is the "
                                      "backstop for the other gates")),
                  file=sys.stderr)
        flagged = refused_by_validator([r[1] for r in rows], history, helpers)
        kept = [r for index, r in enumerate(rows) if index not in flagged]
        print(json.dumps(dict(event='history_filter', archive_rows=len(history),
                              considered=len(rows), refused=len(flagged),
                              kept=len(kept), threshold=MAX_SIMILARITY_TO_HISTORICAL,
                              reasons={reason.split()[0]: sum(
                                  1 for v in flagged.values()
                                  if str(v).split()[0] == reason.split()[0])
                                  for reason in map(str, set(flagged.values()))},
                              all_gates=helpers is not None)),
              file=sys.stderr)
        rows = kept
        if helpers is not None and 'molecule_gate_identity' in helpers:
            gate_evidence = dict(identity=helpers['molecule_gate_identity'],
                                 considered=len(rows)+len(flagged), accepted=len(rows))
    elif arguments.epoch >= MOLECULE_CONFORMER_ACTIVATION:
        # Offline/history opt-out does not disable current chemical validity.
        considered = len(rows)
        flagged = refused_by_validator([r[1] for r in rows], [], helpers)
        rows = [row for index, row in enumerate(rows) if index not in flagged]
        gate_evidence = dict(identity=helpers['molecule_gate_identity'], considered=considered,
                             accepted=len(rows))

    wanted = required_pool_size(arguments.fleet, arguments.count, arguments.spares)
    if len(rows) < wanted:
        print(f'only {len(rows)} rescored molecules for rxn {arguments.reaction}; '
              f'a {arguments.fleet}-uid fleet needs {wanted}. Rescore more of '
              f'the bank, or release to fewer uids.', file=sys.stderr)
        return 1
    chosen = rows[:wanted]

    entropy = atompair_entropy([r[1] for r in chosen[:arguments.count]])
    if entropy < MINIMUM_ENTROPY:
        print(f'AtomPair entropy {entropy:.4f} is below {MINIMUM_ENTROPY}; the '
              f'validator would skip this submission', file=sys.stderr)
        return 1

    # scored_candidates_from_payload rebuilds a Candidate from every row and
    # cross-checks `reaction` and `components` against the identifier, so a row
    # carrying only a name and a score is refused by the allocator even though
    # validate_release_payload accepts it. The two affinity components are the
    # ORIGINAL single-pass measurement: `rescored` stores a mean score and
    # nothing else, and the allocator only carries these through to the
    # artifact. `score` is the three-pass mean, which is what ranks.
    molecules = [dict(name=r[0], smiles=r[1], inchikey=r[2], heavy_atoms=r[3],
                      rotatable_bonds=r[4], score=float(r[5]),
                      scorer=VALIDATOR_EXACT_SCORER, passes=int(r[6]),
                      reaction=arguments.reaction,
                      components=json.loads(r[7]),
                      affinity_probability_binary=float(r[8]),
                      affinity_pred_value=float(r[9]))
                 for r in chosen]
    selection_score = float(sum(m['score'] for m in molecules[:arguments.count]))
    if selection_score < arguments.baseline:
        print(f'selection {selection_score:.4f} does not beat baseline '
              f'{arguments.baseline:.4f}', file=sys.stderr)
        return 1

    payload = dict(
        release=dict(
            schema_version=RELEASE_SCHEMA_VERSION,
            status='qualified',
            reaction=arguments.reaction,
            target=arguments.target,
            epoch=arguments.epoch,
            scorer=VALIDATOR_EXACT_SCORER,
            selected_count=arguments.count,
            selection_score=selection_score,
            exact_count=len(molecules),
            entropy=entropy,
            baseline=dict(score=arguments.baseline),
            fleet_size=arguments.fleet,
        ),
        molecules=molecules,
    )
    if gate_evidence is not None:
        payload['release']['molecule_gate'] = gate_evidence
    out = Path(arguments.output.format(reaction=arguments.reaction))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(out)
    if arguments.complete:
        # Written AFTER the release and digested from what actually landed, so
        # a marker can never authorise a file that was still being written.
        marker = Path(arguments.complete.format(reaction=arguments.reaction))
        marker.parent.mkdir(parents=True, exist_ok=True)
        payload_marker = dict(
            schema_version=1,
            status='complete',
            target=arguments.target,
            epoch=arguments.epoch,
            reaction=arguments.reaction,
            result=str(out.resolve()),
            result_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            # The publisher requires a non-empty iterations list. A bank release
            # has no search iterations; what it has is the bank it was cut from,
            # so that is what is recorded rather than a fabricated round.
            iterations=[dict(source='bank', pool=len(molecules),
                             rescored_available=len(rows),
                             fleet=arguments.fleet)],
        )
        temporary = marker.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(payload_marker, indent=1))
        temporary.replace(marker)

    print(f'{out}: {arguments.count} selected of a {len(molecules)}-row pool '
          f'for {arguments.fleet} uid(s), sum={selection_score:.4f}, '
          f'entropy={entropy:.4f}, epoch={arguments.epoch}'
          + (f', marker={arguments.complete.format(reaction=arguments.reaction)}'
             if arguments.complete else ', NO MARKER (publisher will not commit)'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
