"""Publish one distinct nanobody from every uid our coldkey currently holds.

The exact-score bank is target-bound and reusable across epochs.  This process
reruns the current validator Layer-2 gates over unused repeat-scored sequences,
then binds the best k PAIRWISE DISTINCT of them to the k uids the chain says are
ours this epoch, and delegates each atomic JSON replacement to
publish_checkpoint.  It is safe to invoke repeatedly from a systemd timer: one
process owns the local lock and an already-qualified current-epoch fleet is a
no-op.

Three properties are not negotiable here, because each of them cost real epochs.

*The fleet is discovered from the chain by COLDKEY, every epoch.*  uid numbers
are recycled on deregistration, and this miner spent twenty-one days publishing
into a uid that had become another miner's slot while every diagnostic keyed on
the remembered number followed the new owner.  No configured hotkey or uid list
is accepted, not even as a fallback: configuration must not be able to lie about
identity again.  An empty discovered set is a loud failure, never a quiet skip.

*The k submissions are distinct.*  The validator dedupes by sequence -- k copies
of one molecule score identically and are worth exactly one ticket in its single
draw -- so k uids are only worth holding if they carry k different molecules.

*The publication gates come from nanobody_opt.publication_gates.*  This file used
to hold its own hardcoded copy of the thresholds, in a repository that was not
the one the gates were relaxed in, and so it kept refusing to publish for hours
while the config said otherwise.

``--help`` must work with no chain, no wallet and no keychain: the deploy gate
runs it before stopping the producer.  Nothing here connects or loads a wallet
at import time or during argument parsing.
"""
from __future__ import annotations

from dataclasses import dataclass

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple, Sequence

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compound_miner.nanobody_provider import (  # noqa: E402
    NanobodyCheckpointError,
    validate_nanobody_checkpoint,
    validator_boltzgen_metrics,
)
from nanobody_opt import fleet as fleet_module  # noqa: E402
from nanobody_opt import publication_gates  # noqa: E402
from nanobody_opt import db  # noqa: E402
from nanobody_opt import submission_timing
from nanobody_opt.epoch_deadline import EpochDeadline, run as run_epoch_command

_COMMAND_DEADLINE: EpochDeadline | None = None


CONFIG_URL = "https://raw.githubusercontent.com/metanova-labs/nova/main/config/config.yaml"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/"
    "{target}_nanobodies.csv"
)
DEFAULT_EXACT_INPUTS = (
    "local_neighborhood_round1_exact.json",
    "local_neighborhood_round1_winner_confirm.json",
    "live/epoch25021/local_neighborhood_round2_exact_part1.json",
    "live/epoch25021/finalists_round1_confirm.json",
    "live/epoch25021/local_neighborhood_round2_exact_part2.json",
    "live/epoch25021/new_contenders_confirm.json",
)
DEFAULT_LAYER2_INPUTS = (
    "/root/live/epoch25021/local_neighborhood_round2_layer2.json",
)

# The gates live in nanobody_opt.publication_gates, which the race imports as
# well.  This file keeps no copy: two hardcoded copies of the same thresholds is
# exactly what let a relaxation be deployed while the program that actually
# publishes went on refusing.
RANK_QUALIFICATION_ARGUMENTS = tuple(publication_gates.gate_arguments())

# NOVA_FULL_ARCHIVE_FIELD=1 (see sequential_exact_race.py's identical switch)
# must flip the PUBLISHER's own ranking pass too, not just the race: before
# this, _rank_command always ranked against the small live-epoch field
# (--field-size 49), so nanobody_history/epoch*.json's selection.score sat on
# a 60-400 scale even after the race itself switched to the ~50k-260k
# archive-wide scale -- two different yardsticks for the same fleet, one of
# them stale. User, 2026-10-10: "Fix publisher to use the same [scale]."
_ARCHIVE_FIELD_GATES_PATH = (
    Path(__file__).resolve().parent.parent
    / "nanobody_opt" / "strict_gates_archive_field.json"
)


def _full_archive_field_enabled() -> bool:
    return os.environ.get("NOVA_FULL_ARCHIVE_FIELD") == "1"


def _archive_field_rank_qualification_arguments() -> tuple[str, ...]:
    overrides = publication_gates.load_overrides(_ARCHIVE_FIELD_GATES_PATH)
    return tuple(publication_gates.gate_arguments(overrides))

# Only rank one pays and the validator draws once, so the shortlist that decides
# what is published must be ordered by the chance of a winning draw.
SELECTION_ORDER = fleet_module.REQUIRED_SELECTION_ORDER
# How many ranked candidates the fleet gets to choose from. This is the funnel's
# last stage and it was its narrowest: the generator now proposes four
# clonotypes and a 350-candidate pool carried two, because win-probability
# order concentrates its head on the families with the most observations and
# a new clonotype starts with few. The score bank holds nine clonotypes among
# its best eight hundred by score, so the pool has to reach that deep to see
# them.
#
# This cannot change the lead ticket -- the top of the ordering is the same
# whether the list is thirty or a thousand -- so it is outside the selection
# freeze. The cost is Layer-2 and CDR annotation over more candidates, and
# both are almost entirely cache hits: 350 candidates cost one new TNP
# measurement per cycle.
SHORTLIST_LIMIT = 1000
#: Ranked candidates per uid, so a growing fleet keeps room to find distinct
#: molecules rather than competing for the same thirty.
SHORTLIST_PER_UID = 50

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SUPPORTED_CHAIN_STATE_SCHEMAS = frozenset({1, 2})
PUBLISHER_STATE_SCHEMA_VERSION = 4


class ChainView(NamedTuple):
    """The miner's own chain view: when we are, and who we are."""

    epoch: int
    block: int
    epoch_length: int
    neurons: list[dict[str, Any]]


def _atomic_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(response.content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another nanobody publication is still running") from error
        yield


#: A roster older than this has missed at least one epoch boundary, which is
#: the signature of a miner that has stopped re-syncing. Two epochs of slack
#: so a single failed sync warns rather than halts publication.
MAX_ROSTER_AGE_BLOCKS = 800


def _chain_view(chain_state: Path, netuid: int, max_age: float) -> ChainView:
    """Read the chain view exported by the already-connected miner process.

    The neuron rows travel with the block, so identity and time come from one
    snapshot: a fleet discovered from a stale metagraph is the recycled-uid bug
    with extra steps.
    """
    payload = _read_json(chain_state)
    if payload is None:
        raise RuntimeError(f"miner chain-state file is missing: {chain_state}")
    try:
        schema_version = int(payload["schema_version"])
        state_netuid = int(payload["netuid"])
        block = int(payload["block"])
        epoch_length = int(payload["epoch_length"])
        epoch = int(payload["epoch"])
        age = time.time() - float(payload["updated_at_unix"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("miner chain-state file is malformed") from error
    if schema_version not in SUPPORTED_CHAIN_STATE_SCHEMAS:
        raise RuntimeError(f"unsupported chain-state schema: {schema_version}")
    if state_netuid != netuid:
        raise RuntimeError(f"chain-state netuid is {state_netuid}, expected {netuid}")
    if epoch_length < 1 or epoch != block // epoch_length:
        raise RuntimeError("miner chain-state epoch is internally inconsistent")
    if age < -5 or age > max_age:
        raise RuntimeError(f"miner chain-state is stale: age={age:.1f}s")
    # The FILE being fresh says nothing about the ROSTER being fresh. The
    # miner re-synced its metagraph only when block % epoch_length == 0, a
    # condition its loop stepped over every time, so for five days it
    # exported a roster frozen at process start under an ever-advancing
    # block. Every freshness check passed and we kept assigning sequences to
    # a uid that had been re-registered to another coldkey -- two of eight
    # slots discarded per epoch, silently. Refuse a roster that has not been
    # re-synced within MAX_ROSTER_AGE_BLOCKS rather than publish against it.
    roster_block = payload.get("roster_block")
    if roster_block is None:
        raise RuntimeError(
            "miner chain-state carries no roster_block; the miner predates "
            "roster-age reporting and may be exporting a frozen roster"
        )
    roster_age = block - int(roster_block)
    if roster_age < 0 or roster_age > MAX_ROSTER_AGE_BLOCKS:
        raise RuntimeError(
            "miner chain-state roster is stale: synced at block %d, now %d "
            "(%d blocks); the miner is not re-syncing its metagraph"
            % (int(roster_block), block, roster_age)
        )
    try:
        neurons = fleet_module.neuron_rows(payload.get("neurons"))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("miner chain-state neuron rows are malformed") from error
    if not neurons:
        raise RuntimeError(
            "miner chain-state carries no neuron rows, so the fleet cannot be "
            "discovered by coldkey; upgrade the miner's chain-state export "
            "rather than falling back to a remembered uid"
        )
    return ChainView(epoch, block, epoch_length, neurons)


def _gpu_count() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def _checkpoint_sequence(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != 1:
        return None
    sequence = sequences[0]
    if (
        not isinstance(sequence, str)
        or sequence != sequence.upper()
        or set(sequence) - CANONICAL_AMINO_ACIDS
    ):
        return None
    return sequence


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fleet_module.write_atomic_json(path, payload)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str | None:
    try:
        return _sha256_bytes(path.read_bytes())
    except (FileNotFoundError, NotADirectoryError):
        return None


def _exact_bank_fingerprints(
    exact_inputs: tuple[Path, ...], target: str, minimum: int | None = None
) -> tuple[str, str]:
    """Fingerprint all exact data and the strict-eligible subset separately.

    The publisher needs a full refresh for any new one-shot evidence, while the
    expensive structural qualification pass can only change when a sequence
    with at least the strict minimum of independent observations changes.  This
    mirrors ``rank_exact._observations`` including its observation-id
    deduplication, its complete finite-metric requirement, and its relaxation of
    the observation floor when no sequence reaches it.
    """
    required = set(validator_boltzgen_metrics())
    if minimum is None:
        minimum = _strict_minimum_observations()
    manifest: list[tuple[str, str]] = []
    seen_observations: set[str] = set()
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(set(exact_inputs), key=str):
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            continue
        file_digest = _sha256_bytes(raw)
        manifest.append((str(path), file_digest))
        payload = json.loads(raw)
        if str(payload.get("target")) != target:
            raise RuntimeError(
                f"{path} targets {payload.get('target')!r}, not {target}"
            )
        observation_id = str(payload.get("observation_id") or "").strip()
        source_key = observation_id or file_digest
        if source_key in seen_observations:
            continue
        seen_observations.add(source_key)
        structures = payload.get("structures") or {}
        for by_sequence in payload.get("per_nanobody_components", {}).values():
            for sequence, by_target in by_sequence.items():
                metrics = by_target.get(target)
                if not isinstance(metrics, dict) or not required.issubset(metrics):
                    continue
                try:
                    values = {name: float(metrics[name]) for name in required}
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(value) for value in values.values()):
                    continue
                normalized = str(sequence).strip().upper()
                observations[normalized].append({
                    "observation": source_key,
                    "metrics": values,
                    "structure": structures.get(sequence),
                })
    manifest_digest = _sha256_bytes(
        json.dumps(manifest, separators=(",", ":")).encode()
    )
    # Ranking relaxes the observation floor one step at a time rather than
    # publishing nothing, so the strict-eligible set has to relax with it.
    # Holding this at the unrelaxed floor would empty the subset for a thin
    # epoch, and an empty subset has a constant digest: the cache would stop
    # noticing new evidence exactly while ranking was still selecting on it.
    deepest = max((len(records) for records in observations.values()), default=1)
    effective = max(1, min(minimum, deepest))
    strict_eligible = {
        sequence: records
        for sequence, records in sorted(observations.items())
        if len(records) >= effective
    }
    strict_digest = _sha256_bytes(
        json.dumps(
            strict_eligible, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    return manifest_digest, strict_digest


def _strict_minimum_observations() -> int:
    """The evidence floor, read from the one module that owns it."""
    return int(publication_gates.gate_defaults()["--minimum-observations"])


def _publisher_state_matches(
    state: dict[str, Any] | None,
    *,
    epoch: int,
    target: str,
    exact_fingerprint: str,
    ledger_fingerprint: str | None,
    fleet_fingerprint: str,
) -> bool:
    return bool(
        state
        and state.get("schema_version") == PUBLISHER_STATE_SCHEMA_VERSION
        and state.get("epoch") == epoch
        and state.get("target") == target
        and state.get("exact_fingerprint") == exact_fingerprint
        and state.get("ledger_fingerprint") == ledger_fingerprint
        # A registration or deregistration changes how many distinct sequences
        # must be published, so it must never be served from cache.
        and state.get("fleet_fingerprint") == fleet_fingerprint
    )


def _strict_state_matches(
    state: dict[str, Any] | None,
    *,
    epoch: int,
    target: str,
    strict_fingerprint: str,
    ledger_fingerprint: str | None,
    fleet_fingerprint: str,
) -> bool:
    return bool(
        state
        and state.get("schema_version") == PUBLISHER_STATE_SCHEMA_VERSION
        and state.get("epoch") == epoch
        and state.get("target") == target
        and state.get("strict_fingerprint") == strict_fingerprint
        and state.get("ledger_fingerprint") == ledger_fingerprint
        and state.get("fleet_fingerprint") == fleet_fingerprint
    )


def _write_publisher_state(
    path: Path,
    *,
    epoch: int,
    target: str,
    exact_fingerprint: str,
    strict_fingerprint: str,
    ledger_fingerprint: str | None,
    fleet_fingerprint: str,
) -> None:
    _atomic_json(path, {
        "schema_version": PUBLISHER_STATE_SCHEMA_VERSION,
        "epoch": epoch,
        "target": target,
        "exact_fingerprint": exact_fingerprint,
        "strict_fingerprint": strict_fingerprint,
        "ledger_fingerprint": ledger_fingerprint,
        "fleet_fingerprint": fleet_fingerprint,
        "updated_at_unix": time.time(),
    })


def _archive_and_audit_previous(
    payload: dict[str, Any] | None,
    *,
    current_epoch: int,
    history_dir: Path,
    audit_dir: Path,
    bgen_python: Path,
    nanobody_opt_root: Path,
) -> None:
    """Persist and exactly replay a completed epoch when its API is ready.

    Every uid is audited under its own hotkey.  Auditing one hotkey and
    generalizing was how a recycled uid stayed invisible: the leaderboard row
    that came back belonged to whoever held the number.
    """
    if not isinstance(payload, dict):
        return
    try:
        previous_epoch = int(payload["epoch"])
        target = str(payload["target"])
    except (KeyError, TypeError, ValueError):
        return
    if previous_epoch >= current_epoch:
        return
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or not submissions:
        return
    checkpoint = history_dir / f"epoch{previous_epoch}.json"
    if not checkpoint.is_file():
        _atomic_json(checkpoint, payload)
    for entry in submissions:
        if not isinstance(entry, dict):
            continue
        hotkey = str(entry.get("hotkey") or "").strip()
        if not hotkey:
            continue
        audit = audit_dir / f"epoch{previous_epoch}_{hotkey}.json"
        if audit.is_file():
            continue
        try:
            _run([
                str(bgen_python),
                str(nanobody_opt_root / "audit_validator_score.py"),
                "--epoch", str(previous_epoch),
                "--hotkey", hotkey,
                "--target", target,
                "--checkpoint", str(checkpoint),
                "--output", str(audit),
            ])
        except subprocess.CalledProcessError as error:
            print(
                f"validator audit for completed epoch {previous_epoch} hotkey "
                f"{hotkey} is not ready: exit={error.returncode}",
                flush=True,
            )


def _default_exact_inputs(score_bank_root: Path) -> tuple[Path, ...]:
    """Return independent exact runs without double-counting shard outputs.

    New scoring waves write both per-wave files and an ``exact_all.json``
    aggregate. Only the aggregate is discovered so a single stochastic run
    cannot satisfy the repeat-observation publication gate by appearing twice.
    Explicit legacy runs and confirmation runs remain separate observations.
    """
    paths = [score_bank_root / relative for relative in DEFAULT_EXACT_INPUTS]
    patterns = (
        "*_exact.json",
        "design/*_exact*.json",
        "live/epoch*/**/exact_all.json",
        "live/epoch*/**/*confirm*.json",
    )
    for pattern in patterns:
        paths.extend(sorted(score_bank_root.glob(pattern)))
    return tuple(dict.fromkeys(paths))


def _current_epoch_exact_inputs(
    score_bank_root: Path, epoch: int
) -> tuple[Path, ...]:
    """Return only atomically completed raw scorer aggregates for one epoch.

    Historical evidence is a database concern.  Restricting discovery to the
    current epoch makes the filesystem an ingestion queue instead of an
    ever-growing query engine on the publication critical path.
    """
    root = score_bank_root / "live" / f"epoch{epoch}"
    if not root.is_dir():
        return ()
    from nanobody_opt.exact_checkpoints import exact_paths
    return tuple(exact_paths(root))


def _previous_confirmed_pool(epoch_root: Path, target: str) -> tuple[set[str], tuple[Path, ...]]:
    """Recover a bounded, completed shortlist across a publication outage.

    Scores here only choose which saved evidence to recover. Current-field
    ranking, independent-observation counting, novelty and Layer-2 still run.
    Never walk historical screening pools or launch old GPU work.
    """
    from nanobody_opt_v2.run_live_producer import _artifact_policy_stale
    prior = epoch_root.parent / f"epoch{int(epoch_root.name.removeprefix('epoch')) - 1}"
    candidates = []
    for iteration in sorted((prior / 'producer').glob('iteration_[0-9][0-9]')):
        layer2 = iteration / 'layer2_dedup.json'
        if _artifact_policy_stale(layer2):
            continue
        rounds = iteration / 'race/rounds'
        try:
            eligible = json.loads(layer2.read_text())
            ranking = json.loads((rounds / 'stage_ranking.json').read_text())
        except (OSError, ValueError):
            continue
        if eligible.get('target') != target or ranking.get('target') != target:
            continue
        allowed = set(eligible.get('sequences') or [])
        for candidate in ranking.get('candidates') or []:
            try:
                sequence = str(candidate['sequence']).strip().upper()
                score = float(candidate['selection_score'])
                depth = int(candidate['observation_count'])
            except (KeyError, TypeError, ValueError):
                continue
            if sequence in allowed and depth >= _strict_minimum_observations() and math.isfinite(score):
                candidates.append((score, sequence, rounds))
    selected: dict[str, Path] = {}
    for _, sequence, rounds in sorted(candidates):
        selected.setdefault(sequence, rounds)
        if len(selected) == 64:
            break
    # At most 64 completed repeat files from the immediately preceding epoch;
    # current ingestion is always first, and the command deadline still applies.
    files = sorted({path for rounds in selected.values()
                    for path in rounds.glob('round_[0-9][0-9]/exact_all.json')})[:64]
    return set(selected), tuple(files)


def _publication_exact_inputs(score_bank_root: Path, epoch: int, target: str) -> tuple[Path, ...]:
    _, carryover = _previous_confirmed_pool(score_bank_root / 'live' / f'epoch{epoch}', target)
    return tuple(dict.fromkeys((*_current_epoch_exact_inputs(score_bank_root, epoch), *carryover)))


def _exact_run_descriptor(path: Path, target: str) -> dict[str, Any]:
    """Describe exactly the rows the raw-screen importer must seal."""
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    payload = json.loads(raw)
    if str(payload.get("target")) != target:
        raise RuntimeError(f"{path} targets {payload.get('target')!r}, not {target}")
    run_id = str(payload.get("observation_id") or digest)
    required = set(validator_boltzgen_metrics())
    observation_ids: set[str] = set()
    for group, by_sequence in payload.get("per_nanobody_components", {}).items():
        if not isinstance(by_sequence, dict):
            continue
        for sequence, by_target in by_sequence.items():
            metrics = by_target.get(target) if isinstance(by_target, dict) else None
            if not isinstance(metrics, dict) or not required.issubset(metrics):
                continue
            try:
                values = (float(metrics[name]) for name in required)
                if not all(math.isfinite(value) for value in values):
                    continue
            except (TypeError, ValueError):
                continue
            normalized = str(sequence).strip().upper()
            observation_ids.add(f"local-exact:{run_id}:{group}:{normalized}")
    return {
        "path": path,
        "run_id": run_id,
        "file_sha256": digest,
        "expected_rows": len(observation_ids),
    }


def _exact_run_is_sealed(
    database: Path, *, target: str, descriptor: Mapping[str, Any]
) -> bool:
    if not descriptor["expected_rows"]:
        return False
    # Our own race's sealed-run bookkeeping -- route through database (the
    # local SQLite primary under NOVA_LOCAL_SQLITE=1) via mutation_network's
    # dialect-aware connect, not an unconditional central MySQL connect().
    # `database` used to be accepted and ignored here, same bug fixed in
    # build_top50_index.py (2026-10-10): a Hetzner blip here silently
    # degraded (this already fails closed to False) a check on data that
    # was sitting locally the whole time.
    from nanobody_opt import mutation_network
    try:
        connection = mutation_network._connect(Path(database))
        p = "?" if mutation_network._is_local(connection) else "%s"
        row = connection.execute(
            f"""
            SELECT r.file_sha256, r.expected_rows, r.ingested_rows, r.complete,
                   COUNT(x.observation_id)
            FROM exact_runs r
            LEFT JOIN exact_run_observations x
              ON x.target=r.target AND x.run_id=r.run_id
            WHERE r.target={p} AND r.run_id={p}
            GROUP BY r.target, r.run_id
            """,
            (target, descriptor["run_id"]),
        ).fetchone()
    except Exception:
        return False
    finally:
        if "connection" in locals():
            connection.close()
    expected = int(descriptor["expected_rows"])
    return bool(
        row
        and str(row[0]) == descriptor["file_sha256"]
        and int(row[1]) == expected
        and int(row[2]) == expected
        and int(row[3]) == 1
        and int(row[4]) == expected
    )


def _database_exact_fingerprints(
    database: Path, target: str
) -> tuple[str, str, int]:
    """Fingerprint authoritative sealed raw runs without reading score files."""
    # Same local-SQLite routing as _exact_run_is_sealed above.
    from nanobody_opt import mutation_network
    try:
        connection = mutation_network._connect(Path(database))
        p = "?" if mutation_network._is_local(connection) else "%s"
        rows = connection.execute(
            f"""
            SELECT r.run_id, r.file_sha256, r.expected_rows, r.scorer_protocol,
                   COUNT(x.observation_id),
                   SUM(CASE WHEN a.observation_id IS NOT NULL THEN 1 ELSE 0 END)
            FROM exact_runs r
            JOIN exact_run_observations x
              ON x.target=r.target AND x.run_id=r.run_id
            LEFT JOIN observation_artifacts a
              ON a.observation_id=x.observation_id
            WHERE r.target={p} AND r.complete=1
            GROUP BY r.target, r.run_id
            HAVING COUNT(x.observation_id)=r.expected_rows
            ORDER BY r.run_id
            """,
            (target,),
        ).fetchall()
    except Exception:
        return _sha256_bytes(b"[]"), _sha256_bytes(b"[]"), 0
    finally:
        if "connection" in locals():
            connection.close()

    def _jsonable(value: Any) -> Any:
        # MySQL returns COUNT/SUM as Decimal; json.dumps cannot serialise that.
        from decimal import Decimal

        return int(value) if isinstance(value, Decimal) else value

    manifest = [[_jsonable(v) for v in row[:5]] for row in rows]
    structural = [[_jsonable(v) for v in row] for row in rows]
    return (
        _sha256_bytes(json.dumps(manifest, separators=(",", ":")).encode()),
        _sha256_bytes(json.dumps(structural, separators=(",", ":")).encode()),
        len(rows),
    )


def _import_exact_command(
    *,
    bgen_python: Path,
    nanobody_opt_root: Path,
    database: Path,
    target: str,
    archive: Path,
    exact_inputs: Sequence[Path],
) -> list[str]:
    command = [
        str(bgen_python),
        "-m",
        "nanobody_opt.import_exact_screens",
        "--database", str(database),
        "--target", target,
        "--archive", str(archive),
    ]
    for path in exact_inputs:
        command.extend(["--exact-root", str(path)])
    return command


def _has_new_exact_data(artifact: Path, exact_inputs: tuple[Path, ...]) -> bool:
    """Return whether an atomically completed exact result is newer than publish."""
    try:
        published_at = artifact.stat().st_mtime_ns
    except FileNotFoundError:
        return True
    return any(
        path.is_file() and path.stat().st_mtime_ns > published_at
        for path in exact_inputs
    )


def _is_current_qualified(
    payload: dict[str, Any] | None,
    *,
    target: str,
    epoch: int,
    requirements: dict[str, Any],
) -> bool:
    """Validate one embedded per-uid checkpoint with the consumer's own rules."""
    if payload is None:
        return False
    try:
        validate_nanobody_checkpoint(
            payload,
            expected_target=target,
            expected_epoch=epoch,
            min_sequence_length=int(requirements["min_sequence_length"]),
            max_sequence_length=int(requirements["max_sequence_length"]),
        )
    except (KeyError, TypeError, ValueError, NanobodyCheckpointError):
        return False
    return True


def _fleet_is_current(
    payload: dict[str, Any] | None,
    *,
    target: str,
    epoch: int,
    requirements: dict[str, Any],
    expected_hotkeys: Sequence[str],
    stage: str = "qualified",
) -> bool:
    """Return whether the single artifact already holds this epoch's fleet.

    Every uid we hold must be present, every embedded checkpoint must pass the
    consumer's validation, and the sequences must be pairwise distinct -- a
    duplicate would mean two uids sharing one deduped validator score.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("publication_stage") != stage or payload.get("status") != stage:
        return False
    if str(payload.get("target")) != target or payload.get("epoch") != epoch:
        return False
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or not submissions:
        return False
    if payload.get("unassigned_uids"):
        return False
    seen: set[str] = set()
    hotkeys: set[str] = set()
    for entry in submissions:
        if not isinstance(entry, dict):
            return False
        sequence = str(entry.get("sequence") or "").strip().upper()
        if not sequence or sequence in seen:
            return False
        seen.add(sequence)
        hotkeys.add(str(entry.get("hotkey") or ""))
        checkpoint = entry.get("checkpoint")
        if not _is_current_qualified(
            checkpoint if isinstance(checkpoint, dict) else None,
            target=target,
            epoch=epoch,
            requirements=requirements,
        ):
            return False
        if _checkpoint_sequence(checkpoint) != sequence:
            return False
    return hotkeys == set(expected_hotkeys)


def _covered_current_assignments(
    payload: dict[str, Any] | None,
    *,
    crew: Sequence[fleet_module.FleetMember],
    target: str,
    epoch: int,
) -> list[fleet_module.Assignment]:
    """Return same-epoch assignments that still map to the current fleet.

    A fresh provisional pass is allowed to improve an artifact, but never to
    replace a broader fleet with a shortfall.  This deliberately checks only
    immutable identity and sequence uniqueness: an existing same-epoch fleet
    may predate the current checkpoint schema, yet its commitments are already
    live and remain safer than publishing fewer UIDs.
    """
    if (
        not isinstance(payload, dict)
        or payload.get("epoch") != epoch
        or str(payload.get("target")) != target
    ):
        return []
    try:
        assignments = fleet_module.assignments_from_artifact(payload)
    except (KeyError, TypeError, ValueError):
        return []
    held = {int(member.uid): str(member.hotkey) for member in crew}
    covered: list[fleet_module.Assignment] = []
    seen: set[str] = set()
    for assignment in assignments:
        sequence = str(assignment.sequence).strip().upper()
        if held.get(int(assignment.uid)) != str(assignment.hotkey):
            continue
        if not sequence or sequence in seen:
            continue
        seen.add(sequence)
        covered.append(assignment)
    return covered


def _retire_completed_epoch(
    payload: dict[str, Any] | None, ledger: Path, current_epoch: int
) -> list[str]:
    """Record every sequence a completed epoch published, so none returns.

    All k, not just the winner: the archive rejects any resubmission, so a
    sequence left out of the ledger is a wasted uid the next time it ranks well.
    """
    if not isinstance(payload, dict):
        return []
    try:
        published_epoch = int(payload["epoch"])
    except (KeyError, TypeError, ValueError):
        return []
    if published_epoch >= current_epoch:
        return []
    return fleet_module.record_published(
        ledger, fleet_module.artifact_sequences(payload)
    )


def _live_sequences(payload: dict[str, Any] | None, epoch: int) -> list[str]:
    """Sequences this same epoch already published, which stay selectable."""
    if not isinstance(payload, dict) or payload.get("epoch") != epoch:
        return []
    return fleet_module.artifact_sequences(payload)


def _artifact_passes_scope(payload: dict[str, Any] | None, epoch: int, scope: Path) -> bool:
    """Use the artifact reader's real list contract for current eligibility."""
    return set(_live_sequences(payload, epoch)) <= set(scope.read_text().splitlines())


def _run(command: list[str]) -> None:
    run_epoch_command(command, deadline=_COMMAND_DEADLINE, timeout=300)


def _rank_command(
    *,
    bgen_python: Path,
    nanobody_opt_root: Path,
    exact_inputs: Sequence[Path],
    archive: Path,
    target: str,
    output: Path,
    exclusion: Path,
    epoch: int,
    shortlist: int,
    config: Path | None,
    strict: bool,
    observation_history: Path | None = None,
    diversity_reserve_count: int = 0,
    only_sequences: Path | None = None,
) -> list[str]:
    command = [str(bgen_python), str(nanobody_opt_root / "rank_exact.py")]
    for path in exact_inputs:
        command.extend(["--exact", str(path)])
    if observation_history is not None:
        command.extend(["--observation-history", str(observation_history)])
    command.extend([
        "--archive", str(archive),
        "--target", target,
        "--output", str(output),
        "--exclude-file", str(exclusion),
        "--recent", "300",
        "--field-size", "49",
        "--bootstrap-fields", "64",
        "--limit", str(shortlist),
        "--before-epoch", str(epoch),
    ])
    # --db-field-count needs --observation-history; the rare explicit_exact
    # invocation (observation_history=None, see caller) falls back to the
    # live API instead of crashing on a missing required arg.
    if observation_history is not None:
        command.extend(["--db-field-count", "10", "--minimum-api-fields", "3"])
    else:
        command.extend(["--api-field-count", "10", "--minimum-api-fields", "3"])
    # NOVA_FULL_ARCHIVE_FIELD switches the publisher's own ranking pass to the
    # same full-archive field the race ranks candidates against (see the
    # module-level comment by _full_archive_field_enabled). rank_exact.py
    # ignores --field-size/--bootstrap-fields/--api-field-count/
    # --minimum-api-fields once this is set, so they are left in place above
    # rather than threaded with conditionals.
    full_archive_field = _full_archive_field_enabled()
    if full_archive_field:
        command.append("--full-archive-field")
    if diversity_reserve_count:
        command.extend([
            "--diversity-reserve-count", str(diversity_reserve_count)
        ])
    if only_sequences is not None:
        command.extend(["--only-sequences", str(only_sequences)])
    if strict:
        command.extend(["--pose-config", str(config)])
        command.extend(
            _archive_field_rank_qualification_arguments() if full_archive_field
            else publication_gates.gate_arguments()
        )
    else:
        # The provisional pass exists so that an epoch always submits something;
        # it applies the evidence floor only, never the quality gates. The
        # threshold remains a shared diagnostic basis for competitive rates;
        # fresh-draw field-win probability itself no longer depends on it.
        gates = (
            publication_gates.load_overrides(_ARCHIVE_FIELD_GATES_PATH)
            if full_archive_field else {}
        )
        defaults = publication_gates.gate_defaults()
        command.extend([
            "--minimum-observations",
            defaults["--minimum-observations"],
            "--competitive-score-threshold",
            str(gates.get(
                "--competitive-score-threshold",
                defaults["--competitive-score-threshold"],
            )),
        ])
    command.extend(["--selection-order", SELECTION_ORDER])
    return command


def _strict_pool(
    *,
    args: argparse.Namespace,
    bgen_python: Path,
    epoch_root: Path,
    archive_path: Path,
    target: str,
    exclusion_path: Path,
    epoch: int,
    shortlist: int,
    config_path: Path,
    exact_inputs: Sequence[Path],
    observation_history: Path | None,
    crew: Sequence[Any],
    previous: dict[str, Any] | None,
    candidate_scope: Path | None = None,
) -> Path | None:
    """The sequences strict qualification should look at, or None for the whole shortlist.

    A FRESH provisional-mode ranking supplies the order (the stored one can be an hour old), so the
    candidates qualified are the ones that would be published if the cycle ran this minute.
    """
    size = int(getattr(args, "strict_pool_size", 0) or 0)
    if size <= 0:
        size = 100
    pool_rankings = epoch_root / "strict_pool_rankings.json"
    try:
        _run(_rank_command(
            bgen_python=bgen_python,
            nanobody_opt_root=args.nanobody_opt_root,
            exact_inputs=exact_inputs,
            archive=archive_path,
            target=target,
            output=pool_rankings,
            exclusion=exclusion_path,
            epoch=epoch,
            shortlist=shortlist,
            config=config_path,
            strict=False,
            observation_history=observation_history,
            diversity_reserve_count=len(crew) * 2 + 1,
            only_sequences=candidate_scope,
        ))
        candidates = json.loads(pool_rankings.read_text())["candidates"]
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
        raise RuntimeError('strict pool unavailable; refusing an unbounded upgrade') from error
    sequences = list(dict.fromkeys(
        str(item["sequence"]).strip().upper() for item in candidates[:size]
    ))
    # whatever this epoch already published must stay selectable
    sequences += [s for s in _live_sequences(previous, epoch) if s not in sequences]
    if not sequences:
        raise RuntimeError('strict pool is empty; retaining the published artifact')
    path = epoch_root / "strict_pool_sequences.txt"
    path.write_text("\n".join(sequences) + "\n")
    print(f"strict pool: qualifying the top {len(sequences)} of {len(candidates)} candidates",
          flush=True)
    return path


def _layer2_command(
    *,
    pixi_manifest: Path,
    rankings: Path,
    config: Path,
    output: Path,
    concurrency: int,
    cache_dir: Path,
    pixi: Path,
) -> list[str]:
    return [
        str(pixi), "run", "--manifest-path", str(pixi_manifest),
        "-e", "metanano", "filter-layer2", "--",
        "--input", str(rankings),
        "--config", str(config),
        "--output", str(output),
        "--concurrency", str(concurrency),
        "--igblast-concurrency", str(concurrency),
        "--igblast-batch-size", "1",
        "--cache-dir", str(cache_dir),
    ]


def _novelty_command(
    *,
    pixi_manifest: Path,
    rankings: Path,
    archive: Path,
    target: str,
    output: Path,
    pixi: Path,
) -> list[str]:
    """Replay the validator's own duplicate gates before we commit anything.

    utils/nanobodies.py::is_duplicate drops a UID outright when any of its
    sequences is >=0.95 identical to a top sequence for the target, or shares a
    CDR3 at >=0.85. Local neighbourhood search produces exactly that: our six
    epoch-25093 tickets were 0.95-0.98 identical to the epoch-25091 rank-1
    sequence and 0.94-0.98 identical to each other, so every one of them was
    discarded before scoring and we appeared nowhere on the leaderboard.
    """
    return [
        str(pixi), "run", "--manifest-path", str(pixi_manifest),
        "-e", "metanano", "check-validator-novelty", "--",
        "--input", str(rankings),
        "--archive", str(archive),
        "--target", target,
        "--output", str(output),
    ]


def _validator_novel_sequences(novelty: Path) -> set[str]:
    """Sequences the validator's own gates would accept as novel."""
    payload = json.loads(novelty.read_text())
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise SystemExit(f"{novelty} holds no novelty result list")
    return {str(item).strip().upper() for item in sequences}


def _current_layer2_passes(layer2: Path) -> set[str]:
    """Sequences this epoch's Layer-2 run passed, upper-cased for comparison."""
    payload = json.loads(layer2.read_text())
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"{layer2} holds no Layer-2 candidate list")
    return {
        str(entry.get("sequence") or "").strip().upper()
        for entry in candidates
        if isinstance(entry, Mapping) and entry.get("layer2_passed")
    }


def _novel_layer2_rankings(rankings: Path, novelty: Path, output: Path) -> Path:
    """Spend full TNP only on ranked candidates that can pass novelty."""
    payload = _restrict_pool(
        json.loads(rankings.read_text()), _validator_novel_sequences(novelty),
        gate="validator-novelty", source=novelty,
    )
    _atomic_json(output, payload)
    return output


def _producer_candidate_scope(epoch_root: Path, target: str, previous=None) -> Path | None:
    """Legacy explicit-file scope; sealed history uses the prechecked bank."""
    from nanobody_opt_v2.run_live_producer import _artifact_policy_stale
    allowed = set()
    found = False
    for path in sorted((epoch_root / "producer").glob("iteration_[0-9][0-9]/layer2_dedup.json")):
        if _artifact_policy_stale(path):
            continue
        payload = json.loads(path.read_text())
        if payload.get("target") != target or not isinstance(payload.get("sequences"), list):
            continue
        found = True
        allowed.update(str(s).strip().upper() for s in payload["sequences"] if str(s).strip())
    if not found:
        return None
    epoch = int(epoch_root.name.removeprefix("epoch"))
    allowed.update(_live_sequences(previous, epoch))
    confirmed, _ = _previous_confirmed_pool(epoch_root, target)
    allowed.update(confirmed)
    output = epoch_root / "producer_candidate_scope.txt"
    output.write_text("".join(s + "\n" for s in sorted(allowed)))
    return output


def _stored_comparison_sequences(database: Path, target: str, minimum: int) -> set[str]:
    """Read an evidence-eligible superset, never a merit-limited shortlist.

    The ranker still validates complete finite metric vectors. Counting sealed
    own observations here can over-admit malformed evidence, but cannot promote
    it through that final evidence check. Public rival scores are not own folds.
    """
    from nanobody_opt import mutation_network
    connection = mutation_network._connect(database)
    try:
        p = "?" if mutation_network._is_local(connection) else "%s"
        # The maintained aggregate avoids scanning millions of raw rows each
        # publication cycle. A stale/missing aggregate must not hide candidates.
        ready = False
        if mutation_network._is_local(connection) and mutation_network._aggregate_ready(connection):
            watermark = connection.execute(
                "SELECT value FROM exact_observation_meta WHERE key='source_max_rowid'"
            ).fetchone()
            live = connection.execute("SELECT MAX(rowid) FROM exact_run_observations").fetchone()
            ready = bool(watermark and int(watermark[0]) == int(live[0] or 0))
        if ready:
            rows = connection.execute(
                f"SELECT DISTINCT sequence FROM exact_observation_aggregate "
                f"WHERE target={p} AND count_value>={p}", (target, minimum),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT n.sequence FROM observations o "
                "JOIN nodes n ON n.id=o.node_id "
                "JOIN exact_run_observations x "
                "ON x.observation_id=o.observation_id AND x.target=n.target "
                "JOIN exact_runs r ON r.target=x.target AND r.run_id=x.run_id "
                f"WHERE n.target={p} AND r.complete=1 "
                "AND o.source='local_exact_raw' AND o.metrics_json IS NOT NULL "
                f"GROUP BY n.sequence HAVING COUNT(*)>={p}", (target, minimum),
            ).fetchall()
        return {str(row[0]).strip().upper() for row in rows if row[0]}
    finally:
        connection.close()


def _prechecked_comparison_scope(
    *, args: argparse.Namespace, epoch_root: Path, target: str,
    database: Path, archive: Path, config: Path, exclusion: Path,
    pixi_manifest: Path, ready_artifact: bool = True,
    minimum_observations: int | None = None, cheap_only: bool = False,
    epoch: int | None = None, runner=None,
) -> Path:
    """Apply current gates to stored AND newly confirmed candidates before ranking.

    Reuse sequence measurements, not old eligibility. Incremental proof is
    valid only for this epoch, target, archive, config and gate implementation.
    No exact scorer is invoked and no discovery-budget reservation is made.
    """
    minimum = (int(publication_gates.gate_defaults()["--minimum-observations"])
               if minimum_observations is None else minimum_observations)
    run = runner or _run
    candidates = _stored_comparison_sequences(database, target, minimum)
    with archive.open(newline="") as handle:
        historical = {str(row.get("sequence") or "").strip().upper()
                      for row in csv.DictReader(handle)}
    excluded = {s.strip().upper() for s in exclusion.read_text().splitlines() if s.strip()}
    candidates -= historical | excluded
    proof_key = {
        "schema_version": PUBLISHER_STATE_SCHEMA_VERSION,
        "policy": "incremental-layer2-v1",
        "epoch": int(epoch_root.name.removeprefix("epoch")) if epoch is None else epoch,
        "target": target,
        "archive": _file_digest(archive), "config": _file_digest(config),
        "gates": {name: _file_digest(args.nanobody_opt_root / name) for name in (
            "check_validator_novelty.py", "filter_layer2.py",
        )},
    }
    if cheap_only:
        proof_key.update(confirmation_only=True, minimum_observations=minimum)
    if getattr(args, "adaptive_confirmation_depth", False):
        proof_key["exclusion"] = _file_digest(exclusion)
    proof_path = epoch_root / "comparison_precheck.json"
    proof = _read_json(proof_path) or {}
    prior_eligible = set(proof.get("eligible_sequences", []))
    if minimum != 20 and not cheap_only:
        proof_key["minimum_observations"] = minimum
    if proof.get("key") != proof_key:
        proof = {}
    checked = set(proof.get("checked_sequences", []))
    allowed = set(proof.get("eligible_sequences", [])) & checked
    novelty_checked = set(proof.get("novelty_checked_sequences", []))
    novel = set(proof.get("novel_sequences", [])) & novelty_checked
    pending = candidates - novelty_checked
    if getattr(args, "adaptive_confirmation_depth", False) and len(pending) > 4096:
        # Revalidation must yield usable current proof before the whole bank
        # finishes. Old eligibility is ordering only, never admission.
        from nanobody_opt.filter_layer2 import _developability_cache_digest, cached_developability_result
        cache_keys = {p.stem for p in args.layer2_cache_dir.glob("*.json")}
        measured = {s for s in pending if _developability_cache_digest(s) in cache_keys
                    and (cached_developability_result(s, args.layer2_cache_dir) or {}).get("passed") is True}
        priority = [*sorted(measured), *sorted((prior_eligible & pending) - measured),
                    *sorted(pending - measured - prior_eligible)]
        pending = set(priority[:4096])

    def checkpoint():
        _atomic_json(proof_path, {"key": proof_key,
            "novelty_checked_sequences": sorted(novelty_checked),
            "novel_sequences": sorted(novel),
            "checked_sequences": sorted(checked), "eligible_sequences": sorted(allowed),
            "checked_count": len(candidates & checked),
            "eligible_count": len(candidates & allowed),
            "pending_count": len(candidates - checked),
        })

    if pending:
        inputs = epoch_root / "comparison_precheck_input.json"
        novelty = epoch_root / "comparison_precheck_novelty.json"
        _atomic_json(inputs, {"target": target, "sequences": sorted(pending)})
        novelty.unlink(missing_ok=True)
        run(_novelty_command(
            pixi_manifest=pixi_manifest, rankings=inputs, archive=archive,
            target=target, output=novelty, pixi=args.pixi,
        ) + ["--summary-only"])
        passed_novelty = _validator_novel_sequences(novelty)
        novelty_result = json.loads(novelty.read_text())
        if novelty_result.get("input_count") != len(pending) or not passed_novelty <= pending:
            raise RuntimeError("comparison novelty proof does not cover its input")
        novelty_checked.update(pending)
        novel.update(passed_novelty)
        checked.update(pending - passed_novelty)
        checkpoint()

    # Revalidate available measurements first so a cold historical TNP backlog
    # cannot withhold an otherwise ready fleet. Advance every unchecked entry
    # over subsequent cycles; this is a checkpoint size, never a pool cutoff.
    from nanobody_opt.filter_layer2 import _developability_cache_digest, _measured_developability
    waiting = (candidates & novel) - checked
    cached = set()
    for sequence in (() if cheap_only else waiting):
        measurement = _read_json(args.layer2_cache_dir / (_developability_cache_digest(sequence) + ".json"))
        if (measurement and measurement.get("sequence") == sequence
                and _measured_developability(measurement.get("developability"))):
            cached.add(sequence)
    if cheap_only and getattr(args, "adaptive_confirmation_depth", False):
        from nanobody_opt.filter_layer2 import cached_developability_result
        measured = {s for s in waiting if
                    (cached_developability_result(s, args.layer2_cache_dir) or {}).get("passed") is True}
        ordered_waiting = [*sorted(measured), *sorted((prior_eligible & waiting) - measured),
                           *sorted(waiting - measured - prior_eligible)]
    else:
        ordered_waiting = sorted(waiting)
    batch = (set(ordered_waiting[:2048]) if cheap_only
             else cached or set(sorted(waiting)[:32]))
    if not ready_artifact and candidates & allowed:
        # After a restart or interrupted publish, deliver the proven fleet
        # before starting any further cold work. Resume the backlog next cycle.
        batch = set()
    if batch:
        layer2_input = epoch_root / "comparison_precheck_layer2_input.json"
        layer2 = epoch_root / "comparison_precheck_layer2.json"
        _atomic_json(layer2_input, {"target": target, "sequences": sorted(batch)})
        layer2.unlink(missing_ok=True)
        run(_layer2_command(
            pixi_manifest=pixi_manifest, rankings=layer2_input, config=config,
            output=layer2, concurrency=args.layer2_concurrency,
            cache_dir=args.layer2_cache_dir, pixi=args.pixi,
        ) + (["--igblast-only"] if cheap_only else []))
        passed = ({r["sequence"] for r in json.loads(layer2.read_text()).get("candidates", [])
                   if r.get("igblast_passed") is True}
                  if cheap_only else _current_layer2_passes(layer2))
        if json.loads(layer2.read_text()).get("input_count") != len(batch) or not passed <= batch:
            raise RuntimeError("comparison Layer-2 proof does not cover its input")
        checked.update(batch)
        allowed.update(passed)
        checkpoint()
    eligible = candidates & allowed
    output = epoch_root / "comparison_candidate_scope.txt"
    output.write_text("".join(s + "\n" for s in sorted(eligible)))
    print(f"COMPARISON_PRECHECK candidates={len(candidates)} new_checks={len(pending)} "
          f"eligible={len(eligible)} pending_checks={len(candidates - checked)}", flush=True)
    if not eligible and not cheap_only:
        raise RuntimeError("no comparison candidate passes current pre-checks")
    return output


def _restrict_pool(
    payload: dict[str, Any], allowed: set[str], *, gate: str, source: Path
) -> dict[str, Any]:
    """Drop ranked candidates the gate rejects, keeping win-probability order.

    Order is preserved, so the ranking's own monotonicity assertion still holds
    on the subsequence.
    """
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit("ranking contains no candidate list")
    kept = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and str(candidate.get("sequence") or "").strip().upper() in allowed
    ]
    if not kept:
        raise SystemExit(
            f"no ranked candidate passed {gate} in {source}: there is nothing "
            "publishable this epoch"
        )
    if len(kept) < len(candidates):
        print(
            f"{gate} pool: {len(kept)} of {len(candidates)} ranked "
            "candidate(s) are assignable",
            flush=True,
        )
    return {**payload, "candidates": kept}


#: How many of the archive's most recent rows count as "where the field is
#: standing now". The archive is append-ordered, so the tail is the recent
#: epochs.
FIELD_CROWD_ROWS = 600


def _recent_field_sequences(archive: Path, rows: int = FIELD_CROWD_ROWS) -> list[str]:
    """What the field submitted recently, as a collision hazard list.

    Our own past submissions are in there too and are harmless to include: we
    never resubmit them, so they only ever push us further from a crowded spot.
    """
    try:
        with archive.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sequence" not in reader.fieldnames:
                return []
            tail: deque[str] = deque(maxlen=rows)
            for row in reader:
                sequence = str(row.get("sequence") or "").strip().upper()
                if sequence:
                    tail.append(sequence)
    except OSError:
        return []
    return list(tail)


def _cdr_command(
    *, pixi_manifest: Path, rankings: Path, output: Path, pixi: Path
) -> list[str]:
    """Annotate the ranked pool with CDRs so the fleet can avoid a clonotype.

    Whole-sequence identity is the coarse tier of the validator's duplicate
    rule and not the one that binds a fleet drawn from one lineage: epoch
    25100's seven tickets were all below 0.95 to one another and five were
    discarded anyway, on CDR3s one or two residues apart in seventeen.
    """
    return [
        str(pixi), "run", "--manifest-path", str(pixi_manifest),
        "-e", "metanano", "annotate-cdrs", "--",
        "--input", str(rankings),
        "--output", str(output),
    ]


def _annotated_cdrs(path: Path) -> dict[str, dict[str, str]]:
    """CDRs by sequence, or nothing if the annotation is unavailable.

    Missing annotation degrades to identity-only separation, which is what we
    had before; it must never stop an epoch from publishing.
    """
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}
    cdrs = payload.get("cdrs")
    return cdrs if isinstance(cdrs, dict) else {}


def _publish_fleet(
    *,
    args: argparse.Namespace,
    crew: Sequence[fleet_module.FleetMember],
    rankings: Path,
    layer2: Sequence[Path],
    archive: Path,
    config: Path,
    target: str,
    epoch: int,
    block: int,
    epoch_length: int,
    stage: str,
    bgen_python: Path,
    nanobody_opt_root: Path,
    staging_root: Path,
    novelty: Path,
    cdr_annotations: Path,
) -> tuple[dict[str, Any], list[fleet_module.Assignment]]:
    """Bind k distinct sequences to k uids and build the single fleet artifact."""
    payload = json.loads(rankings.read_text())
    if getattr(args, "adaptive_confirmation_depth", False):
        payload = _restrict_pool(payload,
            {r["sequence"] for r in payload.get("candidates", [])
             if int(r.get("observation_count", 0)) >= _strict_minimum_observations()},
            gate="independent-observations", source=rankings)
    # Assign only from sequences that hold a CURRENT Layer-2 pass. The ranking
    # is ordered by win probability and knows nothing about Layer-2, so the
    # top-k slice routinely contained sequences publish_checkpoint.py then
    # refused with "winner lacks a current Layer-2 pass": in epoch 25092 only
    # three of the top six passed. layer2[0] is this epoch's file; the rest are
    # historical and are not what "current" means.
    payload = _restrict_pool(
        payload,
        _current_layer2_passes(layer2[0]),
        gate="Layer-2",
        source=layer2[0],
    )
    # And the validator's own duplicate gates. Distinctness is not novelty: a
    # sequence within 0.95 of a TOP-50 historical sequence (or sharing its CDR3
    # at >=0.85) is discarded before scoring, which is how the six epoch-25093
    # tickets died. That is what this gate catches, and it is the only identity
    # rule the validator applies -- it compares each uid against the historical
    # index, never our simultaneous tickets against each other. Near-neighbours
    # that clear THIS gate are separate folds and separate draws: measured on
    # the leaderboard, 780 same-epoch same-coldkey pairs at 0.98-0.999 identity
    # score a median 45.5 rank-sum apart, while the 10 pairs at exactly 1.0
    # share one score. See nanobody_opt/fleet.MAXIMUM_FLEET_IDENTITY.
    payload = _restrict_pool(
        payload,
        _validator_novel_sequences(novelty),
        gate="validator-novelty",
        source=novelty,
    )
    # Deliberately NOT crowd-aware. In epoch 25099 our uid 12 lost a duplicate
    # collision to another miner at 0.9528, which argued for avoiding molecules
    # the field is sitting on -- but measured against epoch 25100's pool, 349
    # of 350 candidates are within 0.95 of the archive's recent tail and the
    # single clear one is the WORST in the pool: p90 230.0 and win probability
    # 0.0043, against 108.2 and 0.0431 for the best contested candidate.
    #
    # The whole field has converged on one family, so "uncrowded" selects for
    # distance from what works. A collision costs one ticket; leading with the
    # worst molecule costs the epoch. _recent_field_sequences and
    # fleet.uncrowded are kept for when the pool is no longer degenerate.
    # The GPU producer does not own registrations.  Always prepare a ranked,
    # UID-independent pool large enough for the Pi's target fleet, even when
    # the producer's own chain snapshot temporarily sees fewer uids.  One more
    # sequence remains reserved for a uid gained inside the commit window.
    SPARE_TICKETS = 1
    minimum_pool_size = int(
        getattr(args, "minimum_sequence_pool_size", len(crew))
    )
    pool_target = max(len(crew), minimum_pool_size)
    sequences = fleet_module.distinct_sequences(
        payload, pool_target + SPARE_TICKETS,
        cdrs=_annotated_cdrs(cdr_annotations),
    )
    # ``distinct_sequences`` is deliberately conservative: it stops before
    # assigning two tickets above the validator's pairwise identity gate.  A
    # short compatible fleet must not truncate the handoff artifact, though.
    # Export the compatible set first, followed by exact-distinct candidates
    # from the already Layer-2-passed and validator-novel ranked pool.  The Pi
    # owns the live fleet and makes the final assignment decision.
    ranked_unique: list[str] = []
    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, Mapping):
            continue
        sequence = str(candidate.get("sequence") or "").strip().upper()
        if sequence and sequence not in ranked_unique:
            ranked_unique.append(sequence)
    handoff_candidates: list[str] = []
    for sequence in [*sequences, *ranked_unique]:
        if sequence not in handoff_candidates:
            handoff_candidates.append(sequence)
    if minimum_pool_size > 0 and len(handoff_candidates) < pool_target:
        # Soft target, not a hard exit: publishing the safely-assignable fresh
        # entries beats a whole-fleet outage when the pool is a few short.
        uncovered = max(0, pool_target - len(handoff_candidates))
        print(
            f"nanobody sequence pool shortfall: need {pool_target} exact-distinct "
            f"Layer-2-passed validator-novel sequences, got "
            f"{len(handoff_candidates)}; publishing the available pool and "
            f"reporting {uncovered} uncovered uids",
            flush=True,
        )
    fleet_sequences = sequences[: len(crew)]
    sequence_pool = handoff_candidates[:pool_target]
    assigned_sequences = set(fleet_sequences)
    spare_sequences = [
        sequence for sequence in handoff_candidates
        if sequence not in assigned_sequences
    ]
    try:
        blocks = fleet_module.commitment_blocks(
            epoch=epoch,
            epoch_length=epoch_length,
            reserve_blocks=args.reserve_blocks,
            count=len(fleet_sequences),
            block=block if block >= 0 else None,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    assignments = fleet_module.assign_sequences(crew, fleet_sequences, blocks)
    short = fleet_module.uncovered(crew, assignments)
    if short:
        print(
            "NANOBODY_FLEET_SHORTFALL only "
            f"{len(assignments)} distinct qualified sequence(s) for {len(crew)} "
            "uid(s); these uids publish nothing this epoch: "
            + ", ".join(f"uid {member.uid}" for member in short),
            file=sys.stderr,
            flush=True,
        )
    requirements = yaml.safe_load(config.read_text())["nanobody_requirements"]
    checkpoints: dict[str, dict[str, Any]] = {}
    # Per-uid, isolated, for the same reason the commit loop is: one uid whose
    # winner "lacks a current Layer-2 pass" used to abort the whole fleet from
    # inside this loop, so the uids already staged never reached an artifact and
    # nothing committed. Epochs 25089 and 25091 published two uids of six that
    # way. A uid that cannot be published is reported as uncovered, which is
    # exactly what the shortfall path already means.
    published: list[fleet_module.Assignment] = []
    rejected: list[tuple[int, str]] = []
    for assignment in assignments:
        destination = staging_root / stage / f"uid{assignment.uid}.json"
        command = [
            str(bgen_python), str(nanobody_opt_root / "publish_checkpoint.py"),
            "--rankings", str(rankings),
        ]
        for path in layer2:
            command.extend(["--layer2", str(path)])
        command.extend([
            "--archive", str(archive),
            "--config", str(config),
            "--target", target,
            "--epoch", str(epoch),
            "--sequence", assignment.sequence,
            "--output", str(destination),
        ])
        if getattr(args, "adaptive_confirmation_depth", False):
            command.extend(["--minimum-observations", str(_strict_minimum_observations())])
        if stage == "provisional":
            command.append("--provisional")
        try:
            _run(command)
        except subprocess.CalledProcessError as error:
            rejected.append((assignment.uid, f"publisher exited {error.returncode}"))
            continue
        checkpoint = _read_json(destination)
        if not _is_current_qualified(
            checkpoint, target=target, epoch=epoch, requirements=requirements
        ):
            rejected.append((assignment.uid, "invalid checkpoint"))
            continue
        if _checkpoint_sequence(checkpoint) != assignment.sequence:
            # Never publishable: a uid holding a sequence it was not assigned is
            # how two uids end up committing the same molecule, which the
            # validator dedupes to one score.
            rejected.append(
                (assignment.uid, "published a sequence it was not assigned")
            )
            continue
        checkpoints[assignment.sequence] = checkpoint
        published.append(assignment)

    if rejected:
        print(
            f"{PUBLISH_FLEET_REJECTED} {len(rejected)} of {len(assignments)} "
            "uid(s) could not be published this epoch: "
            + ", ".join(f"uid {uid} ({why})" for uid, why in rejected),
            file=sys.stderr,
            flush=True,
        )
    if not published:
        raise SystemExit(
            f"no uid could be published for epoch {epoch}: "
            + ", ".join(f"uid {uid} ({why})" for uid, why in rejected)
        )

    artifact = fleet_module.fleet_artifact(
        target=target,
        epoch=epoch,
        stage=stage,
        assignments=published,
        checkpoints=checkpoints,
        unassigned=fleet_module.uncovered(crew, published),
        sequence_pool=sequence_pool,
    )
    if getattr(args, "_confirmation_depth", None) is not None:
        artifact["confirmation_depth"] = args._confirmation_depth
    if getattr(args, "_submission_timing", None) is not None:
        _COMMAND_DEADLINE.remaining()
        current = _chain_view(args.chain_state, args.netuid, args.maximum_chain_state_age)
        artifact["submission_handoff"] = submission_timing.publication_receipt(
            epoch=epoch, timing=args._submission_timing,
            confirmation=args._confirmation_at_start, block=current.block,
            epoch_length=epoch_length)
    return artifact, published, CommitContext(
        spares=tuple(spare_sequences),
        known_uids=frozenset(member.uid for member in crew),
    )


BLOCK_SECONDS = 12.0
# The validator reads a commitment only while start_block < commit_block <
# end - BLACKOUT_BLOCKS. Canonical definition lives in nanobody_opt.fleet; the
# publisher re-exports it so the reserve schedule and the commit wait share one
# blackout boundary.
BLACKOUT_BLOCKS = fleet_module.BLACKOUT_BLOCKS
# Poll faster than the miner rewrites the chain view (~11s), so the reserve
# opening is noticed within a block rather than two or three. Epoch 25098's
# first commit landed at elapsed 313 against a schedule of 306, and epoch
# 25097's winner committed at 306: the gap is polling lag, and ties go to the
# earlier block.
COMMIT_WAIT_POLL_SECONDS = 4.0
PUBLISH_FLEET_REJECTED = "NANOBODY_FLEET_UID_REJECTED"
COMMIT_FLEET_FAILED = "COMMIT_FLEET_UID_FAILED"
# Registration can change after the GPU host's chain snapshot and before the
# Pi's commit window. Prepare headroom for a 13-member fleet plus churn; the Pi
# remains the only authority that binds this ranked pool to UIDs.
MINIMUM_SEQUENCE_POOL_SIZE = 16
COMMIT_FLEET_INCOMPLETE = "COMMIT_FLEET_INCOMPLETE"


def _sleep(seconds: float) -> None:
    """Indirection so a test can advance the chain instead of real time."""
    time.sleep(seconds)


def _await_commit_block(
    args: argparse.Namespace, target_block: int, epoch: int
) -> bool:
    """Hold until the chain reaches this assignment's scheduled block.

    Every scored submission in the field commits inside the end-of-epoch
    reserve: in epoch 25093 the leaderboard's blocks_elapsed ran 306 to 344 of
    361, and our two scored uids sat at 336 and 344 while the four we committed
    early were not scored at all. fleet.commitment_blocks() already computes
    that schedule; this is what honours it.

    The wait is bounded by the epoch and by the blackout: the validator reads a
    commit only while commit_block < end - BLACKOUT_BLOCKS, so a scheduled block
    in the blackout -- or a live block that drifted there while we waited --
    returns False and the caller skips the commit rather than spending an
    extrinsic the validator will never read.
    """
    while True:
        chain = _chain_view(args.chain_state, args.netuid, args.maximum_chain_state_age)
        if chain.epoch != epoch:
            raise SystemExit(
                f"epoch rolled to {chain.epoch} while waiting for the "
                f"commitment reserve of epoch {epoch}"
            )
        epoch_end = (epoch + 1) * int(chain.epoch_length)
        if target_block >= epoch_end - BLACKOUT_BLOCKS:
            return False
        if chain.block >= epoch_end - BLACKOUT_BLOCKS:
            return False
        if chain.block >= target_block:
            return True
        remaining = target_block - chain.block
        _sleep(min(BLOCK_SECONDS * remaining, COMMIT_WAIT_POLL_SECONDS))


def _committed_this_epoch(ledger: Path, epoch: int) -> set[int]:
    """uids that already hold a commitment written this epoch."""
    payload = _read_json(ledger)
    if not isinstance(payload, dict) or int(payload.get("epoch", -1)) != int(epoch):
        return set()
    return {int(uid) for uid in payload.get("uids", [])}


def _record_commitment(ledger: Path, epoch: int, uid: int) -> None:
    """Remember it immediately, so a crash mid-fleet does not re-commit.

    Only ever called once the commitment is known to belong to a uid we hold.
    See ``_commitment_is_ours``: a commitment is written against a HOTKEY, not a
    uid, so an extrinsic sent with a hotkey that has since been deregistered
    succeeds and lands nowhere the validator will read for that uid. Recording
    that as done also suppresses the retry, because the ledger is what the next
    pass consults.
    """
    committed = _committed_this_epoch(ledger, epoch)
    committed.add(int(uid))
    _atomic_json(ledger, {"epoch": int(epoch), "uids": sorted(committed)})


def _commitment_is_ours(
    held: dict[int, str] | None,
    assignment: fleet_module.Assignment,
) -> bool:
    """Whether a commitment just sent actually lands on a uid we hold.

    ``held`` is a fresh uid -> hotkey view of our own coldkey. ``None`` means the
    chain could not be re-read, and an unverifiable commitment is not recorded:
    the cost of not recording is one duplicate attempt next pass, which the
    chain's own rate limit refuses harmlessly. The cost of recording wrongly is
    a ticket silently forfeited for the epoch.
    """
    if held is None:
        return False
    return held.get(assignment.uid) == assignment.hotkey


FLEET_UID_LOST = "NANOBODY_FLEET_UID_LOST"
# A uid the chain handed us DURING the reserve, after the fleet was planned.
FLEET_UID_GAINED = "NANOBODY_FLEET_UID_GAINED"
# A planned uid dropped at its own commit block, before the extrinsic was sent.
FLEET_UID_SKIPPED = "NANOBODY_FLEET_UID_SKIPPED"
# An extrinsic went out but the chain does not show it landing on a uid we hold.
COMMIT_UNVERIFIED = "COMMIT_FLEET_UID_UNVERIFIED"


HOTKEY_UNREGISTERED = "NANOBODY_HOTKEY_UNREGISTERED"


def report_unregistered_hotkeys(
    *, wallet_root: Path, wallet_name: str, chain: ChainView, coldkey: str
) -> list[str]:
    """Name the local hotkeys the chain does not register to us. Report only.

    The box held sixteen key files for seven registered uids, so knowing which
    are live is worth having. Moving them automatically is not: a hotkey that
    loses its slot VANISHES from the metagraph, which is indistinguishable from
    one the purchase agent has just created and not yet registered. Archiving
    on absence therefore deletes the next slot we were about to buy, and in
    testing it did exactly that. Purging stays a deliberate act.
    """
    ours = {
        str(row["hotkey"])
        for row in chain.neurons
        if str(row.get("coldkey") or "") == coldkey
    }
    if not ours:
        # Discovering nothing is a bad chain view, not an empty fleet.
        return []
    addresses = fleet_module.local_hotkey_addresses(wallet_root, wallet_name)
    idle = sorted(name for address, name in addresses.items() if address not in ours)
    if idle:
        print(
            f"{HOTKEY_UNREGISTERED} {len(idle)} of {len(addresses)} local "
            f"hotkey(s) are not registered to us: {', '.join(idle)}",
            flush=True,
        )
    return idle


def verify_fleet_ownership(
    assignments: Sequence[fleet_module.Assignment],
    chain: ChainView,
    epoch: int,
) -> list[int]:
    """Re-check, after committing, that every uid we signed for is still ours.

    The commit driver refuses a uid the chain no longer maps to our hotkey, but
    that check happens before the extrinsic and a slot can be recycled between
    then and the validator reading the epoch. In epoch 25101 we committed for
    uid 46 and it scored 131 for another coldkey: the registration, the ticket
    and the epoch's work for that slot were all spent on someone else's uid.

    Reporting only. Nothing can be undone at this point; the value is knowing
    the fleet is smaller than it looks before deciding to buy more.
    """
    held = {int(row["uid"]): str(row["hotkey"]) for row in chain.neurons}
    lost: list[int] = []
    for assignment in assignments:
        if held.get(assignment.uid) != assignment.hotkey:
            lost.append(assignment.uid)
            print(
                f"{FLEET_UID_LOST} uid={assignment.uid} "
                f"hotkey={assignment.hotkey} is no longer ours at epoch "
                f"{epoch}; that commitment scores for whoever holds it now",
                file=sys.stderr,
                flush=True,
            )
    if lost:
        print(
            f"fleet ownership: {len(assignments) - len(lost)} of "
            f"{len(assignments)} committed uid(s) still ours",
            flush=True,
        )
    return lost


@dataclass(frozen=True)
class CommitContext:
    """What the commit path needs beyond the assignments themselves.

    ``known_uids`` is every uid the chain attributed to us when the fleet was
    planned, including any the publisher then rejected. A uid missing from it is
    one the chain handed us DURING the reserve; a uid present in it but absent
    from the assignments was deliberately left out, and must not be revived.
    """

    spares: tuple[str, ...] = ()
    known_uids: frozenset[int] = frozenset()


def _next_reserve_block(args: argparse.Namespace, epoch: int) -> int | None:
    """The next block inside this epoch's reserve that has not gone by yet.

    A uid gained mid-reserve was not in the planned schedule, so it has no block
    of its own. One extrinsic per block is the pace the chain accepts, so it
    takes the next one still ahead of us, and None once the epoch's reserve is
    spent -- where "spent" now means the blackout boundary (end - BLACKOUT_BLOCKS),
    not the literal rollover, because a commit at or after end - 16 is never read.
    """
    view = _chain_view(args.chain_state, args.netuid, args.maximum_chain_state_age)
    block = int(view.block)
    epoch_end = (epoch + 1) * int(view.epoch_length)
    candidate = block + 1
    return candidate if candidate < epoch_end - BLACKOUT_BLOCKS else None


def load_molecules_by_uid(path: str | None) -> dict[int, str]:
    """Per-uid compound submissions, keyed by uid, as a comma-separated string.

    FAIL-SAFE BY CONSTRUCTION. A missing file, an unreadable one, or a uid with
    no entry all yield "", and commit_submission then emits the same "~|<seq>"
    this fleet has always sent. Compounds must never be able to cost a nanobody
    submission -- the commitment is ONE message carrying both sections, so a
    molecule list that arrives malformed would otherwise take the sequence down
    with it.
    """
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        print(f"MOLECULES_UNREADABLE path={path} ({error}); "
              f"committing nanobodies alone", file=sys.stderr, flush=True)
        return {}
    mapping: dict[int, str] = {}
    for key, value in (payload or {}).items():
        try:
            uid = int(key)
        except (TypeError, ValueError):
            continue
        names = value.split(",") if isinstance(value, str) else list(value or [])
        names = [str(name).strip() for name in names if str(name).strip()]
        if not names or len(set(names)) != len(names):
            continue
        if any(any(c.isspace() for c in name) or "|" in name or not name.isascii()
               for name in names):
            print(f"MOLECULES_MALFORMED uid={uid}; skipping its compound section",
                  file=sys.stderr, flush=True)
            continue
        mapping[uid] = ",".join(names)
    return mapping


def _commit_fleet(
    assignments: Sequence[fleet_module.Assignment],
    *,
    args: argparse.Namespace,
    template: str | None,
    artifact: Path,
    epoch: int,
    pause_seconds: float,
    ledger: Path,
    coldkey: str | None = None,
    context: CommitContext | None = None,
) -> None:
    """Send one commitment per uid, one block apart, inside the reserve.

    Sent as fast as the chain accepts them. ``max_regs_per_block`` is 1, but
    that caps REGISTRATIONS, not commitments: in epoch 25096 uid 60 and uid 61
    committed in the same block from different hotkeys. Pacing cost us that
    epoch -- we tied the winner's score of 43.0 exactly and lost the tiebreak,
    which is block_submitted, because our fleet landed at elapsed 328-350
    against their 326.  With no command configured the per-uid miner processes
    commit from the artifact themselves and this is a no-op; the schedule is
    recorded either way.
    """
    if not template:
        return
    # One commitment per uid per epoch. The publisher republishes several times
    # inside an epoch and used to re-commit the whole fleet each pass; the
    # chain's commitment rate limit then refused the later ones, and a uid
    # whose only attempt that epoch was refused kept a commitment from the
    # PREVIOUS epoch. Those fall outside the validator's window -- it reads
    # start_block < commit < end - 10 -- and are never read at all. Epoch
    # 25101 had 49 commitments read of a 64-uid subnet, and three of ours
    # never reached the nanobody stage.
    already = _committed_this_epoch(ledger, epoch)
    pending = [item for item in assignments if item.uid not in already]
    if len(pending) < len(assignments):
        print(
            f"commit: {len(assignments) - len(pending)} uid(s) already hold a "
            f"commitment for epoch {epoch}; committing {len(pending)}",
            flush=True,
        )
    assignments = pending
    if not assignments:
        return
    failures: list[tuple[int, int]] = []
    # Ownership is resolved once when the fleet is planned, but the commits then
    # wait out the reserve one block at a time -- twenty minutes or more. Subnet
    # 68 is full, so a registration recycles the lowest pruning score and a slot
    # can change hands inside that gap. In epoch 25114 uid 59 passed to another
    # coldkey 108 blocks before we committed to it, and the ticket was spent on
    # their uid. Re-reading the chain at each block is what closes that window;
    # the post-commit verification below only reports the loss afterwards.
    def _fresh_fleet() -> dict[int, str] | None:
        if not coldkey:
            return None
        try:
            view = _chain_view(
                args.chain_state, args.netuid, args.maximum_chain_state_age
            )
        except Exception as error:  # a stale read must not cancel the commits
            print(f"commit: chain re-read failed ({error}); committing on the "
                  f"planned fleet, and ownership is UNVERIFIED for this uid",
                  file=sys.stderr, flush=True)
            return None
        return {int(row["uid"]): str(row["hotkey"]) for row in view.neurons
                if str(row.get("coldkey") or "") == coldkey}

    context = context or CommitContext()
    # Seeded with every uid known at plan time, so a uid the publisher rejected
    # is not mistaken for one the chain just handed us.
    planned = {assignment.uid for assignment in assignments} | set(context.known_uids)
    unused = [str(item).strip().upper() for item in context.spares]
    molecules_by_uid = load_molecules_by_uid(getattr(args, "molecules_file", None))
    if molecules_by_uid:
        print(f"MOLECULES_LOADED uids={sorted(molecules_by_uid)}", flush=True)
    gained: list[fleet_module.Assignment] = []

    for index, assignment in enumerate(assignments):
        command = [
            part.format(
                uid=assignment.uid,
                hotkey=assignment.hotkey,
                sequence=assignment.sequence,
                molecules=molecules_by_uid.get(assignment.uid, ""),
                block=assignment.commit_block,
                epoch=epoch,
                artifact=str(artifact),
            )
            for part in shlex.split(template)
        ]
        # The schedule, not a fixed delay: one block apart INSIDE the reserve.
        if not _await_commit_block(args, assignment.commit_block, epoch):
            print(
                f"commit: uid {assignment.uid} skipped; its scheduled block "
                f"{assignment.commit_block} falls inside the validator's "
                f"blackout (end - {BLACKOUT_BLOCKS})",
                flush=True,
            )
            unused.insert(0, assignment.sequence)
            continue

        held = _fresh_fleet()
        if held is not None:
            if held.get(assignment.uid) != assignment.hotkey:
                # Spending the extrinsic now would commit our sequence to a uid
                # that scores for whoever holds it. Keep the sequence instead:
                # it is still distinct and a gained uid below can carry it.
                print(
                    f"{FLEET_UID_SKIPPED} uid={assignment.uid} "
                    f"hotkey={assignment.hotkey} left our coldkey before its "
                    f"commit block {assignment.commit_block}; not committing",
                    file=sys.stderr,
                    flush=True,
                )
                unused.insert(0, assignment.sequence)
                continue
            for uid, hotkey in sorted(held.items()):
                if uid in planned or uid in already:
                    continue
                planned.add(uid)
                gained.append(
                    fleet_module.Assignment(
                        uid=uid, hotkey=hotkey, sequence="", commit_block=0
                    )
                )
                print(
                    f"{FLEET_UID_GAINED} uid={uid} hotkey={hotkey} joined our "
                    f"coldkey during epoch {epoch}'s reserve",
                    flush=True,
                )

        if index and pause_seconds > 0:
            time.sleep(pause_seconds)
        try:
            _run(command)
            if _commitment_is_ours(_fresh_fleet(), assignment):
                _record_commitment(ledger, epoch, assignment.uid)
            else:
                print(
                    f"{COMMIT_UNVERIFIED} uid={assignment.uid} "
                    f"hotkey={assignment.hotkey} committed but the chain does "
                    f"not map that hotkey to the uid; not recorded, so the next "
                    f"pass may retry",
                    file=sys.stderr,
                    flush=True,
                )
        except subprocess.CalledProcessError as error:
            # One uid's chain error must never cost the rest of the fleet its
            # commitment: every uid gets its own attempt inside the reserve.
            failures.append((assignment.uid, error.returncode))
            print(
                f"{COMMIT_FLEET_FAILED} uid={assignment.uid} "
                f"hotkey={assignment.hotkey} rc={error.returncode}",
                file=sys.stderr,
                flush=True,
            )
    # A uid gained mid-reserve is a ticket we already paid a registration for.
    # Commit for it from the sequences the planned fleet did not use, in the
    # blocks still left of the reserve, rather than leaving the slot idle for an
    # epoch. Each sequence is distinct and none is committed twice, so this
    # cannot collide with the planned fleet.
    for member in gained:
        if not unused:
            print(
                f"commit: uid {member.uid} joined mid-reserve but no distinct "
                f"sequence is left to commit for it",
                flush=True,
            )
            break
        try:
            block = _next_reserve_block(args, epoch)
        except SystemExit:
            raise
        except Exception as error:
            print(f"commit: no reserve block left for uid {member.uid} ({error})",
                  flush=True)
            break
        if block is None:
            print(f"commit: epoch {epoch}'s reserve is spent; uid {member.uid} "
                  f"waits for the next epoch", flush=True)
            break
        sequence = unused.pop(0)
        late = fleet_module.Assignment(
            uid=member.uid, hotkey=member.hotkey,
            sequence=sequence, commit_block=block,
        )
        if not _await_commit_block(args, block, epoch):
            print(
                f"commit: uid {member.uid} skipped; block {block} is inside "
                f"the validator's blackout (end - {BLACKOUT_BLOCKS})",
                flush=True,
            )
            unused.insert(0, sequence)
            continue
        command = [
            part.format(
                uid=late.uid, hotkey=late.hotkey, sequence=late.sequence,
                molecules=molecules_by_uid.get(late.uid, ""),
                block=late.commit_block, epoch=epoch, artifact=str(artifact),
            )
            for part in shlex.split(template)
        ]
        try:
            _run(command)
            if not _commitment_is_ours(_fresh_fleet(), late):
                print(
                    f"{COMMIT_UNVERIFIED} uid={late.uid} hotkey={late.hotkey} "
                    f"committed but the chain does not map that hotkey to the "
                    f"uid; not recorded",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            _record_commitment(ledger, epoch, late.uid)
            assignments = list(assignments) + [late]
        except subprocess.CalledProcessError as error:
            failures.append((late.uid, error.returncode))
            print(
                f"{COMMIT_FLEET_FAILED} uid={late.uid} hotkey={late.hotkey} "
                f"rc={error.returncode}",
                file=sys.stderr,
                flush=True,
            )

    try:
        verify_fleet_ownership(
            assignments,
            _chain_view(args.chain_state, args.netuid, args.maximum_chain_state_age),
            epoch,
        )
    except RuntimeError as error:
        # A stale chain view must not turn a healthy commit run into a failure.
        print(f"fleet ownership unverified: {error}", file=sys.stderr, flush=True)

    if failures:
        detail = ", ".join(f"uid {uid} (rc {rc})" for uid, rc in failures)
        print(
            f"{COMMIT_FLEET_INCOMPLETE} {len(failures)} of "
            f"{len(assignments)} uid(s) did not commit this epoch: {detail}",
            file=sys.stderr,
            flush=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    submission_timing.add_arguments(parser)
    from nanobody_opt.confirmation_depth import add_arguments
    add_arguments(parser)
    parser.add_argument("--network", default="finney")
    parser.add_argument("--netuid", type=int, default=68)
    parser.add_argument(
        "--target",
        help="optional target assertion; production resolves the target online",
    )
    parser.add_argument("--expected-gpus", type=int, default=10)
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument("--score-bank-root", type=Path, default=Path("/root"))
    parser.add_argument(
        "--nanobody-opt-root",
        type=Path,
        default=Path("/root/nanobody_opt"),
        help="deployed producer source; the runtime root holds no other copy",
    )
    parser.add_argument("--pixi", type=Path, default=Path("/usr/local/bin/pixi"))
    parser.add_argument("--layer2-concurrency", type=int, default=40)
    parser.add_argument(
        "--shortlist-limit",
        type=int,
        default=SHORTLIST_LIMIT,
        help=(
            "minimum number of ranked candidates carried into Layer 2; "
            "fleet-size coverage can raise this floor"
        ),
    )
    parser.add_argument(
        "--layer2-cache-dir", type=Path, default=Path("/root/layer2_cache")
    )
    parser.add_argument(
        "--chain-state",
        type=Path,
        default=Path("/root/artifacts/chain_state.json"),
    )
    parser.add_argument("--maximum-chain-state-age", type=float, default=90)
    parser.add_argument(
        "--artifact", type=Path, default=Path("/root/artifacts/nanobody_result.json")
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("/root/artifacts/nanobody_history"),
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("/root/artifacts/validator_audits"),
    )
    parser.add_argument(
        "--coldkey",
        help=(
            "our coldkey ss58; the fleet is discovered from the chain by this "
            "address every epoch. Defaults to the wallet's public coldkeypub."
        ),
    )
    parser.add_argument("--wallet-name", default="mining_ops")
    parser.add_argument(
        "--wallet-path", type=Path, default=Path.home() / ".bittensor" / "wallets"
    )
    parser.add_argument(
        "--reserve-blocks",
        type=int,
        default=36,
        help="end-of-epoch blocks reserved for commitments, one uid per block",
    )
    parser.add_argument(
        "--minimum-sequence-pool-size",
        type=int,
        default=MINIMUM_SEQUENCE_POOL_SIZE,
        help=(
            "minimum UID-independent ranked sequence pool prepared for the "
            "submit host, regardless of the producer's current crew view"
        ),
    )
    parser.add_argument(
        "--molecules-file",
        help=(
            "JSON {uid: \"name1,name2,...\"} of per-uid compound submissions. "
            "Absent or unreadable commits nanobodies alone, unchanged."
        ),
    )
    parser.add_argument(
        "--commit-command",
        help=(
            "optional per-uid commitment command; {uid} {hotkey} {sequence} "
            "{molecules} {block} {epoch} {artifact} are substituted"
        ),
    )
    parser.add_argument(
        "--strict-qualification",
        action="store_true",
        help=(
            "run the strict rank_exact upgrade after the provisional fleet "
            "has committed. It costs about a third of an epoch, so it is "
            "gated on --strict-qualification-reserve-blocks"
        ),
    )
    parser.add_argument(
        "--strict-pool-size",
        type=int,
        default=100,
        help=(
            "qualify only the top N candidates of a fresh provisional ranking in the strict pass "
            "(plus anything already published this epoch); zero retains the bounded default of 100"
        ),
    )
    parser.add_argument(
        "--strict-qualification-reserve-blocks",
        type=int,
        default=150,
        help=(
            "only start the optional strict upgrade with this many blocks remaining; "
            "every child command also has a five-minute cap and live epoch cancellation"
        ),
    )
    parser.add_argument(
        "--commit-pause-seconds",
        type=float,
        default=0.0,
        help=(
            "delay between one uid's commitment and the next. Zero: ties are "
            "broken by block_submitted and every block of delay is a tie lost"
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("/root/artifacts/nanobody_published_sequences.txt"),
    )
    parser.add_argument(
        "--lock", type=Path, default=Path("/run/lock/nova-nanobody-publisher.lock")
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("/root/artifacts/nanobody_publisher_state.json"),
        help="processed exact-bank state, independent of checkpoint replacement",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="assert the miner's chain-state epoch equals this; never a substitute",
    )
    parser.add_argument("--exact", action="append", type=Path)
    parser.add_argument(
        "--observation-database",
        type=Path,
        default=Path("/root/artifacts/nanobody_mutations.sqlite"),
        help=(
            "authoritative sealed raw exact-run store; normal publication "
            "ingests only current-epoch exact_all.json files, then ranks this "
            "database without scanning historical files"
        ),
    )
    parser.add_argument("--historical-layer2", action="append", type=Path)
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=0,
        help="repeat forever at this interval, logging failures and retrying",
    )
    return parser


def _publish_once(args: argparse.Namespace) -> None:
    # Reset even on failure/rollover; a ten-pass decision belongs to one epoch.
    with publication_gates.evidence_floor():
        return _publish_once_current_epoch(args)


def _publish_once_current_epoch(args: argparse.Namespace) -> None:
    global _COMMAND_DEADLINE
    with _exclusive_lock(args.lock):
        if args.expected_gpus and _gpu_count() != args.expected_gpus:
            raise SystemExit(
                f"expected {args.expected_gpus} GPUs on the nanobody host; fail closed"
            )
        chain = _chain_view(
            args.chain_state, args.netuid, args.maximum_chain_state_age
        )
        epoch, block, epoch_length = chain.epoch, chain.block, chain.epoch_length
        args._submission_timing = submission_timing.configured_plan(args)
        args._confirmation_at_start = submission_timing.confirmation_digest(args.runtime_root, epoch)
        publication_reserve = max(args.reserve_blocks,
            (args._submission_timing or {}).get("publication_reserve_blocks", 0))
        _COMMAND_DEADLINE = EpochDeadline(args.chain_state, epoch, publication_reserve,
                                         args.maximum_chain_state_age)
        if args._submission_timing is not None:
            _COMMAND_DEADLINE.remaining()
        if args.epoch is not None and args.epoch != epoch:
            raise SystemExit(
                f"requested epoch {args.epoch} is not the chain epoch {epoch}"
            )
        print(f"current epoch={epoch} block={block} epoch_length={epoch_length}")

        # Identity, every epoch, from the chain: uid numbers are recycled and a
        # remembered one silently follows whoever holds it next.
        coldkey = fleet_module.resolve_coldkey(
            args.coldkey,
            wallet_root=args.wallet_path,
            wallet_name=args.wallet_name,
        )
        crew = fleet_module.discover_fleet(chain.neurons, coldkey, netuid=args.netuid)
        args._submission_timing = submission_timing.configured_plan(args, len(crew))
        if args._submission_timing is not None:
            _atomic_json(args.runtime_root / "artifacts/submission_handoff.json",
                         {**args._submission_timing, "epoch": epoch})
            _COMMAND_DEADLINE = EpochDeadline(args.chain_state, epoch,
                max(args.reserve_blocks, args._submission_timing["publication_reserve_blocks"]),
                args.maximum_chain_state_age)
            _COMMAND_DEADLINE.remaining()
        report_unregistered_hotkeys(
            wallet_root=args.wallet_path,
            wallet_name=args.wallet_name,
            chain=chain,
            coldkey=coldkey,
        )
        fleet_module.require_local_hotkeys(crew, args.wallet_path, args.wallet_name)
        fleet_fingerprint = fleet_module.fleet_fingerprint(crew)
        print(
            f"fleet: {len(crew)} registered uid(s) for coldkey {coldkey}: "
            + ", ".join(f"uid {member.uid}={member.hotkey}" for member in crew),
            flush=True,
        )

        epoch_root = args.runtime_root / "live" / f"epoch{epoch}"
        config_path = epoch_root / "config.yaml"
        archive_path = epoch_root / "archive.csv"
        rankings_path = epoch_root / "provisional_rankings.json"
        current_layer2_path = epoch_root / "provisional_layer2.json"
        commit_ledger = epoch_root / "committed_uids.json"
        novelty_path = epoch_root / "provisional_novelty.json"
        cdr_path = epoch_root / "provisional_cdrs.json"
        _atomic_download(CONFIG_URL, config_path)
        config = yaml.safe_load(config_path.read_text())
        configured_target = str(config["protein_selection"]["nanobody"]["target"])
        if args.target is not None and configured_target != args.target:
            raise SystemExit(
                f"live target is {configured_target!r}, exact bank is {args.target!r}; fail closed"
            )
        target = configured_target
        args._confirmation_depth = None
        if getattr(args, "adaptive_confirmation_depth", False):
            from nanobody_opt import confirmation_depth as depth
            args._confirmation_depth = depth.read(epoch_root / "producer", epoch=epoch, target=target)
            publication_gates.set_evidence_floor((args._confirmation_depth or {}).get(
                "minimum_observations", depth.TARGET))
        requirements = config["nanobody_requirements"]

        previous = _read_json(args.artifact)
        expected_hotkeys = [member.hotkey for member in crew]
        current_qualified = _fleet_is_current(
            previous,
            target=target,
            epoch=epoch,
            requirements=requirements,
            expected_hotkeys=expected_hotkeys,
        )
        bgen_python = args.runtime_root / ".pixi/envs/bgen/bin/python"
        pixi_manifest = args.runtime_root / "pixi.toml"
        if not bgen_python.is_file() or not pixi_manifest.is_file():
            raise SystemExit("locked nanobody Pixi runtimes are not installed")

        # Normal operation uses SQLite as the sole historical query surface.
        # Current runs and a bounded completed shortlist from the prior epoch
        # are ingested once, sealed, then omitted from rank_exact. Explicit
        # --exact remains a break-glass/testing path with the old semantics.
        explicit_exact = tuple(args.exact or ())
        missing_exact = [str(path) for path in explicit_exact if not path.is_file()]
        if missing_exact:
            raise SystemExit(
                "score-bank inputs named on the command line are missing: "
                + ", ".join(missing_exact)
            )
        archive_ready = False
        if explicit_exact:
            exact_inputs = explicit_exact
            observation_history = None
            exact_fingerprint, strict_fingerprint = _exact_bank_fingerprints(
                exact_inputs, target
            )
            sealed_run_count = 0
        else:
            exact_inputs = ()
            observation_history = args.observation_database
            current_inputs = _publication_exact_inputs(
                args.score_bank_root, epoch, target
            )
            descriptors = [
                _exact_run_descriptor(path, target) for path in current_inputs
            ]
            unsealed = [
                descriptor
                for descriptor in descriptors
                if not _exact_run_is_sealed(
                    args.observation_database,
                    target=target,
                    descriptor=descriptor,
                )
            ]
            if unsealed:
                _atomic_download(ARCHIVE_URL.format(target=target), archive_path)
                archive_ready = True
                _run(_import_exact_command(
                    bgen_python=bgen_python,
                    nanobody_opt_root=args.nanobody_opt_root,
                    database=args.observation_database,
                    target=target,
                    archive=archive_path,
                    exact_inputs=[Path(item["path"]) for item in unsealed],
                ))
                incomplete = [
                    str(item["path"])
                    for item in unsealed
                    if not _exact_run_is_sealed(
                        args.observation_database,
                        target=target,
                        descriptor=item,
                    )
                ]
                if incomplete:
                    raise SystemExit(
                        "raw exact ingestion did not seal every current run: "
                        + ", ".join(incomplete)
                    )
            (
                exact_fingerprint,
                strict_fingerprint,
                sealed_run_count,
            ) = _database_exact_fingerprints(args.observation_database, target)
            if not sealed_run_count:
                raise SystemExit(
                    "observation database has no complete sealed raw exact run "
                    f"for {target}; run import_exact_screens once to backfill it"
                )
        _archive_and_audit_previous(
            previous,
            current_epoch=epoch,
            history_dir=args.history_dir,
            audit_dir=args.audit_dir,
            bgen_python=bgen_python,
            nanobody_opt_root=args.nanobody_opt_root,
        )
        retired = _retire_completed_epoch(previous, args.ledger, epoch)
        if retired:
            print(
                f"retired {len(retired)} published sequence(s) from the previous "
                "epoch into the ledger",
                flush=True,
            )
        ledger_fingerprint = _file_digest(args.ledger)
        publisher_state = _read_json(args.state)
        # A fingerprint match means the QUALIFICATION is unchanged, not that
        # every planned uid actually committed. _commit_fleet logs-and-skips
        # failures, and the state is written BEFORE the commit, so a no-op here
        # would silently forfeit any uid whose commit failed. Re-derive the
        # planned uids and retry delivery when any is missing from the commit
        # ledger.
        committed_uids = _committed_this_epoch(commit_ledger, epoch)
        prior_uids = {
            int(assignment.uid)
            for assignment in fleet_module.assignments_from_artifact(previous)
        }
        uncommitted = prior_uids - committed_uids
        if uncommitted:
            print(
                f"qualified epoch {epoch} fleet unchanged but {len(uncommitted)} "
                "uid(s) lack a commitment; retrying delivery",
                file=sys.stderr,
                flush=True,
            )
        effective_ledger = args.ledger
        if getattr(args, "pi_handoff", False):
            from nanobody_opt import publication_ledger
            effective_ledger = publication_ledger.effective(
                args.ledger, epoch=epoch, target=target, history=args.history_dir,
                outcomes=args.runtime_root / "artifacts/pi_submission_outcomes.json",
                output=epoch_root / "submission_exclusions.txt")
        exclusion_path = fleet_module.write_exclusion_file(
            epoch_root / "excluded_sequences.txt",
            effective_ledger,
            _live_sequences(previous, epoch),
        )
        if not archive_ready:
            _atomic_download(ARCHIVE_URL.format(target=target), archive_path)
        # A bank/crew match alone cannot reuse eligibility under changed rules.
        gate_fingerprint = [_file_digest(config_path), _file_digest(archive_path),
                            _file_digest(exclusion_path)]
        if getattr(args, "adaptive_confirmation_depth", False):
            gate_fingerprint.append(args._confirmation_depth)
        if args._submission_timing is not None:
            gate_fingerprint.append(args._confirmation_at_start)
        exact_fingerprint = _sha256_bytes(json.dumps([exact_fingerprint, gate_fingerprint]).encode())
        strict_fingerprint = _sha256_bytes(json.dumps([strict_fingerprint, gate_fingerprint]).encode())

        historical_layer2 = tuple(
            args.historical_layer2 or map(Path, DEFAULT_LAYER2_INPUTS)
        )
        # An input the operator NAMED must exist -- a typo there should stop the
        # run. A default must not: DEFAULT_EXACT_INPUTS and DEFAULT_LAYER2_INPUTS
        # are paths from epoch 25021 on a machine that no longer exists, and
        # requiring them meant a rebuilt box deferred publication for ever while
        # sitting on a perfectly good score bank the glob had already found.
        explicit = [*(args.historical_layer2 or ())]
        missing = [str(path) for path in explicit if not path.is_file()]
        if missing:
            raise SystemExit(
                "score-bank inputs named on the command line are missing: "
                + ", ".join(missing)
            )
        historical_layer2 = tuple(path for path in historical_layer2 if path.is_file())
        # Publishing with no evidence at all is the thing to refuse.
        if explicit_exact and not exact_inputs:
            raise SystemExit(
                "no explicit exact score-bank inputs remain after validation"
            )

        shortlist_floor = int(getattr(args, "shortlist_limit", SHORTLIST_LIMIT))
        if shortlist_floor < len(crew):
            raise SystemExit(
                "shortlist limit must cover every registered uid: "
                f"need at least {len(crew)}, got {shortlist_floor}"
            )
        shortlist = max(shortlist_floor, len(crew) * SHORTLIST_PER_UID)
        if observation_history is not None:
            candidate_scope = _prechecked_comparison_scope(
                args=args, epoch_root=epoch_root, target=target,
                database=observation_history, archive=archive_path, config=config_path,
                exclusion=exclusion_path, pixi_manifest=pixi_manifest,
                ready_artifact=bool(_live_sequences(previous, epoch)),
            )
        else:
            candidate_scope = _producer_candidate_scope(epoch_root, target, previous)
        # Newly checked cached entries can improve the shortlist without a new
        # GPU fold; that progress must invalidate the publication no-op too.
        scope_fingerprint = _file_digest(candidate_scope) if candidate_scope else None
        exact_fingerprint = _sha256_bytes(json.dumps([exact_fingerprint, scope_fingerprint]).encode())
        strict_fingerprint = _sha256_bytes(json.dumps([strict_fingerprint, scope_fingerprint]).encode())
        previous_prechecked = (
            observation_history is None
            or _artifact_passes_scope(previous, epoch, candidate_scope)
        )
        current_qualified = current_qualified and previous_prechecked
        if (
            current_qualified
            and _publisher_state_matches(
                publisher_state,
                epoch=epoch,
                target=target,
                exact_fingerprint=exact_fingerprint,
                ledger_fingerprint=ledger_fingerprint,
                fleet_fingerprint=fleet_fingerprint,
            )
            and not uncommitted
        ):
            print(
                f"qualified epoch {epoch} fleet, score bank and uid set are "
                "unchanged; no-op"
            )
            return
        if current_qualified:
            # The provisional pass is the net that keeps an epoch from submitting
            # nothing. A qualified fleet is already live for this epoch, so the
            # net would only downgrade it; go straight to looking for an upgrade.
            print(
                f"epoch {epoch} already holds a qualified fleet for all "
                f"{len(crew)} uid(s); skipping the provisional pass",
                flush=True,
            )
            assignments = fleet_module.assignments_from_artifact(previous)
            # Skipping the provisional pass also skips the fleet plan that
            # builds the commit context, but both re-commit paths below still
            # need one -- without it this branch raises UnboundLocalError and
            # an already-qualified epoch stops re-committing entirely.  The
            # crew is the same chain view the planned fleet would have used;
            # no spare is known here, and inventing one would revive a uid the
            # earlier pass may have deliberately left out.
            commit_context = CommitContext(
                known_uids=frozenset(member.uid for member in crew),
            )
        else:
            _run(_rank_command(
                bgen_python=bgen_python,
                nanobody_opt_root=args.nanobody_opt_root,
                exact_inputs=exact_inputs,
                archive=archive_path,
                target=target,
                output=rankings_path,
                exclusion=exclusion_path,
                epoch=epoch,
                shortlist=shortlist,
                config=config_path,
                strict=False,
                observation_history=observation_history,
                diversity_reserve_count=len(crew) * 2 + 1,
                only_sequences=candidate_scope,
            ))
            _run(_novelty_command(
                pixi_manifest=pixi_manifest,
                rankings=rankings_path,
                archive=archive_path,
                target=target,
                output=novelty_path,
                pixi=args.pixi,
            ))
            novel_rankings = _novel_layer2_rankings(
                rankings_path, novelty_path,
                epoch_root / "provisional_novel_rankings.json",
            )
            _run(_layer2_command(
                pixi_manifest=pixi_manifest,
                rankings=novel_rankings,
                config=config_path,
                output=current_layer2_path,
                concurrency=args.layer2_concurrency,
                cache_dir=args.layer2_cache_dir,
                pixi=args.pixi,
            ))
            _run(_cdr_command(
                pixi_manifest=pixi_manifest,
                rankings=rankings_path,
                output=cdr_path,
                pixi=args.pixi,
            ))

            _assert_same_epoch(args, epoch, "retrying on timer")

            artifact, assignments, commit_context = _publish_fleet(
                args=args,
                crew=crew,
                rankings=rankings_path,
                layer2=[current_layer2_path, *historical_layer2],
                archive=archive_path,
                config=config_path,
                target=target,
                epoch=epoch,
                block=block,
                epoch_length=epoch_length,
                stage="provisional",
                bgen_python=bgen_python,
                nanobody_opt_root=args.nanobody_opt_root,
                staging_root=epoch_root / "checkpoints",
                novelty=novelty_path,
                cdr_annotations=cdr_path,
            )
            previous_assignments = _covered_current_assignments(
                previous,
                crew=crew,
                target=target,
                epoch=epoch,
            )
            if previous_prechecked and len(previous_assignments) > len(assignments):
                print(
                    "NANOBODY_FLEET_RETAINED broader same-epoch artifact: "
                    f"previous={len(previous_assignments)} "
                    f"challenger={len(assignments)}",
                    flush=True,
                )
                artifact = previous
                assignments = previous_assignments
                commit_context = CommitContext(
                    known_uids=frozenset(member.uid for member in crew),
                )
            _write_submission_artifact(args, artifact)
            _report(artifact, "provisional")
            ledger_fingerprint = _record_fleet(args.ledger, assignments)
            # Commit what we already have before the strict rank, which has
            # run over twenty minutes and can outlast the epoch. A committed
            # provisional fleet beats a qualified one that never lands; the
            # qualified attempt below re-commits when the chain's commitment
            # rate limit allows, and is a harmless no-op when it does not.
            _commit_fleet(
                assignments,
                args=args,
                ledger=commit_ledger,
                template=args.commit_command,
                artifact=args.artifact,
                epoch=epoch,
                pause_seconds=args.commit_pause_seconds,
                coldkey=coldkey,
                context=commit_context,
            )

        # Two separate reasons this used to be off, and they are not the same
        # reason. It cost a third of an epoch AND it ran BEFORE the commit, so
        # a slow strict rank meant the fleet reached the chain late or not at
        # all. The commit moved ahead of it, so the second reason is gone: by
        # the time control arrives here the provisional fleet is already on
        # chain and everything below is an upgrade attempt that can only
        # improve on it or be discarded. The first reason is handled by the
        # block reserve rather than by never running: an upgrade that cannot
        # finish inside this epoch is not started.
        skip_strict: str | None = None
        if not args.strict_qualification:
            skip_strict = "strict qualification is disabled"
        else:
            # The provisional fleet is already on chain by now, so everything
            # from here is optional.  A chain read that fails must therefore
            # skip the upgrade, never raise: turning an optional upgrade into
            # a failure path would cost the epoch a cycle it has already paid
            # for.
            try:
                current = _chain_view(
                    args.chain_state, args.netuid, args.maximum_chain_state_age
                )
            except (RuntimeError, SystemExit, OSError, ValueError) as error:
                current = None
                skip_strict = f"the chain view is unreadable ({error})"
            strict_window = (
                0 if current is None
                else current.epoch_length - (current.block % current.epoch_length)
            )
            if current is not None and (
                strict_window < args.strict_qualification_reserve_blocks
            ):
                skip_strict = (
                    "too little of the epoch is left for the strict upgrade "
                    f"({strict_window} blocks, need "
                    f"{args.strict_qualification_reserve_blocks})"
                )
        if skip_strict is not None:
            print(
                f"{skip_strict}; the committed provisional fleet stands "
                "for this epoch",
                flush=True,
            )
            _write_publisher_state(
                args.state,
                epoch=epoch,
                target=target,
                exact_fingerprint=exact_fingerprint,
                strict_fingerprint=strict_fingerprint,
                ledger_fingerprint=ledger_fingerprint,
                fleet_fingerprint=fleet_fingerprint,
            )
            return

        if _strict_state_matches(
            publisher_state,
            epoch=epoch,
            target=target,
            strict_fingerprint=strict_fingerprint,
            ledger_fingerprint=ledger_fingerprint,
            fleet_fingerprint=fleet_fingerprint,
        ):
            print(
                "strict-eligible exact evidence is unchanged; "
                "retaining previous qualification result",
                flush=True,
            )
            _write_publisher_state(
                args.state,
                epoch=epoch,
                target=target,
                exact_fingerprint=exact_fingerprint,
                strict_fingerprint=strict_fingerprint,
                ledger_fingerprint=ledger_fingerprint,
                fleet_fingerprint=fleet_fingerprint,
            )
            _commit_fleet(
                assignments,
                args=args,
                ledger=commit_ledger,
                template=args.commit_command,
                artifact=args.artifact,
                epoch=epoch,
                pause_seconds=args.commit_pause_seconds,
                coldkey=coldkey,
                context=commit_context,
            )
            return

        qualified_rankings = epoch_root / "qualified_rankings.json"
        qualified_layer2 = epoch_root / "qualified_layer2.json"
        qualified_novelty = epoch_root / "qualified_novelty.json"
        qualified_cdrs = epoch_root / "qualified_cdrs.json"
        strict_only = _strict_pool(
            args=args,
            bgen_python=bgen_python,
            epoch_root=epoch_root,
            archive_path=archive_path,
            target=target,
            exclusion_path=exclusion_path,
            epoch=epoch,
            shortlist=shortlist,
            config_path=config_path,
            exact_inputs=exact_inputs,
            observation_history=observation_history,
            crew=crew,
            previous=previous,
            candidate_scope=candidate_scope,
        )
        strict_only = strict_only or candidate_scope
        try:
            _run(_rank_command(
                bgen_python=bgen_python,
                nanobody_opt_root=args.nanobody_opt_root,
                exact_inputs=exact_inputs,
                archive=archive_path,
                target=target,
                output=qualified_rankings,
                exclusion=exclusion_path,
                epoch=epoch,
                shortlist=shortlist,
                config=config_path,
                strict=True,
                observation_history=observation_history,
                diversity_reserve_count=len(crew) * 2 + 1,
                only_sequences=strict_only,
            ))
            if strict_only is not None:
                qualified_count = len(json.loads(qualified_rankings.read_text())["candidates"])
                if qualified_count < len(crew):
                    # too few survived qualification to fill the fleet: a clean "no upgrade" outcome
                    raise subprocess.CalledProcessError(1, "strict pool too thin")
        except subprocess.CalledProcessError:
            print(
                "no robust repeat-observation upgrade is available yet; "
                "retaining the provisional fleet",
                flush=True,
            )
            _write_publisher_state(
                args.state,
                epoch=epoch,
                target=target,
                exact_fingerprint=exact_fingerprint,
                strict_fingerprint=strict_fingerprint,
                ledger_fingerprint=ledger_fingerprint,
                fleet_fingerprint=fleet_fingerprint,
            )
            _commit_fleet(
                assignments,
                args=args,
                ledger=commit_ledger,
                template=args.commit_command,
                artifact=args.artifact,
                epoch=epoch,
                pause_seconds=args.commit_pause_seconds,
                coldkey=coldkey,
                context=commit_context,
            )
            return
        _run(_novelty_command(
            pixi_manifest=pixi_manifest,
            rankings=qualified_rankings,
            archive=archive_path,
            target=target,
            output=qualified_novelty,
            pixi=args.pixi,
        ))
        qualified_novel_rankings = _novel_layer2_rankings(
            qualified_rankings, qualified_novelty,
            epoch_root / "qualified_novel_rankings.json",
        )
        _run(_layer2_command(
            pixi_manifest=pixi_manifest,
            rankings=qualified_novel_rankings,
            config=config_path,
            output=qualified_layer2,
            concurrency=args.layer2_concurrency,
            cache_dir=args.layer2_cache_dir,
            pixi=args.pixi,
        ))
        _run(_cdr_command(
            pixi_manifest=pixi_manifest,
            rankings=qualified_rankings,
            output=qualified_cdrs,
            pixi=args.pixi,
        ))
        _assert_same_epoch(args, epoch, "retaining the provisional fleet")

        artifact, assignments, commit_context = _publish_fleet(
            args=args,
            crew=crew,
            rankings=qualified_rankings,
            layer2=[qualified_layer2, *historical_layer2],
            archive=archive_path,
            config=config_path,
            target=target,
            epoch=epoch,
            block=block,
            epoch_length=epoch_length,
            stage="qualified",
            bgen_python=bgen_python,
            nanobody_opt_root=args.nanobody_opt_root,
            staging_root=epoch_root / "checkpoints",
            novelty=qualified_novelty,
            cdr_annotations=qualified_cdrs,
        )
        _write_submission_artifact(args, artifact)
        _report(artifact, "qualified")
        ledger_fingerprint = _record_fleet(args.ledger, assignments)
        _write_publisher_state(
            args.state,
            epoch=epoch,
            target=target,
            exact_fingerprint=exact_fingerprint,
            strict_fingerprint=strict_fingerprint,
            ledger_fingerprint=ledger_fingerprint,
            fleet_fingerprint=fleet_fingerprint,
        )
        _commit_fleet(
            assignments,
            args=args,
            ledger=commit_ledger,
            template=args.commit_command,
            artifact=args.artifact,
            epoch=epoch,
            pause_seconds=args.commit_pause_seconds,
                coldkey=coldkey,
                context=commit_context,
        )


def _assert_same_epoch(args: argparse.Namespace, epoch: int, consequence: str) -> None:
    """Refuse to publish into an epoch that rolled while we were scoring."""
    if getattr(args, "pi_handoff", False) and _COMMAND_DEADLINE is not None:
        _COMMAND_DEADLINE.remaining()
    current = _chain_view(
        args.chain_state, args.netuid, args.maximum_chain_state_age
    )
    if current.epoch != epoch:
        raise SystemExit(
            f"epoch rolled from {epoch} to {current.epoch} at block "
            f"{current.block}; {consequence}"
        )


def _write_submission_artifact(args, artifact):
    if getattr(args, "pi_handoff", False):
        from nanobody_opt import publication_ledger
        sequences = list(dict.fromkeys([
            *fleet_module.artifact_sequences(artifact), *artifact.get("sequence_pool", [])]))
        fleet_module.record_published(args.ledger, sequences)
        publication_ledger.record(args.ledger, epoch=artifact["epoch"],
                                  target=artifact["target"], sequences=sequences)
    _atomic_json(args.artifact, artifact)


def _record_fleet(
    ledger: Path, assignments: Sequence[fleet_module.Assignment]
) -> str | None:
    """Write all k published sequences to the ledger and refresh its digest."""
    appended = fleet_module.record_published(
        ledger, [item.sequence for item in assignments]
    )
    if appended:
        print(
            f"ledger: recorded {len(appended)} newly published sequence(s)",
            flush=True,
        )
    return _file_digest(ledger)


def _report(artifact: dict[str, Any], stage: str) -> None:
    for entry in artifact.get("submissions", []):
        print(
            f"NANOBODY_CHECKPOINT_OK stage={stage} epoch={artifact['epoch']} "
            f"uid={entry['uid']} block={entry['commit_block']} "
            f"sequence={entry['sequence']}",
            flush=True,
        )


def main() -> None:
    args = _parser().parse_args()
    if args.watch_interval <= 0:
        try:
            _publish_once(args)
        except (fleet_module.EmptyFleetError, fleet_module.MissingHotkeyError) as error:
            print(str(error), file=sys.stderr, flush=True)
            raise SystemExit(3) from error
        return

    interval = max(15.0, args.watch_interval)
    print(f"watching for epoch rollover every {interval:.0f}s", flush=True)
    while True:
        try:
            _publish_once(args)
        except (fleet_module.EmptyFleetError, fleet_module.MissingHotkeyError) as error:
            # Never a quiet skip: silence here is the 21-day bug.
            print(str(error), file=sys.stderr, flush=True)
        except SystemExit as error:
            print(f"publication deferred: {error}", flush=True)
        except Exception:
            traceback.print_exc()
        time.sleep(interval)


if __name__ == "__main__":
    main()
