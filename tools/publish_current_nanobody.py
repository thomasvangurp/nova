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

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Sequence

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

# Only rank one pays and the validator draws once, so the shortlist that decides
# what is published must be ordered by the chance of a winning draw.
SELECTION_ORDER = fleet_module.REQUIRED_SELECTION_ORDER
SHORTLIST_LIMIT = 30

CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SUPPORTED_CHAIN_STATE_SCHEMAS = frozenset({1, 2})
PUBLISHER_STATE_SCHEMA_VERSION = 2


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
    deduplication and complete finite-metric requirement.
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
    strict_eligible = {
        sequence: records
        for sequence, records in sorted(observations.items())
        if len(records) >= minimum
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


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


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
) -> list[str]:
    command = [str(bgen_python), str(nanobody_opt_root / "rank_exact.py")]
    for path in exact_inputs:
        command.extend(["--exact", str(path)])
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
        "--api-field-count", "10",
        "--minimum-api-fields", "3",
    ])
    if strict:
        command.extend(["--pose-config", str(config)])
        command.extend(RANK_QUALIFICATION_ARGUMENTS)
    else:
        # The provisional pass exists so that an epoch always submits something;
        # it applies the evidence floor only, never the quality gates. The
        # winning threshold is not a quality gate: it is what "win probability"
        # is measured against, so both passes must order against the same one.
        command.extend([
            "--minimum-observations",
            publication_gates.gate_defaults()["--minimum-observations"],
            "--competitive-score-threshold",
            publication_gates.gate_defaults()["--competitive-score-threshold"],
        ])
    command.extend(["--selection-order", SELECTION_ORDER])
    return command


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
) -> tuple[dict[str, Any], list[fleet_module.Assignment]]:
    """Bind k distinct sequences to k uids and build the single fleet artifact."""
    payload = json.loads(rankings.read_text())
    sequences = fleet_module.distinct_sequences(payload, len(crew))
    try:
        blocks = fleet_module.commitment_blocks(
            epoch=epoch,
            epoch_length=epoch_length,
            reserve_blocks=args.reserve_blocks,
            count=len(sequences),
            block=block if block >= 0 else None,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    assignments = fleet_module.assign_sequences(crew, sequences, blocks)
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
    )
    return artifact, published


PUBLISH_FLEET_REJECTED = "NANOBODY_FLEET_UID_REJECTED"
COMMIT_FLEET_FAILED = "COMMIT_FLEET_UID_FAILED"
COMMIT_FLEET_INCOMPLETE = "COMMIT_FLEET_INCOMPLETE"


def _commit_fleet(
    assignments: Sequence[fleet_module.Assignment],
    *,
    template: str | None,
    artifact: Path,
    epoch: int,
    pause_seconds: float,
) -> None:
    """Send one commitment per uid, one block apart, inside the reserve.

    ``max_regs_per_block`` is 1, so extrinsics are paced a block apart rather
    than sent together.  With no command configured the per-uid miner processes
    commit from the artifact themselves and this is a no-op; the schedule is
    recorded either way.
    """
    if not template:
        return
    failures: list[tuple[int, int]] = []
    for index, assignment in enumerate(assignments):
        command = [
            part.format(
                uid=assignment.uid,
                hotkey=assignment.hotkey,
                sequence=assignment.sequence,
                block=assignment.commit_block,
                epoch=epoch,
                artifact=str(artifact),
            )
            for part in shlex.split(template)
        ]
        if index and pause_seconds > 0:
            time.sleep(pause_seconds)
        try:
            _run(command)
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
        default=55,
        help="end-of-epoch blocks reserved for commitments, one uid per block",
    )
    parser.add_argument(
        "--commit-command",
        help=(
            "optional per-uid commitment command; {uid} {hotkey} {sequence} "
            "{block} {epoch} {artifact} are substituted"
        ),
    )
    parser.add_argument(
        "--strict-qualification",
        action="store_true",
        help=(
            "run the strict rank_exact upgrade after the provisional fleet "
            "has committed. Off by default: it has produced no upgrade in "
            "live operation and costs about a third of an epoch"
        ),
    )
    parser.add_argument(
        "--commit-pause-seconds",
        type=float,
        default=12.0,
        help="one block between extrinsics; max_regs_per_block is 1",
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
    parser.add_argument("--historical-layer2", action="append", type=Path)
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=0,
        help="repeat forever at this interval, logging failures and retrying",
    )
    return parser


def _publish_once(args: argparse.Namespace) -> None:
    with _exclusive_lock(args.lock):
        if args.expected_gpus and _gpu_count() != args.expected_gpus:
            raise SystemExit(
                f"expected {args.expected_gpus} GPUs on the nanobody host; fail closed"
            )
        chain = _chain_view(
            args.chain_state, args.netuid, args.maximum_chain_state_age
        )
        epoch, block, epoch_length = chain.epoch, chain.block, chain.epoch_length
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
        _atomic_download(CONFIG_URL, config_path)
        config = yaml.safe_load(config_path.read_text())
        configured_target = str(config["protein_selection"]["nanobody"]["target"])
        if args.target is not None and configured_target != args.target:
            raise SystemExit(
                f"live target is {configured_target!r}, exact bank is {args.target!r}; fail closed"
            )
        target = configured_target
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
        exact_inputs = tuple(args.exact or _default_exact_inputs(args.score_bank_root))
        exact_fingerprint, strict_fingerprint = _exact_bank_fingerprints(
            exact_inputs, target
        )
        bgen_python = args.runtime_root / ".pixi/envs/bgen/bin/python"
        pixi_manifest = args.runtime_root / "pixi.toml"
        if not bgen_python.is_file() or not pixi_manifest.is_file():
            raise SystemExit("locked nanobody Pixi runtimes are not installed")
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
        if current_qualified and _publisher_state_matches(
            publisher_state,
            epoch=epoch,
            target=target,
            exact_fingerprint=exact_fingerprint,
            ledger_fingerprint=ledger_fingerprint,
            fleet_fingerprint=fleet_fingerprint,
        ):
            print(
                f"qualified epoch {epoch} fleet, score bank and uid set are "
                "unchanged; no-op"
            )
            return
        exclusion_path = fleet_module.write_exclusion_file(
            epoch_root / "excluded_sequences.txt",
            args.ledger,
            _live_sequences(previous, epoch),
        )
        _atomic_download(ARCHIVE_URL.format(target=target), archive_path)

        historical_layer2 = tuple(
            args.historical_layer2 or map(Path, DEFAULT_LAYER2_INPUTS)
        )
        # An input the operator NAMED must exist -- a typo there should stop the
        # run. A default must not: DEFAULT_EXACT_INPUTS and DEFAULT_LAYER2_INPUTS
        # are paths from epoch 25021 on a machine that no longer exists, and
        # requiring them meant a rebuilt box deferred publication for ever while
        # sitting on a perfectly good score bank the glob had already found.
        explicit = [
            *(args.exact or ()),
            *(args.historical_layer2 or ()),
        ]
        missing = [str(path) for path in explicit if not path.is_file()]
        if missing:
            raise SystemExit(
                "score-bank inputs named on the command line are missing: "
                + ", ".join(missing)
            )
        exact_inputs = tuple(path for path in exact_inputs if path.is_file())
        historical_layer2 = tuple(path for path in historical_layer2 if path.is_file())
        # Publishing with no evidence at all is the thing to refuse.
        if not exact_inputs:
            raise SystemExit(
                "no exact score-bank inputs found under "
                f"{args.score_bank_root}: the producer has not written an "
                "exact_all.json yet, so there is nothing qualified to publish"
            )

        shortlist = max(SHORTLIST_LIMIT, len(crew))
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
            ))
            _run(_layer2_command(
                pixi_manifest=pixi_manifest,
                rankings=rankings_path,
                config=config_path,
                output=current_layer2_path,
                concurrency=args.layer2_concurrency,
                cache_dir=args.layer2_cache_dir,
                pixi=args.pixi,
            ))

            _assert_same_epoch(args, epoch, "retrying on timer")

            artifact, assignments = _publish_fleet(
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
            )
            _atomic_json(args.artifact, artifact)
            _report(artifact, "provisional")
            ledger_fingerprint = _record_fleet(args.ledger, assignments)
            # Commit what we already have before the strict rank, which has
            # run over twenty minutes and can outlast the epoch. A committed
            # provisional fleet beats a qualified one that never lands; the
            # qualified attempt below re-commits when the chain's commitment
            # rate limit allows, and is a harmless no-op when it does not.
            _commit_fleet(
                assignments,
                template=args.commit_command,
                artifact=args.artifact,
                epoch=epoch,
                pause_seconds=args.commit_pause_seconds,
            )

        if not args.strict_qualification:
            # Cut from the live path. Six consecutive epochs produced zero
            # qualified upgrades while the strict rank consumed roughly twenty
            # minutes of a seventy-two minute epoch, and until the provisional
            # commit moved ahead of it, it was the reason a published fleet
            # reached the chain late or not at all. Run it offline against the
            # score bank, or pass --strict-qualification to restore it.
            print(
                "strict qualification is disabled; the committed provisional "
                "fleet stands for this epoch",
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
                template=args.commit_command,
                artifact=args.artifact,
                epoch=epoch,
                pause_seconds=args.commit_pause_seconds,
            )
            return

        qualified_rankings = epoch_root / "qualified_rankings.json"
        qualified_layer2 = epoch_root / "qualified_layer2.json"
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
            ))
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
                template=args.commit_command,
                artifact=args.artifact,
                epoch=epoch,
                pause_seconds=args.commit_pause_seconds,
            )
            return
        _run(_layer2_command(
            pixi_manifest=pixi_manifest,
            rankings=qualified_rankings,
            config=config_path,
            output=qualified_layer2,
            concurrency=args.layer2_concurrency,
            cache_dir=args.layer2_cache_dir,
            pixi=args.pixi,
        ))
        _assert_same_epoch(args, epoch, "retaining the provisional fleet")

        artifact, assignments = _publish_fleet(
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
        )
        _atomic_json(args.artifact, artifact)
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
            template=args.commit_command,
            artifact=args.artifact,
            epoch=epoch,
            pause_seconds=args.commit_pause_seconds,
        )


def _assert_same_epoch(args: argparse.Namespace, epoch: int, consequence: str) -> None:
    """Refuse to publish into an epoch that rolled while we were scoring."""
    current = _chain_view(
        args.chain_state, args.netuid, args.maximum_chain_state_age
    )
    if current.epoch != epoch:
        raise SystemExit(
            f"epoch rolled from {epoch} to {current.epoch} at block "
            f"{current.block}; {consequence}"
        )


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
