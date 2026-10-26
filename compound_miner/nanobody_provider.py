"""Fail-closed transport and validation for nanobody mining checkpoints."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from config.config_loader import load_boltzgen_metrics
from nanobody_opt import publication_gates


NULL_SUBMISSION = "~"
SUPPORTED_SCHEMA_VERSIONS = frozenset({2, 3})
# The fleet artifact wraps one schema-2/3 checkpoint per uid we hold.
FLEET_SCHEMA_VERSION = 4
ALLOWED_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class NanobodyCheckpointError(ValueError):
    """Raised when a checkpoint cannot safely be included in a commitment."""

    def __init__(self, message="", *, reason_code="checkpoint_invalid"):
        super().__init__(message)
        self.reason_code = reason_code


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)


def _single_target(value: Any, field: str) -> str:
    if isinstance(value, str):
        targets = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        targets = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise NanobodyCheckpointError(f"{field} must contain exactly one target")
    if len(targets) != 1:
        raise NanobodyCheckpointError(f"{field} must contain exactly one target")
    return targets[0]


@lru_cache(maxsize=1)
def validator_boltzgen_metrics() -> frozenset[str]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "boltzgen_config.yaml"
    return frozenset(load_boltzgen_metrics(str(config_path)))


def _read_complete_snapshot(path: Path) -> dict[str, Any]:
    """Read one inode once, so a concurrent atomic rename is always complete."""
    try:
        with path.open("rb") as handle:
            raw = handle.read()
    except FileNotFoundError as exc:
        raise NanobodyCheckpointError(f"checkpoint is missing: {path}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NanobodyCheckpointError("checkpoint is malformed or partial JSON") from exc
    if not isinstance(payload, dict):
        raise NanobodyCheckpointError("checkpoint root must be a JSON object")
    return payload


def _validate_metrics(value: Any) -> None:
    if not isinstance(value, dict):
        raise NanobodyCheckpointError("BoltzGen metrics must be an object")
    missing = validator_boltzgen_metrics() - value.keys()
    if missing:
        raise NanobodyCheckpointError(
            "BoltzGen metrics are incomplete: missing " + ", ".join(sorted(missing))
        )
    for name in validator_boltzgen_metrics():
        metric = value[name]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise NanobodyCheckpointError(f"BoltzGen metric {name} is not numeric")
        if not math.isfinite(float(metric)):
            raise NanobodyCheckpointError(f"BoltzGen metric {name} is not finite")


def _finite_number(mapping: Mapping[str, Any], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NanobodyCheckpointError(f"checkpoint {name} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise NanobodyCheckpointError(f"checkpoint {name} is not finite")
    return number


def _integer(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NanobodyCheckpointError(f"checkpoint {name} is not an integer")
    return value


def _gate(flag: str) -> float:
    """Read one publication threshold from the module that owns all of them.

    These numbers were literals here until the day the gates were relaxed in
    nanobody_opt.publication_gates: ranking then qualified candidates this
    consumer went on rejecting, so the miner held a checkpoint it refused to
    commit. A consumer that is stricter than the gate it claims to enforce is a
    silent publication ban, not a safety margin.
    """
    return float(publication_gates.gate_defaults()[flag])


def _checkpoint_evidence_floor(payload: Mapping[str, Any], default: int) -> int:
    if "confirmation_depth" not in payload:
        return default
    from nanobody_opt.confirmation_depth import validate
    decision = validate(payload["confirmation_depth"], epoch=payload.get("epoch"),
                        target=payload.get("target"))
    if decision is None:
        raise NanobodyCheckpointError("invalid epoch confirmation-depth decision")
    return decision["minimum_observations"]


def _validate_robust_exact_evidence(payload: Mapping[str, Any]) -> None:
    exact = payload.get("exact")
    selection = payload.get("selection")
    if not isinstance(exact, Mapping) or not isinstance(selection, Mapping):
        raise NanobodyCheckpointError("robust exact-selection evidence is missing")
    minimum_observations = _checkpoint_evidence_floor(payload, int(_gate("--minimum-observations")))
    if _integer(exact, "observation_count") < minimum_observations:
        raise NanobodyCheckpointError(
            f"checkpoint has fewer than {minimum_observations} exact observations"
        )
    pose = exact.get("pose_stability")
    if not isinstance(pose, Mapping):
        raise NanobodyCheckpointError("checkpoint pose-stability evidence is missing")
    minimum_structures = int(_gate("--minimum-structure-observations"))
    if _integer(pose, "structure_observation_count") < minimum_structures:
        raise NanobodyCheckpointError(
            f"checkpoint has fewer than {minimum_structures} retained complexes"
        )
    if _finite_number(pose, "contact_jaccard_median") < _gate(
        "--minimum-median-contact-jaccard"
    ):
        raise NanobodyCheckpointError("checkpoint contact stability is below threshold")
    if _finite_number(pose, "cdr3_rmsd_max_angstrom") > _gate("--maximum-cdr3-rmsd"):
        raise NanobodyCheckpointError("checkpoint CDR3 stability is below threshold")
    # Current validator config's generation hotspot is not the observed contact
    # patch of the strongest public P20809 poses. Preserve this field for audit,
    # but rely on repeat contact Jaccard and target-aligned CDR3 RMSD as gates.
    if _finite_number(pose, "epitope_occupancy_min") < _gate(
        "--minimum-epitope-occupancy"
    ):
        raise NanobodyCheckpointError("checkpoint epitope occupancy is below threshold")
    if _finite_number(selection, "competitive_rate") < _gate(
        "--minimum-competitive-rate"
    ):
        raise NanobodyCheckpointError("checkpoint field competitive rate is below threshold")
    if _finite_number(selection, "observation_competitive_rate") < _gate(
        "--minimum-observation-competitive-rate"
    ):
        raise NanobodyCheckpointError(
            "checkpoint observation competitive rate is below threshold"
        )
    if _finite_number(selection, "score_p90") > _gate("--maximum-p90-rank-sum"):
        raise NanobodyCheckpointError("checkpoint p90 rank sum exceeds threshold")
    categories = selection.get("p90_category_rank_sums")
    if not isinstance(categories, Mapping):
        raise NanobodyCheckpointError("checkpoint category-tail evidence is missing")
    for category, flag in (
        ("confidence", "--maximum-p90-confidence-rank-sum"),
        ("physical_interaction", "--maximum-p90-physical-interaction-rank-sum"),
        ("developability", "--maximum-p90-developability-rank-sum"),
    ):
        if _finite_number(categories, category) > _gate(flag):
            raise NanobodyCheckpointError(
                f"checkpoint {category} p90 rank sum exceeds threshold"
            )


def _validate_provisional_exact_evidence(payload: Mapping[str, Any]) -> None:
    """Validate an optimistic checkpoint produced by the first screen batch."""
    exact = payload.get("exact")
    selection = payload.get("selection")
    if not isinstance(exact, Mapping) or not isinstance(selection, Mapping):
        raise NanobodyCheckpointError("provisional exact-selection evidence is missing")
    if _integer(exact, "observation_count") < _checkpoint_evidence_floor(payload, 1):
        raise NanobodyCheckpointError("checkpoint has no exact observations")
    _finite_number(selection, "score")
    _finite_number(selection, "score_p90")


def _validate_live_trial_evidence(payload: Mapping[str, Any], boltzgen: Mapping[str, Any]) -> None:
    """Accept measured fast-baseline evidence without claiming equivalence.

    The caller has already checked the actual current PR100 eligibility rules.
    This stage relaxes only our local quality-certification requirement.
    """
    from nanobody_opt import rules
    if payload['epoch'] < rules.ACTIVATION_EPOCH:
        raise NanobodyCheckpointError('live trials require current PR100 eligibility')
    layer_one = payload['validation']['layer_1']
    for flag, code in (("historically_unique", "layer_1_historically_unique_failed"),
                       ("duplicate_gate_passed", "layer_1_duplicate_gate_failed")):
        if layer_one.get(flag) is not True:
            raise NanobodyCheckpointError("live trial novelty gate did not pass", reason_code=code)
    reason = rules.novelty_reason(layer_one.get("novelty"), payload['sequences'][0], payload['target'])
    if reason:
        raise NanobodyCheckpointError("live trial novelty evidence is invalid", reason_code=reason)
    if (boltzgen.get('mode') != 'live_fast_baseline'
            or any(boltzgen.get(k) is not False for k in
                   ('exact', 'runtime_matched', 'validator_matched', 'scoring_equivalence_qualified'))):
        raise NanobodyCheckpointError('live trial must explicitly disclaim exact-scoring equivalence')
    provenance = boltzgen.get('provenance')
    if not isinstance(provenance, Mapping):
        raise NanobodyCheckpointError('live baseline provenance is missing')
    if provenance.get('epoch') != payload['epoch'] or provenance.get('target') != payload['target']:
        raise NanobodyCheckpointError('live baseline provenance is for another context')
    for key in ('scorer_sha256', 'settings_sha256'):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
            raise NanobodyCheckpointError('live baseline provenance digest is invalid')
    settings = provenance.get('settings')
    if not isinstance(settings, dict) or not settings or rules.digest(settings) != provenance['settings_sha256']:
        raise NanobodyCheckpointError('live baseline settings do not match their digest')
    _validate_metrics(boltzgen.get('metrics'))
    _validate_provisional_exact_evidence(payload)


def validate_nanobody_checkpoint(
    payload: Mapping[str, Any],
    *,
    expected_target: str,
    expected_epoch: int,
    min_sequence_length: int,
    max_sequence_length: int,
    epoch_start_block_hash: str | None = None,
    epoch_length: int | None = None,
) -> str:
    """Return the sole qualified sequence or reject the entire checkpoint."""
    version = payload.get("schema_version")
    if isinstance(version, bool) or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise NanobodyCheckpointError(f"unsupported schema_version: {version!r}")
    if version == 2:
        stage = "qualified"
        if payload.get("status") != "qualified":
            raise NanobodyCheckpointError("checkpoint status is not qualified")
    else:
        stage = payload.get("publication_stage")
        if stage not in {"provisional", "qualified", "live_trial"}:
            raise NanobodyCheckpointError("checkpoint publication stage is invalid")
        if payload.get("status") != stage:
            raise NanobodyCheckpointError("checkpoint status does not match publication stage")
    if _single_target(payload.get("target"), "checkpoint target") != expected_target:
        raise NanobodyCheckpointError("checkpoint target does not match current challenge")

    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != expected_epoch:
        raise NanobodyCheckpointError("checkpoint epoch is stale or does not match current challenge")

    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or len(sequences) != 1:
        raise NanobodyCheckpointError("checkpoint must contain exactly one sequence")
    sequence = sequences[0]
    if not isinstance(sequence, str) or not sequence:
        raise NanobodyCheckpointError("nanobody sequence must be a non-empty string")
    if sequence != sequence.upper() or set(sequence) - ALLOWED_AMINO_ACIDS:
        raise NanobodyCheckpointError("nanobody sequence must contain uppercase canonical amino acids")
    if not min_sequence_length <= len(sequence) <= max_sequence_length:
        raise NanobodyCheckpointError(
            f"nanobody sequence length {len(sequence)} is outside configured range "
            f"{min_sequence_length}..{max_sequence_length}"
        )

    from nanobody_opt import rules
    if expected_epoch >= rules.ACTIVATION_EPOCH:
        try:
            if type(epoch_length) is not int or epoch_length < 1:
                raise ValueError("independently supplied epoch length is required")
            config = rules.pinned_config(expected_epoch)
            expected = rules.challenge(config, expected_epoch, epoch_start_block_hash, epoch_length)
            if expected_target != expected["target"]:
                raise ValueError("PR100 target mismatch")
            rules.validate_evidence(payload.get("pr100_evidence"), sequence, expected, config)
        except (ValueError, TypeError, KeyError) as error:
            message = str(error)
            code = {"missing or mismatched current-challenge evidence": "pr100_challenge_mismatch",
                    "filter version mismatch": "pr100_filter_mismatch"}.get(message, message)
            if code not in rules.HANDOFF_REJECTION_CODES:
                code = "pr100_evidence_invalid"
            raise NanobodyCheckpointError("PR100 validation: " + message, reason_code=code) from error

    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise NanobodyCheckpointError("validation declaration is missing")
    for layer in ("layer_1", "layer_2"):
        result = validation.get(layer)
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise NanobodyCheckpointError(f"{layer} validation did not pass")
    boltzgen = validation.get("boltzgen")
    if not isinstance(boltzgen, dict):
        raise NanobodyCheckpointError("exact BoltzGen validation is missing")
    if stage == 'live_trial':
        _validate_live_trial_evidence(payload, boltzgen)
        return sequence
    runtime_matched = boltzgen.get("runtime_matched") is True
    legacy_matched = boltzgen.get("validator_matched") is True
    if boltzgen.get("exact") is not True or not (runtime_matched or legacy_matched):
        raise NanobodyCheckpointError("BoltzGen validation is not runtime-matched exact scoring")
    _validate_metrics(boltzgen.get("metrics"))
    if stage == "qualified":
        _validate_robust_exact_evidence(payload)
    else:
        _validate_provisional_exact_evidence(payload)
    return sequence


def fleet_submission(payload: Mapping[str, Any], hotkey: str) -> Mapping[str, Any]:
    """Return the one fleet entry this hotkey must commit.

    The publisher writes a single artifact holding one distinct sequence per uid,
    because k separate files can be read as a torn fleet.  A miner process
    therefore selects its own entry BY ITS OWN HOTKEY: not by uid, which is
    recycled on deregistration, and not by position, which reorders whenever the
    registered set changes.
    """
    address = str(hotkey or "").strip()
    if not address:
        raise NanobodyCheckpointError(
            "a fleet artifact needs this miner's hotkey to select its submission"
        )
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or not submissions:
        raise NanobodyCheckpointError("fleet artifact holds no submissions")
    matches = [
        entry for entry in submissions
        if isinstance(entry, Mapping) and str(entry.get("hotkey") or "") == address
    ]
    if len(matches) != 1:
        raise NanobodyCheckpointError(
            f"fleet artifact holds {len(matches)} submissions for this hotkey, not 1"
        )
    entry = matches[0]
    checkpoint = entry.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise NanobodyCheckpointError("fleet submission carries no checkpoint")
    sequence = str(entry.get("sequence") or "").strip().upper()
    sequences = checkpoint.get("sequences")
    embedded = (
        str(sequences[0]).strip().upper()
        if isinstance(sequences, list) and sequences and isinstance(sequences[0], str)
        else ""
    )
    if not sequence or sequence != embedded:
        raise NanobodyCheckpointError(
            "fleet submission sequence disagrees with its own checkpoint"
        )
    return checkpoint


def is_fleet_artifact(payload: Any) -> bool:
    """Return whether this payload is the multi-uid fleet artifact."""
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == FLEET_SCHEMA_VERSION
        and payload.get("artifact_kind") == "nanobody-fleet"
    )


def load_qualified_nanobody(
    challenge_params: Mapping[str, Any],
    config: Any,
    *,
    path: str | os.PathLike[str] | None = None,
    hotkey: str | None = None,
) -> str:
    """Load and validate the current local checkpoint; raise on every defect."""
    configured_path = path if path is not None else os.environ.get("NOVA_NANOBODY_RESULT")
    if not configured_path:
        raise NanobodyCheckpointError("NOVA_NANOBODY_RESULT is not configured")
    expected_target = _single_target(challenge_params.get("nanobody_target"), "challenge target")
    epoch = challenge_params.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise NanobodyCheckpointError("current challenge epoch is missing")
    payload = _read_complete_snapshot(Path(configured_path))
    if is_fleet_artifact(payload):
        payload = fleet_submission(payload, hotkey or "")
    return validate_nanobody_checkpoint(
        payload,
        expected_target=expected_target,
        expected_epoch=epoch,
        min_sequence_length=int(_config_value(config, "min_sequence_length")),
        max_sequence_length=int(_config_value(config, "max_sequence_length")),
        epoch_start_block_hash=challenge_params.get("epoch_start_block_hash"),
        epoch_length=challenge_params.get("epoch_length"),
    )


def fetch_remote_checkpoint(destination: str | os.PathLike[str]) -> None:
    """SCP to a sibling temp file and atomically replace the local checkpoint."""
    source = os.environ.get("NOVA_NANOBODY_SSH_SOURCE")
    if not source:
        return
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "scp", "-q", "-O",
        "-P", os.environ.get("NOVA_NANOBODY_SSH_PORT", "22"),
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=yes",
    ]
    identity = os.environ.get("NOVA_NANOBODY_SSH_IDENTITY")
    known_hosts = os.environ.get("NOVA_NANOBODY_SSH_KNOWN_HOSTS")
    if identity:
        command.extend(["-i", identity])
    if known_hosts:
        command.extend(["-o", f"UserKnownHostsFile={known_hosts}"])

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        completed = subprocess.run(
            [*command, source, temporary_name],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise NanobodyCheckpointError(
                f"SCP checkpoint fetch failed: {detail or completed.returncode}"
            )
        os.replace(temporary_name, destination_path)
        temporary_name = None
    except (OSError, subprocess.SubprocessError) as exc:
        raise NanobodyCheckpointError(f"SCP checkpoint fetch failed: {exc}") from exc
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def fetch_and_load_qualified_nanobody(
    challenge_params: Mapping[str, Any], config: Any
) -> str:
    """Pull the producer result when configured, then validate the local snapshot."""
    destination = os.environ.get("NOVA_NANOBODY_RESULT")
    if not destination:
        raise NanobodyCheckpointError("NOVA_NANOBODY_RESULT is not configured")
    fetch_remote_checkpoint(destination)
    return load_qualified_nanobody(challenge_params, config, path=destination)
