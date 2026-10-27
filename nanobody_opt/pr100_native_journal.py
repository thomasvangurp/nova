"""Allowlisted, durable observations; never a cache or publication authority.

Each invocation owns a new attempt UUID. Old records are counted for audit only:
they are neither loaded as measurements nor reused as independent repeats. The
runtime identity deliberately says that model-weight identity is incomplete.
"""
from __future__ import annotations

from collections import Counter
import copy
from decimal import Decimal, InvalidOperation
import errno
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import uuid

from nanobody_opt import rules


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pr100-native-observation-v1"
METRICS = ("total_cdr_length", "cdr3_length", "cdr3_compactness",
           "surface_hydrophobic_patches", "positive_charge_patches",
           "negative_charge_patches")
A3D_METRICS = ("a3d_total", "a3d_average", "a3d_maximum", "a3d_minimum",
               "a3d_positive_auc", "radius_angstrom")
A3D_ENUMS = {"ph_scale": {"default-scale"},
            "surface_method": {"FreeSASA-2.0.1-NACCESS-LR100"},
            "mode": {"static-no-minimization"},
            "status": {"ok", "error", "disabled"}}
A3D_HASHES = ("sequence_hash", "input_structure_sha256", "normalized_structure_sha256")
ERROR_KINDS = frozenset({"none", "timeout", "no_profile", "profile_exception",
                         "cancelled", "unknown"})
REJECTION_REASONS = frozenset({"developability_red_flags", "psh_threshold",
                               "aggrescan_threshold", "developability_failed"})
GATE_REASONS = REJECTION_REASONS | frozenset({"", "developability_unavailable",
    "psh_unavailable", "aggrescan_unavailable_or_failed", "aggrescan_sequence_mismatch",
    "aggrescan_provenance_mismatch", "observation_unavailable"})
OUTCOMES = ("passed", "rejected", "inconclusive")
SOURCES = frozenset({"existing_native_reference", "fresh_baseline_proposal"})
PACKAGES = ("abnumber", "anarci", "anarcii", "biopython", "numpy", "torch",
            "ImmuneBuilder", "openmm", "TNP")
SOURCE_FILES = {
    "adapter": "nanobody_opt/pr100_filter.py",
    "worker": "nanobody_opt/pr100_live_worker.py",
    "journal": "nanobody_opt/pr100_native_journal.py",
    "rules": "nanobody_opt/rules.py",
    "annotation": "utils/germline_identity.py",
    "aggregation_adapter": "utils/aggrescan3d.py",
    "backend_config": "NOVA-nanobody-filter/metanano/config.py",
    "backend_service": "NOVA-nanobody-filter/metanano/services/developability_service.py",
    "backend_filter": "NOVA-nanobody-filter/metanano/filters/developability.py",
    "backend_tnp": "NOVA-nanobody-filter/metanano/utils/tnp_wrapper.py",
    "backend_aggregation": "NOVA-nanobody-filter/metanano/utils/aggrescan3d_wrapper.py",
}


def _hash(value, length=64):
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value) else None


def _number(value):
    try:
        return value if type(value) in (int, float) and math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def _decimal(value):
    # No arbitrary __str__, exception text, NaN, Infinity or excessively large
    # decimal expansion can pass through to a diagnostic record.
    if type(value) not in (str, int, float, Decimal):
        return None
    try:
        text = str(value)
    except (OverflowError, ValueError):
        return None
    if len(text) > 128 or not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", text):
        return None
    try:
        number = Decimal(text)
        return str(number) if number.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _enum(value, choices, default="unknown"):
    return value if isinstance(value, str) and value in choices else default


def _profile_fields(value):
    """Project only typed metrics, booleans, hashes and closed-vocabulary codes."""
    value = value if isinstance(value, dict) else {}
    metrics = {key: number for key in METRICS if (number := _number(value.get(key))) is not None}
    raw = value.get("aggrescan3d")
    raw = raw if isinstance(raw, dict) else {}
    a3d = {key: decimal for key in A3D_METRICS if (decimal := _decimal(raw.get(key))) is not None}
    a3d.update({key: _enum(raw.get(key), choices) for key, choices in A3D_ENUMS.items()})
    a3d.update({key: digest for key in A3D_HASHES if (digest := _hash(raw.get(key))) is not None})
    if type(raw.get("residue_count")) is int and raw["residue_count"] >= 0:
        a3d["residue_count"] = raw["residue_count"]
    a3d["passed"] = raw.get("passed") if type(raw.get("passed")) is bool else None
    missing = [key for key in METRICS if key not in metrics]
    missing += ["aggrescan3d." + key for key in ("a3d_average", "a3d_maximum") if key not in a3d]
    return metrics, missing, a3d


def observation(value, requirements, sequence):
    """Describe one native completion without changing its scientific verdict.

    Errors, unavailable profiles and unverifiable provenance are inconclusive.
    No message, stderr, environment, free-text reason or raw flags are retained.
    """
    raw = value if isinstance(value, dict) else {}
    metrics, missing, a3d = _profile_fields(raw)
    error = not isinstance(value, dict) or bool(raw.get("error"))
    error_kind = ("none" if raw.get("error_kind") is None and not error
                  else _enum(raw.get("error_kind"), ERROR_KINDS))
    error = error or error_kind != "none"
    try:
        reason = _enum(rules.developability_reason(value, requirements, sequence=sequence),
                       GATE_REASONS, "observation_unavailable")
        if (reason == "aggrescan_unavailable_or_failed" and not error and not missing
                and a3d.get("status") == "ok" and a3d.get("passed") is False):
            # The publication gate checks the backend's passed flag before its
            # numeric caps. A completed cap rejection therefore shares a reason
            # with an unavailable tool. For diagnostics only, ask the same gate
            # to validate provenance and caps without that early false flag.
            # Never mutate the profile or turn this into an acceptance decision.
            diagnostic = {**raw, "aggrescan3d": {**raw["aggrescan3d"], "passed": True}}
            if (Decimal(a3d["a3d_maximum"]) >= 0
                    and rules.developability_reason(
                        diagnostic, requirements, sequence=sequence) == "aggrescan_threshold"):
                reason = "aggrescan_threshold"
    except Exception:
        reason = "observation_unavailable"
    outcome = ("inconclusive" if error or reason not in REJECTION_REASONS | {""}
               else "passed" if not reason else "rejected")
    return {"metrics": metrics, "missing_metrics": missing, "aggrescan3d": a3d,
            "error": error, "error_kind": error_kind,
            "backend_passed": raw.get("passed") if type(raw.get("passed")) is bool else None,
            "has_red_flags": bool(raw.get("red_flags")),
            "gate_reason": reason, "outcome": outcome,
            "execution_censored": outcome == "inconclusive"}


def _safe_observation(value):
    """Reproject even caller-supplied observations before writing any bytes."""
    value = value if isinstance(value, dict) else {}
    metrics = value.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    metrics, missing, a3d = _profile_fields({**metrics, "aggrescan3d": value.get("aggrescan3d")})
    reason = _enum(value.get("gate_reason"), GATE_REASONS, "observation_unavailable")
    error = value.get("error") is not False
    outcome = _enum(value.get("outcome"), OUTCOMES, "inconclusive")
    if error or reason not in REJECTION_REASONS | {""}:
        outcome = "inconclusive"
    elif outcome == "passed" and (reason or missing):
        outcome = "inconclusive"
    return {"metrics": metrics, "missing_metrics": missing, "aggrescan3d": a3d,
            "error": error, "error_kind": _enum(value.get("error_kind"), ERROR_KINDS),
            "backend_passed": value.get("backend_passed") if type(value.get("backend_passed")) is bool else None,
            "has_red_flags": value.get("has_red_flags") is True,
            "gate_reason": reason, "outcome": outcome,
            "execution_censored": outcome == "inconclusive"}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context(directory, job, requirements):
    """Record observed identities without importing models or executing tools."""
    failures = Counter()

    def attempt(key, operation):
        try:
            return operation()
        except Exception:
            failures[key] += 1
            return None

    stamp_path = ROOT / "build_stamp.json"
    stamp = attempt("build_stamp", lambda: json.loads(stamp_path.read_text()))
    stamp = stamp if isinstance(stamp, dict) else {}
    source = {name: attempt("source_file", lambda path=path: _file_hash(ROOT / path))
              for name, path in SOURCE_FILES.items()}
    packages = {}
    for package in PACKAGES:
        version = attempt("package_version", lambda package=package: importlib.metadata.version(package))
        # Local-version strings may contain arbitrary text: keep their digest,
        # never their literal contents. Do not import any numerical package.
        packages[package] = {"version": version if isinstance(version, str) and re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?", version) else None,
            "version_sha256": hashlib.sha256(version.encode()).hexdigest() if isinstance(version, str) else None}
    tools = {}
    native_tools = job.get("native_tools")
    for name in ("aggrescan3d-rs", "freesasa", "hmmscan", "mkdssp", "TNP"):
        path = (Path(native_tools) / name if name in ("aggrescan3d-rs", "freesasa")
                and isinstance(native_tools, str) else shutil.which(name))
        tools[name] = {"sha256": attempt("tool_identity", lambda path=path: _file_hash(path)) if path else None}
    context = {
        "schema": SCHEMA, "observational": True, "reusable": False,
        "model_weight_identity": "incomplete", "runtime_identity_status": "partial",
        "source_revision": _hash(stamp.get("git_sha"), 40),
        "declared_source_revision": _hash(job.get("source_sha"), 40),
        "build_stamp_sha256": attempt("build_stamp", lambda: _file_hash(stamp_path)),
        "source_sha256": source,
        "input_sha256": attempt("input_identity", lambda: _file_hash(directory / "input.json")),
        "challenge_sha256": attempt("challenge_identity", lambda: rules.digest(job.get("challenge"))),
        "requirements_sha256": attempt("requirements_identity", lambda: rules.digest(requirements)),
        "filter_revision": _hash(rules.FILTER_SHA, 40),
        "upstream_revision": _hash(rules.UPSTREAM_SHA, 40),
        "runtime": {"python_version": list(sys.version_info[:3]),
                    "python_binary_sha256": attempt("python_identity", lambda: _file_hash(sys.executable)),
                    "packages": packages, "tools": tools},
    }
    context["identity_errors"] = dict(failures)
    return context


class Journal:
    """One writer's append-only attempt, with fail-open diagnostic persistence.

    ``repeat`` is one-based. ``save`` returns True only after fsync. Progress
    counts are for this attempt alone; previous records are audit counts only.
    This class never exposes or republishes a previously recorded profile.
    """

    def __init__(self, directory, job, requirements):
        self.directory = Path(directory)
        self.path = self.directory / "native-observations.jsonl"
        self.attempt_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._errors = Counter()
        self._errnos = Counter()
        self._saved = set()
        self._seen = 0
        self._outcomes = Counter()
        self._first_outcomes = Counter()
        self._previous_records = 0
        self._previous_attempts = set()
        self._invalid_history = 0
        self._torn_tail_bytes = 0
        self._needs_separator = False
        self._header_saved = False
        try:
            self.context = _context(self.directory, job, requirements)
        except Exception as error:
            self._error("runtime_identity", error)
            self.context = {"schema": SCHEMA, "observational": True, "reusable": False,
                            "model_weight_identity": "incomplete", "runtime_identity_status": "unavailable"}
        self.context_digest = rules.digest(self.context)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._recover()
        except Exception as error:
            self._error("recovery", error)
            self._needs_separator = True
        self._header_saved = self._append(self._header())

    def _error(self, code, error):
        self._errors[code] += 1
        number = getattr(error, "errno", None)
        if type(number) is int:
            self._errnos[str(number)] += 1

    def _header(self):
        return {"record_type": "attempt", "schema": SCHEMA, "attempt_id": self.attempt_id,
                "context_digest": self.context_digest, "context": self.context,
                "timestamp_unix": time.time(), "observational": True, "reusable": False}

    def _recover(self):
        """Inspect valid complete records, ignoring torn/corrupt records as evidence.

        A separator before this attempt prevents a torn tail from swallowing its
        header. Later reads can still recover subsequent complete attempts.
        """
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        except FileNotFoundError:
            return
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OSError(errno.EINVAL, "journal must be a regular file")
            for line in stream:
                if not line.endswith(b"\n"):
                    self._torn_tail_bytes = len(line)
                    self._needs_separator = True
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    claimed = row.pop("record_digest")
                    if (_hash(claimed) is None or rules.digest(row) != claimed
                            or row.get("schema") != SCHEMA or row.get("reusable") is not False
                            or row.get("observational") is not True
                            or not isinstance(row.get("attempt_id"), str)
                            or not re.fullmatch(r"[0-9a-f]{32}", row["attempt_id"])
                            or row.get("record_type") not in ("attempt", "observation")):
                        raise ValueError("invalid audit record")
                    self._previous_attempts.add(row["attempt_id"])
                    if row["record_type"] == "observation":
                        self._previous_records += 1
                except (ValueError, TypeError, KeyError, AttributeError):
                    self._invalid_history += 1

    def _append(self, record):
        # A single callback writer is used by the worker. flock also prevents
        # interleaving if an operator accidentally starts a second invocation.
        try:
            import fcntl
            record = {**record, "record_digest": rules.digest(record)}
            data = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND
                         | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK, 0o600)
            with os.fdopen(fd, "ab", buffering=0) as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise OSError(errno.EINVAL, "journal must be a regular file")
                # Diagnostics must not wait indefinitely behind another writer.
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if self._needs_separator:
                    stream.write(b"\n")
                    self._needs_separator = False
                # FileIO writes can be short. Complete every byte before fsync.
                remaining = memoryview(data)
                while remaining:
                    written = stream.write(remaining)
                    if not written:
                        raise OSError(errno.EIO, "incomplete journal write")
                    remaining = remaining[written:]
                os.fsync(stream.fileno())
            if not self._header_saved:
                directory_fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return True
        except Exception as error:
            self._needs_separator = True
            self._error("append", error)
            return False

    def save(self, row, repeat, observed):
        with self._lock:
            self._seen += 1
            try:
                sequence_hash = rules.sequence_hash(row["sequence"])
                if type(repeat) is not int or repeat < 1:
                    raise ValueError("repeat must be a positive integer")
                key = (sequence_hash, repeat)
                if key in self._saved:
                    self._errors["duplicate_observation"] += 1
                    return False
                annotation = row.get("pr100_annotation")
                annotation = annotation if isinstance(annotation, dict) else {}
                safe = _safe_observation(observed)
                record = {"record_type": "observation", "schema": SCHEMA,
                          "attempt_id": self.attempt_id, "context_digest": self.context_digest,
                          "sequence_hash": sequence_hash, "repeat": repeat,
                          "timestamp_unix": time.time(), "observational": True, "reusable": False,
                          "annotation_identity": _hash(annotation.get("annotation_identity")),
                          "source": _enum(row.get("source"), SOURCES),
                          "reference_hash": _hash(row.get("reference_hash")), "observation": safe}
                if not self._header_saved:
                    self._header_saved = self._append(self._header())
                if not self._header_saved or not self._append(record):
                    return False
                self._saved.add(key)
                self._outcomes[safe["outcome"]] += 1
                if repeat == 1:
                    self._first_outcomes[safe["outcome"]] += 1
                return True
            except Exception as error:
                self._error("observation", error)
                return False

    def progress(self):
        with self._lock:
            return {"schema": SCHEMA, "attempt_id": self.attempt_id,
                    "context_digest": self.context_digest, "observational": True, "reusable": False,
                    "model_weight_identity": "incomplete",
                    "status": "degraded" if self._errors else "ok",
                    "observations_seen": self._seen, "observations_durable": len(self._saved),
                    "unique_sequences_durable": len({key[0] for key in self._saved}),
                    "outcomes": {key: self._outcomes[key] for key in OUTCOMES},
                    "first_pass_outcomes": {key: self._first_outcomes[key] for key in OUTCOMES},
                    "repeat_observations_durable": sum(repeat > 1 for _, repeat in self._saved),
                    "errors": dict(self._errors), "errno_counts": dict(self._errnos),
                    "identity_errors": self.context.get("identity_errors", {}),
                    "previous_records_audit_only": self._previous_records,
                    "previous_attempts_audit_only": len(self._previous_attempts),
                    "invalid_history_records": self._invalid_history,
                    "ignored_torn_tail_bytes": self._torn_tail_bytes}


def _progress_fields(value):
    """Small snapshot; arbitrary worker/backend fields never cross over."""
    value = value if isinstance(value, dict) else {}
    result = {}
    for key in ("epoch", "input_count", "completed", "not_run", "not_started", "not_completed",
                "in_flight", "workers", "repeats", "observation_errors", "native_passed"):
        number = value.get(key)
        if type(number) is int and 0 <= number <= 2 ** 53:
            result[key] = number
    for key in ("native_classification_complete", "novelty_checked"):
        if type(value.get(key)) is bool:
            result[key] = value[key]
    result["status"] = _enum(value.get("status"), {"running", "complete", "interrupted"})
    outcomes = value.get("measurement_outcomes")
    if isinstance(outcomes, dict):
        result["measurement_outcomes"] = {
            key: outcomes[key] for key in OUTCOMES
            if type(outcomes.get(key)) is int and 0 <= outcomes[key] <= 2 ** 53}
    return result


class AsyncJournal:
    """Bounded writer with no disk work on the profiling event loop.

    True from save means queued, not durable. Progress exposes the last
    acknowledged inner-journal snapshot plus pending/dropped counts. A daemon
    owns initialization, fsync, recovery and snapshot I/O. A stuck filesystem
    can strand that thread, but cannot delay a profile or process shutdown.
    Close waits at most its explicit timeout, capped at one second.
    """

    def __init__(self, directory, job, requirements, *, queue_size=128):
        if type(queue_size) is not int or not 1 <= queue_size <= 4096:
            raise ValueError("queue_size must be an integer from 1 to 4096")
        self.directory = Path(directory)
        self.path = self.directory / "native-observations.jsonl"
        self._queue = queue.Queue(maxsize=queue_size)
        # Bounded in-memory submissions only; no I/O under this lock. Public
        # callers acquire non-blockingly; progress never acquires any lock.
        self._submit_lock = threading.Lock()
        self._closed = False
        self._ready = False
        self._initialization_failed = False
        self._done = threading.Event()
        self._submitted = 0
        self._accepted = 0
        self._processed = 0
        self._dropped = 0
        self._write_failed = 0
        self._submit_errors = Counter()
        self._writer_errors = Counter()
        self._writer_errnos = Counter()
        self._cached = {
            "schema": SCHEMA, "attempt_id": None, "context_digest": None,
            "observational": True, "reusable": False, "model_weight_identity": "incomplete",
            "status": "initializing", "observations_seen": 0, "observations_durable": 0,
            "unique_sequences_durable": 0, "outcomes": {key: 0 for key in OUTCOMES},
            "first_pass_outcomes": {key: 0 for key in OUTCOMES},
            "repeat_observations_durable": 0, "errors": {}, "errno_counts": {},
            "identity_errors": {}, "previous_records_audit_only": 0,
            "previous_attempts_audit_only": 0, "invalid_history_records": 0,
            "ignored_torn_tail_bytes": 0,
        }
        # Retain only context inputs, never environment or unrelated job fields.
        context_job = {key: copy.deepcopy(job[key]) for key in
                       ("source_sha", "native_tools", "challenge") if key in job}
        context_requirements = copy.deepcopy(requirements)
        self._thread = threading.Thread(target=self._run,
            args=(context_job, context_requirements), name="pr100-observation-writer", daemon=True)
        try:
            self._thread.start()
        except Exception as error:
            self._writer_error("thread_start", error)
            self._initialization_failed = True
            self._done.set()

    def _writer_error(self, code, error):
        self._writer_errors[code] += 1
        number = getattr(error, "errno", None)
        if type(number) is int:
            self._writer_errnos[str(number)] += 1

    def save(self, row, repeat, observed, progress=None):
        """Enqueue a small allowlisted copy, or explicitly drop without waiting."""
        self._submitted += 1
        if not self._submit_lock.acquire(blocking=False):
            self._dropped += 1
            self._submit_errors["submission_busy"] += 1
            return False
        try:
            if self._closed or self._initialization_failed:
                self._dropped += 1
                self._submit_errors["closed" if self._closed else "initialization"] += 1
                return False
            sequence = row.get("sequence") if isinstance(row, dict) else None
            if (type(sequence) is not str or not 1 <= len(sequence) <= 4096
                    or type(repeat) is not int or not 1 <= repeat <= 2 ** 31):
                self._dropped += 1
                self._submit_errors["invalid_observation"] += 1
                return False
            annotation = row.get("pr100_annotation")
            annotation = annotation if isinstance(annotation, dict) else {}
            safe_row = {"sequence": sequence, "source": _enum(row.get("source"), SOURCES),
                        "reference_hash": _hash(row.get("reference_hash")),
                        "pr100_annotation": {"annotation_identity": _hash(annotation.get("annotation_identity"))}}
            item = (safe_row, repeat, _safe_observation(observed),
                    _progress_fields(progress) if progress is not None else None)
            # Caps queue memory even for pathological numeric backend values.
            if len(json.dumps(item, allow_nan=False).encode()) > 16384:
                self._dropped += 1
                self._submit_errors["observation_too_large"] += 1
                return False
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                self._submit_errors["queue_full"] += 1
                return False
            self._accepted += 1
            return True
        except Exception:
            self._dropped += 1
            self._submit_errors["invalid_observation"] += 1
            return False
        finally:
            self._submit_lock.release()

    def progress(self):
        # Writer replaces this small snapshot after inner locks are released.
        # Main-thread readers never touch the inner Journal lock or disk.
        result = copy.deepcopy(self._cached)
        errors = dict(self._submit_errors)
        for key, count in self._writer_errors.copy().items():
            errors[key] = errors.get(key, 0) + count
        pending = max(0, self._accepted - self._processed)
        result.update(
            writer_ready=self._ready, writer_alive=self._thread.is_alive(),
            writer_done=self._done.is_set(), closed=self._closed,
            queue_capacity=self._queue.maxsize, observations_submitted=self._submitted,
            observations_accepted=self._accepted, observations_processed=self._processed,
            observations_pending=pending, observations_dropped=self._dropped,
            observations_write_failed=self._write_failed,
            async_errors=errors, async_errno_counts=dict(self._writer_errnos),
        )
        if errors or result["status"] == "degraded":
            result["status"] = "degraded"
        elif pending:
            result["status"] = "pending" if self._ready else "initializing"
        return result

    def close(self, timeout=1.0):
        """Reject new observations and drain for at most the explicit timeout."""
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 1:
            raise ValueError("close timeout must be between zero and one second")
        self._closed = True
        if self._thread.ident is not None:
            self._thread.join(timeout=float(timeout))
        return self.progress()

    def _can_finish(self):
        if not self._closed or not self._submit_lock.acquire(blocking=False):
            return False
        try:
            return self._queue.empty()
        finally:
            self._submit_lock.release()

    def _write_progress(self, snapshot):
        """Write separate diagnostics, never result.json or publication inputs."""
        value = {**snapshot, "schema": "pr100-native-progress-v1", "observational": True,
                 "reusable": False, "observation_journal": self.progress()}
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=".native-progress-", suffix=".tmp", dir=self.directory)
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / "native-progress.json")
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _run(self, job, requirements):
        try:
            try:
                inner = Journal(self.directory, job, requirements)
                self._cached = inner.progress()
                self._ready = True
            except Exception as error:
                self._writer_error("initialization", error)
                self._initialization_failed = True
                return
            while not self._can_finish():
                try:
                    item = self._queue.get(timeout=0.025)
                except queue.Empty:
                    continue
                row, repeat, observed, snapshot = item
                try:
                    saved = inner.save(row, repeat, observed)
                    self._cached = inner.progress()
                    if not saved:
                        self._write_failed += 1
                        self._writer_errors["save"] += 1
                except Exception as error:
                    self._write_failed += 1
                    self._writer_error("save", error)
                finally:
                    self._processed += 1
                    self._queue.task_done()
                if snapshot is not None:
                    try:
                        self._write_progress(snapshot)
                    except Exception as error:
                        self._writer_error("progress_snapshot", error)
        finally:
            if self._initialization_failed:
                # Serializes only queue bookkeeping, never I/O. A submit already
                # in progress finishes before the failure drain is counted.
                with self._submit_lock:
                    while True:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            break
                        self._processed += 1
                        self._write_failed += 1
                        self._queue.task_done()
            self._done.set()
