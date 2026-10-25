"""Publication history is not proof of an actual submission.

Keep the original ledger intact. Release an unused entry only with a recent
Pi outcome snapshot covering its last publication epoch, and never release a
sequence that appeared in any Pi submission input or validator result.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

SCHEMA = "pi-submission-outcomes-v1"
MAXIMUM_AGE = 900


def identity(sequence):
    return hashlib.sha256(sequence.strip().upper().encode()).hexdigest()


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def record(ledger, *, epoch, target, sequences):
    """Persist the attempt before exposing the artifact to Pi, including reuses."""
    ledger = Path(ledger)
    path = ledger.with_suffix(".publication_events.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        events = read(path)
        entries = events.setdefault(target, {})
        for sequence in sequences:
            key = identity(sequence)
            entries[key] = max(int(entries.get(key, 0)), int(epoch))
        save(path, events)


def effective(ledger, *, epoch, target, history, outcomes, output, now=None):
    """Return a current exclusion file; uncertainty retains the original ban."""
    ledger = Path(ledger)
    proof = read(outcomes)
    clock = time.time() if now is None else now
    try:
        valid = (proof["schema"] == SCHEMA and proof["target"] == target
                 and 0 <= clock - float(proof["generated_at"]) <= MAXIMUM_AGE)
        covered = set(proof["covered_epochs"])
        blocked = set(proof["possibly_submitted_hashes"])
        valid = (valid and bool(covered) and all(type(e) is int and 0 <= e < epoch for e in covered)
                 and all(isinstance(h, str) and len(h) == 64 for h in blocked))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        return ledger
    contiguous_start = max(covered)
    while contiguous_start - 1 in covered:
        contiguous_start -= 1
    events = read(ledger.with_suffix(".publication_events.json")).get(target, {})
    if not isinstance(events, dict) or any(type(value) is not int for value in events.values()):
        return ledger
    events = dict(events)
    # Bootstrap final artifacts conservatively; intermediate revisions are
    # covered by the durable event journal once this policy is installed.
    for past in range(max(min(covered), epoch - 60), epoch):
        artifact = read(Path(history) / f"epoch{past}.json")
        if artifact.get("epoch") != past or artifact.get("target") != target:
            continue
        for row in artifact.get("submissions", []):
            if isinstance(row, dict) and isinstance(row.get("sequence"), str):
                key = identity(row["sequence"])
                events[key] = max(int(events.get(key, 0)), past)
    sequences = ledger.read_text().splitlines() if ledger.exists() else []
    retained, released = [], []
    for sequence in sequences:
        key = identity(sequence)
        last = events.get(key)
        # Require continuous finalized coverage after the last publication.
        known_unused = (type(last) is int and last in covered and key not in blocked
                        and last >= contiguous_start)
        (released if known_unused else retained).append(sequence)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, prefix=output.name + ".")
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write("".join(s + "\n" for s in retained))
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    save(output.with_suffix(".receipt.json"), {
        "policy": SCHEMA, "epoch": epoch, "target": target,
        "proof_generated_at": proof["generated_at"], "retained": len(retained),
        "released_unused": len(released), "last_finalized_epoch": max(covered),
    })
    return output
