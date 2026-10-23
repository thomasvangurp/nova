"""Durable per-epoch budget for new screening dispatches, independent of caches."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

POLICY = "unseen-candidates-v2"


def identity(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def snapshot(root: Path, epoch: int, limit: int) -> dict:
    path = root / "fresh_screen_budget.json"
    data = json.loads(path.read_text()) if path.exists() else {
        "policy": POLICY, "epoch": epoch, "limit": limit, "reserved": [],
    }
    if (data["policy"], data["epoch"], data["limit"]) != (POLICY, epoch, limit):
        raise ValueError("fresh screen budget identity mismatch")
    if len(set(data["reserved"])) != len(data["reserved"]) or len(data["reserved"]) > limit:
        raise ValueError("invalid fresh screen budget ledger")
    return data


def pending(records: list[dict], root: Path, epoch: int, limit: int) -> list[dict]:
    used = set(snapshot(root, epoch, limit)["reserved"])
    return [r for r in records if identity(r["sequence"]) not in used][:limit - len(used)]


def reserve(sequences: list[str], root: Path, epoch: int, limit: int) -> list[str]:
    """Charge before dispatch. A crash cannot reset the cap or repeat a charge.

    An interrupted/failed dispatch stays charged and is reported separately
    from successful evaluations; silently retrying it could exceed the cap.
    Cheap proposal rejection happens before this ledger and costs no budget.
    """
    root.mkdir(parents=True, exist_ok=True)
    with (root / "fresh_screen_budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = snapshot(root, epoch, limit)
        used = set(data["reserved"])
        selected = []
        for sequence in sequences:
            key = identity(sequence)
            if key not in used and len(used) < limit:
                selected.append(sequence)
                used.add(key)
        data["reserved"] = sorted(used)
        path = root / "fresh_screen_budget.json"
        temporary = path.with_suffix(".next")
        with temporary.open("w") as handle:
            json.dump(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return selected
