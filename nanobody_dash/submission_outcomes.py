"""Pi evidence for distinguishing unused publications from submission inputs."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time

from nanobody_opt.publication_ledger import SCHEMA, identity, save


def export(database: Path, archives: Path, output: Path, *, target="P20809", now=None):
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT epoch,sequence FROM entries WHERE sequence IS NOT NULL").fetchall()
    finalized = {int(epoch) for epoch, _ in rows}
    blocked = {identity(sequence) for _, sequence in rows if sequence}
    archived = set()
    for folder in Path(archives).glob("epoch[0-9]*"):
        suffix = folder.name.removeprefix("epoch")
        if not suffix.isdecimal():
            continue
        epoch = int(suffix)
        for path in folder.glob("nanobody_result*.json"):
            try:
                artifact = json.loads(path.read_text())
                if artifact.get("epoch") != epoch or artifact.get("target") != target:
                    continue
                pool = list(artifact.get("sequence_pool") or [])
                pool += [r["sequence"] for r in artifact.get("submissions", []) if r.get("sequence")]
                if not pool or not all(isinstance(s, str) and s for s in pool):
                    continue
                blocked.update(identity(sequence) for sequence in pool)
                if path.name == "nanobody_result.json":
                    archived.add(epoch)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
    covered = sorted(archived & finalized)
    if not covered:
        raise ValueError("no finalized archived submission epochs")
    payload = {"schema": SCHEMA, "target": target,
               "generated_at": time.time() if now is None else now,
               "covered_epochs": covered, "possibly_submitted_hashes": sorted(blocked)}
    save(output, payload)
    return {"covered_epochs": len(payload["covered_epochs"]),
            "latest_finalized_epoch": max(covered), "protected_hashes": len(blocked)}
