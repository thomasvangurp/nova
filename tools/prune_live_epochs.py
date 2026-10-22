"""Free disk by pruning old live epoch directories, after their measurements are safely in SQLite.

Each ``/root/live/epoch<N>/producer`` tree holds 0.5-2.7 GB of regenerable search output (candidate
CSVs, race rounds, structures).  The *measurements* in it are the raw ``exact_all.json`` payloads,
which ``nanobody_opt.import_exact_screens`` writes to the mutation database as ``local_exact_raw``
observations.  This tool, for every epoch older than ``--keep`` newest directories:

  1. recomputes the observation ids the importer produces for that epoch's exact payloads;
  2. if any id is missing from SQLite, runs the importer (the same command the top-up uses) and
     recomputes; an epoch that still has a missing id is left untouched;
  3. only then removes regenerable producer output and writes ``epoch<N>/PRUNED.json``.
     Raw payloads and structures stay at their original per-pass paths.

It is a dry run unless ``--execute`` is given.  Rankings, tickets (``checkpoints``) and the other
small top-level files are kept.  Run it with the interpreter that can import ``nanobody_opt``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from nanobody_opt.storage_retention import prune_generated_tree, retained_evidence

EPOCH_DIR = re.compile(r"^epoch(\d+)$")
ID_CHUNK = 500


def epoch_dirs(live_root: Path) -> list[Path]:
    found = [(int(m.group(1)), p) for p in live_root.iterdir()
             if (m := EPOCH_DIR.match(p.name)) and p.is_dir() and not p.is_symlink()]
    return [p for _, p in sorted(found)]


def prune_candidates(live_root: Path, keep: int) -> list[Path]:
    """Every epoch directory except the ``keep`` newest ones."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    dirs = epoch_dirs(live_root)
    return dirs[:-keep] if len(dirs) > keep else []


def tree_stats(path: Path) -> tuple[int, float]:
    """(total bytes, newest file mtime) of a tree; mtime 0.0 when empty."""
    total, newest = 0, 0.0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                info = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += info.st_size
            newest = max(newest, info.st_mtime)
    return total, newest


def tree_bytes(path: Path) -> int:
    return tree_stats(path)[0]


def expected_ids(epoch_dir: Path, target: str) -> list[str]:
    """Observation ids the importer would write for this epoch's raw exact payloads."""
    import pandas as pd
    from nanobody_opt.import_exact_screens import (
        _bootstrap_fields, _sha256, discover_exact_files, exact_observations)

    archive = epoch_dir / "archive.csv"
    if not archive.is_file():
        raise FileNotFoundError(f"{archive} is missing; cannot recompute the importer ids")
    fields = _bootstrap_fields(pd.read_csv(archive), recent=300, field_size=49, count=1, seed=0)
    paths = discover_exact_files([epoch_dir])
    observations, _summary = exact_observations(
        paths, target=target, fields=fields, metadata={}, archive_sha256=_sha256(archive))
    return sorted({item["observation_id"] for item in observations})


def missing_ids(database: Path, ids: list[str]) -> list[str]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=120)
    try:
        connection.execute("PRAGMA query_only=1")
        have: set[str] = set()
        for start in range(0, len(ids), ID_CHUNK):
            chunk = ids[start:start + ID_CHUNK]
            marks = ",".join("?" * len(chunk))
            have.update(row[0] for row in connection.execute(
                f"SELECT observation_id FROM observations WHERE observation_id IN ({marks})", chunk))
    finally:
        connection.close()
    return [item for item in ids if item not in have]


def run_importer(python: str, database: Path, target: str, epoch_dir: Path) -> int:
    command = [python, "-m", "nanobody_opt.import_exact_screens", "--database", str(database),
               "--target", target, "--archive", str(epoch_dir / "archive.csv"),
               "--exact-root", str(epoch_dir)]
    return subprocess.run(command, capture_output=True, text=True, timeout=1800).returncode


def remove_producer_tree(live_root: Path, epoch_dir: Path) -> int:
    """Prune producer output while retaining every raw payload and pose path."""
    producer = epoch_dir / "producer"
    if producer.is_symlink() or not producer.is_dir():
        return 0
    if producer.resolve() != (live_root.resolve() / epoch_dir.name / "producer"):
        raise RuntimeError(f"refusing to delete {producer}: not directly under {live_root}")
    return prune_generated_tree(producer)["removed_bytes"]


def prune_epoch(args: argparse.Namespace, epoch_dir: Path) -> dict:
    result: dict = {"epoch_dir": epoch_dir.name}
    producer = epoch_dir / "producer"
    if not producer.is_dir():
        result.update(producer_bytes=0, status="already_pruned")
        return result
    if not any(not p.is_symlink() and p.is_file() and not retained_evidence(p)
               for p in producer.rglob("*")):
        result.update(producer_bytes=0, status="already_pruned")
        return result
    result["producer_bytes"], newest = tree_stats(producer)
    idle_seconds = args.min_idle_minutes * 60
    if idle_seconds and time.time() - newest < idle_seconds:
        result["status"] = "recently_modified"  # an old epoch directory that is being worked on again
        return result
    ids = expected_ids(epoch_dir, args.target)
    absent = missing_ids(args.database, ids)
    result.update(observation_ids=len(ids), missing_before=len(absent))
    if absent:
        if not args.execute:
            result["status"] = "would_import_then_prune"
            return result
        result["importer_rc"] = run_importer(args.python, args.database, args.target, epoch_dir)
        absent = missing_ids(args.database, ids)
        result["missing_after"] = len(absent)
    if absent:
        result["status"] = "kept_missing_measurements"
        return result
    if not args.execute:
        result["status"] = "would_prune"
        return result
    freed = remove_producer_tree(args.live_root, epoch_dir)
    marker = {"pruned_at": time.time(), "verified_observation_ids": len(ids), "freed_bytes": freed,
              "kept": "rankings, checkpoints, config, raw exact payloads and per-pass structures at their original paths"}
    (epoch_dir / "PRUNED.json").write_text(json.dumps(marker, indent=1))
    result.update(status="pruned", freed_bytes=freed)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live-root", type=Path, default=Path("/root/live"))
    parser.add_argument("--database", type=Path, default=Path("/root/artifacts/nanobody_mutations.sqlite"))
    parser.add_argument("--target", default="P20809")
    parser.add_argument("--keep", type=int, default=10, help="newest epoch directories left untouched")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the importer subprocess")
    parser.add_argument("--min-idle-minutes", type=float, default=30.0,
                        help="skip an epoch whose producer tree changed more recently than this")
    parser.add_argument("--quiet", action="store_true", help="do not print epochs that are already pruned")
    parser.add_argument("--execute", action="store_true", help="import missing rows and delete (default: dry run)")
    args = parser.parse_args(argv)

    freed_total = 0
    for epoch_dir in prune_candidates(args.live_root, args.keep):
        try:
            result = prune_epoch(args, epoch_dir)
        except Exception as error:  # one bad epoch must not stop the others, and is never deleted
            result = {"epoch_dir": epoch_dir.name, "status": "error", "error": f"{type(error).__name__}: {error}"[:200]}
        freed_total += result.get("freed_bytes", 0)
        if not (args.quiet and result.get("status") == "already_pruned"):
            print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"summary": True, "at": time.strftime("%FT%T%z"), "execute": bool(args.execute),
                      "freed_gb": round(freed_total / 1e9, 1),
                      "free_gb_now": round(shutil.disk_usage(args.live_root).free / 1e9, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
