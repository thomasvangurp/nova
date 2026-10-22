"""Remove regenerable output without invalidating retained pose paths."""
from __future__ import annotations

import os
from pathlib import Path


def retained_evidence(path: Path) -> bool:
    # Keep the original path: observation_artifacts and existing readers refer
    # to it. A basename is NOT a pose identity (all replicates share it).
    return path.name.endswith((".cif", ".cif.gz", ".pdb", ".pdb.gz")) or path.name == "exact_all.json"


def prune_generated_tree(root: Path) -> dict[str, int]:
    """Keep raw payloads and every distinct pose in place; never follow links.

    Callers must first verify that the measurements are sealed in SQLite and
    that the tree is no longer being written. Symlinks are retained, including
    links to structure directories. Repeated calls are harmless.
    """
    result = {"removed_files": 0, "removed_bytes": 0, "retained_files": 0}
    if root.is_symlink() or not root.is_dir():
        return result
    for directory, dirs, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or retained_evidence(path):
                result["retained_files"] += 1
                continue
            size = path.stat().st_size
            path.unlink()
            result["removed_files"] += 1
            result["removed_bytes"] += size
        for name in dirs:
            path = Path(directory) / name
            if not path.is_symlink() and not any(path.iterdir()):
                path.rmdir()
    if not any(root.iterdir()):
        root.rmdir()
    return result
