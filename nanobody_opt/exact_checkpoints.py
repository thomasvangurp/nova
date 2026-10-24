"""One aggregate per scoring run, including durable completed-wave evidence."""
from pathlib import Path


def exact_paths(root: Path, prefix: str = "**") -> list[Path]:
    selected: dict[Path, Path] = {}
    for path in sorted(root.glob(f"{prefix}/exact_partial_[0-9][0-9][0-9].json")):
        selected[path.parent] = path
    for path in root.glob(f"{prefix}/exact_all.json"):
        selected[path.parent] = path
    return sorted(selected.values())
