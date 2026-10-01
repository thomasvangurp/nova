#!/usr/bin/env python3
"""Download only the public, revision-pinned EasyNano model snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

from huggingface_hub import snapshot_download


ROOT = Path("/root/easynano-models")
MODELS = {
    "ESMFold2": ("Biohub/ESMFold2", "69869f737beffec5294845ede23db5fc0b4f509e"),
    "ESMFold2-Fast": ("Biohub/ESMFold2-Fast", "45fe8656f5b3ef493c17fcf9abe9a2968902e712"),
    "ESMC-6B": ("Biohub/ESMC-6B", "af1602ba7406f521b11bf8f81d52af378cde09e4"),
}


def main() -> int:
    logging.disable(logging.CRITICAL)
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for destination, (repository, revision) in MODELS.items():
            local = ROOT / destination
            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_dir=local,
                local_dir_use_symlinks=False,
            )
            files = [path for path in local.rglob("*") if path.is_file()]
            records.append({
                "repository": repository,
                "revision": revision,
                "destination": str(local),
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
            })
    except Exception as error:
        (ROOT / "download-status.json").write_text(json.dumps({
            "stage": "failed", "error_class": type(error).__name__,
        }) + "\n")
        return 1
    (ROOT / "download-status.json").write_text(json.dumps({
        "stage": "complete", "models": records,
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
