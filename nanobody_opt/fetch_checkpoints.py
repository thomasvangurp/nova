"""Fetch exact BoltzGen checkpoints through Hugging Face's Xet client."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="boltzgen/boltzgen-1")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--specs-json", required=True)
    args = parser.parse_args()

    specs = json.loads(args.specs_json)
    args.local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_count = 0
    for spec in specs:
        name, expected = spec["name"], spec["sha256"]
        destination = args.local_dir / name
        if destination.is_file() and sha256(destination) == expected:
            print(f"CACHED {name}", flush=True)
            continue
        downloaded = Path(hf_hub_download(
            repo_id=args.repo,
            filename=name,
            local_dir=args.local_dir,
            force_download=True,
        ))
        actual = sha256(downloaded)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}: {actual} != {expected}")
        downloaded_count += 1
        print(f"OK {name}", flush=True)
    print(f"ALL_CHECKPOINTS_OK DOWNLOADED={downloaded_count}")


if __name__ == "__main__":
    main()
