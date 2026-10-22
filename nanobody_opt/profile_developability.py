#!/usr/bin/env python3
"""Profile the full six-flag TNP developability gate for a batch of sequences.

Runs in the metanano pixi environment. Reuses ``filter_layer2._cached_developability``
so the per-sequence cache under ``--cache-dir`` is shared with the Layer-2 gate
(one TNP run per sequence, ever). Writes ``{sequence: profile}`` keyed by the
normalized sequence so the caller can attach the profile to parent candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt.filter_layer2 import _cached_developability


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    target = str(payload.get("target", ""))
    sequences = []
    for item in payload.get("sequences", []):
        sequence = str(item).strip().upper() if isinstance(item, str) else ""
        if sequence and sequence not in sequences:
            sequences.append(sequence)
    if not sequences:
        args.output.write_text(json.dumps({"target": target, "profiles": {}}, indent=2) + "\n")
        return 0

    rows = [{"sequence": sequence} for sequence in sequences]
    results = _cached_developability(rows, args.cache_dir, args.concurrency)
    profiles = {}
    for row, profile in zip(rows, results, strict=True):
        profiles[row["sequence"]] = profile
    args.output.write_text(json.dumps({"target": target, "profiles": profiles}, indent=2) + "\n")
    print(f"DEVELOPABILITY_PROFILES_READY target={target} sequences={len(profiles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
