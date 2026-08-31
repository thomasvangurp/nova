#!/usr/bin/env python3
"""Merge resumable Boltz score shards while rejecting mixed oracle settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _key(record: dict[str, Any]) -> str:
    return str(record.get("inchikey") or record.get("name") or record["smiles"])


def merge(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one input is required")
    payloads = [json.loads(path.read_text()) for path in paths]
    inference = payloads[0]["inference"]
    oracle_inference = {
        key: value for key, value in inference.items() if key != "preprocessing_threads"
    }
    scorers = {row["scorer"] for payload in payloads for row in payload["scores"]}
    if (
        any(
            {key: value for key, value in payload["inference"].items() if key != "preprocessing_threads"}
            != oracle_inference
            for payload in payloads[1:]
        )
        or len(scorers) != 1
    ):
        raise ValueError("refusing to merge score files produced by different oracle settings")
    records: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for record in payload["scores"]:
            key = _key(record)
            if key in records and float(records[key]["score"]) != float(record["score"]):
                raise ValueError(f"conflicting duplicate score for {key}")
            records[key] = record
    scores = sorted(records.values(), key=lambda row: (int(row["reaction"]), str(row["name"])))
    return {
        "reaction": 0,
        "count": len(scores),
        "wall_seconds": float(sum(payload.get("wall_seconds", 0.0) for payload in payloads)),
        "gpus": payloads[0].get("gpus", []),
        "shards_per_gpu": payloads[0].get("shards_per_gpu", 1),
        "inference": inference,
        "sources": [str(path) for path in paths],
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "count": result["count"]}))


if __name__ == "__main__":
    main()
