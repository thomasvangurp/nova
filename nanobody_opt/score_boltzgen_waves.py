"""Exact-score nanobodies in independently publishable, resumable waves.

Each wave invokes the validator-matched sharded scorer as a separate process.
Its output is validated and atomically promoted before the next wave begins, so
a live producer can rank and confirm new contenders without waiting for the
entire candidate bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _sequences(path: Path) -> list[str]:
    raw = json.loads(path.read_text())
    items = raw if isinstance(raw, list) else raw.get("cases") or raw.get("sequences") or []
    sequences = [item.get("sequence") if isinstance(item, dict) else item for item in items]
    sequences = [item.strip().upper() for item in sequences if isinstance(item, str) and item.strip()]
    return list(dict.fromkeys(sequences))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _components(payload: dict, target: str) -> dict[str, dict]:
    if str(payload.get("target")) != target:
        raise ValueError(
            f"wave targets {payload.get('target')!r}, expected {target!r}"
        )
    merged: dict[str, dict] = {}
    for by_sequence in payload.get("per_nanobody_components", {}).values():
        for sequence, by_target in by_sequence.items():
            if target not in by_target:
                raise ValueError(f"wave result lacks {target} metrics for {sequence}")
            merged[sequence] = by_target
    return merged


def _validated_wave(
    path: Path,
    expected: list[str],
    target: str,
    require_structures: bool = False,
) -> dict:
    payload = json.loads(path.read_text())
    found = _components(payload, target)
    if set(found) != set(expected) or len(found) != len(expected):
        raise ValueError(
            f"wave result mismatch: expected {len(expected)}, found {len(found)}"
        )
    if int(payload.get("n", -1)) != len(expected):
        raise ValueError("wave result count does not match its input")
    if require_structures and set(payload.get("structures", {})) != set(expected):
        raise ValueError("wave result lacks required retained structures")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=1)
    parser.add_argument(
        "--wave-size",
        type=int,
        default=80,
        help="candidates promoted together; 80 is eight per GPU on a 10-GPU box",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=1,
        help="sequences sharing one BoltzGen predict call; validator parity requires one",
    )
    parser.add_argument("--scorer", type=Path)
    parser.add_argument(
        "--capture-structures",
        action="store_true",
        help="retain one refolded target complex per exact observation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="score only the first N ranked sequences for a short update sweep",
    )
    args = parser.parse_args()

    if (
        args.wave_size < 1
        or args.predict_batch_size < 1
        or (args.limit is not None and args.limit < 1)
    ):
        raise SystemExit("wave size, predict batch size, and limit must be positive")
    sequences = _sequences(args.sequences)
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise SystemExit(f"no sequences parsed from {args.sequences}")
    scorer = args.scorer or Path(__file__).with_name("score_boltzgen_batch.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.is_file():
        observation_id = str(json.loads(manifest_path.read_text())["observation_id"])
    else:
        observation_id = str(uuid.uuid4())
    completed = []
    all_components: dict[str, dict] = {}
    all_structures: dict[str, dict] = {}

    for wave_index, start in enumerate(range(0, len(sequences), args.wave_size)):
        wave = sequences[start : start + args.wave_size]
        stem = f"wave_{wave_index:03d}"
        wave_input = args.output_dir / f"{stem}_input.json"
        wave_output = args.output_dir / f"{stem}_exact.json"
        _atomic_json(wave_input, {"target": args.target, "sequences": wave})

        if wave_output.exists():
            payload = _validated_wave(
                wave_output, wave, args.target, args.capture_structures
            )
            action = "RESUMED"
        else:
            temporary_output = args.output_dir / f".{stem}_exact.incomplete.json"
            try:
                temporary_output.unlink()
            except FileNotFoundError:
                pass
            command = [
                sys.executable,
                str(scorer),
                "--sequences",
                str(wave_input),
                "--config",
                str(args.config),
                "--target",
                args.target,
                "--gpus",
                args.gpus,
                "--shards-per-gpu",
                str(args.shards_per_gpu),
                "--batch-size",
                str(args.predict_batch_size),
                "--work-root",
                str(args.output_dir / "work"),
                "--out",
                str(temporary_output),
            ]
            if args.capture_structures:
                command.extend([
                    "--structure-dir",
                    str(args.output_dir / "structures"),
                    "--require-structures",
                ])
            subprocess.run(command, check=True)
            payload = _validated_wave(
                temporary_output, wave, args.target, args.capture_structures
            )
            os.replace(temporary_output, wave_output)
            action = "READY"

        all_components.update(_components(payload, args.target))
        for sequence, by_target in payload.get("structures", {}).items():
            all_structures.setdefault(sequence, {}).update(by_target)
        completed.append(
            {
                "wave": wave_index,
                "start": start,
                "count": len(wave),
                "input": str(wave_input),
                "output": str(wave_output),
                "sha256": _sha256(wave_output),
            }
        )
        _atomic_json(
            manifest_path,
            {
                "observation_id": observation_id,
                "target": args.target,
                "source": str(args.sequences),
                "source_sha256": _sha256(args.sequences),
                "candidate_count": len(sequences),
                "wave_size": args.wave_size,
                "completed_count": sum(item["count"] for item in completed),
                "completed_waves": completed,
            },
        )
        print(
            f"WAVE_{action} {wave_index}: {len(wave)} candidates -> {wave_output}",
            flush=True,
        )

    if len(all_components) != len(sequences):
        raise SystemExit(
            f"wave merge mismatch: expected {len(sequences)}, found {len(all_components)}"
        )
    _atomic_json(
        args.output_dir / "exact_all.json",
        {
            "target": args.target,
            "observation_id": observation_id,
            "n": len(sequences),
            "wave_count": len(completed),
            "per_nanobody_components": {"0": all_components},
            "structures": all_structures,
        },
    )
    print(
        f"ALL_WAVES_READY: {len(sequences)} candidates -> "
        f"{args.output_dir / 'exact_all.json'}"
    )


if __name__ == "__main__":
    main()
