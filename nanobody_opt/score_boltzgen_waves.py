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

from nanobody_opt.diffusion_sample_selection import (
    EXPECTED_DIFFUSION_SAMPLES,
    SCORER_PROTOCOL,
    require_expected_sample_count,
)


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
    expected_diffusion_samples: int = EXPECTED_DIFFUSION_SAMPLES,
) -> dict:
    payload = json.loads(path.read_text())
    found = _components(payload, target)
    if set(found) != set(expected) or len(found) != len(expected):
        raise ValueError(
            f"wave result mismatch: expected {len(expected)}, found {len(found)}"
        )
    if int(payload.get("n", -1)) != len(expected):
        raise ValueError("wave result count does not match its input")
    if payload.get("scorer_protocol") != SCORER_PROTOCOL:
        raise ValueError("wave predates validator dual-sample fidelity")
    if require_structures and set(payload.get("structures", {})) != set(expected):
        raise ValueError("wave result lacks required retained structures")
    if require_structures:
        for sequence in expected:
            record = payload["structures"][sequence].get(target, {})
            if not all(record.get(key) for key in (
                "path", "writer_path", "sample_selection"
            )):
                raise ValueError(
                    f"wave result lacks dual-sample structures for {sequence}"
                )
            require_expected_sample_count(
                record["sample_selection"],
                expected=expected_diffusion_samples,
                context=f"wave result for {sequence}",
            )
    return payload


def _archive_legacy_wave(path: Path) -> Path:
    """Preserve, but never resume, an output from the old scoring protocol."""
    archived = path.with_name(f"{path.stem}.legacy-pre-dual-sample.json")
    if archived.exists():
        archived = path.with_name(
            f"{path.stem}.legacy-pre-dual-sample-{uuid.uuid4().hex}.json"
        )
    os.replace(path, archived)
    return archived


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length_waves(sequences: list[str], wave_size: int) -> list[list[str]]:
    """Keep every admitted candidate while satisfying the fast scorer's shape.

    Preserve first-seen length order and candidate order within each length.
    Homogeneous inputs keep their existing wave boundaries and resume identity.
    """
    by_length: dict[int, list[str]] = {}
    for sequence in sequences:
        by_length.setdefault(len(sequence), []).append(sequence)
    return [bucket[start:start + wave_size]
            for bucket in by_length.values()
            for start in range(0, len(bucket), wave_size)]


def _wave_command(
    *,
    scorer: Path,
    wave_input: Path,
    config: Path,
    target: str,
    gpus: str,
    shards_per_gpu: int,
    predict_batch_size: int,
    output_dir: Path,
    out: Path,
    capture_structures: bool,
) -> list[str]:
    """The scorer argv for one wave.

    Extracted so the daemon job builder can be tested against the command
    that actually runs. The previous test built a command by hand and so
    agreed with a job builder that was wired to the wrong CLI entirely.
    """
    command = [
        sys.executable,
        str(scorer),
        "--sequences", str(wave_input),
        "--config", str(config),
        "--target", target,
        "--gpus", gpus,
        "--shards-per-gpu", str(shards_per_gpu),
        "--batch-size", str(predict_batch_size),
        "--work-root", str(output_dir / "work"),
        "--out", str(out),
    ]
    if capture_structures:
        command.extend([
            "--structure-dir", str(output_dir / "structures"),
            "--require-structures",
        ])
    return command


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
    parser.add_argument(
        "--expected-diffusion-samples",
        type=int,
        default=EXPECTED_DIFFUSION_SAMPLES,
        help=(
            "reject a wave folded with a different diffusion-sample count; the "
            "reported metrics are a maximum over samples, so fewer samples "
            "biases the observation instead of only adding noise"
        ),
    )
    args = parser.parse_args()

    if args.expected_diffusion_samples < 1:
        raise SystemExit("expected diffusion-sample count must be positive")
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
        _atomic_json(manifest_path, {
            "observation_id": observation_id, "target": args.target,
            "source": str(args.sequences), "source_sha256": _sha256(args.sequences),
            "candidate_count": len(sequences), "completed_count": 0,
            "completed_waves": [],
        })
    completed = []
    all_components: dict[str, dict] = {}
    all_structures: dict[str, dict] = {}

    start = 0
    for wave_index, wave in enumerate(_length_waves(sequences, args.wave_size)):
        stem = f"wave_{wave_index:03d}"
        wave_input = args.output_dir / f"{stem}_input.json"
        wave_output = args.output_dir / f"{stem}_exact.json"
        _atomic_json(wave_input, {"target": args.target, "sequences": wave})

        if wave_output.exists():
            try:
                payload = _validated_wave(
                    wave_output,
                    wave,
                    args.target,
                    args.capture_structures,
                    args.expected_diffusion_samples,
                )
            except ValueError as error:
                archived = _archive_legacy_wave(wave_output)
                print(
                    f"WAVE_PROTOCOL_REFRESH {wave_index}: {error}; "
                    f"preserved={archived}",
                    flush=True,
                )
            else:
                action = "RESUMED"
        if not wave_output.exists():
            temporary_output = args.output_dir / f".{stem}_exact.incomplete.json"
            try:
                temporary_output.unlink()
            except FileNotFoundError:
                pass
            command = _wave_command(
                scorer=scorer,
                wave_input=wave_input,
                config=args.config,
                target=args.target,
                gpus=args.gpus,
                shards_per_gpu=args.shards_per_gpu,
                predict_batch_size=args.predict_batch_size,
                output_dir=args.output_dir,
                out=temporary_output,
                capture_structures=args.capture_structures,
            )
            # The daemon is NOT hooked here either. `command` runs the fast
            # adapter, whose flags are --work-root/--structure-dir; the
            # scorer's own are --shard-root/--capture-structures. The hook
            # belongs where the scorer argv finally exists: in the adapter.
            subprocess.run(command, check=True)
            payload = _validated_wave(
                temporary_output,
                wave,
                args.target,
                args.capture_structures,
                args.expected_diffusion_samples,
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
                "sequence_length": len(wave[0]),
                "input": str(wave_input),
                "output": str(wave_output),
                "sha256": _sha256(wave_output),
            }
        )
        # A later length group may fail or hit the epoch deadline. Preserve
        # completed observations for ranking/ingestion without marking this
        # whole dispatch complete. The final aggregate uses the same run ID.
        _atomic_json(args.output_dir / f"exact_partial_{wave_index:03d}.json", {
            "target": args.target, "scorer_protocol": SCORER_PROTOCOL,
            "observation_id": observation_id, "n": len(all_components),
            "wave_count": len(completed), "complete": False,
            "per_nanobody_components": {"0": all_components},
            "structures": all_structures,
        })
        _atomic_json(
            manifest_path,
            {
                "observation_id": observation_id,
                "target": args.target,
                "source": str(args.sequences),
                "source_sha256": _sha256(args.sequences),
                "candidate_count": len(sequences),
                "wave_size": args.wave_size,
                "wave_partition": "first-seen-length",
                "completed_count": sum(item["count"] for item in completed),
                "completed_waves": completed,
            },
        )
        print(
            f"WAVE_{action} {wave_index}: {len(wave)} candidates -> {wave_output}",
            flush=True,
        )
        start += len(wave)

    if len(all_components) != len(sequences):
        raise SystemExit(
            f"wave merge mismatch: expected {len(sequences)}, found {len(all_components)}"
        )
    _atomic_json(
        args.output_dir / "exact_all.json",
        {
            "target": args.target,
            "scorer_protocol": SCORER_PROTOCOL,
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
