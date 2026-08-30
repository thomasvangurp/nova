#!/usr/bin/env python3
"""Score a proxy shortlist with the validator's pinned open-source Boltz path."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from compound_miner.targets import P40261_SEQUENCE


def record_id(smiles: str, seed: int = 68) -> int:
    digest = hashlib.sha256(smiles.encode()).digest()
    return (int.from_bytes(digest[:8], "little") ^ seed) % (2**31 - 1)


def load_records(archive: Path, reaction: int, limit: int) -> list[dict[str, Any]]:
    with sqlite3.connect(archive) as connection:
        rows = connection.execute(
            """SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds,proxy_mean,proxy_std
               FROM candidates WHERE reaction=? ORDER BY proxy_mean + 0.35 * proxy_std DESC LIMIT ?""",
            (reaction, limit),
        ).fetchall()
    columns = ("name", "smiles", "reaction", "components", "inchikey", "heavy_atoms", "rotatable_bonds", "proxy_mean", "proxy_std")
    return [dict(zip(columns, row)) for row in rows]


def _load_metrics(prediction_directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in prediction_directory.iterdir():
        if path.name.startswith(("affinity", "confidence")) and path.suffix == ".json":
            result.update(json.loads(path.read_text()))
    return result


def score_worker(arguments: tuple[int, int, list[dict[str, Any]], str, str]) -> list[dict[str, Any]]:
    shard, gpu, records, scratch_root, msa_path = arguments
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from boltz.main import predict

    scratch = Path(scratch_root) / f"shard{shard}"
    inputs = scratch / "inputs"
    outputs = scratch / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for record in records:
        identifier = record_id(record["smiles"])
        payload = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": None, "msa": msa_path}},
                {"ligand": {"id": "B", "smiles": record["smiles"]}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        payload["sequences"][0]["protein"]["sequence"] = P40261_SEQUENCE
        (inputs / f"{identifier}_P40261.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    started = time.monotonic()
    predict(
        data=str(inputs),
        out_dir=str(outputs),
        recycling_steps=3,
        sampling_steps=100,
        diffusion_samples=1,
        sampling_steps_affinity=100,
        diffusion_samples_affinity=3,
        output_format="mmcif",
        seed=68,
        affinity_mw_correction=True,
        override=False,
        num_workers=0,
    )
    elapsed = time.monotonic() - started
    results = []
    for record in records:
        identifier = record_id(record["smiles"])
        metrics = _load_metrics(outputs / "boltz_results_inputs" / "predictions" / f"{identifier}_P40261")
        probability = float(metrics["affinity_probability_binary"])
        prediction = float(metrics["affinity_pred_value"])
        results.append(
            {
                **record,
                "components": json.loads(record["components"]),
                "affinity_probability_binary": probability,
                "affinity_pred_value": prediction,
                "score": (probability - prediction) / int(record["heavy_atoms"]),
                "scorer": "boltz-2.2.1-validator",
                "elapsed_seconds": elapsed / max(1, len(records)),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reaction", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=1)
    parser.add_argument("--msa", type=Path, default=Path("data/msa_files/P40261.a3m"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()
    records = load_records(args.archive, args.reaction, args.limit)
    if not records:
        raise SystemExit("archive shortlist is empty")
    gpus = [int(value) for value in args.gpus.split(",")]
    worker_gpus = [gpu for gpu in gpus for _ in range(args.shards_per_gpu)]
    partitions = [records[index:: len(worker_gpus)] for index in range(len(worker_gpus))]
    scratch = tempfile.mkdtemp(prefix="nova-boltz-")
    started = time.monotonic()
    try:
        context = mp.get_context("spawn")
        with context.Pool(len(worker_gpus)) as pool:
            chunks = pool.map(
                score_worker,
                [(index, gpu, partition, scratch, str(args.msa.resolve())) for index, (gpu, partition) in enumerate(zip(worker_gpus, partitions)) if partition],
            )
        results = [item for chunk in chunks for item in chunk]
        payload = {
            "reaction": args.reaction,
            "count": len(results),
            "wall_seconds": time.monotonic() - started,
            "gpus": gpus,
            "shards_per_gpu": args.shards_per_gpu,
            "scores": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(args.output)
        print(json.dumps({key: value for key, value in payload.items() if key != "scores"}))
    finally:
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
