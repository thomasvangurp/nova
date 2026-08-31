#!/usr/bin/env python3
"""Score a proxy shortlist with the validator's pinned open-source Boltz path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compound_miner.targets import P40261_SEQUENCE


def record_id(smiles: str, seed: int = 68) -> int:
    digest = hashlib.sha256(smiles.encode()).digest()
    return (int.from_bytes(digest[:8], "little") ^ seed) % (2**31 - 1)


def load_records(archive: Path, reaction: int, limit: int) -> list[dict[str, Any]]:
    """Build a fixed acquisition portfolio instead of trusting one weak proxy rank.

    Half of each reaction batch exploits predicted mean, one quarter samples high
    epistemic uncertainty, and the remainder is spread deterministically over the
    full proxy-ranked archive.  The fixed ordering also keeps host timing from
    changing which molecules receive an exact score.
    """
    with sqlite3.connect(archive) as connection:
        reactions = range(1, 6) if reaction == 0 else (reaction,)
        rows = []
        for value in reactions:
            pool = connection.execute(
                """SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds,proxy_mean,proxy_std
                   FROM candidates WHERE reaction=?
                   ORDER BY proxy_mean DESC, name ASC""",
                (value,),
            ).fetchall()
            if not pool:
                continue
            chosen: list[tuple[Any, ...]] = []
            names: set[str] = set()

            def add(items: list[tuple[Any, ...]], quota: int) -> None:
                for item in items:
                    if len(chosen) >= quota or len(chosen) >= limit:
                        break
                    if item[0] not in names:
                        chosen.append(item)
                        names.add(item[0])

            exploit_end = math.ceil(limit * 0.50)
            add(pool, exploit_end)
            uncertainty_end = exploit_end + math.ceil(limit * 0.25)
            add(sorted(pool, key=lambda item: (-float(item[8]), item[0])), uncertainty_end)
            remaining = [item for item in pool if item[0] not in names]
            if remaining:
                needed = limit - len(chosen)
                # Even ranks are stable, broad, and reproducible across machines.
                positions = [int(index * len(remaining) / max(1, needed)) for index in range(needed)]
                add([remaining[min(position, len(remaining) - 1)] for position in positions], limit)
            rows.extend(chosen)
    columns = ("name", "smiles", "reaction", "components", "inchikey", "heavy_atoms", "rotatable_bonds", "proxy_mean", "proxy_std")
    return [dict(zip(columns, row)) for row in rows]


def _load_metrics(prediction_directory: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in prediction_directory.iterdir():
        if path.name.startswith(("affinity", "confidence")) and path.suffix == ".json":
            result.update(json.loads(path.read_text()))
    return result


def score_worker(arguments: tuple[int, int, list[dict[str, Any]], str, str, dict[str, Any]]) -> list[dict[str, Any]]:
    shard, gpu, records, scratch_root, msa_path, inference = arguments
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
        recycling_steps=inference["recycling_steps"],
        sampling_steps=inference["sampling_steps"],
        diffusion_samples=inference["diffusion_samples"],
        sampling_steps_affinity=inference["sampling_steps_affinity"],
        diffusion_samples_affinity=inference["diffusion_samples_affinity"],
        skip_affinity_structure=inference["skip_affinity_structure"],
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
                "scorer": (
                    "boltz-2.2.1-r"
                    f"{inference['recycling_steps']}-s{inference['sampling_steps']}"
                    f"-d{inference['diffusion_samples']}-as{inference['sampling_steps_affinity']}"
                    f"-ad{inference['diffusion_samples_affinity']}"
                    f"-skip{int(inference['skip_affinity_structure'])}"
                ),
                "elapsed_seconds": elapsed / max(1, len(records)),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reaction", type=int, choices=range(0, 6), required=True, help="1-5, or 0 for a balanced batch from all reactions")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=1)
    parser.add_argument("--msa", type=Path, default=Path("data/msa_files/P40261.a3m"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-json", type=Path, help="Reuse the exact molecule set from an earlier score file")
    parser.add_argument("--recycling-steps", type=int, default=3)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--diffusion-samples", type=int, default=1)
    parser.add_argument("--sampling-steps-affinity", type=int, default=100)
    parser.add_argument("--diffusion-samples-affinity", type=int, default=3)
    parser.add_argument("--skip-affinity-structure", action="store_true")
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args()
    if args.records_json:
        source_records = json.loads(args.records_json.read_text())["scores"]
        reactions = range(1, 6) if args.reaction == 0 else (args.reaction,)
        records = []
        for reaction in reactions:
            reaction_records = [item for item in source_records if int(item["reaction"]) == reaction]
            reaction_records.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
            records.extend(reaction_records[: args.limit])
        for record in records:
            if not isinstance(record["components"], str):
                record["components"] = json.dumps(record["components"])
    else:
        records = load_records(args.archive, args.reaction, args.limit)
    if not records:
        raise SystemExit("archive shortlist is empty")
    gpus = [int(value) for value in args.gpus.split(",")]
    worker_gpus = [gpu for gpu in gpus for _ in range(args.shards_per_gpu)]
    partitions = [records[index:: len(worker_gpus)] for index in range(len(worker_gpus))]
    scratch = tempfile.mkdtemp(prefix="nova-boltz-")
    inference = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "sampling_steps_affinity": args.sampling_steps_affinity,
        "diffusion_samples_affinity": args.diffusion_samples_affinity,
        "skip_affinity_structure": args.skip_affinity_structure,
    }
    started = time.monotonic()
    try:
        context = mp.get_context("spawn")
        with context.Pool(len(worker_gpus)) as pool:
            chunks = pool.map(
                score_worker,
                [(index, gpu, partition, scratch, str(args.msa.resolve()), inference) for index, (gpu, partition) in enumerate(zip(worker_gpus, partitions)) if partition],
            )
        results = [item for chunk in chunks for item in chunk]
        payload = {
            "reaction": args.reaction,
            "count": len(results),
            "wall_seconds": time.monotonic() - started,
            "gpus": gpus,
            "shards_per_gpu": args.shards_per_gpu,
            "inference": inference,
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
