"""Backfill every raw local exact-screen observation into the mutation database.

Ranker outputs contain only a bounded leader list.  This importer reads the raw
``exact_all.json``/legacy exact payloads instead, preserving bad outcomes and
independent repeats.  Scalar graph scores are deterministic validator-shaped
replays against fixed historical fields; all ten raw metrics remain canonical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nanobody_opt.mutation_network import (
    record_exact_run,
    rebuild_nearest_edges,
    update_observation_store,
)
from nanobody_opt.simulate_round import METRICS


EPOCH_PATTERNS = (
    re.compile(r"epoch(\d+)", re.IGNORECASE),
    re.compile(r"[_/-](\d{5})(?:[_/.]|$)"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_from_path(path: Path) -> int | None:
    value = str(path)
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(value)
        if match:
            return int(match.group(1))
    return None


def _is_exact_payload(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() == ".json"
        and "exact" in name
        and not name.startswith("wave_")
    )


def discover_exact_files(roots: Iterable[Path]) -> list[Path]:
    """Prefer merged exact payloads and exclude duplicate per-wave fragments."""
    found: dict[str, Path] = {}
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*.json")
        for path in candidates:
            if not path.is_file() or not _is_exact_payload(path):
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            components = payload.get("per_nanobody_components")
            if not isinstance(components, dict) or not components:
                continue
            digest = _sha256(path)
            previous = found.get(digest)
            if previous is None or (
                path.name == "exact_all.json" and previous.name != "exact_all.json"
            ):
                found[digest] = path
    return sorted(found.values(), key=lambda path: str(path))


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_proposal_metadata(roots: Iterable[Path]) -> dict[tuple[int | None, str], dict]:
    """Index Layer-2/CSV proposal provenance by epoch and sequence."""
    indexed: dict[tuple[int | None, str], dict[str, Any]] = {}

    def retain(path: Path, records: Iterable[dict[str, Any]]) -> None:
        epoch = epoch_from_path(path)
        for raw in records:
            sequence = str(raw.get("sequence", "")).strip().upper()
            if not sequence:
                continue
            record = {
                str(key): _plain(value)
                for key, value in raw.items()
                if key != "sequence"
            }
            record["metadata_source_path"] = str(path)
            key = (epoch, sequence)
            previous = indexed.get(key, {})
            # Richer later records may add Layer-2 and graph fields without
            # replacing already captured parent/proposal identity.
            indexed[key] = {**record, **previous}

    seen_paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            try:
                if path.suffix.lower() == ".csv":
                    header = pd.read_csv(path, nrows=0)
                    if not {"sequence", "parent_sequence"}.issubset(header.columns):
                        continue
                    retain(path, pd.read_csv(path).to_dict(orient="records"))
                elif "layer2" in path.name.lower() and path.suffix.lower() == ".json":
                    payload = json.loads(path.read_text())
                    records = [
                        *payload.get("candidates", []),
                        *payload.get("rejected", []),
                    ]
                    retain(path, (item for item in records if isinstance(item, dict)))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return indexed


def _bootstrap_fields(
    archive: pd.DataFrame,
    *,
    recent: int,
    field_size: int,
    count: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    reference = archive.tail(recent).reset_index(drop=True)
    if len(reference) < field_size:
        raise ValueError("archive is smaller than the requested competitor field")
    missing = set(METRICS) - set(reference.columns)
    if missing:
        raise ValueError(f"archive lacks validator metrics: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    fields = []
    for _ in range(count):
        selected = reference.iloc[
            rng.choice(len(reference), field_size, replace=False)
        ]
        fields.append({
            metric: np.unique(pd.to_numeric(selected[metric]).to_numpy(dtype=float))
            for metric in METRICS
        })
    return fields


def validator_replay_distribution(
    metrics: dict[str, float], fields: list[dict[str, np.ndarray]]
) -> list[float]:
    scores = []
    for field in fields:
        total = 0
        for metric, direction in METRICS.items():
            value = float(metrics[metric])
            competitors = field[metric]
            better = competitors < value if direction == "min" else competitors > value
            total += 1 + int(np.count_nonzero(better))
        scores.append(float(total))
    return scores


def exact_observations(
    paths: Iterable[Path],
    *,
    target: str,
    fields: list[dict[str, np.ndarray]],
    metadata: dict[tuple[int | None, str], dict[str, Any]],
    archive_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    observations = []
    counters = {
        "files": 0,
        "component_rows": 0,
        "complete_observations": 0,
        "incomplete_observations": 0,
        "with_parent_metadata": 0,
        "with_structure": 0,
    }
    for path in paths:
        payload = json.loads(path.read_text())
        if str(payload.get("target")) != target:
            continue
        counters["files"] += 1
        file_sha256 = _sha256(path)
        run_id = str(payload.get("observation_id") or file_sha256)
        epoch = epoch_from_path(path)
        structures = payload.get("structures") or {}
        for group, by_sequence in payload.get("per_nanobody_components", {}).items():
            if not isinstance(by_sequence, dict):
                continue
            for sequence, by_target in by_sequence.items():
                counters["component_rows"] += 1
                sequence = str(sequence).strip().upper()
                metrics = by_target.get(target) if isinstance(by_target, dict) else None
                if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
                    counters["incomplete_observations"] += 1
                    continue
                values = {metric: float(metrics[metric]) for metric in METRICS}
                if not all(math.isfinite(value) for value in values.values()):
                    counters["incomplete_observations"] += 1
                    continue
                distribution = validator_replay_distribution(values, fields)
                proposal = metadata.get((epoch, sequence), metadata.get((None, sequence), {}))
                structure = (
                    structures.get(sequence, {}).get(target)
                    if isinstance(structures.get(sequence), dict)
                    else None
                )
                observation = {
                    "observation_id": f"local-exact:{run_id}:{group}:{sequence}",
                    "sequence": sequence,
                    "target": target,
                    "epoch": epoch,
                    "source": "local_exact_raw",
                    "score_kind": "local_exact_validator_replay_rank_sum",
                    "score": float(np.median(distribution)),
                    "observation_count": 1,
                    "metrics": values,
                    "score_p90": float(np.quantile(distribution, 0.9)),
                    "score_distribution": distribution,
                    "exact_source_path": str(path),
                    "exact_file_sha256": file_sha256,
                    "exact_run_id": run_id,
                    "exact_group": str(group),
                    "scorer_protocol": payload.get("scorer_protocol"),
                    "archive_sha256": archive_sha256,
                    "structure": structure,
                    **proposal,
                }
                if proposal.get("parent_sequence"):
                    counters["with_parent_metadata"] += 1
                if structure:
                    counters["with_structure"] += 1
                observations.append(observation)
                counters["complete_observations"] += 1
    return observations, counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--exact-root", action="append", required=True, type=Path)
    parser.add_argument("--metadata-root", action="append", default=[], type=Path)
    parser.add_argument("--recent", type=int, default=300)
    parser.add_argument("--field-size", type=int, default=49)
    parser.add_argument("--bootstrap-fields", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-edges", action="store_true")
    parser.add_argument("--edge-workers", type=int, default=40)
    parser.add_argument("--edge-limit", type=int, default=16)
    parser.add_argument("--maximum-edge-distance", type=int, default=6)
    args = parser.parse_args()
    if min(
        args.recent,
        args.field_size,
        args.bootstrap_fields,
        args.edge_workers,
        args.edge_limit,
        args.maximum_edge_distance,
    ) < 1:
        parser.error("replay sizes must be positive")

    paths = discover_exact_files(args.exact_root)
    archive = pd.read_csv(args.archive)
    fields = _bootstrap_fields(
        archive,
        recent=args.recent,
        field_size=args.field_size,
        count=args.bootstrap_fields,
        seed=args.random_seed,
    )
    metadata = load_proposal_metadata(args.metadata_root)
    observations, summary = exact_observations(
        paths,
        target=args.target,
        fields=fields,
        metadata=metadata,
        archive_sha256=_sha256(args.archive),
    )
    summary.update({
        "discovered_exact_files": len(paths),
        "proposal_metadata_records": len(metadata),
        "unique_observation_ids": len({
            item["observation_id"] for item in observations
        }),
        "dry_run": bool(args.dry_run),
    })
    if not args.dry_run:
        summary["database_rows_written"] = update_observation_store(
            args.database, observations
        )
        runs: dict[str, list[dict[str, Any]]] = {}
        for observation in observations:
            runs.setdefault(str(observation["exact_run_id"]), []).append(observation)
        sealed = []
        for run_id, records in sorted(runs.items()):
            first = records[0]
            sealed.append(record_exact_run(
                args.database,
                target=args.target,
                run_id=run_id,
                epoch=first.get("epoch"),
                source_path=Path(str(first["exact_source_path"])),
                file_sha256=str(first["exact_file_sha256"]),
                scorer_protocol=(
                    str(first["scorer_protocol"])
                    if first.get("scorer_protocol") else None
                ),
                observation_ids=[str(item["observation_id"]) for item in records],
            ))
        summary["exact_runs_recorded"] = len(sealed)
        summary["exact_runs_complete"] = sum(item["complete"] for item in sealed)
        if args.rebuild_edges:
            summary["graph_rebuild"] = rebuild_nearest_edges(
                args.database,
                target=args.target,
                edge_limit=args.edge_limit,
                maximum_distance=args.maximum_edge_distance,
                workers=args.edge_workers,
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
