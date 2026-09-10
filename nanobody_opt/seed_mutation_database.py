"""Seed and refresh the persistent scored VHH mutation database.

The durable store is SQLite/WAL.  Public validator fields are fetched and
normalized concurrently, then one transaction upserts their observations.
The optional nearest-edge pass is also parallel and never changes observation
history, so an interrupted rebuild can simply be restarted.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nanobody_opt.mutation_network import (
    rebuild_nearest_edges,
    update_observation_store,
)
from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT
from nanobody_opt.select_elite_parents import _epoch_candidates, _fetch
from nanobody_opt.simulate_round import METRICS


def _observation(
    *,
    epoch: int,
    sequence: str,
    score: float,
    metrics: dict[str, float],
    target: str,
    api_root: str,
) -> dict[str, Any]:
    return {
        "observation_id": f"public-validator:epoch{epoch}:{sequence}",
        "sequence": sequence,
        "score": score,
        "score_kind": "validator_final_rank_sum",
        "source": "public_validator_api",
        "epoch": epoch,
        "target": target,
        "observation_count": 1,
        "metrics": metrics,
        "api_root": api_root.rstrip("/"),
    }


def _snapshot_group(
    rows: list[dict[str, Any]],
    *,
    target: str,
    api_root: str,
) -> list[dict[str, Any]]:
    observations = []
    for row in rows:
        try:
            epoch = int(row["epoch"])
            sequence = str(row["sequence"]).strip().upper()
            score = float(row["final_score"])
            metrics = {name: float(row[name]) for name in METRICS}
        except (KeyError, TypeError, ValueError):
            continue
        observations.append(_observation(
            epoch=epoch,
            sequence=sequence,
            score=score,
            metrics=metrics,
            target=target,
            api_root=api_root,
        ))
    return observations


def snapshot_observations(
    path: Path,
    *,
    target: str,
    api_root: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Convert a saved public-field snapshot in parallel by epoch."""
    frame = pd.read_csv(path)
    required = {"epoch", "sequence", "final_score", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"snapshot is missing required columns: {missing}")
    groups = [
        group.to_dict("records")
        for _, group in frame.groupby("epoch", sort=False)
    ]
    observations: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _snapshot_group,
                group,
                target=target,
                api_root=api_root,
            )
            for group in groups
        ]
        for future in as_completed(futures):
            observations.extend(future.result())
    return observations


def fetch_observations(
    epochs: Iterable[int],
    *,
    target: str,
    api_root: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Fetch exact public validator observations with bounded concurrency."""
    observations: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_fetch, api_root, int(epoch)) for epoch in epochs
        ]
        for future in as_completed(futures):
            epoch, payload = future.result()
            for item in _epoch_candidates(payload, target):
                observations.append(_observation(
                    epoch=epoch,
                    sequence=str(item["sequence"]),
                    score=float(item["validator_final_score"]),
                    metrics={
                        name: float(item["metrics"][name]) for name in METRICS
                    },
                    target=target,
                    api_root=api_root,
                ))
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--api-first-epoch", type=int)
    parser.add_argument("--before-epoch", type=int)
    parser.add_argument("--api-root", default=LEADERBOARD_API_ROOT)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--rebuild-edges", action="store_true")
    parser.add_argument("--edge-limit", type=int, default=16)
    parser.add_argument("--maximum-edge-distance", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or args.edge_limit < 1 or args.maximum_edge_distance < 1:
        parser.error("worker and edge limits must be positive")
    if (args.api_first_epoch is None) != (args.before_epoch is None):
        parser.error("--api-first-epoch and --before-epoch must be used together")

    observations: list[dict[str, Any]] = []
    if args.snapshot is not None:
        observations.extend(snapshot_observations(
            args.snapshot,
            target=args.target,
            api_root=args.api_root,
            workers=args.workers,
        ))
        print(
            f"MUTATION_DB_SNAPSHOT_READY observations={len(observations)}",
            flush=True,
        )
    if args.api_first_epoch is not None:
        fetched = fetch_observations(
            range(args.api_first_epoch, args.before_epoch),
            target=args.target,
            api_root=args.api_root,
            workers=args.workers,
        )
        observations.extend(fetched)
        print(
            f"MUTATION_DB_API_READY observations={len(fetched)}",
            flush=True,
        )
    if not observations:
        raise SystemExit("no snapshot or API observations were available")
    # Snapshot and API records share a stable observation id, so API refreshes
    # replace cached rows instead of double-counting them.
    deduplicated = {
        str(item["observation_id"]): item for item in observations
    }
    written = update_observation_store(args.database, deduplicated.values())
    print(
        f"MUTATION_DB_OBSERVATIONS_COMMITTED count={written} "
        f"database={args.database}",
        flush=True,
    )
    if args.rebuild_edges:
        result = rebuild_nearest_edges(
            args.database,
            target=args.target,
            edge_limit=args.edge_limit,
            maximum_distance=args.maximum_edge_distance,
            workers=args.workers,
        )
        print("MUTATION_DB_GRAPH_READY " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
