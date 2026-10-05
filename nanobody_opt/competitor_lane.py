"""Broad single-edit parent lane, wired into the producer.

Generates single-substitution candidates at the rival's framework positions,
across the UNFILTERED public archive (any rank, any miner) -- the strategy the
elite competitor wins with. Replaces/out-weights the elite-centric
`select_elite_parents` bank as a parent source.

The position set is the measured concentration (reconstruct_results.json, the
5GhREcsK parent reconstruction): walk-forward recall 84% at the top-10 and 94%
at the top-15, so it is stable and safe to pin rather than re-learn each epoch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nanobody_opt import gates
from nanobody_opt.competitor_position_lane import competitor_position_candidates

# 1-indexed hot positions, from the measured edit-position distribution (see
# analysis/nanobody-gap/parent-selection-20260929/reconstruct_results.json).
HOT_POSITIONS = [1, 3, 114, 122, 121, 124, 116, 123, 31, 100, 126, 111, 73, 115, 127]


def _load_pool(archive: Path) -> list[str]:
    """Every public sequence in the archive, deduped, order preserved."""
    import pandas as pd

    frame = pd.read_csv(archive)
    column = "sequence"
    if column not in frame:
        raise SystemExit(f"archive lacks a {column!r} column; got {list(frame.columns)}")
    seen: set[str] = set()
    pool: list[str] = []
    for value in frame[column]:
        seq = str(value).strip().upper()
        if seq and seq not in seen:
            seen.add(seq)
            pool.append(seq)
    if not pool:
        raise SystemExit("archive contains no sequences")
    return pool


def _eligible_parents(pool: list[str], top50: list[str], min_distance: int) -> list[str]:
    """Parents whose min Hamming distance to the top-50 is >= min_distance, so a
    single-edit child can land >=7 edits and clear the whole-seq 0.95 dedup."""
    import numpy as np

    top = np.array([[ord(ch) for ch in s] for s in top50], dtype=np.uint8)
    if top.size == 0:
        return list(pool)
    length = top.shape[1]
    eligible: list[str] = []
    for s in pool:
        seq = np.frombuffer(s.encode(), dtype=np.uint8)
        if seq.shape[0] != length:
            continue
        if int((top != seq).sum(axis=1).min()) >= min_distance:
            eligible.append(s)
    return eligible


def generate(archive: Path, positions: list[int], limit: int | None,
             pool_limit: int = 2000, top50: list[str] | None = None,
             min_parent_distance: int = 6) -> dict[str, Any]:
    pool = _load_pool(archive)
    # Nanobodies vary in length; the hot positions are measured against the
    # modal-length sequence, so restrict the pool to that length (a single
    # allowed-residue table must match every parent).
    from collections import Counter

    modal_length = Counter(len(s) for s in pool).most_common(1)[0][0]
    pool = [s for s in pool if len(s) == modal_length]
    # Only keep parents far enough from the top-50 that a single-edit can clear
    # the whole-seq 0.95 dedup: a parent at distance D yields single-edits at
    # D-1..D+1, and 0.95 identity == <=6 edits is a dup, so D>=6 is the eligible
    # band (the exact survivor count is decided later by pre_gpu_dedup).
    if top50:
        pool = _eligible_parents(pool, top50, min_parent_distance)
    # Shuffle so the pool_limit cap and the later candidate cap sample a DIVERSE
    # set of parents, not a single tight lineage of the first few.
    import random

    random.shuffle(pool)
    # Sample down so the single-edit fan-out (pool x positions x residues)
    # stays bounded.
    if pool_limit and len(pool) > pool_limit:
        pool = pool[:pool_limit]
    length = modal_length
    # Any standard residue is a valid single substitution at a hot position;
    # the downstream layer2/dedup gates do the actual quality filtering.
    allowed = [sorted(gates.ALLOWED_AAS) for _ in range(length)]
    candidates = competitor_position_candidates(
        pool, allowed=allowed, positions=positions
    )
    items = [
        {"sequence": seq, "parent": meta["parent"], "position": meta["position"],
         "source": "competitor_position_lane"}
        for seq, meta in candidates.items()
    ]
    if limit:
        items = items[:limit]
    return {
        "source": "competitor_position_lane",
        "positions": positions,
        "pool_size": len(pool),
        "sequences": [item["sequence"] for item in items],
        "candidates": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positions", default=",".join(map(str, HOT_POSITIONS)),
                        help="comma-separated 1-indexed positions (default: measured hot set)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of emitted candidates (default: unlimited)")
    parser.add_argument("--pool-limit", type=int, default=2000,
                        help="sample the archive down to its most recent N sequences (default 2000)")
    parser.add_argument("--top50", type=Path, default=None,
                        help="build_top50_index.json; when given, only parents >= min-parent-distance edits from the top-50 are kept")
    parser.add_argument("--min-parent-distance", type=int, default=6,
                        help="minimum Hamming distance from the top-50 a parent must have (default 6)")
    args = parser.parse_args()
    positions = [int(p) for p in args.positions.split(",") if p.strip()]
    top50 = None
    if args.top50 is not None:
        top50 = [str(s).strip().upper() for s in
                 (json.loads(args.top50.read_text()).get("sequences") or [])]
    payload = generate(args.archive, positions, args.limit, pool_limit=args.pool_limit,
                       top50=top50, min_parent_distance=args.min_parent_distance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"COMPETITOR_LANE_READY pool={payload['pool_size']} "
        f"candidates={len(payload['candidates'])} output={args.output}"
    )


if __name__ == "__main__":
    main()
