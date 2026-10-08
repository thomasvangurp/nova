"""Harvest our elite children bank: submit the top directly + mutate a few.

The replay DB already holds ~105k children whose exact validator replay
rank-sum we measured, and the top-2000 are all admissible against the HF-archive
dedup (and better than any rival's submissions). So each epoch we walk down the
bank: emit the top-N children DIRECTLY (no mutation -- they are already the best
sequences we have), and single-edit the next-M as cheap local exploration.
The published ledger ensures each child is used once.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from nanobody_opt import db, gates
from nanobody_opt.competitor_lane import POSITION_RESIDUE_ORDER
from nanobody_opt.competitor_position_lane import (
    competitor_position_candidates,
)

HOT_POSITIONS = [1, 3, 114, 122, 121, 124, 116, 123, 31, 100, 126, 111, 73, 127]


def top_children(db_path: str, ledger_path: str, k: int) -> list[tuple[str, float]]:
    """Top-``k`` children by validator replay rank-sum, skipping published ones.

    Prefers ``local_exact_validator_replay_rank_sum`` over the fast replay when
    both exist for the same sequence. Lower rank-sum = better.
    """
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT n.sequence, o.score, o.score_kind
        FROM observations o JOIN nodes n ON n.id = o.node_id
        WHERE n.target = 'P20809'
          AND o.score_kind IN ('local_exact_validator_replay_rank_sum',
                               'local_validator_replay_rank_sum')
        GROUP BY n.sequence
        ORDER BY o.score ASC
        """
    ).fetchall()
    conn.close()

    published: set[str] = set()
    ledger = Path(ledger_path)
    if ledger.is_file():
        published = {
            line.strip().upper() for line in ledger.read_text().splitlines()
            if line.strip()
        }

    best: dict[str, float] = {}
    for seq, score, kind in rows:
        seq = str(seq).strip().upper()
        if seq in published:
            continue
        if seq not in best or kind == "local_exact_validator_replay_rank_sum":
            best[seq] = float(score)
    return sorted(best.items(), key=lambda kv: kv[1])[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--direct", type=int, default=7)
    parser.add_argument("--mutate", type=int, default=6)
    parser.add_argument("--positions", default=",".join(map(str, HOT_POSITIONS)))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    positions = [int(x) for x in args.positions.split(",") if x.strip()]
    kids = top_children(str(args.db), str(args.ledger), args.direct + args.mutate)
    if not kids:
        raise SystemExit("no unpublished children found in the bank")

    direct = kids[: args.direct]
    mutators = kids[args.direct : args.direct + args.mutate]

    modal = len(direct[0][0])
    allowed: list[list[str]] = []
    for i in range(modal):
        order = POSITION_RESIDUE_ORDER.get(i + 1)
        if order:
            rest = [r for r in sorted(gates.ALLOWED_AAS) if r not in order]
            allowed.append(order + rest)
        else:
            allowed.append(sorted(gates.ALLOWED_AAS))

    candidates: list[dict[str, Any]] = []
    for seq, _score in direct:
        candidates.append({
            "sequence": seq, "parent": seq, "position": 0,
            "source": "children_direct",
        })
    # One single-edit per mutator child (6 total): the best residue at the first
    # hot position, per the exhaustive-optima POSITION_RESIDUE_ORDER. Keeps each
    # UID a distinct ticket rather than ~266 edits per child.
    first_pos = positions[0]
    first_residue = allowed[first_pos - 1][0]
    for seq, _score in mutators:
        cand = seq[: first_pos - 1] + first_residue + seq[first_pos:]
        candidates.append({
            "sequence": cand, "parent": seq, "position": first_pos,
            "source": "children_mutate",
        })

    payload = {
        "source": "children_lane",
        "direct": [s for s, _ in direct],
        "mutators": [s for s, _ in mutators],
        "sequences": [c["sequence"] for c in candidates],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"CHILDREN_LANE_READY direct={len(direct)} mutate={len(mutators)} "
        f"candidates={len(candidates)} output={args.output}"
    )


if __name__ == "__main__":
    main()
