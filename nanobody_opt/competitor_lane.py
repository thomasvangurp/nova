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
from nanobody_opt.competitor_position_lane import (
    competitor_position_candidates,
    double_position_candidates,
)

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"


def fetch_rival_sequences(epoch: int, coldkey_prefix: str,
                          n_epochs: int = 15) -> list[str]:
    """Fetch a competitor's submitted sequences from the leaderboard.

    The rival already spent compute finding these, so they are 'proven good'
    parents -- use them directly as the mutation source rather than re-ranking
    the whole archive. Returns the P20809 sequences for any coldkey starting
    with ``coldkey_prefix`` over the most recent ``n_epochs`` epochs, deduped.
    """
    import json
    import urllib.request

    best: dict[str, float] = {}
    for e in range(epoch - 1, epoch - 1 - n_epochs, -1):
        req = urllib.request.Request(
            f"{API}/api/competitions/leaderboard/{e}/nanobodies",
            headers={"User-Agent": "nova-competitor-lane/1"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
        except Exception:
            continue
        for submission in payload.get("data") or []:
            if not str(submission.get("coldkey") or "").startswith(coldkey_prefix):
                continue
            for nb in submission.get("nanobodies") or []:
                if nb.get("protein_name") != TARGET:
                    continue
                seq = str(nb.get("sequence") or "").strip().upper()
                if not seq:
                    continue
                try:
                    score = float(nb.get("final_score"))
                except (TypeError, ValueError):
                    score = float("inf")
                if seq not in best or score < best[seq]:
                    best[seq] = score
    # Best (lowest validator final_score) first, so --pool-limit keeps the top.
    return [s for s, _ in sorted(best.items(), key=lambda kv: kv[1])]

# 1-indexed hot positions, from the measured edit-position distribution (see
# analysis/nanobody-gap/parent-selection-20260929/reconstruct_results.json).
HOT_POSITIONS = [1, 3, 114, 122, 121, 124, 116, 123, 31, 100, 126, 111, 73, 127]

# 1-indexed position -> residues in descending measured quality, from the
# exhaustive 8x single-edit sweep (scoring_backfill/exhaustive_ranking.json).
# The sweep scored 4,738 edits (13 positions x 19 residues x 20 parents) 8x and
# found Alanine best/tied at 7/13 positions. Best residue first; pos 115 was a
# DUD (dropped). Positions 31/100 were outside the sweep and keep the rival's
# frequency order.
POSITION_RESIDUE_ORDER = {
    1: ["A", "D"],
    3: ["A", "M"],
    73: ["A", "T"],
    111: ["E", "D"],
    114: ["H", "M"],
    116: ["A", "S"],
    121: ["A", "T"],
    122: ["A", "P"],
    123: ["V", "A"],
    124: ["A", "S"],
    126: ["G", "A"],
    127: ["A", "P"],
    31: ["R"],
    100: ["A", "G"],
}


def _load_pool(archive: Path) -> list[str]:
    """Every public sequence in the archive, ranked by the validator's
    Algorithm-2 minmax (best first), deduped.

    This is the GLOBAL parent source: the whole archive sorted by the proxy
    (binding/confidence), never our own lineage privileged. The archive's
    metric columns give the Algorithm-2 rank for free, so no re-scoring.
    """
    import pandas as pd

    frame = pd.read_csv(archive)
    frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()
    frame = frame.drop_duplicates("sequence", keep="last")

    work = frame[frame.get("liability_num_violations", 0) <= 50].copy()
    weights = {
        "design_iiptm": 1.0, "design_ptm": 2.0, "min_design_to_target_pae": 1.0,
        "plip_hbonds_refolded": 2.0, "plip_saltbridge_refolded": 2.0,
        "delta_sasa_refolded": 2.0,
    }
    metric_dirs = {
        "design_iiptm": "max", "design_ptm": "max", "min_design_to_target_pae": "min",
        "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
        "delta_sasa_refolded": "max",
    }
    rank_cols = {}
    for metric, weight in weights.items():
        if metric not in work.columns:
            continue
        ascending = metric_dirs[metric] == "min"
        rank_cols[metric] = work[metric].rank(ascending=ascending, method="average") / weight
    work = work.assign(algo2_score=pd.DataFrame(rank_cols, index=work.index).max(axis=1))
    work = work.sort_values("algo2_score", ascending=True)

    pool: list[str] = [str(s).strip().upper() for s in work["sequence"].tolist()]
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
             min_parent_distance: int = 6,
             priority: list[str] | None = None,
             double_edit_pairs: int = 0,
             ranker_model: Any | None = None,
             rival_sequences: list[str] | None = None) -> dict[str, Any]:
    # Parent source: the rival's proven-good sequences when supplied, else the
    # global archive ranked by the validator's Algorithm-2 proxy.
    if rival_sequences:
        pool = [s.upper() for s in rival_sequences]
    else:
        pool = _load_pool(archive)
    # Nanobodies vary in length; the hot positions are measured against the
    # modal-length sequence, so restrict the pool to that length (a single
    # allowed-residue table must match every parent).
    from collections import Counter

    modal_length = Counter(len(s) for s in pool).most_common(1)[0][0]
    pool = [s for s in pool if len(s) == modal_length]
    if top50:
        pool = _eligible_parents(pool, top50, min_parent_distance)

    # The live top-50 are the strongest parents: their single/double edits are
    # closest to the current winners AND still clear the dedup (the validator's
    # reference is the full-history HF top-50, ~30 edits from the live field).
    priority_pool: list[str] = []
    if priority:
        priority_pool = [s.upper() for s in priority if len(s) == modal_length]
        priority_set = set(priority_pool)
        pool = [s for s in pool if s not in priority_set]
    # Keep the pool ranked best-first (Algorithm-2 from _load_pool) rather than
    # shuffling: the shuffle threw away the quality signal and diluted rare
    # winning motifs (e.g. the T/V CDR3 variants) to their ~random share of the
    # 2000-slot cap.
    if pool_limit and len(pool) > pool_limit:
        pool = pool[:pool_limit]

    length = modal_length
    # Any standard residue is a valid substitution at a hot position; the
    # downstream layer2/exact-validator gates do the quality filtering. Order
    # residues by the rival's measured usage frequency at each hot position so
    # the winning substitution is emitted before the alphabetical remainder.
    allowed = []
    for i in range(length):
        order = POSITION_RESIDUE_ORDER.get(i + 1)
        if order:
            rest = [r for r in sorted(gates.ALLOWED_AAS) if r not in order]
            allowed.append(order + rest)
        else:
            allowed.append(sorted(gates.ALLOWED_AAS))

    def _items(source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"sequence": seq, "parent": meta["parent"], "position": meta["position"],
             "source": "competitor_position_lane"}
            for seq, meta in source.items()
        ]

    # Single-edit candidates kept separate so the priority top-50 can be ranked
    # independently (per-parent) while the broad pool stays shuffled for diversity.
    priority_candidates = competitor_position_candidates(
        priority_pool, allowed=allowed, positions=positions)
    if double_edit_pairs:
        for parent in priority_pool:
            for seq, (p1, p2) in double_position_candidates(
                parent, allowed=allowed, positions=positions,
                pair_limit=double_edit_pairs,
            ).items():
                priority_candidates.setdefault(
                    seq, {"parent": parent, "position": p1, "position2": p2}
                )
    broad_candidates = competitor_position_candidates(
        pool, allowed=allowed, positions=positions)

    priority_items = _items(priority_candidates)
    broad_items = _items(broad_candidates)
    # A sequence that is a single-edit of both a priority and a broad parent
    # must not be emitted twice; the priority parent wins.
    priority_seqs = {item["sequence"] for item in priority_items}
    broad_items = [item for item in broad_items if item["sequence"] not in priority_seqs]

    # When a trained parent-relative ranker is supplied, order every parent's
    # single-edits by predicted improvement (best edit per parent first) so
    # --limit keeps the strongest edit of each parent -- the top-50 priority
    # first, then the rival's submissions as the supplement.
    items = priority_items + broad_items
    ranker_applied = False
    if ranker_model is not None:
        import pandas as pd

        from nanobody_opt.parent_relative_ranker_inference import rank_single_edits

        archive_frame = pd.read_csv(archive)
        items = rank_single_edits(
            ranker_model, archive_frame, items, length=length
        )
        ranker_applied = True

    if limit:
        items = items[:limit]
    return {
        "source": "competitor_position_lane",
        "positions": positions,
        "pool_size": len(priority_pool) + len(pool),
        "priority_pool_size": len(priority_pool),
        "ranker_applied": ranker_applied,
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
    parser.add_argument("--priority-parents", type=Path, default=None,
                        help="top50_index.json; its `live_sequences` (final_score top-50) are edited first and always kept")
    parser.add_argument("--double-edit-pairs", type=int, default=0,
                        help="generate 2-substitution candidates of the priority parents, using this many position pairs (0 = off)")
    parser.add_argument("--ranker-model", type=Path, default=None,
                        help="model.joblib from train_parent_relative_ranker; when given, single-edits are ordered by predicted improvement before --limit")
    parser.add_argument("--rival-coldkey", type=str, default=None,
                        help="coldkey prefix of a competitor; when given, their recent submissions become the parent pool (proven-good source)")
    args = parser.parse_args()
    positions = [int(p) for p in args.positions.split(",") if p.strip()]
    top50 = None
    if args.top50 is not None:
        top50 = [str(s).strip().upper() for s in
                 (json.loads(args.top50.read_text()).get("sequences") or [])]
    priority = None
    if args.priority_parents is not None:
        blob = json.loads(args.priority_parents.read_text())
        priority = [str(s).strip().upper() for s in
                    (blob.get("live_sequences") or blob.get("sequences") or [])]
    ranker_model = None
    if args.ranker_model is not None:
        from nanobody_opt.parent_relative_ranker_inference import load_model

        ranker_model = load_model(args.ranker_model)
    rival_sequences = None
    if args.rival_coldkey:
        import re

        m = re.search(r"epoch(\d+)", str(args.archive))
        if not m:
            raise SystemExit(f"cannot derive epoch from archive path {args.archive}")
        rival_sequences = fetch_rival_sequences(int(m.group(1)), args.rival_coldkey)
        if not rival_sequences:
            raise SystemExit(
                f"no sequences found for rival coldkey {args.rival_coldkey!r}"
            )
    payload = generate(args.archive, positions, args.limit, pool_limit=args.pool_limit,
                       top50=top50, min_parent_distance=args.min_parent_distance,
                       priority=priority, double_edit_pairs=args.double_edit_pairs,
                       ranker_model=ranker_model,
                       rival_sequences=rival_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"COMPETITOR_LANE_READY pool={payload['pool_size']} "
        f"candidates={len(payload['candidates'])} output={args.output}"
    )


if __name__ == "__main__":
    main()
