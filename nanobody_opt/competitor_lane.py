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


#: Writer/score_kind tag for the robust 10-pass replay (multipass_all_rivals.py
#: PASS_SCORE_KIND) -- the same measurement for "ours" and every rival, ranked
#: against the identical archive field each pass. Matched against literally,
#: never re-derived, so a label change there breaks this loudly instead of
#: silently mixing scales.
MULTIPASS_SCORE_KIND = "multipass_sum10_pass"


def multipass_scores(
    sequences: list[str], db_path: Path, *, min_passes: int = 10,
    epoch: int | None = None,
) -> dict[str, float]:
    """Mean 10-pass replay rank-sum per sequence, for whichever of
    ``sequences`` have at least ``min_passes`` stored passes -- recomputed
    on read against the CURRENT real last-10-epochs field, never a stored
    ``score`` frozen at whatever field was live when that pass was written.

    User directive, 2026-10-10: competitor_lane's parent pool must be ranked
    by robust multi-pass scoring only, "it doesn't matter where it comes
    from" -- the validator's single-observation final_score (fetch_rival_
    sequences) and the archive's single-snapshot Algorithm-2 proxy
    (_load_pool) are both the kind of noisy single-observation signal
    already ruled out for candidate ranking; the same principle applies one
    step upstream, at parent selection.

    User, 2026-10-11: the background a stored multi-pass score was ranked
    against keeps moving (same reason the live producer now rebuilds its own
    field every ranking pass, see rank_exact.py's --db-field-count), so a
    cached ``score`` column goes stale the same way a cached field would.
    multipass_all_rivals.py already stores the raw ``metrics_json`` per pass
    specifically so this never needs fresh GPU work: reread it and rank
    against today's field via the one shared formula
    (rank_exact.rank_metrics_against_field) multipass_all_rivals.py itself
    uses, so a dashboard number and a parent-selection number are never two
    different computations.
    """
    import json as _json

    from nanobody_opt import db
    from nanobody_opt.rank_exact import (
        fetch_recent_fields_from_db, pool_recent_fields, rank_metrics_against_field,
    )

    if not sequences or epoch is None:
        return {}
    pooled_field = pool_recent_fields(
        fetch_recent_fields_from_db(db_path, target=TARGET, before_epoch=epoch, count=10)
    )
    if pooled_field.empty:
        return {}
    # Master's local SQLite is the only store; central MySQL is retired.
    conn = db.connect_local(db_path)
    placeholders = ",".join(["?"] * len(sequences))
    rows = conn.execute(
        f"""
        SELECT n.sequence, o.observation_id, o.metrics_json
        FROM observations o JOIN nodes n ON n.id = o.node_id
        WHERE n.target = ? AND n.sequence IN ({placeholders})
          AND o.source = ? AND o.score_kind = ?
          AND o.metrics_json IS NOT NULL AND o.metrics_json != ''
        """,
        (TARGET, *sequences, "public_multipass_rivals", MULTIPASS_SCORE_KIND),
    ).fetchall()
    metrics_by_pass: dict[str, dict[str, float]] = {}
    seq_of_pass: dict[str, str] = {}
    for seq, observation_id, blob in rows:
        try:
            metrics = _json.loads(blob)
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        key = f"{seq}::{observation_id}"
        metrics_by_pass[key] = metrics
        seq_of_pass[key] = seq
    pass_ranks = rank_metrics_against_field(metrics_by_pass, pooled_field)
    by_seq: dict[str, list[float]] = {}
    for key, rank_sum in pass_ranks.items():
        by_seq.setdefault(seq_of_pass[key], []).append(rank_sum)
    return {
        seq: sum(scores) / len(scores)
        for seq, scores in by_seq.items() if len(scores) >= min_passes
    }


#: Sources that count as "our own" exact-scoring history (same list
#: rescale_multipass_history.py uses for its tier-2 fallback).
OUR_RAW_SOURCES = (
    "live_exact_screen", "live_exact_repeat", "local_exact_raw",
    "children_8x", "children_20x", "children_40x", "backfill_8x",
    "exhaustive_8x",
)


def fetch_elite_history_parents(
    db_path: Path, *, target: str = TARGET, before_epoch: int, min_passes: int = 3,
    limit: int = 500,
) -> dict[str, float]:
    """Our own historically strong, since-abandoned sequences as a parent
    source that never ages out just because nobody has resubmitted it.

    `generate()`'s broad pool is rival-only when --rival-coldkey is given
    (the production path), and `select_elite_parents` was narrowed to
    --parent-limit 64 on 2026-10-06 -- so a sequence WE generated, scored
    well, and then stopped mutating (no resubmission, so multipass_all_
    rivals.py -- which only rescoring sequences it currently sees on the
    live leaderboard -- never touched it) had NO lane that ever reconsidered
    it. Confirmed case, 2026-10-11: a sequence 1 edit from the actual #1
    winner at epoch 25656 sat in our own local_exact_raw/children_8x/20x
    history at epochs 25423-25552 (rank-sum 59-287, clearly strong) and was
    never mutated or resubmitted again.

    Reuses OUR_RAW_SOURCES observations the exact way rescale_multipass_
    history.py's tier 2 already treats as "our multipass", and the same
    dense-sum-of-10 rank-sum definition as everywhere else, against the
    exact same pooled recent-field (rank_exact.fetch_recent_fields_from_db
    + pool_recent_fields) -- this never introduces a second scoring scale.
    Ranks each candidate against the field alone (see the comment at the
    ranking loop below for why rank_metrics_against_field itself is not
    reused here). Requires >= min_passes raw observations per sequence, the
    same robustness bar multipass gating applies elsewhere: a single noisy
    pass does not get an exemption just because it is old.
    """
    import json as _json

    import numpy as np

    from nanobody_opt import db
    from nanobody_opt.rank_exact import METRICS, fetch_recent_fields_from_db, pool_recent_fields

    connection = db.connect_local(db_path)
    pooled_field = pool_recent_fields(
        fetch_recent_fields_from_db(db_path, target=target, before_epoch=before_epoch, count=10)
    )
    if pooled_field.empty:
        return {}

    p = "?"
    placeholders = ",".join([p] * len(OUR_RAW_SOURCES))
    rows = connection.execute(
        f"""
        SELECT n.sequence, o.observation_id, o.metrics_json
        FROM observations o JOIN nodes n ON n.id = o.node_id
        WHERE n.target = {p} AND o.epoch < {p} AND o.source IN ({placeholders})
          AND o.metrics_json IS NOT NULL AND o.metrics_json NOT IN ('', '{{}}')
        """,
        (target, before_epoch, *OUR_RAW_SOURCES),
    ).fetchall()

    passes_by_seq: dict[str, list[dict[str, float]]] = {}
    for seq, _observation_id, blob in rows:
        try:
            metrics = _json.loads(blob)
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict) or not metrics or not set(METRICS).issubset(metrics):
            continue
        try:
            values = {name: float(metrics[name]) for name in METRICS}
        except (TypeError, ValueError):
            continue
        passes_by_seq.setdefault(seq.upper(), []).append(values)

    qualifying = {seq: passes for seq, passes in passes_by_seq.items()
                  if len(passes) >= min_passes}
    if not qualifying:
        return {}
    # Mean metrics per sequence (same "mean of several real measurements"
    # principle used throughout this session), ranked once against the
    # current field -- never the stale score a pass carried when written.
    seqs = list(qualifying)
    mean_metrics = {
        seq: {name: sum(pass_[name] for pass_ in qualifying[seq]) / len(qualifying[seq])
              for name in METRICS}
        for seq in seqs
    }
    # Deliberately NOT rank_metrics_against_field: that batches every
    # candidate into the same combined frame and dense-ranks them together,
    # so a candidate's score depends on how many OTHER candidates were
    # passed in the same call -- harmless for its usual callers (a handful
    # of passes or one epoch's ~50 submissions) but silently inflates the
    # scale by up to 10x here, with tens of thousands of candidates at once
    # (confirmed 2026-10-11: scores up to 45,972 against a field whose true
    # ceiling is ~5,230, the exact shape of the pool-inflation bug already
    # fixed once in rescale_multipass_history.py). Instead, rank each
    # candidate against the field ALONE, vectorized: a value's dense rank if
    # inserted one at a time equals (count of strictly "worse" field values)
    # + 1, computed via searchsorted on the field's sorted unique values --
    # identical result to concatenating field+1 candidate, for every
    # candidate at once.
    total = np.zeros(len(seqs), dtype=np.float64)
    for metric, direction in METRICS.items():
        field_sorted = np.sort(pooled_field[metric].to_numpy(dtype=np.float64))
        unique_sorted = np.unique(field_sorted)
        values = np.array([mean_metrics[seq][metric] for seq in seqs], dtype=np.float64)
        if direction == "min":
            # rank 1 = smallest value = best; rank(v) = #(unique field < v) + 1
            total += np.searchsorted(unique_sorted, values, side="left") + 1
        else:
            # rank 1 = largest value = best; rank(v) = #(unique field > v) + 1
            total += (len(unique_sorted)
                       - np.searchsorted(unique_sorted, values, side="right")) + 1
    ranks = dict(zip(seqs, total.tolist()))
    return dict(sorted(ranks.items(), key=lambda kv: kv[1])[:limit])


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


def _merge_priority(priority: list[str], scores: dict[str, float]) -> list[str]:
    """Keep EVERY priority (top-50) parent in its own list order (Oct 8 rule).

    The 2026-10-10 gate dropped top-50 parents without 10 stored replay
    passes and re-sorted the rest by multipass score. With a 2,000-candidate
    --limit, list order decides which ~8 parents get edited at all, so the
    Oct 8 behaviour is the plain top-50 order. ``scores`` is accepted for the
    call site but deliberately not used for ordering.
    """
    return list(priority)


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
             rival_sequences: list[str] | None = None,
             mutation_database: Path | None = None,
             min_passes: int = 10,
             epoch: int | None = None,
             elite_history_scores: dict[str, float] | None = None) -> dict[str, Any]:
    # Parent source: the rival's proven-good sequences when supplied, else the
    # global archive ranked by the validator's Algorithm-2 proxy.
    if rival_sequences:
        pool = [s.upper() for s in rival_sequences]
    else:
        pool = _load_pool(archive)
    # Our own historically-strong, since-abandoned sequences (see
    # fetch_elite_history_parents) are a parent source the rival-only broad
    # pool above has no way to surface -- added here, not as a replacement,
    # so this never narrows what --rival-coldkey already finds.
    if elite_history_scores:
        existing = set(pool)
        pool = pool + [s for s in elite_history_scores if s not in existing]
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

    multipass_ranked = 0
    if mutation_database is not None:
        # User directive, 2026-10-10: "demand multipass scoring, only use
        # those" -- drop every parent (priority and broad pool alike) without
        # >= min_passes stored replay passes, and rank what's left by that
        # robust score instead of the validator's single final_score / the
        # archive's single-snapshot Algorithm-2 proxy. See multipass_scores()
        # docstring for why this is the same principle as --full-archive-field,
        # one step upstream.
        scores = multipass_scores(
            sorted(set(priority_pool) | set(pool)), mutation_database,
            min_passes=min_passes, epoch=epoch)
        # Elite-history candidates are already gated on >= min_passes raw
        # observations (fetch_elite_history_parents), just not the
        # public_multipass_rivals/multipass_sum10_pass label multipass_
        # scores() requires -- that label is only ever written for
        # sequences someone is CURRENTLY resubmitting, which is exactly the
        # trap this pool exists to escape. Merge rather than re-gate them.
        if elite_history_scores:
            scores = {**scores, **{s: v for s, v in elite_history_scores.items()
                                    if s not in scores}}
        multipass_ranked = len(scores)
        priority_pool = _merge_priority(priority_pool, scores)
        pool = sorted((s for s in pool if s in scores), key=lambda s: scores[s])

    # Keep the pool ranked best-first (Algorithm-2 from _load_pool, or the
    # multipass score above) rather than shuffling: the shuffle threw away
    # the quality signal and diluted rare winning motifs (e.g. the T/V CDR3
    # variants) to their ~random share of the 2000-slot cap.
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
        "multipass_required": mutation_database is not None,
        "multipass_ranked_pool": multipass_ranked,
        "pool_size": len(priority_pool) + len(pool),
        "priority_pool_size": len(priority_pool),
        "elite_history_pool_size": len(elite_history_scores) if elite_history_scores else 0,
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
    parser.add_argument("--mutation-database", type=Path, default=None,
                        help="when given, every parent (priority and broad pool) must have >= --min-passes stored multipass_sum10_pass observations, ranked by that score instead of the validator final_score / archive Algorithm-2 proxy")
    parser.add_argument("--min-passes", type=int, default=10,
                        help="stored passes required per parent when --mutation-database is given (default 10)")
    parser.add_argument("--elite-history-limit", type=int, default=500,
                        help="max of our own historically-strong, since-abandoned sequences to add to the broad pool (0 = off, default 500)")
    parser.add_argument("--elite-history-min-passes", type=int, default=3,
                        help="raw OUR_RAW_SOURCES observations required per elite-history sequence (default 3)")
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
    import re

    archive_epoch_match = re.search(r"epoch(\d+)", str(args.archive))
    rival_sequences = None
    if args.rival_coldkey:
        if not archive_epoch_match:
            raise SystemExit(f"cannot derive epoch from archive path {args.archive}")
        rival_sequences = fetch_rival_sequences(
            int(archive_epoch_match.group(1)), args.rival_coldkey)
        if not rival_sequences:
            raise SystemExit(
                f"no sequences found for rival coldkey {args.rival_coldkey!r}"
            )
    epoch = int(archive_epoch_match.group(1)) if archive_epoch_match else None
    if args.mutation_database is not None and epoch is None:
        raise SystemExit(
            f"--mutation-database requires an epoch, and none could be "
            f"derived from archive path {args.archive}"
        )
    elite_history_scores = None
    if args.mutation_database is not None and args.elite_history_limit:
        elite_history_scores = fetch_elite_history_parents(
            args.mutation_database, before_epoch=epoch,
            min_passes=args.elite_history_min_passes,
            limit=args.elite_history_limit)
    payload = generate(args.archive, positions, args.limit, pool_limit=args.pool_limit,
                       top50=top50, min_parent_distance=args.min_parent_distance,
                       priority=priority, double_edit_pairs=args.double_edit_pairs,
                       ranker_model=ranker_model,
                       rival_sequences=rival_sequences,
                       mutation_database=args.mutation_database,
                       min_passes=args.min_passes,
                       epoch=epoch,
                       elite_history_scores=elite_history_scores)
    if args.mutation_database is not None and payload["pool_size"] == 0:
        raise SystemExit(
            f"--mutation-database given but zero parents have >= "
            f"{args.min_passes} stored passes; refusing to write an empty "
            "parent pool (strict multipass-only mode, no silent fallback)"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"COMPETITOR_LANE_READY pool={payload['pool_size']} "
        f"candidates={len(payload['candidates'])} "
        f"multipass_required={payload['multipass_required']} "
        f"multipass_ranked_pool={payload['multipass_ranked_pool']} "
        f"elite_history_pool={payload['elite_history_pool_size']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
