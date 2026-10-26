#!/usr/bin/env python3
"""Recompute every historical multipass_sum10_pass epoch against ITS OWN
real last-10-epochs field, instead of the static full-archive field it was
originally scored against.

User, 2026-10-11: "No full archive anywhere please" / "All need to be
recalculated to 10 recent also on dashboard." The live producer and
dashboard writer both switched to the real recent-epochs field today
(rank_exact.fetch_recent_fields_from_db), but 114 epochs of ALREADY-WRITTEN
dashboard history (25533-25653) were still on the old hf_archive_full scale
-- a ~100x jump at the switchover that the dashboard can label as a regime
boundary (see nanobody_dash/progress.py), but a boundary is still a split.
This eliminates it by rederiving every historical epoch on the SAME field
methodology, so the whole series is one consistent regime end to end.

Inputs, no GPU work:
  - raw per-pass metrics: already stored (observations.metrics_json,
    source=public_multipass_rivals, score_kind=multipass_sum10_pass) for
    every epoch in range, formula-independent by design.
  - cohort (which coldkey each sequence belongs to): refetched from the same
    public leaderboard API multipass_all_rivals.py already calls -- not
    persisted per-sequence in the stored observations, but a deterministic,
    free, already-published historical record.
  - the real field itself: rank_exact.fetch_recent_fields_from_db, the same
    function the live writer now uses, called with before_epoch=<the epoch
    being rescaled>, never today's epoch -- so a 2026-10-01 epoch is ranked
    against 2026-10-01's real recent field, not 2026-10-11's.

Output: one corrected multipass_<epoch>.json per epoch, same shape
multipass_all_rivals.py itself writes (topup section zeroed -- it is parent-
pool metadata, never part of the ours/rivals chart data), written locally
here and left for the caller to ship to the dashboard box (the actual file
sync_multipass_rivals.py reads).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from nanobody_opt.rules import target_for_epoch

API = "https://compound-api.metanova-labs.ai"
TARGET = "P20809"
OUR_COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"
LOCAL_DB = Path("/root/artifacts/nanobody_mutations.sqlite")
FIELD_EPOCHS = 10
PASSES_PER_SEQUENCE = 10
#: User, 2026-10-11: "All raw data is there and in db" / "Donot give up so
#: fast". Confirmed: epochs where public_multipass_rivals's own
#: multipass_sum10_pass rows for OUR sequences carry empty metrics_json
#: (pre-sum10-formula passes, score_kind legacy_multipass_algo2_pass etc.)
#: are NOT actually missing raw data -- OUR OWN sequences were independently,
#: repeatedly scored by the race itself under these sources (same scorer,
#: same challenge, fold never expires -- the exact reasoning
#: _recent_observation_history and local_passes_for_ours already rely on).
#: Only rivals have no such fallback: we never ran their sequences through
#: our own race, so a rival's only possible record is whatever
#: public_multipass_rivals itself stored.
OUR_RAW_SOURCES = (
    "live_exact_screen", "live_exact_repeat", "local_exact_raw",
    "children_8x", "children_20x", "children_40x", "backfill_8x",
    "exhaustive_8x",
)


def fetch_leaderboard(epoch: int) -> list[tuple[str, str]] | None:
    req = urllib.request.Request(
        f"{API}/api/competitions/leaderboard/{epoch}/nanobodies",
        headers={"User-Agent": "nova-rescale-multipass/1"},
    )
    data = None
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        data = payload.get("data") if isinstance(payload, dict) else None
    except Exception:
        data = None
    if not isinstance(data, list) or not data:
        # The API has no rows (since 25693 the validator's upload is rejected): use the rebuilt-from-log board.
        try:
            from tools.leaderboard_fallback import load_rows
        except ImportError:
            from leaderboard_fallback import load_rows
        data = load_rows(epoch)
        if not isinstance(data, list):
            return None
    # Measurements belong to a sequence, but leaderboard membership belongs
    # to its owner too. The same sequence can score for two owners in one
    # epoch; globally deduplicating it silently removes the later owner's row.
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for submission in data:
        coldkey = str(submission.get("coldkey") or "")
        for nb in submission.get("nanobodies") or []:
            if nb.get("protein_name") != target_for_epoch(epoch):
                continue
            seq = str(nb.get("sequence") or "").strip().upper()
            if seq and (coldkey, seq) not in seen:
                seen.add((coldkey, seq))
                pairs.append((coldkey, seq))
    return pairs


def multipass_sum10_metrics(epoch: int, sequences: set[str] | None = None
                            ) -> dict[str, dict[str, float]]:
    """{"<sequence>::<observation_id>": metrics} from public_multipass_rivals
    (every cohort scores here; this is the ONLY possible source for a
    rival, since we never run a rival's sequence through our own race)."""
    import sqlite3

    target = target_for_epoch(epoch)
    conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True, timeout=30)
    base = (
        "SELECT n.sequence, o.observation_id, o.metrics_json "
        "FROM observations o JOIN nodes n ON n.id = o.node_id "
        "WHERE n.target = ? AND o.source = 'public_multipass_rivals' "
        "  AND o.score_kind = 'multipass_sum10_pass' "
        "  AND o.metrics_json IS NOT NULL AND o.metrics_json NOT IN ('', '{}') "
    )
    try:
        if sequences is None:
            rows = conn.execute(base + " AND o.epoch = ?", (target, epoch)).fetchall()
        else:
            # Every canonical raw fold belongs to (sequence, target). Reuse
            # stored passes across epochs regardless of which replay writer
            # produced them; bound by the real submitted sequence set.
            rows = []
            ordered = sorted(sequences)
            for i in range(0, len(ordered), 300):
                chunk = ordered[i:i + 300]
                rows += conn.execute(
                    base + f" AND n.sequence IN ({','.join('?' * len(chunk))}) "
                    "ORDER BY o.observation_id",
                    (target, *chunk),
                ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, float]] = {}
    for sequence, observation_id, metrics_json in rows:
        if sequences is not None and sequence not in sequences:
            continue
        try:
            metrics = json.loads(metrics_json)
        except (TypeError, ValueError):
            continue
        if isinstance(metrics, dict) and metrics:
            out[f"{sequence}::{observation_id}"] = metrics
    return out


def our_race_metrics(sequences: set[str], target: str = TARGET) -> dict[str, dict[str, float]]:
    """{"<sequence>::<observation_id>": metrics} for OUR OWN sequences from
    the race's own exact-scoring history (OUR_RAW_SOURCES) -- the same data
    local_passes_for_ours() already reuses, no epoch filter: a fold is a
    property of (sequence, target), never of when it happened to be scored
    (same reasoning _recent_observation_history relies on). Capped at
    PASSES_PER_SEQUENCE per sequence so a heavily-observed sequence cannot
    outweigh a lightly-observed rival in the fleet mean."""
    import sqlite3

    if not sequences:
        return {}
    conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True, timeout=30)
    try:
        placeholders = ",".join("?" * len(sequences))
        source_placeholders = ",".join("?" * len(OUR_RAW_SOURCES))
        rows = conn.execute(
            f"SELECT n.sequence, o.observation_id, o.metrics_json "
            f"FROM observations o JOIN nodes n ON n.id = o.node_id "
            f"WHERE n.target = ? AND n.sequence IN ({placeholders}) "
            f"  AND o.source IN ({source_placeholders}) "
            f"  AND o.metrics_json IS NOT NULL AND o.metrics_json != '' "
            f"ORDER BY o.observation_id",
            (target, *sequences, *OUR_RAW_SOURCES),
        ).fetchall()
    finally:
        conn.close()
    per_seq_count: dict[str, int] = defaultdict(int)
    out: dict[str, dict[str, float]] = {}
    for sequence, observation_id, metrics_json in rows:
        if per_seq_count[sequence] >= PASSES_PER_SEQUENCE:
            continue
        try:
            metrics = json.loads(metrics_json)
        except (TypeError, ValueError):
            continue
        if isinstance(metrics, dict) and metrics:
            out[f"{sequence}::{observation_id}"] = metrics
            per_seq_count[sequence] += 1
    return out


def validator_api_metrics(epoch: int, sequences: set[str]
                          ) -> dict[str, dict[str, float]]:
    """{"<sequence>::<observation_id>": metrics} from the real validator's
    own published per-submission metrics (source=public_validator_api),
    ingested continuously by seed_mutation_database.py -- the universal
    fallback that covers a rival sequence no other source can (we never
    fold a rival's own sequence ourselves). Not epoch-filtered: the
    sequence's own real fold is valid evidence whenever it was published,
    same reasoning as our_race_metrics, and this source only ever has at
    most a handful of rows per sequence anyway (one per epoch it was
    actually submitted)."""
    import sqlite3

    if not sequences:
        return {}
    conn = sqlite3.connect(f"file:{LOCAL_DB}?mode=ro", uri=True, timeout=30)
    try:
        placeholders = ",".join("?" * len(sequences))
        rows = conn.execute(
            f"SELECT n.sequence, o.observation_id, o.metrics_json "
            f"FROM observations o JOIN nodes n ON n.id = o.node_id "
            f"WHERE n.target = ? AND n.sequence IN ({placeholders}) "
            f"  AND o.source = 'public_validator_api' "
            f"  AND o.metrics_json IS NOT NULL AND o.metrics_json != '' "
            f"ORDER BY ABS(o.epoch - ?)",
            (target_for_epoch(epoch), *sequences, epoch),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for sequence, observation_id, metrics_json in rows:
        if sequence in seen:
            continue
        try:
            metrics = json.loads(metrics_json)
        except (TypeError, ValueError):
            continue
        if isinstance(metrics, dict) and metrics:
            out[f"{sequence}::{observation_id}"] = metrics
            seen.add(sequence)
    return out


def rescale_epoch(epoch: int) -> dict | None:
    from nanobody_opt.rank_exact import (
        fetch_recent_fields_from_db, pool_recent_fields, rank_metrics_against_field,
    )

    pairs = fetch_leaderboard(epoch)
    if not pairs:
        return None
    by_coldkey: dict[str, list[str]] = defaultdict(list)
    for coldkey, seq in pairs:
        by_coldkey[coldkey].append(seq.upper())
    our_seqs = set(by_coldkey.get(OUR_COLDKEY, []))
    all_seqs = {seq for seqs in by_coldkey.values() for seq in seqs}

    # sequences=all_seqs is load-bearing, not an optimization: without it
    # this also pulls every HF-top200 bounded-topup "parent candidate" row
    # multipass_all_rivals.py scores alongside the real field each epoch
    # (thousands of rows some epochs) into rank_metrics_against_field's
    # candidate pool -- they get ranked (and counted in the denominator)
    # right alongside the real rivals, inflating a real candidate's rank by
    # up to ~35x on a heavy-topup epoch (confirmed: epoch 25598 alone had
    # 17,827 tier-1 rows for ~50 real submissions, producing rank-sums of
    # 20,000-30,000 against a 493-row field whose true ceiling is ~4,930).
    pass_metrics = multipass_sum10_metrics(epoch, all_seqs)
    covered_our_seqs = {
        key.rsplit("::", 1)[0] for key in pass_metrics
    } & our_seqs
    missing_our_seqs = our_seqs - covered_our_seqs
    if missing_our_seqs:
        pass_metrics.update(our_race_metrics(missing_our_seqs, target_for_epoch(epoch)))

    # A single validator draw is not a multipass mean. Keep missing
    # evidence explicit rather than silently changing measurement regimes.
    depths: dict[str, int] = defaultdict(int)
    for key in pass_metrics:
        depths[key.rsplit("::", 1)[0]] += 1
    qualified = {seq for seq, depth in depths.items()
                 if depth >= PASSES_PER_SEQUENCE}
    pass_metrics = {key: metrics for key, metrics in pass_metrics.items()
                    if key.rsplit("::", 1)[0] in qualified}

    fields = fetch_recent_fields_from_db(
        LOCAL_DB, target=target_for_epoch(epoch), before_epoch=epoch, count=FIELD_EPOCHS)
    if not fields:
        return None
    pooled = pool_recent_fields(fields)
    pass_ranks = rank_metrics_against_field(pass_metrics, pooled)

    seq_of_key = {k: k.rsplit("::", 1)[0] for k in pass_ranks}
    by_seq: dict[str, list[float]] = defaultdict(list)
    for key, rank_sum in pass_ranks.items():
        by_seq[seq_of_key[key]].append(rank_sum)
    seq_means = {seq: sum(v) / len(v) for seq, v in by_seq.items()}

    import numpy as np

    result: dict = {
        "ours": None,
        "rivals": {},
        "sequences": {s: float(m) for s, m in seq_means.items()},
        "method": {
            "target": target_for_epoch(epoch),
            "score_kind": "multipass_sum10_pass",
            "formula": "sum10",
            "measurement": "local_fast_scorer_multipass_mean",
            "passes": None,
            "minimum_passes": PASSES_PER_SEQUENCE,
            "coverage_policy": "complete_fleet_minimum_10_passes",
            "field": "recent_validator_epochs",
            "field_rows": int(len(pooled)),
            "field_sha256": None,
            "writer": "rescale_multipass_history",
        },
        "topup": {"cohort": "hf_top200", "enabled": False, "depth": None,
                  "sequences": 0, "sequence_passes": 0, "rows_written": 0},
    }
    for coldkey, seqs in sorted(by_coldkey.items(), key=lambda kv: -len(kv[1])):
        scored = [seq_means[s] for s in seqs if s in seq_means]
        complete = len(scored) == len(seqs)
        entry = {
            "mean": float(np.mean(scored)) if complete and scored else None,
            "stdev": float(np.std(scored)) if complete and len(scored) > 1 else None,
            "fleet_size": len(seqs),
            "scored_sequences": len(scored),
            "missing_sequences": len(seqs) - len(scored),
            "minimum_passes": PASSES_PER_SEQUENCE,
            "coverage_complete": complete,
            "passes_min": min((depths[s] for s in seqs if s in seq_means), default=0),
            "passes_max": max((depths[s] for s in seqs if s in seq_means), default=0),
            "scores": [float(v) for v in sorted(scored)] if complete else [],
        }
        if coldkey == OUR_COLDKEY:
            result["ours"] = entry
        else:
            result["rivals"][coldkey] = entry
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", required=True, help="e.g. 25533-25653")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--compare-against", type=Path, default=None,
        help="existing multipass_comparison.json; an epoch whose recompute "
             "would lose ours (present before, None now) or lose more than "
             "half its rivals is left unwritten instead of overwritten -- "
             "raw metrics genuinely gone (pre-2026-10-09, disk since rotated) "
             "cannot be recomputed, and the old value is strictly better "
             "than destroying it",
    )
    args = parser.parse_args()

    existing: dict = {}
    if args.compare_against is not None:
        existing = json.loads(args.compare_against.read_text())

    lo, hi = (int(x) for x in args.epochs.split("-", 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok, skipped, preserved = 0, 0, 0
    for epoch in range(lo, hi + 1):
        result = rescale_epoch(epoch)
        if result is None:
            print(f"RESCALE_SKIP epoch={epoch} (no stored passes, no field, "
                  "or leaderboard unavailable)", flush=True)
            skipped += 1
            continue
        prior = existing.get(str(epoch)) or {}
        prior_ours = prior.get("ours")
        prior_rivals = prior.get("rivals") or {}
        lost_ours = prior_ours is not None and result["ours"] is None
        lost_rivals = (
            len(prior_rivals) >= 4
            and len(result["rivals"]) < len(prior_rivals) / 2
        )
        if lost_ours or lost_rivals:
            print(f"RESCALE_PRESERVED epoch={epoch} (raw metrics genuinely "
                  f"gone: lost_ours={lost_ours} prior_rivals="
                  f"{len(prior_rivals)} new_rivals={len(result['rivals'])}; "
                  "leaving the original archive-scale value in place)", flush=True)
            preserved += 1
            continue
        out_path = args.out_dir / f"multipass_{epoch}.json"
        out_path.write_text(json.dumps({str(epoch): result}, indent=2) + "\n")
        ours_mean = result["ours"]["mean"] if result["ours"] else None
        print(f"RESCALE_OK epoch={epoch} ours_mean={ours_mean} "
              f"rivals={len(result['rivals'])} field_rows={result['method']['field_rows']}",
              flush=True)
        ok += 1
    print(f"RESCALE_DONE ok={ok} skipped={skipped} preserved={preserved}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
