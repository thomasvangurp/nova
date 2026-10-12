#!/usr/bin/env python3
"""Epoch-independent replay queue: which sequences should the slave replay next.

A fold is a property of (sequence, target), so a replay pass is useful whatever
epoch the sequence is later submitted in. This tool lists sequences still short
of PASSES canonical passes (source public_multipass_rivals, score_kind
multipass_sum10_pass, non-empty metrics) in priority order and emits one batch
sized to a seq-pass budget. Read-only on the DB; the only state it writes is a
small attempts log so a sequence that never gets a pass is not retried forever.

Priority (rival-submission replay stays ahead of ALL of this; that gate lives in
tools/replay_queue_loop.sh):
  0  shadow tournament picks (tools/shadow_tournament.py): the top-13 of every ticket-ordering
     rule for each closed epoch, newest first, so the rules can be compared on full coverage
  1  live top-50 parents (newest top50_index.json live_sequences), then the
     parents the newest competitor_parents.json actually mutates
  2  best provisional children of the live race (own screen/raw rank-sum, this
     and last epoch) -- the ones headed for confirmation with 1-2 observations
  3  our own historical near-misses (best rank vs the CURRENT field, >=3 raw
     passes), as filler so the GPUs are never idle

Usage (master, bgen env):
  replay_queue.py --out /tmp/batch.json [--budget-seq-passes 350] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root")

DB = "/root/artifacts/nanobody_mutations.sqlite"
TARGET = "P20809"
SUPPORTED_LENGTH = 127  # the fast scorer only supports this length; others hang or are refused
SOURCE, KIND = "public_multipass_rivals", "multipass_sum10_pass"
STATE_DIR = Path("/root/artifacts/replay_queue")
ATTEMPTS = STATE_DIR / "attempts.json"
ELITE_CACHE = STATE_DIR / "elite_history.json"
ELITE_TTL_SECONDS = 6 * 3600
MAX_ATTEMPTS_PER_DAY = 3
OWN_SCORE_KINDS = ("local_validator_replay_rank_sum", "local_exact_validator_replay_rank_sum")


def chain_epoch() -> int:
    return int(json.load(open("/root/artifacts/chain_state.json"))["epoch"])


def newest(pattern: str) -> str | None:
    paths = glob.glob(pattern)
    return max(paths, key=os.path.getmtime) if paths else None


def tier0_shadow() -> list[str]:
    """Sequences the shadow tournament needs replayed (every ordering rule's top-13 per closed
    epoch, newest epoch first). Never allowed to break the queue."""
    try:
        sys.path.insert(0, "/root/tools")
        import shadow_tournament

        return shadow_tournament.pending_sequences()
    except Exception as error:  # noqa: BLE001
        print(f"SHADOW_TIER_SKIPPED {type(error).__name__}: {error}", file=sys.stderr)
        return []


def tier1_parents() -> list[str]:
    seqs: list[str] = []
    top50 = newest("/root/live/epoch*/producer/iteration_*/top50_index.json")
    if top50:
        blob = json.load(open(top50))
        seqs += [str(s).strip().upper() for s in blob.get("live_sequences") or blob.get("sequences") or []]
    parents = newest("/root/live/epoch*/producer/competitor_parents.json")
    if parents:
        for cand in json.load(open(parents)).get("candidates") or []:
            p = str(cand.get("parent") or "").strip().upper()
            if p:
                seqs.append(p)
    return list(dict.fromkeys(seqs))


def tier2_children(conn: sqlite3.Connection, epoch: int, limit: int = 300) -> list[str]:
    marks = ",".join("?" * len(OWN_SCORE_KINDS))
    rows = conn.execute(
        f"SELECT n.sequence, AVG(o.score) m FROM observations o JOIN nodes n ON n.id = o.node_id "
        f"WHERE n.target = ? AND o.epoch >= ? AND o.score_kind IN ({marks}) "
        f"AND o.source IN ('live_exact_screen','local_exact_raw','live_exact_repeat') "
        f"GROUP BY n.sequence ORDER BY m ASC LIMIT ?",
        (TARGET, epoch - 1, *OWN_SCORE_KINDS, limit),
    ).fetchall()
    return [str(r[0]).strip().upper() for r in rows]


def tier3_history(epoch: int, limit: int = 3000) -> list[str]:
    if ELITE_CACHE.exists() and time.time() - ELITE_CACHE.stat().st_mtime < ELITE_TTL_SECONDS:
        return json.load(open(ELITE_CACHE))["sequences"]
    from nanobody_opt.competitor_lane import fetch_elite_history_parents

    scores = fetch_elite_history_parents(Path(DB), before_epoch=epoch, min_passes=3, limit=limit)
    seqs = [s for s, _ in sorted(scores.items(), key=lambda kv: kv[1])]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ELITE_CACHE.write_text(json.dumps({"sequences": seqs, "epoch": epoch}))
    return seqs


def canonical_counts(conn: sqlite3.Connection, seqs: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(0, len(seqs), 400):
        chunk = seqs[i:i + 400]
        rows = conn.execute(
            f"SELECT n.sequence, COUNT(*) FROM observations o JOIN nodes n ON n.id = o.node_id "
            f"WHERE n.target = ? AND n.sequence IN ({','.join('?' * len(chunk))}) "
            f"AND o.source = ? AND o.score_kind = ? AND o.metrics_json NOT IN ('', '{{}}') "
            f"GROUP BY n.sequence", (TARGET, *chunk, SOURCE, KIND)).fetchall()
        counts.update({str(s).upper(): int(c) for s, c in rows})
    return counts


def load_attempts() -> dict[str, list[float]]:
    try:
        return json.load(open(ATTEMPTS))
    except (OSError, ValueError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--budget-seq-passes", type=int, default=350)
    ap.add_argument("--max-sequences", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-limit", type=int, default=0,
                    help="verification only: cap sequences, and need per sequence at --test-passes")
    ap.add_argument("--test-passes", type=int, default=2)
    args = ap.parse_args()

    epoch = chain_epoch()
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    now = time.time()
    attempts = load_attempts()

    tiers: list[tuple[int, list[str]]] = [(0, tier0_shadow()), (1, tier1_parents()), (2, tier2_children(conn, epoch))]
    items: list[dict] = []
    seen: set[str] = set()
    budget = args.budget_seq_passes
    if args.test_limit:
        budget = args.test_limit * args.test_passes
    stats: dict[int, dict[str, int]] = {}

    def take(tier: int, seqs: list[str]) -> None:
        nonlocal budget
        counts = canonical_counts(conn, [s for s in seqs if s not in seen and len(s) == SUPPORTED_LENGTH])
        stat = stats.setdefault(tier, {"listed": 0, "short": 0, "taken": 0})
        for s in seqs:
            if s in seen or len(s) != SUPPORTED_LENGTH:
                continue
            seen.add(s)
            stat["listed"] += 1
            need = args.passes - counts.get(s, 0)
            if args.test_limit:
                need = min(need, args.test_passes)
            if need < 1:
                continue
            stat["short"] += 1
            recent = [t for t in attempts.get(s, []) if now - t < 86400]
            if len(recent) >= MAX_ATTEMPTS_PER_DAY and not args.test_limit:
                continue
            if budget < need or len(items) >= (args.test_limit or args.max_sequences):
                continue
            items.append({"sequence": s, "need": need, "tier": tier})
            budget -= need
            stat["taken"] += 1

    for tier, seqs in tiers:
        take(tier, seqs)
    if budget > 0 and len(items) < (args.test_limit or args.max_sequences):
        take(3, tier3_history(epoch))

    print(json.dumps({"epoch": epoch, "sequences": len(items),
                      "seq_passes": sum(i["need"] for i in items), "tiers": stats}, sort_keys=True))
    if args.dry_run or not args.out:
        return 0 if items else 4
    if not items:
        return 4
    batch_id = f"{int(now)}"
    args.out.write_text(json.dumps({"batch_id": batch_id, "epoch": epoch, "items": items}))
    for it in items:
        attempts.setdefault(it["sequence"], []).append(now)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ATTEMPTS.write_text(json.dumps({s: [t for t in ts if now - t < 86400] for s, ts in attempts.items()
                                    if any(now - t < 86400 for t in ts)}))
    print(f"REPLAY_QUEUE_BATCH id={batch_id} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
