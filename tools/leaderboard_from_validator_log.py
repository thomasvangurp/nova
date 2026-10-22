"""Rebuild a nanobody leaderboard from the validator's own log when the public API never publishes the epoch.

2026-10-13, epoch 25693: the validator's upload to the Dashboard API was rejected (HTTP 422 on non-integer
``competition.min_heavy_atoms`` / ``min_rotatable_bonds`` / ``max_rotatable_bonds``), so the public leaderboard stayed
404 while ranking and payouts went ahead.  The validator still logs, at DEBUG, one line ``Final boltzgen scores: {uid:
{sequence: {target: {...metrics, rank_sum, ...}}}}`` for every scored UID.  That line carries the same sequence,
metrics and final score (``rank_sum``) the API would have served.

What this does NOT have: the validator's commit-order tie-break (taken from the logged winner UID), per-UID molecule
scores, and ``blocks_elapsed``.  The output marks itself as derived (``source``), and the dashboard stores that in
``epoch_provenance`` so a derived epoch is never mistaken for an API one.

Usage:
  uv run --with bittensor python tools/leaderboard_from_validator_log.py --epoch 25693 --dump scores_line.txt \
      --winner-uid 14 --database data/nanobody_dash.sqlite --output-dir data/leaderboard_from_validator_log
"""
from __future__ import annotations

import argparse
import ast
import json
import sqlite3
import sys
import time
from pathlib import Path

TARGET = "P20809"
NETUID = 68


def parse_final_scores(text: str) -> dict[int, dict]:
    """``{uid: {"sequence", "final_score", "metrics"}}`` from the logged dict (the text may carry a log prefix)."""
    start = text.index("{")
    raw = ast.literal_eval(text[start:].strip())
    out: dict[int, dict] = {}
    for uid, sequences in raw.items():
        if len(sequences) != 1:
            raise ValueError(f"UID {uid} has {len(sequences)} sequences; expected exactly one")
        (sequence, targets), = sequences.items()
        metrics = targets[TARGET]
        out[int(uid)] = {"sequence": sequence, "final_score": float(metrics["rank_sum"]),
                         "metrics": {k: v for k, v in metrics.items()}}
    return out


def build_rows(scores: dict[int, dict], mapping: dict[int, dict], winner_uid: int | None = None) -> list[dict]:
    """API-shaped rows, best first.  Ties go to ``winner_uid`` (the validator's logged winner), then to the lower UID."""
    ordered = sorted(scores.items(),
                     key=lambda item: (item[1]["final_score"], 0 if item[0] == winner_uid else 1, item[0]))
    if winner_uid is not None and ordered and ordered[0][0] != winner_uid:
        raise ValueError(f"logged winner UID {winner_uid} does not have the best score "
                         f"(best is UID {ordered[0][0]} at {ordered[0][1]['final_score']})")
    unmapped = [uid for uid, _ in ordered if not (mapping.get(uid) or {}).get("hotkey") or not (mapping.get(uid) or {}).get("coldkey")]
    if unmapped:
        raise ValueError(f"no hotkey/coldkey for UIDs {unmapped}")
    rows = []
    for rank, (uid, item) in enumerate(ordered, start=1):
        who = mapping[uid]
        rows.append({
            "uid": uid, "coldkey": who.get("coldkey"), "hotkey": who.get("hotkey"), "blocks_elapsed": None,
            "nanobody_leaderboard_rank": rank, "final_score": item["final_score"], "molecule_final_score": None,
            "nanobodies": [{"sequence": item["sequence"], "protein_name": TARGET, "final_score": str(item["final_score"]),
                            **{k: str(v) for k, v in item["metrics"].items()}}],
            "source": "validator_log",
        })
    return rows


def chain_mapping(uids) -> dict[int, dict]:
    import bittensor as bt  # imported lazily: tests and parsing do not need the chain
    subtensor = bt.Subtensor("finney")

    def query(item, params):
        value = subtensor.query(item, params)
        return getattr(value, "value", value)

    mapping = {}
    for uid in sorted(uids):
        hotkey = str(query(("SubtensorModule", "Keys"), [NETUID, uid]))
        mapping[uid] = {"hotkey": hotkey, "coldkey": str(query(("SubtensorModule", "Owner"), [hotkey])),
                        "registered_block": int(query(("SubtensorModule", "BlockAtRegistration"), [NETUID, uid]))}
    return mapping


def resolve_mapping(chain: dict[int, dict], close_block: int | None, fallback_rows: list[dict] | None) -> dict[int, dict]:
    """Chain owners as of now, except UIDs re-registered after the epoch closed.

    A UID that changed hands after ``close_block`` is not who was scored; take its occupant from ``fallback_rows``
    (a nearby published leaderboard).  Without one the epoch cannot be attributed correctly, so refuse.
    """
    fallback = {int(r["uid"]): r for r in (fallback_rows or [])}
    resolved, unresolved = {}, []
    for uid, who in chain.items():
        if close_block is not None and who.get("registered_block", 0) > close_block:
            old = fallback.get(uid)
            if old and old.get("hotkey") and old.get("coldkey"):
                resolved[uid] = {"hotkey": old["hotkey"], "coldkey": old["coldkey"], "from": "fallback_leaderboard"}
            else:
                unresolved.append(uid)
        else:
            resolved[uid] = {"hotkey": who["hotkey"], "coldkey": who["coldkey"], "from": "chain"}
    if unresolved:
        raise ValueError(f"UIDs {unresolved} were re-registered after block {close_block} and have no fallback occupant")
    return resolved


PROVENANCE_SCHEMA = ("CREATE TABLE IF NOT EXISTS epoch_provenance (epoch INTEGER PRIMARY KEY, source TEXT NOT NULL, "
                     "note TEXT, recorded_at REAL NOT NULL)")


def store(database: Path, epoch: int, rows: list[dict], note: str) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nanobody_dash import collect
    connection = collect.connect(database)
    try:
        connection.execute(PROVENANCE_SCHEMA)
        # a corrected rerun must replace the epoch, not sit beside rows with a different hotkey
        connection.execute("DELETE FROM entries WHERE epoch=?", (epoch,))
        stored = collect.store_epoch(connection, epoch, rows)
        connection.execute("INSERT OR REPLACE INTO epoch_provenance (epoch, source, note, recorded_at) VALUES (?,?,?,?)",
                           (epoch, "validator_log", note, time.time()))
        connection.commit()
    finally:
        connection.close()
    return stored


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--dump", type=Path, required=True, help="file holding the 'Final boltzgen scores: {...}' log line")
    parser.add_argument("--winner-uid", type=int, help="winner UID from the logged 'nanobody ranking' line (tie-break)")
    parser.add_argument("--mapping-json", type=Path, help="{uid: {hotkey, coldkey}} instead of querying the chain")
    parser.add_argument("--close-block", type=int, help="block the epoch closed at; UIDs re-registered after it use the fallback")
    parser.add_argument("--fallback-leaderboard", type=Path, help="a nearby published leaderboard JSON (e.g. the previous epoch)")
    parser.add_argument("--winner-coldkey", help="coldkey prefix the log shows being paid; must match the winner row")
    parser.add_argument("--database", type=Path, default=Path("data/nanobody_dash.sqlite"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/leaderboard_from_validator_log"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    scores = parse_final_scores(args.dump.read_text())
    if args.mapping_json:
        mapping = {int(k): v for k, v in json.loads(args.mapping_json.read_text()).items()}
    else:
        fallback = None
        if args.fallback_leaderboard:
            body = json.loads(args.fallback_leaderboard.read_text())
            fallback = body["data"] if isinstance(body, dict) else body
        mapping = resolve_mapping(chain_mapping(scores), args.close_block, fallback)
    rows = build_rows(scores, mapping, args.winner_uid)
    if args.winner_coldkey and not (rows[0]["coldkey"] or "").startswith(args.winner_coldkey):
        raise SystemExit(f"winner row coldkey {rows[0]['coldkey'][:10]} does not match the paid coldkey {args.winner_coldkey}")
    print(f"epoch {args.epoch}: {len(rows)} UIDs, winner UID {rows[0]['uid']} at {rows[0]['final_score']:.0f}, "
          f"median {rows[len(rows) // 2]['final_score']:.0f}")
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.epoch}.json").write_text(json.dumps({"success": True, "source": "validator_log", "data": rows}))
    note = "public API never published this epoch (validator dashboard upload rejected); rebuilt from the validator log"
    print("stored rows:", store(args.database, args.epoch, rows, note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
