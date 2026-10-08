#!/usr/bin/env python3
"""Schedule one adaptive confirmation observation for prospective bank hits.

Run ``select`` after every bank ingest, score the emitted ``*-next.json`` once,
ingest it, and call ``select`` again before spending another pass.  This is the
important boundary: pass three is never committed before pass two is visible.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compound_miner.information_efficient import (  # noqa: E402
    GATE_PASSES,
    NORMAL_SEED_NOISE,
    select_confirmation_candidates,
)
from compound_miner.novelty_gate import default_cache, prefilter  # noqa: E402
from compound_miner.types import Candidate  # noqa: E402
from compound_miner.validity import HistoricalIndex, ValidityGate  # noqa: E402

TOP_N = 20


def _rows(connection: sqlite3.Connection, reaction: int, batch: int, offset: int) -> list[tuple]:
    return connection.execute(
        """SELECT ca.name, ca.smiles, ca.reaction, ca.components, ca.inchikey,
                  ca.heavy_atoms, ca.rotatable_bonds, ca.proxy_mean, ca.proxy_std,
                  ss.observation_count, ss.mean_score, ss.sd_score
           FROM score_summaries ss JOIN candidates ca ON ca.name = ss.name
           WHERE ca.reaction = ?
           ORDER BY ss.mean_score DESC, ca.name ASC LIMIT ? OFFSET ?""",
        (reaction, batch, offset),
    ).fetchall()


def _candidate(row: tuple) -> Candidate:
    return Candidate(
        str(row[0]), str(row[1]), int(row[2]), tuple(json.loads(row[3])), str(row[4]),
        int(row[5]), int(row[6]), float(row[7]), float(row[8]),
    )


def select(args: argparse.Namespace) -> dict:
    started = time.time()
    history = HistoricalIndex(args.historical_molecules)
    gate = ValidityGate(history)
    cache = default_cache(history)
    uri = f"file:{Path(args.bank).resolve()}?mode=ro"
    confirmed: list[dict] = []
    pending: list[dict] = []
    scanned = 0
    offset = 0
    threshold = None
    stop_at = None
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        while offset < args.scan_limit:
            rows = _rows(connection, args.reaction, args.batch, offset)
            if not rows:
                break
            offset += len(rows)
            candidates = [_candidate(row) for row in rows]
            _, kept, _ = prefilter(candidates, history, cache)
            stop_chunk = False
            for index in kept:
                row = rows[index]
                candidate = candidates[index]
                mean = float(row[10])
                passes = int(row[9])
                if stop_at is not None and mean < stop_at:
                    stop_chunk = True
                    break
                if not gate.accepts(candidate, check_similarity=False):
                    continue
                item = {
                    "name": candidate.name,
                    "candidate": candidate,
                    "passes": passes,
                    "mean": mean,
                    "sd": float(row[11] or 0.0),
                    "components": candidate.components,
                }
                if passes >= GATE_PASSES:
                    if len(confirmed) < TOP_N:
                        confirmed.append(item)
                        if len(confirmed) == TOP_N:
                            threshold = mean
                            stop_at = threshold - args.scan_margin
                else:
                    pending.append(item)
            scanned += len(rows)
            if stop_chunk or (stop_at is not None and float(rows[-1][10]) < stop_at):
                break

    if threshold is None:
        if confirmed:
            threshold = min(item["mean"] for item in confirmed)
        elif pending:
            # With no eligible incumbent, learn around the best exact row
            # instead of treating every molecule as an equal open slot.
            threshold = max(item["mean"] for item in pending)

    decisions = [] if threshold is None else select_confirmation_candidates(
        pending,
        threshold=threshold,
        incumbent_components=[item["components"] for item in confirmed[:TOP_N]],
        cap=args.max,
        minimum_probability=args.minimum_probability,
        admission_margin=args.admission_margin,
        noise_floor=args.noise_floor,
        information_fraction=args.information_fraction,
        diversity_fraction=args.diversity_fraction,
    )
    pending_by_name = {item["name"]: item for item in pending}
    chosen = [(decision, pending_by_name[decision.name]) for decision in decisions]
    prefix = Path(args.prefix)
    output = prefix.with_name(f"{prefix.name}-next.json")
    selection_path = prefix.with_name(f"{prefix.name}.selection.json")
    output.unlink(missing_ok=True)
    selection_path.unlink(missing_ok=True)

    reasons = Counter(decision.reason for decision, _ in chosen)
    summary = {
        "event": "adaptive_confirm_selected",
        "reaction": args.reaction,
        "threshold": None if threshold is None else round(threshold, 6),
        "confirmed_top": len(confirmed),
        "pending_seen": len(pending),
        "selected": len(chosen),
        "passes_planned": len(chosen),
        "reasons": dict(reasons),
        "minimum_probability": args.minimum_probability,
        "admission_margin": args.admission_margin,
        "scanned": scanned,
        "seconds": round(time.time() - started, 2),
        "file": str(output) if chosen else None,
    }
    if not chosen:
        return {**summary, "event": "adaptive_confirm", "passes_added": 0, "molecules": []}

    payload = {
        "schema_version": 1,
        "status": "complete",
        "expected_count": len(chosen),
        "kind": f"rxn{args.reaction}-adaptive-confirm-next",
        "policy": "information-efficient-v1",
        "candidates": [],
    }
    molecules = []
    for decision, item in chosen:
        candidate = item["candidate"]
        payload["candidates"].append({
            "name": candidate.name,
            "smiles": candidate.smiles,
            "reaction": candidate.reaction,
            "components": list(candidate.components),
            "inchikey": candidate.inchikey,
            "heavy_atoms": candidate.heavy_atoms,
            "rotatable_bonds": candidate.rotatable_bonds,
            "proxy_mean": candidate.proxy_mean,
            "proxy_std": candidate.proxy_std,
            "confirmation_reason": decision.reason,
            "entry_probability": decision.probability,
            "expected_portfolio_improvement": decision.expected_improvement,
        })
        molecules.append({
            "name": candidate.name,
            "passes": item["passes"],
            "mean": item["mean"],
            "reason": decision.reason,
            "entry_probability": decision.probability,
            "expected_portfolio_improvement": decision.expected_improvement,
            "standard_error": decision.standard_error,
            "component_novelty": decision.component_novelty,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    selection_path.write_text(json.dumps({**summary, "molecules": molecules}, separators=(",", ":"), sort_keys=True) + "\n")
    return summary


def report(args: argparse.Namespace) -> dict:
    prefix = Path(args.prefix)
    selection = json.loads(prefix.with_name(f"{prefix.name}.selection.json").read_text())
    threshold = selection["threshold"]
    uri = f"file:{Path(args.bank).resolve()}?mode=ro"
    molecules = []
    passes_added = 0
    reached = 0
    survived = 0
    with sqlite3.connect(uri, uri=True, timeout=60) as connection:
        for item in selection["molecules"]:
            row = connection.execute(
                "SELECT observation_count, mean_score FROM score_summaries WHERE name=?",
                (item["name"],),
            ).fetchone()
            passes_after, mean_after = (
                (int(row[0]), float(row[1])) if row else (item["passes"], item["mean"])
            )
            added = passes_after - int(item["passes"])
            passes_added += added
            reached += passes_after >= GATE_PASSES
            still = threshold is None or mean_after >= threshold
            survived += passes_after >= GATE_PASSES and still
            molecules.append({
                "name": item["name"],
                "reason": item["reason"],
                "passes": [item["passes"], passes_after],
                "prior_mean": round(float(item["mean"]), 5),
                "updated_mean": round(mean_after, 5),
                "delta": round(mean_after - float(item["mean"]), 5),
                "entry_probability": round(float(item["entry_probability"]), 4),
                "survived": bool(still),
            })
    return {
        "event": "adaptive_confirm",
        "reaction": selection["reaction"],
        "threshold": threshold,
        "selected": len(molecules),
        "reasons": selection.get("reasons") or {},
        "passes_planned": selection["passes_planned"],
        "passes_added": passes_added,
        "reached_gate_passes": reached,
        "survived_threshold": survived,
        "molecules": molecules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "report"))
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--reaction", type=int, default=3)
    parser.add_argument("--historical-molecules", type=Path)
    parser.add_argument("--max", type=int, default=40)
    parser.add_argument("--minimum-probability", type=float, default=0.10)
    parser.add_argument("--admission-margin", type=float, default=0.010)
    parser.add_argument("--noise-floor", type=float, default=NORMAL_SEED_NOISE)
    parser.add_argument("--information-fraction", type=float, default=0.20)
    parser.add_argument("--diversity-fraction", type=float, default=0.15)
    parser.add_argument("--scan-margin", type=float, default=0.040)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--scan-limit", type=int, default=200_000)
    args = parser.parse_args()
    if args.mode == "select":
        if args.historical_molecules is None:
            parser.error("select needs --historical-molecules")
        summary = select(args)
    else:
        summary = report(args)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

