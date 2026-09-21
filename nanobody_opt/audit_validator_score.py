"""Replay one finalized public nanobody score exactly from validator metrics.

This is a post-epoch parity check.  It uses the metric vector and complete
competitor field published by the dashboard API, applies the validator's dense
ranks, and refuses to write a successful audit unless category and total scores
match the API.  A pre-submission local BoltzGen run is intentionally not treated
as the same observation: validator inference is stochastic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT
from nanobody_opt.simulate_round import METRICS, category_rank_sums


CATEGORY_FIELDS = {
    "confidence": "confidence_rank",
    "physical_interaction": "physical_interaction_rank",
    "developability": "developability_rank",
}


def _metrics(nanobody: dict[str, Any]) -> dict[str, float]:
    values = {name: float(nanobody[name]) for name in METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("nanobody metrics contain a non-finite value")
    return values


def _matching_nanobodies(
    payload: Any, *, target: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError("leaderboard API response is not successful")
    found = []
    for submission in payload.get("data", []):
        if not isinstance(submission, dict):
            continue
        for nanobody in submission.get("nanobodies", []):
            if (
                isinstance(nanobody, dict)
                and str(nanobody.get("protein_name")) == target
                and set(METRICS).issubset(nanobody)
            ):
                found.append((submission, nanobody))
    return found


def replay_validator_score(
    payload: Any,
    *,
    epoch: int,
    hotkey: str,
    target: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    submissions = payload.get("data", []) if isinstance(payload, dict) else []
    mine = [item for item in submissions if item.get("hotkey") == hotkey]
    if len(mine) != 1:
        raise ValueError(f"expected one submission for hotkey {hotkey}, found {len(mine)}")
    candidates = [
        item for item in mine[0].get("nanobodies", [])
        if isinstance(item, dict) and (target is None or item.get("protein_name") == target)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one matching nanobody for hotkey {hotkey}, found {len(candidates)}"
        )
    nanobody = candidates[0]
    resolved_target = str(nanobody["protein_name"])
    mine_metrics = _metrics(nanobody)
    mine_hash = str(nanobody.get("sequence_hash") or "")
    mine_sequence = str(nanobody.get("sequence") or "").strip().upper()

    competitor_rows = []
    for _, other in _matching_nanobodies(payload, target=resolved_target):
        other_hash = str(other.get("sequence_hash") or "")
        other_sequence = str(other.get("sequence") or "").strip().upper()
        if (mine_hash and other_hash == mine_hash) or (
            not mine_hash and other_sequence == mine_sequence
        ):
            continue
        competitor_rows.append(_metrics(other))
    if not competitor_rows:
        raise ValueError("leaderboard has no competitor nanobodies")
    field = pd.DataFrame(competitor_rows, columns=list(METRICS))
    replayed_categories = category_rank_sums(mine_metrics, field)
    replayed_total = float(sum(replayed_categories.values()))
    api_categories = {
        category: float(nanobody[field_name])
        for category, field_name in CATEGORY_FIELDS.items()
    }
    api_total = float(nanobody["final_score"])
    if replayed_categories != api_categories or replayed_total != api_total:
        raise ValueError(
            "validator score replay mismatch: "
            f"replayed={replayed_categories}/{replayed_total}, "
            f"api={api_categories}/{api_total}"
        )

    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "exact_match",
        "epoch": int(epoch),
        "target": resolved_target,
        "hotkey": hotkey,
        "sequence": mine_sequence,
        "sequence_hash": mine_hash,
        "field_size": len(competitor_rows) + 1,
        "rank_method": "dense",
        "api": {"categories": api_categories, "final_score": api_total},
        "local_replay": {
            "categories": replayed_categories,
            "final_score": replayed_total,
        },
        "audited_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if checkpoint is not None and isinstance(checkpoint.get("submissions"), list):
        entries = [entry for entry in checkpoint["submissions"] if entry.get("hotkey") == hotkey]
        if len(entries) != 1:
            raise ValueError("expected one receipt for the audited hotkey")
        entry = entries[0]
        sequence_matches = str(entry.get("sequence", "")).upper() == mine_sequence
        if not sequence_matches:
            raise ValueError("snapshot sequence does not match the validator sequence")
        nested = entry.get("checkpoint") or None
        if nested and nested.get("sequences") != [mine_sequence]:
            raise ValueError("nested checkpoint describes a different sequence")
        audit["snapshot_comparison"] = {
            "epoch_matches": checkpoint.get("epoch") == epoch,
            "target_matches": checkpoint.get("target") == resolved_target,
            "sequence_matches": sequence_matches,
            "checkpoint_available": nested is not None,
        }
        if mine[0].get("uid") is not None:
            audit["snapshot_comparison"]["uid_matches"] = entry.get("uid") == mine[0]["uid"]
        audit["prediction_at_commit"] = entry.get("prediction_at_commit")
        if nested and audit["prediction_at_commit"] is None:
            selection = nested.get("selection") or {}
            if selection.get("win_probability") is not None:
                audit["prediction_at_commit"] = {
                    "event": "rank_sum_below_fixed_threshold",
                    "probability": selection.get("win_probability"),
                    "threshold": selection.get("competitive_score_threshold"),
                    "reference": selection.get("reference"),
                    "observation_count": (nested.get("exact") or {}).get("observation_count"),
                }
        checkpoint = nested
    if checkpoint is not None:
        local_metrics = checkpoint.get("exact", {}).get("metrics", {})
        checkpoint_sequence = str((checkpoint.get("sequences") or [""])[0]).upper()
        metric_deltas = {
            name: mine_metrics[name] - float(local_metrics[name])
            for name in METRICS
            if name in local_metrics
        }
        audit["checkpoint_comparison"] = {
            "epoch_matches": int(checkpoint.get("epoch", -1)) == int(epoch),
            "target_matches": str(checkpoint.get("target")) == resolved_target,
            "sequence_matches": checkpoint_sequence == mine_sequence,
            "validator_minus_local_metrics": metric_deltas,
            "semantics": (
                "metric deltas compare independent stochastic observations; the "
                "exact equality guarantee applies to final score replay from the "
                "validator-published metric field"
            ),
        }
    return audit


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--hotkey", required=True)
    parser.add_argument("--target")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-root", default=LEADERBOARD_API_ROOT)
    args = parser.parse_args()

    url = (
        f"{args.api_root.rstrip('/')}/api/competitions/leaderboard/"
        f"{args.epoch}/nanobodies"
    )
    request = Request(url, headers={"User-Agent": "nova-nanobody-auditor/1"})
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    checkpoint = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
    audit = replay_validator_score(
        payload,
        epoch=args.epoch,
        hotkey=args.hotkey,
        target=args.target,
        checkpoint=checkpoint,
    )
    audit["source"] = url
    if args.output:
        _atomic_json(args.output, audit)
        print(f"VALIDATOR_SCORE_EXACT_MATCH {args.epoch}: {args.output}")
    else:
        print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
