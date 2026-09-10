"""Build a live, score-qualified parent bank from completed public rounds.

Parent sequences are public and are therefore never submitted.  They are used
only as proposal centers.  Selection is based on the validator-published final
score, not file recency or an archive-wide proxy.  Keeping the leading two
sequences from each completed epoch preserves both winners and close runners-up
before the bank is globally ordered for current Layer-2 revalidation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT
from nanobody_opt.simulate_round import METRICS


def _fetch(api_root: str, epoch: int) -> tuple[int, Any | None]:
    url = f"{api_root.rstrip('/')}/api/competitions/leaderboard/{epoch}/nanobodies"
    request = Request(url, headers={"User-Agent": "nova-elite-parent-selector/1"})
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            return epoch, json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return epoch, None


def _epoch_candidates(payload: Any, target: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    candidates = []
    for submission in payload.get("data", []):
        if not isinstance(submission, dict):
            continue
        for nanobody in submission.get("nanobodies", []):
            if not isinstance(nanobody, dict) or nanobody.get("protein_name") != target:
                continue
            sequence = str(nanobody.get("sequence") or "").strip().upper()
            try:
                metrics = {name: float(nanobody[name]) for name in METRICS}
                score = float(nanobody["final_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not sequence
                or not all(residue in "ACDEFGHIKLMNPQRSTVWY" for residue in sequence)
                or not math.isfinite(score)
                or not all(math.isfinite(value) for value in metrics.values())
            ):
                continue
            candidates.append({
                "sequence": sequence,
                "selection_score": score,
                "validator_final_score": score,
                "validator_categories": {
                    "confidence": float(nanobody.get("confidence_rank", math.nan)),
                    "physical_interaction": float(
                        nanobody.get("physical_interaction_rank", math.nan)
                    ),
                    "developability": float(
                        nanobody.get("developability_rank", math.nan)
                    ),
                },
                "metrics": metrics,
            })
    candidates.sort(key=lambda item: (item["validator_final_score"], item["sequence"]))
    return candidates


def select_elite_parents(
    payloads: dict[int, Any],
    *,
    target: str,
    per_epoch: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Select score-leading parents while retaining their epoch provenance."""
    by_sequence: dict[str, dict[str, Any]] = {}
    for epoch in sorted(payloads, reverse=True):
        for epoch_rank, candidate in enumerate(
            _epoch_candidates(payloads[epoch], target)[:per_epoch], start=1
        ):
            sequence = str(candidate["sequence"])
            observation = {
                "epoch": epoch,
                "epoch_rank": epoch_rank,
                "validator_final_score": candidate["validator_final_score"],
            }
            if sequence not in by_sequence:
                by_sequence[sequence] = {
                    **candidate,
                    "parent_source": "public_leaderboard_api",
                    "public_observations": [observation],
                }
            else:
                by_sequence[sequence]["public_observations"].append(observation)
                if candidate["validator_final_score"] < by_sequence[sequence][
                    "validator_final_score"
                ]:
                    observations = by_sequence[sequence]["public_observations"]
                    by_sequence[sequence] = {
                        **candidate,
                        "parent_source": "public_leaderboard_api",
                        "public_observations": observations,
                    }
    ordered = sorted(
        by_sequence.values(),
        key=lambda item: (
            item["validator_final_score"],
            min(entry["epoch_rank"] for entry in item["public_observations"]),
            item["sequence"],
        ),
    )
    return ordered[:limit]


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
    parser.add_argument("--target", required=True)
    parser.add_argument("--before-epoch", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--per-epoch", type=int, default=2)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--scan-limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--api-root", default=LEADERBOARD_API_ROOT)
    args = parser.parse_args()
    for name in ("epochs", "per_epoch", "limit", "scan_limit", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    epochs = range(
        args.before_epoch - 1,
        max(args.before_epoch - args.scan_limit - 1, 0),
        -1,
    )
    payloads: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_fetch, args.api_root, epoch) for epoch in epochs]
        for future in as_completed(futures):
            epoch, payload = future.result()
            if payload is not None and _epoch_candidates(payload, args.target):
                payloads[epoch] = payload
    complete_epochs = sorted(payloads, reverse=True)[: args.epochs]
    selected = select_elite_parents(
        {epoch: payloads[epoch] for epoch in complete_epochs},
        target=args.target,
        per_epoch=args.per_epoch,
        limit=args.limit,
    )
    if not selected:
        raise SystemExit("no complete public nanobody parents were available")
    _atomic_json(args.output, {
        "target": args.target,
        "before_epoch": args.before_epoch,
        "source": args.api_root.rstrip("/"),
        "complete_epochs": complete_epochs,
        "per_epoch": args.per_epoch,
        "sequences": [item["sequence"] for item in selected],
        "candidates": selected,
    })
    print(
        f"ELITE_PARENTS_READY target={args.target} epochs={len(complete_epochs)} "
        f"parents={len(selected)} output={args.output}"
    )
    for item in selected[:10]:
        provenance = item["public_observations"][0]
        print(
            f"  score={item['validator_final_score']:.1f} "
            f"epoch={provenance['epoch']} rank={provenance['epoch_rank']} "
            f"{item['sequence']}"
        )


if __name__ == "__main__":
    main()
