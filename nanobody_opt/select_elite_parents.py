"""Build a live, score-qualified parent bank from completed public rounds.

Parent sequences are public and are therefore never submitted.  They are used
only as proposal centers.  Selection is based on the validator-published final
score, not file recency or an archive-wide proxy.  The primary lane keeps the
leading two sequences from each completed epoch.  A bounded family-neighbour
lane also inspects ranks immediately below them, retaining close sequence
neighbours with diverse terminal residues.  That lane preserves useful local
precursors which a top-two-only policy silently discarded.
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
    neighbor_per_epoch: int = 0,
    neighbor_distance: int = 6,
    neighbor_limit: int = 0,
    terminal_variants_per_residue: int = 3,
) -> list[dict[str, Any]]:
    """Select epoch elites plus bounded, terminally diverse family neighbours."""
    if per_epoch < 1 or limit < 1:
        raise ValueError("per_epoch and limit must be positive")
    if min(
        neighbor_per_epoch,
        neighbor_distance,
        neighbor_limit,
        terminal_variants_per_residue,
    ) < 0:
        raise ValueError("neighbour selection arguments cannot be negative")

    ranked_epochs: dict[int, list[dict[str, Any]]] = {}
    for epoch in sorted(payloads, reverse=True):
        unique: dict[str, dict[str, Any]] = {}
        for candidate in _epoch_candidates(payloads[epoch], target):
            unique.setdefault(str(candidate["sequence"]), candidate)
        ranked_epochs[epoch] = list(unique.values())

    def observation(epoch: int, epoch_rank: int, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "epoch_rank": epoch_rank,
            "validator_final_score": candidate["validator_final_score"],
        }

    def merge(
        destination: dict[str, dict[str, Any]],
        candidate: dict[str, Any],
        *,
        epoch: int,
        epoch_rank: int,
        lane: str,
        nearest_elite_distance: int = 0,
    ) -> None:
        sequence = str(candidate["sequence"])
        seen = destination.get(sequence)
        current_observation = observation(epoch, epoch_rank, candidate)
        if seen is None:
            destination[sequence] = {
                **candidate,
                "parent_source": "public_leaderboard_api",
                "parent_selection_lane": lane,
                "nearest_elite_distance": nearest_elite_distance,
                "public_observations": [current_observation],
            }
            return
        seen["public_observations"].append(current_observation)
        if lane == "epoch_elite":
            seen["parent_selection_lane"] = lane
            seen["nearest_elite_distance"] = 0
        else:
            seen["nearest_elite_distance"] = min(
                int(seen.get("nearest_elite_distance", nearest_elite_distance)),
                nearest_elite_distance,
            )
        if candidate["validator_final_score"] < seen["validator_final_score"]:
            preserved = {
                key: seen[key]
                for key in (
                    "parent_source",
                    "parent_selection_lane",
                    "nearest_elite_distance",
                    "public_observations",
                )
            }
            destination[sequence] = {**candidate, **preserved}

    anchors: dict[str, dict[str, Any]] = {}
    for epoch in sorted(ranked_epochs, reverse=True):
        for epoch_rank, candidate in enumerate(ranked_epochs[epoch][:per_epoch], start=1):
            merge(
                anchors,
                candidate,
                epoch=epoch,
                epoch_rank=epoch_rank,
                lane="epoch_elite",
            )

    ordered_anchors = sorted(
        anchors.values(),
        key=lambda item: (
            item["validator_final_score"],
            min(entry["epoch_rank"] for entry in item["public_observations"]),
            item["sequence"],
        ),
    )
    if neighbor_per_epoch == 0 or neighbor_limit == 0 or len(ordered_anchors) >= limit:
        return ordered_anchors[:limit]

    anchor_sequences = [str(item["sequence"]) for item in ordered_anchors]
    neighbours: dict[str, dict[str, Any]] = {}
    for epoch in sorted(ranked_epochs, reverse=True):
        start = per_epoch
        stop = per_epoch + neighbor_per_epoch
        for epoch_rank, candidate in enumerate(
            ranked_epochs[epoch][start:stop], start=start + 1
        ):
            sequence = str(candidate["sequence"])
            if sequence in anchors:
                merge(
                    anchors,
                    candidate,
                    epoch=epoch,
                    epoch_rank=epoch_rank,
                    lane="epoch_elite",
                )
                continue
            distances = [
                sum(left != right for left, right in zip(sequence, anchor, strict=True))
                for anchor in anchor_sequences
                if len(anchor) == len(sequence)
            ]
            if not distances or min(distances) > neighbor_distance:
                continue
            merge(
                neighbours,
                candidate,
                epoch=epoch,
                epoch_rank=epoch_rank,
                lane="family_neighbor",
                nearest_elite_distance=min(distances),
            )

    # A sequence can be an anchor in one epoch and a lower-ranked observation
    # in another.  Re-sort after merging those observations because the latter
    # may contain its best public score.
    ordered_anchors = sorted(
        anchors.values(),
        key=lambda item: (
            item["validator_final_score"],
            min(entry["epoch_rank"] for entry in item["public_observations"]),
            item["sequence"],
        ),
    )

    neighbour_order = sorted(
        neighbours.values(),
        key=lambda item: (
            item["validator_final_score"],
            item["nearest_elite_distance"],
            min(entry["epoch_rank"] for entry in item["public_observations"]),
            item["sequence"],
        ),
    )
    # Reserve several strong examples for each observed C-terminal residue.
    # UID 2 in epoch 25037 came from precisely such a rank-nine, D-terminal
    # family member and was invisible to the former global score truncation.
    diverse: list[dict[str, Any]] = []
    per_terminal: dict[str, int] = {}
    for item in neighbour_order:
        terminal = str(item["sequence"])[-1]
        if per_terminal.get(terminal, 0) >= terminal_variants_per_residue:
            continue
        diverse.append(item)
        per_terminal[terminal] = per_terminal.get(terminal, 0) + 1
    chosen: list[dict[str, Any]] = []
    chosen_sequences: set[str] = set()
    for item in [*diverse, *neighbour_order]:
        sequence = str(item["sequence"])
        if sequence in chosen_sequences:
            continue
        chosen.append(item)
        chosen_sequences.add(sequence)
        if len(chosen) >= min(neighbor_limit, limit - len(ordered_anchors)):
            break
    return [*ordered_anchors, *chosen][:limit]


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
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--neighbor-per-epoch",
        type=int,
        default=8,
        help="inspect these additional ranks below each epoch's elite lane",
    )
    parser.add_argument("--neighbor-distance", type=int, default=6)
    parser.add_argument("--neighbor-limit", type=int, default=32)
    parser.add_argument("--terminal-variants-per-residue", type=int, default=3)
    parser.add_argument("--scan-limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--api-root", default=LEADERBOARD_API_ROOT)
    args = parser.parse_args()
    for name in ("epochs", "per_epoch", "limit", "scan_limit", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "neighbor_per_epoch",
        "neighbor_distance",
        "neighbor_limit",
        "terminal_variants_per_residue",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")

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
        neighbor_per_epoch=args.neighbor_per_epoch,
        neighbor_distance=args.neighbor_distance,
        neighbor_limit=args.neighbor_limit,
        terminal_variants_per_residue=args.terminal_variants_per_residue,
    )
    if not selected:
        raise SystemExit("no complete public nanobody parents were available")
    _atomic_json(args.output, {
        "target": args.target,
        "before_epoch": args.before_epoch,
        "source": args.api_root.rstrip("/"),
        "complete_epochs": complete_epochs,
        "per_epoch": args.per_epoch,
        "neighbor_per_epoch": args.neighbor_per_epoch,
        "neighbor_distance": args.neighbor_distance,
        "neighbor_limit": args.neighbor_limit,
        "terminal_variants_per_residue": args.terminal_variants_per_residue,
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
