"""Build a live, score-qualified parent bank from completed public rounds.

Parent sequences are public and are therefore never submitted. They are used
only as proposal centers. Eligibility spans every public observation in the
persistent mutation database; recency is provenance, never an exclusion rule.
Selection is based on validator-published score and family coverage. A bounded
family-neighbour lane inspects ranks immediately below the epoch elites and is
reserved capacity even when the global elite anchor set is already large.
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

from nanobody_opt.basins import basin_id
from urllib.request import Request, urlopen

from nanobody_opt.basins import basin_id, rank_basin_evidence
from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT
from nanobody_opt.mutation_network import load_observations, update_observation_store
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
        coldkey = str(submission.get("coldkey") or "")
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
                # Not used by the general elite-parent pool (which stays
                # coldkey-blind on purpose); only priority_coldkey_candidates
                # below reads this, to guarantee a slot for specific,
                # consistently-strong rivals regardless of capacity limits.
                "coldkey": coldkey,
            })
    candidates.sort(key=lambda item: (item["validator_final_score"], item["sequence"]))
    return candidates


def priority_coldkey_candidates(
    payloads: dict[int, Any],
    *,
    target: str,
    priority_coldkeys: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Each priority coldkey's single best-scoring submission across the given
    epochs, as a guaranteed-reserved parent, bypassing capacity competition.

    Exists because a handful of rival coldkeys win far more than the field
    average (one measured at a 38.8% top-10 rate against ~18-20% average),
    yet compete for the same limited parent slots as every other rival --
    routinely losing that competition and taking 1-11 epochs to become one of
    our own parents even after winning. This lane is additive: it never
    removes or displaces anything select_elite_parents already chose.
    """
    if not priority_coldkeys:
        return []
    best: dict[str, dict[str, Any]] = {}
    for epoch in sorted(payloads):
        for candidate in _epoch_candidates(payloads[epoch], target):
            coldkey = candidate.get("coldkey") or ""
            if coldkey not in priority_coldkeys:
                continue
            current = best.get(coldkey)
            if current is None or candidate["validator_final_score"] < current["validator_final_score"]:
                best[coldkey] = {**candidate, "epoch": epoch}
    results = []
    for coldkey in priority_coldkeys:
        candidate = best.get(coldkey)
        if candidate is None:
            continue
        sequence = str(candidate["sequence"])
        results.append({
            "sequence": sequence,
            "selection_score": candidate["validator_final_score"],
            "validator_final_score": candidate["validator_final_score"],
            "metrics": candidate["metrics"],
            "parent_source": "priority_coldkey",
            "parent_selection_lane": "priority_coldkey",
            "nearest_elite_distance": 0,
            "basin_core_id": basin_id(sequence),
            "basin_rank": 0,
            "basin_allocation_reserved": True,
            "public_observations": [{
                "epoch": candidate["epoch"],
                "epoch_rank": None,
                "validator_final_score": candidate["validator_final_score"],
            }],
        })
    return results


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
    recent_epoch_window: int = 16,
    recent_anchor_limit: int = 16,
    recent_neighbor_limit: int = 16,
    basin_core_limit: int = 5,
    basin_parent_limit: int = 32,
    priority_coldkeys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Select a balanced bank of global and recent family representatives.

    Every historical observation remains eligible.  The recent quotas prevent
    globally good but stale families from consuming the complete active bank;
    the remainder is still selected globally so recency never replaces quality.

    priority_coldkeys additively reserves one guaranteed slot per listed
    coldkey for its single best recent submission, on top of (never instead
    of) everything below -- see priority_coldkey_candidates.
    """
    if per_epoch < 1 or limit < 1:
        raise ValueError("per_epoch and limit must be positive")
    if min(
        neighbor_per_epoch,
        neighbor_distance,
        neighbor_limit,
        terminal_variants_per_residue,
        recent_epoch_window,
        recent_anchor_limit,
        recent_neighbor_limit,
        basin_core_limit,
        basin_parent_limit,
    ) < 0:
        raise ValueError("neighbour selection arguments cannot be negative")

    priority = priority_coldkey_candidates(
        payloads, target=target, priority_coldkeys=priority_coldkeys,
    )
    priority_sequences = {item["sequence"] for item in priority}

    def with_priority(selection: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not priority:
            return selection
        rest = [item for item in selection if str(item["sequence"]) not in priority_sequences]
        return [*priority, *rest]

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

    def allocate_by_basin(
        legacy_selection: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reserve parent capacity by strong core, not by individual sequence."""
        if basin_core_limit == 0 or basin_parent_limit == 0:
            return legacy_selection[:limit]
        all_sequences: dict[str, dict[str, Any]] = {}
        evidence_rows: list[dict[str, Any]] = []
        for epoch in sorted(ranked_epochs, reverse=True):
            for epoch_rank, candidate in enumerate(ranked_epochs[epoch], start=1):
                sequence = str(candidate["sequence"])
                evidence_rows.append({
                    "sequence": sequence,
                    "score": candidate["validator_final_score"],
                    "epoch": epoch,
                })
                current = all_sequences.get(sequence)
                entry = observation(epoch, epoch_rank, candidate)
                if current is None:
                    all_sequences[sequence] = {
                        **candidate,
                        "parent_source": "public_leaderboard_api",
                        "parent_selection_lane": "basin_core",
                        "nearest_elite_distance": 0,
                        "public_observations": [entry],
                    }
                else:
                    current["public_observations"].append(entry)
                    if (
                        candidate["validator_final_score"]
                        < current["validator_final_score"]
                    ):
                        preserved = {
                            key: current[key]
                            for key in (
                                "parent_source",
                                "parent_selection_lane",
                                "nearest_elite_distance",
                                "public_observations",
                            )
                        }
                        all_sequences[sequence] = {**candidate, **preserved}

        basin_evidence = rank_basin_evidence(
            evidence_rows,
            recent_epoch_window=recent_epoch_window,
        )
        top_ids = [
            identifier
            for identifier, stats in sorted(
                basin_evidence.items(), key=lambda pair: int(pair[1]["basin_rank"])
            )
            if int(stats["basin_rank"]) <= basin_core_limit
        ]

        def annotate(item: dict[str, Any]) -> dict[str, Any]:
            identifier = basin_id(str(item["sequence"]))
            stats = basin_evidence.get(identifier, {})
            return {**item, "basin_core_id": identifier, **stats}

        grouped: dict[str, list[dict[str, Any]]] = {identifier: [] for identifier in top_ids}
        for item in all_sequences.values():
            identifier = basin_id(str(item["sequence"]))
            if identifier in grouped:
                grouped[identifier].append(annotate(item))
        for identifier in grouped:
            grouped[identifier].sort(key=lambda item: (
                float(item["validator_final_score"]),
                -max(int(entry["epoch"]) for entry in item["public_observations"]),
                str(item["sequence"]),
            ))

        reserved: list[dict[str, Any]] = []
        capacity = min(limit, basin_parent_limit)
        index = 0
        while len(reserved) < capacity:
            added = False
            for identifier in top_ids:
                items = grouped[identifier]
                if index >= len(items):
                    continue
                item = items[index]
                item["basin_allocation_reserved"] = True
                item["parent_history_scope"] = (
                    "recent"
                    if any(
                        int(entry["epoch"])
                        in set(sorted(ranked_epochs, reverse=True)[:recent_epoch_window])
                        for entry in item["public_observations"]
                    )
                    else "global"
                )
                reserved.append(item)
                added = True
                if len(reserved) >= capacity:
                    break
            if not added:
                break
            index += 1

        selected = list(reserved)
        used = {str(item["sequence"]) for item in selected}
        for item in legacy_selection:
            sequence = str(item["sequence"])
            if sequence in used:
                continue
            selected.append(annotate(item))
            used.add(sequence)
            if len(selected) >= limit:
                break
        return selected[:limit]

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
    if neighbor_per_epoch == 0 or neighbor_limit == 0:
        return with_priority(allocate_by_basin(ordered_anchors[:limit]))

    # The former early return here silently disabled this lane as soon as the
    # elite bank itself filled ``limit``. Reserve the requested neighbour
    # capacity first, then backfill it with anchors only when no supported
    # neighbours exist.
    neighbor_capacity = min(neighbor_limit, max(0, limit - 1))
    anchor_capacity = limit - neighbor_capacity
    recent_epochs = set(sorted(ranked_epochs, reverse=True)[:recent_epoch_window])

    def recent_observations(item: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            entry for entry in item["public_observations"]
            if int(entry["epoch"]) in recent_epochs
        ]

    def recent_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Round-robin recent epochs so one prolific round cannot own a quota."""
        chosen: list[dict[str, Any]] = []
        chosen_sequences: set[str] = set()
        maximum_rank = max(
            (
                int(entry["epoch_rank"])
                for item in items
                for entry in recent_observations(item)
            ),
            default=0,
        )
        for epoch_rank in range(1, maximum_rank + 1):
            for epoch in sorted(recent_epochs, reverse=True):
                eligible = [
                    item for item in items
                    if str(item["sequence"]) not in chosen_sequences
                    and any(
                        int(entry["epoch"]) == epoch
                        and int(entry["epoch_rank"]) == epoch_rank
                        for entry in recent_observations(item)
                    )
                ]
                if not eligible:
                    continue
                item = min(
                    eligible,
                    key=lambda value: (
                        value["validator_final_score"], value["sequence"]
                    ),
                )
                chosen.append(item)
                chosen_sequences.add(str(item["sequence"]))
        return chosen

    # Scale the configured quota down for tests/smaller deployments while
    # always leaving at least half of each lane available to global quality.
    recent_anchor_capacity = min(
        recent_anchor_limit,
        (anchor_capacity + 1) // 2,
    )
    recent_anchors = recent_order(ordered_anchors)
    selected_anchors = recent_anchors[:recent_anchor_capacity]
    selected_anchor_sequences = {
        str(item["sequence"]) for item in selected_anchors
    }
    selected_anchors.extend(
        item for item in ordered_anchors
        if str(item["sequence"]) not in selected_anchor_sequences
    )
    selected_anchors = selected_anchors[:anchor_capacity]
    selected_anchor_sequences = {
        str(item["sequence"]) for item in selected_anchors
    }
    for item in selected_anchors:
        item["parent_history_scope"] = (
            "recent" if str(item["sequence"]) in {
                str(value["sequence"])
                for value in recent_anchors[:recent_anchor_capacity]
            } else "global"
        )

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
    chosen: list[dict[str, Any]] = []
    chosen_sequences: set[str] = set()
    per_terminal: dict[str, int] = {}

    def add_neighbours(
        candidates: list[dict[str, Any]], capacity: int, scope: str
    ) -> None:
        if capacity <= 0:
            return
        added = 0
        # Terminal diversity is a first pass, not a hard exclusion: a sparse
        # cohort must still fill its promised recent/global coverage quota.
        for enforce_diversity in (True, False):
            for item in candidates:
                sequence = str(item["sequence"])
                if sequence in chosen_sequences:
                    continue
                terminal = sequence[-1]
                if (
                    enforce_diversity
                    and per_terminal.get(terminal, 0)
                    >= terminal_variants_per_residue
                ):
                    continue
                item["parent_history_scope"] = scope
                chosen.append(item)
                chosen_sequences.add(sequence)
                per_terminal[terminal] = per_terminal.get(terminal, 0) + 1
                added += 1
                if added >= capacity:
                    return

    recent_neighbor_capacity = min(
        recent_neighbor_limit,
        (neighbor_capacity + 1) // 2,
    )
    add_neighbours(
        recent_order(neighbour_order), recent_neighbor_capacity, "recent"
    )
    add_neighbours(
        neighbour_order,
        neighbor_capacity - len(chosen),
        "global",
    )
    selected_sequences = {
        str(item["sequence"]) for item in [*selected_anchors, *chosen]
    }
    backfill = [
        item for item in ordered_anchors
        if str(item["sequence"]) not in selected_sequences
    ]
    for item in backfill:
        item["parent_history_scope"] = "global"
    return with_priority(allocate_by_basin([*selected_anchors, *chosen, *backfill][:limit]))


def _database_payloads(
    path: Path,
    *,
    target: str,
    before_epoch: int,
) -> dict[int, Any]:
    """Rehydrate public validator rows without imposing a recency window."""
    payloads: dict[int, list[dict[str, Any]]] = {}
    for item in load_observations(path, target=target):
        epoch = item.get("epoch")
        if (
            item.get("score_kind") != "validator_final_rank_sum"
            or epoch is None
            or int(epoch) >= before_epoch
        ):
            continue
        metrics = item.get("metrics") or {}
        if any(name not in metrics for name in METRICS):
            continue
        payloads.setdefault(int(epoch), []).append({
            "protein_name": target,
            "sequence": item["sequence"],
            "final_score": item["score"],
            "confidence_rank": math.nan,
            "physical_interaction_rank": math.nan,
            "developability_rank": math.nan,
            **{name: metrics[name] for name in METRICS},
        })
    return {
        epoch: {"success": True, "data": [{"nanobodies": candidates}]}
        for epoch, candidates in payloads.items()
    }


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="completed epochs to retain; zero uses all database/API history",
    )
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
    parser.add_argument("--recent-epoch-window", type=int, default=16)
    parser.add_argument("--recent-anchor-limit", type=int, default=16)
    parser.add_argument("--recent-neighbor-limit", type=int, default=16)
    parser.add_argument("--basin-core-limit", type=int, default=5)
    parser.add_argument("--basin-parent-limit", type=int, default=32)
    parser.add_argument(
        "--priority-coldkeys",
        default="",
        help="comma-separated coldkeys, each guaranteed one parent slot for "
             "its best recent submission regardless of capacity limits; "
             "empty disables the lane entirely",
    )
    parser.add_argument("--scan-limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--api-root", default=LEADERBOARD_API_ROOT)
    parser.add_argument(
        "--observation-database",
        type=Path,
        help="SQLite scored mutation store; all prior public epochs are eligible",
    )
    args = parser.parse_args()
    for name in ("per_epoch", "limit", "scan_limit", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.epochs < 0:
        parser.error("--epochs cannot be negative")
    for name in (
        "neighbor_per_epoch",
        "neighbor_distance",
        "neighbor_limit",
        "terminal_variants_per_residue",
        "recent_epoch_window",
        "recent_anchor_limit",
        "recent_neighbor_limit",
        "basin_core_limit",
        "basin_parent_limit",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative")

    epochs = range(
        args.before_epoch - 1,
        max(args.before_epoch - args.scan_limit - 1, 0),
        -1,
    )
    payloads: dict[int, Any] = {}  # DB read skipped: frontier uses the API only
    api_payloads: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_fetch, args.api_root, epoch) for epoch in epochs]
        for future in as_completed(futures):
            epoch, payload = future.result()
            if payload is not None and _epoch_candidates(payload, args.target):
                payloads[epoch] = payload
                api_payloads[epoch] = payload
    if args.observation_database is not None and api_payloads:
        update_observation_store(
            args.observation_database,
            (
                {
                    "observation_id": (
                        f"public-validator-api:epoch{epoch}:"
                        f"{candidate['sequence']}"
                    ),
                    "sequence": candidate["sequence"],
                    "score": candidate["validator_final_score"],
                    "score_kind": "validator_final_rank_sum",
                    "source": "public_validator_api",
                    "epoch": epoch,
                    "target": args.target,
                    "observation_count": 1,
                    "metrics": candidate["metrics"],
                }
                for epoch, payload in api_payloads.items()
                for candidate in _epoch_candidates(payload, args.target)
                if epoch < args.before_epoch
            ),
        )
    ordered_epochs = sorted(payloads, reverse=True)
    complete_epochs = (
        ordered_epochs[: args.epochs] if args.epochs else ordered_epochs
    )
    # Frontier selection: the competitive frontier (validator_final_score >= 100)
    # instead of the slow dedup-locked basin. The score selection above is
    # skipped (it read the DB + basin allocation, ~3 min); select from the full
    # API pool instead.
    _pool = {}
    for _epoch in complete_epochs:
        for _cand in _epoch_candidates(payloads[_epoch], target=args.target):
            _pool.setdefault(_cand["sequence"], _cand)
    _frontier = sorted(
        (c for c in _pool.values() if c["validator_final_score"] >= 100),
        key=lambda c: (c["validator_final_score"], c["sequence"]),
    )
    selected = _frontier[: args.limit]
    for _cand in selected:
        _cand.setdefault("parent_source", "public_leaderboard_api")
        _cand["parent_selection_lane"] = "frontier"
        _cand.setdefault("nearest_elite_distance", 0)
        _cand.setdefault("public_observations", [{
            "epoch": 0, "epoch_rank": 0,
            "validator_final_score": _cand["validator_final_score"],
        }])
    print(f"frontier selection: pool={len(_pool)} frontier={len(_frontier)} "
          f"selected={len(selected)}", flush=True)
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
        "recent_epoch_window": args.recent_epoch_window,
        "recent_anchor_limit": args.recent_anchor_limit,
        "recent_neighbor_limit": args.recent_neighbor_limit,
        "basin_core_limit": args.basin_core_limit,
        "basin_parent_limit": args.basin_parent_limit,
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
