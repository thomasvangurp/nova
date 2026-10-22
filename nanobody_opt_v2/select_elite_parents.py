"""Build the live parent bank from completed public rounds (frontier path).

Parent sequences are public and are therefore never submitted; they are used
only as proposal centres. The pool is the latest public observation per
sequence from the completed-epoch leaderboard API.

The live bank ranks independent folds against one shared current ten-epoch
field and selects by that multipass mean, with at least ten folds per parent.
Family and coldkey reservations cannot displace stronger measured parents.
Gate failures remain repair-only and occupy at most a quarter of the bank.
Every generated child must pass the exact novelty gate before submission.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from urllib.request import Request, urlopen

from nanobody_opt.basins import basin_id, rank_basin_evidence
from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT
from nanobody_opt.mutation_network import load_observations, update_observation_store
from nanobody_opt.simulate_round import METRICS

#: CDR-only developability "Red Region" bounds (metanano/filters/developability.py,
#: July 2025). Framework edits preserve the CDRs, so a parent whose L/L3/C is
#: out of range can only produce doomed framework-edit children; it must be
#: repaired via CDR3 edits instead of taking normal single-edit seed seats.
DEVELOPABILITY_CDR_RED_REGION = {
    "total_cdr_length": (20.0, 39.0),
    "cdr3_length": (5.0, 23.0),
    "cdr3_compactness": (0.56, 1.61),
}


def _fetch(api_root: str, epoch: int) -> tuple[int, Any | None]:
    url = f"{api_root.rstrip('/')}/api/competitions/leaderboard/{epoch}/nanobodies?limit=10000"
    request = Request(url, headers={"User-Agent": "nova-elite-parent-selector/1"})
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310
            return epoch, json.load(response)
    except HTTPError as error:
        return epoch, {"success": False, "http_status": error.code}
    except (URLError, TimeoutError, json.JSONDecodeError):
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
    newest_epoch: dict[str, int] = {}
    for epoch in sorted(payloads, reverse=True):
        for candidate in _epoch_candidates(payloads[epoch], target):
            coldkey = candidate.get("coldkey") or ""
            # A configured key spec may be a full coldkey or a coldkey prefix
            # (e.g. "5GhREcsK"); match either, so the rival's best recent
            # submission is guaranteed a slot without knowing its full SS58
            # address ahead of time. First matching spec wins.
            keyspec = next(
                (
                    spec for spec in priority_coldkeys
                    if coldkey == spec or coldkey.startswith(spec)
                ),
                None,
            )
            if keyspec is None:
                continue
            # Pick each rival's NEWEST completed epoch first, then the best
            # submission within that epoch. Comparing raw validator_final_score
            # across epochs was wrong: a stale epoch with a lower absolute score
            # would displace the rival's current winning submission.
            if keyspec in newest_epoch and newest_epoch[keyspec] > epoch:
                continue
            if keyspec not in best or candidate["validator_final_score"] < best[keyspec]["validator_final_score"]:
                best[keyspec] = {**candidate, "epoch": epoch}
            newest_epoch[keyspec] = epoch
    results = []
    for keyspec in priority_coldkeys:
        candidate = best.get(keyspec)
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


def _validator_parent_pool(
    pool: list[dict[str, Any]], *, target: str,
) -> list[dict[str, Any]]:
    """Build one fresh HF Algorithm-2 index in metanano per selection.

    Dependency/network diagnostics stay private because they may contain auth
    data. Any failure aborts selection before the parent output is written.
    """
    with tempfile.TemporaryDirectory(prefix="validator-parent-gate-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        sequences = sorted({str(item["sequence"]) for item in pool})
        request.write_text(json.dumps({"target": target, "sequences": sequences}))
        try:
            result = subprocess.run(
                ["/usr/local/bin/pixi", "run", "--manifest-path", "/root/pixi.toml",
                 "--locked", "--environment", "metanano", "python",
                 str(Path(__file__).with_name("validator_parent_gate.py")),
                 "--input", str(request), "--output", str(response)],
                cwd="/root", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=900, check=False,
            )
            if result.returncode:
                raise ValueError("metanano gate failed")
            payload = json.loads(response.read_text())
            rows = payload["results"]
            if (payload["target"] != target or payload["algorithm2_top_count"] != 50
                    or not isinstance(rows, dict) or set(rows) != set(sequences)
                    or any(not isinstance(row, dict)
                           or type(row.get("duplicate")) is not bool
                           or not isinstance(row.get("cdr3"), str)
                           or not isinstance(row.get("identity"), (int, float))
                           or not isinstance(row.get("cdr3_similarity"), (int, float))
                           for row in rows.values())):
                raise ValueError("incomplete gate response")
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            raise SystemExit("validator parent gate unavailable; refusing unsafe seeds") from None
    def _in_top50_basin(row: dict[str, Any]) -> bool:
        # Top-50-basin routing flag. A parent may seed a CDR3-preserving lane
        # (single-edit / framework / basin-terminal / historical singles) only
        # when its clonotype is NOT already in the validator's top-50 basin,
        # because such a lane inherits the blocked CDR3 and every child is then
        # rejected by entry_unique_for_protein_hf (>=0.95 whole-sequence
        # identity, or CDR3 >=0.85 similar to a top-50 CDR3 = same clonotype).
        # The parent gate's is_duplicate may omit the CDR3 check, so re-derive
        # the basin flag from the explicit scalars -- NOT from ``duplicate``
        # alone -- so a CDR3-clonotype parent is never mistaken for clear.
        return bool(
            row["duplicate"]
            or row["identity"] >= 0.95
            or row["cdr3_similarity"] >= 0.85
        )

    return [
        {**item,
         "parent_cdr3": rows[str(item["sequence"])]["cdr3"],
         "parent_validator_duplicate": rows[str(item["sequence"])]["duplicate"],
         "parent_identity": rows[str(item["sequence"])]["identity"],
         "parent_cdr3_similarity": rows[str(item["sequence"])]["cdr3_similarity"],
         "parent_top50_identity_in_basin": _in_top50_basin(rows[str(item["sequence"])])}
        for item in pool
    ]


def _cdr_developability_in_range(profile: dict[str, Any] | None) -> bool:
    """True when the CDR-only red flags (L/L3/C) are all inside their ranges.

    These three are inherited unchanged by framework edits, so they are the
    only ones a parent must satisfy to be a safe single-edit/multi-edit
    framework anchor. PSH/PPC/PNC are full-sequence surface metrics and are
    deliberately NOT enforced here -- they are re-checked on the child at the
    Layer-2 gate.
    """
    if not isinstance(profile, dict):
        return False
    try:
        L = float(profile["total_cdr_length"])
        L3 = float(profile["cdr3_length"])
        C = float(profile["cdr3_compactness"])
    except (KeyError, TypeError, ValueError):
        return False
    lo, hi = DEVELOPABILITY_CDR_RED_REGION["total_cdr_length"]
    if not lo <= L <= hi:
        return False
    lo, hi = DEVELOPABILITY_CDR_RED_REGION["cdr3_length"]
    if not lo <= L3 <= hi:
        return False
    lo, hi = DEVELOPABILITY_CDR_RED_REGION["cdr3_compactness"]
    return lo <= C <= hi


def _developability_parent_pool(
    pool: list[dict[str, Any]], *, target: str, cache_dir: Path,
) -> list[dict[str, Any]]:
    """Annotate each parent with its full six-flag TNP profile (cached).

    Runs the metanano developability service once per parent (the TNP CLI is
    ~2s/uncached sequence, but the per-sequence cache under ``cache_dir`` is
    shared with the Layer-2 gate, so each sequence is profiled exactly once).
    A parent that cannot be profiled fails closed (out-of-range).
    """
    with tempfile.TemporaryDirectory(prefix="developability-parent-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        sequences = sorted({str(item["sequence"]) for item in pool})
        request.write_text(json.dumps({"target": target, "sequences": sequences}))
        try:
            result = subprocess.run(
                ["/usr/local/bin/pixi", "run", "--manifest-path", "/root/pixi.toml",
                 "--locked", "--environment", "metanano", "python",
                 str(Path(__file__).resolve().parents[1] / "nanobody_opt" / "profile_developability.py"),
                 "--input", str(request), "--output", str(response),
                 "--cache-dir", str(cache_dir)],
                cwd="/root", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=3600, check=False,
            )
            if result.returncode:
                raise ValueError("developability profile failed")
            profiles = json.loads(response.read_text()).get("profiles", {})
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            # GAP 5: FAIL CLOSED -- an unmeasured parent must not silently enter
            # the single-edit/framework/lineage lanes (which preserve its CDR3).
            # Route every parent to the repair/multi-edit lane, where a CDR3
            # mismatch is forced, until a profile is available again.
            print(f"SELECT_PARENTS_DEGRADED: developability profiling unavailable "
                  f"({error}); routing all parents to the repair lane", flush=True)
            for item in pool:
                item["parent_developability"] = None
                item["parent_cdr_developability_in_range"] = False
            return pool
    for item in pool:
        profile = profiles.get(str(item["sequence"]))
        item["parent_developability"] = profile
        item["parent_cdr_developability_in_range"] = _cdr_developability_in_range(profile)
    return pool


def _unrelated_family(cdr3: str) -> bool:
    """True when a CDR3 belongs to the historical AIGL/GIGY/TIGL family (~17
    edits from the winner), not the winner's VFLGAHPCPLE / VFLGSHPCPLE core.

    The top miners score by landing ~3 CDR3 edits from the winner clonotype while
    keeping its GAHPCPL core. The unrelated family carries the LTGW/KTGW terminal
    motif without that core, and must not get ordinary seats. Anything else
    (winner-family or an unannotated/synthetic CDR3) is kept -- the dedup gate,
    not this selection, is the authority on it.
    """
    if not cdr3:
        return False
    has_unrelated_tail = "LTGW" in cdr3 or "KTGW" in cdr3
    has_winner_core = "GAHPCPL" in cdr3 or "GSHPCPL" in cdr3
    return has_unrelated_tail and not has_winner_core


def _same_cdr3_family(left: str, right: str) -> bool:
    """Group CDR3s within 15% edit distance; never use whole-VHH identity.

    Length-normalized edit distance handles insertions/deletions as well as
    substitutions. Fresh validator nonduplicate status supplies novelty
    headroom; this additional separation keeps reserved families distinct.
    """
    if not left or not right:
        return False
    previous = list(range(len(right) + 1))
    for index, residue in enumerate(left, 1):
        current = [index]
        for column, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (residue != other)))
        previous = current
    return previous[-1] <= 0.15 * max(len(left), len(right))


def _current_parent_evidence(pool, *, database: Path, target: str, before_epoch: int,
                             minimum_passes: int = 10) -> dict[str, Any]:
    """Re-rank independent stored folds against one shared ten-epoch field.

    Historical rank sums and aggregate ranking rows are not independent folds.
    Scope the indexed history read to this candidate bank before loading metrics.
    """
    import sqlite3
    from nanobody_opt.competitor_lane import OUR_RAW_SOURCES, MULTIPASS_SCORE_KIND
    from nanobody_opt.rank_exact import (
        fetch_recent_fields_from_db, pool_recent_fields, rank_metrics_against_field,
    )
    fields = fetch_recent_fields_from_db(database, target=target, before_epoch=before_epoch, count=10)
    if len(fields) != 10:
        raise ValueError("parent merit ranking requires ten complete reference epochs")
    field = pool_recent_fields(fields)
    sequences = sorted({item["sequence"] for item in pool})
    raw = {}
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        for offset in range(0, len(sequences), 300):
            batch = sequences[offset:offset + 300]
            rows = connection.execute(
                "SELECT n.sequence, o.observation_id, o.source, o.score_kind, o.metrics_json "
                "FROM nodes n CROSS JOIN observations o "
                "WHERE o.node_id=n.id AND n.target=? AND n.sequence IN ("
                + ",".join("?" for _ in batch) + ") AND o.observation_count=1 "
                "AND o.metrics_json IS NOT NULL AND o.metrics_json NOT IN ('', '{}')",
                (target, *batch),
            )
            for sequence, observation_id, source, kind, blob in rows:
                if not (source in OUR_RAW_SOURCES or (
                    source == "public_multipass_rivals" and kind == MULTIPASS_SCORE_KIND
                )):
                    continue
                try:
                    metrics = json.loads(blob)
                    if not all(math.isfinite(float(metrics[name])) for name in METRICS):
                        continue
                except (ValueError, TypeError, KeyError):
                    continue
                raw[f"{sequence}::{observation_id}"] = metrics
    finally:
        connection.close()
    ranks = rank_metrics_against_field(raw, field)
    by_sequence = {}
    for key, score in ranks.items():
        by_sequence.setdefault(key.rsplit("::", 1)[0], []).append(score)
    return {
        "scores": {seq: sum(values) / len(values) for seq, values in by_sequence.items()
                   if len(values) >= minimum_passes},
        "depths": {seq: len(values) for seq, values in by_sequence.items()},
        "field_epochs": [epoch for epoch, _ in fields], "field_rows": len(field),
        "minimum_passes": minimum_passes, "before_epoch": before_epoch,
    }


def _merit_parents(pool, evidence, *, limit: int) -> list[dict[str, Any]]:
    """Select globally by current multipass mean, independent of family or owner.

    Gate failures remain repair-only and use at most a quarter of the bank.
    Every selected parent has the required independent folds; unmeasured parents
    cannot displace measured parents through a family or coldkey reservation.
    """
    measured = []
    for original in pool:
        seq = original["sequence"]
        score = evidence["scores"].get(seq)
        depth = evidence["depths"].get(seq, 0)
        if score is None or not math.isfinite(score) or depth < evidence["minimum_passes"]:
            continue
        item = dict(original)
        repair = not (item.get("parent_validator_duplicate") is False
                      and item.get("parent_top50_identity_in_basin") is False
                      and item.get("parent_cdr_developability_in_range") is True)
        item.update(parent_multipass_mean=score, parent_multipass_observations=depth,
                    parent_multipass_epoch=evidence["before_epoch"],
                    parent_multipass_relative_rank=score / (10 * evidence["field_rows"]),
                    selection_score=score, selection_score_kind="current_multipass_mean",
                    parent_selection_lane="frontier", parent_frontier_lane="frontier_multipass",
                    diversity_reserved=False)
        if repair:
            item.update(parent_validator_duplicate=True, parent_developability_repair=True)
        measured.append((repair, item))
    measured.sort(key=lambda pair: (pair[1]["parent_multipass_mean"], pair[1]["sequence"]))
    ordinary_count = sum(not repair for repair, _ in measured)
    repair_limit = min(limit // 4, ordinary_count // 3)
    selected = []
    repairs = 0
    for repair, item in measured:
        if repair and repairs >= repair_limit:
            continue
        selected.append(item)
        repairs += int(repair)
        if len(selected) >= limit:
            break
    return selected


def _frontier_parents(
    pool: list[dict[str, Any]],
    comparison: dict[str, Any],
    *,
    before_epoch: int,
    limit: int,
    diversity_k: int = 2,
    priority_sequences: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Exploit multipass-ranked parents with a quarter reserved for exploration.

    Nonduplicates are retained first; duplicates are at most a quarter of the bank.
    Unmeasured parents are interleaved using their public epoch percentile.
    """
    from statistics import median

    # Normalize sum-of-ten ranks by ten times the replay field size, not by
    # the selectively measured cohort. Use the latest eligible measurement.
    measured: dict[str, tuple[int, float, float]] = {}
    for epoch_text, result in comparison.items():
        if not str(epoch_text).isdigit() or not isinstance(result, dict):
            continue
        epoch = int(epoch_text)
        method = result.get("method") or {}
        if (epoch >= before_epoch or method.get("formula") != "sum10"
                or method.get("field") != "recent_validator_epochs"
                or (method.get("minimum_passes") or 0) < 10):
            continue
        field_rows = method.get("field_rows")
        if not isinstance(field_rows, (int, float)) or not math.isfinite(field_rows) or field_rows <= 0:
            continue
        values = {
            seq: float(value) for seq, value in (result.get("sequences") or {}).items()
            if isinstance(value, (int, float)) and math.isfinite(value)
        }
        for sequence, value in values.items():
            if sequence not in measured or epoch > measured[sequence][0]:
                measured[sequence] = (epoch, value / (10 * field_rows), value)

    families: dict[str, list[float]] = {}
    for item in pool:
        sequence = str(item["sequence"])
        loop = item["parent_cdr3"]
        if sequence in measured:
            epoch, relative_rank, mean = measured[sequence]
            item.update(parent_multipass_epoch=epoch,
                        parent_multipass_relative_rank=relative_rank,
                        parent_multipass_mean=mean)
            families.setdefault(loop or sequence, []).append(relative_rank)
    family_medians = {loop: median(values) for loop, values in families.items()}

    def quality(item: dict[str, Any]) -> tuple[Any, ...]:
        return (measured.get(str(item["sequence"]), (0, math.inf, 0))[1],
                family_medians.get(item["parent_cdr3"] or item["sequence"], math.inf),
                item["validator_final_score"], item["sequence"])

    ranked = sorted(pool, key=quality)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    def add(item: dict[str, Any], lane: str) -> None:
        if item["sequence"] not in used and len(selected) < limit:
            item["parent_selection_lane"] = "frontier"
            item["parent_frontier_lane"] = lane
            selected.append(item)
            used.add(item["sequence"])

    unmeasured = [item for item in pool if str(item["sequence"]) not in measured]
    def public_quality(item: dict[str, Any]) -> tuple[Any, ...]:
        observation = (item.get("public_observations") or [{}])[0]
        return (item.get("public_epoch_percentile", math.inf),
                -int(observation.get("epoch", 0)),
                observation.get("epoch_rank", math.inf), item["sequence"])
    unmeasured.sort(key=public_quality)
    exploitation = [item for item in ranked if str(item["sequence"]) in measured]
    exploration = min(len(unmeasured), max(1, (limit + 3) // 4)) if limit > 1 else 0
    explored = 0
    for position in range(1, limit + 1):
        explore_now = explored < (position * exploration // limit)
        if unmeasured and (explore_now or not exploitation):
            add(unmeasured.pop(0), "frontier_exploration")
            explored += 1
        elif exploitation:
            add(exploitation.pop(0), "frontier_multipass")
        else:
            break
    # Refill from the FULL gated pool, before imposing the actual-bank cap.
    ordered = selected + [item for item in ranked if item["sequence"] not in used]
    nondup = [item for item in ordered if item.get("parent_validator_duplicate") is False]
    duplicate = [item for item in ordered if item.get("parent_validator_duplicate") is True]
    # Developability hoist: framework-edit children inherit the CDR-only red
    # flags (L/L3/C) unchanged, so an out-of-range parent can only produce
    # doomed single-edits. Route it to the multi-edit lane (repair pool) where
    # CDR3 edits can move C back into range.
    nondup_in_range = [
        item for item in nondup
        if item.get("parent_cdr_developability_in_range", False)
    ]
    # GAP 2: stamp ALL out-of-range parents (nondup AND already-novelty-dup)
    # as repair-only, so both routes keep a repair seat and no already-dup
    # out-of-range parent is silently dropped from the multi-edit lane.
    nondup_repair = [
        {**item,
         "parent_validator_duplicate": True,
         "parent_developability_repair": True}
        for item in nondup
        if not item.get("parent_cdr_developability_in_range", False)
    ]
    duplicate = [
        {**item, "parent_developability_repair": True}
        if not item.get("parent_cdr_developability_in_range", False)
        else item
        for item in duplicate
    ]
    nondup = nondup_in_range
    duplicate = duplicate + nondup_repair

    # Min-multipass anchor (2026-10-21): the parent with the LOWEST
    # parent_multipass_mean leads the bank, regardless of duplicate status. A
    # duplicate parent (identity >= 0.95) cannot be single-edited (its children
    # are 100% duplicates) but its 2-3-edit children cross back below the
    # threshold, so it is the best MULTI-edit anchor. Non-duplicate parents stay
    # in the bank for single-editing. Child safety is established by the
    # unchanged commit-time gate, never by the parent's duplicate status.
    anchors: list[dict[str, Any]] = []
    measured_items = [
        item for item in pool
        if item.get("parent_multipass_mean") is not None
    ]
    if measured_items:
        anchors = [
            min(measured_items,
                key=lambda item: (item["parent_multipass_mean"], item["sequence"]))
        ]
    anchor_sequences = {item["sequence"] for item in anchors}
    # Priority coldkey submissions get guaranteed head slots, additively.
    priority_items = [
        item for item in nondup
        if item["sequence"] in priority_sequences
    ]
    head_sequences = anchor_sequences | {item["sequence"] for item in priority_items}
    head_items = [item for item in nondup if item["sequence"] in head_sequences]
    head_set = set(head_sequences)

    # --- Diversified clonotype allocation (2026-10-21 redesign) ---
    # The previous logic grabbed the min-multipass anchor's ENTIRE CDR3 family
    # into `family_sequences`, packing the bank with the winner's clonotype
    # (VFLGAHPCPLE) -- every single-edit of which is a top-50 duplicate. Instead
    # allocate ordinary seats round-robin across DISTINCT clonotype clusters so
    # several novel CDR3s each hold seats. A cluster whose CDR3 is in the top-50
    # basin (parent_top50_identity_in_basin) gets NO ordinary seat (repair-only).
    cluster_keys: list[str] = []
    clusters: list[list[dict[str, Any]]] = []
    for item in nondup:
        if item["sequence"] in head_set:
            continue
        if item.get("parent_top50_identity_in_basin") is True:
            continue  # blocked clonotype: no ordinary single-edit seat
        loop = item.get("parent_cdr3") or item["sequence"]
        placed = False
        for idx, key in enumerate(cluster_keys):
            if _same_cdr3_family(loop, key):
                clusters[idx].append(item)
                placed = True
                break
        if not placed:
            cluster_keys.append(loop)
            clusters.append([item])
    # Within each cluster, best (min-multipass) member first.
    for cluster in clusters:
        cluster.sort(key=lambda i: (
            i.get("parent_multipass_mean") is None,
            i.get("parent_multipass_mean", math.inf),
            str(i["sequence"]),
        ))

    # A small secondary historical diversity reserve, capped at 10% of the bank.
    reserve = []
    loops = set()
    historical_budget = min(diversity_k, limit // 10)
    cluster_sequences = {i["sequence"] for c in clusters for i in c}
    for item in nondup:
        if len(reserve) >= historical_budget:
            break
        if not item.get("historical_reservoir", False):
            continue
        if item["sequence"] in head_sequences or item["sequence"] in cluster_sequences:
            continue
        loop = item.get("parent_cdr3")
        if loop and not any(_same_cdr3_family(loop, other) for other in loops):
            reserve.append(item)
            loops.add(loop)
    reserved = {item["sequence"] for item in reserve}

    # TWEAK 1: bound ordinary seats to the winner-family. The top miners score
    # by landing ~3 CDR3 edits from VFLGAHPCPLE/VFLGSHPCPLE while keeping the
    # core motif; the unrelated AIGL/GIGY/TIGL families (~17 edits away) are
    # historical, low-scoring, and over-diversify the bank -- no ordinary seat.
    winner_clusters = [
        cluster for key, cluster in zip(cluster_keys, clusters)
        if not _unrelated_family(key)
    ]

    # Round-robin across WINNER-FAMILY clusters only: one parent per distinct
    # winner clonotype per pass, so no single clonotype dominates and the bank
    # spans the ~5-11 clonotypes in the winner's 2-3-edit neighborhood.
    diversified: list[dict[str, Any]] = []
    while any(winner_clusters) and len(head_items) + len(diversified) + len(reserve) < limit:
        progressed = False
        for cluster in winner_clusters:
            if not cluster:
                continue
            if len(head_items) + len(diversified) + len(reserve) >= limit:
                break
            diversified.append(cluster.pop(0))
            progressed = True
        if not progressed:
            break
    selected = (head_items + diversified + reserve)[:limit]
    repair_dups = [item for item in duplicate if item.get("parent_developability_repair")]
    regular_dups = [item for item in duplicate if not item.get("parent_developability_repair")]
    # GAP 2d: the multi-edit CDR3-repair lane must always keep at least one
    # seat, even when the in-range non-dup bank fills the limit -- otherwise
    # `len(selected)//3` truncates to zero and repair is silently dropped.
    if repair_dups and len(selected) >= limit:
        selected = selected[: limit - 1]
    # 3*dup <= nondup is equivalent to dup <= 25% of the ACTUAL bank.
    dup_budget = min(
        limit - len(selected),
        max(len(selected) // 3, 1 if repair_dups else 0),
    )
    selected += (repair_dups + regular_dups)[:dup_budget]
    for item in selected:
        item["parent_selection_lane"] = "frontier"
        item.setdefault("parent_frontier_lane", "frontier_nonduplicate_refill")
        item["diversity_reserved"] = item["sequence"] in reserved
    # GAP 2: the 3:1 dup cap applies to NOVELTY duplicates only. Repair parents
    # (out-of-range CDR, stamped duplicate to reach the multi-edit lane) are not
    # novelty duplicates, so a tiny in-range bank must not fail the ratio and
    # drop the reserved repair seat.
    novelty_dups = sum(
        item.get("parent_validator_duplicate") is True
        and not item.get("parent_developability_repair")
        for item in selected
    )
    nondups = sum(item.get("parent_validator_duplicate") is False for item in selected)
    assert 3 * novelty_dups <= nondups
    # The consumer sorts selection_score and accepts only lane="frontier".
    for rank, item in enumerate(selected, start=1):
        item["selection_score"] = float(rank)
        item["selection_score_kind"] = "frontier_policy_rank"
    return selected


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


def _parser() -> argparse.ArgumentParser:
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
        default="5GhREcsK",
        help="legacy compatibility option; current merit ranking reserves no coldkey slots",
    )
    parser.add_argument("--diversity-k", type=int, default=2)
    parser.add_argument("--historical-reservoir", type=Path,
                        default=Path("/root/artifacts/agreedfix_historical_parents.json"))
    parser.add_argument("--scan-limit", type=int, default=40)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--api-root", default=LEADERBOARD_API_ROOT)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/root/layer2_cache"),
        help="shared TNP developability cache (same dir filter_layer2 uses)",
    )
    parser.add_argument(
        "--observation-database",
        type=Path,
        help="SQLite scored mutation store; all prior public epochs are eligible",
    )
    return parser


def main() -> None:
    parser = _parser()
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
        futures = [
            executor.submit(_fetch, args.api_root, epoch)
            for epoch in epochs
        ]
        for future in as_completed(futures):
            epoch, payload = future.result()
            if (isinstance(payload, dict) and payload.get("success") is True
                    and isinstance(payload.get("data"), list)
                    and _epoch_candidates(payload, args.target)):
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
    # Latest public observation per sequence, with real provenance.
    _pool = {}
    for _epoch in ordered_epochs:
        epoch_candidates = _epoch_candidates(payloads[_epoch], target=args.target)
        for _rank, _cand in enumerate(epoch_candidates, start=1):
            _cand["public_epoch_percentile"] = _rank / len(epoch_candidates)
            _cand["public_observations"] = [{
                "epoch": _epoch, "epoch_rank": _rank,
                "validator_final_score": _cand["validator_final_score"],
            }]
            _pool.setdefault(_cand["sequence"], _cand)
    # Reservoir carries public parent evidence, never the 1733-row merged pool.
    reservoir = json.loads(args.historical_reservoir.read_text())
    if reservoir.get("target") in (None, args.target):
        for candidate in reservoir.get("candidates", []):
            sequence = str(candidate["sequence"])
            item = _pool.setdefault(sequence, dict(candidate))
            item["historical_reservoir"] = True
    else:
        print(
            "SELECT_PARENTS_DEGRADED: historical parent reservoir target "
            f"{reservoir.get('target')!r} != {args.target!r}; skipping reservoir",
            flush=True,
        )
    if not _pool:
        print(
            "SELECT_PARENTS_DEGRADED: no public or reservoir parents available; "
            "writing marked-empty bank",
            flush=True,
        )
        _atomic_json(args.output, {
            "target": args.target,
            "before_epoch": args.before_epoch,
            "source": args.api_root.rstrip("/"),
            "complete_epochs": [],
            "selection_policy": "agreedfix-2026-10-20-complete+developability-hoist",
            "gated_pool_size": 0,
            "nonduplicate_count": 0,
            "duplicate_count": 0,
            "sequences": [],
            "candidates": [],
        })
        print(f"ELITE_PARENTS_READY target={args.target} parents=0 output={args.output}")
        return
    if args.observation_database is None:
        raise SystemExit("current parent merit ranking requires an observation database")
    evidence = _current_parent_evidence(
        list(_pool.values()), database=args.observation_database,
        target=args.target, before_epoch=args.before_epoch,
    )
    measured_pool = [item for item in _pool.values() if item["sequence"] in evidence["scores"]]
    if not measured_pool:
        raise SystemExit("no parents have ten independent folds on the current field")
    eligible_pool = _validator_parent_pool(measured_pool, target=args.target)
    eligible_pool = _developability_parent_pool(
        eligible_pool, target=args.target, cache_dir=args.cache_dir,
    )
    selected = _merit_parents(eligible_pool, evidence, limit=max(args.limit, 155))
    _atomic_json(args.output.with_suffix(".ranking.json"), {
        "policy": "current_field_multipass_merit", "before_epoch": args.before_epoch,
        "field_epochs": evidence["field_epochs"], "field_rows": evidence["field_rows"],
        "minimum_passes": evidence["minimum_passes"], "eligible_pool": len(eligible_pool),
        "measured_pool": len(evidence["scores"]), "selected": len(selected),
        "parents": [{"sequence": c["sequence"], "mean": c["parent_multipass_mean"],
                     "passes": c["parent_multipass_observations"]} for c in selected],
    })
    for _cand in selected:
        _cand.setdefault("parent_source", "public_leaderboard_api")
        _cand.setdefault("nearest_elite_distance", 0)
        _cand.setdefault("public_observations", [{
            "epoch": 0, "epoch_rank": 0,
            "validator_final_score": _cand["validator_final_score"],
        }])
    if not selected:
        print(
            "SELECT_PARENTS_DEGRADED: selection empty; writing marked-empty bank",
            flush=True,
        )
        _atomic_json(args.output, {
            "target": args.target,
            "before_epoch": args.before_epoch,
            "source": args.api_root.rstrip("/"),
            "complete_epochs": ordered_epochs,
            "selection_policy": "agreedfix-2026-10-20-complete+developability-hoist",
            "gated_pool_size": len(eligible_pool),
            "nonduplicate_count": 0,
            "duplicate_count": 0,
            "sequences": [],
            "candidates": [],
        })
        print(f"ELITE_PARENTS_READY target={args.target} parents=0 output={args.output}")
        return
    print(f"frontier selection: pool={len(_pool)} selected={len(selected)} "
          f"multipass={sum('parent_multipass_mean' in c for c in selected)} "
          f"multipass_range={min(c['parent_multipass_mean'] for c in selected):.1f}.."
          f"{max(c['parent_multipass_mean'] for c in selected):.1f}", flush=True)
    _atomic_json(args.output, {
        "target": args.target,
        "before_epoch": args.before_epoch,
        "source": args.api_root.rstrip("/"),
        "complete_epochs": ordered_epochs,
        "parent_ranking_policy": "current_field_multipass_merit",
        "parent_reference_epochs": evidence["field_epochs"],
        "diversity_k": args.diversity_k,
        "historical_reservoir": str(args.historical_reservoir),
        "selection_policy": "agreedfix-2026-10-20-complete+developability-hoist",
        "gated_pool_size": len(eligible_pool),
        "nonduplicate_count": sum(c["parent_validator_duplicate"] is False for c in selected),
        "duplicate_count": sum(c["parent_validator_duplicate"] is True for c in selected),
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
            f"  multipass={item['parent_multipass_mean']:.1f} "
            f"epoch={provenance['epoch']} rank={provenance['epoch_rank']} "
            f"{item['sequence']}"
        )


if __name__ == "__main__":
    main()
