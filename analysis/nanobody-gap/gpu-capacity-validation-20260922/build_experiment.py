#!/usr/bin/env python3
"""Freeze equal proposal arms for the GPU-capacity validation experiment."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
GAP = HERE.parent
BASELINE = HERE / "input" / "current_production_candidates.json"
CLOSURE = GAP / "hamming2-fast-20260922" / "manifest.json"
TARGET = "P20809"
ARM_SIZE = 288
PUBLIC_DB = GAP / "mutations.sqlite"
DASHBOARD_DB = HERE.parents[2] / "data" / "nanobody_dash.sqlite"


def balanced_closure(manifest: dict, side: str, limit: int) -> list[str]:
    """Take an unscored, basin-balanced prefix of one closure side."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_by_group: dict[str, set[str]] = defaultdict(set)
    for case in manifest["cases"]:
        sequence = str(case["sequence"])
        for source in case["provenance"]:
            if source.get("anchor_side") != side:
                continue
            edit_count = int(source.get("edit_count", 0))
            if edit_count == 0:
                continue
            group = str(source["group"])
            if sequence not in seen_by_group[group]:
                grouped[group].append((edit_count, sequence))
                seen_by_group[group].add(sequence)
    for group in grouped:
        grouped[group].sort(key=lambda item: (item[0], item[1]))
    selected: list[str] = []
    selected_set: set[str] = set()
    offsets = defaultdict(int)
    groups = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for group in groups:
            rows = grouped[group]
            while offsets[group] < len(rows):
                sequence = rows[offsets[group]][1]
                offsets[group] += 1
                if sequence in selected_set:
                    continue
                selected.append(sequence)
                selected_set.add(sequence)
                progressed = True
                break
            if len(selected) == limit:
                break
        if not progressed:
            break
    if len(selected) != limit:
        raise RuntimeError(f"{side} closure supplied {len(selected)}, need {limit}")
    return selected


def main() -> None:
    baseline_payload = json.loads(BASELINE.read_text())
    closure = json.loads(CLOSURE.read_text())
    baseline = list(dict.fromkeys(map(str, baseline_payload["sequences"])))[:ARM_SIZE]
    if len(baseline) != ARM_SIZE:
        raise RuntimeError("production arm is too small")
    arms = {
        "current_production": baseline,
        "ours_hamming2": balanced_closure(closure, "ours", ARM_SIZE),
        "rival_hamming2": balanced_closure(closure, "rival", ARM_SIZE),
    }

    provenance: dict[str, list[dict]] = defaultdict(list)
    anchors: list[str] = []
    anchor_seen: set[str] = set()
    for case in closure["cases"]:
        sequence = str(case["sequence"])
        provenance[sequence].extend(case["provenance"])
        if any(
            source.get("group") != "easynano-three-way"
            and int(source.get("edit_count", -1)) == 0
            for source in case["provenance"]
        ) and sequence not in anchor_seen:
            anchors.append(sequence)
            anchor_seen.add(sequence)
    if len(anchors) != 24:
        raise RuntimeError(f"expected 24 anchors, found {len(anchors)}")

    membership: dict[str, list[str]] = defaultdict(list)
    for arm, sequences in arms.items():
        for sequence in sequences:
            membership[sequence].append(arm)
    ordered = list(dict.fromkeys([*baseline, *arms["ours_hamming2"], *arms["rival_hamming2"], *anchors]))
    connection = sqlite3.connect(PUBLIC_DB)
    public_sequences = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT DISTINCT n.sequence
            FROM observations o JOIN nodes n ON n.id=o.node_id
            WHERE o.source='public_validator_api' AND n.target=?
            """,
            (TARGET,),
        )
    }
    public_snapshot = connection.execute(
        """
        SELECT MIN(epoch), MAX(epoch), COUNT(DISTINCT epoch)
        FROM observations
        WHERE source='public_validator_api'
        """
    ).fetchone()
    connection.close()
    dashboard = sqlite3.connect(DASHBOARD_DB)
    public_sequences.update(
        str(row[0])
        for row in dashboard.execute(
            "SELECT DISTINCT sequence FROM entries WHERE sequence IS NOT NULL"
        )
    )
    dashboard_snapshot = dashboard.execute(
        "SELECT MIN(epoch), MAX(epoch), COUNT(DISTINCT epoch) FROM entries"
    ).fetchone()
    dashboard.close()
    cases = []
    for sequence in ordered:
        arm_groups: dict[str, list[str]] = defaultdict(list)
        for source in provenance.get(sequence, []):
            side = source.get("anchor_side")
            if side == "ours" and sequence in arms["ours_hamming2"]:
                arm_groups["ours_hamming2"].append(str(source["group"]))
            if side == "rival" and sequence in arms["rival_hamming2"]:
                arm_groups["rival_hamming2"].append(str(source["group"]))
        cases.append({
            "sequence": sequence,
            "arms": membership.get(sequence, []),
            "control_anchor": sequence in anchor_seen,
            "known_public": sequence in public_sequences,
            "arm_groups": {key: sorted(set(value)) for key, value in arm_groups.items()},
            "closure_provenance": provenance.get(sequence, []),
        })

    preregistration = {
        "proposal_arm_size": ARM_SIZE,
        "discovery_passes": 4,
        "shortlist_proposals_per_arm": 24,
        "anchors_always_retained": 24,
        "confirmation_passes": 20,
        "confirmation_selection_prefixes": [1, 2, 4, 8, 12],
        "confirmation_evaluation_passes": list(range(13, 21)),
        "primary_metric": "complete ten-metric dense-rank sum",
        "primary_gate": (
            "a challenger arm selected on passes 1-12 must beat the current-production "
            "arm on at least 5 of 8 untouched passes"
        ),
        "basin_gate": (
            "rival closure must beat ours in at least 8 of 12 matched basins on "
            "the untouched confirmation passes"
        ),
        "capacity_decision": (
            "adopt the smallest repeat depth whose untouched performance is not "
            "improved by the next measured depth"
        ),
    }
    manifest = {
        "target": TARGET,
        "experiment": "equal-budget proposal arms plus frozen-shortlist confirmation",
        "source_epoch": 25274,
        "baseline_source_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "closure_source_sha256": hashlib.sha256(CLOSURE.read_bytes()).hexdigest(),
        "arm_size": ARM_SIZE,
        "arm_counts": {key: len(value) for key, value in arms.items()},
        "anchor_count": len(anchors),
        "unique_sequence_count": len(ordered),
        "known_public_sequence_count": sum(
            sequence in public_sequences for sequence in ordered
        ),
        "public_novelty_snapshot": {
            "mutation_graph": {
                "minimum_epoch": public_snapshot[0],
                "maximum_epoch": public_snapshot[1],
                "epoch_count": public_snapshot[2],
            },
            "leaderboard": {
                "minimum_epoch": dashboard_snapshot[0],
                "maximum_epoch": dashboard_snapshot[1],
                "epoch_count": dashboard_snapshot[2],
            },
        },
        "preregistration": preregistration,
        "cases": cases,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "sequences.json").write_text(json.dumps({
        "target": TARGET, "sequences": ordered,
    }, indent=2) + "\n")
    print(json.dumps({
        "arms": manifest["arm_counts"],
        "anchors": len(anchors),
        "unique": len(ordered),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
