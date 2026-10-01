#!/usr/bin/env python3
"""Freeze the H2 selector proof before draws 9-16 exist."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanobody_opt.rank_exact import LEADERBOARD_API_ROOT, leaderboard_field

from selector import (
    FLEET_NONINFERIORITY_METRICS,
    FLEET_SIZE,
    METRICS,
    TARGET,
    current_selector,
    hard_gate_thresholds,
    maximum_independent_capacity,
    passes_hard_gates,
    repaired_selector,
)


HERE = Path(__file__).resolve().parent
GAP = HERE.parent
SOURCE = GAP / "gpu-capacity-validation-20260922"
HOLDOUT = GAP / "holdout-recall-20260922"
DB = GAP / "mutations.sqlite"
DEV_EPOCHS = tuple(range(25231, 25253))
RESERVED_EPOCHS = tuple(range(25261, 25269))
CODE_FILES = (
    "selector.py",
    "build_preregistration.py",
    "analyze_fresh.py",
    "run_remote.sh",
    "timing_watch.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_draw(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    if payload.get("scorer_protocol") != "validator-dual-sample-v1":
        raise RuntimeError(f"wrong development protocol: {path}")
    found = {}
    for by_sequence in payload["per_nanobody_components"].values():
        for sequence, by_target in by_sequence.items():
            metrics = by_target[TARGET]
            found[sequence] = {metric: float(metrics[metric]) for metric in METRICS}
    if len(found) != 888:
        raise RuntimeError(f"development draw is not full cohort: {path}")
    return found


def development_fields() -> list[dict[str, Any]]:
    connection = sqlite3.connect(DB)
    rows = connection.execute(
        """
        SELECT o.epoch, n.sequence, o.metrics_json
        FROM observations o JOIN nodes n ON n.id=o.node_id
        WHERE o.source='public_validator_api' AND n.target=?
          AND o.epoch BETWEEN ? AND ?
        ORDER BY o.epoch, n.sequence
        """,
        (TARGET, DEV_EPOCHS[0], DEV_EPOCHS[-1]),
    ).fetchall()
    connection.close()
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for epoch, _, raw in rows:
        metrics = json.loads(raw)
        if all(metric in metrics for metric in METRICS):
            grouped[int(epoch)].append({
                metric: float(metrics[metric]) for metric in METRICS
            })
    fields = [
        {"epoch": epoch, "records": grouped[epoch]} for epoch in DEV_EPOCHS
    ]
    if any(len(field["records"]) < 30 for field in fields):
        raise RuntimeError("development field has fewer than 30 complete records")
    return fields


def fetch_reserved_field(epoch: int) -> dict[str, Any]:
    url = (
        f"{LEADERBOARD_API_ROOT}/api/competitions/leaderboard/"
        f"{epoch}/nanobodies"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "nova-offline-selector-proof/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    frame = leaderboard_field(payload, TARGET)
    if len(frame) < 30:
        raise RuntimeError(f"reserved epoch {epoch} has only {len(frame)} records")
    records = [
        {metric: float(row[metric]) for metric in METRICS}
        for row in frame.to_dict(orient="records")
    ]
    return {"epoch": epoch, "target": TARGET, "records": records}


def noninferiority_margins(
    fleet: list[str], draws: list[dict[str, dict[str, float]]]
) -> dict[str, float]:
    margins = {}
    for metric in FLEET_NONINFERIORITY_METRICS:
        by_draw = [
            statistics.median(draw[sequence][metric] for sequence in fleet)
            for draw in draws
        ]
        center = statistics.median(by_draw)
        margins[metric] = statistics.median(abs(value - center) for value in by_draw)
    return margins


def main() -> None:
    for draw in range(9, 17):
        if (HERE / f"fresh_{draw:02d}" / "exact_all.json").exists():
            raise RuntimeError("future draw already exists before preregistration")
    draw_paths = [
        SOURCE / f"discovery_{draw:02d}" / "exact_all.json"
        for draw in range(1, 5)
    ] + [
        HOLDOUT / f"heldout_{draw:02d}" / "exact_all.json"
        for draw in range(5, 9)
    ]
    draws = [load_draw(path) for path in draw_paths]
    source_manifest_path = SOURCE / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    metadata = {case["sequence"]: case for case in source_manifest["cases"]}
    ours = [
        case["sequence"] for case in source_manifest["cases"]
        if "ours_hamming2" in case["arms"] and not case.get("known_public", False)
    ]
    current_all = [
        case["sequence"] for case in source_manifest["cases"]
        if "current_production" in case["arms"] and not case.get("known_public", False)
    ]
    current = sorted(
        current_all,
        key=lambda sequence: (hashlib.sha256(sequence.encode()).hexdigest(), sequence),
    )[:len(ours)]
    if len(ours) != 282 or len(current) != 282:
        raise RuntimeError("equal proposal budget is not 282 per arm")
    fields = development_fields()
    write_json(HERE / "development_fields.json", fields)
    gates = hard_gate_thresholds(current, draws)
    gated_ours = [
        sequence for sequence in ours if passes_hard_gates(sequence, draws, gates)
    ]
    capacity = {
        "A_current": maximum_independent_capacity(current),
        "B_ours_h2": maximum_independent_capacity(ours),
        "C_ours_h2_after_catastrophic_gates": maximum_independent_capacity(gated_ours),
    }
    common_capacity = min(record["capacity"] for record in capacity.values())
    if common_capacity < FLEET_SIZE:
        raise RuntimeError("preregistered eight-ticket fleet is infeasible")
    arm_a = current_selector(current, draws, fields, fleet_size=FLEET_SIZE)
    arm_b = current_selector(ours, draws, fields, fleet_size=FLEET_SIZE)
    arm_c = repaired_selector(
        ours, draws, fields, metadata, gates, fleet_size=FLEET_SIZE
    )
    if any(len(arm["fleet"]) != FLEET_SIZE for arm in (arm_a, arm_b, arm_c)):
        raise RuntimeError("selector did not freeze equal-size fleets")
    pools = {
        "A_current": current,
        "B_and_C_ours_h2": ours,
        "selection_rule": (
            "all 282 novel ours-H2 candidates; lowest SHA256 282 of 288 "
            "current candidates, chosen without performance data"
        ),
    }
    write_json(HERE / "pools.json", pools)
    selected = {
        "A": arm_a,
        "B": arm_b,
        "C": arm_c,
    }
    write_json(HERE / "selected_fleets.json", selected)
    sequence_source = SOURCE / "sequences.json"
    shutil.copyfile(sequence_source, HERE / "sequences.json")

    # Fetch and freeze the later fields only after all fleets are selected.
    reserved_hashes = {}
    reserved_sizes = {}
    for epoch in RESERVED_EPOCHS:
        field = fetch_reserved_field(epoch)
        path = HERE / "reserved_fields" / f"epoch_{epoch}.json"
        write_json(path, field)
        reserved_hashes[str(epoch)] = sha256(path)
        reserved_sizes[str(epoch)] = len(field["records"])

    code_hashes = {name: sha256(HERE / name) for name in CODE_FILES}
    margins = noninferiority_margins(arm_a["fleet"], draws)
    preregistration = {
        "experiment": "offline-h2-repaired-selector-proof-20260922",
        "frozen_before_fresh_scoring": True,
        "frozen_unix": time.time(),
        "target": TARGET,
        "scorer_protocol": "validator-dual-sample-v1",
        "fresh_draws": list(range(9, 17)),
        "candidate_count_per_draw": 888,
        "proposal_budget_per_arm": 282,
        "fleet_size_per_arm": FLEET_SIZE,
        "development_draws": list(range(1, 9)),
        "development_draw_hashes": {
            str(index): sha256(path)
            for index, path in enumerate(draw_paths, 1)
        },
        "development_public_fields": list(DEV_EPOCHS),
        "development_fields_sha256": sha256(HERE / "development_fields.json"),
        "reserved_evaluation_fields": list(RESERVED_EPOCHS),
        "reserved_field_hashes": reserved_hashes,
        "reserved_field_sizes": reserved_sizes,
        "reserved_fields_used_in_selection": False,
        "source_manifest_sha256": sha256(source_manifest_path),
        "sequence_manifest_sha256": sha256(HERE / "sequences.json"),
        "pool_manifest_sha256": sha256(HERE / "pools.json"),
        "frozen_fleet_artifact_sha256": sha256(HERE / "selected_fleets.json"),
        "frozen_code_hashes": code_hashes,
        "identity_capacity_audit": capacity,
        "catastrophic_per_candidate_gates": gates,
        "fleet_metric_noninferiority_margins": margins,
        "arms": {
            "A": {
                "proposal_pool": "A_current",
                "selector": "exact-current-production-replay",
                "fleet": arm_a["fleet"],
            },
            "B": {
                "proposal_pool": "B_and_C_ours_h2",
                "selector": "exact-current-production-replay",
                "fleet": arm_b["fleet"],
            },
            "C": {
                "proposal_pool": "B_and_C_ours_h2",
                "selector": "cross-fitted-robust-MILP",
                "fleet": arm_c["fleet"],
            },
        },
        "proof_gate": {
            "effective_independent_blocks": 8,
            "individual_wins_per_entry": "C strictly beats A in at least 6/8 scorer draws",
            "top10_per_entry": "C strictly beats A in at least 6/8 scorer draws",
            "leave_one_draw_out": "both aggregate directions remain strictly favorable in every 7/8 leave-one-draw-out view",
            "whole_fleet": "aggregate C>A, at least two positive draws, and no draw contributes more than half of positive gain",
            "hard_metrics": "for every frozen metric, median favorable delta >=0 and worst draw >= -A-development MAD margin",
            "runtime": "max fresh 8-GPU draw time *2 for four GPUs *1.10 guard <=3660 seconds",
            "decision": "all gates must pass; otherwise no-deploy",
        },
        "runtime_gate": {
            "deadline_seconds": 3660.0,
            "basis": "conservative (360 epoch blocks - 55 live reserve blocks) * 12 seconds",
            "projection_guard_multiplier": 1.10,
        },
        "comparisons": {
            "proposal_effect": "A_to_B",
            "selector_effect": "B_to_C",
            "combined_treatment": "A_to_C",
        },
        "change_log_before_freeze": [
            "Initial seven intersected quartile gates were rejected before freeze because they destroyed valid fleet feasibility.",
            "A decile-tail attempt was also rejected before freeze after the exact identity audit showed selection infeasibility.",
            "Final per-candidate gates reject only liabilities worse than the current-control worst median; H-bond/interface/developability moved to fresh fleet-level noninferiority.",
            "Common fleet size set to the exact minimum identity-cap capacity: eight.",
            "Greedy repaired fleet construction was replaced before freeze by exact binary scenario-coverage MILP to avoid a greedy identity trap.",
        ],
    }
    write_json(HERE / "preregistration.json", preregistration)
    (HERE / "PREREGISTRATION.sha256").write_text(
        sha256(HERE / "preregistration.json") + "  preregistration.json\n"
    )
    write_json(HERE / "run_metadata.json", {
        "experiment": preregistration["experiment"],
        "candidate_count": 888,
        "fresh_draws": list(range(9, 17)),
        "hourly_cost": 4.14111111111111,
        "preregistration_sha256": sha256(HERE / "preregistration.json"),
        "sequence_sha256": sha256(HERE / "sequences.json"),
        "draw_timings": [],
    })
    print(json.dumps({
        "frozen": True,
        "preregistration_sha256": sha256(HERE / "preregistration.json"),
        "fleet_sizes": {name: len(arm["fleet"]) for name, arm in selected.items()},
        "reserved_field_sizes": reserved_sizes,
        "identity_capacities": {name: row["capacity"] for name, row in capacity.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
