#!/usr/bin/env python3
"""Supersede the defective v1 runtime gate before any fresh scoring."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
GAP = HERE.parent
V1 = HERE / "superseded_v1"
V1_SHA256 = "06d65aaf1160573e32b3961b38749600396394c3a1f69c88ad817a9fc32438e0"
CODE_FILES = (
    "selector.py",
    "build_preregistration.py",
    "supersede_preregistration_v2.py",
    "analyze_fresh.py",
    "run_remote.sh",
    "timing_watch.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def completion_times(root: Path, names: list[str]) -> list[float]:
    return [(root / name / "exact_all.json").stat().st_mtime for name in names]


def consecutive_intervals(values: list[float]) -> list[float]:
    return [right - left for left, right in zip(values, values[1:])]


def timing_measurements() -> dict[str, Any]:
    capacity = GAP / "gpu-capacity-validation-20260922"
    h2 = GAP / "hamming2-fast-20260922"
    recall = GAP / "holdout-recall-20260922"

    discovery_paths = [
        capacity / f"discovery_{index:02d}" / "exact_all.json"
        for index in range(1, 5)
    ]
    discovery_intervals = consecutive_intervals(
        [path.stat().st_mtime for path in discovery_paths]
    )
    h2_paths = [
        h2 / f"pass_{index:02d}" / "exact_all.json"
        for index in range(1, 7)
    ]
    h2_intervals = consecutive_intervals(
        [path.stat().st_mtime for path in h2_paths]
    )
    confirmation_pair_times = []
    confirmation_paths = []
    for pair in range(10):
        paths = [
            capacity / f"confirmation_{2 * pair + offset:02d}" / "exact_all.json"
            for offset in (1, 2)
        ]
        confirmation_paths.extend(paths)
        confirmation_pair_times.append(max(path.stat().st_mtime for path in paths))
    confirmation_intervals = consecutive_intervals(confirmation_pair_times)

    recall_metadata_path = recall / "run_metadata.json"
    recall_metadata = json.loads(recall_metadata_path.read_text())
    recent_full_durations = [
        float(row["duration_seconds"])
        for row in recall_metadata["draw_timings"]
        if row.get("complete")
    ]
    if len(recent_full_durations) != 4:
        raise RuntimeError("expected four prior full-cohort timing observations")

    # Use the slowest recent directly measured full-cohort pass and the median
    # paired 96-candidate pass to identify t = L + g*N/G.  The independently
    # observed 690-candidate median is a cross-check, not a fit input.
    full_seconds = max(recent_full_durations)
    small_seconds = statistics.median(confirmation_intervals)
    full_work = 888.0 / 8.0
    small_work = 96.0 / 4.0
    gpu_seconds = (full_seconds - small_seconds) / (full_work - small_work)
    fixed_load = small_seconds - gpu_seconds * small_work
    h2_predicted = fixed_load + gpu_seconds * (690.0 / 8.0)
    h2_observed = statistics.median(h2_intervals)
    if fixed_load <= 0 or gpu_seconds <= 0 or h2_observed > h2_predicted:
        raise RuntimeError("timing calibration is not conservative on the 690 check")

    source_paths = [
        recall_metadata_path,
        *discovery_paths,
        *h2_paths,
        *confirmation_paths,
    ]
    return {
        "model": "wall_seconds = model_loads*L + gpu_seconds_per_observation*observations/GPU_count + selector_CPU_seconds",
        "fixed_load_seconds_L": fixed_load,
        "gpu_seconds_per_observation": gpu_seconds,
        "fit_endpoints": {
            "full_888_on_8_gpu_seconds": full_seconds,
            "full_888_source_draw_seconds": recent_full_durations,
            "paired_96_on_4_gpu_median_seconds": small_seconds,
            "paired_96_intervals_seconds": confirmation_intervals,
        },
        "cross_checks_not_fit": {
            "discovery_888_on_8_gpu_intervals_seconds": discovery_intervals,
            "discovery_888_on_8_gpu_median_seconds": statistics.median(discovery_intervals),
            "h2_690_on_8_gpu_intervals_seconds": h2_intervals,
            "h2_690_on_8_gpu_median_seconds": h2_observed,
            "h2_690_on_8_gpu_model_prediction_seconds": h2_predicted,
        },
        "selector_cpu_benchmark_seconds": {
            "A_current": 4.791625,
            "B_current_H2": 4.711212,
            "C_repaired_H2": 12.152582,
            "frozen_current_budget": 6.0,
            "frozen_treatment_budget": 15.0,
            "note": "Measured locally immediately before v2 freeze; frozen budgets round upward and the final 10% guard covers the entire workload.",
        },
        "source_sha256": {
            str(path.relative_to(GAP)): sha256(path) for path in source_paths
        },
    }


def workload_projection(
    loads: int,
    observations: int,
    selector_cpu_seconds: float,
    fixed_load: float,
    gpu_seconds: float,
) -> dict[str, float | int]:
    unguarded = (
        loads * fixed_load
        + gpu_seconds * observations / 4.0
        + selector_cpu_seconds
    )
    return {
        "model_loads": loads,
        "candidate_observations": observations,
        "selector_cpu_seconds": selector_cpu_seconds,
        "unguarded_seconds_at_4_gpu": unguarded,
        "guarded_seconds_at_4_gpu": unguarded * 1.10,
    }


def main() -> None:
    v1_path = V1 / "preregistration.json"
    if sha256(v1_path) != V1_SHA256:
        raise RuntimeError("preserved v1 hash mismatch")
    if sha256(HERE / "preregistration.json") != V1_SHA256:
        raise RuntimeError("active artifact is not the preserved v1 starting point")
    future = [
        str(path.relative_to(HERE))
        for draw in range(9, 17)
        if (path := HERE / f"fresh_{draw:02d}" / "exact_all.json").exists()
    ]
    if future:
        raise RuntimeError(f"fresh observations already exist: {future}")

    archive_files = sorted(
        path for path in V1.iterdir() if path.is_file() and path.name != "ARCHIVE.sha256"
    )
    (V1 / "ARCHIVE.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in archive_files)
    )

    measurements = timing_measurements()
    measurement_path = HERE / "runtime_model_measurements.json"
    write_json(measurement_path, measurements)
    fixed_load = float(measurements["fixed_load_seconds_L"])
    gpu_seconds = float(measurements["gpu_seconds_per_observation"])
    treatment = workload_projection(7, 7 * 282, 15.0, fixed_load, gpu_seconds)
    current = workload_projection(
        11, 4 * 300 + 4 * 180 + 3 * 108, 6.0, fixed_load, gpu_seconds
    )

    v1 = json.loads(v1_path.read_text())
    v2 = dict(v1)
    v2.update({
        "experiment": "offline-h2-repaired-selector-proof-20260922-v2",
        "preregistration_version": 2,
        "frozen_unix": time.time(),
        "supersedes": {
            "version": 1,
            "preregistration_sha256": V1_SHA256,
            "preserved_path": "superseded_v1/preregistration.json",
            "reason": "v1 projected one 888-candidate pass instead of the complete repeated production tournament",
            "scientific_change": "runtime gate only; proposal pools, selected fleets, evaluation fields, biological gates, and outcome gates are unchanged",
        },
        "runtime_model_measurements_sha256": sha256(measurement_path),
        "runtime_gate": {
            "deadline_seconds": 3660.0,
            "deadline_basis": "conservative (360 epoch blocks - 55 live reserve blocks) * 12 seconds",
            "production_gpu_count": 4,
            "projection_guard_multiplier": 1.10,
            "fixed_load_per_observation_model": {
                "fixed_load_seconds": fixed_load,
                "gpu_seconds_per_observation": gpu_seconds,
                "measurement_artifact": "runtime_model_measurements.json",
                "measurement_artifact_sha256": sha256(measurement_path),
            },
            "incremental_workloads": {
                "treatment": {
                    **treatment,
                    "formula": "7*282 candidate repeats + 7 model loads + repaired-selector CPU",
                },
                "current": {
                    **current,
                    "formula": "4*300 + 4*180 + 3*108 candidate repeats + 11 model loads + current-selector CPU",
                },
            },
            "broad_one_shot_screen": {
                "candidate_count": 2800,
                "status": "unchanged in A and C; excluded because it cancels in the incremental comparison",
            },
            "fresh_recalibration": {
                "candidate_count": 888,
                "gpu_count": 8,
                "rule": "replace frozen gpu-seconds/observation only when the slowest fresh full-888 draw implies a larger value; never reduce it",
            },
            "pass_rule": "treatment guarded wall <=3660 seconds AND treatment guarded wall <= current configured tournament guarded wall",
        },
    })
    v2["proof_gate"] = dict(v1["proof_gate"])
    v2["proof_gate"]["runtime"] = (
        "full incremental treatment (7*282 observations, 7 loads, selector CPU, 10% guard) "
        "must fit 3660s and not exceed the current 4*300+4*180+3*108, 11-load tournament; "
        "the unchanged broad 2800 one-shot cancels"
    )
    v2["change_log_before_freeze"] = [
        *v1["change_log_before_freeze"],
        "v1 was preserved and superseded before fresh scoring because its runtime gate covered only one full-888 draw.",
        "v2 changes only the runtime proof: complete treatment/current workloads, selector CPU, fixed-load/per-observation calibration, fresh conservative recalibration, and a 10% whole-workload guard.",
    ]
    v2["frozen_code_hashes"] = {
        name: sha256(HERE / name) for name in CODE_FILES
    }
    write_json(HERE / "preregistration.json", v2)
    v2_hash = sha256(HERE / "preregistration.json")
    (HERE / "PREREGISTRATION.sha256").write_text(
        f"{v2_hash}  preregistration.json\n"
    )
    supersession = {
        "v1_preregistration_sha256": V1_SHA256,
        "v1_archive_manifest_sha256": sha256(V1 / "ARCHIVE.sha256"),
        "v2_preregistration_sha256": v2_hash,
        "runtime_measurements_sha256": sha256(measurement_path),
        "fresh_draws_present_before_v2_freeze": future,
        "fresh_scoring_started_before_v2_freeze": False,
        "research_host_state": "exited; no resume issued while constructing v2",
        "reason": v2["supersedes"]["reason"],
    }
    supersession_path = HERE / "supersession_v1_to_v2.json"
    write_json(supersession_path, supersession)
    (HERE / "FREEZE_MANIFEST.sha256").write_text(
        f"{v2_hash}  preregistration.json\n"
        f"{sha256(measurement_path)}  runtime_model_measurements.json\n"
        f"{sha256(supersession_path)}  supersession_v1_to_v2.json\n"
        f"{sha256(V1 / 'ARCHIVE.sha256')}  superseded_v1/ARCHIVE.sha256\n"
    )
    metadata = json.loads((V1 / "run_metadata.json").read_text())
    metadata.update({
        "experiment": v2["experiment"],
        "preregistration_sha256": v2_hash,
        "preregistration_version": 2,
        "supersedes_preregistration_sha256": V1_SHA256,
        "draw_timings": [],
    })
    write_json(HERE / "run_metadata.json", metadata)
    print(json.dumps({
        "frozen": True,
        "v1_sha256": V1_SHA256,
        "v2_sha256": v2_hash,
        "fresh_draws_present": future,
        "treatment_guarded_seconds": treatment["guarded_seconds_at_4_gpu"],
        "current_guarded_seconds": current["guarded_seconds_at_4_gpu"],
        "deadline_seconds": 3660.0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
