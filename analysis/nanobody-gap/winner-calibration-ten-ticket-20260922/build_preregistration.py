#!/usr/bin/env python3
"""Build and freeze the historical calibration plus literal ten-ticket A/B."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
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

from nanobody_dash import DEFAULT_COLDKEY

from experiment import (
    METRICS,
    NONINFERIORITY_METRICS,
    TARGET,
    dedupe_field_entries,
    identity_audit,
    live_order_fill,
    load_draw,
    median_absolute_deviation,
    sequence_identity,
    sha256,
    write_json,
    valid_nanobody_sequence,
)


HERE = Path(__file__).resolve().parent
GAP = HERE.parent
PROOF = GAP / "h2-selector-proof-20260922"
CAPACITY = GAP / "gpu-capacity-validation-20260922"
HOLDOUT = GAP / "holdout-recall-20260922"
DB = ROOT / "data" / "nanobody_dash.sqlite"
HISTORY_CUTOFF = 25278
PANEL_1 = tuple(range(25261, 25269))
PANEL_2 = tuple(range(25269, 25277))
FRESH_DRAWS = tuple(range(17, 33))
API_ROOT = "https://compound-api.metanova-labs.ai"
CODE_FILES = (
    "experiment.py",
    "build_preregistration.py",
    "analyze_fresh.py",
    "test_frozen_semantics.py",
    "run_remote.sh",
    "timing_watch.py",
)


def complete_metrics(record: dict[str, Any]) -> dict[str, float] | None:
    # The live API's protein_id is numeric. rank_exact.leaderboard_field filters
    # on protein_name, whose value is the configured target accession.
    if str(record.get("protein_name") or "") != TARGET:
        return None
    if not all(record.get(metric) is not None for metric in METRICS):
        return None
    return {metric: float(record[metric]) for metric in METRICS}


def owner_label(coldkey: str) -> str:
    if coldkey == DEFAULT_COLDKEY:
        return "US"
    digest = hashlib.sha256(f"owner-label-v1|{coldkey}".encode()).hexdigest()
    return f"rival_{digest[:16]}"


def fetch_epoch(epoch: int) -> dict[str, Any]:
    url = f"{API_ROOT}/api/competitions/leaderboard/{epoch}/nanobodies"
    request = urllib.request.Request(
        url, headers={"User-Agent": "nova-offline-calibration-prereg/1"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:  # noqa: S310
        payload = json.load(response)
    sanitized = []
    for submission in payload.get("data", []):
        label = owner_label(str(submission.get("coldkey") or ""))
        for nanobody in submission.get("nanobodies") or []:
            metrics = complete_metrics(nanobody)
            sequence = str(nanobody.get("sequence") or "").strip().upper()
            if metrics is None or not sequence:
                continue
            sanitized.append({
                "owner_label": label,
                "sequence": sequence,
                "metrics": metrics,
            })
    unique, audit = dedupe_field_entries(sanitized)
    return {"epoch": epoch, "target": TARGET, "entries": unique, "deduplication_audit": audit}


def fleet_bucket(size: int) -> str:
    if size <= 7:
        return "le7"
    if size <= 9:
        return "8to9"
    return "ge10"


def allocate_nonwinning_slots(
    strata: dict[str, list[dict[str, Any]]], total: int
) -> dict[str, int]:
    """Frozen min-one plus capped largest-remainder allocation."""
    allocation = {
        name: 1 for name, rows in strata.items()
        if any(not row["winning_epoch"] for row in rows)
    }
    remaining = total - sum(allocation.values())
    if remaining < 0:
        raise RuntimeError("fewer matched slots than nonempty nonwinner strata")
    weights = {}
    for name in allocation:
        rows = strata[name]
        winner_count = sum(row["winning_epoch"] for row in rows)
        nonwinner_count = sum(not row["winning_epoch"] for row in rows)
        weights[name] = winner_count or nonwinner_count
    while remaining:
        active = [
            name for name in allocation
            if allocation[name]
            < sum(not row["winning_epoch"] for row in strata[name])
        ]
        if not active:
            raise RuntimeError("stratified allocation exhausted before 27 slots")
        total_weight = sum(weights[name] for name in active)
        quotas = {
            name: remaining * weights[name] / total_weight for name in active
        }
        floors = {
            name: min(
                math.floor(quotas[name]),
                sum(not row["winning_epoch"] for row in strata[name])
                - allocation[name],
            )
            for name in active
        }
        for name, amount in floors.items():
            allocation[name] += amount
        remaining -= sum(floors.values())
        if not remaining:
            break
        for name in sorted(
            active,
            key=lambda item: (
                -(quotas[item] - math.floor(quotas[item])),
                -weights[item],
                item,
            ),
        ):
            if not remaining:
                break
            capacity = sum(
                not row["winning_epoch"] for row in strata[name]
            )
            if allocation[name] < capacity:
                allocation[name] += 1
                remaining -= 1
    return allocation


def load_history() -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    connection = sqlite3.connect(DB)
    rows = connection.execute(
        """
        SELECT epoch, coldkey, score, sequence
        FROM entries WHERE epoch <= ?
        ORDER BY epoch, score, sequence
        """,
        (HISTORY_CUTOFF,),
    ).fetchall()
    connection.close()
    by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for epoch, coldkey, score, sequence in rows:
        if not valid_nanobody_sequence(sequence):
            raise RuntimeError(f"epoch {epoch} contains a null or scorer-invalid sequence")
        by_epoch[int(epoch)].append({
            "ours": coldkey == DEFAULT_COLDKEY,
            "score": float(score),
            "sequence": sequence.strip().upper(),
        })
    eligible = []
    for epoch, records in sorted(by_epoch.items()):
        ours = [row for row in records if row["ours"]]
        rival_scores = [row["score"] for row in records if not row["ours"]]
        if not ours or not rival_scores:
            continue
        best_rival = min(rival_scores)
        eligible.append({
            "epoch": epoch,
            "fleet_size": len(ours),
            "winning_epoch": any(row["score"] <= best_rival for row in ours),
            "winner_count": sum(row["score"] <= best_rival for row in ours),
            "ours": [
                {
                    "sequence": row["sequence"],
                    "actual_field_beater": row["score"] <= best_rival,
                }
                for row in ours
            ],
        })
    return eligible, by_epoch


def sample_history(eligible: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Ordered-index tertiles make the rule robust to irregular epoch gaps.
    total_epochs = len(eligible)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(eligible):
        tertile = min(2, index * 3 // total_epochs)
        name = f"t{tertile}_{fleet_bucket(row['fleet_size'])}"
        copied = dict(row)
        copied["stratum"] = name
        copied["ordered_epoch_index"] = index
        strata[name].append(copied)
    winning = [
        row for rows in strata.values() for row in rows if row["winning_epoch"]
    ]
    allocation = allocate_nonwinning_slots(strata, len(winning))
    selected_nonwinning = []
    stratum_audit = {}
    for name, rows in sorted(strata.items()):
        nonwinning = [row for row in rows if not row["winning_epoch"]]
        selected_count = allocation.get(name, 0)
        chosen = sorted(
            nonwinning,
            key=lambda row: (
                hashlib.sha256(
                    f"historical-calibration-v1|{row['epoch']}".encode()
                ).hexdigest(),
                row["epoch"],
            ),
        )[:selected_count]
        probability = selected_count / len(nonwinning) if nonwinning else None
        weight = len(nonwinning) / selected_count if selected_count else None
        for row in chosen:
            row["sampling_probability"] = probability
            row["epoch_weight"] = weight
            row["cohort_role"] = "sampled_nonwinning_epoch"
        selected_nonwinning.extend(chosen)
        stratum_audit[name] = {
            "winner_epochs": sum(row["winning_epoch"] for row in rows),
            "nonwinner_epochs": len(nonwinning),
            "sampled_nonwinner_epochs": selected_count,
            "nonwinner_inclusion_probability": probability,
            "nonwinner_epoch_weight": weight,
            "ordered_index_min": min(row["ordered_epoch_index"] for row in rows),
            "ordered_index_max": max(row["ordered_epoch_index"] for row in rows),
        }
    for row in winning:
        row["sampling_probability"] = 1.0
        row["epoch_weight"] = 1.0
        row["cohort_role"] = "all_winning_epochs"
    selected = sorted([*winning, *selected_nonwinning], key=lambda row: row["epoch"])
    audit = {
        "history_cutoff_epoch": HISTORY_CUTOFF,
        "eligible_epoch_count": len(eligible),
        "full_history_entry_count": sum(row["fleet_size"] for row in eligible),
        "full_history_field_beating_entries": sum(row["winner_count"] for row in eligible),
        "full_history_corrected_rate": (
            sum(row["winner_count"] for row in eligible)
            / sum(row["fleet_size"] for row in eligible)
        ),
        "winning_epoch_groups": len(winning),
        "selected_nonwinning_epochs": len(selected_nonwinning),
        "selected_epochs": len(selected),
        "strata": stratum_audit,
        "allocation_rule": (
            "ordered-epoch-index tertiles x fleet buckets <=7/8-9/>=10; "
            "minimum one nonwinner per nonempty stratum; remaining of 27 "
            "slots by capped largest remainder using winner-epoch count, "
            "falling back to N_h for zero-winner strata; within-stratum "
            "lowest SHA256('historical-calibration-v1|epoch')"
        ),
    }
    return selected, audit


def load_development_draws() -> list[dict[str, dict[str, float]]]:
    paths = [
        CAPACITY / f"discovery_{draw:02d}" / "exact_all.json"
        for draw in range(1, 5)
    ] + [
        HOLDOUT / f"heldout_{draw:02d}" / "exact_all.json"
        for draw in range(5, 9)
    ]
    return [load_draw(path, 888) for path in paths]


def build_fleets() -> tuple[dict[str, Any], list[dict[str, dict[str, float]]]]:
    frozen = json.loads((PROOF / "selected_fleets.json").read_text())
    a_order = [row["sequence"] for row in frozen["A"]["ranking"]]
    b_order = [row["sequence"] for row in frozen["B"]["ranking"]]
    a10 = live_order_fill(a_order, 10)
    b8 = list(frozen["B"]["fleet"])
    if b8 != live_order_fill(b_order, 8):
        raise RuntimeError("prior exact current-selector B8 does not replay")
    b10 = live_order_fill(a_order, 10, preselected=b8)
    if len(a10) != 10 or len(b10) != 10 or b10[:8] != b8:
        raise RuntimeError("literal ten-ticket fleet is not feasible")
    a_audit = identity_audit(a10)
    b_audit = identity_audit(b10)
    if b_audit["pairs_at_or_above_validator_0_95"]:
        raise RuntimeError("B10 crosses the hard validator duplicate line")
    draws = load_development_draws()
    margins = {}
    for metric in NONINFERIORITY_METRICS:
        per_draw = [
            statistics.median(draw[sequence][metric] for sequence in a10)
            for draw in draws
        ]
        margins[metric] = median_absolute_deviation(per_draw)
    return ({
        "A": {
            "fleet": a10,
            "composition": {"current_selector": 10},
            "identity_audit": a_audit,
            "selection": "exact live current-selector order and diversity semantics",
        },
        "B": {
            "fleet": b10,
            "composition": {"ours_hamming2_current_selector": 8, "current_order_fill": 2},
            "identity_audit": b_audit,
            "selection": (
                "exact frozen H2 B8, then current candidates in exact current-selector "
                "order; prefer <0.94 and farthest-first fallback strictly <0.95"
            ),
        },
        "development_noninferiority_mad_margins": margins,
        "source_selected_fleets_sha256": sha256(PROOF / "selected_fleets.json"),
    }, draws)


def main() -> None:
    for draw in FRESH_DRAWS:
        if (HERE / f"fresh_{draw:02d}" / "exact_all.json").exists():
            raise RuntimeError(f"fresh draw {draw} exists before preregistration")
    eligible, _ = load_history()
    selected_epochs, sampling_audit = sample_history(eligible)
    if sampling_audit["winning_epoch_groups"] != 27:
        raise RuntimeError("history snapshot no longer has frozen 27 winning groups")
    if sampling_audit["full_history_field_beating_entries"] != 33:
        raise RuntimeError("history snapshot no longer has frozen 33 field beaters")
    if sampling_audit["full_history_entry_count"] != 1495:
        raise RuntimeError("history snapshot no longer has frozen 1,495 entries")
    fleets, development_draws = build_fleets()

    fetch_epochs = sorted(
        set(row["epoch"] for row in selected_epochs) | set(PANEL_1) | set(PANEL_2)
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        fetched = dict(zip(fetch_epochs, executor.map(fetch_epoch, fetch_epochs)))

    calibration_epochs = []
    calibration_sequences = []
    for row in selected_epochs:
        public = fetched[row["epoch"]]
        public_ours = {
            item["sequence"] for item in public["entries"]
            if item["owner_label"] == "US"
        }
        expected_ours = {item["sequence"] for item in row["ours"]}
        if not expected_ours <= public_ours:
            raise RuntimeError(f"epoch {row['epoch']} is missing public US sequences")
        rivals = [
            item for item in public["entries"] if item["owner_label"] != "US"
        ]
        if not rivals:
            raise RuntimeError(f"epoch {row['epoch']} has no complete rival field")
        entries = []
        for index, item in enumerate(row["ours"]):
            entries.append({
                "entry_id": f"e{row['epoch']}_us{index:02d}",
                "sequence": item["sequence"],
                "actual_field_beater": item["actual_field_beater"],
            })
            calibration_sequences.append(item["sequence"])
        calibration_epochs.append({
            "epoch": row["epoch"],
            "stratum": row["stratum"],
            "cohort_role": row["cohort_role"],
            "sampling_probability": row["sampling_probability"],
            "epoch_weight": row["epoch_weight"],
            "entries": entries,
            "rivals": rivals,
            "public_field_deduplication_audit": public["deduplication_audit"],
        })
    if len(calibration_sequences) != len(set(calibration_sequences)):
        raise RuntimeError("calibration sequences are unexpectedly duplicated")

    panel_payload = {}
    for name, epochs in (("panel_25261_25268", PANEL_1), ("panel_25269_25276", PANEL_2)):
        fields = []
        for epoch in epochs:
            rivals = [
                item for item in fetched[epoch]["entries"]
                if item["owner_label"] != "US"
            ]
            if len(rivals) < 30:
                raise RuntimeError(f"panel epoch {epoch} has fewer than 30 rivals")
            fields.append({
                "epoch": epoch,
                "rivals": rivals,
                "public_field_deduplication_audit": fetched[epoch]["deduplication_audit"],
            })
        panel_payload[name] = fields

    finalists = list(dict.fromkeys(fleets["A"]["fleet"] + fleets["B"]["fleet"]))
    combined = list(dict.fromkeys(calibration_sequences + finalists))
    overlap = len(calibration_sequences) + len(finalists) - len(combined)
    write_json(HERE / "historical_sampling.json", sampling_audit)
    write_json(HERE / "calibration_cohort.json", {
        "target": TARGET,
        "epochs": calibration_epochs,
        "entry_record_count": len(calibration_sequences),
        "unique_sequence_count": len(set(calibration_sequences)),
    })
    write_json(HERE / "rival_panels.json", panel_payload)
    write_json(HERE / "fleets.json", fleets)
    write_json(HERE / "sequences.json", {"target": TARGET, "sequences": combined})

    # No raw public owner identifiers may persist in any frozen data artifact.
    for path in (
        HERE / "historical_sampling.json",
        HERE / "calibration_cohort.json",
        HERE / "rival_panels.json",
        HERE / "fleets.json",
        HERE / "sequences.json",
    ):
        if DEFAULT_COLDKEY in path.read_text():
            raise RuntimeError(f"raw own coldkey leaked into {path.name}")

    fixed_load = 35.9383817212335
    gpu_seconds = 5.460900761615271
    hourly_cost = 4.14111111111111
    per_draw_projected = fixed_load + gpu_seconds * len(combined) / 8
    projected_total = per_draw_projected * len(FRESH_DRAWS)
    weighted_denominator = sum(
        epoch["epoch_weight"] * len(epoch["entries"])
        for epoch in calibration_epochs
    )
    preregistration = {
        "experiment": "historical-winner-calibration-and-literal-ten-ticket-ab-20260922",
        "frozen_before_fresh_scoring": True,
        "frozen_unix": time.time(),
        "target": TARGET,
        "scorer_protocol": "validator-dual-sample-v1",
        "fresh_draws": list(FRESH_DRAWS),
        "existing_draw17_plus_absent_at_freeze": True,
        "candidate_manifest_count": len(combined),
        "calibration_entry_records": len(calibration_sequences),
        "calibration_unique_sequences": len(set(calibration_sequences)),
        "ab_unique_finalists": len(finalists),
        "calibration_ab_overlap": overlap,
        "development_only": {
            "draws": list(range(1, 9)),
            "fields": list(range(25231, 25253)),
            "no_draw9_16_outcomes_used": True,
            "source_fleet_artifact_sha256": fleets["source_selected_fleets_sha256"],
        },
        "test1_historical_calibration": {
            "history_cutoff_epoch": HISTORY_CUTOFF,
            "full_history_rate": "33/1495",
            "full_history_rate_value": 33 / 1495,
            "winning_epoch_groups": 27,
            "same_epoch_losing_controls": 223,
            "sampling_artifact": "historical_sampling.json",
            "sampling_artifact_sha256": sha256(HERE / "historical_sampling.json"),
            "cohort_artifact": "calibration_cohort.json",
            "cohort_artifact_sha256": sha256(HERE / "calibration_cohort.json"),
            "weighted_actual_rate_audit": 33 / weighted_denominator,
            "weighted_denominator_audit": weighted_denominator,
            "joint_ranking_semantics": (
                "for each epoch x draw, rank all US entries together with all "
                "other-coldkey rivals; multiple US entries may win; US entries "
                "compete for global top-ten slots"
            ),
            "primary_gates": {
                "winner_recovery": (
                    "order entries by descending frequency of joint entry-win over "
                    "the 16 intact future draws, then ascending mean joint rank sum, "
                    "then entry_id; the first entry is an actual field-beater in at "
                    "least 20/27 winning epoch groups"
                ),
                "absolute_calibration": (
                    "full-history corrected 33/1495 rate lies in frozen 95% "
                    "two-level draw-block plus epoch/stratum bootstrap interval of "
                    "the IPW predicted win rate"
                ),
                "winner_control_separation": (
                    "95% bootstrap lower bound of IPW mean predicted-win-rate "
                    "difference (actual field beaters minus controls) is >0"
                ),
            },
            "bootstrap": {
                "replicates": 5000,
                "seed": "sha256('winner-calibration-bootstrap-v1') first 64 bits",
                "units": (
                    "resample 16 intact scorer draws; independently resample epoch "
                    "clusters within sampling stratum x cohort role"
                ),
            },
            "secondary": ["IPW Brier score", "IPW weighted AUC", "top10 rate"],
        },
        "test2_literal_ten_ticket": {
            "fleets_artifact": "fleets.json",
            "fleets_artifact_sha256": sha256(HERE / "fleets.json"),
            "fleet_size": 10,
            "live_identity_semantics": (
                "prefer pairwise identity <0.94; farthest-first fallback permitted "
                "only below hard validator duplicate line 0.95"
            ),
            "panels": ["25261-25268", "25269-25276"],
            "panels_artifact_sha256": sha256(HERE / "rival_panels.json"),
            "joint_ranking_semantics": (
                "rank all rivals plus all ten arm entries jointly for every "
                "draw x field; wins tie/beat best rival; top10 uses global cutoff"
            ),
            "proof_gates": {
                "draw_blocks": (
                    "B strictly exceeds A for individual wins/entry and top10/entry "
                    "in at least 12/16 scorer-draw blocks"
                ),
                "leave_one_draw_out": (
                    "aggregate B-A individual-win and top10 count deltas remain "
                    "strictly positive in every 15/16 LODO view"
                ),
                "both_panels": (
                    "aggregate B-A individual-win and top10 deltas are strictly "
                    "positive separately in both frozen field panels"
                ),
                "not_two_draw_dominated": (
                    "for both outcomes at least 3 draw blocks have positive delta "
                    "and the largest two positive contributions are <=50% of total"
                ),
                "interface_liability_noninferiority": (
                    "for every frozen metric, median favorable B-A delta >=0 and "
                    "worst block >= negative A development-draw MAD"
                ),
                "runtime": (
                    "fresh conservative production projection for 7*282 repeats + "
                    "7 loads + 6s selector CPU, 10% guard, is <=3660s and <= the "
                    "configured current tournament projection"
                ),
                "decision": "all gates pass or NO-DEPLOY",
            },
        },
        "runtime": {
            "research_gpu_count": 8,
            "production_gpu_count": 4,
            "hourly_cost": hourly_cost,
            "fixed_load_seconds": fixed_load,
            "frozen_gpu_seconds_per_observation": gpu_seconds,
            "projected_seconds_per_fresh_draw": per_draw_projected,
            "projected_total_seconds_16_draws": projected_total,
            "projected_total_minutes": projected_total / 60,
            "projected_scoring_cost": projected_total / 3600 * hourly_cost,
            "research_guard_multiplier": 1.10,
            "research_guarded_total_seconds": projected_total * 1.10,
            "production_deadline_seconds": 3660.0,
            "production_treatment": {
                "candidate_observations": 7 * 282,
                "model_loads": 7,
                "selector_cpu_seconds": 6.0,
            },
            "production_current": {
                "candidate_observations": 4 * 300 + 4 * 180 + 3 * 108,
                "model_loads": 11,
                "selector_cpu_seconds": 6.0,
            },
            "projection_guard_multiplier": 1.10,
            "broad_2800_one_shot": "unchanged between arms and cancels",
        },
        "frozen_hashes": {
            "sequence_manifest": sha256(HERE / "sequences.json"),
            "calibration_cohort": sha256(HERE / "calibration_cohort.json"),
            "historical_sampling": sha256(HERE / "historical_sampling.json"),
            "rival_panels": sha256(HERE / "rival_panels.json"),
            "fleets": sha256(HERE / "fleets.json"),
            "development_draws": {
                str(index): sha256(path)
                for index, path in enumerate([
                    *(
                        CAPACITY / f"discovery_{draw:02d}" / "exact_all.json"
                        for draw in range(1, 5)
                    ),
                    *(
                        HOLDOUT / f"heldout_{draw:02d}" / "exact_all.json"
                        for draw in range(5, 9)
                    ),
                ], 1)
            },
            "code": {name: sha256(HERE / name) for name in CODE_FILES},
        },
    }
    write_json(HERE / "preregistration.json", preregistration)
    (HERE / "PREREGISTRATION.sha256").write_text(
        sha256(HERE / "preregistration.json") + "  preregistration.json\n"
    )
    freeze_files = [
        "preregistration.json",
        "historical_sampling.json",
        "calibration_cohort.json",
        "rival_panels.json",
        "fleets.json",
        "sequences.json",
        *CODE_FILES,
    ]
    (HERE / "FREEZE_MANIFEST.sha256").write_text("".join(
        f"{sha256(HERE / name)}  {name}\n" for name in freeze_files
    ))
    write_json(HERE / "run_metadata.json", {
        "experiment": preregistration["experiment"],
        "candidate_count": len(combined),
        "fresh_draws": list(FRESH_DRAWS),
        "hourly_cost": hourly_cost,
        "preregistration_sha256": sha256(HERE / "preregistration.json"),
        "sequence_sha256": sha256(HERE / "sequences.json"),
        "draw_timings": [],
    })
    print(json.dumps({
        "frozen": True,
        "preregistration_sha256": sha256(HERE / "preregistration.json"),
        "candidate_count": len(combined),
        "calibration_records": len(calibration_sequences),
        "ab_finalists": len(finalists),
        "fleet_a_identity": fleets["A"]["identity_audit"],
        "fleet_b_identity": fleets["B"]["identity_audit"],
        "projected_minutes": projected_total / 60,
        "projected_cost": projected_total / 3600 * hourly_cost,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
