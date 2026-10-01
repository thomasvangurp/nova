#!/usr/bin/env python3
"""Frozen offline selectors for the H2 proposal/selection proof.

Every scoring operation keeps the validator's complete ten-metric vector
intact.  No metric is independently resampled.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from nanobody_opt.fleet import MAXIMUM_FLEET_IDENTITY, distinct_sequences
from nanobody_opt.simulate_round import METRICS
from nanobody_opt.win_probability import rank_by_fresh_draw_probability


TARGET = "P20809"
# The exact current selector can emit only eight non-colliding sequences from
# the equal-budget H2 pool at its frozen 0.94 identity limit.  All proof arms
# therefore use eight tickets; per-entry outcomes remain directly comparable.
FLEET_SIZE = 8
FLEET_NONINFERIORITY_METRICS = {
    "liability_score": "min",
    "liability_num_violations": "min",
    "plip_hbonds_refolded": "max",
    "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min",
    "interaction_pae": "min",
    "delta_sasa_refolded": "max",
}
PER_CANDIDATE_HARD_METRICS = {
    "liability_score": "min",
    "liability_num_violations": "min",
}


def dense_rank_sums(records: Sequence[Mapping[str, float]]) -> list[int]:
    totals = [0] * len(records)
    for metric, direction in METRICS.items():
        values = sorted(
            {float(record[metric]) for record in records},
            reverse=(direction == "max"),
        )
        ranks = {value: index + 1 for index, value in enumerate(values)}
        for index, record in enumerate(records):
            totals[index] += ranks[float(record[metric])]
    return totals


def candidate_field_outcome(
    metrics: Mapping[str, float], field: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    scores = dense_rank_sums([*field, metrics])
    candidate = scores[-1]
    competitor = scores[:-1]
    cutoff = sorted(scores)[min(9, len(scores) - 1)]
    return {
        "rank_sum": candidate,
        "win": candidate <= min(competitor),
        "top10": candidate <= cutoff,
    }


def current_selector(
    sequences: Sequence[str],
    draws: Sequence[Mapping[str, Mapping[str, float]]],
    fields: Sequence[Mapping[str, Any]],
    *,
    fleet_size: int = FLEET_SIZE,
) -> dict[str, Any]:
    """Replay the exact live scalar selector and fleet de-duplication path."""
    candidates = []
    for sequence in sequences:
        per_draw = [
            statistics.median(
                candidate_field_outcome(draw[sequence], field["records"])[
                    "rank_sum"
                ]
                for field in fields
            )
            for draw in draws
        ]
        candidates.append({
            "sequence": sequence,
            "settled_deviations": 0,
            "per_observation_median_rank_sum": per_draw,
            "score_median": statistics.median(per_draw),
            "score_p90": quantile(per_draw, 0.9),
        })
    ranked = rank_by_fresh_draw_probability(candidates)
    ranking = {
        "selection_order": "win-probability",
        "candidates": ranked,
    }
    fleet = distinct_sequences(
        ranking,
        fleet_size,
        value_share=0.0,
        maximum_identity=MAXIMUM_FLEET_IDENTITY,
    )
    return {
        "fleet": fleet,
        "fleet_size": len(fleet),
        "ranking": ranked,
        "policy": {
            "ranker": "rank_by_fresh_draw_probability",
            "fleet_selector": "fleet.distinct_sequences",
            "selection_order": "win-probability",
            "value_share": 0.0,
            "maximum_identity": MAXIMUM_FLEET_IDENTITY,
            "settled_deviations": 0,
        },
    }


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def hard_gate_thresholds(
    reference_sequences: Sequence[str],
    draws: Sequence[Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float | str]]:
    """Freeze catastrophic liability gates at the current-control worst."""
    thresholds: dict[str, dict[str, float | str]] = {}
    for metric, direction in PER_CANDIDATE_HARD_METRICS.items():
        values = [
            statistics.median(draw[sequence][metric] for draw in draws)
            for sequence in reference_sequences
        ]
        probability = 1.0 if direction == "min" else 0.0
        thresholds[metric] = {
            "direction": direction,
            "threshold": quantile(values, probability),
            "reference_quantile": probability,
            "meaning": "reject only candidates worse than every current-control median",
        }
    return thresholds


def passes_hard_gates(
    sequence: str,
    draws: Sequence[Mapping[str, Mapping[str, float]]],
    gates: Mapping[str, Mapping[str, float | str]],
) -> bool:
    for metric, gate in gates.items():
        value = statistics.median(draw[sequence][metric] for draw in draws)
        threshold = float(gate["threshold"])
        if gate["direction"] == "min" and value > threshold:
            return False
        if gate["direction"] == "max" and value < threshold:
            return False
    return True


def _families(sequence: str, metadata: Mapping[str, Mapping[str, Any]]) -> list[str]:
    groups = metadata[sequence].get("arm_groups", {}).get("ours_hamming2", [])
    return sorted(str(group) for group in groups) or ["unassigned"]


def _sequence_identity(left: str, right: str) -> float:
    return sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))


def maximum_independent_capacity(
    sequences: Sequence[str], *, maximum_identity: float = MAXIMUM_FLEET_IDENTITY
) -> dict[str, Any]:
    """Exact maximum non-colliding fleet size under the live identity cap."""
    sequences = list(sequences)
    edges = [
        (left, right)
        for left in range(len(sequences))
        for right in range(left + 1, len(sequences))
        if _sequence_identity(sequences[left], sequences[right]) >= maximum_identity
    ]
    row = []
    column = []
    data = []
    for index, (left, right) in enumerate(edges):
        row.extend((index, index))
        column.extend((left, right))
        data.extend((1.0, 1.0))
    constraints = []
    if edges:
        matrix = coo_matrix(
            (data, (row, column)), shape=(len(edges), len(sequences))
        ).tocsr()
        constraints.append(
            LinearConstraint(matrix, np.zeros(len(edges)), np.ones(len(edges)))
        )
    result = milp(
        c=-np.ones(len(sequences)),
        integrality=np.ones(len(sequences)),
        bounds=Bounds(np.zeros(len(sequences)), np.ones(len(sequences))),
        constraints=constraints,
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise RuntimeError("maximum independent-set audit failed")
    witness = [
        sequence for sequence, value in zip(sequences, result.x, strict=True)
        if value > 0.5
    ]
    return {
        "capacity": len(witness),
        "witness": witness,
        "candidate_count": len(sequences),
        "identity_edges": len(edges),
        "maximum_identity": maximum_identity,
        "solver": "scipy.optimize.milp exact binary independent set",
    }


def repaired_selector(
    sequences: Sequence[str],
    draws: Sequence[Mapping[str, Mapping[str, float]]],
    fields: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Mapping[str, float | str]],
    *,
    fleet_size: int = FLEET_SIZE,
) -> dict[str, Any]:
    """Cross-fitted robust fleet selection on four intact development blocks."""
    if len(draws) != 8 or len(fields) != 22:
        raise ValueError("repaired selector is frozen for 8 draws and 22 fields")
    draw_blocks = ([0, 1], [2, 3], [4, 5], [6, 7])
    field_blocks = ([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16], [17, 18, 19, 20, 21])
    eligible = [
        sequence for sequence in sequences
        if passes_hard_gates(sequence, draws, gates)
    ]
    if len(eligible) < fleet_size:
        raise ValueError("hard gates leave fewer candidates than the fleet")

    scenario_block: dict[int, int] = {}
    win_scenarios: dict[str, set[int]] = {sequence: set() for sequence in eligible}
    top10_scenarios: dict[str, set[int]] = {sequence: set() for sequence in eligible}
    fold_records = []
    scenario = 0
    for fold, (test_draws, test_fields) in enumerate(
        zip(draw_blocks, field_blocks, strict=True)
    ):
        train_draws = [draw for index, draw in enumerate(draws) if index not in test_draws]
        train_fields = [field for index, field in enumerate(fields) if index not in test_fields]
        training = current_selector(eligible, train_draws, train_fields, fleet_size=fleet_size)
        training_rank = {
            item["sequence"]: index + 1 for index, item in enumerate(training["ranking"])
        }
        block_scenarios = []
        for draw_index in test_draws:
            for field_index in test_fields:
                scenario_block[scenario] = fold
                block_scenarios.append(scenario)
                field = fields[field_index]["records"]
                for sequence in eligible:
                    outcome = candidate_field_outcome(draws[draw_index][sequence], field)
                    if outcome["win"]:
                        win_scenarios[sequence].add(scenario)
                    if outcome["top10"]:
                        top10_scenarios[sequence].add(scenario)
                scenario += 1
        fold_records.append({
            "fold": fold + 1,
            "heldout_draw_numbers": [index + 1 for index in test_draws],
            "heldout_field_epochs": [fields[index]["epoch"] for index in test_fields],
            "training_draw_numbers": [
                index + 1 for index in range(len(draws)) if index not in test_draws
            ],
            "training_field_epochs": [field["epoch"] for field in train_fields],
            "scenario_ids": block_scenarios,
            "training_rank": training_rank,
        })

    def block_rates(sequence: str, events: Mapping[str, set[int]]) -> list[float]:
        return [
            sum(scenario_id in events[sequence] for scenario_id, block in scenario_block.items() if block == fold)
            / sum(block == fold for block in scenario_block.values())
            for fold in range(4)
        ]

    diagnostics = {}
    for sequence in eligible:
        wins = block_rates(sequence, win_scenarios)
        top10 = block_rates(sequence, top10_scenarios)
        diagnostics[sequence] = {
            "win_rate_by_block": wins,
            "top10_rate_by_block": top10,
            "win_rate_median": statistics.median(wins),
            "win_rate_worst": min(wins),
            "top10_rate_median": statistics.median(top10),
            "top10_rate_worst": min(top10),
            "training_rank_median": statistics.median(
                fold["training_rank"][sequence] for fold in fold_records
            ),
        }

    scenario_count = len(scenario_block)
    candidate_count = len(eligible)
    variable_count = candidate_count + 2 * scenario_count
    objective = np.zeros(variable_count)
    for index, sequence in enumerate(eligible):
        diag = diagnostics[sequence]
        objective[index] = -(
            100.0 * diag["win_rate_worst"]
            + 50.0 * diag["win_rate_median"]
            + 25.0 * diag["top10_rate_worst"]
            + 10.0 * diag["top10_rate_median"]
            - 0.001 * diag["training_rank_median"]
        )
    objective[candidate_count:candidate_count + scenario_count] = -1000.0
    objective[candidate_count + scenario_count:] = -10.0

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: Mapping[int, float], low: float, high: float) -> None:
        row_index = len(lower)
        for column, value in coefficients.items():
            rows.append(row_index)
            columns.append(column)
            values.append(value)
        lower.append(low)
        upper.append(high)

    add_constraint({index: 1.0 for index in range(candidate_count)}, fleet_size, fleet_size)
    for left in range(candidate_count):
        for right in range(left + 1, candidate_count):
            if _sequence_identity(eligible[left], eligible[right]) >= MAXIMUM_FLEET_IDENTITY:
                add_constraint({left: 1.0, right: 1.0}, 0.0, 1.0)
    family_members: dict[str, list[int]] = defaultdict(list)
    for index, sequence in enumerate(eligible):
        for family in _families(sequence, metadata):
            family_members[family].append(index)
    for members in family_members.values():
        add_constraint({index: 1.0 for index in members}, 0.0, 1.0)
    for scenario_id in range(scenario_count):
        win_variable = candidate_count + scenario_id
        top10_variable = candidate_count + scenario_count + scenario_id
        add_constraint(
            {win_variable: 1.0, **{
                index: -1.0 for index, sequence in enumerate(eligible)
                if scenario_id in win_scenarios[sequence]
            }},
            -math.inf,
            0.0,
        )
        add_constraint(
            {top10_variable: 1.0, **{
                index: -1.0 for index, sequence in enumerate(eligible)
                if scenario_id in top10_scenarios[sequence]
            }},
            -math.inf,
            0.0,
        )
    matrix = coo_matrix(
        (values, (rows, columns)), shape=(len(lower), variable_count)
    ).tocsr()
    solution = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=[LinearConstraint(matrix, np.array(lower), np.array(upper))],
        options={"time_limit": 180},
    )
    if not solution.success or solution.x is None:
        raise ValueError("robust fleet MILP is infeasible")
    fixed = [
        sequence for sequence, value in zip(eligible, solution.x[:candidate_count], strict=True)
        if value > 0.5
    ]
    if len(fixed) != fleet_size:
        raise RuntimeError("robust fleet MILP returned the wrong fleet size")

    selected: list[str] = []
    covered_wins: set[int] = set()
    covered_top10: set[int] = set()
    trace = []
    remaining = set(fixed)
    while remaining:
        sequence = max(
            remaining,
            key=lambda item: (
                len(win_scenarios[item] - covered_wins),
                len(top10_scenarios[item] - covered_top10),
                diagnostics[item]["win_rate_worst"],
                diagnostics[item]["top10_rate_worst"],
                item,
            ),
        )
        new_wins = win_scenarios[sequence] - covered_wins
        new_top10 = top10_scenarios[sequence] - covered_top10
        selected.append(sequence)
        covered_wins |= new_wins
        covered_top10 |= new_top10
        remaining.remove(sequence)
        trace.append({
            "slot": len(selected),
            "sequence": sequence,
            "assigned_family": _families(sequence, metadata)[0],
            "families": _families(sequence, metadata),
            "marginal_win_scenarios": len(new_wins),
            "marginal_top10_scenarios": len(new_top10),
            **diagnostics[sequence],
        })
    family_counts = defaultdict(int)
    for sequence in selected:
        family_counts[_families(sequence, metadata)[0]] += 1
    return {
        "fleet": selected,
        "fleet_size": len(selected),
        "eligible_after_hard_gates": len(eligible),
        "hard_gates": gates,
        "crossfit_folds": fold_records,
        "candidate_diagnostics": diagnostics,
        "selection_trace": trace,
        "family_counts": dict(sorted(family_counts.items())),
        "covered_oof_win_scenarios": len(covered_wins),
        "covered_oof_top10_scenarios": len(covered_top10),
        "oof_scenarios": len(scenario_block),
        "policy": {
            "complete_metrics": list(METRICS),
            "crossfit": "four paired leave-one-draw-block/field-block folds",
            "candidate_objective": "median/worst-block individual win and top10 rate",
            "fleet_objective": "greedy marginal intact-scenario win/top10 coverage",
            "maximum_members_per_basin": 1,
            "optimizer": "binary MILP with intact-scenario coverage variables",
            "maximum_identity": MAXIMUM_FLEET_IDENTITY,
        },
    }
