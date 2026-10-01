#!/usr/bin/env python3
"""Replay fixed-cost within-epoch allocation policies on the frozen A pool.

This is deliberately an allocation test, not a new scoring run.  The 282
current-production candidates already have sixteen complete, independent
validator-shaped observations.  Four draw blocks are held out in turn; a
policy may use only the other twelve observations to prune and select, and is
then evaluated against eight reserved public competitor fields.

The repeat-observation budget is fixed at 2,244, matching the configured
300 -> 180 -> 108 tournament (4, 4, 3 rounds).  Policies with fewer model
loads can spend the saved launches on a few more candidate observations, but
the primary comparisons below remain at 2,244 observations or less.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROOF = ROOT / "analysis/nanobody-gap/h2-selector-proof-20260922"
GPU = ROOT / "analysis/nanobody-gap/gpu-capacity-validation-20260922"
RECALL = ROOT / "analysis/nanobody-gap/holdout-recall-20260922"
HERE = Path(__file__).resolve().parent
TARGET = "P20809"
FLEET_SIZE = 10
REPEAT_BUDGET = 2244
MAXIMUM_IDENTITY = 0.94

from nanobody_opt.simulate_round import METRICS  # noqa: E402
from nanobody_opt.win_probability import rank_by_fresh_draw_probability  # noqa: E402


DRAW_PATHS = [
    *(GPU / f"discovery_{index:02d}/exact_all.json" for index in range(1, 5)),
    *(GPU / f"heldout_{index:02d}/exact_all.json" for index in range(5, 8)),
    RECALL / "heldout_08/exact_all.json",
    *(PROOF / f"fresh_{index:02d}/exact_all.json" for index in range(9, 17)),
]


# (rounds at this width, optional number retained after those rounds).
# The initial one-pass screen of all 282 candidates is outside REPEAT_BUDGET.
POLICIES: dict[str, tuple[tuple[int, int | None], ...]] = {
    # Exact configured shape, scaled only because this frozen pool has 282
    # candidates rather than the configured maximum of 300.
    "current_4_4_3": ((4, 169), (4, 101), (3, None)),
    # Spend the same 2,244 candidate-observations but postpone the decision.
    "late_5_6": ((5, 139), (6, None)),
    "late_6_4": ((6, 138), (4, None)),
    "late_7_2": ((7, 135), (2, None)),
    # Give everybody eight repeats.  This is twelve observations (0.5%) over
    # the observation budget but uses three fewer model launches than current.
    "uniform_8": ((8, None),),
    # Same-cohort controls.  The frozen pool has only 282 candidates, so the
    # literal current policy consumes 2,107 rather than the 2,244 it consumes
    # at the live width of 300.  These arms spend at most 2,107 too, separating
    # allocation shape from the 137-observation cohort-size difference.
    "equal2107_late_5_6": ((5, 116), (6, None)),
    "equal2107_late_6_3": ((6, 138), (3, None)),
    "equal2107_late_6_4_depth11": ((6, 103), (4, None)),
    "equal2107_late_6_5_depth12": ((6, 83), (5, None)),
    "equal2107_late_7_1": ((7, 133), (1, None)),
    "equal2107_late_7_4_depth12": ((7, 33), (4, None)),
    "equal2107_uniform_7": ((7, None),),
}


def load_draw(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text())
    found: dict[str, dict[str, float]] = {}
    for wave in payload["per_nanobody_components"].values():
        for sequence, by_target in wave.items():
            metrics = by_target[TARGET]
            if set(METRICS).issubset(metrics):
                found[sequence] = {name: float(metrics[name]) for name in METRICS}
    return found


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


def outcome(metrics: Mapping[str, float], field: Sequence[Mapping[str, float]]) -> tuple[int, bool, bool]:
    scores = dense_rank_sums([*field, metrics])
    candidate = scores[-1]
    competitors = scores[:-1]
    cutoff = sorted(scores)[min(9, len(scores) - 1)]
    return candidate, candidate <= min(competitors), candidate <= cutoff


def outcome_table(
    sequences: Sequence[str],
    draws: Sequence[Mapping[str, Mapping[str, float]]],
    fields: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Vectorize exact dense-rank insertion for every draw/candidate/field."""
    shape = (len(draws), len(sequences), len(fields))
    scores = np.zeros(shape, dtype=np.int16)
    wins = np.zeros(shape, dtype=np.bool_)
    top10 = np.zeros(shape, dtype=np.bool_)
    for draw_index, draw in enumerate(draws):
        for field_index, field_record in enumerate(fields):
            field = field_record["records"]
            candidate_total = np.zeros(len(sequences), dtype=np.int16)
            competitor_total = np.zeros((len(sequences), len(field)), dtype=np.int16)
            for metric, direction in METRICS.items():
                candidate = np.array(
                    [float(draw[sequence][metric]) for sequence in sequences]
                )
                competitor = np.array([float(row[metric]) for row in field])
                unique = np.unique(competitor)
                if direction == "max":
                    candidate_rank = 1 + (unique[None, :] > candidate[:, None]).sum(axis=1)
                    competitor_rank = 1 + (unique[None, :] > competitor[:, None]).sum(axis=1)
                    candidate_better = candidate[:, None] > competitor[None, :]
                else:
                    candidate_rank = 1 + (unique[None, :] < candidate[:, None]).sum(axis=1)
                    competitor_rank = 1 + (unique[None, :] < competitor[:, None]).sum(axis=1)
                    candidate_better = candidate[:, None] < competitor[None, :]
                candidate_is_new = ~np.isin(candidate, unique)
                competitor_total += competitor_rank[None, :] + (
                    candidate_better & candidate_is_new[:, None]
                )
                candidate_total += candidate_rank
            scores[draw_index, :, field_index] = candidate_total
            wins[draw_index, :, field_index] = (
                candidate_total <= competitor_total.min(axis=1)
            )
            top10[draw_index, :, field_index] = (
                (competitor_total < candidate_total[:, None]).sum(axis=1) <= 9
            )
    return {"scores": scores, "wins": wins, "top10": top10}


def identity(left: str, right: str) -> float:
    return sum(a == b for a, b in zip(left, right)) / max(len(left), len(right))


def family_slate(ranked: Sequence[str], keep: int) -> list[str]:
    """Mirror sequential_exact_race._keep_a_family_slate."""
    if keep >= len(ranked):
        return list(ranked)
    slate: list[str] = []
    for sequence in ranked:
        if len(slate) >= keep:
            break
        if all(identity(sequence, chosen) < 0.95 for chosen in slate):
            slate.append(sequence)
    if len(slate) < keep:
        taken = set(slate)
        slate.extend(sequence for sequence in ranked if sequence not in taken)
    return slate[:keep]


def fleet(ranked: Sequence[str], count: int = FLEET_SIZE) -> list[str]:
    chosen: list[str] = []
    for sequence in ranked:
        if all(identity(sequence, prior) < MAXIMUM_IDENTITY for prior in chosen):
            chosen.append(sequence)
            if len(chosen) == count:
                break
    # Mirror the publisher: after the preferred 0.94 separation is exhausted,
    # fill farthest-first up to (but never including) the validator's 0.95
    # duplicate line.  Returning fewer than ten is legitimate and is reflected
    # in the per-entry denominator below.
    remaining = [sequence for sequence in ranked if sequence not in chosen]
    while remaining and len(chosen) < count:
        pick = min(
            remaining,
            key=lambda sequence: max(
                (identity(sequence, prior) for prior in chosen), default=0.0
            ),
        )
        if any(identity(pick, prior) >= 0.95 for prior in chosen):
            break
        chosen.append(pick)
        remaining.remove(pick)
    if not chosen:
        raise RuntimeError("ranking yielded no usable fleet entry")
    return chosen


def observation_scores(
    sequences: Iterable[str],
    draw_indices: Sequence[int],
    table: Mapping[str, np.ndarray],
    sequence_index: Mapping[str, int],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for sequence in sequences:
        row = sequence_index[sequence]
        result[sequence] = np.median(
            table["scores"][list(draw_indices), row, :], axis=1
        ).astype(float).tolist()
    return result


def rank_internal(
    alive: Sequence[str],
    observed: Mapping[str, Sequence[int]],
    table: Mapping[str, np.ndarray],
    sequence_index: Mapping[str, int],
) -> list[str]:
    candidates = []
    for sequence in alive:
        scores = observation_scores(
            [sequence], observed[sequence], table, sequence_index
        )[sequence]
        candidates.append({
            "sequence": sequence,
            "per_observation_median_rank_sum": scores,
            "score_median": statistics.median(scores),
        })
    return [item["sequence"] for item in rank_by_fresh_draw_probability(candidates)]


def rank_external(
    alive: Sequence[str],
    observed: Mapping[str, Sequence[int]],
    table: Mapping[str, np.ndarray],
    sequence_index: Mapping[str, int],
) -> list[str]:
    """Rank the event that each ticket actually earns: beating the field."""
    rows = []
    for sequence in alive:
        row = sequence_index[sequence]
        draw_indices = list(observed[sequence])
        scores = table["scores"][draw_indices, row, :].reshape(-1)
        wins = int(table["wins"][draw_indices, row, :].sum())
        top10 = int(table["top10"][draw_indices, row, :].sum())
        n = int(scores.size)
        # Jeffreys smoothing makes a 0/N result comparable at unequal depths.
        win_rate = (wins + 0.5) / (n + 1.0)
        top10_rate = (top10 + 0.5) / (n + 1.0)
        rows.append((
            -win_rate,
            -top10_rate,
            statistics.fmean(scores),
            statistics.median(scores),
            sequence,
        ))
    rows.sort()
    return [row[-1] for row in rows]


RANKERS = {"current_internal": rank_internal, "field_win": rank_external}


def replay(
    sequences: Sequence[str],
    train: Sequence[int],
    table: Mapping[str, np.ndarray],
    sequence_index: Mapping[str, int],
    policy: Sequence[tuple[int, int | None]],
    ranker_name: str,
) -> dict[str, Any]:
    ranker = RANKERS[ranker_name]
    alive = list(sequences)
    observed: dict[str, list[int]] = {sequence: [train[0]] for sequence in sequences}
    cursor = 1
    trace = []
    repeat_observations = 0
    for rounds, retain in policy:
        indices = list(train[cursor:cursor + rounds])
        if len(indices) != rounds:
            raise RuntimeError("policy consumes more than twelve training draws")
        cursor += rounds
        for sequence in alive:
            observed[sequence].extend(indices)
        repeat_observations += len(alive) * rounds
        ranked = ranker(alive, observed, table, sequence_index)
        trace.append({
            "alive_before_prune": len(alive),
            "rounds": rounds,
            "observations_per_survivor": len(observed[alive[0]]),
            "retain": retain,
        })
        if retain is not None:
            alive = family_slate(ranked, retain)
    ranked = ranker(alive, observed, table, sequence_index)
    selected = fleet(ranked)
    return {
        "fleet": selected,
        "alive": len(alive),
        "repeat_observations": repeat_observations,
        "model_loads": cursor - 1,
        "survivor_observations": min(len(observed[sequence]) for sequence in alive),
        "trace": trace,
    }


def evaluate(
    chosen: Sequence[str],
    test: Sequence[int],
    table: Mapping[str, np.ndarray],
    sequence_index: Mapping[str, int],
) -> dict[str, Any]:
    wins = 0
    top10 = 0
    fleet_wins = 0
    field_count = table["scores"].shape[2]
    entries = len(chosen) * len(test) * field_count
    by_draw = []
    for draw_index in test:
        draw_wins = 0
        draw_top10 = 0
        draw_fleet_wins = 0
        rows = [sequence_index[sequence] for sequence in chosen]
        block_wins = table["wins"][draw_index, rows, :]
        block_top10 = table["top10"][draw_index, rows, :]
        draw_wins = int(block_wins.sum())
        draw_top10 = int(block_top10.sum())
        draw_fleet_wins = int(block_wins.any(axis=0).sum())
        wins += draw_wins
        top10 += draw_top10
        fleet_wins += draw_fleet_wins
        by_draw.append({
            "draw": draw_index + 1,
            "wins": draw_wins,
            "top10": draw_top10,
            "fleet_wins": draw_fleet_wins,
        })
    return {
        "entries": entries,
        "wins": wins,
        "win_per_entry": wins / entries,
        "top10": top10,
        "top10_per_entry": top10 / entries,
        "fleet_field_opportunities": len(test) * field_count,
        "fleet_wins": fleet_wins,
        "fleet_win_rate": fleet_wins / (len(test) * field_count),
        "by_draw": by_draw,
    }


def main() -> None:
    pools = json.loads((PROOF / "pools.json").read_text())
    sequences = list(pools["A_current"])
    draws = [load_draw(path) for path in DRAW_PATHS]
    if len(sequences) != 282 or any(not set(sequences).issubset(draw) for draw in draws):
        raise RuntimeError("frozen pool/draw mismatch")
    development_fields = json.loads((PROOF / "development_fields.json").read_text())
    reserved_fields = [
        json.loads(path.read_text())
        for path in sorted((PROOF / "reserved_fields").glob("epoch_*.json"))
    ]
    sequence_index = {sequence: index for index, sequence in enumerate(sequences)}
    development_table = outcome_table(sequences, draws, development_fields)
    reserved_table = outcome_table(sequences, draws, reserved_fields)

    folds = []
    for fold in range(4):
        test = list(range(4 * fold, 4 * fold + 4))
        train = [index for index in range(16) if index not in test]
        fold_result: dict[str, Any] = {
            "fold": fold + 1,
            "train_draws": [index + 1 for index in train],
            "test_draws": [index + 1 for index in test],
            "arms": {},
        }
        for ranker_name in RANKERS:
            for policy_name, policy in POLICIES.items():
                replayed = replay(
                    sequences, train, development_table, sequence_index, policy, ranker_name
                )
                tested = evaluate(
                    replayed["fleet"], test, reserved_table, sequence_index
                )
                fold_result["arms"][f"{ranker_name}/{policy_name}"] = {
                    **replayed,
                    "heldout": tested,
                }
        folds.append(fold_result)

    summary = {}
    arms = sorted(folds[0]["arms"])
    for arm in arms:
        records = [fold["arms"][arm] for fold in folds]
        entries = sum(record["heldout"]["entries"] for record in records)
        wins = sum(record["heldout"]["wins"] for record in records)
        top10 = sum(record["heldout"]["top10"] for record in records)
        fleet_opportunities = sum(
            record["heldout"]["fleet_field_opportunities"] for record in records
        )
        fleet_wins = sum(record["heldout"]["fleet_wins"] for record in records)
        summary[arm] = {
            "repeat_observations": records[0]["repeat_observations"],
            "model_loads": records[0]["model_loads"],
            "survivor_observations": records[0]["survivor_observations"],
            "entries": entries,
            "wins": wins,
            "win_per_entry": wins / entries,
            "top10": top10,
            "top10_per_entry": top10 / entries,
            "fleet_field_opportunities": fleet_opportunities,
            "fleet_wins": fleet_wins,
            "fleet_win_rate": fleet_wins / fleet_opportunities,
            "win_folds": [record["heldout"]["wins"] for record in records],
            "top10_folds": [record["heldout"]["top10"] for record in records],
            "fleet_overlap_by_fold": [
                len(set(records[left]["fleet"]) & set(records[right]["fleet"]))
                for left in range(len(records))
                for right in range(left + 1, len(records))
            ],
        }

    baseline = summary["current_internal/current_4_4_3"]
    for arm, record in summary.items():
        record["delta_vs_live_shape"] = {
            "wins": record["wins"] - baseline["wins"],
            "top10": record["top10"] - baseline["top10"],
            "fleet_wins": record["fleet_wins"] - baseline["fleet_wins"],
        }

    output = {
        "experiment": "fixed-budget-allocation-crossvalidated-replay-v1",
        "candidate_pool": "A_current",
        "candidate_count": len(sequences),
        "fleet_size": FLEET_SIZE,
        "repeat_budget": REPEAT_BUDGET,
        "development_field_epochs": [field["epoch"] for field in development_fields],
        "reserved_field_epochs": [field["epoch"] for field in reserved_fields],
        "draw_paths": [str(path.relative_to(ROOT)) for path in DRAW_PATHS],
        "policies": {
            name: [[rounds, retain] for rounds, retain in stages]
            for name, stages in POLICIES.items()
        },
        "summary": summary,
        "folds": folds,
    }
    (HERE / "analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    for arm, record in sorted(
        summary.items(), key=lambda item: (-item[1]["wins"], -item[1]["top10"])
    ):
        print(
            f"{arm:36s} obs={record['repeat_observations']:4d} "
            f"loads={record['model_loads']:2d} n={record['survivor_observations']:2d} "
            f"wins={record['wins']:3d}/{record['entries']} "
            f"top10={record['top10']:4d}/{record['entries']} "
            f"fleet={record['fleet_wins']:3d}/{record['fleet_field_opportunities']}"
        )


if __name__ == "__main__":
    main()
