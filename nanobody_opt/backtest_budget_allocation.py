"""Measure exact-observation noise and choose a breadth/replication budget split.

The validator publishes a dense rank sum over ten metrics against the competitor
field of one epoch.  One BoltzGen observation of a fixed sequence is a draw from
a distribution around that sequence's mean metric vector, so the published score
is a noisy realisation even when the sequence is fixed.

This module measures three quantities from evidence that already exists:

1.  ``sigma_observation``: the spread, in leaderboard rank points, of repeated
    exact observations of one sequence scored into that epoch's real field.
2.  ``sigma_quality``: the spread of true candidate quality across a screened
    pool, recovered by subtracting observation variance from the observed
    one-shot spread.
3.  the expected published score of a two-stage policy that screens ``K``
    candidates once, replicates a shortlist of ``s``, and publishes the best
    pooled mean, for a fixed observation budget.

Observations are read from live race artifacts (``exact_all.json``); competitor
fields are read from cached leaderboard payloads.  Nothing here re-runs
BoltzGen, so the backtest is cheap and repeatable.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nanobody_opt.rank_exact import leaderboard_field
from nanobody_opt.simulate_round import METRICS, dense_rank_sums, rank_sum_in_field

METRIC_NAMES = list(METRICS)
_EXACT_PATTERN = re.compile(
    r"epoch(?P<epoch>\d+)/producer/iteration_(?P<iteration>\d+)/race/"
    r"(?:screen/tranche_(?P<tranche>\d+)|candidates/(?P<candidate>[0-9a-f]+)/"
    r"repeat_(?P<repeat>\d+))/exact_all\.json$"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observation_rows(document: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    """Return ``(sequence, metrics)`` pairs from one ``exact_all.json``."""
    rows: list[tuple[str, dict[str, float]]] = []
    components = document.get("per_nanobody_components", {})
    if not isinstance(components, dict):
        return rows
    for wave in components.values():
        if not isinstance(wave, dict):
            continue
        for sequence, targets in wave.items():
            if not isinstance(targets, dict):
                continue
            for metrics in targets.values():
                if not isinstance(metrics, dict):
                    continue
                if not all(name in metrics for name in METRIC_NAMES):
                    continue
                rows.append(
                    (sequence, {name: float(metrics[name]) for name in METRIC_NAMES})
                )
    return rows


def collect_observations(live_root: Path) -> pd.DataFrame:
    """Read every live exact observation under ``live_root``.

    Screen tranches contribute one observation per candidate; repeat directories
    contribute the independent observations of a replicated finalist.
    """
    records: list[dict[str, Any]] = []
    pattern = str(live_root / "epoch*/producer/iteration_*/race/**/exact_all.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        match = _EXACT_PATTERN.search(path)
        if match is None:
            continue
        try:
            document = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        kind = "screen" if match.group("tranche") else "repeat"
        for sequence, metrics in _observation_rows(document):
            records.append(
                {
                    "epoch": int(match.group("epoch")),
                    "iteration": int(match.group("iteration")),
                    "kind": kind,
                    "replicate": int(match.group("repeat") or 1),
                    "protocol": str(document.get("scorer_protocol", "legacy")),
                    "sequence": sequence,
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(records)


def load_fields(
    field_dir: Path, target: str, *, exclude_uid: int | None = None
) -> dict[int, pd.DataFrame]:
    """Load cached leaderboard payloads as per-epoch competitor fields.

    ``exclude_uid`` removes our own submission, so a candidate is scored against
    the competitors it would actually have faced rather than against a field
    that already contains the sequence we published.
    """
    fields: dict[int, pd.DataFrame] = {}
    for path in sorted(field_dir.glob("epoch*.json")):
        epoch = int(path.stem.replace("epoch", ""))
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        if exclude_uid is not None and isinstance(payload, dict):
            payload = dict(payload)
            payload["data"] = [
                submission
                for submission in payload.get("data", [])
                if not (
                    isinstance(submission, dict)
                    and submission.get("uid") == exclude_uid
                )
            ]
        frame = leaderboard_field(payload, target)
        if len(frame) >= 20:
            fields[epoch] = frame
    return fields


def published_ranks(field_dir: Path, target: str, uid: int) -> list[int]:
    """Leaderboard ranks actually published for ``uid``, newest epoch last."""
    ranks: list[int] = []
    for path in sorted(field_dir.glob("epoch*.json")):
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        for submission in payload.get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(submission, dict) or submission.get("uid") != uid:
                continue
            if not any(
                isinstance(nanobody, dict)
                and str(nanobody.get("protein_name")) == target
                for nanobody in submission.get("nanobodies", [])
            ):
                continue
            rank = submission.get("nanobody_leaderboard_rank")
            if rank is not None:
                ranks.append(int(rank))
    return ranks


def measured_policy(observations: pd.DataFrame) -> dict[str, float]:
    """What the live producer actually spends per epoch, not what it is configured to."""
    screens = observations[observations.kind == "screen"]
    repeats = observations[observations.kind == "repeat"]
    screened = screens.groupby("epoch").sequence.nunique()
    replicated = repeats.groupby("epoch").sequence.nunique()
    per_finalist = repeats.groupby(["epoch", "sequence"]).size()
    return {
        "epochs": int(screened.size),
        "breadth": float(screened.mean()),
        "shortlist": float(replicated.reindex(screened.index).fillna(0).mean()),
        "replicates": float(per_finalist.mean()),
        "observations": float(
            screened.mean()
            + replicated.reindex(screened.index).fillna(0).mean()
            * (per_finalist.mean() - 1)
        ),
    }


def _score_into_field(frame: pd.DataFrame, field: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            rank_sum_in_field(
                {name: float(row[name]) for name in METRIC_NAMES},
                field,
            )
            for _, row in frame.iterrows()
        ]
    )


def observation_noise(
    observations: pd.DataFrame,
    fields: dict[int, pd.DataFrame],
    *,
    minimum_replicates: int = 5,
) -> dict[str, Any]:
    """Spread of repeated observations of one sequence, in rank points."""
    spreads: list[float] = []
    groups = 0
    repeats = observations[observations.kind == "repeat"]
    for (epoch, sequence), group in repeats.groupby(["epoch", "sequence"]):
        field = fields.get(int(epoch))
        if field is None or len(group) < minimum_replicates:
            continue
        scores = _score_into_field(group, field)
        spreads.append(float(scores.std(ddof=1)))
        groups += 1
    if not spreads:
        raise ValueError("no replicated sequence has a cached competitor field")
    return {
        "groups": groups,
        # The median of the per-sequence spreads is the reported estimate: a
        # handful of replicated finalists have unusually unstable folds, and the
        # mean lets those few groups set the noise level for every candidate.
        "sigma_observation": float(np.median(spreads)),
        "sigma_observation_mean": float(np.mean(spreads)),
    }


def pool_spread(
    observations: pd.DataFrame,
    fields: dict[int, pd.DataFrame],
    sigma_observation: float,
) -> dict[str, Any]:
    """Observed and noise-corrected spread of screened candidate quality."""
    observed: list[float] = []
    means: list[float] = []
    per_epoch: dict[str, dict[str, float]] = {}
    screens = observations[observations.kind == "screen"].drop_duplicates(
        ["epoch", "sequence"]
    )
    for epoch, group in screens.groupby("epoch"):
        field = fields.get(int(epoch))
        if field is None or len(group) < 40:
            continue
        scores = _score_into_field(group, field)
        observed.append(float(scores.std(ddof=1)))
        means.append(float(scores.mean()))
        per_epoch[str(int(epoch))] = {
            "candidates": int(len(group)),
            "mean_rank_sum": float(scores.mean()),
            "sd_rank_sum": float(scores.std(ddof=1)),
        }
    sd_observed = float(np.mean(observed))
    sigma_quality = float(np.sqrt(max(sd_observed**2 - sigma_observation**2, 0.0)))
    return {
        "epochs": len(observed),
        "pool_mean_rank_sum": float(np.mean(means)),
        "sd_observed": sd_observed,
        "sigma_quality": sigma_quality,
        "one_observation_correlation": float(
            sigma_quality / np.hypot(sigma_quality, sigma_observation)
        ),
        "per_epoch": per_epoch,
    }


def two_stage_expectation(
    breadth: int,
    shortlist: int,
    replicates: int,
    *,
    sigma_quality: float,
    sigma_observation: float,
    sigma_screen: float | None = None,
    trials: int = 4000,
    seed: int = 7,
) -> float:
    """Expected published score of screen-then-replicate, relative to pool mean.

    ``breadth`` candidates are observed once, the best ``shortlist`` by that
    single observation receive ``replicates - 1`` further observations, and the
    best pooled mean is published.  The validator then draws its own unbiased
    observation, so the expectation of the published score is the true quality
    of the selected candidate.

    ``sigma_screen`` lets the screening pass use a cheaper, noisier measurement
    than the replication pass.  A screening measurement only has to rank, so a
    faster setting with a constant offset is admissible there; the replication
    measurement is compared against absolute qualification gates and against the
    validator's own scale, so it must stay on the validated setting.  Screen and
    replication observations are therefore never pooled across settings: only
    the ``replicates - 1`` confirmation observations decide the selection.
    """
    if shortlist > breadth or shortlist < 1 or replicates < 1:
        raise ValueError("invalid allocation")
    screen_noise = sigma_observation if sigma_screen is None else sigma_screen
    mixed_scales = sigma_screen is not None
    if mixed_scales and replicates < 2:
        raise ValueError(
            "a cheaper screening scale requires at least one confirmation "
            "observation on the validated scale"
        )
    rng = np.random.default_rng(seed)
    selected = np.empty(trials)
    for index in range(trials):
        quality = rng.normal(0.0, sigma_quality, breadth)
        first = quality + rng.normal(0.0, screen_noise, breadth)
        keep = np.argsort(first)[:shortlist]
        if replicates > 1:
            extra = quality[keep][:, None] + rng.normal(
                0.0, sigma_observation, (shortlist, replicates - 1)
            )
            pooled = (
                extra.mean(axis=1)
                if mixed_scales
                else (first[keep] + extra.sum(axis=1)) / replicates
            )
        else:
            pooled = first[keep]
        selected[index] = quality[keep][int(np.argmin(pooled))]
    return float(selected.mean())


def allocation_grid(
    budget: int,
    *,
    sigma_quality: float,
    sigma_observation: float,
    trials: int = 2500,
) -> list[dict[str, Any]]:
    """Search breadth/shortlist/replicate splits of a fixed observation budget."""
    results: list[dict[str, Any]] = []
    for fraction in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        breadth = max(int(budget * fraction), 1)
        remaining = budget - breadth
        for shortlist in (1, 5, 8, 12, 16, 24, 32, 40, 60, 80, 120):
            if shortlist > breadth:
                continue
            replicates = 1 + (remaining // shortlist if shortlist else 0)
            if replicates < 1 or breadth + shortlist * (replicates - 1) > budget:
                continue
            if shortlist > 1 and replicates == 1:
                continue
            results.append(
                {
                    "breadth": breadth,
                    "shortlist": shortlist,
                    "replicates": replicates,
                    "observations": breadth + shortlist * (replicates - 1),
                    "expected_published_minus_pool_mean": two_stage_expectation(
                        breadth,
                        shortlist,
                        replicates,
                        sigma_quality=sigma_quality,
                        sigma_observation=sigma_observation,
                        trials=trials,
                    ),
                }
            )
    results.sort(key=lambda item: item["expected_published_minus_pool_mean"])
    return results


def wall_clock_plan(
    *,
    sigma_quality: float,
    sigma_observation: float,
    load_seconds: float,
    seconds_per_sequence: float,
    gpu_count: int,
    window_seconds: float,
    bookkeeping_seconds: float,
    overlap: bool,
    tranche_sizes: tuple[int, ...] = (150, 200),
    breadths: tuple[int, ...] = (600, 800, 1000, 1200, 1400, 1600, 1800, 2000),
    shortlists: tuple[int, ...] = (16, 24, 32, 40, 60, 150),
    trials: int = 2500,
) -> list[dict[str, Any]]:
    """Best screen/replicate split that actually fits one epoch's wall clock.

    An observation budget is not free: every scorer invocation pays a model load
    before its first fold, so the tranche must be large enough to amortise it,
    and the per-tranche ranking and database work either overlaps the next
    tranche's folding or is added to the critical path.
    """

    def invocation_seconds(count: int) -> float:
        return load_seconds + seconds_per_sequence * count / gpu_count

    def wall(breadth: int, tranche: int, shortlist: int, rounds: int) -> float:
        invocations = -(-breadth // tranche)
        screen = sum(
            invocation_seconds(min(tranche, breadth - index * tranche))
            for index in range(invocations)
        )
        serial = 0.0 if overlap else invocations * bookkeeping_seconds
        return screen + serial + rounds * invocation_seconds(shortlist)

    plans: list[dict[str, Any]] = []
    for tranche in tranche_sizes:
        for breadth in breadths:
            for shortlist in shortlists:
                if shortlist > breadth:
                    continue
                rounds = 0
                while wall(breadth, tranche, shortlist, rounds + 1) <= window_seconds:
                    rounds += 1
                if rounds < 1:
                    continue
                plans.append(
                    {
                        "breadth": breadth,
                        "tranche": tranche,
                        "shortlist": shortlist,
                        "replication_rounds": rounds,
                        "replicates": rounds + 1,
                        "observations": breadth + shortlist * rounds,
                        "sequences_per_shard": shortlist / gpu_count,
                        "wall_seconds": round(
                            wall(breadth, tranche, shortlist, rounds), 1
                        ),
                        "expected_published_minus_pool_mean": two_stage_expectation(
                            breadth,
                            shortlist,
                            rounds + 1,
                            sigma_quality=sigma_quality,
                            sigma_observation=sigma_observation,
                            trials=trials,
                        ),
                    }
                )
    plans.sort(key=lambda item: item["expected_published_minus_pool_mean"])
    return plans


def rank_outcome(
    expected_score: float,
    fields: dict[int, pd.DataFrame],
    sigma_observation: float,
    *,
    trials: int = 40000,
    seed: int = 11,
) -> dict[str, float]:
    """Rank distribution of one submission whose expected score is known."""
    rng = np.random.default_rng(seed)
    pools = [dense_rank_sums(field) for field in fields.values()]
    ranks = np.empty(trials)
    for index in range(trials):
        pool = pools[int(rng.integers(len(pools)))]
        draw = rng.normal(expected_score, sigma_observation)
        ranks[index] = 1 + int((pool < draw).sum())
    return {
        "expected_score": float(expected_score),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "probability_rank_1": float((ranks == 1).mean()),
        "probability_top_3": float((ranks <= 3).mean()),
        "probability_top_5": float((ranks <= 5).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-root",
        type=Path,
        help="directory holding epoch*/producer/... live race artifacts",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        help="previously collected observation CSV instead of --live-root",
    )
    parser.add_argument("--write-observations", type=Path)
    parser.add_argument("--field-dir", type=Path, required=True)
    parser.add_argument("--target", default="P20809")
    parser.add_argument(
        "--budget",
        type=int,
        action="append",
        help="observation budgets to search; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-uid", type=int, default=8)
    parser.add_argument("--scorer-load-seconds", type=float, default=33.0)
    parser.add_argument("--scorer-seconds-per-sequence", type=float, default=14.5)
    parser.add_argument("--gpu-count", type=int, default=10)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=3672.0,
        help="scoring time in one epoch after the commitment reserve",
    )
    parser.add_argument("--bookkeeping-seconds", type=float, default=84.0)
    args = parser.parse_args()

    if args.live_root is None and args.observations is None:
        parser.error("pass --live-root or --observations")
    observations = (
        collect_observations(args.live_root)
        if args.live_root is not None
        else pd.read_csv(args.observations)
    )
    if observations.empty:
        parser.error("no exact observations found")
    if args.write_observations is not None:
        args.write_observations.parent.mkdir(parents=True, exist_ok=True)
        observations.to_csv(args.write_observations, index=False)

    fields = load_fields(args.field_dir, args.target, exclude_uid=args.exclude_uid)
    if not fields:
        parser.error("no cached competitor fields found")

    noise = observation_noise(observations, fields)
    pool = pool_spread(observations, fields, noise["sigma_observation"])
    budgets = args.budget or [170, 350, 600, 1200, 2000, 3200]
    grids = {
        str(budget): allocation_grid(
            budget,
            sigma_quality=pool["sigma_quality"],
            sigma_observation=noise["sigma_observation"],
        )[:5]
        for budget in budgets
    }
    outcomes = {
        budget: rank_outcome(
            pool["pool_mean_rank_sum"]
            + grid[0]["expected_published_minus_pool_mean"],
            fields,
            noise["sigma_observation"],
        )
        for budget, grid in grids.items()
    }
    plans = {
        overlap: wall_clock_plan(
            sigma_quality=pool["sigma_quality"],
            sigma_observation=noise["sigma_observation"],
            load_seconds=args.scorer_load_seconds,
            seconds_per_sequence=args.scorer_seconds_per_sequence,
            gpu_count=args.gpu_count,
            window_seconds=args.window_seconds,
            bookkeeping_seconds=args.bookkeeping_seconds,
            overlap=overlap == "overlapped",
        )[:5]
        for overlap in ("overlapped", "serial")
    }
    measured = measured_policy(observations)
    measured_expectation = two_stage_expectation(
        max(int(measured["breadth"]), 1),
        max(int(round(measured["shortlist"])), 1),
        max(int(round(measured["replicates"])), 1),
        sigma_quality=pool["sigma_quality"],
        sigma_observation=noise["sigma_observation"],
    )
    payload = {
        "target": args.target,
        "measured_policy": {
            **measured,
            "expected_published_minus_pool_mean": measured_expectation,
            "rank_outcome": rank_outcome(
                pool["pool_mean_rank_sum"] + measured_expectation,
                fields,
                noise["sigma_observation"],
            ),
            "published_ranks": published_ranks(
                args.field_dir, args.target, args.exclude_uid
            ),
        },
        "wall_clock_plans": plans,
        "wall_clock_model": {
            "scorer_load_seconds": args.scorer_load_seconds,
            "scorer_seconds_per_sequence": args.scorer_seconds_per_sequence,
            "gpu_count": args.gpu_count,
            "window_seconds": args.window_seconds,
            "bookkeeping_seconds": args.bookkeeping_seconds,
        },
        "epochs_with_fields": sorted(fields),
        "observation_counts": {
            "screen": int((observations.kind == "screen").sum()),
            "repeat": int((observations.kind == "repeat").sum()),
            "sequences": int(observations.sequence.nunique()),
        },
        "observation_noise": noise,
        "pool_spread": pool,
        "allocation_grid": grids,
        "rank_outcome": outcomes,
    }
    payload["result_sha256"] = _sha256(json.dumps(payload, sort_keys=True))
    _atomic_json(args.output, payload)
    print(
        f"sigma_observation={noise['sigma_observation']:.1f} "
        f"sigma_quality={pool['sigma_quality']:.1f} "
        f"pool_mean={pool['pool_mean_rank_sum']:.1f}",
        flush=True,
    )
    plan = plans["overlapped"][0]
    print(
        f"wall-clock plan: breadth={plan['breadth']} tranche={plan['tranche']} "
        f"shortlist={plan['shortlist']} replicates={plan['replicates']} "
        f"observations={plan['observations']} "
        f"expected={plan['expected_published_minus_pool_mean']:.1f} "
        f"(serial: {plans['serial'][0]['expected_published_minus_pool_mean']:.1f}, "
        f"measured today: {measured_expectation:.1f})",
        flush=True,
    )
    for budget, grid in grids.items():
        best = grid[0]
        print(
            f"budget={budget:>5} breadth={best['breadth']:>5} "
            f"shortlist={best['shortlist']:>3} replicates={best['replicates']:>3} "
            f"expected={best['expected_published_minus_pool_mean']:.1f} "
            f"median_rank={outcomes[budget]['median_rank']:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
