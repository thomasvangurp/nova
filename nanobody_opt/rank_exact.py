"""Rank exact nanobody observations with the validator's dense-rank objective.

BoltzGen inference is stochastic, so a live finalist must survive more than one
exact observation. This utility applies the validator's per-metric dense ranks
to historical live fields. Selection uses the complete observation-by-field
score distribution: first the probability of a competitive score, then its
upper tail. Lower is better. These are robustness statistics, not a prediction
of the next stochastic validator observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt import publication_gates
from nanobody_opt.local_neighborhood import (
    FIELD_CONSENSUS_THRESHOLD,
    field_consensus,
)
from nanobody_opt.simulate_round import (
    METRICS,
    METRIC_CATEGORIES,
)
from nanobody_opt.win_probability import rank_by_win_probability
from nanobody_opt.pose_stability import (
    exact_sample_selections,
    exact_structure_paths,
    residue_numbers,
    summarize_sequence_structures,
)


#: How much of the field's top the validator dedupes against.
CROWD_LIMIT = 50
LEADERBOARD_API_ROOT = "https://compound-api.metanova-labs.ai"


def _compile_validator_field(field: pd.DataFrame) -> dict[str, np.ndarray]:
    """Precompute the unique values that define each dense-rank scale."""
    return {
        metric: np.sort(np.unique(
            pd.to_numeric(field[metric], errors="coerce")
            .dropna()
            .to_numpy(dtype=np.float64)
        ))
        for metric in METRICS
    }


# Selection orders on the plain sum of dense ranks, the same quantity the
# validator's own score is built from. A weighted variant ran for eleven epochs
# in two shapes and is reverted.
#
# What it was for: over 25103-25112 our median category ranks against the field
# were level on developability (-0.2) and physical interaction (-0.7) and behind
# on confidence (+29.2, worse in 9 of 10). Since the validator's score is the sum
# of the three, confidence was the whole gap, and within it the rank is driven by
# interaction_pae (Spearman +0.81 against confidence_rank over 494 pooled scored
# submissions), min_design_to_target_pae (+0.73) and design_to_target_iptm
# (-0.73). Leaning the ordering on those looked obvious.
#
# What happened: the gap did not close and the ranks got worse. Across the three
# configurations -- unweighted, weighted, and weighted plus the generator fixes --
# best-rank medians were 3, 8 and 18, mean confidence gaps +3, +27 and +25, and
# the single win in that stretch belongs to the unweighted ordering.
#
# That is not proof the weighting caused it: the unweighted baseline is only
# three epochs, an outage broke the series, and each epoch is one draw. But
# eleven epochs across two weighted shapes produced no win and no improvement in
# the very quantity they targeted, so the weighting is not earning its
# complexity and the ordering goes back to what produced 25115.
#
# The likelier explanation is upstream anyway. Our local screen's rank
# correlation with the validator is +0.139 (n=55 molecules scored by both), and
# our own screen re-measuring the SAME molecule agrees with itself at only +0.373
# (n=3,963, median spread 55 rank points). Reweighting an ordering built on a
# signal that noisy cannot do much, whatever the weights.


def _compiled_metric_ranks(
    metrics: dict[str, float], compiled: dict[str, np.ndarray]
) -> dict[str, float]:
    """Each metric's dense rank in its field, the shared basis for both sums."""
    ranks: dict[str, float] = {}
    for metric, direction in METRICS.items():
        values = compiled[metric]
        value = float(metrics[metric])
        if direction == "min":
            rank = int(np.searchsorted(values, value, side="left")) + 1
        else:
            rank = len(values) - int(
                np.searchsorted(values, value, side="right")
            ) + 1
        ranks[metric] = float(rank)
    return ranks


def _compiled_category_rank_sums(
    metrics: dict[str, float], compiled: dict[str, np.ndarray]
) -> dict[str, float]:
    """Exactly reproduce pandas dense ranks without rebuilding DataFrames.

    A candidate's ascending dense rank is one plus the number of distinct field
    values below it; descending is one plus the number above it.  Computing
    those counts with binary search is equivalent to the validator's
    ``Series.rank(method="dense")`` and makes runtime scale with observations,
    not pandas object construction.
    """
    ranks = _compiled_metric_ranks(metrics, compiled)
    return {
        category: float(sum(ranks[metric] for metric in names))
        for category, names in METRIC_CATEGORIES.items()
    }


def settled_deviation_count(sequence: str, consensus: dict) -> int:
    """Settled positions this sequence disagrees with the field on.

    Measured over 1,678 live tickets, each one costs about 7-12 rank points, and a
    molecule carrying none averages final_score 132.7 -- better than the leading
    rival's own mean.  Ranking has been blind to this: it orders on screened
    metrics, which agree with the validator at r=0.19, while this structural
    signal is measured at 14 points per unit.  We already generate ~27% of
    candidates that break nothing and submit ones averaging 1.21.
    """
    count = 0
    for position, (residue, share) in consensus.items():
        if share < FIELD_CONSENSUS_THRESHOLD or not residue:
            continue
        if position < len(sequence) and sequence[position] != residue:
            count += 1
    return count


def _observations(paths: list[Path], target: str) -> dict[str, list[dict[str, float]]]:
    found: dict[str, list[dict[str, float]]] = defaultdict(list)
    seen_observations: set[str] = set()
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if str(payload.get("target")) != target:
            raise SystemExit(f"{path} targets {payload.get('target')!r}, not {target}")
        observation_id = str(payload.get("observation_id") or "").strip()
        source_key = observation_id or hashlib.sha256(raw).hexdigest()
        if source_key in seen_observations:
            continue
        seen_observations.add(source_key)
        for by_sequence in payload.get("per_nanobody_components", {}).values():
            for sequence, by_target in by_sequence.items():
                metrics = by_target.get(target)
                if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
                    continue
                values = {name: float(metrics[name]) for name in METRICS}
                if all(math.isfinite(value) for value in values.values()):
                    found[str(sequence)].append(values)
    return found


def _excluded_sequences(paths: list[Path]) -> set[str]:
    """Read newline-delimited sequence ledgers, ignoring comments and blanks."""
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        excluded.update(
            line.strip().upper()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return excluded


def leaderboard_field(payload: Any, target: str) -> pd.DataFrame:
    """Flatten one dashboard response into the validator's unique metric field."""
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return pd.DataFrame(columns=list(METRICS))
    by_hash: dict[str, dict[str, float]] = {}
    for submission in payload.get("data", []):
        if not isinstance(submission, dict):
            continue
        for nanobody in submission.get("nanobodies", []):
            if (
                not isinstance(nanobody, dict)
                or str(nanobody.get("protein_name")) != target
                or not set(METRICS).issubset(nanobody)
            ):
                continue
            try:
                metrics = {name: float(nanobody[name]) for name in METRICS}
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in metrics.values()):
                continue
            sequence_hash = str(nanobody.get("sequence_hash") or "")
            sequence = str(nanobody.get("sequence") or "").strip().upper()
            key = sequence_hash or sequence
            if key:
                by_hash.setdefault(key, metrics)
    return pd.DataFrame(by_hash.values(), columns=list(METRICS))


def leaderboard_crowd(payload: Any, target: str, limit: int) -> list[str]:
    """The best `limit` sequences of one epoch's field, best first.

    The validator DEDUPES against the top of the field, so a ticket that sits
    within the dedupe identity of one of these is discarded before it is scored
    and the registration that carried it is wasted.  Measured on epoch 25138's
    committed fleet, 6 of 7 tickets sat at or above 0.95 positional identity to
    the previous epoch's top-50 -- which is why we commit seven and see two to
    five scored.  Positional identity understates it: the validator aligns, and
    an alignment can only find more.
    """
    scored: list[tuple[float, str]] = []
    for item in (payload.get("data") or []) if isinstance(payload, Mapping) else []:
        if not isinstance(item, Mapping):
            continue
        for nanobody in item.get("nanobodies") or []:
            if not isinstance(nanobody, Mapping):
                continue
            if str(nanobody.get("protein_name") or target) != target:
                continue
            sequence = str(nanobody.get("sequence") or "").strip().upper()
            score = nanobody.get("final_score")
            if sequence and isinstance(score, (int, float)):
                scored.append((float(score), sequence))
    scored.sort()
    seen: set[str] = set()
    crowd: list[str] = []
    for _, sequence in scored:
        if sequence not in seen:
            seen.add(sequence)
            crowd.append(sequence)
        if len(crowd) >= limit:
            break
    return crowd


def fetch_leaderboard_fields(
    api_root: str,
    *,
    before_epoch: int,
    target: str,
    count: int,
    scan_limit: int,
) -> tuple[list[tuple[int, pd.DataFrame]], list[str]]:
    """Fetch recent fields, and the most recent field's top sequences."""
    fields: list[tuple[int, pd.DataFrame]] = []
    crowd: list[str] = []
    root = api_root.rstrip("/")
    for epoch in range(before_epoch - 1, max(before_epoch - scan_limit - 1, 0), -1):
        url = f"{root}/api/competitions/leaderboard/{epoch}/nanobodies"
        request = Request(url, headers={"User-Agent": "nova-nanobody-ranker/1"})
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        field = leaderboard_field(payload, target)
        if len(field) < 2:
            continue
        if not crowd:
            # The newest field we could read is the best available proxy for the
            # one we are about to be judged against; everyone is mutating the
            # same winners, so the crowd is stable epoch to epoch.
            crowd = leaderboard_crowd(payload, target, CROWD_LIMIT)
        fields.append((epoch, field))
        if len(fields) >= count:
            break
    return fields, crowd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", action="append", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--minimum-observations",
        type=int,
        default=int(publication_gates.gate_defaults()["--minimum-observations"]),
        help=(
            "observations a candidate needs before it may be selected; relaxed "
            "one at a time if no candidate reaches it, never below one"
        ),
    )
    parser.add_argument(
        "--competitive-score-threshold",
        type=float,
        default=75.0,
        help="rank sum treated as competitive in historical live fields",
    )
    parser.add_argument(
        "--minimum-competitive-rate",
        type=float,
        default=0.0,
        help="fail closed below this observation-by-field success fraction",
    )
    parser.add_argument(
        "--minimum-observation-competitive-rate",
        type=float,
        default=0.0,
        help=(
            "fail closed unless this fraction of independent exact observations "
            "has a median historical-field score at or below the competitive "
            "threshold"
        ),
    )
    parser.add_argument(
        "--maximum-p90-rank-sum",
        type=float,
        help="optional fail-closed upper-tail ceiling",
    )
    parser.add_argument(
        "--maximum-p90-confidence-rank-sum",
        type=float,
        help="optional fail-closed upper-tail ceiling for five confidence metrics",
    )
    parser.add_argument(
        "--maximum-p90-physical-interaction-rank-sum",
        type=float,
        help="optional fail-closed upper-tail ceiling for three interface metrics",
    )
    parser.add_argument(
        "--maximum-p90-developability-rank-sum",
        type=float,
        help="optional fail-closed upper-tail ceiling for two liability metrics",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        type=Path,
        help="Newline-delimited sequences that must never be selected again",
    )
    parser.add_argument("--recent", type=int, default=300)
    parser.add_argument("--field-size", type=int, default=49)
    parser.add_argument("--bootstrap-fields", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--before-epoch",
        type=int,
        help="use completed public leaderboard fields strictly before this epoch",
    )
    parser.add_argument("--api-field-count", type=int, default=0)
    parser.add_argument("--minimum-api-fields", type=int, default=3)
    parser.add_argument("--api-scan-limit", type=int, default=40)
    parser.add_argument("--leaderboard-api-root", default=LEADERBOARD_API_ROOT)
    parser.add_argument(
        "--selection-order",
        choices=("competitive-rate", "p90", "win-probability"),
        default="competitive-rate",
        help="ordering after all fail-closed qualification gates pass",
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        help="live config used to evaluate retained complex-pose stability",
    )
    parser.add_argument("--minimum-structure-observations", type=int, default=0)
    parser.add_argument("--minimum-median-contact-jaccard", type=float, default=0.0)
    parser.add_argument("--maximum-cdr3-rmsd", type=float)
    parser.add_argument("--minimum-epitope-occupancy", type=float, default=0.0)
    args = parser.parse_args()

    if args.minimum_observations < 1:
        parser.error("--minimum-observations must be positive")
    for name in (
        "minimum_competitive_rate",
        "minimum_observation_competitive_rate",
    ):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between zero and one")
    for name in (
        "minimum_median_contact_jaccard",
        "minimum_epitope_occupancy",
    ):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between zero and one")
    if args.minimum_structure_observations < 0:
        parser.error("--minimum-structure-observations cannot be negative")
    pose_requested = bool(
        args.minimum_structure_observations
        or args.minimum_median_contact_jaccard
        or args.maximum_cdr3_rmsd is not None
        or args.minimum_epitope_occupancy
    )
    if pose_requested and args.pose_config is None:
        parser.error("pose-stability gates require --pose-config")

    full_archive = pd.read_csv(args.archive)
    historical_sequences = set(
        full_archive.sequence.dropna().astype(str).str.strip().str.upper()
    )
    excluded_sequences = _excluded_sequences(args.exclude_file)
    archive = full_archive.tail(args.recent).reset_index(drop=True)
    if len(archive) < args.field_size:
        raise SystemExit("archive is smaller than the requested competitor field")
    missing = set(METRICS) - set(archive.columns)
    if missing:
        raise SystemExit(f"archive lacks metrics: {sorted(missing)}")

    api_fields: list[tuple[int, pd.DataFrame]] = []
    crowd_sequences: list[str] = []
    if args.api_field_count:
        if args.before_epoch is None:
            parser.error("--api-field-count requires --before-epoch")
        api_fields, crowd_sequences = fetch_leaderboard_fields(
            args.leaderboard_api_root,
            before_epoch=args.before_epoch,
            target=args.target,
            count=args.api_field_count,
            scan_limit=args.api_scan_limit,
        )
        if len(api_fields) < args.minimum_api_fields:
            raise SystemExit(
                f"only {len(api_fields)} complete {args.target} leaderboard fields "
                f"were available; require {args.minimum_api_fields}"
            )

    if api_fields:
        fields = [field for _, field in api_fields]
        reference = {
            "source": "public_leaderboard_api",
            "api_root": args.leaderboard_api_root.rstrip("/"),
            "epochs": [epoch for epoch, _ in api_fields],
            "field_sizes": [len(field) for _, field in api_fields],
        }
    else:
        rng = np.random.default_rng(args.random_seed)
        fields = [
            archive.iloc[rng.choice(len(archive), args.field_size, replace=False)]
            for _ in range(args.bootstrap_fields)
        ]
        reference = {
            "source": "archive_bootstrap",
            "recent_rows": len(archive),
            "field_size": args.field_size,
            "field_count": len(fields),
            "random_seed": args.random_seed,
        }
    compiled_fields = [_compile_validator_field(field) for field in fields]
    # The field the ranker already replays is also the field whose consensus
    # decides which positions are settled, so it costs one pass over sequences
    # we have loaded anyway.
    consensus = field_consensus(
        [
            str(sequence).strip().upper()
            for sequence in archive.sequence.dropna().astype(str)
        ],
        int(archive.sequence.dropna().astype(str).str.len().mode().iloc[0]),
    ) if "sequence" in archive.columns else {}
    structure_paths: dict[str, list[Path]] = {}
    writer_structure_paths: dict[str, list[Path]] = {}
    sample_selections: dict[str, list[dict[str, Any]]] = {}
    pose_challenge: dict[str, Any] | None = None
    if args.pose_config is not None:
        import yaml

        pose_challenge = yaml.safe_load(args.pose_config.read_text())[
            "protein_selection"
        ]["nanobody"]
        if str(pose_challenge["target"]) != args.target:
            raise SystemExit("pose config target does not match exact target")
        structure_paths = exact_structure_paths(
            args.exact, args.target, pose_kind="confidence"
        )
        writer_structure_paths = exact_structure_paths(
            args.exact, args.target, pose_kind="writer"
        )
        sample_selections = exact_sample_selections(args.exact, args.target)
    ranked = []
    for sequence, observations in _observations(args.exact, args.target).items():
        normalized_sequence = sequence.strip().upper()
        if (
            normalized_sequence in historical_sequences
            or normalized_sequence in excluded_sequences
        ):
            continue
        category_matrix = [
            [
                _compiled_category_rank_sums(metrics, field)
                for field in compiled_fields
            ]
            for metrics in observations
        ]
        # Ordering runs on the weighted sum; the category matrix above still
        # carries the validator's own unweighted view for the emitted estimate.
        score_matrix = [
            [float(sum(categories.values())) for categories in by_field]
            for by_field in category_matrix
        ]
        score_distribution = np.asarray(score_matrix, dtype=np.float64).reshape(-1)
        per_observation = [float(np.median(scores)) for scores in score_matrix]
        observation_competitive = [
            score <= args.competitive_score_threshold
            for score in per_observation
        ]
        observation_competitive_rate = float(np.mean(observation_competitive))
        score_median = float(np.median(score_distribution))
        score_p90 = float(np.quantile(score_distribution, 0.9))
        competitive_rate = float(
            np.mean(score_distribution <= args.competitive_score_threshold)
        )
        if competitive_rate < args.minimum_competitive_rate:
            continue
        if (
            observation_competitive_rate
            < args.minimum_observation_competitive_rate
        ):
            continue
        if (
            args.maximum_p90_rank_sum is not None
            and score_p90 > args.maximum_p90_rank_sum
        ):
            continue
        # Score sharing averages each raw metric across validators before the
        # epoch field is ranked.  Our independent repeats are a local estimate
        # of that same expectation, so their arithmetic mean is the only
        # validator-shaped aggregate.  The observation-by-field p90 below is
        # retained as a qualification stress test; it must not leak into the
        # metric vector presented as the local validator-score estimate.
        aggregate = {
            name: float(np.mean([item[name] for item in observations]))
            for name in METRICS
        }
        metric_standard_errors = {
            name: float(
                np.std([item[name] for item in observations], ddof=1)
                / math.sqrt(len(observations))
            )
            if len(observations) > 1
            else 0.0
            for name in METRICS
        }
        aggregate_categories = [
            _compiled_category_rank_sums(aggregate, field)
            for field in compiled_fields
        ]
        aggregate_field_scores = [
            float(sum(categories.values()))
            for categories in aggregate_categories
        ]
        aggregate_category_scores = {
            category: [
                categories[category] for categories in aggregate_categories
            ]
            for category in ("confidence", "physical_interaction", "developability")
        }
        category_distributions = {
            category: [
                float(categories[category])
                for by_field in category_matrix
                for categories in by_field
            ]
            for category in ("confidence", "physical_interaction", "developability")
        }
        p90_categories = {
            category: float(np.quantile(scores, 0.9))
            for category, scores in category_distributions.items()
        }
        category_ceilings = {
            "confidence": args.maximum_p90_confidence_rank_sum,
            "physical_interaction": (
                args.maximum_p90_physical_interaction_rank_sum
            ),
            "developability": args.maximum_p90_developability_rank_sum,
        }
        if any(
            ceiling is not None and p90_categories[category] > ceiling
            for category, ceiling in category_ceilings.items()
        ):
            continue
        pose_summary: dict[str, Any] = {"structure_observation_count": 0}
        writer_pose_summary: dict[str, Any] = {
            "structure_observation_count": 0
        }
        if pose_challenge is not None:
            pose_summary = summarize_sequence_structures(
                normalized_sequence,
                structure_paths.get(normalized_sequence, []),
                target_chain=str(pose_challenge["structure_chain"]),
                nanobody_chain="B",
                epitope=residue_numbers(
                    str(pose_challenge["structure_binding_site"])
                ),
            )
            writer_pose_summary = summarize_sequence_structures(
                normalized_sequence,
                writer_structure_paths.get(normalized_sequence, []),
                target_chain=str(pose_challenge["structure_chain"]),
                nanobody_chain="B",
                epitope=residue_numbers(
                    str(pose_challenge["structure_binding_site"])
                ),
            )
        if (
            int(pose_summary["structure_observation_count"])
            < args.minimum_structure_observations
        ):
            continue
        if (
            float(pose_summary.get("contact_jaccard_median", 0.0))
            < args.minimum_median_contact_jaccard
        ):
            continue
        if (
            args.maximum_cdr3_rmsd is not None
            and float(pose_summary.get("cdr3_rmsd_max_angstrom", math.inf))
            > args.maximum_cdr3_rmsd
        ):
            continue
        if (
            float(pose_summary.get("epitope_occupancy_min", 0.0))
            < args.minimum_epitope_occupancy
        ):
            continue
        selections = sample_selections.get(normalized_sequence, [])
        selection_summary = {
            "observation_count": len(selections),
            "disagreement_count": sum(
                bool(item.get("sample_selection_disagrees"))
                for item in selections
            ),
            "disagreement_rate": (
                float(np.mean([
                    bool(item.get("sample_selection_disagrees"))
                    for item in selections
                ]))
                if selections else None
            ),
            "observations": selections,
        }
        ranked.append({
            "sequence": normalized_sequence,
            "settled_deviations": settled_deviation_count(
                normalized_sequence, consensus
            ),
            "observation_count": len(observations),
            "selection_score": score_p90,
            "competitive_score_threshold": args.competitive_score_threshold,
            "competitive_rate": competitive_rate,
            "observation_competitive_rate": observation_competitive_rate,
            "observation_competitive": observation_competitive,
            "score_median": score_median,
            "score_p90": score_p90,
            "score_worst": float(np.max(score_distribution)),
            "per_observation_field_rank_sums": score_matrix,
            "per_observation_median_rank_sum": per_observation,
            "p90_category_rank_sums": p90_categories,
            "aggregate_median_rank_sum": float(np.median(aggregate_field_scores)),
            "aggregate_p90_rank_sum": float(np.quantile(aggregate_field_scores, 0.9)),
            "aggregate_median_category_rank_sums": {
                category: float(np.median(scores))
                for category, scores in aggregate_category_scores.items()
            },
            "metric_aggregation": "arithmetic_mean",
            "metrics": aggregate,
            "metric_standard_errors": metric_standard_errors,
            "observations": observations,
            "pose_stability": pose_summary,
            "validator_writer_pose_stability": writer_pose_summary,
            "diffusion_sample_selection": selection_summary,
        })

    # Selecting on a one-observation estimate is selecting on noise, but
    # publishing nothing is the only outcome that certainly pays nothing.  Take
    # the deepest evidence this epoch actually produced rather than either.
    observation_floor = args.minimum_observations
    if ranked:
        # Relax against the depth of EVERY observed candidate, not only those
        # that cleared qualification.  The publisher's strict-eligibility cache
        # computes its own floor the same way, over all observations, because it
        # cannot evaluate the gates -- so taking it post-gate here let the two
        # disagree: when the deep candidates all failed, ranking dropped to a
        # shallower floor the cache was not watching, and a change in the
        # evidence ranking was actually using would not invalidate it.  Matching
        # the cache's basis removes that by construction.  It costs nothing the
        # relaxation was for: its purpose is a thin epoch that never reached the
        # floor, and an epoch whose candidates all fail the gates publishes
        # nothing under either basis.
        deepest = max(
            len(records)
            for records in _observations(args.exact, args.target).values()
        )
        observation_floor = max(1, min(observation_floor, deepest))
        if observation_floor < args.minimum_observations:
            print(
                "EXACT_OBSERVATION_FLOOR_RELAXED "
                f"from={args.minimum_observations} to={observation_floor}",
                flush=True,
            )
        ranked = [
            item for item in ranked
            if item["observation_count"] >= observation_floor
        ]

    # Settled-site breakage outranks the screen, and this is the one place the
    # decision is actually made: proposal ordering only decides what gets
    # SCREENED.  The evidence is lopsided -- our screened metrics agree with the
    # validator at r=0.19, while a settled deviation is measured at ~14 rank
    # points across 1,427 live tickets.  It is a coarse integer key, so within a
    # tier the screen still does all the ordering; it only stops us spending a
    # uid on a molecule that breaks something when one that does not is
    # available, which measurement says it is for ~27% of what we generate.
    def _deviation_tier(item: dict[str, Any]) -> int:
        return int(item.get("settled_deviations", 0))

    if args.selection_order == "win-probability":
        # Only rank one pays and the validator draws once, so the target is the
        # chance of a winning draw, not the best average.  Spread is an asset:
        # epoch 25062 was won by a molecule that drew 85 rank points better than
        # its own average and ranked 13th of 46 on averaged quality.
        ranked = rank_by_win_probability(ranked, args.competitive_score_threshold)
        ranked.sort(key=_deviation_tier)   # stable: preserves win-probability order within a tier
    elif args.selection_order == "p90":
        ranked.sort(key=lambda item: (
            _deviation_tier(item),
            item["score_p90"],
            item["score_median"],
            -item["observation_competitive_rate"],
            -item["competitive_rate"],
            item["score_worst"],
            item["sequence"],
        ))
    else:
        ranked.sort(key=lambda item: (
            _deviation_tier(item),
            -item["observation_competitive_rate"],
            -item["competitive_rate"],
            item["score_p90"],
            item["score_median"],
            item["score_worst"],
            item["sequence"],
        ))
    selected = ranked[: args.limit]
    if not selected:
        raise SystemExit(
            "no complete exact candidate cleared qualification "
            f"(observation floor {observation_floor})"
        )
    payload = {
        "target": args.target,
        "exact_files": [str(path) for path in args.exact],
        "unique_exact_observation_count": len({
            str(json.loads(path.read_text()).get("observation_id") or "")
            or hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.exact
        }),
        "minimum_observations": args.minimum_observations,
        "observation_floor_applied": observation_floor,
        "qualification": {
            "competitive_score_threshold": args.competitive_score_threshold,
            "minimum_competitive_rate": args.minimum_competitive_rate,
            "minimum_observation_competitive_rate": (
                args.minimum_observation_competitive_rate
            ),
            "maximum_p90_rank_sum": args.maximum_p90_rank_sum,
            "maximum_p90_category_rank_sums": {
                "confidence": args.maximum_p90_confidence_rank_sum,
                "physical_interaction": (
                    args.maximum_p90_physical_interaction_rank_sum
                ),
                "developability": args.maximum_p90_developability_rank_sum,
            },
            "pose_stability": {
                "minimum_structure_observations": (
                    args.minimum_structure_observations
                ),
                "minimum_median_contact_jaccard": (
                    args.minimum_median_contact_jaccard
                ),
                "maximum_cdr3_rmsd": args.maximum_cdr3_rmsd,
                "minimum_epitope_occupancy": args.minimum_epitope_occupancy,
            },
        },
        "rank_method": "dense",
        "selection_order": args.selection_order,
        "score_semantics": (
            "raw metrics use the validator score-share arithmetic mean; dense ranks "
            "are replayed over historical live fields because the current field is "
            "unknowable before reveal; observation-by-field p90 is a separate "
            "qualification stress test"
        ),
        "reference": reference,
        # The field the validator dedupes against.  Fleet selection must keep
        # every committed ticket clear of this, or the ticket is discarded
        # before it is scored and the registration that carried it is wasted.
        "crowd": crowd_sequences,
        "excluded_sequence_count": len(excluded_sequences),
        "candidate_count": len(ranked),
        "sequences": [item["sequence"] for item in selected],
        "candidates": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"ranked {len(ranked)} candidates; wrote top {len(selected)} to {args.output}")
    for item in selected[:10]:
        print(
            f"  observations={item['observation_competitive_rate']:.1%} "
            f"field_replays={item['competitive_rate']:.1%} "
            f"p90={item['score_p90']:.1f} median={item['score_median']:.1f} "
            f"n={item['observation_count']} {item['sequence']}"
        )


if __name__ == "__main__":
    main()
