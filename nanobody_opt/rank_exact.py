"""Rank exact nanobody observations with the validator's dense-rank objective.

BoltzGen inference is stochastic, so a live finalist must survive more than one
exact observation. This utility applies the validator's unweighted per-metric
dense ranks to historical live fields. Final selection uses the intact
observation score vectors to calculate each molecule's empirical probability
of beating the other qualified molecules on one fresh draw. Lower rank sums are
better; p90 and threshold rates remain qualification diagnostics only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanobody_opt import consensus_policy, publication_gates, shrinkage
from nanobody_opt.local_neighborhood import (
    FIELD_CONSENSUS_THRESHOLD,
    field_consensus,
)
from nanobody_opt.mutation_network import load_exact_artifacts
from nanobody_opt.pose_stability import (
    exact_sample_selections,
    exact_structure_paths,
    residue_numbers,
    summarize_sequence_structures,
)
from nanobody_opt.simulate_round import METRIC_CATEGORIES, METRICS
from nanobody_opt.win_probability import (
    rank_by_fresh_draw_probability,
)

LEADERBOARD_API_ROOT = "https://compound-api.metanova-labs.ai"
VALIDATOR_DUPLICATE_IDENTITY = 0.95


def _exact_run_ids(paths: list[Path]) -> set[str]:
    result = set()
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        result.add(str(payload.get("observation_id") or hashlib.sha256(raw).hexdigest()))
    return result


def _sequence_identity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shared = sum(a == b for a, b in zip(left, right))
    return shared / max(len(left), len(right))


def _with_diversity_reserve(
    ranked: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    count: int,
    *,
    maximum_identity: float = VALIDATOR_DUPLICATE_IDENTITY,
) -> tuple[list[dict[str, Any]], int]:
    """Append the smallest ranked tail that exposes ``count`` distinct tickets.

    The top-N remains intact.  Reserve candidates are discovered by scanning
    the full ordered cohort, then emitted in that same order, so the ranker's
    best-first contract is preserved while downstream fleet selection can see
    beyond a single overrepresented basin.
    """
    if count <= 0:
        return selected, 0
    reserve: list[str] = []
    for candidate in ranked:
        sequence = str(candidate.get("sequence") or "").strip().upper()
        if not sequence or any(
            _sequence_identity(sequence, other) >= maximum_identity
            for other in reserve
        ):
            continue
        reserve.append(sequence)
        if len(reserve) >= count:
            break
    wanted = {
        str(candidate.get("sequence") or "").strip().upper()
        for candidate in selected
    }
    extra = set(reserve) - wanted
    wanted.update(reserve)
    augmented = [
        candidate
        for candidate in ranked
        if str(candidate.get("sequence") or "").strip().upper() in wanted
    ]
    return augmented, len(extra)


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


# The selector deliberately has no per-metric weighting policy. Selection uses
# the same plain dense-rank sum as the validator and represents stochastic
# uncertainty through repeated observations and fresh-draw win probability.
# A weighted variant ran for eleven epochs in two shapes and is reverted.
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


METRIC_INDEX = {metric: index for index, metric in enumerate(METRICS)}


def _observation_rank_array(observations: list[dict[str, float]],
                            compiled: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Vectorised _compiled_metric_ranks over every observation and field.

    Same searchsorted arithmetic, so ranks are identical; the scalar form was
    ~10 Python-level binary searches per observation per field and dominated
    rank_exact once observation history was merged (13 min on 25398).
    """
    values = np.asarray([[float(obs[metric]) for metric in METRICS] for obs in observations],
                        dtype=np.float64).reshape(len(observations), len(METRICS))
    ranks = np.empty((len(observations), len(compiled), len(METRICS)), dtype=np.float64)
    for f, field in enumerate(compiled):
        for j, (metric, direction) in enumerate(METRICS.items()):
            column = field[metric]
            if direction == "min":
                ranks[:, f, j] = np.searchsorted(column, values[:, j], side="left") + 1
            else:
                ranks[:, f, j] = len(column) - np.searchsorted(column, values[:, j], side="right") + 1
    return ranks


# Win-probability objective.  The published rank-sum weights ten metrics
# equally, but only the confidence metrics carry over from our scorer to the
# validator.  Replayed on 132 epochs of our own pools with fresh draws, ordering
# win probability on these three plus the construction constraints below beat
# the ten-metric order by +2.5 top-10 points (95% CI +0.9..+4.1).
CONFIDENCE_OBJECTIVE = ("design_to_target_iptm", "design_iiptm", "design_ptm")


def _objective_rank_sum(metrics: dict[str, float], compiled: dict[str, np.ndarray],
                        objective: str) -> float:
    ranks = _compiled_metric_ranks(metrics, compiled)
    if objective == "confidence":
        return float(sum(ranks[metric] for metric in CONFIDENCE_OBJECTIVE))
    return float(sum(ranks.values()))


def _public_edit_distance(sequences: list[str], public: set[str], cap: int = 3) -> dict[str, int]:
    """Hamming distance to the nearest public sequence, capped at ``cap + 1``.

    Same-length sequences only.  Pigeonhole index: any match within ``cap``
    edits shares at least one of ``cap + 1`` blocks exactly.
    """
    blocks = cap + 1
    by_length: dict[int, dict[tuple[int, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    bounds_by_length: dict[int, list[tuple[int, int]]] = {}
    for sequence in public:
        n = len(sequence)
        bounds = bounds_by_length.setdefault(
            n, [(n * b // blocks, n * (b + 1) // blocks) for b in range(blocks)])
        for b, (lo, hi) in enumerate(bounds):
            by_length[n][(b, sequence[lo:hi])].append(sequence)
    distances: dict[str, int] = {}
    for sequence in sequences:
        n = len(sequence); best = cap + 1
        index = by_length.get(n)
        if index:
            seen: set[str] = set()
            for b, (lo, hi) in enumerate(bounds_by_length[n]):
                for other in index.get((b, sequence[lo:hi]), ()):
                    if other in seen:
                        continue
                    seen.add(other)
                    d = sum(x != y for x, y in zip(sequence, other))
                    if d < best:
                        best = d
        distances[sequence] = best
    return distances


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


def deviation_tier(item: Mapping[str, Any]) -> int:
    """How many settled sites this candidate breaks: the coarse ordering key."""
    return int(item.get("settled_deviations", 0))


def win_probability_order_key(item: Mapping[str, Any]) -> tuple[int, int]:
    """The order the win-probability selection is written in.

    Module-level on purpose.  The tier was applied as a one-line lambda inside
    main() and the tests re-implemented that lambda on their own fixtures, so
    both agreed while the ranker emitted an order fleet.distinct_sequences
    refused -- "0.0148 follows 0.0" -- and the fleet committed nothing for an
    epoch.  One definition, exercised by the tests and the ranker alike.

    The chance key comes first: the deviation rule chooses between molecules
    that can win, so a clean molecule with no chance of a winning draw must
    never displace one that has a chance.  The sort is stable, so the
    win-probability order the screen chose survives inside each tier.
    """
    return consensus_policy.order_key(item, "tier")[:2]


def _order_candidates(
    candidates: list[dict[str, Any]],
    selection_order: str,
    competitive_score_threshold: float,
    consensus_ranking_policy: str = "tier",
) -> list[dict[str, Any]]:
    """Apply the one production ordering in a reusable, non-pose stage."""
    ranked = list(candidates)
    consensus_policy.checked_policy(consensus_ranking_policy)
    if selection_order == "win-probability":
        # Estimate the event that actually pays: each candidate receives one
        # untouched draw of the validator's complete ten-metric objective and
        # the best draw wins.  The former normal-tail approximation optimized
        # crossing a fixed threshold instead of beating this candidate field.
        ranked = rank_by_fresh_draw_probability(ranked)
        ranked = consensus_policy.ordered_candidates(ranked, consensus_ranking_policy)
    elif selection_order == "p90":
        ranked.sort(key=lambda item: (
            deviation_tier(item) if consensus_ranking_policy == "tier" else 0,
            item["score_p90"],
            item["score_median"],
            -item["observation_competitive_rate"],
            -item["competitive_rate"],
            item["score_worst"],
            item["sequence"],
        ))
    else:
        ranked.sort(key=lambda item: (
            deviation_tier(item) if consensus_ranking_policy == "tier" else 0,
            -item["observation_competitive_rate"],
            -item["competitive_rate"],
            item["score_p90"],
            item["score_median"],
            item["score_worst"],
            item["sequence"],
        ))
    return ranked


def _pose_order_is_subset_invariant(
    candidates: list[dict[str, Any]],
    selection_order: str,
    competitive_score_threshold: float,
) -> bool:
    """Whether pose failures cannot reorder the unevaluated tail.

    The p90 and competitive-rate orders contain no cohort statistic.  Fresh-
    draw probability does: removing a pose failure changes the opponent field.
    There is one common live case we can prove safe.  With one observation per
    candidate, the empirical tournament order is exactly the score order (ties
    use the same stable fallback), so removing candidates cannot reverse it.
    """
    if selection_order != "win-probability":
        return True
    score_rows = [
        item.get("per_observation_median_rank_sum") or []
        for item in candidates
    ]
    return bool(score_rows) and all(len(scores) == 1 for scores in score_rows)


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


def compact_persisted_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the transient observation-by-field matrix from a rank artifact.

    ``per_observation_field_rank_sums`` is needed while this process computes
    qualification statistics, but no downstream consumer reads it.  Persisting
    it duplicated every historical field score for every shortlisted sequence:
    live screen/stage rank files reached ~810 MB each and eventually filled the
    producer disk.  Keep the per-observation medians and every derived statistic
    that selection/publication consumes; record only the matrix dimensions for
    auditability.
    """
    persisted = dict(item)
    matrix = persisted.pop("per_observation_field_rank_sums", None) or []
    persisted["field_replay_matrix_shape"] = [
        len(matrix),
        max((len(row) for row in matrix), default=0),
    ]
    return persisted


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


def _historical_observations(database: Path, target: str,
                             known: dict[str, list[dict[str, float]]],
                             excluded_run_ids: set[str] | None = None,
                             ) -> tuple[dict[str, list[dict[str, float]]], int]:
    """Folds from earlier epochs, which are still valid evidence today.

    The challenge does not change between epochs -- one config digest across
    every epoch, one target across every node in the database -- so a fold is
    a property of (sequence, target) and never expires. The screen was
    ranking on whatever this epoch happened to produce, a single fold for
    most candidates, while the database held a mean of 39.8 observations for
    the same sequences. Measured out of sample over 22 epochs, ranking on the
    accumulated evidence lifts agreement with the validator's realized
    confidence_rank from +0.165 to +0.365.

    Only `target` rows are read, and only our own folds: the public
    leaderboard rows we ingest for analysis are somebody else's measurement
    and must never enter our own estimate.
    """
    import sqlite3

    found: dict[str, list[dict[str, float]]] = defaultdict(list)
    added = 0
    try:
        connection = sqlite3.connect(
            f"file:{database}?mode=ro", uri=True, timeout=60)
    except sqlite3.Error:
        return found, 0
    try:
        rows = connection.execute(
            "SELECT n.sequence, o.metrics_json, x.run_id "
            "FROM observations o "
            "JOIN nodes n ON n.id = o.node_id "
            "JOIN exact_run_observations x "
            "ON x.observation_id=o.observation_id AND x.target=n.target "
            "JOIN exact_runs r ON r.target=x.target "
            "AND r.run_id=x.run_id AND r.complete=1 "
            "WHERE n.target = ? AND o.source='local_exact_raw' "
            "AND o.metrics_json IS NOT NULL",
            (target,))
        for sequence, blob, run_id in rows:
            if str(run_id or "") in (excluded_run_ids or set()):
                continue
            try:
                metrics = json.loads(blob)
            except (TypeError, ValueError):
                continue
            if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
                continue
            try:
                values = {name: float(metrics[name]) for name in METRICS}
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in values.values()):
                continue
            found[str(sequence)].append(values)
            added += 1
    except sqlite3.Error:
        return found, 0
    finally:
        connection.close()
    return found, added


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


def fetch_leaderboard_fields(
    api_root: str,
    *,
    before_epoch: int,
    target: str,
    count: int,
    scan_limit: int,
) -> list[tuple[int, pd.DataFrame]]:
    """Fetch recent complete, target-matching fields from the public API."""
    fields: list[tuple[int, pd.DataFrame]] = []
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
        fields.append((epoch, field))
        if len(fields) >= count:
            break
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-metric-policy",
        choices=("plain",),
        default="plain",
        help="compatibility field; only the validator's plain dense-rank sum is supported",
    )
    parser.add_argument("--exact", action="append", default=[], type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--diversity-reserve-count",
        type=int,
        default=0,
        help=(
            "append the minimum ranked tail needed to expose this many "
            "pairwise validator-distinct sequences"
        ),
    )
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
    parser.add_argument(
        "--observation-history", type=Path, default=None,
        help=(
            "sqlite mutation database whose earlier-epoch folds are merged "
            "into this epoch's evidence; the challenge is identical across "
            "epochs so those folds remain valid"
        ),
    )
    parser.add_argument(
        "--metric-shrinkage", dest="metric_shrinkage",
        action="store_true", default=True,
        help=(
            "shrink each candidate's mean toward the pool mean by how little "
            "evidence supports it (default: on)"
        ),
    )
    parser.add_argument(
        "--no-metric-shrinkage", dest="metric_shrinkage", action="store_false",
        help="rank on raw means regardless of how few folds support them",
    )
    parser.add_argument("--consensus-ranking-policy", choices=consensus_policy.POLICIES,
                        default="tier", help="final ordering only; generation is unaffected")
    parser.add_argument("--paired-consensus-shortlist", action="store_true",
                        help="emit a capped shared pool covering both final policies")
    parser.add_argument("--minimum-median-contact-jaccard", type=float, default=0.0)
    parser.add_argument("--maximum-cdr3-rmsd", type=float)
    parser.add_argument("--minimum-epitope-occupancy", type=float, default=0.0)
    parser.add_argument("--win-objective", choices=("confidence", "full"), default="full",
                        help="metrics the fresh-draw win probability ranks on; gates and "
                             "diagnostics keep the full ten-metric rank sum")
    parser.add_argument("--maximum-public-edits", type=int, default=0,
                        help="keep candidates within this many edits of a recent public "
                             "sequence (and not identical to one); 0 disables")
    parser.add_argument("--public-parent-window", type=int, default=600,
                        help="most recent archive rows that count as public parents")
    parser.add_argument("--liability-quantile", type=float, default=0.0,
                        help="drop candidates whose mean liability score is above this "
                             "quantile of the candidate pool; 0 disables")
    parser.add_argument("--minimum-constrained-pool", type=int, default=40,
                        help="skip the construction constraints for this run if fewer "
                             "candidates than this would survive them")
    parser.add_argument("--only-sequences", type=Path, default=None,
                        help="rank only the sequences listed in this newline-delimited "
                             "file; history and shrinkage still use the full pool")
    args = parser.parse_args()

    if not args.exact and args.observation_history is None:
        parser.error("at least one --exact or --observation-history is required")

    if args.paired_consensus_shortlist and (
        args.selection_order != "win-probability" or args.limit < 2 or args.limit % 2
    ):
        parser.error("paired consensus shortlist requires win-probability and an even limit >= 2")
    if args.diversity_reserve_count < 0:
        parser.error("--diversity-reserve-count cannot be negative")
    if args.paired_consensus_shortlist and args.diversity_reserve_count:
        parser.error("diversity reserve is not supported by the paired shortlist")

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
    if args.api_field_count:
        if args.before_epoch is None:
            parser.error("--api-field-count requires --before-epoch")
        api_fields = fetch_leaderboard_fields(
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
        if args.observation_history is not None:
            current_run_ids = _exact_run_ids(args.exact)
            for sequence, records in load_exact_artifacts(
                args.observation_history, target=args.target
            ).items():
                for record in records:
                    if str(record.get("run_id") or "") in current_run_ids:
                        continue
                    confidence = record.get("confidence_path")
                    writer = record.get("writer_path")
                    if confidence:
                        structure_paths.setdefault(sequence, []).append(
                            Path(str(confidence))
                        )
                    if writer:
                        writer_structure_paths.setdefault(sequence, []).append(
                            Path(str(writer))
                        )
                    selection = record.get("sample_selection")
                    if isinstance(selection, dict):
                        sample_selections.setdefault(sequence, []).append(selection)
    observations_by_sequence = _observations(args.exact, args.target)
    if args.observation_history is not None:
        exact_run_ids = _exact_run_ids(args.exact)
        history, added = _historical_observations(
            args.observation_history,
            args.target,
            observations_by_sequence,
            excluded_run_ids=exact_run_ids,
        )
        for sequence, folds in history.items():
            observations_by_sequence[sequence].extend(folds)
        depths = [len(v) for v in observations_by_sequence.values()] or [0]
        print(
            "EXACT_OBSERVATION_HISTORY merged=%d sequences=%d median_depth=%d"
            % (added, len(observations_by_sequence),
               sorted(depths)[len(depths) // 2]),
            flush=True,
        )
    # Depth is uneven once history is merged -- a new proposal has one fold
    # and a previous finalist has scores of them -- so the means are not
    # equally trustworthy and must not be compared as if they were.
    shrunk_by_sequence: dict[str, dict[str, float]] = {}
    if args.metric_shrinkage:
        try:
            shrunk_by_sequence = shrinkage.shrunk_metrics(
                observations_by_sequence, tuple(METRICS))
        except (ValueError, statistics.StatisticsError):
            shrunk_by_sequence = {}
    ranked = []
    only_sequences = None
    if args.only_sequences is not None:
        only_sequences = {line.strip().upper() for line in
                          args.only_sequences.read_text().splitlines() if line.strip()}
    for sequence, observations in observations_by_sequence.items():
        normalized_sequence = sequence.strip().upper()
        if only_sequences is not None and normalized_sequence not in only_sequences:
            continue
        if (
            normalized_sequence in historical_sequences
            or normalized_sequence in excluded_sequences
        ):
            continue
        # ranks[observation, field, metric], each rank computed once and
        # shared by the category sums and the win objective.
        rank_array = _observation_rank_array(observations, compiled_fields)
        category_sums = {
            category: rank_array[:, :, [METRIC_INDEX[m] for m in names]].sum(axis=2)
            for category, names in METRIC_CATEGORIES.items()
        }
        # Keep the objective identical to the validator. Repeated observations
        # feed the fresh-draw distribution; they do not justify changing the
        # ten-metric arithmetic into a hand-weighted surrogate objective.
        score_matrix = rank_array.sum(axis=2).tolist()
        score_distribution = np.asarray(score_matrix, dtype=np.float64).reshape(-1)
        per_observation = [float(np.median(scores)) for scores in score_matrix]
        # Gates and diagnostics keep the full ten-metric scores; only the
        # fresh-draw win probability ranks on the chosen objective.
        win_per_observation = per_observation if args.win_objective == "full" else [
            float(np.median(row)) for row in rank_array[
                :, :, [METRIC_INDEX[m] for m in CONFIDENCE_OBJECTIVE]].sum(axis=2)
        ]
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
        # Shrinkage replaces the raw mean where it could be estimated. A
        # single fold keeps ~37% of its deviation from the pool mean and
        # forty folds keep ~96%, so one lucky measurement can no longer
        # outrank a candidate we actually know something about.
        for name, value in (shrunk_by_sequence.get(sequence) or {}).items():
            if name in aggregate and math.isfinite(value):
                aggregate[name] = float(value)
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
            category: category_sums[category].reshape(-1).tolist()
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
            "per_observation_median_rank_sum": win_per_observation,
            "per_observation_full_rank_sum": per_observation,
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
            for records in observations_by_sequence.values()
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

    constrained = list(ranked)
    if args.maximum_public_edits > 0 and constrained:
        public = set(full_archive.sequence.dropna().astype(str).str.strip().str.upper()
                     .tail(args.public_parent_window))
        distance = _public_edit_distance([item["sequence"] for item in constrained], public,
                                         cap=args.maximum_public_edits)
        constrained = [item for item in constrained
                       if 1 <= distance[item["sequence"]] <= args.maximum_public_edits]
    if 0 < args.liability_quantile < 1 and constrained:
        liabilities = sorted(float(item["metrics"]["liability_score"]) for item in constrained)
        cut = liabilities[int(args.liability_quantile * (len(liabilities) - 1))]
        constrained = [item for item in constrained
                       if float(item["metrics"]["liability_score"]) <= cut]
    if args.maximum_public_edits > 0 or 0 < args.liability_quantile < 1:
        if len(constrained) >= args.minimum_constrained_pool:
            print(f"CONSTRUCTION_CONSTRAINTS_APPLIED kept={len(constrained)} of={len(ranked)}",
                  flush=True)
            ranked = constrained
        else:
            print(f"CONSTRUCTION_CONSTRAINTS_SKIPPED kept={len(constrained)} of={len(ranked)} "
                  f"minimum={args.minimum_constrained_pool}", flush=True)

    # Pose stability dominates strict runtime.  Order the cheap scalar evidence
    # first, then walk that order.  Early stopping is used only where removing a
    # pose failure provably cannot change the order; mixed-depth win-probability
    # pools retain the exhaustive behavior and therefore the exact old result.
    preliminary = _order_candidates(
        ranked, args.selection_order, args.competitive_score_threshold,
        args.consensus_ranking_policy,
    )
    pose_early_stop = (
        pose_challenge is not None
        and not args.paired_consensus_shortlist
        and not args.diversity_reserve_count
        and _pose_order_is_subset_invariant(
            preliminary,
            args.selection_order,
            args.competitive_score_threshold,
        )
    )
    if pose_challenge is not None:
        pose_survivors: list[dict[str, Any]] = []
        target_chain = str(pose_challenge["structure_chain"])
        epitope = residue_numbers(str(pose_challenge["structure_binding_site"]))
        for item in preliminary:
            sequence = str(item["sequence"])
            pose_summary = summarize_sequence_structures(
                sequence,
                structure_paths.get(sequence, []),
                target_chain=target_chain,
                nanobody_chain="B",
                epitope=epitope,
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
            accepted = dict(item)
            accepted["pose_stability"] = pose_summary
            pose_survivors.append(accepted)
            if pose_early_stop and len(pose_survivors) >= args.limit:
                break
        ranked = pose_survivors
    else:
        ranked = [
            {**item, "pose_stability": {"structure_observation_count": 0}}
            for item in preliminary
        ]

    # Recompute cohort-normalized win-probability fields on the actual pose
    # survivors.  In exhaustive mode this is byte-for-byte the old cohort; in
    # guarded early-stop mode the proof above guarantees the selected order.
    ranked = _order_candidates(
        ranked, args.selection_order, args.competitive_score_threshold,
        args.consensus_ranking_policy,
    )
    selected = (
        consensus_policy.ordered_candidates(
            consensus_policy.paired_shortlist(ranked, args.limit),
            args.consensus_ranking_policy,
        ) if args.paired_consensus_shortlist else ranked[:args.limit]
    )
    selected, diversity_reserve_added = _with_diversity_reserve(
        ranked,
        selected,
        args.diversity_reserve_count,
    )
    if not selected:
        raise SystemExit(
            "no complete exact candidate cleared qualification "
            f"(observation floor {observation_floor})"
        )

    # The validator-writer pose is audit-only: no qualification or ordering
    # gate reads it.  Computing it for every rejected candidate doubled pose
    # work.  Attach it only to the selected records that are actually emitted.
    if pose_challenge is not None:
        target_chain = str(pose_challenge["structure_chain"])
        epitope = residue_numbers(str(pose_challenge["structure_binding_site"]))
        for item in selected:
            sequence = str(item["sequence"])
            item["validator_writer_pose_stability"] = (
                summarize_sequence_structures(
                    sequence,
                    writer_structure_paths.get(sequence, []),
                    target_chain=target_chain,
                    nanobody_chain="B",
                    epitope=epitope,
                )
            )
    else:
        for item in selected:
            item["validator_writer_pose_stability"] = {
                "structure_observation_count": 0
            }
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
        "selection_metric_policy": args.selection_metric_policy,
        "prediction_event": "best_selection_rank_sum_on_one_fresh_draw",
        "consensus_ranking_policy": args.consensus_ranking_policy,
        "consensus_shortlist": {
            "mode": "paired" if args.paired_consensus_shortlist else "single",
            "capacity": args.limit,
            "per_policy_prefix": args.limit // 2 if args.paired_consensus_shortlist else None,
            "qualified_candidate_count": len(ranked),
        },
        "diversity_reserve": {
            "requested_distinct_count": args.diversity_reserve_count,
            "added_candidate_count": diversity_reserve_added,
            "maximum_identity": VALIDATOR_DUPLICATE_IDENTITY,
        },
        "score_semantics": (
            "raw metrics use the validator score-share arithmetic mean; dense ranks "
            "are replayed over historical live fields because the current field is "
            "unknowable before reveal; observation-by-field p90 is a separate "
            "qualification stress test"
        ),
        "reference": reference,
        "excluded_sequence_count": len(excluded_sequences),
        "candidate_count": len(ranked),
        "sequences": [item["sequence"] for item in selected],
        "candidates": [compact_persisted_candidate(item) for item in selected],
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
