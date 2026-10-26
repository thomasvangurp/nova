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
import os
from concurrent.futures import ProcessPoolExecutor
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
from nanobody_opt import consensus_policy, db, publication_gates, shrinkage
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


def cold_start_reference(archive, observations, *, target, epoch, minimum_rows):
    """Use measured new-target folds only while the public field is being born.

    This is an internal selection reference, never evidence of competitive
    performance against rivals or a comparable old-target rank-sum.
    """
    from nanobody_opt import rules
    if (target != rules.TARGET or epoch is None
            or not rules.ACTIVATION_EPOCH <= epoch < rules.ACTIVATION_EPOCH + 3
            or len(archive) >= minimum_rows):
        return None
    records = []
    for sequence, folds in observations.items():
        if (not folds or any(any(k not in fold or not math.isfinite(float(fold[k]))
                                 for k in METRICS) for fold in folds)):
            continue
        records.append({"sequence": sequence, **{
            key: statistics.mean(float(fold[key]) for fold in folds) for key in METRICS}})
    if len(records) < 2:
        raise ValueError("new-target ranking needs at least two complete measured candidates")
    return pd.DataFrame(records)


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
    rank_workers: int = 1,
    rank_workers_minimum_candidates: int = 64,
) -> list[dict[str, Any]]:
    """Apply the one production ordering in a reusable, non-pose stage."""
    ranked = list(candidates)
    consensus_policy.checked_policy(consensus_ranking_policy)
    if selection_order == "win-probability":
        # Estimate the event that actually pays: each candidate receives one
        # untouched draw of the validator's complete ten-metric objective and
        # the best draw wins.  The former normal-tail approximation optimized
        # crossing a fixed threshold instead of beating this candidate field.
        ranked = rank_by_fresh_draw_probability(
            ranked, workers=rank_workers,
            workers_minimum_candidates=rank_workers_minimum_candidates,
            competitive_score_threshold=competitive_score_threshold,
        )
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
    elif selection_order == "multipass":
        # The validator's actual metric: dense sum-of-10 rank-sum against the
        # current last-10-epochs field (rank_metrics_against_field), the same
        # formula the dashboard and competitor_lane parent pool use. Do NOT
        # lead with deviation_tier here: the consensus (few settled deviations)
        # is the blocked basin that scores worst on multipass, so tier-first
        # would surface it ahead of the diverse/admissible candidates.
        ranked.sort(key=lambda item: (
            item["multipass_rank"],
            item["score_median"],
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


#: Set by _historical_observations when the history read fails; reported in
#: the rankings payload so a degraded ranking is visible in its own output.
_HISTORY_READ_ERROR: str | None = None


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
    # A failed read still returns an empty history (a missing history must not
    # block the provisional publish), but it is no longer silent: the failure
    # goes to stderr with a traceback, to the box failure log the liveness
    # checker reads, and into the rankings payload as
    # observation_history.status=read_failed (ARCHITECTURE_AUDIT §0.4, P0b).
    global _HISTORY_READ_ERROR
    # This is our own race's folds (source='local_exact_raw'), the same data
    # NOVA_LOCAL_SQLITE's local primary holds as soon as the race writes it --
    # route through mutation_network._connect(database) (dialect-aware, the
    # same path the write side uses) instead of an unconditional central
    # MySQL connect(), mirroring the build_top50_index.py fix (2026-10-10):
    # `database` used to be accepted and ignored here too, so every call on
    # this -- the hottest call path in the pipeline, hit by every race
    # tranche/round via sequential_exact_race.py AND every publish pass via
    # publish_current_nanobody.py -- paid a cross-continent round trip (and
    # silently threw away the +0.165->+0.365 accuracy lift on every Hetzner
    # blip) for data already sitting locally.
    from nanobody_opt import mutation_network
    try:
        connection = mutation_network._connect(Path(database))
    except Exception as error:
        _HISTORY_READ_ERROR = f"connect: {type(error).__name__}: {error}"[:500]
        from nanobody_opt.failure_log import record_failure
        record_failure("rank_exact", "EXACT_OBSERVATION_HISTORY_READ_FAILED",
                       error, stage="connect", target=target)
        return found, 0
    try:
        p = "?" if mutation_network._is_local(connection) else "%s"
        rows = connection.execute(
            "SELECT n.sequence, o.metrics_json, x.run_id "
            "FROM observations o "
            "JOIN nodes n ON n.id = o.node_id "
            "JOIN exact_run_observations x "
            "ON x.observation_id=o.observation_id AND x.target=n.target "
            "JOIN exact_runs r ON r.target=x.target "
            "AND r.run_id=x.run_id AND r.complete=1 "
            f"WHERE n.target = {p} AND o.source='local_exact_raw' "
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
    except Exception as error:
        _HISTORY_READ_ERROR = f"query: {type(error).__name__}: {error}"[:500]
        from nanobody_opt.failure_log import record_failure
        record_failure("rank_exact", "EXACT_OBSERVATION_HISTORY_READ_FAILED",
                       error, stage="query", target=target, partial_rows=added)
        return found, 0
    finally:
        connection.close()
    return found, added


def _parse_fold_blob(blob: str) -> dict[str, float] | None:
    """One metrics_json blob -> per-metric float values, or None if skipped.

    The single validation the historical read applies to every fold: all ten
    METRICS present and finite.  Shared by the full-read fallback and the
    incremental-aggregate candidate read so the two can never disagree on
    which folds count (Fix 2).
    """
    try:
        metrics = json.loads(blob)
    except (TypeError, ValueError):
        return None
    if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
        return None
    try:
        values = {name: float(metrics[name]) for name in METRICS}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


from contextlib import contextmanager


@contextmanager
def _history_snapshot(database):
    from nanobody_opt import mutation_network
    try:
        connection = mutation_network._connect(Path(database))
    except Exception:
        yield None
        return
    try:
        yield connection
    finally:
        connection.close()


def _read_aggregate_stats(
    database: Path, target: str, *, _connection=None,
) -> tuple[dict[str, dict[str, tuple[float, float, int]]] | None, int | None]:
    """Read the incremental per-(sequence, metric) aggregate (Fix 2).

    Returns ``(stats, watermark)`` where ``stats`` maps sequence ->
    {metric: (sum, sum_sq, count)} for the full pool, and ``watermark`` is the
    aggregate's source ``MAX(exact_run_observations.rowid)``.  When the
    aggregate is absent, unreadable, or stale (its watermark does not match the
    live ``MAX(rowid)``), returns ``(None, watermark)`` so the caller falls back
    to the exact old full-pool read rather than rank on out-of-date numbers.
    """
    from nanobody_opt import mutation_network
    try:
        connection = _connection if _connection is not None else mutation_network._connect(Path(database))
    except Exception:
        return None, None
    try:
        p = "?" if mutation_network._is_local(connection) else "%s"
        if not mutation_network._aggregate_ready(connection):
            return None, None
        rows = connection.execute(
            f"SELECT sequence, metric, sum_value, sum_sq, count_value "
            f"FROM exact_observation_aggregate WHERE target={p}",
            (target,),
        ).fetchall()
        meta_row = connection.execute(
            "SELECT value FROM exact_observation_meta WHERE key='source_max_rowid'"
        ).fetchone()
        live_row = connection.execute(
            "SELECT MAX(rowid) FROM exact_run_observations"
        ).fetchone()
    except Exception:
        return None, None
    finally:
        if _connection is None:
            connection.close()
    live_max = int(live_row[0]) if live_row and live_row[0] is not None else 0
    watermark = int(meta_row[0]) if meta_row else None
    if watermark is None or watermark != live_max:
        return None, watermark
    stats: dict[str, dict[str, tuple[float, float, int]]] = defaultdict(dict)
    for sequence, metric, total, total_sq, count in rows:
        key = str(sequence).strip().upper()
        stats[key][str(metric)] = (float(total), float(total_sq), int(count))
    return dict(stats), watermark


def _candidate_observation_folds(
    database: Path,
    target: str,
    sequences: set[str],
    excluded_run_ids: set[str] | None, *, _connection=None,
) -> dict[str, list[dict[str, float]]]:
    """Full per-observation vectors for the small candidate set only (Fix 2).

    Reads the same four-way join the old full-pool read used, restricted to
    ``sequences`` so it json.loads a few hundred folds instead of 1.04M, and
    applies the same excluded_run_ids filter and per-row validation.
    """
    from nanobody_opt import mutation_network
    found: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not sequences:
        return found
    ordered = sorted(sequences)
    try:
        connection = _connection if _connection is not None else mutation_network._connect(Path(database))
    except Exception:
        return found
    try:
        p = "?" if mutation_network._is_local(connection) else "%s"
        placeholders = ",".join([p] * len(ordered))
        rows = connection.execute(
            "SELECT n.sequence, o.metrics_json, x.run_id "
            "FROM observations o "
            "JOIN nodes n ON n.id = o.node_id "
            "JOIN exact_run_observations x "
            "ON x.observation_id=o.observation_id AND x.target=n.target "
            "JOIN exact_runs r ON r.target=x.target "
            "AND r.run_id=x.run_id AND r.complete=1 "
            f"WHERE n.target = {p} AND o.source='local_exact_raw' "
            "AND o.metrics_json IS NOT NULL "
            f"AND n.sequence IN ({placeholders})",
            (target, *ordered),
        )
        excluded = excluded_run_ids or set()
        for sequence, blob, run_id in rows:
            if str(run_id or "") in excluded:
                continue
            values = _parse_fold_blob(blob)
            if values is None:
                continue
            found[str(sequence)].append(values)
    finally:
        if _connection is None:
            connection.close()
    return found


def _excluded_run_stats(
    database: Path, target: str, excluded_run_ids: set[str], *, _connection=None,
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """(sum, sum_sq, count) per (sequence, metric) for complete runs whose
    ``run_id`` is in ``excluded_run_ids`` (the current epoch's runs already
    read from the --exact files).  Subtracting these from the aggregate
    reproduces the old excluded_run_ids filter for the shrinkage pool."""
    from nanobody_opt import mutation_network
    if not excluded_run_ids:
        return {}
    ordered = sorted(excluded_run_ids)
    stats: dict[str, dict[str, tuple[float, float, int]]] = defaultdict(dict)
    try:
        connection = _connection if _connection is not None else mutation_network._connect(Path(database))
    except Exception:
        return {}
    try:
        p = "?" if mutation_network._is_local(connection) else "%s"
        placeholders = ",".join([p] * len(ordered))
        rows = connection.execute(
            "SELECT n.sequence, o.metrics_json "
            "FROM observations o "
            "JOIN nodes n ON n.id = o.node_id "
            "JOIN exact_run_observations x "
            "ON x.observation_id=o.observation_id AND x.target=n.target "
            "JOIN exact_runs r ON r.target=x.target "
            "AND r.run_id=x.run_id AND r.complete=1 "
            f"WHERE n.target = {p} AND o.source='local_exact_raw' "
            "AND o.metrics_json IS NOT NULL "
            f"AND x.run_id IN ({placeholders})",
            (target, *ordered),
        )
        for sequence, blob in rows:
            values = _parse_fold_blob(blob)
            if values is None:
                continue
            key = str(sequence).strip().upper()
            for metric, value in values.items():
                total, total_sq, count = stats[key].get(metric, (0.0, 0.0, 0))
                stats[key][metric] = (
                    total + value, total_sq + value * value, count + 1)
    finally:
        if _connection is None:
            connection.close()
    return dict(stats)


def _folds_to_stats(
    folds_by_sequence: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """(sum, sum_sq, count) per (sequence, metric) from full fold vectors."""
    stats: dict[str, dict[str, tuple[float, float, int]]] = {}
    for sequence, folds in folds_by_sequence.items():
        key = str(sequence).strip().upper()
        row: dict[str, tuple[float, float, int]] = {}
        for fold in folds:
            for metric, value in fold.items():
                if isinstance(value, (int, float)):
                    total, total_sq, count = row.get(metric, (0.0, 0.0, 0))
                    row[metric] = (total + value, total_sq + value * value,
                                   count + 1)
        stats[key] = row
    return stats


def _add_stats(
    target: dict[str, dict[str, tuple[float, float, int]]],
    source: dict[str, dict[str, tuple[float, float, int]]],
    sign: int,
) -> None:
    """In place: ``target[seq][metric] += sign * source[seq][metric]``."""
    for sequence, metrics in source.items():
        row = target.setdefault(sequence, {})
        for metric, (total, total_sq, count) in metrics.items():
            have = row.get(metric, (0.0, 0.0, 0))
            row[metric] = (
                have[0] + sign * total,
                have[1] + sign * total_sq,
                have[2] + sign * count,
            )


def _candidate_set(
    pool_sequences: object,
    only_sequences: set[str] | None,
    historical_sequences: set[str],
    excluded_sequences: set[str],
) -> set[str]:
    """Sequences in the pool that would survive _rank_candidate's filters.

    Mirrors the three early-return filters in :func:`_rank_candidate` so the
    candidate set (the only sequences that need full per-observation vectors)
    is known before the expensive per-candidate read.
    """
    candidates: set[str] = set()
    for sequence in pool_sequences:
        normalized = str(sequence).strip().upper()
        if only_sequences is not None and normalized not in only_sequences:
            continue
        if normalized in historical_sequences or normalized in excluded_sequences:
            continue
        candidates.add(normalized)
    return candidates


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


def fetch_recent_fields_from_db(
    database: Path, *, target: str, before_epoch: int, count: int,
) -> list[tuple[int, pd.DataFrame]]:
    """Real per-epoch validator fields read from the local cache, no live call.

    seed_mutation_database.py already keeps
    observations(source='public_validator_api', score_kind=
    'validator_final_rank_sum') continuously synced with the public
    leaderboard -- every live submission across every coldkey, by epoch, with
    raw metrics (confirmed 2026-10-11: 853 distinct epochs cached, current
    through the live epoch). Reading that here means ranking -- the hottest
    call path in the pipeline, hit by every race tranche/round -- never makes
    a live HTTP call in its own critical path; the one network dependency
    moves to the separate, already-running sync process, mirroring the
    _recent_observation_history fix for the same reason (dialect-aware
    mutation_network._connect, never an unconditional central MySQL hop).
    """
    from nanobody_opt import mutation_network

    try:
        connection = mutation_network._connect(database)
    except Exception as error:
        # Local store only (no MySQL fallback); a missing store is loud, not an
        # empty field passed off as a real one.
        print(f"RECENT_FIELD_READ_FAILED {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        from nanobody_opt.failure_log import record_failure
        record_failure("rank_exact", "RECENT_FIELD_READ_FAILED", error)
        return []
    try:
        # Some deployments of mutation_network only ever speak central MySQL
        # (no local-sqlite branch, no _is_local helper at all -- confirmed on
        # the dashboard box 2026-10-11); default to the MySQL placeholder
        # whenever that helper is missing, rather than assuming sqlite.
        is_local = getattr(mutation_network, "_is_local", None)
        p = "?" if (is_local is not None and is_local(connection)) else "%s"
        epoch_rows = connection.execute(
            "SELECT DISTINCT o.epoch FROM observations o "
            "JOIN nodes n ON n.id = o.node_id "
            f"WHERE n.target = {p} AND o.source = 'public_validator_api' "
            f"AND o.epoch < {p} ORDER BY o.epoch DESC LIMIT {p}",
            (target, before_epoch, count),
        )
        epochs = [int(row[0]) for row in epoch_rows]
        if not epochs:
            return []
        placeholders = ",".join([p] * len(epochs))
        rows = connection.execute(
            "SELECT o.epoch, o.metrics_json FROM observations o "
            "JOIN nodes n ON n.id = o.node_id "
            f"WHERE n.target = {p} AND o.source = 'public_validator_api' "
            f"AND o.epoch IN ({placeholders})",
            (target, *epochs),
        )
        by_epoch: dict[int, list[dict[str, float]]] = defaultdict(list)
        for epoch, blob in rows:
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
            by_epoch[int(epoch)].append(values)
    except Exception:
        return []
    finally:
        connection.close()
    return [
        (epoch, pd.DataFrame(values))
        for epoch, values in sorted(by_epoch.items(), reverse=True)
        if len(values) >= 2
    ]


def pool_recent_fields(fields: list[tuple[int, pd.DataFrame]]) -> pd.DataFrame:
    """One field from fetch_recent_fields_from_db's per-epoch list, for
    callers that rank many candidates against the same pooled background in
    one go (fetch once, reuse for every candidate -- see
    rank_metrics_against_field)."""
    if not fields:
        return pd.DataFrame(columns=list(METRICS))
    return pd.concat([df for _, df in fields], ignore_index=True)


def rank_metrics_against_field(
    metrics_by_key: Mapping[str, Mapping[str, float]], pooled_field: pd.DataFrame,
) -> dict[str, float]:
    """Dense sum-of-10 rank-sum for each key's raw metrics against
    ``pooled_field`` (typically pool_recent_fields' output).

    The one ranking formula multipass_all_rivals.py (dashboard + parent-pool
    writer) and competitor_lane.py (parent-pool reader, on-demand recompute
    from stored metrics_json) both call, so a dashboard number and a
    parent-selection number are never two different computations of "rank
    against the recent field" -- user, 2026-10-11: "ensure dashboard uses
    this logic."
    """
    # Each key is ranked against the field ALONE. Concatenating every key into
    # one frame made a score depend on how many other keys shared the call
    # (a 60k-sequence call inflated scores ~10x; a 500-pass epoch ~2x). A
    # value's dense rank when inserted on its own is (number of strictly
    # better unique field values) + 1 -- identical to ranking it alone,
    # computed for all keys at once.
    keys = [key for key, values in metrics_by_key.items() if set(METRICS).issubset(values)]
    if not keys:
        return {}
    total = np.zeros(len(keys), dtype=np.float64)
    for metric, direction in METRICS.items():
        unique = np.unique(pooled_field[metric].to_numpy(dtype=np.float64))
        values = np.array([float(metrics_by_key[key][metric]) for key in keys], dtype=np.float64)
        if direction == "min":
            total += np.searchsorted(unique, values, side="left") + 1
        else:
            total += len(unique) - np.searchsorted(unique, values, side="right") + 1
    return dict(zip(keys, total.tolist()))


def _mean_pass_rank(
    sequence: str, observations: list[dict[str, float]],
    pooled_field: pd.DataFrame,
) -> float:
    """The validator's true multipass: rank EACH fold against the field, then
    mean the per-pass rank-sums.

    competitor_lane.multipass_scores / rescale_epoch rank each of a sequence's
    ~10 passes and average; ranking a single pooled aggregate instead would be a
    subtly different (lower-variance, but wrong) metric. A candidate with one
    observation so far is a 1-pass estimate of that mean -- it is returned
    as-is (the shrinkage/aggregation upstream already regularises the folds).
    """
    per_pass = {
        f"{sequence}::{index}": fold
        for index, fold in enumerate(observations)
    }
    ranks = rank_metrics_against_field(per_pass, pooled_field)
    if not ranks:
        return float("inf")
    return float(np.mean(list(ranks.values())))


#: Read-only context for _rank_candidate, set once per process: directly
#: in the main process for a serial run, or via _init_rank_worker in each
#: pool worker (a separate process, so it cannot see main()'s closure).
_RANK_CONTEXT: dict[str, Any] | None = None


def _init_rank_worker(context: dict[str, Any]) -> None:
    global _RANK_CONTEXT
    _RANK_CONTEXT = context


def _rank_candidate(item):
    """One sequence's candidacy check: identical to the old loop body,
    run in a worker process so RANK_WORKERS candidates are computed at
    once instead of one Python-bytecode-bound core carrying all of them.
    """
    sequence, observations = item
    ctx = _RANK_CONTEXT
    args = ctx["args"]
    compiled_fields = ctx["compiled_fields"]
    shrunk_by_sequence = ctx["shrunk_by_sequence"]
    consensus = ctx["consensus"]
    sample_selections = ctx["sample_selections"]
    historical_sequences = ctx["historical_sequences"]
    excluded_sequences = ctx["excluded_sequences"]
    only_sequences = ctx["only_sequences"]
    normalized_sequence = sequence.strip().upper()
    if only_sequences is not None and normalized_sequence not in only_sequences:
        return None
    if (
        normalized_sequence in historical_sequences
        or normalized_sequence in excluded_sequences
    ):
        return None
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
        return None
    if (
        observation_competitive_rate
        < args.minimum_observation_competitive_rate
    ):
        return None
    if (
        args.maximum_p90_rank_sum is not None
        and score_p90 > args.maximum_p90_rank_sum
    ):
        return None
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
        return None
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
    multipass_rank = _mean_pass_rank(
        normalized_sequence, observations, ctx["multipass_field"],
    ) if ctx.get("multipass_field") is not None else float("inf")
    return {
        "sequence": normalized_sequence,
        "settled_deviations": settled_deviation_count(
            normalized_sequence, consensus
        ),
        "observation_count": len(observations),
        # The selection value IS the multipass field-rank under the multipass
        # selection order; score_p90 is only a robustness qualification guard,
        # never the selection metric. Persisting the rank under selection_score
        # keeps every downstream consumer (checkpoint, fleet) honest even when
        # they read selection_score directly instead of multipass_rank.
        "selection_score": (
            multipass_rank
            if args.selection_order == "multipass"
            else score_p90
        ),
        "multipass_rank": multipass_rank,
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
    }


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
    parser.add_argument(
        "--full-archive-field",
        action="store_true",
        help=(
            "rank against the entire HF archive as one field instead of a small "
            "live-epoch field; ignores --api-field-count/--field-size/"
            "--bootstrap-fields. See ARCHITECTURE_REDESIGN new finding 7."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--before-epoch",
        type=int,
        help="use completed public leaderboard fields strictly before this epoch",
    )
    parser.add_argument("--api-field-count", type=int, default=0)
    parser.add_argument(
        "--db-field-count", type=int, default=0,
        help=(
            "like --api-field-count but reads real recent-epoch fields from "
            "the already-synced local observation history (source="
            "public_validator_api) instead of a live leaderboard API call; "
            "takes priority over --api-field-count when both are set, and "
            "requires --observation-history"
        ),
    )
    parser.add_argument("--minimum-api-fields", type=int, default=3)
    parser.add_argument("--api-scan-limit", type=int, default=40)
    parser.add_argument("--leaderboard-api-root", default=LEADERBOARD_API_ROOT)
    parser.add_argument(
        "--selection-order",
        choices=("competitive-rate", "p90", "win-probability", "multipass"),
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
    parser.add_argument("--current-exact-candidates-only", action="store_true",
                        help="use history as evidence only for candidates present in --exact inputs")
    parser.add_argument("--rank-workers", type=int,
                        default=min(32, os.cpu_count() or 1),
                        help="process pool size for the per-candidate ranking loop "
                             "(pure Python, GIL-bound to one core otherwise); 1 disables "
                             "the pool and ranks serially")
    parser.add_argument("--rank-workers-minimum-candidates", type=int, default=64,
                        help="use the pool only above this many candidates; below it, "
                             "process-start overhead outweighs the parallel gain")
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
    observations_by_sequence = _observations(args.exact, args.target)
    cold_reference = cold_start_reference(full_archive, observations_by_sequence,
        target=args.target, epoch=args.before_epoch, minimum_rows=args.field_size)
    if cold_reference is not None:
        full_archive = cold_reference
        args.field_size = min(args.field_size, len(full_archive))
        args.db_field_count = args.api_field_count = 0
    excluded_sequences = _excluded_sequences(args.exclude_file)
    archive = full_archive.tail(args.recent).reset_index(drop=True)
    if len(archive) < args.field_size:
        raise SystemExit("archive is smaller than the requested competitor field")
    missing = set(METRICS) - set(archive.columns)
    if missing:
        raise SystemExit(f"archive lacks metrics: {sorted(missing)}")

    if args.full_archive_field:
        # Rank against the entire HF archive as ONE field, not a small live-epoch
        # sample. The validator's own final_score ranks a submission only against
        # its ~50-sequence live epoch field (see validator_scores_from_observations),
        # but that single live draw is high-variance and noisy to select against.
        # The full-archive rank-sum is the lower-variance, multi-pass-averaged
        # metric that actually shows our measured, repeated gap against rivals
        # (ARCHITECTURE_REDESIGN new finding 7) -- deliberately NOT matching the
        # validator's literal field size; that was investigated and rejected.
        compiled_fields = [_compile_validator_field(full_archive)]
        reference = {
            "source": "full_hf_archive",
            "archive_rows": len(full_archive),
        }
    else:
        api_fields: list[tuple[int, pd.DataFrame]] = []
        field_reference_source = "public_leaderboard_api"
        if args.db_field_count:
            if args.before_epoch is None:
                parser.error("--db-field-count requires --before-epoch")
            if args.observation_history is None:
                parser.error("--db-field-count requires --observation-history")
            api_fields = fetch_recent_fields_from_db(
                args.observation_history,
                target=args.target,
                before_epoch=args.before_epoch,
                count=args.db_field_count,
            )
            field_reference_source = "db_cache_public_validator_api"
            from nanobody_opt import rules
            early_target = (args.target == rules.TARGET and args.before_epoch is not None
                            and rules.ACTIVATION_EPOCH <= args.before_epoch < rules.ACTIVATION_EPOCH + 3)
            if len(api_fields) < args.minimum_api_fields and not early_target:
                raise SystemExit(
                    f"only {len(api_fields)} complete {args.target} cached "
                    f"fields were available; require {args.minimum_api_fields}"
                )
        elif args.api_field_count:
            if args.before_epoch is None:
                parser.error("--api-field-count requires --before-epoch")
            api_fields = fetch_leaderboard_fields(
                args.leaderboard_api_root,
                before_epoch=args.before_epoch,
                target=args.target,
                count=args.api_field_count,
                scan_limit=args.api_scan_limit,
            )
            from nanobody_opt import rules
            early_target = (args.target == rules.TARGET and args.before_epoch is not None
                            and rules.ACTIVATION_EPOCH <= args.before_epoch < rules.ACTIVATION_EPOCH + 3)
            if len(api_fields) < args.minimum_api_fields and not early_target:
                raise SystemExit(
                    f"only {len(api_fields)} complete {args.target} leaderboard "
                    f"fields were available; require {args.minimum_api_fields}"
                )

        if api_fields:
            fields = [field for _, field in api_fields]
            reference = {
                "source": field_reference_source,
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
    if cold_reference is not None:
        reference.update(source="new_target_local_cohort", target=args.target,
                         competitive_history_available=False,
                         comparable_to_prior_target=False)
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
    observations_by_sequence = _observations(args.exact, args.target)
    only_sequences = None
    if args.only_sequences is not None:
        only_sequences = {line.strip().upper() for line in
                          args.only_sequences.read_text().splitlines() if line.strip()}
    if args.current_exact_candidates_only:
        current = set(observations_by_sequence)
        only_sequences = current if only_sequences is None else current & only_sequences
        if not only_sequences:
            raise SystemExit("current exact candidate cohort is empty")
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
    # pool_depths always maps every sequence in the merged pool to its fold
    # count; it feeds the depth print and the observation-floor relaxation
    # without materialising every fold (Fix 2).
    pool_depths: dict[str, int] = {}
    aggregate_stats: dict[str, dict[str, tuple[float, float, int]]] | None = None
    if args.observation_history is not None:
        with _history_snapshot(args.observation_history) as history_connection:
            exact_run_ids = _exact_run_ids(args.exact)
            aggregate_stats, _watermark = _read_aggregate_stats(
                args.observation_history, args.target, _connection=history_connection)
            if aggregate_stats is not None:
                # Fix 2 fast path.  Reconstruct the same merged pool the old
                # full-pool read produced, but from the numeric aggregate; read the
                # full per-observation vectors ONLY for the small candidate set.
                if exact_run_ids:
                    _add_stats(
                        aggregate_stats,
                        _excluded_run_stats(
                            args.observation_history, args.target, exact_run_ids,
                            _connection=history_connection),
                        -1,
                    )
                _add_stats(aggregate_stats,
                           _folds_to_stats(observations_by_sequence), 1)
                pool_depths = {
                    seq: max((count for _, _, count in metrics.values()), default=0)
                    for seq, metrics in aggregate_stats.items()
                }
                current_folds = sum(len(v) for v in observations_by_sequence.values())
                added = sum(pool_depths.values()) - current_folds
                # The full stored bank participates in comparison. Do not
                # materialize hundreds of thousands of one-pass histories
                # that cannot satisfy this invocation's evidence floor.
                # Keep their aggregate statistics for unchanged shrinkage.
                candidates = _candidate_set(
                    (seq for seq, depth in pool_depths.items()
                     if depth >= args.minimum_observations), only_sequences,
                    historical_sequences, excluded_sequences,
                )
                candidate_folds = _candidate_observation_folds(
                    args.observation_history, args.target,
                    candidates, excluded_run_ids=exact_run_ids,
                    _connection=history_connection,
                )
                for sequence in candidates:
                    observations_by_sequence[sequence] = (
                        observations_by_sequence.get(sequence, [])
                        + candidate_folds.get(sequence, [])
                    )
                # Non-candidates keep an entry so _rank_candidate can apply its
                # own filters and count the pool; their observations are never read
                # (they return before touching them).
                for sequence in pool_depths:
                    observations_by_sequence.setdefault(sequence, [])
                history_mode = "aggregate"
            else:
                # Aggregate absent or stale: fall back to the exact old full-pool
                # read, preserving the previous behaviour verbatim.
                history, added = _historical_observations(
                    args.observation_history,
                    args.target,
                    observations_by_sequence,
                    excluded_run_ids=exact_run_ids,
                )
                for sequence, folds in history.items():
                    observations_by_sequence[sequence].extend(folds)
                pool_depths = {
                    seq: len(v) for seq, v in observations_by_sequence.items()
                }
                history_mode = "full_read"
            depths = sorted(pool_depths.values())
            print(
                "EXACT_OBSERVATION_HISTORY merged=%d sequences=%d median_depth=%d mode=%s"
                % (added, len(pool_depths),
                   depths[len(depths) // 2] if depths else 0, history_mode),
                flush=True,
            )
    else:
        pool_depths = {
            seq: len(v) for seq, v in observations_by_sequence.items()
        }
    # Depth is uneven once history is merged -- a new proposal has one fold
    # and a previous finalist has scores of them -- so the means are not
    # equally trustworthy and must not be compared as if they were.
    shrunk_by_sequence: dict[str, dict[str, float]] = {}
    if args.metric_shrinkage:
        try:
            if aggregate_stats is not None:
                shrunk_by_sequence = shrinkage.shrunk_metrics_from_stats(
                    aggregate_stats, tuple(METRICS))
            else:
                shrunk_by_sequence = shrinkage.shrunk_metrics(
                    observations_by_sequence, tuple(METRICS))
        except (ValueError, statistics.StatisticsError):
            shrunk_by_sequence = {}
    ranked = []
    items = [
        (sequence, observations)
        for sequence, observations in observations_by_sequence.items()
        if len(observations) >= args.minimum_observations
        and (only_sequences is None or sequence in only_sequences)
        and sequence not in historical_sequences and sequence not in excluded_sequences
    ]
    # Load structures only for candidates with enough independent evidence.
    # The 11K screening pool used to load and qualify all retained poses before
    # applying the depth floor, blocking both confirmation and publication.
    if pose_challenge is not None and args.observation_history is not None:
        current_run_ids = _exact_run_ids(args.exact)
        for sequence, records in load_exact_artifacts(
            args.observation_history, target=args.target,
            sequences={sequence for sequence, _ in items},
        ).items():
            for record in records:
                if str(record.get('run_id') or '') in current_run_ids:
                    continue
                for key, destination in [('confidence_path', structure_paths),
                                         ('writer_path', writer_structure_paths)]:
                    if record.get(key):
                        destination.setdefault(sequence, []).append(Path(str(record[key])))
                selection = record.get('sample_selection')
                if isinstance(selection, dict):
                    sample_selections.setdefault(sequence, []).append(selection)
    # The multipass selection order needs the CURRENT last-10-epochs field,
    # recomputed at selection time (the field shifts as epochs roll, so a
    # cached field would rank against a stale cohort).
    multipass_field = None
    if args.selection_order == "multipass":
        if cold_reference is not None:
            multipass_field = cold_reference
        elif args.observation_history is not None and args.before_epoch is not None:
            multipass_field = pool_recent_fields(fetch_recent_fields_from_db(
                args.observation_history, target=args.target,
                before_epoch=args.before_epoch, count=10))
            if multipass_field.empty:
                raise SystemExit("multipass selection requires a non-empty field")
        elif compiled_fields:
            # Rare explicit-exact path has no DB history: fall back to the
            # compiled (archive/API) field rather than hard-failing.
            multipass_field = compiled_fields[0]
    global _RANK_CONTEXT
    _RANK_CONTEXT = {
        "args": args,
        "compiled_fields": compiled_fields,
        "multipass_field": multipass_field,
        "shrunk_by_sequence": shrunk_by_sequence,
        "consensus": consensus,
        "sample_selections": sample_selections,
        "historical_sequences": historical_sequences,
        "excluded_sequences": excluded_sequences,
        "only_sequences": only_sequences,
    }
    workers = max(1, args.rank_workers)
    if workers > 1 and len(items) >= args.rank_workers_minimum_candidates:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_rank_worker,
            initargs=(_RANK_CONTEXT,),
        ) as pool:
            results = list(pool.map(
                _rank_candidate, items,
                chunksize=max(1, len(items) // (workers * 4)),
            ))
    else:
        results = [_rank_candidate(item) for item in items]
    for result in results:
        if result is not None:
            ranked.append(result)

    # Selecting on a one-observation estimate is selecting on noise, but
    # publishing nothing is the only outcome that certainly pays nothing.  Take
    # the deepest evidence this epoch actually produced rather than either.
    observation_floor = args.minimum_observations
    if ranked:
        # The requested independent-observation floor is a hard requirement.
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

    # Pose stability dominates strict runtime.  The preliminary ordering is only
    # needed when there is a pose challenge to walk; without one it is a discarded
    # duplicate of the final ordering below, so it is skipped to avoid paying the
    # exact fresh-draw tournament twice per invocation.
    if pose_challenge is not None:
        preliminary = _order_candidates(
            ranked, args.selection_order, args.competitive_score_threshold,
            args.consensus_ranking_policy,
            rank_workers=args.rank_workers,
            rank_workers_minimum_candidates=args.rank_workers_minimum_candidates,
        )
        pose_early_stop = (
            not args.paired_consensus_shortlist
            and not args.diversity_reserve_count
            and _pose_order_is_subset_invariant(
                preliminary,
                args.selection_order,
                args.competitive_score_threshold,
            )
        )
    else:
        preliminary = ranked
        pose_early_stop = False
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
        rank_workers=args.rank_workers,
        rank_workers_minimum_candidates=args.rank_workers_minimum_candidates,
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
        "observation_history": {
            "requested": args.observation_history is not None,
            "status": (
                "read_failed" if _HISTORY_READ_ERROR
                else "merged" if args.observation_history is not None
                else "not_requested"
            ),
            "error": _HISTORY_READ_ERROR,
        },
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
