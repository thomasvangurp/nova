"""Build an elite parent bank straight from the mutation database.

Override seam for ``select_elite_parents``: the producer only runs
``select_elite_parents`` when ``<producer_root>/elite_parents.json`` does not
already exist, so writing this file first makes it the parent bank for the
next producer generation.

Selection rule (matches the saved ``prepare_children.json`` ``elite_parents``
list): a sequence is public iff it has a ``validator_final_rank_sum``
observation.  Its best-available score is the mean of its
``local_validator_replay_rank_sum`` scores when any exist, otherwise its best
single-pass ``validator_final_rank_sum`` score.  Lower is better.  The bank is
the top ``fraction`` (default 12%) of public sequences by that score.

The emitted JSON mirrors the ``candidates``/``sequences`` schema written by
``select_elite_parents.main``, so ``filter_layer2`` and ``local_neighborhood``
consume it unchanged.  Only ``sequence`` and ``selection_score`` are load-
bearing; ``parent_selection_lane`` must stay ``epoch_elite`` so
``local_neighborhood`` classifies these as core seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from nanobody_opt import db

DEFAULT_TARGET = "P20809"
DEFAULT_FRACTION = 0.12
DEFAULT_SINGLE_KIND = "validator_final_rank_sum"
DEFAULT_REPLAY_KIND = "local_validator_replay_rank_sum"
N_TERMINAL_WIDTH = 10
C_TERMINAL_WIDTH = 17


def basin_id(sequence: str) -> str:
    """Inline copy of ``nanobody_opt.basins.basin_id`` (self-contained CLI)."""
    normalized = sequence.strip().upper()
    if len(normalized) <= N_TERMINAL_WIDTH + C_TERMINAL_WIDTH:
        core = normalized
    else:
        end = len(normalized) - C_TERMINAL_WIDTH
        core = normalized[N_TERMINAL_WIDTH:end]
    return hashlib.sha256(core.encode("ascii")).hexdigest()[:16]


def _connect(path: Path) -> db.Connection:
    return db.connect()


def load_public(path: Path, target: str, single_kind: str) -> dict[str, dict[str, Any]]:
    """Every single-pass public observation, keyed by sequence.

    Returns ``sequence -> {"observations": [(score, epoch, metrics)], ...}``.
    """
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT n.sequence, o.score, o.epoch, o.metrics_json
            FROM observations o
            JOIN nodes n ON n.id = o.node_id
            WHERE n.target = %s AND o.score_kind = %s
            ORDER BY o.epoch, o.score, n.sequence
            """,
            (target, single_kind),
        ).fetchall()
    public: dict[str, dict[str, Any]] = {}
    for sequence, score, epoch, metrics_json in rows:
        try:
            metrics = json.loads(metrics_json) if metrics_json else {}
        except (TypeError, json.JSONDecodeError):
            metrics = {}
        entry = public.setdefault(sequence, {"observations": []})
        entry["observations"].append((float(score), int(epoch), metrics))
    for entry in public.values():
        entry["best_single"] = min(obs[0] for obs in entry["observations"])
    return public


def load_replay_means(path: Path, target: str, replay_kind: str) -> dict[str, float]:
    """Mean multi-pass replay score per sequence (lower is better).

    Replaced by :func:`load_corrected_scores` for the parent ranking: the stored
    ``score`` sums all ten metric dense ranks, whereas the validator ranks each
    of the three metric categories and sums those.  Kept for reference.
    """
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT n.sequence, AVG(o.score), COUNT(*)
            FROM observations o
            JOIN nodes n ON n.id = o.node_id
            WHERE n.target = %s AND o.score_kind = %s
            GROUP BY n.sequence
            """,
            (target, replay_kind),
        ).fetchall()
    return {sequence: float(mean) for sequence, mean, _count in rows}


def load_corrected_scores(path: Path, target: str, replay_kind: str) -> dict[str, float]:
    """Corrected validator ``final_score`` per sequence from the replay metrics.

    The validator does not sum all ten metric dense ranks; it ranks each of the
    three metric categories within the field and sums those three ranks (see
    ``simulate_round.validator_final_scores``).  The replay metrics_json already
    carries ``interaction_pae`` and the other nine metrics, so this recomputes the
    correct score without re-scoring.  Lower is better.
    """
    from nanobody_opt.mutation_network import load_observations
    from nanobody_opt.simulate_round import validator_scores_from_observations

    observations = load_observations(path, target=target)
    return validator_scores_from_observations(
        observations, score_kind=replay_kind, by_epoch=True
    )


def _epoch_rank_table(public: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per-sequence ``public_observations`` with within-epoch rank, 1-based."""
    by_epoch: dict[int, list[tuple[float, str]]] = {}
    for sequence, entry in public.items():
        for score, epoch, _metrics in entry["observations"]:
            by_epoch.setdefault(epoch, []).append((score, sequence))
    ranks: dict[tuple[int, str], int] = {}
    for epoch, scored in by_epoch.items():
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        for rank, (_score, sequence) in enumerate(scored, start=1):
            ranks[(epoch, sequence)] = rank
    return {
        sequence: [
            {
                "epoch": epoch,
                "epoch_rank": ranks[(epoch, sequence)],
                "validator_final_score": score,
            }
            for score, epoch, _metrics in entry["observations"]
        ]
        for sequence, entry in public.items()
    }


def build_candidates(
    public: dict[str, dict[str, Any]],
    replay_means: dict[str, float],
    *,
    fraction: float = DEFAULT_FRACTION,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Rank public sequences by best-available score and cut the top fraction."""
    if not public:
        return [], 0

    def best(sequence: str) -> float:
        mean = replay_means.get(sequence)
        if mean is not None:
            return mean
        return public[sequence]["best_single"]

    ranked = sorted(public, key=lambda seq: (best(seq), seq))
    count = limit if limit > 0 else round(len(ranked) * fraction)
    count = max(1, min(count, len(ranked)))
    selected = ranked[:count]

    observation_table = _epoch_rank_table(public)
    candidates: list[dict[str, Any]] = []
    for index, sequence in enumerate(selected):
        entry = public[sequence]
        best_single = entry["best_single"]
        # Metrics from the best-scoring single-pass observation (same ten keys
        # in every metrics_json for this target).
        metrics: dict[str, Any] = {}
        for score, _epoch, obs_metrics in entry["observations"]:
            if score == best_single:
                metrics = obs_metrics
                break
        score = best(sequence)
        candidates.append({
            "sequence": sequence,
            "selection_score": score,
            "validator_final_score": best_single,
            "metrics": metrics,
            "parent_source": "public_leaderboard_db_top12",
            "parent_selection_lane": "epoch_elite",
            "nearest_elite_distance": 0,
            "parent_history_scope": "global",
            "basin_core_id": basin_id(sequence),
            "basin_rank": index,
            "basin_allocation_reserved": False,
            "public_observations": observation_table[sequence],
        })
    return candidates, len(ranked)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="share of public leaderboard sequences to keep (default 0.12)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="exact parent count override; zero means use --fraction",
    )
    parser.add_argument("--before-epoch", type=int, default=None)
    parser.add_argument("--single-score-kind", default=DEFAULT_SINGLE_KIND)
    parser.add_argument("--replay-score-kind", default=DEFAULT_REPLAY_KIND)
    args = parser.parse_args()

    if not args.database.is_file():
        parser.error(f"--database does not exist: {args.database}")
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    public = load_public(args.database, args.target, args.single_score_kind)
    corrected_scores = load_corrected_scores(
        args.database, args.target, args.replay_score_kind
    )
    candidates, total_public = build_candidates(
        public, corrected_scores, fraction=args.fraction, limit=args.limit
    )
    if not candidates:
        raise SystemExit("no public leaderboard sequences in database")

    complete_epochs = sorted(
        {
            epoch
            for entry in public.values()
            for _score, epoch, _metrics in entry["observations"]
        }
    )
    _atomic_json(args.output, {
        "target": args.target,
        "before_epoch": args.before_epoch,
        "source": "nanobody_mutations.sqlite:top_fraction",
        "generator": "elite_parents_from_db",
        "fraction": args.fraction,
        "complete_epochs": complete_epochs,
        "sequences": [item["sequence"] for item in candidates],
        "candidates": candidates,
    })
    print(
        f"ELITE_PARENTS_DB_READY target={args.target} public={total_public} "
        f"parents={len(candidates)} output={args.output}"
    )
    for item in candidates[:10]:
        provenance = item["public_observations"][0]
        print(
            f"  score={item['selection_score']:.2f} "
            f"epoch={provenance['epoch']} rank={provenance['epoch_rank']} "
            f"{item['sequence']}"
        )


if __name__ == "__main__":
    main()
