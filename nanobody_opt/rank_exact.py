"""Rank exact nanobody observations with the validator's dense-rank objective.

BoltzGen inference is stochastic, so a live finalist must survive more than one
exact observation. This utility applies the validator's per-metric dense ranks
to deterministic bootstrap fields and uses the candidate's worst observation
median as its conservative selection statistic. Lower is better. The statistic
is field-relative; it is not presented as a prediction of the next API score.
"""
from __future__ import annotations

import argparse
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
from nanobody_opt.simulate_round import (
    METRICS,
    category_rank_sums,
    rank_sum_in_field,
)


LEADERBOARD_API_ROOT = "https://compound-api.metanova-labs.ai"


def _observations(paths: list[Path], target: str) -> dict[str, list[dict[str, float]]]:
    found: dict[str, list[dict[str, float]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text())
        if str(payload.get("target")) != target:
            raise SystemExit(f"{path} targets {payload.get('target')!r}, not {target}")
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
    parser.add_argument("--exact", action="append", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--minimum-observations", type=int, default=1)
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
    args = parser.parse_args()

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
    ranked = []
    for sequence, observations in _observations(args.exact, args.target).items():
        normalized_sequence = sequence.strip().upper()
        if (
            normalized_sequence in historical_sequences
            or normalized_sequence in excluded_sequences
        ):
            continue
        if len(observations) < args.minimum_observations:
            continue
        per_observation = [
            float(np.median([rank_sum_in_field(metrics, field) for field in fields]))
            for metrics in observations
        ]
        aggregate = {
            name: float(np.median([item[name] for item in observations]))
            for name in METRICS
        }
        aggregate_field_scores = [
            rank_sum_in_field(aggregate, field) for field in fields
        ]
        aggregate_category_scores = {
            category: [
                category_rank_sums(aggregate, field)[category]
                for field in fields
            ]
            for category in ("confidence", "physical_interaction", "developability")
        }
        ranked.append({
            "sequence": normalized_sequence,
            "observation_count": len(observations),
            "selection_score": max(per_observation),
            "per_observation_median_rank_sum": per_observation,
            "aggregate_median_rank_sum": float(np.median(aggregate_field_scores)),
            "aggregate_p90_rank_sum": float(np.quantile(aggregate_field_scores, 0.9)),
            "aggregate_median_category_rank_sums": {
                category: float(np.median(scores))
                for category, scores in aggregate_category_scores.items()
            },
            "metrics": aggregate,
            "observations": observations,
        })

    ranked.sort(key=lambda item: (
        item["selection_score"], item["aggregate_p90_rank_sum"], item["sequence"]
    ))
    selected = ranked[: args.limit]
    if not selected:
        raise SystemExit("no complete exact candidates met the observation requirement")
    payload = {
        "target": args.target,
        "exact_files": [str(path) for path in args.exact],
        "minimum_observations": args.minimum_observations,
        "rank_method": "dense",
        "score_semantics": (
            "validator-compatible per-metric dense rank sum against historical "
            "reference fields; not a prediction of the next stochastic observation"
        ),
        "reference": reference,
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
            f"  score={item['selection_score']:.1f} "
            f"p90={item['aggregate_p90_rank_sum']:.1f} "
            f"n={item['observation_count']} {item['sequence']}"
        )


if __name__ == "__main__":
    main()
