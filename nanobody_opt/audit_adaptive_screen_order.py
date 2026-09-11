"""Replay fixed versus parent-balanced exact-screen tranche ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from nanobody_opt.sequential_exact_race import _balanced_screen_tranche


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _balanced_order(frame: pd.DataFrame, tranche_size: int) -> list[str]:
    sequences = frame.sequence.astype(str).tolist()
    metadata = {
        str(row.sequence): {"parent_sequence": str(row.parent_sequence)}
        for row in frame.itertuples()
    }
    pending = list(sequences)
    parent_counts: Counter[str] = Counter()
    ordered: list[str] = []
    while pending:
        tranche = _balanced_screen_tranche(
            pending, metadata, parent_counts, tranche_size
        )
        for sequence in tranche:
            pending.remove(sequence)
            parent_counts[str(metadata[sequence]["parent_sequence"])] += 1
        ordered.extend(tranche)
    return ordered


def _prefix_metrics(
    order: list[str], outcomes: pd.DataFrame, prefixes: list[int]
) -> dict[str, dict[str, float | int]]:
    indexed = outcomes.set_index("sequence")
    high_score_count = max(1, round(len(outcomes) * 0.1))
    output = {}
    for prefix in prefixes:
        cohort = indexed.loc[order[:prefix]]
        high = int((cohort.bootstrap_rank <= high_score_count).sum())
        output[str(prefix)] = {
            "best_bootstrap_rank": int(cohort.bootstrap_rank.min()),
            "top_10_count": int((cohort.bootstrap_rank <= 10).sum()),
            "top_10pct_count": high,
            "top_10pct_recall": high / high_score_count,
            "distinct_parents": int(cohort.parent_sequence.nunique()),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tranche-size", type=int, default=20)
    parser.add_argument("--prefixes", default="20,40,80,160,320")
    args = parser.parse_args()
    if args.tranche_size < 1:
        parser.error("--tranche-size must be positive")

    proposals = pd.read_csv(args.proposals)
    outcomes = pd.read_csv(args.outcomes)
    required = {"sequence", "parent_sequence", "bootstrap_rank"}
    missing = required - set(outcomes.columns)
    if missing:
        raise SystemExit(f"outcomes lack columns: {sorted(missing)}")
    proposal_order = {
        str(sequence): index for index, sequence in enumerate(proposals.sequence)
    }
    outcomes["proposal_order"] = outcomes.sequence.map(proposal_order)
    if outcomes.proposal_order.isna().any():
        raise SystemExit("some scored candidates are absent from proposals")
    outcomes = outcomes.sort_values("proposal_order").reset_index(drop=True)
    baseline = outcomes.sequence.astype(str).tolist()
    balanced = _balanced_order(outcomes, args.tranche_size)
    prefixes = [
        min(int(value), len(outcomes))
        for value in args.prefixes.split(",")
        if value.strip()
    ]
    payload = {
        "candidate_count": len(outcomes),
        "parent_count": int(outcomes.parent_sequence.nunique()),
        "protocol": {
            "tranche_size": args.tranche_size,
            "high_scorer_definition": "best 10% by bootstrap validator rank",
            "baseline": "fixed proposal order",
            "adaptive": (
                "rebuild pending queue after every tranche and select the "
                "least-screened parent, preserving proposal order for ties"
            ),
            "graph_score_policy": (
                "no total-score neighbour pruning: saved exact graph evidence "
                "is too weak for safe elimination"
            ),
        },
        "inputs": {
            "proposals": {
                "path": str(args.proposals),
                "sha256": _sha256(args.proposals),
            },
            "outcomes": {
                "path": str(args.outcomes),
                "sha256": _sha256(args.outcomes),
            },
        },
        "fixed_order": _prefix_metrics(baseline, outcomes, prefixes),
        "parent_balanced": _prefix_metrics(balanced, outcomes, prefixes),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
