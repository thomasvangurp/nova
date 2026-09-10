"""Reconstruct and compare recent per-epoch nanobody winners.

The public archive intentionally stores only immutable sequence measurements,
not epoch numbers.  The Hugging Face commit messages do retain the epoch and
batch size.  Exact duplicates are rejected by the validator, so a unique
archive can be partitioned from its tail using those batch sizes without
guessing a fixed number of submissions per epoch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from nanobody_opt.simulate_round import METRICS, dense_rank_sums


COMMIT_API = (
    "https://huggingface.co/api/datasets/Metanova/Submission-Archive/commits/main"
)
COMMIT_RE = re.compile(
    r"Added (?P<count>\d+) nanobody sequences from epoch (?P<epoch>\d+) "
    r"to (?P<filename>[^/]+_nanobodies[.]csv)"
)


def parse_epoch_commits(
    commits: list[dict[str, Any]], filename: str
) -> list[dict[str, Any]]:
    """Return newest-first batch boundaries for one target archive."""
    batches = []
    for commit in commits:
        match = COMMIT_RE.fullmatch(str(commit.get("title", "")))
        if not match or match.group("filename") != filename:
            continue
        batches.append({
            "epoch": int(match.group("epoch")),
            "count": int(match.group("count")),
            "commit": str(commit["id"]),
            "date": commit.get("date"),
        })
    return batches


def fetch_epoch_commits(
    api: str, filename: str, epochs: int, page_size: int = 100
) -> list[dict[str, Any]]:
    """Fetch enough public commit pages to cover the requested epochs."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    batches: list[dict[str, Any]] = []
    page = 0
    while len(batches) < epochs:
        query = urlencode({"p": page, "limit": page_size})
        with urlopen(f"{api}?{query}", timeout=30) as response:  # noqa: S310
            commits = json.load(response)
        if not commits:
            break
        batches.extend(parse_epoch_commits(commits, filename))
        page += 1
    if len(batches) < epochs:
        raise ValueError(
            f"commit history exposes {len(batches)} {filename} batches; "
            f"requested {epochs}"
        )
    return batches[:epochs]


def partition_recent_epochs(
    archive: pd.DataFrame, batches: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Partition unique archive rows from newest to oldest batch."""
    if "sequence_hash" in archive and archive.sequence_hash.duplicated().any():
        raise ValueError("archive contains duplicate sequence hashes")
    end = len(archive)
    groups = []
    for batch in batches:
        count = int(batch["count"])
        if count <= 0 or count > end:
            raise ValueError(
                f"invalid epoch {batch['epoch']} batch size {count} at row {end}"
            )
        start = end - count
        groups.append((batch, archive.iloc[start:end].copy().reset_index(drop=True)))
        end = start
    return groups


def rank_epoch(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the validator's ten-metric rank sum within one epoch."""
    missing = set(METRICS) - set(frame)
    if missing:
        raise ValueError(f"archive lacks score metrics: {sorted(missing)}")
    ranked = frame.copy()
    ranked["epoch_rank_sum"] = dense_rank_sums(
        ranked, METRICS, na_option="bottom"
    )
    return ranked.sort_values(
        ["epoch_rank_sum", "sequence"], kind="stable"
    ).reset_index(drop=True)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pairwise_identity(sequences: list[str]) -> dict[str, float | None]:
    identities = []
    for index, left in enumerate(sequences):
        for right in sequences[index + 1 :]:
            if len(left) != len(right):
                continue
            identities.append(
                sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
            )
    if not identities:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": float(np.min(identities)),
        "median": float(np.median(identities)),
        "maximum": float(np.max(identities)),
    }


def analyze_epoch_groups(
    groups: list[tuple[dict[str, Any], pd.DataFrame]]
) -> dict[str, Any]:
    winners = []
    field_rows = []
    for batch, frame in groups:
        ranked = rank_epoch(frame)
        winner = ranked.iloc[0]
        record = {
            **batch,
            "field_size": len(frame),
            "sequence": str(winner.sequence),
            "epoch_rank_sum": float(winner.epoch_rank_sum),
        }
        for metric in METRICS:
            record[metric] = _finite(winner[metric])
        winners.append(record)
        field_rows.extend(frame.to_dict("records"))

    lengths = Counter(len(item["sequence"]) for item in winners)
    modal_length = lengths.most_common(1)[0][0]
    modal_winners = [
        item["sequence"] for item in winners if len(item["sequence"]) == modal_length
    ]
    modal_field = [
        str(item["sequence"])
        for item in field_rows
        if len(str(item.get("sequence", ""))) == modal_length
    ]
    variable_positions = []
    for index in range(modal_length):
        winner_counts = Counter(seq[index] for seq in modal_winners)
        field_counts = Counter(seq[index] for seq in modal_field)
        residue, count = winner_counts.most_common(1)[0]
        winner_frequency = count / len(modal_winners)
        field_frequency = field_counts[residue] / len(modal_field)
        if len(winner_counts) > 1 or abs(winner_frequency - field_frequency) >= 0.1:
            variable_positions.append({
                "position": index + 1,
                "winner_consensus": residue,
                "winner_frequency": winner_frequency,
                "field_frequency": field_frequency,
                "winner_residues": dict(winner_counts.most_common()),
            })

    medians = {
        metric: float(np.median([
            item[metric] for item in winners if item[metric] is not None
        ]))
        for metric in ["epoch_rank_sum", *METRICS]
    }
    return {
        "epoch_count": len(groups),
        "winners": winners,
        "winner_summary": {
            "unique_sequence_count": len(set(item["sequence"] for item in winners)),
            "modal_length": modal_length,
            "modal_length_winner_count": len(modal_winners),
            "pairwise_identity": _pairwise_identity(modal_winners),
            "metric_medians": medians,
            "variable_or_enriched_positions": variable_positions,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--commit-api", default=COMMIT_API)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    archive = pd.read_csv(args.archive)
    batches = fetch_epoch_commits(args.commit_api, args.filename, args.epochs)
    result = analyze_epoch_groups(partition_recent_epochs(archive, batches))
    result.update({
        "archive": str(args.archive),
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "filename": args.filename,
        "commit_api": args.commit_api,
        "partition_basis": (
            "newest-first Hugging Face commit batch sizes over a unique, "
            "append-only validator archive"
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    summary = result["winner_summary"]
    identity = summary["pairwise_identity"]
    print(
        f"EPOCH_WINNERS_READY epochs={result['epoch_count']} "
        f"unique={summary['unique_sequence_count']} "
        f"median_pair_identity={identity['median']:.3f} output={args.output}"
    )
    for winner in result["winners"]:
        print(
            f"  epoch={winner['epoch']} n={winner['field_size']} "
            f"rank_sum={winner['epoch_rank_sum']:.1f} "
            f"iIPTM={winner['design_iiptm']:.5f} "
            f"liability={winner['liability_score']:.0f}/"
            f"{winner['liability_num_violations']:.0f} "
            f"{winner['sequence']}"
        )


if __name__ == "__main__":
    main()
