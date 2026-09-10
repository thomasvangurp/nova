"""Replay the validator's exact historical and top-50 similarity gates.

The cheap generation gate uses k-mer Jaccard to avoid obviously crowded
families. Finalists must also pass the validator's full alignment/CDR duplicate
rules against the live Algorithm-2 top 50; this command provides that proof
from the archive Ansible fetched for the current target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from nanobody_opt.diversity import minmax_weighted_rank
from nanobody_opt.filter_layer2 import _load_candidates


def duplicate_reason(identity: float, cdr_similarity: dict[str, float] | None) -> str | None:
    """Mirror ``utils.nanobodies.is_duplicate`` without importing miner code."""
    if identity >= 0.95:
        return "near-identical sequence"
    if cdr_similarity is not None:
        cdr3 = cdr_similarity.get("CDR3", 0.0)
        if cdr3 >= 0.85:
            return f"CDR3 identity {cdr3:.0%} (same clonotype)"
        if (
            cdr3 >= 0.80
            and cdr_similarity.get("CDR1", 0.0) >= 0.90
            and cdr_similarity.get("CDR2", 0.0) >= 0.90
        ):
            return f"CDR3={cdr3:.0%} with conserved CDR1/CDR2"
    if identity >= 0.90 and cdr_similarity is None:
        return f"whole-sequence identity {identity:.0%} (no CDR data)"
    return None


def _build_search_engine(top: pd.DataFrame):
    from metanano.config import SearchConfig
    from metanano.search import IndexManager, SearchEngine
    from metanano.utils.alignment import AlignmentEngine
    from metanano.utils.cdr_utils import extract_cdrs
    from metanano.utils.kmer import generate_kmers

    config = SearchConfig()
    index = IndexManager(config)
    engine = SearchEngine(config, index, AlignmentEngine(config.fine_alignment))
    for selection_rank, row in enumerate(top.itertuples(index=False), start=1):
        sequence = str(row.sequence).strip().upper()
        cdrs = extract_cdrs(sequence)
        if cdrs is None:
            raise RuntimeError(
                f"failed to annotate Algorithm-2 top-50 sequence {selection_rank}"
            )
        sequence_id = str(getattr(row, "sequence_hash", "")) or hashlib.sha256(
            sequence.encode("ascii")
        ).hexdigest()
        index.add_sequence(
            sequence_id, sequence, cdrs, generate_kmers(sequence, k=config.k)
        )
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--max-liability-violations", type=int, default=50)
    args = parser.parse_args()

    candidates = _load_candidates(args.input)
    if not candidates:
        raise SystemExit("no candidate sequences found")
    archive = pd.read_csv(args.archive)
    historical = set(archive.sequence.dropna().astype(str).str.strip().str.upper())
    eligible = archive[
        archive.liability_num_violations <= args.max_liability_violations
    ].copy()
    top = minmax_weighted_rank(eligible, k=args.top)
    if len(top) != args.top:
        raise SystemExit(f"expected {args.top} top sequences, got {len(top)}")
    engine = _build_search_engine(top)

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        sequence = candidate["sequence"]
        exact_historical = sequence in historical
        search = engine.search(sequence, include_alignment=True)
        matches = []
        for match in search.matches:
            reason = duplicate_reason(match.identity, match.cdr_similarity)
            matches.append(
                {
                    "target_id": match.target_id,
                    "identity": match.identity,
                    "cdr_similarity": match.cdr_similarity,
                    "duplicate_reason": reason,
                }
            )
        duplicate_matches = [match for match in matches if match["duplicate_reason"]]
        results.append(
            {
                **candidate,
                "exact_historical": exact_historical,
                "indexed_top": search.total_indexed,
                "coarse_candidates": search.total_candidates,
                "maximum_identity": max(
                    (match["identity"] for match in matches), default=0.0
                ),
                "duplicate_matches": duplicate_matches,
                "validator_novel": not exact_historical and not duplicate_matches,
                "matches": matches,
            }
        )

    passed = [item for item in results if item["validator_novel"]]
    payload = {
        "target": args.target,
        "archive": str(args.archive),
        "archive_count": len(archive),
        "algorithm2_top_count": len(top),
        "input_count": len(results),
        "pass_count": len(passed),
        "sequences": [item["sequence"] for item in passed],
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"validator novelty: {len(passed)}/{len(results)} pass against "
        f"{len(archive):,} historical sequences and Algorithm-2 top {len(top)}"
    )
    for item in results:
        label = item.get("mutation", item["sequence"][:12])
        print(
            f"  {'PASS' if item['validator_novel'] else 'FAIL'} {label}: "
            f"exact={item['exact_historical']} coarse={item['coarse_candidates']} "
            f"max_identity={item['maximum_identity']:.3f} "
            f"duplicates={len(item['duplicate_matches'])}"
        )


if __name__ == "__main__":
    main()
