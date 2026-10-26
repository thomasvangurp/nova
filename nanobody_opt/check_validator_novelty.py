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

from nanobody_opt.filter_layer2 import _load_candidates


def _rank_binders(archive: pd.DataFrame, top: int, max_violations: int):
    """The validator's own top-k selection, not our reimplementation of it.

    utils/nanobodies.py::index_top_sequences calls
    utils.minmax_weighted_rank.rank_binders. This module used
    nanobody_opt.diversity.minmax_weighted_rank -- a different function with a
    different score -- so the replay indexed a different fifty sequences than
    the validator and reported "30/30 novel" with zero coarse candidates while
    our fleet sat at 0.95+ identity to the live rank-1 sequence. Replaying a
    gate with a lookalike of its inputs proves nothing.
    """
    if top == 0:
        return archive.copy()
    # Loaded by path, not as utils.minmax_weighted_rank: utils/__init__.py
    # pulls in rdkit, which the metanano environment does not carry. This is
    # still the validator's own source file, not a copy of it.
    import importlib.util

    source = Path(__file__).resolve().parent.parent / "utils" / "minmax_weighted_rank.py"
    spec = importlib.util.spec_from_file_location("_nova_minmax_weighted_rank", source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the validator's ranking from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rank_binders = module.rank_binders

    return rank_binders(
        archive, k=top, max_liability_violations=max_violations
    ).head(top)


def duplicate_reason(
    identity: float, cdr_similarity: dict[str, float] | None,
    identity_threshold: float = 0.95,
    no_cdr_gate: bool = False,
) -> str | None:
    """Mirror ``utils.nanobodies.is_duplicate`` without importing miner code.

    ``identity_threshold`` relaxes only the whole-sequence identity bar (the
    parent/seed gate passes 0.97 so near-boundary parents whose offspring can
    cross below 0.95 are kept). ``no_cdr_gate`` disables the CDR3 clonotype and
    conserved-CDR checks entirely -- used ONLY by the parent/seed gate, because
    a parent that shares the CDR3 clonotype can still mutate the CDR3 into an
    admissible child. The child/candidate gate keeps the full rule.
    """
    if identity >= identity_threshold:
        return "near-identical sequence"
    if not no_cdr_gate:
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


def _build_search_engine(top: pd.DataFrame, *, all_archive: bool = False):
    if all_archive:
        from nanobody_opt.pr100_filter import activate_filter
        activate_filter()
    from metanano.config import SearchConfig
    from metanano.search import IndexManager, SearchEngine
    from metanano.utils.alignment import AlignmentEngine
    from metanano.utils.cdr_utils import extract_cdrs
    from metanano.utils.kmer import generate_kmers

    config = SearchConfig()
    index = IndexManager(config)
    engine = SearchEngine(config, index, AlignmentEngine(config.fine_alignment))
    numbered = {}
    if all_archive and len(top):
        from abnumber import Chain
        # Match PR100's archive-side AbNumber batch annotation, retaining rows
        # with missing CDR data for its whole-sequence fallback check.
        items = {str(i): str(s).strip().upper() for i, s in enumerate(top.sequence)}
        for start in range(0, len(items), 512):
            batch = {str(i): items[str(i)] for i in range(start, min(start + 512, len(items)))}
            chains, _ = Chain.batch(batch, scheme="imgt")
            numbered.update(chains)
    for selection_rank, row in enumerate(top.itertuples(index=False), start=1):
        sequence = str(row.sequence).strip().upper()
        if all_archive:
            chain = numbered.get(str(selection_rank - 1))
            cdrs = ({key: str(getattr(chain, key + "_seq") or "")
                     for key in ("cdr1", "cdr2", "cdr3")} if chain is not None else None)
        else:
            cdrs = extract_cdrs(sequence)
        if cdrs is None and not all_archive:
            # Match index_top_sequences: a top-50 sequence whose CDRs cannot be
            # annotated is skipped (its igblast fallback is broken upstream --
            # called with a path against a Dict[str, str] signature), so it is
            # simply not indexed and cannot produce a duplicate match.
            continue
        sequence_id = str(getattr(row, "sequence_hash", "")) or hashlib.sha256(
            sequence.encode("ascii")
        ).hexdigest()
        index.add_sequence(
            sequence_id, sequence, cdrs, generate_kmers(sequence, k=config.k)
        )
    return engine


#: Set once per worker process by _init_worker, never pickled/shared across
#: processes directly -- each worker builds its own SearchEngine from `top`.
_WORKER_ENGINE = None
_WORKER_HISTORICAL: set[str] = set()
_WORKER_IDENTITY_THRESHOLD = 0.95
_WORKER_NO_CDR_GATE = False


def _init_worker(top: pd.DataFrame, historical: set[str],
                 identity_threshold: float = 0.95,
                 no_cdr_gate: bool = False, all_archive: bool = False) -> None:
    global _WORKER_ENGINE, _WORKER_HISTORICAL, _WORKER_IDENTITY_THRESHOLD, _WORKER_NO_CDR_GATE
    _WORKER_ENGINE = (_build_search_engine(top, all_archive=True) if all_archive
                      else _build_search_engine(top))
    _WORKER_HISTORICAL = historical
    _WORKER_IDENTITY_THRESHOLD = identity_threshold
    _WORKER_NO_CDR_GATE = no_cdr_gate


def _search_one(candidate: dict[str, Any]) -> dict[str, Any]:
    """Exactly the per-candidate body of the old serial loop, unchanged --
    only the engine's origin (built once per worker, not once per process
    overall) and the dispatch (pool.map instead of a for loop) differ.

    coarse_min_shared/coarse_jaccard are None here, matching the validator's
    nanobody_validity.py call: None resolves to the SearchEngine's configured
    coarse-filter defaults (min_shared_kmers/jaccard_threshold), so the
    candidate set -- and therefore is_duplicate -- is byte-identical to the
    validator's, not a superset.
    """
    sequence = candidate["sequence"]
    exact_historical = sequence in _WORKER_HISTORICAL
    search = _WORKER_ENGINE.search(
        sequence,
        include_alignment=True,
        exclude_ids=None,
        coarse_min_shared=None,
        coarse_jaccard=None,
    )
    matches = []
    for match in search.matches:
        reason = duplicate_reason(
            match.identity, match.cdr_similarity,
            identity_threshold=_WORKER_IDENTITY_THRESHOLD,
            no_cdr_gate=_WORKER_NO_CDR_GATE,
        )
        matches.append(
            {
                "target_id": match.target_id,
                "identity": match.identity,
                "cdr_similarity": match.cdr_similarity,
                "duplicate_reason": reason,
            }
        )
    duplicate_matches = [match for match in matches if match["duplicate_reason"]]
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--max-liability-violations", type=int, default=50)
    parser.add_argument(
        "--identity-threshold", type=float, default=0.95,
        help=(
            "whole-sequence identity bar for 'near-identical sequence'; the "
            "parent/seed gate passes 0.97 so near-boundary parents whose 1-edit "
            "offspring can cross below 0.95 are kept. The child/candidate gate "
            "uses the default 0.95."
        ),
    )
    parser.add_argument(
        "--no-cdr-gate", action="store_true",
        help=(
            "disable the CDR3 clonotype and conserved-CDR duplicate rules, "
            "leaving ONLY whole-sequence identity. Used by the parent/seed gate "
            "(a parent sharing the CDR3 clonotype can still mutate the CDR3 into "
            "an admissible child); the child/candidate gate never passes this."
        ),
    )
    parser.add_argument(
        "--concurrency", type=int, default=64,
        help=(
            "candidate searches run in parallel worker processes (each "
            "candidate's search only reads the shared top-50 index, so "
            "this is embarrassingly parallel); 1 reproduces the old serial "
            "behavior exactly, including error messages"
        ),
    )
    parser.add_argument(
        "--rejected-output", type=Path, default=None,
        help=(
            "also write the input sequences the validator would reject, one "
            "per line; the publisher's pre-shortlist gate reads this"
        ),
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help=(
            "leave the per-candidate detail out of --output and stdout; a "
            "20,000-candidate pre-filter would otherwise write ~770 MB"
        ),
    )
    args = parser.parse_args()
    from nanobody_opt import rules
    all_archive = args.target == rules.TARGET
    if all_archive:
        args.top = 0

    candidates = _load_candidates(args.input)
    if not candidates:
        # A valid-but-empty candidate set is a successful empty gate, not a
        # failure: the historical fallback downstream must be allowed to run
        # when a generation produces zero Layer-2 survivors. Malformed input
        # still raises inside _load_candidates (json.loads / csv parse).
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "target": args.target,
            "archive": str(args.archive),
            "archive_count": 0,
            "algorithm2_top_count": 0,
            "input_count": 0,
            "pass_count": 0,
            "sequences": [],
            "candidates": [],
            "all_results": [],
        }, indent=2) + "\n")
        return
    archive = pd.read_csv(args.archive)
    historical = set(archive.sequence.dropna().astype(str).str.strip().str.upper())
    # No pre-filter of our own: rank_binders applies the liability filter
    # itself, exactly as index_top_sequences hands it the raw frame.
    top = _rank_binders(archive, args.top, args.max_liability_violations)
    if args.top and len(top) != args.top:
        # Degrade instead of aborting: a thin archive must not kill the gate.
        # A partial/empty top index simply yields fewer duplicate matches (an
        # empty index passes everything as novel, matching the validator's own
        # behaviour when it has nothing to index).
        print(
            f"VALIDATOR_NOVELTY_DEGRADED: expected {args.top} top sequences, "
            f"got {len(top)}; indexing the available {len(top)}",
            flush=True,
        )

    # Exactly the call neurons/validator/nanobody_validity.py makes: it passes
    # coarse_min_shared=None, coarse_jaccard=None, which the SearchEngine
    # resolves to its configured coarse-filter defaults (min_shared_kmers /
    # jaccard_threshold). _search_one passes None/None too, so the candidate
    # set -- and therefore is_duplicate -- is byte-identical to the validator.
    # (A previous comment here claimed the validator "disables" the prefilter
    # and that explicit zeros were required; the checked validator source and
    # SearchEngine show the opposite: None = configured defaults.)
    #
    # PARALLELIZED (2026-10-10): each candidate's full alignment search
    # against the fixed top-50 index is independent of every other
    # candidate's -- nothing here is shared mutable state, so this is a
    # correctness-preserving reshape of the same serial loop, not a
    # different algorithm. The engine is rebuilt once per WORKER (not once
    # per candidate), amortizing its cost across however many candidates
    # that worker handles.
    if args.concurrency <= 1:
        _init_worker(top, historical, args.identity_threshold, args.no_cdr_gate, all_archive)
        results = [_search_one(candidate) for candidate in candidates]
    else:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.concurrency, len(candidates)),
            initializer=_init_worker,
            initargs=(top, historical, args.identity_threshold, args.no_cdr_gate, all_archive),
        ) as pool:
            results = list(pool.map(_search_one, candidates))

    passed = [item for item in results if item["validator_novel"]]
    payload = {
        "target": args.target,
        "archive": str(args.archive),
        "archive_count": len(archive),
        "algorithm2_top_count": len(top),
        "input_count": len(results),
        "pass_count": len(passed),
        "sequences": [item["sequence"] for item in passed],
    }
    if not args.summary_only:
        # `candidates` = the survivors, so a downstream race reads the filtered
        # set the same way pre_gpu_dedup emitted it. Full per-candidate detail
        # (including rejects + `validator_novel`) stays in `all_results`.
        payload["candidates"] = passed
        payload["all_results"] = results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.rejected_output is not None:
        args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
        args.rejected_output.write_text("".join(
            f"{item['sequence']}\n" for item in results if not item["validator_novel"]
        ))
    print(
        f"validator novelty: {len(passed)}/{len(results)} pass against "
        f"{len(archive):,} historical sequences and Algorithm-2 top {len(top)}"
    )
    for item in () if args.summary_only else results:
        label = item.get("mutation", item["sequence"][:12])
        print(
            f"  {'PASS' if item['validator_novel'] else 'FAIL'} {label}: "
            f"exact={item['exact_historical']} coarse={item['coarse_candidates']} "
            f"max_identity={item['maximum_identity']:.3f} "
            f"duplicates={len(item['duplicate_matches'])}"
        )


if __name__ == "__main__":
    main()
