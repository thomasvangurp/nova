"""Audit generator reachability and 320-slot allocation on sealed epochs.

This is the fast, decomposed counterpart to the full model-guided beam replay.
For each held-out top-10 sequence it first asks whether the prior-only proposal
grammar can express the sequence.  It then injects that sequence into the
frozen organic pool and measures whether the heuristic and pairwise-model
allocators would retain it at the same budget.  Held-out sequences never alter
parents, donor alphabets, the organic pool, or fitted model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nanobody_opt.backtest_mutation_ranker import (
    Parent,
    Proposal,
    generate_proposals,
    mutation_description,
    nearest_parent,
    rank_field,
    select_parents,
)
from nanobody_opt.backtest_pairwise_beam import (
    model_allocation,
    quality_round_robin,
)
from nanobody_opt.train_pairwise_ranker_v2 import symmetric_win_probability


@dataclass(frozen=True)
class Reachability:
    reachable: bool
    historically_excluded: bool
    minimum_edits: int | None
    sources: tuple[str, ...]
    proposals: tuple[Proposal, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
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


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def supported_routes(
    sequence: str,
    parents: list[Parent],
    donors: list[str],
    *,
    historical: set[str],
    maximum_edits: int = 6,
) -> Reachability:
    """Return every prior-supported route in the broad proposal grammar."""
    if sequence in historical:
        return Reachability(False, True, None, (), ())
    length = len(sequence)
    compatible_donors = [item for item in donors if len(item) == length]
    compatible_parents = [item for item in parents if len(item.sequence) == length]
    if not compatible_donors or not compatible_parents:
        return Reachability(False, False, None, (), ())
    allowed = [
        {donor[position] for donor in compatible_donors}
        for position in range(length)
    ]
    routes: dict[tuple[str, str], Proposal] = {}
    minimum_edits: int | None = None

    def add(parent: Parent, source: str, changes: list[int]) -> None:
        nonlocal minimum_edits
        edits, mutations = mutation_description(parent.sequence, sequence)
        routes.setdefault(
            (parent.sequence, source),
            Proposal(sequence, parent, source, edits, mutations),
        )
        minimum_edits = edits if minimum_edits is None else min(minimum_edits, edits)

    for parent in compatible_parents:
        changes = [
            position
            for position, (before, after) in enumerate(
                zip(parent.sequence, sequence, strict=True)
            )
            if before != after
        ]
        if not 1 <= len(changes) <= maximum_edits:
            continue
        if not all(sequence[position] in allowed[position] for position in changes):
            continue
        if len(changes) == 1:
            add(parent, "exhaustive_single", changes)
        if 2 <= len(changes) and sequence in compatible_donors:
            add(parent, "coherent_donor_shell", changes)
        for source, positions in (
            ("n_terminal_shell", set(range(0, min(10, length)))),
            ("c_terminal_shell", set(range(max(0, length - 17), length))),
        ):
            if not set(changes).issubset(positions):
                continue
            if any(
                all(donor[position] == sequence[position] for position in changes)
                for donor in compatible_donors
            ):
                add(parent, source, changes)
        if len(changes) >= 2:
            add(parent, "position_recombination", changes)
    proposals = tuple(routes.values())
    return Reachability(
        bool(proposals),
        False,
        minimum_edits,
        tuple(sorted({proposal.source for proposal in proposals})),
        proposals,
    )


def _score(
    model: Any, proposals: list[Proposal], *, length: int
) -> list[tuple[Proposal, float]]:
    if not proposals:
        return []
    probabilities = symmetric_win_probability(
        model,
        [item.sequence for item in proposals],
        [item.parent.sequence for item in proposals],
        length=length,
    )
    return [
        (proposal, float(probability))
        for proposal, probability in zip(proposals, probabilities, strict=True)
    ]


def _model_rank(
    scored: list[tuple[Proposal, float]], sequence: str
) -> int | None:
    best: dict[str, float] = {}
    for proposal, probability in scored:
        best[proposal.sequence] = max(
            probability, best.get(proposal.sequence, -math.inf)
        )
    if sequence not in best:
        return None
    return 1 + sum(value > best[sequence] for value in best.values())


def _unique_pairs(proposals: list[Proposal]) -> list[Proposal]:
    return list({
        (proposal.parent.sequence, proposal.sequence): proposal
        for proposal in proposals
    }.values())


def _diagnosis(reachable: bool, generated: bool, selected: bool) -> str:
    if not reachable:
        return "operator_unreachable"
    if not generated:
        return "generation_miss"
    if not selected:
        return "allocation_miss"
    return "included"


def audit(
    model: Any,
    snapshot: pd.DataFrame,
    *,
    holdout_epochs: int,
    length: int,
    maximum_edits: int,
    random_candidates_per_parent: int,
    budget: int,
    parent_floor: int,
    exploration: int,
    seed: int,
    progress_path: Path | None = None,
) -> pd.DataFrame:
    fields = [
        (int(epoch), frame.reset_index(drop=True))
        for epoch, frame in snapshot.groupby("epoch", sort=True)
    ]
    start = max(0, len(fields) - holdout_epochs)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for completed, index in enumerate(range(start, len(fields)), start=1):
        epoch, raw_field = fields[index]
        preceding = fields[:index]
        parents = select_parents(preceding)
        donors = list(dict.fromkeys(
            str(sequence)
            for _, frame in preceding[-16:]
            for sequence in frame.sequence
            if len(str(sequence)) == length
        ))
        historical = {
            str(sequence)
            for _, frame in preceding
            for sequence in frame.sequence
        }
        organic = _unique_pairs(generate_proposals(
            parents,
            donors,
            historical=historical,
            length=length,
            random_candidates_per_parent=random_candidates_per_parent,
            seed=seed + epoch,
        ))
        organic_sequences = {proposal.sequence for proposal in organic}
        organic_pairs = {
            (proposal.parent.sequence, proposal.sequence)
            for proposal in organic
        }
        if len(organic_sequences) < budget:
            raise ValueError(
                f"epoch {epoch} has only {len(organic_sequences)} proposals for "
                f"budget {budget}"
            )
        organic_scored = _score(model, organic, length=length)
        heuristic_selected = {
            proposal.sequence
            for proposal in quality_round_robin(organic, budget=budget)
        }
        model_selected = {
            proposal.sequence
            for proposal in model_allocation(
                organic_scored,
                budget=budget,
                parent_floor=min(parent_floor, len(parents)),
                exploration=min(exploration, max(0, budget - parent_floor)),
                seed=seed + epoch,
            )
        }
        field = rank_field(
            raw_field[raw_field.sequence.str.len() == length].reset_index(drop=True)
        ).drop_duplicates("sequence", keep="first").head(10)
        for _, item in field.iterrows():
            sequence = str(item.sequence)
            routes = supported_routes(
                sequence,
                parents,
                donors,
                historical=historical,
                maximum_edits=maximum_edits,
            )
            generated = sequence in organic_sequences
            injection = list(routes.proposals)
            if not injection:
                parent = nearest_parent(sequence, parents)
                edits, mutations = mutation_description(parent.sequence, sequence)
                injection = [Proposal(
                    sequence,
                    parent,
                    "heldout_oracle_only",
                    edits,
                    mutations,
                )]
            injection = [
                proposal
                for proposal in injection
                if (proposal.parent.sequence, proposal.sequence)
                not in organic_pairs
            ]
            injected_organic = _unique_pairs([*organic, *injection])
            injection_scored = _score(model, injection, length=length)
            injected_scored = [*organic_scored, *injection_scored]
            heuristic_counterfactual = {
                proposal.sequence
                for proposal in quality_round_robin(
                    injected_organic, budget=budget
                )
            }
            model_counterfactual = {
                proposal.sequence
                for proposal in model_allocation(
                    injected_scored,
                    budget=budget,
                    parent_floor=min(parent_floor, len(parents)),
                    exploration=min(
                        exploration, max(0, budget - parent_floor)
                    ),
                    seed=seed + epoch,
                )
            }
            reachable = routes.reachable
            rows.append({
                "epoch": epoch,
                "actual_field_rank": int(item.actual_field_rank),
                "actual_final_score": float(item.final_score),
                "sequence": sequence,
                "parent_count": len(parents),
                "organic_pool_size": len(organic_sequences),
                "operator_reachable": reachable,
                "historically_excluded": routes.historically_excluded,
                "minimum_supported_edits": routes.minimum_edits,
                "supported_sources": ";".join(routes.sources),
                "organically_generated": generated,
                "heuristic_organic_in320": sequence in heuristic_selected,
                "model_organic_in320": sequence in model_selected,
                "heuristic_injected_in320": (
                    sequence in heuristic_counterfactual if reachable else None
                ),
                "model_injected_in320": (
                    sequence in model_counterfactual if reachable else None
                ),
                "model_oracle_rank": _model_rank(injected_scored, sequence),
                "heuristic_diagnosis": _diagnosis(
                    reachable, generated, sequence in heuristic_selected
                ),
                "model_diagnosis": _diagnosis(
                    reachable, generated, sequence in model_selected
                ),
                "maximum_parent_epoch": max(parent.epoch for parent in parents),
                "leakage_check": max(parent.epoch for parent in parents) < epoch,
            })
        partial = pd.DataFrame(rows)
        if progress_path is not None:
            _atomic_csv(progress_path, partial)
        elapsed = time.monotonic() - started
        remaining = len(fields) - start - completed
        eta = elapsed / completed * remaining
        print(
            f"SHORTLIST_AUDIT_PROGRESS epochs={completed}/{len(fields) - start} "
            f"epoch={epoch} elapsed_minutes={elapsed / 60:.1f} "
            f"eta_minutes={eta / 60:.1f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, budget: int) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for cutoff in (1, 5, 10):
        group = frame[frame.actual_field_rank <= cutoff]
        reachable = group[group.operator_reachable]
        cohorts[f"top{cutoff}"] = {
            "epochs": int(group.epoch.nunique()),
            "sequences": len(group),
            "operator_reachability": float(group.operator_reachable.mean()),
            "organic_generation_recall": float(group.organically_generated.mean()),
            "heuristic_organic_shortlist_recall": float(
                group.heuristic_organic_in320.mean()
            ),
            "model_organic_shortlist_recall": float(
                group.model_organic_in320.mean()
            ),
            "heuristic_conditional_injected_recall": (
                None
                if reachable.empty
                else float(reachable.heuristic_injected_in320.mean())
            ),
            "model_conditional_injected_recall": (
                None
                if reachable.empty
                else float(reachable.model_injected_in320.mean())
            ),
            "median_model_oracle_rank": float(group.model_oracle_rank.median()),
            "heuristic_failure_counts": {
                str(key): int(value)
                for key, value in group.heuristic_diagnosis.value_counts().items()
            },
            "model_failure_counts": {
                str(key): int(value)
                for key, value in group.model_diagnosis.value_counts().items()
            },
        }
    return {
        "objective": (
            "held-out winner and high-scorer inclusion in equal-budget "
            "proposal shortlists"
        ),
        "budget": budget,
        "all_parent_epochs_precede_target": bool(frame.leakage_check.all()),
        "cohorts": cohorts,
    }


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--maximum-edits", type=int, default=6)
    parser.add_argument("--random-candidates-per-parent", type=int, default=500)
    parser.add_argument("--budget", type=int, default=320)
    parser.add_argument("--parent-floor", type=int, default=32)
    parser.add_argument("--exploration", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260910)
    args = parser.parse_args()
    if min(
        args.holdout_epochs,
        args.sequence_length,
        args.maximum_edits,
        args.random_candidates_per_parent,
        args.budget,
        args.parent_floor,
    ) < 1:
        parser.error("counts and budget must be positive")
    if not 0 <= args.exploration <= args.budget:
        parser.error("exploration must be between zero and budget")

    snapshot = pd.read_csv(args.api_snapshot)
    snapshot.sequence = snapshot.sequence.astype(str).str.strip().str.upper()
    snapshot = snapshot[
        snapshot.sequence.str.len() == args.sequence_length
    ].reset_index(drop=True)
    model = joblib.load(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    placements_path = args.output_dir / "placements.csv"
    placements = audit(
        model,
        snapshot,
        holdout_epochs=args.holdout_epochs,
        length=args.sequence_length,
        maximum_edits=args.maximum_edits,
        random_candidates_per_parent=args.random_candidates_per_parent,
        budget=args.budget,
        parent_floor=args.parent_floor,
        exploration=args.exploration,
        seed=args.random_seed,
        progress_path=placements_path,
    )
    _atomic_csv(placements_path, placements)
    summary = summarize(placements, args.budget)
    summary["config"] = {
        "api_snapshot": str(args.api_snapshot),
        "api_snapshot_sha256": _sha256(args.api_snapshot),
        "model": str(args.model),
        "model_sha256": _sha256(args.model),
        "holdout_epochs": args.holdout_epochs,
        "sequence_length": args.sequence_length,
        "maximum_edits": args.maximum_edits,
        "random_candidates_per_parent": args.random_candidates_per_parent,
        "budget": args.budget,
        "parent_floor": args.parent_floor,
        "exploration": args.exploration,
        "random_seed": args.random_seed,
        "heldout_sequences_used_for_training_or_generation": False,
    }
    summary_path = args.output_dir / "summary.json"
    _atomic_json(summary_path, summary)
    _atomic_json(args.output_dir / "manifest.json", {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
