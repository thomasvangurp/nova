"""Leakage-free end-to-end test of the V2 pairwise acquisition model.

The public-field V2 headline ranks sequences that miners eventually submitted.
That measures ordering but not discovery.  This module instead freezes the
trained model, reconstructs parents and residue alphabets from earlier epochs
only, performs a model-guided position-wise beam search, and asks whether the
held-out winner entered the same 160/320-candidate acquisition budget.

The comparison route uses the earlier broad stochastic proposal generator and
a quality-parent round-robin allocation.  Held-out rows are never inserted into
either proposal pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from nanobody_opt.backtest_mutation_ranker import (
    Parent,
    Proposal,
    generate_proposals,
    mutation_description,
    rank_field,
    select_parents,
)
from nanobody_opt.train_pairwise_ranker_v2 import symmetric_win_probability


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epoch_fields(snapshot: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    fields = []
    for epoch, frame in snapshot.groupby("epoch", sort=True):
        fields.append((int(epoch), frame.reset_index(drop=True)))
    return fields


def _donors(
    preceding: list[tuple[int, pd.DataFrame]], *, length: int, epochs: int = 16
) -> list[str]:
    return list(dict.fromkeys(
        str(sequence)
        for _, frame in preceding[-epochs:]
        for sequence in frame.sequence
        if len(str(sequence)) == length
    ))


def _historical(preceding: list[tuple[int, pd.DataFrame]]) -> set[str]:
    return {
        str(sequence)
        for _, frame in preceding
        for sequence in frame.sequence
    }


def model_beam_proposals(
    model: Any,
    parents: list[Parent],
    donors: list[str],
    *,
    historical: set[str],
    length: int,
    beam_width: int,
    maximum_edits: int,
) -> list[tuple[Proposal, float]]:
    """Search supported residue recombinations without seeing the held-out field."""
    if beam_width < 1 or maximum_edits < 1:
        raise ValueError("beam width and maximum edits must be positive")
    compatible = [sequence for sequence in donors if len(sequence) == length]
    if not compatible:
        return []
    allowed = [
        sorted({sequence[position] for sequence in compatible})
        for position in range(length)
    ]
    frontier = {parent.sequence: [parent.sequence] for parent in parents}
    by_parent = {parent.sequence: parent for parent in parents}
    discovered: dict[tuple[str, str], tuple[Proposal, float]] = {}

    for _depth in range(1, maximum_edits + 1):
        expansions: list[tuple[Parent, str]] = []
        for parent_sequence, sequences in frontier.items():
            parent = by_parent[parent_sequence]
            seen: set[str] = set()
            for sequence in sequences:
                for position, residues in enumerate(allowed):
                    # Each depth changes one previously untouched position, so
                    # a sequence cannot cycle or silently exceed the edit cap.
                    if sequence[position] != parent.sequence[position]:
                        continue
                    for residue in residues:
                        if residue == parent.sequence[position]:
                            continue
                        candidate = (
                            sequence[:position] + residue + sequence[position + 1 :]
                        )
                        if candidate not in seen:
                            seen.add(candidate)
                            expansions.append((parent, candidate))
        if not expansions:
            break
        probabilities = symmetric_win_probability(
            model,
            [sequence for _, sequence in expansions],
            [parent.sequence for parent, _ in expansions],
            length=length,
        )
        ranked: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (parent, sequence), probability in zip(
            expansions, probabilities, strict=True
        ):
            ranked[parent.sequence].append((sequence, float(probability)))
        next_frontier: dict[str, list[str]] = {}
        for parent_sequence, rows in ranked.items():
            parent = by_parent[parent_sequence]
            rows.sort(key=lambda item: (-item[1], item[0]))
            chosen = rows[:beam_width]
            next_frontier[parent_sequence] = [sequence for sequence, _ in chosen]
            for sequence, probability in chosen:
                if sequence in historical:
                    continue
                edits, mutations = mutation_description(parent.sequence, sequence)
                proposal = Proposal(
                    sequence=sequence,
                    parent=parent,
                    source="pairwise_model_beam",
                    edit_distance=edits,
                    mutations=mutations,
                )
                discovered[(parent.sequence, sequence)] = (proposal, probability)
        frontier = next_frontier
    return list(discovered.values())


def model_allocation(
    proposals: list[tuple[Proposal, float]],
    *,
    budget: int,
    parent_floor: int,
    exploration: int,
    seed: int,
) -> list[Proposal]:
    """Allocate model, parent-coverage, and deterministic exploration lanes."""
    if parent_floor + exploration > budget:
        raise ValueError("parent floor plus exploration exceeds budget")
    ranked = sorted(proposals, key=lambda item: (-item[1], item[0].sequence))
    by_parent: dict[str, list[tuple[Proposal, float]]] = defaultdict(list)
    for proposal, probability in ranked:
        by_parent[proposal.parent.sequence].append((proposal, probability))
    parent_order = sorted(
        by_parent,
        key=lambda sequence: (
            by_parent[sequence][0][0].parent.final_score,
            sequence,
        ),
    )
    selected: list[Proposal] = []
    used: set[str] = set()
    for parent_sequence in parent_order[:parent_floor]:
        for proposal, _ in by_parent[parent_sequence]:
            if proposal.sequence not in used:
                selected.append(proposal)
                used.add(proposal.sequence)
                break
    exploit_limit = max(0, budget - exploration)
    for proposal, _ in ranked:
        if len(selected) >= exploit_limit:
            break
        if proposal.sequence not in used:
            selected.append(proposal)
            used.add(proposal.sequence)
    remaining = [item[0] for item in ranked if item[0].sequence not in used]
    remaining.sort(key=lambda proposal: (
        hashlib.sha256(f"{seed}:{proposal.sequence}".encode()).hexdigest(),
        proposal.sequence,
    ))
    for proposal in remaining:
        if len(selected) >= budget:
            break
        selected.append(proposal)
        used.add(proposal.sequence)
    return selected


def quality_round_robin(
    proposals: Iterable[Proposal], *, budget: int
) -> list[Proposal]:
    """Prior route: strong parents first, balanced without learned ordering."""
    grouped: dict[str, list[Proposal]] = defaultdict(list)
    for proposal in proposals:
        grouped[proposal.parent.sequence].append(proposal)
    source_priority = {
        "exhaustive_single": 0,
        "coherent_donor_shell": 1,
        "n_terminal_shell": 2,
        "c_terminal_shell": 2,
        "position_recombination": 3,
    }
    for rows in grouped.values():
        rows.sort(key=lambda proposal: (
            source_priority.get(proposal.source, 9),
            proposal.edit_distance,
            proposal.sequence,
        ))
    parent_order = sorted(
        grouped,
        key=lambda sequence: (
            grouped[sequence][0].parent.final_score,
            sequence,
        ),
    )
    selected: list[Proposal] = []
    used: set[str] = set()
    maximum = max((len(rows) for rows in grouped.values()), default=0)
    for index in range(maximum):
        for parent_sequence in parent_order:
            rows = grouped[parent_sequence]
            if index >= len(rows):
                continue
            proposal = rows[index]
            if proposal.sequence in used:
                continue
            selected.append(proposal)
            used.add(proposal.sequence)
            if len(selected) >= budget:
                return selected
    return selected


def _placement(
    selected: list[Proposal], field: pd.DataFrame, winner: str
) -> dict[str, Any]:
    selected_sequences = {proposal.sequence for proposal in selected}
    ranked = rank_field(field).drop_duplicates("sequence", keep="first")
    matched = ranked[ranked.sequence.isin(selected_sequences)]
    return {
        "selected": len(selected_sequences),
        "winner_selected": winner in selected_sequences,
        "field_matches": len(matched),
        "best_matched_field_rank": (
            None if matched.empty else int(matched.actual_field_rank.min())
        ),
        "best_matched_final_score": (
            None if matched.empty else float(matched.final_score.min())
        ),
    }


def backtest(
    model: Any,
    snapshot: pd.DataFrame,
    *,
    holdout_epochs: int,
    length: int,
    beam_width: int,
    maximum_edits: int,
    random_candidates_per_parent: int,
    budgets: tuple[int, ...],
    parent_floor: int,
    exploration: int,
    seed: int,
    target_epochs: set[int] | None = None,
) -> pd.DataFrame:
    fields = _epoch_fields(snapshot)
    holdout = {epoch for epoch, _ in fields[-holdout_epochs:]}
    if target_epochs is not None:
        unavailable = target_epochs - holdout
        if unavailable:
            raise ValueError(
                "target epochs must be inside the sealed holdout: "
                + ",".join(str(epoch) for epoch in sorted(unavailable))
            )
        holdout &= target_epochs
    rows: list[dict[str, Any]] = []
    for index, (epoch, field) in enumerate(fields):
        if epoch not in holdout:
            continue
        preceding = fields[:index]
        parents = select_parents(preceding)
        donors = _donors(preceding, length=length)
        historical = _historical(preceding)
        winner = str(rank_field(field).iloc[0].sequence)
        baseline_pool = generate_proposals(
            parents,
            donors,
            historical=historical,
            length=length,
            random_candidates_per_parent=random_candidates_per_parent,
            seed=seed + epoch,
        )
        beam_pool = model_beam_proposals(
            model,
            parents,
            donors,
            historical=historical,
            length=length,
            beam_width=beam_width,
            maximum_edits=maximum_edits,
        )
        baseline_sequences = {proposal.sequence for proposal in baseline_pool}
        beam_sequences = {proposal.sequence for proposal, _ in beam_pool}
        for budget in budgets:
            baseline = quality_round_robin(baseline_pool, budget=budget)
            model_selected = model_allocation(
                beam_pool,
                budget=budget,
                parent_floor=min(parent_floor, len(parents)),
                exploration=min(exploration, max(0, budget - parent_floor)),
                seed=seed + epoch,
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "budget": budget,
                "winner_sequence": winner,
                "winner_score": float(rank_field(field).iloc[0].final_score),
                "maximum_parent_epoch": max(parent.epoch for parent in parents),
                "leakage_check": max(parent.epoch for parent in parents) < epoch,
                "baseline_pool_size": len(baseline_sequences),
                "model_beam_pool_size": len(beam_sequences),
                "baseline_winner_generated": winner in baseline_sequences,
                "model_winner_generated": winner in beam_sequences,
            }
            row.update({
                f"baseline_{key}": value
                for key, value in _placement(baseline, field, winner).items()
            })
            row.update({
                f"model_{key}": value
                for key, value in _placement(model_selected, field, winner).items()
            })
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    by_budget: dict[str, Any] = {}
    for budget, group in frame.groupby("budget", sort=True):
        comparable = group.dropna(
            subset=["baseline_best_matched_field_rank", "model_best_matched_field_rank"]
        )
        by_budget[str(int(budget))] = {
            "epochs": len(group),
            "all_parent_epochs_precede_target": bool(group.leakage_check.all()),
            "baseline_winner_generation_rate": float(
                group.baseline_winner_generated.mean()
            ),
            "model_winner_generation_rate": float(group.model_winner_generated.mean()),
            "baseline_winner_selection_rate": float(
                group.baseline_winner_selected.mean()
            ),
            "model_winner_selection_rate": float(group.model_winner_selected.mean()),
            "baseline_total_field_matches": int(group.baseline_field_matches.sum()),
            "model_total_field_matches": int(group.model_field_matches.sum()),
            "paired_epochs_with_field_matches": len(comparable),
            "model_better_paired_epochs": int((
                comparable.model_best_matched_field_rank
                < comparable.baseline_best_matched_field_rank
            ).sum()),
            "baseline_better_paired_epochs": int((
                comparable.baseline_best_matched_field_rank
                < comparable.model_best_matched_field_rank
            ).sum()),
            "ties_paired_epochs": int((
                comparable.baseline_best_matched_field_rank
                == comparable.model_best_matched_field_rank
            ).sum()),
            "baseline_epochs_with_top1_match": int((
                group.baseline_best_matched_field_rank <= 1
            ).sum()),
            "model_epochs_with_top1_match": int((
                group.model_best_matched_field_rank <= 1
            ).sum()),
            "baseline_epochs_with_top5_match": int((
                group.baseline_best_matched_field_rank <= 5
            ).sum()),
            "model_epochs_with_top5_match": int((
                group.model_best_matched_field_rank <= 5
            ).sum()),
            "baseline_epochs_with_top10_match": int((
                group.baseline_best_matched_field_rank <= 10
            ).sum()),
            "model_epochs_with_top10_match": int((
                group.model_best_matched_field_rank <= 10
            ).sum()),
        }
    return {"budgets": by_budget}


def main() -> None:
    import joblib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--maximum-edits", type=int, default=6)
    parser.add_argument("--random-candidates-per-parent", type=int, default=500)
    parser.add_argument("--budgets", default="160,320")
    parser.add_argument("--parent-floor", type=int, default=32)
    parser.add_argument("--exploration", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=20260910)
    parser.add_argument(
        "--target-epochs",
        help=(
            "optional comma-separated subset of the sealed holdout; used by "
            "the resumable shard runner"
        ),
    )
    args = parser.parse_args()
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",")}))
    if not budgets or min(budgets) < 1:
        parser.error("budgets must be positive")
    target_epochs = (
        None
        if not args.target_epochs
        else {int(value) for value in args.target_epochs.split(",")}
    )
    snapshot = pd.read_csv(args.api_snapshot)
    snapshot.sequence = snapshot.sequence.astype(str).str.strip().str.upper()
    snapshot = snapshot[
        snapshot.sequence.str.len() == args.sequence_length
    ].reset_index(drop=True)
    model = joblib.load(args.model)
    placements = backtest(
        model,
        snapshot,
        holdout_epochs=args.holdout_epochs,
        length=args.sequence_length,
        beam_width=args.beam_width,
        maximum_edits=args.maximum_edits,
        random_candidates_per_parent=args.random_candidates_per_parent,
        budgets=budgets,
        parent_floor=args.parent_floor,
        exploration=args.exploration,
        seed=args.random_seed,
        target_epochs=target_epochs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    placements.to_csv(args.output_dir / "placements.csv", index=False)
    summary = summarize(placements)
    summary["config"] = {
        "holdout_epochs": args.holdout_epochs,
        "sequence_length": args.sequence_length,
        "beam_width": args.beam_width,
        "maximum_edits": args.maximum_edits,
        "random_candidates_per_parent": args.random_candidates_per_parent,
        "budgets": budgets,
        "parent_floor": args.parent_floor,
        "exploration": args.exploration,
        "random_seed": args.random_seed,
        "target_epochs": (
            None if target_epochs is None else sorted(target_epochs)
        ),
        "heldout_sequences_inserted_into_proposals": False,
        "api_snapshot_sha256": _sha256(args.api_snapshot),
        "model_sha256": _sha256(args.model),
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    _atomic_json(args.output_dir / "manifest.json", {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
