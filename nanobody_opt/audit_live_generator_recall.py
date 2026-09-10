"""Leakage-free replay of the live 384-to-320 nanobody proposal generator.

For every held-out epoch this audit reconstructs the full prior public parent
bank, writes only pre-epoch archive/winner inputs, and invokes the production
``local_neighborhood`` module with live budgets. It measures organic winner and
top-5 recall before Layer-2. Parent Layer-2 is deliberately not retroactively
replayed; that assumption is explicit in the output rather than hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from nanobody_opt.select_elite_parents import select_elite_parents
from nanobody_opt.simulate_round import METRICS, dense_rank_sums


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


def _payload(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    nanobodies = []
    for row in frame.to_dict("records"):
        nanobodies.append({
            "protein_name": target,
            "sequence": row["sequence"],
            "final_score": row["final_score"],
            "confidence_rank": 0,
            "physical_interaction_rank": 0,
            "developability_rank": 0,
            **{name: row[name] for name in METRICS},
        })
    return {"success": True, "data": [{"nanobodies": nanobodies}]}


def _rank_field(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["actual_rank_sum"] = dense_rank_sums(ranked, METRICS)
    ranked["actual_field_rank"] = ranked["final_score"].rank(
        method="min", ascending=True
    ).astype(int)
    return ranked.sort_values(
        ["final_score", "sequence"], kind="stable"
    ).drop_duplicates("sequence", keep="first")


def _replay_epoch(
    *,
    epoch: int,
    fields: dict[int, pd.DataFrame],
    target: str,
    config: Path,
    output_dir: Path,
    budget: int,
) -> list[dict[str, Any]]:
    prior_epochs = [value for value in sorted(fields) if value < epoch]
    prior = pd.concat([fields[value] for value in prior_epochs], ignore_index=True)
    payloads = {
        value: _payload(fields[value], target) for value in prior_epochs
    }
    selected = select_elite_parents(
        payloads,
        target=target,
        per_epoch=2,
        limit=64,
        neighbor_per_epoch=8,
        neighbor_distance=6,
        neighbor_limit=32,
        terminal_variants_per_residue=3,
    )
    epoch_dir = output_dir / f"epoch{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    archive = epoch_dir / "prior_archive.csv"
    prior.to_csv(archive, index=False)
    parent_path = epoch_dir / "prior_parents_layer2_assumed.json"
    _atomic_json(parent_path, {
        "target": target,
        "assumption": (
            "public historical parents are treated as Layer-2 valid; "
            "candidate Layer-2 is not simulated"
        ),
        "candidates": [
            {**item, "selection_score": item["validator_final_score"],
             "layer2_passed": True}
            for item in selected
        ],
    })
    winners = []
    for value in prior_epochs:
        leader = _rank_field(fields[value]).iloc[0]
        winners.append({
            "epoch": value,
            "sequence": str(leader.sequence),
            "epoch_rank_sum": float(leader.final_score),
        })
    winner_path = epoch_dir / "prior_winners.json"
    _atomic_json(winner_path, {"winners": winners})
    candidates = epoch_dir / "candidates.csv"
    command = [
        sys.executable,
        "-m", "nanobody_opt.local_neighborhood",
        "--archive", str(archive),
        "--config", str(config),
        "--layer2-seeds", str(parent_path),
        "--winner-priors", str(winner_path),
        "--output", str(candidates),
        "--seed-count", "32",
        "--neighbor-seed-count", "32",
        "--minimum-seeds", "1",
        "--per-seed", "12",
        "--proposal-limit", "384",
        "--framework-fraction", "0.5",
        "--framework-per-seed", "64",
        "--terminal-width", "17",
        "--maximum-terminal-haplotype-mutations", "6",
        "--motif-fraction", "0.25",
        "--single-mutation-fraction", "0.75",
        "--maximum-motif-mutations", "6",
        "--maximum-exhaustive-singles", "100000",
        "--random-seed", str(epoch * 100 + 1),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    (epoch_dir / "generator.stdout").write_text(completed.stdout)
    (epoch_dir / "generator.stderr").write_text(completed.stderr)
    if completed.returncode:
        return [{
            "epoch": epoch,
            "generator_failed": True,
            "generator_returncode": completed.returncode,
            "leakage_check": max(prior_epochs) < epoch,
        }]
    generated = pd.read_csv(candidates)
    shortlisted = generated.head(budget)
    route_by_sequence = (
        generated.sort_values("screen_priority")
        .drop_duplicates("sequence", keep="first")
        .set_index("sequence")
        .to_dict("index")
    )
    shortlist_rank = {
        sequence: rank
        for rank, sequence in enumerate(shortlisted.sequence, start=1)
    }
    heldout = _rank_field(fields[epoch]).head(10)
    rows = []
    for item in heldout.itertuples(index=False):
        sequence = str(item.sequence)
        route = route_by_sequence.get(sequence, {})
        rows.append({
            "epoch": epoch,
            "actual_field_rank": int(item.actual_field_rank),
            "actual_final_score": float(item.final_score),
            "sequence": sequence,
            "parent_count": len(selected),
            "proposal_count": len(generated),
            "organically_generated": sequence in route_by_sequence,
            "in_top320": sequence in shortlist_rank,
            "screen_rank": shortlist_rank.get(sequence),
            "proposal_type": route.get("proposal_type"),
            "proposal_lane": route.get("proposal_lane"),
            "edit_distance": route.get("edit_distance"),
            "parent_sequence": route.get("parent_sequence"),
            "generator_failed": False,
            "leakage_check": max(prior_epochs) < epoch,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-epochs", type=int, default=20)
    parser.add_argument("--budget", type=int, default=320)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.holdout_epochs, args.budget, args.workers) < 1:
        parser.error("holdout, budget, and worker counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = pd.read_csv(args.api_snapshot)
    fields = {
        int(epoch): frame.reset_index(drop=True)
        for epoch, frame in snapshot.groupby("epoch", sort=True)
    }
    heldout = sorted(fields)[-args.holdout_epochs:]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _replay_epoch,
                epoch=epoch,
                fields=fields,
                target=args.target,
                config=args.config,
                output_dir=args.output_dir,
                budget=args.budget,
            ): epoch
            for epoch in heldout
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            epoch = futures[future]
            epoch_rows = future.result()
            rows.extend(epoch_rows)
            print(
                f"LIVE_GENERATOR_AUDIT_PROGRESS {completed}/{len(heldout)} "
                f"epoch={epoch}",
                flush=True,
            )
    placements = pd.DataFrame(rows).sort_values(
        ["epoch", "actual_field_rank"], kind="stable"
    )
    placements.to_csv(args.output_dir / "placements.csv", index=False)
    successful = placements[~placements.generator_failed]
    summary: dict[str, Any] = {
        "objective": "organic held-out inclusion by the production generator",
        "layer2_replay": False,
        "parent_layer2_assumption": "historical public parents treated as valid",
        "budget": args.budget,
        "holdout_epochs": len(heldout),
        "failed_epochs": sorted(
            placements.loc[placements.generator_failed, "epoch"].unique().tolist()
        ),
        "all_parent_epochs_precede_target": bool(placements.leakage_check.all()),
        "cohorts": {},
    }
    for cutoff in (1, 5, 10):
        cohort = successful[successful.actual_field_rank <= cutoff]
        summary["cohorts"][f"top{cutoff}"] = {
            "sequences": len(cohort),
            "organic_generation_recall": float(cohort.organically_generated.mean()),
            "top320_recall": float(cohort.in_top320.mean()),
            "included_epochs": sorted(
                cohort.loc[cohort.in_top320, "epoch"].unique().tolist()
            ),
        }
    _atomic_json(args.output_dir / "summary.json", summary)
    manifest = {
        "snapshot_sha256": hashlib.sha256(
            args.api_snapshot.read_bytes()
        ).hexdigest(),
        "files": sorted(
            str(path.relative_to(args.output_dir))
            for path in args.output_dir.rglob("*")
            if path.is_file()
        ),
    }
    _atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
