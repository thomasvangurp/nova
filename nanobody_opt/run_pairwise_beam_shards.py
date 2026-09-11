"""Run the leakage-free pairwise beam replay as resumable epoch shards.

Each child receives the complete frozen API snapshot and evaluates exactly one
epoch from its sealed tail. The underlying backtest therefore sees all prior
epochs but no future field data. Completed shards are reused only after their
input hashes, epoch, budgets, row count, and leakage guard all validate.
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

from nanobody_opt.backtest_pairwise_beam import summarize


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


def heldout_epochs(
    snapshot_path: Path, *, holdout_epochs: int, length: int
) -> list[int]:
    frame = pd.read_csv(snapshot_path, usecols=["epoch", "sequence"])
    frame.sequence = frame.sequence.astype(str).str.strip().str.upper()
    available = sorted(
        int(epoch)
        for epoch in frame.loc[
            frame.sequence.str.len() == length, "epoch"
        ].unique()
    )
    if len(available) < holdout_epochs:
        raise ValueError(
            f"snapshot has {len(available)} usable epochs, fewer than "
            f"holdout {holdout_epochs}"
        )
    return available[-holdout_epochs:]


def shard_is_complete(
    shard_dir: Path,
    *,
    epoch: int,
    budgets: tuple[int, ...],
    snapshot_sha256: str,
    model_sha256: str,
) -> bool:
    try:
        summary = json.loads((shard_dir / "summary.json").read_text())
        placements = pd.read_csv(shard_dir / "placements.csv")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False
    config = summary.get("config", {})
    if config.get("api_snapshot_sha256") != snapshot_sha256:
        return False
    if config.get("model_sha256") != model_sha256:
        return False
    if config.get("target_epochs") != [epoch]:
        return False
    if set(placements.epoch.astype(int)) != {epoch}:
        return False
    if tuple(sorted(placements.budget.astype(int).unique())) != budgets:
        return False
    return len(placements) == len(budgets) and bool(
        placements.leakage_check.all()
    )


def _run_shard(
    *,
    epoch: int,
    snapshot: Path,
    model: Path,
    output_dir: Path,
    holdout_epochs: int,
    length: int,
    beam_width: int,
    maximum_edits: int,
    random_candidates_per_parent: int,
    budgets: tuple[int, ...],
    parent_floor: int,
    exploration: int,
    seed: int,
    nice: int,
) -> None:
    shard_dir = output_dir / "shards" / f"epoch{epoch}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "nanobody_opt.backtest_pairwise_beam",
        "--api-snapshot",
        str(snapshot),
        "--model",
        str(model),
        "--output-dir",
        str(shard_dir),
        "--holdout-epochs",
        str(holdout_epochs),
        "--target-epochs",
        str(epoch),
        "--sequence-length",
        str(length),
        "--beam-width",
        str(beam_width),
        "--maximum-edits",
        str(maximum_edits),
        "--random-candidates-per-parent",
        str(random_candidates_per_parent),
        "--budgets",
        ",".join(str(value) for value in budgets),
        "--parent-floor",
        str(parent_floor),
        "--exploration",
        str(exploration),
        "--random-seed",
        str(seed),
    ]
    if nice:
        command = ["nice", "-n", str(nice), *command]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    (shard_dir / "stdout.log").write_text(completed.stdout)
    (shard_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"epoch {epoch} shard failed with status {completed.returncode}; "
            f"see {shard_dir / 'stderr.log'}"
        )


def _manifest(output_dir: Path, epochs: list[int]) -> dict[str, Any]:
    paths = [output_dir / "placements.csv", output_dir / "summary.json"]
    for epoch in epochs:
        shard = output_dir / "shards" / f"epoch{epoch}"
        paths.extend([shard / "placements.csv", shard / "summary.json"])
    return {
        str(path.relative_to(output_dir)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-snapshot", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--holdout-epochs", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=127)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--maximum-edits", type=int, default=6)
    parser.add_argument("--random-candidates-per-parent", type=int, default=500)
    parser.add_argument("--budgets", default="160,320")
    parser.add_argument("--parent-floor", type=int, default=32)
    parser.add_argument("--exploration", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=20260910)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--nice",
        type=int,
        default=15,
        help="niceness increment for child shards; 0 disables adjustment",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute otherwise valid completed shards",
    )
    args = parser.parse_args()
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",")}))
    if not budgets or min(budgets) < 1:
        parser.error("budgets must be positive")
    if min(
        args.holdout_epochs,
        args.sequence_length,
        args.beam_width,
        args.maximum_edits,
        args.random_candidates_per_parent,
        args.parent_floor,
        args.workers,
    ) < 1:
        parser.error("counts, widths, and budgets must be positive")
    if not 0 <= args.nice <= 19:
        parser.error("nice must be between 0 and 19")
    if not 0 <= args.exploration <= min(budgets):
        parser.error("exploration must be between zero and the smallest budget")

    snapshot = args.api_snapshot.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    epochs = heldout_epochs(
        snapshot,
        holdout_epochs=args.holdout_epochs,
        length=args.sequence_length,
    )
    snapshot_sha256 = _sha256(snapshot)
    model_sha256 = _sha256(model)
    output_dir.mkdir(parents=True, exist_ok=True)

    pending: list[int] = []
    reused: list[int] = []
    for epoch in epochs:
        complete = shard_is_complete(
            output_dir / "shards" / f"epoch{epoch}",
            epoch=epoch,
            budgets=budgets,
            snapshot_sha256=snapshot_sha256,
            model_sha256=model_sha256,
        )
        if complete and not args.no_resume:
            reused.append(epoch)
        else:
            pending.append(epoch)

    common = {
        "snapshot": snapshot,
        "model": model,
        "output_dir": output_dir,
        "holdout_epochs": args.holdout_epochs,
        "length": args.sequence_length,
        "beam_width": args.beam_width,
        "maximum_edits": args.maximum_edits,
        "random_candidates_per_parent": args.random_candidates_per_parent,
        "budgets": budgets,
        "parent_floor": args.parent_floor,
        "exploration": args.exploration,
        "seed": args.random_seed,
        "nice": args.nice,
    }
    finished = set(reused)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_shard, epoch=epoch, **common): epoch
            for epoch in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            epoch = futures[future]
            future.result()
            if not shard_is_complete(
                output_dir / "shards" / f"epoch{epoch}",
                epoch=epoch,
                budgets=budgets,
                snapshot_sha256=snapshot_sha256,
                model_sha256=model_sha256,
            ):
                raise RuntimeError(f"epoch {epoch} shard failed validation")
            finished.add(epoch)
            partial = pd.concat(
                [
                    pd.read_csv(
                        output_dir
                        / "shards"
                        / f"epoch{value}"
                        / "placements.csv"
                    )
                    for value in sorted(finished)
                ],
                ignore_index=True,
            ).sort_values(["epoch", "budget"], kind="stable")
            _atomic_csv(output_dir / "progress.csv", partial)
            print(
                f"PAIRWISE_BEAM_SHARD_PROGRESS {completed}/{len(pending)} "
                f"epoch={epoch} reused={len(reused)}",
                flush=True,
            )

    placements = pd.concat(
        [
            pd.read_csv(
                output_dir / "shards" / f"epoch{epoch}" / "placements.csv"
            )
            for epoch in epochs
        ],
        ignore_index=True,
    ).sort_values(["epoch", "budget"], kind="stable")
    _atomic_csv(output_dir / "placements.csv", placements)
    summary = summarize(placements)
    summary["config"] = {
        "api_snapshot": str(snapshot),
        "api_snapshot_sha256": snapshot_sha256,
        "model": str(model),
        "model_sha256": model_sha256,
        "holdout_epochs": args.holdout_epochs,
        "heldout_epoch_range": [epochs[0], epochs[-1]],
        "sequence_length": args.sequence_length,
        "beam_width": args.beam_width,
        "maximum_edits": args.maximum_edits,
        "random_candidates_per_parent": args.random_candidates_per_parent,
        "budgets": budgets,
        "parent_floor": args.parent_floor,
        "exploration": args.exploration,
        "random_seed": args.random_seed,
        "workers": args.workers,
        "child_nice_increment": args.nice,
        "heldout_sequences_inserted_into_proposals": False,
        "one_sealed_epoch_per_process": True,
        "reused_epochs": reused,
        "executed_epochs": pending,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "manifest.json", {
        "files": _manifest(output_dir, epochs),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
