#!/usr/bin/env python3
"""Build a fresh, challenge-bound compound portfolio for one live epoch."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compound_miner.archive import CandidateArchive
from compound_miner.database import ReactionDatabase
from compound_miner.history import aggregate_observations, sync_history, sync_novelty
from compound_miner.proxy import TargetProxy
from compound_miner.search import SearchEngine
from compound_miner.validity import HistoricalIndex


_ACTIVE_PROCESS: subprocess.Popen | None = None


def _terminate_active_process(signum: int, _frame: object) -> None:
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")))
    os.replace(temporary, path)


def _write_dynamic_baselines(frame: pd.DataFrame, target: str, path: Path) -> None:
    """Record public bests for strategy evaluation, not submission blocking."""
    reactions: dict[str, dict[str, int | float | None]] = {}
    for reaction in range(1, 6):
        rows = frame[frame["reaction"] == reaction]
        if rows.empty:
            reactions[str(reaction)] = {"epoch": 0, "uid": None, "score": 0.0}
            continue
        submissions = rows[["epoch", "uid", "submission_score"]].drop_duplicates()
        best = submissions.sort_values(
            ["submission_score", "epoch"], ascending=[False, False]
        ).iloc[0]
        reactions[str(reaction)] = {
            "epoch": int(best["epoch"]),
            "uid": int(best["uid"]),
            "score": float(best["submission_score"]),
        }
    _atomic_json(
        path,
        {"schema_version": 1, "target": target, "reactions": reactions},
    )


def _write_release_baselines(target: str, path: Path) -> None:
    """Allow the first valid exact set; later writes are naturally monotonic."""
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "target": target,
            "reactions": {
                str(reaction): {"epoch": 0, "uid": None, "score": 0.0}
                for reaction in range(1, 6)
            },
        },
    )


def _run_until(arguments: list[str], deadline: float) -> None:
    global _ACTIVE_PROCESS
    remaining = deadline - time.monotonic()
    if remaining <= 60:
        raise TimeoutError("not enough epoch time remains to start exact qualification")
    process = subprocess.Popen(arguments, cwd=ROOT, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        process.wait(timeout=remaining - 30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        raise TimeoutError("epoch search exceeded its submission deadline")
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, arguments)
    _ACTIVE_PROCESS = None


def main() -> None:
    signal.signal(signal.SIGTERM, _terminate_active_process)
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--reaction", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("artifacts/epochs"))
    parser.add_argument("--history-root", type=Path, default=Path("history"))
    parser.add_argument("--proposals", type=int, default=250_000)
    parser.add_argument("--similarity-shortlist", type=int, default=50_000)
    parser.add_argument("--construction-workers", type=int, default=40)
    parser.add_argument("--similarity-workers", type=int, default=32)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=3)
    parser.add_argument("--preprocessing-threads", type=int, default=1)
    parser.add_argument("--reduced-limit", type=int, default=1_024)
    parser.add_argument("--exact-limit", type=int, default=128)
    parser.add_argument("--deadline-seconds", type=int, default=3_900)
    parser.add_argument("--scratch-root", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--skip-history-refresh", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    deadline = started + args.deadline_seconds
    work = args.work_root / args.target / f"epoch{args.epoch}" / f"rxn{args.reaction}"
    work.mkdir(parents=True, exist_ok=True)
    history_path = args.history_root / f"{args.target}_scores.csv"
    novelty_path = args.history_root / f"{args.target}_molecules.csv"
    if not args.skip_history_refresh:
        try:
            history = sync_history(history_path, target=args.target, workers=12)
        except Exception:
            if not history_path.is_file():
                raise
            history = pd.read_csv(history_path)
        try:
            sync_novelty(novelty_path, target=args.target)
        except Exception:
            if not novelty_path.is_file():
                raise
    else:
        history = pd.read_csv(history_path)

    _write_dynamic_baselines(history, args.target, work / "public_benchmarks.json")
    baselines = work / "release_baselines.json"
    _write_release_baselines(args.target, baselines)
    model_path = work / "proxy.joblib"
    TargetProxy(trees=args.trees).fit(aggregate_observations(history)).save(model_path)

    archive_path = work / "candidates.sqlite"
    engine = SearchEngine(
        ReactionDatabase(),
        history,
        TargetProxy.load(model_path),
        CandidateArchive(archive_path),
        HistoricalIndex(novelty_path),
    )
    search_metrics = engine.run_round(
        reaction=args.reaction,
        proposals=args.proposals,
        exact_shortlist=args.exact_limit,
        similarity_shortlist=args.similarity_shortlist,
        result_path=None,
        construction_workers=args.construction_workers,
        similarity_workers=args.similarity_workers,
    )
    _atomic_json(work / "search_metrics.json", search_metrics)

    qualification = [
        sys.executable,
        str(ROOT / "tools" / "build_exact_checkpoint.py"),
        "--target",
        args.target,
        "--epoch",
        str(args.epoch),
        "--reaction",
        str(args.reaction),
        "--archive",
        str(archive_path),
        "--result",
        str(args.result),
        "--baselines",
        str(baselines),
        "--work-dir",
        str(work / "boltz"),
        "--gpus",
        args.gpus,
        "--shards-per-gpu",
        str(args.shards_per_gpu),
        "--preprocessing-threads",
        str(args.preprocessing_threads),
        "--reduced-limit",
        str(args.reduced_limit),
        "--exact-limit",
        str(args.exact_limit),
        "--scratch-root",
        str(args.scratch_root),
    ]
    _run_until(qualification, deadline)
    _atomic_json(
        work / "complete.json",
        {
            "target": args.target,
            "epoch": args.epoch,
            "reaction": args.reaction,
            "elapsed_seconds": time.monotonic() - started,
            "result": str(args.result.resolve()),
        },
    )


if __name__ == "__main__":
    main()
