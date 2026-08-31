#!/usr/bin/env python3
"""Build a fresh, challenge-bound compound portfolio for one live epoch."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compound_miner.archive import CandidateArchive
from compound_miner.database import ReactionDatabase
from compound_miner.history import aggregate_observations, sync_history, sync_novelty
from compound_miner.proxy import TargetProxy
from compound_miner.release import ReleaseGateError, validate_release_payload
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


def _release_score(path: Path, target: str, epoch: int, reaction: int) -> float | None:
    """Return a verified live score, or ``None`` for a stale/invalid result."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        validate_release_payload(
            payload,
            reaction=reaction,
            target=target,
            epoch=epoch,
        )
        return float(payload["release"]["selection_score"])
    except (OSError, ValueError, TypeError, KeyError, ReleaseGateError):
        return None


def _promote_if_improved(
    staged: Path,
    destination: Path,
    target: str,
    epoch: int,
    reaction: int,
) -> tuple[bool, float]:
    """Atomically publish a staged exact release only when it improves."""
    staged_score = _release_score(staged, target, epoch, reaction)
    if staged_score is None:
        raise ReleaseGateError(f"staged result is not a valid exact release: {staged}")
    incumbent_score = _release_score(destination, target, epoch, reaction)
    if incumbent_score is not None and staged_score <= incumbent_score:
        return False, incumbent_score

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name, dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(staged.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return True, staged_score


def _can_start_iteration(
    remaining: float,
    observed_seconds: list[float],
    minimum_seconds: float,
    reserve_seconds: float,
) -> bool:
    """Require enough budget for a complete measured pass plus commit headroom."""
    estimate = max([minimum_seconds, *observed_seconds])
    return remaining >= estimate + reserve_seconds


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
    parser.add_argument(
        "--diverse-retention",
        action="store_true",
        help="Retain deterministic exploitation, uncertainty, and coverage lanes",
    )
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=3)
    parser.add_argument("--preprocessing-threads", type=int, default=1)
    parser.add_argument("--reduced-limit", type=int, default=1_024)
    parser.add_argument("--exact-limit", type=int, default=128)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--minimum-iteration-seconds",
        type=float,
        default=900.0,
        help="Do not start another full proposal/Boltz pass without this much time",
    )
    parser.add_argument(
        "--iteration-reserve-seconds",
        type=float,
        default=120.0,
        help="Keep this much deadline headroom for atomic publication and commitment",
    )
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
    iteration_seconds: list[float] = []
    # A restarted/upgraded runner may inherit the original one-shot portfolio.
    # Treat it as pass zero so recovery continues into fresh chemistry instead
    # of paying to score the same molecules again.
    legacy_reduced = work / "boltz" / f"rxn{args.reaction}_reduced.json"
    prior_reduced: list[Path] = (
        [legacy_reduced] if legacy_reduced.is_file() else []
    )
    iterations: list[dict[str, object]] = []
    stop_reason = "max_iterations"
    for iteration in range(1, args.max_iterations + 1):
        remaining = deadline - time.monotonic()
        if iteration > 1 and not _can_start_iteration(
            remaining,
            iteration_seconds,
            args.minimum_iteration_seconds,
            args.iteration_reserve_seconds,
        ):
            stop_reason = "insufficient_time_for_measured_iteration"
            break

        iteration_started = time.monotonic()
        search_metrics = engine.run_round(
            reaction=args.reaction,
            proposals=args.proposals,
            exact_shortlist=args.exact_limit,
            similarity_shortlist=args.similarity_shortlist,
            result_path=None,
            construction_workers=args.construction_workers,
            similarity_workers=args.similarity_workers,
            diverse_retention=args.diverse_retention,
        )
        _atomic_json(
            work / f"search_metrics_iteration_{iteration}.json", search_metrics
        )

        qualification_work = work / "boltz" / f"iteration_{iteration}"
        staged_result = work / f"result_iteration_{iteration}.json"
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
            str(staged_result),
            "--baselines",
            str(baselines),
            "--work-dir",
            str(qualification_work),
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
        for prior in prior_reduced:
            qualification.extend(("--exclude-records-json", str(prior)))
        try:
            _run_until(qualification, deadline)
        except TimeoutError:
            if iteration == 1:
                raise
            stop_reason = "deadline_during_optional_iteration"
            break

        reduced = qualification_work / f"rxn{args.reaction}_reduced.json"
        prior_reduced.append(reduced)
        promoted, score = _promote_if_improved(
            staged_result,
            args.result,
            args.target,
            args.epoch,
            args.reaction,
        )
        duration = time.monotonic() - iteration_started
        iteration_seconds.append(duration)
        iterations.append(
            {
                "iteration": iteration,
                "elapsed_seconds": duration,
                "promoted": promoted,
                "incumbent_score": score,
                "remaining_seconds": max(0.0, deadline - time.monotonic()),
                "search_metrics": search_metrics,
            }
        )
        _atomic_json(
            work / "progress.json",
            {
                "target": args.target,
                "epoch": args.epoch,
                "reaction": args.reaction,
                "iterations": iterations,
            },
        )

    _atomic_json(
        work / "complete.json",
        {
            "target": args.target,
            "epoch": args.epoch,
            "reaction": args.reaction,
            "elapsed_seconds": time.monotonic() - started,
            "result": str(args.result.resolve()),
            "iterations": iterations,
            "stop_reason": stop_reason,
        },
    )


if __name__ == "__main__":
    main()
