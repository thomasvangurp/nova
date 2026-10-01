"""Reconstruct strict rankings without touching the live publisher or chain.

Run with the deployed bgen Python. Each completed epoch uses only the exact
inputs named by its archived provisional ranking, its retained config/archive,
and public competitor fields strictly before that epoch. Output is explicitly
marked retrospective, not evidence that a qualified submission was published.
Child output is captured and discarded; only allowlisted diagnostics escape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def emit(**fields):
    print(json.dumps(fields, sort_keys=True), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(payload, target, gates):
    if payload.get("target") != target or not payload.get("candidates"):
        raise ValueError("invalid ranking target or empty candidates")
    qualification = payload["qualification"]
    pose_gates = qualification["pose_stability"]
    for name in ("minimum_structure_observations", "minimum_median_contact_jaccard",
                 "maximum_cdr3_rmsd", "minimum_epitope_occupancy"):
        if float(pose_gates[name]) != float(gates["--" + name.replace("_", "-")]):
            raise ValueError("pose gates differ from deployed qualification policy")
    for candidate in payload["candidates"]:
        if candidate["observation_count"] < int(gates["--minimum-observations"]):
            raise ValueError("missing scalar evidence")
        for field, flag in (
            ("competitive_rate", "--minimum-competitive-rate"),
            ("observation_competitive_rate", "--minimum-observation-competitive-rate"),
        ):
            value = float(candidate[field])
            if not math.isfinite(value) or value < float(gates[flag]):
                raise ValueError("candidate failed competitive-rate gate")
        ceilings = [(float(candidate["score_p90"]), float(gates["--maximum-p90-rank-sum"]))]
        ceilings.extend((float(candidate["p90_category_rank_sums"][category]),
                         float(gates[flag])) for category, flag in (
            ("confidence", "--maximum-p90-confidence-rank-sum"),
            ("physical_interaction", "--maximum-p90-physical-interaction-rank-sum"),
            ("developability", "--maximum-p90-developability-rank-sum"),
        ))
        if any(not math.isfinite(value) or value > ceiling for value, ceiling in ceilings):
            raise ValueError("candidate failed upper-tail quality gate")
        pose = candidate["pose_stability"]
        if int(pose["structure_observation_count"]) < int(gates["--minimum-structure-observations"]):
            raise ValueError("missing structural evidence")
        checks = (
            (float(pose.get("contact_jaccard_median", 0)), float(gates["--minimum-median-contact-jaccard"]), "minimum"),
            (float(pose.get("cdr3_rmsd_max_angstrom", math.inf)), float(gates["--maximum-cdr3-rmsd"]), "maximum"),
            (float(pose.get("epitope_occupancy_min", 0)), float(gates["--minimum-epitope-occupancy"]), "minimum"),
        )
        if any(not math.isfinite(value) or (value < limit if direction == "minimum" else value > limit)
               for value, limit, direction in checks):
            raise ValueError("candidate failed structural gate")


def run_epoch(root, epoch, timeout, limit, cache_runner=None, cache_dir=None, pose_workers=1):
    sys.path.insert(0, str(root))
    from nanobody_opt import publication_gates

    directory = root / "live" / f"epoch{epoch}"
    destination = directory / "qualified_rankings.json"
    if destination.exists():
        emit(epoch=epoch, status="already_exists", output=str(destination))
        return True
    provisional = directory / "provisional_rankings.json"
    archived = json.loads(provisional.read_text())
    exact = [Path(name) for name in archived["exact_files"]]
    if not exact or any(not path.is_file() for path in exact):
        raise ValueError("missing archived exact input")
    for path in exact:
        for part in path.parts:
            if part.startswith("epoch") and part[5:].isdigit() and int(part[5:]) > epoch:
                raise ValueError("future-epoch input rejected")
    config, archive = directory / "config.yaml", directory / "archive.csv"
    exclusion = directory / "excluded_sequences.txt"
    sources = [root / "nanobody_opt" / name for name in
               ("rank_exact.py", "pose_stability.py", "publication_gates.py", "win_probability.py")]
    manifest = {str(path): digest(path) for path in [provisional, config, archive, exclusion, *exact, *sources]}
    if cache_runner is not None:
        for path in (cache_runner, cache_runner.parent / "qualified_pose_cache.py",
                     root / "nanobody_opt" / "regions.py"):
            manifest[str(path)] = digest(path)
    handle, temporary = tempfile.mkstemp(prefix=".qualified_backfill_", suffix=".json", dir=directory)
    os.close(handle)
    temporary = Path(temporary)
    command = [sys.executable, str(sources[0])]
    for path in exact:
        command.extend(["--exact", str(path)])
    command.extend([
        "--archive", str(archive), "--target", archived["target"],
        "--output", str(temporary), "--exclude-file", str(exclusion),
        "--recent", "300", "--field-size", "49", "--bootstrap-fields", "64",
        "--limit", str(limit), "--before-epoch", str(epoch), "--api-field-count", "10",
        "--minimum-api-fields", "3", "--pose-config", str(config),
        "--selection-order", archived.get("selection_order", "win-probability"),
        *publication_gates.gate_arguments(),
    ])
    # Newer deployed rankers expose explicit policy flags. Preserve the policy
    # recorded for this epoch rather than silently taking today's defaults.
    rank_source = sources[0].read_text()
    for flag, key, default in (
        ("--consensus-ranking-policy", "consensus_ranking_policy", "tier"),
        ("--selection-metric-policy", "selection_metric_policy", "plain"),
    ):
        if flag in rank_source:
            command.extend([flag, archived.get(key, default)])
    if archived.get("consensus_shortlist", {}).get("mode") == "paired":
        command.append("--paired-consensus-shortlist")
    if cache_runner is not None:
        command = [sys.executable, str(cache_runner), "--rank-script", str(sources[0]),
                   "--cache-dir", str(cache_dir), "--pose-workers", str(pose_workers), "--", *command[2:]]
    environment = {name: os.environ[name] for name in ("PATH", "HOME", "LANG") if name in os.environ}
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    start = time.monotonic()
    emit(epoch=epoch, status="running", exact_file_count=len(exact), limit=limit, timeout_seconds=timeout)
    try:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, start_new_session=True)
        while True:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                process.communicate(timeout=min(30, remaining))
                break
            except subprocess.TimeoutExpired:
                emit(epoch=epoch, status="running", elapsed_seconds=round(time.monotonic() - start, 2))
        result = process
        if result.returncode:
            # Never emit captured streams or arbitrary exception messages.
            emit(epoch=epoch, status="rank_failed", returncode=result.returncode,
                 elapsed_seconds=round(time.monotonic() - start, 2))
            return False
        if any(digest(Path(path)) != expected for path, expected in manifest.items()):
            raise ValueError("an input changed during reconstruction")
        payload = json.loads(temporary.read_text())
        validate(payload, archived["target"], publication_gates.gate_defaults())
        payload["reconstruction"] = {
            "retrospective": True, "epoch": epoch, "chain_submission_performed": False,
            "input_capture": "retained files at reconstruction time; not an immutable commit-time snapshot",
            "input_sha256": manifest, "elapsed_seconds": round(time.monotonic() - start, 2),
        }
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        # Exclusive installation: never replace an existing qualified artifact.
        os.link(temporary, destination)
        emit(epoch=epoch, status="complete", output=str(destination),
             candidate_count=payload.get("candidate_count"), selected_count=len(payload["candidates"]),
             elapsed_seconds=payload["reconstruction"]["elapsed_seconds"],
             pose_cache=payload.get("pose_cache", {}),
             best_probability=payload["candidates"][0].get("win_probability"))
        return True
    except subprocess.TimeoutExpired:
        emit(epoch=epoch, status="timeout", elapsed_seconds=round(time.monotonic() - start, 2))
        return False
    finally:
        if "process" in locals() and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--cache-runner", type=Path)
    parser.add_argument("--pose-cache-dir", type=Path, default=Path("/root/artifacts/qualified_pose_cache"))
    parser.add_argument("--pose-workers", type=int, default=1)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.limit <= 0:
        parser.error("timeout and limit must be positive")
    os.nice(15)
    ok = True
    for epoch in args.epochs:
        try:
            ok = run_epoch(args.runtime_root, epoch, args.timeout_seconds, args.limit,
                           args.cache_runner, args.pose_cache_dir, args.pose_workers) and ok
        except Exception as error:
            emit(epoch=epoch, status="failed", error_type=type(error).__name__)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
