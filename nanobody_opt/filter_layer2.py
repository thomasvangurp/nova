"""Run the validator's actual IgBLAST and TNP gates before GPU scoring.

The cheap gates in ``gates.py`` and ``diversity.py`` deliberately have no
external dependencies. This second stage mirrors ``validate_nanobodies`` from
upstream NOVA: one batched IgBLAST call, the live nativeness/humanness and VHH
thresholds, then the same concurrent DevelopabilityService used by validators.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import tempfile

#: Bump when the TNP tool or CDR annotation changes; invalidates the
#: developability cache so a stale profile can't bypass the hoisted gate.
DEVELOPABILITY_CACHE_VERSION = "tnp-2026-07-redregion-v1"
import multiprocessing as mp
from pathlib import Path
from typing import Any

import yaml


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        blob = json.loads(path.read_text())
        if isinstance(blob, dict):
            blob = blob.get(
                "candidates", blob.get("sequences", blob.get("cases", blob))
            )
        if isinstance(blob, dict):
            rows = [{"sequence": seq, **(meta if isinstance(meta, dict) else {})}
                    for seq, meta in blob.items()]
        elif isinstance(blob, list):
            rows = [({"sequence": item} if isinstance(item, str) else dict(item))
                    for item in blob]
        else:
            raise ValueError("JSON input must contain a list or sequence mapping")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        seq = str(row.get("sequence", "")).strip().upper()
        if seq and seq not in seen:
            seen.add(seq)
            normalized.append({**row, "sequence": seq})
    return normalized


def _score_value(result: Any, name: str) -> Any:
    return getattr(result, name, None) if not isinstance(result, dict) else result.get(name)


def _igblast_chunk(items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Score one isolated IgBLAST/ANARCI batch in a worker process."""
    from dataclasses import asdict
    from metanano.utils import igblast_nativeness

    return [asdict(result) for result in igblast_nativeness.score_sequences(dict(items))]


def _cached_igblast(
    keyed: dict[str, str], cache_dir: Path, concurrency: int, batch_size: int
) -> list[dict[str, Any]]:
    """Run independent IgBLAST batches in parallel and cache exact results."""
    cache_dir = cache_dir / "igblast"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    for seq_id, sequence in keyed.items():
        path = cache_dir / f"{seq_id}.json"
        try:
            cached = json.loads(path.read_text())
            if cached.get("sequence") == sequence:
                results[seq_id] = cached["igblast"]
                continue
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        missing.append((seq_id, sequence))

    print(f"IgBLAST cache: {len(keyed) - len(missing):,} hit; "
          f"{len(missing):,} missing", flush=True)
    chunks = [missing[start : start + batch_size]
              for start in range(0, len(missing), batch_size)]
    workers = max(1, min(200, concurrency, len(chunks))) if chunks else 0
    completed = 0

    def checkpoint(chunk, scored):
        nonlocal completed
        if len(chunk) != len(scored):
            raise RuntimeError(
                f"IgBLAST returned {len(scored)} results for {len(chunk)} sequences"
            )
        expected = {seq_id: sequence for seq_id, sequence in chunk}
        for result in scored:
            seq_id = result.get("sequence_id")
            if seq_id not in expected:
                raise RuntimeError(f"unexpected IgBLAST sequence ID: {seq_id!r}")
            results[seq_id] = result
            payload = {"sequence": expected[seq_id], "igblast": result}
            (cache_dir / f"{seq_id}.json").write_text(json.dumps(payload) + "\n")
        completed += len(chunk)
        if completed == len(missing) or completed % 256 < len(chunk):
            print(f"IgBLAST checkpoint: {completed:,}/{len(missing):,} "
                  "newly measured", flush=True)

    if workers == 1:
        for chunk in chunks:
            checkpoint(chunk, _igblast_chunk(chunk))
    elif workers:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {executor.submit(_igblast_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                checkpoint(futures[future], future.result())

    if len(results) != len(keyed):
        raise RuntimeError(
            f"IgBLAST result count mismatch: expected {len(keyed)}, got {len(results)}"
        )
    return [results[seq_id] for seq_id in keyed]


async def _developability(sequences: list[str], concurrency: int) -> list[dict[str, Any]]:
    # Fast path: a persistent TNP daemon loads NanoBodyBuilder2 once and runs
    # the same n_threads=1 TNP pipeline (ANARCI -> predict -> OpenMM refine ->
    # compactness -> PSH/PPC/PNC -> six flags) over a parallel worker pool,
    # instead of a subprocess per sequence (TNP -> NanoBodyBuilder2) that re-imports
    # torch/ImmuneBuilder/openmm and re-reads 700 MB of weights off disk each time.
    import nanobody_opt.tnp_daemon as tnp_daemon
    if tnp_daemon.ENABLED:
        try:
            if tnp_daemon.available():
                return await asyncio.to_thread(tnp_daemon.profile_batch, sequences)
        except Exception as error:  # fall back to the subprocess path, as today
            print("TNP_DAEMON_FALLBACK error=%s: %s"
                  % (type(error).__name__, error), flush=True)

    from metanano.config import Config
    from metanano.services.async_manager import AsyncServiceManager
    from metanano.services.developability_service import DevelopabilityService

    cfg = Config()
    cfg.async_config.max_concurrent_tnp = concurrency
    # Feed the whole checkpoint group into the semaphore-controlled service.
    # Rigid concurrency-sized waves leave nearly every core idle while the
    # final one or two slow structures finish; a rolling queue starts their
    # replacements as soon as any slot becomes free.
    cfg.async_config.batch_size = max(1, len(sequences))
    # Raise the default asyncio executor so asyncio.to_thread spawns up to
    # `concurrency` TNP subprocesses instead of the 32-thread default cap
    # (which is what collapsed --concurrency 200 to ~32, and GPU-memory
    # contention with the scorer daemon collapsed it further to ~10).
    import concurrent.futures
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    )
    manager = AsyncServiceManager(cfg.async_config)
    service = DevelopabilityService(cfg.developability, manager)
    try:
        return await service.analyze_batch_async(sequences)
    finally:
        await manager.shutdown()


def _measured_developability(result: Any) -> bool:
    """Distinguish a biological TNP decision from an execution failure."""
    return (
        isinstance(result, dict)
        and isinstance(result.get("passed"), bool)
        and isinstance(result.get("flags"), dict)
    )


def _developability_cache_digest(sequence: str) -> str:
    """Sequence + tool/annotation version key, so a TNP change invalidates cache.

    A sequence-only key would silently reuse a profile measured under an older
    TNP tool or CDR annotation, defeating the hoisted developability gate.
    """
    return hashlib.sha256(
        (DEVELOPABILITY_CACHE_VERSION + "\n" + sequence).encode("ascii")
    ).hexdigest()


def cached_developability_result(sequence: str, cache_dir: Path) -> dict | None:
    """Read only a measured result for this exact input and tool version."""
    path = cache_dir / f"{_developability_cache_digest(sequence)}.json"
    try:
        cached = json.loads(path.read_text())
        if (cached.get("sequence") == sequence
                and _measured_developability(cached.get("developability"))):
            return cached["developability"]
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return None


def developability_rejection(result: Any) -> str:
    if not _measured_developability(result):
        return "developability_unavailable"
    return "" if result["passed"] else "developability"


def _cached_developability(
    rows: list[dict[str, Any]], cache_dir: Path, concurrency: int
) -> list[dict[str, Any]]:
    """Return TNP results in input order, checkpointing every completed wave."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(rows)
    missing: list[int] = []
    for index, row in enumerate(rows):
        cached = cached_developability_result(row["sequence"], cache_dir)
        if cached is not None:
            results[index] = cached
            continue
        missing.append(index)

    print(f"TNP cache: {len(rows) - len(missing):,} hit; {len(missing):,} missing")
    checkpoint_size = concurrency * 4
    for start in range(0, len(missing), checkpoint_size):
        indices = missing[start : start + checkpoint_size]
        wave = asyncio.run(_developability(
            [rows[index]["sequence"] for index in indices], concurrency
        ))
        pending_writes: list[tuple[str, str]] = []
        for index, result in zip(indices, wave, strict=True):
            results[index] = result
            if _measured_developability(result):
                digest = _developability_cache_digest(rows[index]["sequence"])
                payload = {"sequence": rows[index]["sequence"], "developability": result}
                # Flush after this wave, outside the concurrent compute, so a
                # restart does not discard every previously completed wave.
                pending_writes.append((digest, json.dumps(payload) + "\n"))
        for digest, blob in pending_writes:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(blob)
                temporary.replace(cache_dir / f"{digest}.json")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        print(f"TNP checkpoint: {min(start + len(indices), len(missing)):,}/"
              f"{len(missing):,} newly measured", flush=True)

    if any(result is None for result in results):
        raise RuntimeError("internal error: missing TNP result after checkpointed run")
    return [result for result in results if result is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--igblast-concurrency", type=int, default=16)
    parser.add_argument("--igblast-batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("layer2_cache"))
    parser.add_argument("--limit", type=int, default=0, help="canary/debug prefix only")
    parser.add_argument("--tail", type=int, default=0, help="canary/debug suffix only")
    parser.add_argument(
        "--igblast-only",
        action="store_true",
        help="stop after the sequence-only nativeness/humanness/VHH gate",
    )
    args = parser.parse_args()

    rows = _load_candidates(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if args.tail:
        rows = rows[-args.tail :]
    if not rows:
        raise SystemExit("no candidate sequences found")

    requirements = yaml.safe_load(args.config.read_text())["nanobody_requirements"]
    min_nativeness = float(requirements["min_nativeness_score"])
    min_humanness = float(requirements["min_human_framework_score"])
    enforce_vhh = bool(requirements.get("enforce_vhh_hallmarks", True))

    keyed = {
        hashlib.sha256(row["sequence"].encode("ascii")).hexdigest(): row["sequence"]
        for row in rows
    }
    igblast_concurrency = max(1, min(200, args.igblast_concurrency))
    igblast_batch_size = max(1, args.igblast_batch_size)
    scored = _cached_igblast(
        keyed, args.cache_dir, igblast_concurrency, igblast_batch_size
    )
    by_id = {_score_value(result, "sequence_id"): result for result in scored}

    survivors: list[dict[str, Any]] = []
    igblast_rejected: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for row in rows:
        seq_id = hashlib.sha256(row["sequence"].encode("ascii")).hexdigest()
        result = by_id.get(seq_id)
        if result is None:
            reason = "missing_igblast_result"
            details: dict[str, Any] = {}
        else:
            features = _score_value(result, "features") or {}
            nativeness = _score_value(result, "vhh_nativeness")
            humanness = _score_value(result, "human_framework")
            domain = ("VHH" if features.get("fr2_pos49", "X") in "EQ"
                      and features.get("fr2_pos50", "X") == "R" else "VH")
            details = {
                "nativeness_score": nativeness,
                "human_framework_score": humanness,
                "domain_type": domain,
            }
            if _score_value(result, "hard_reject"):
                reason = "igblast_hard_reject"
                details["igblast_reason"] = _score_value(result, "reject_reason")
            elif enforce_vhh and domain != "VHH":
                reason = "not_vhh"
            elif nativeness is None or float(nativeness) < min_nativeness:
                reason = "low_nativeness"
            elif humanness is None or float(humanness) < min_humanness:
                reason = "low_humanness"
            else:
                reason = ""

        record = {**row, **details}
        if reason:
            record["layer2_passed"] = False
            record["layer2_rejection"] = reason
            rejected[reason] = rejected.get(reason, 0) + 1
            igblast_rejected.append(record)
        else:
            survivors.append(record)

    print(f"IgBLAST: {len(survivors):,}/{len(rows):,} pass live thresholds")
    for reason, count in sorted(rejected.items(), key=lambda item: -item[1]):
        print(f"  rejected by {reason}: {count:,}")

    if args.igblast_only:
        for row in survivors:
            row["igblast_passed"] = True
        output = {
            "thresholds": {
                "min_nativeness_score": min_nativeness,
                "min_human_framework_score": min_humanness,
                "enforce_vhh_hallmarks": enforce_vhh,
            },
            "igblast_only": True,
            "input_count": len(rows),
            "pass_count": len(survivors),
            "rejection_counts": rejected,
            "sequences": [row["sequence"] for row in survivors],
            "candidates": survivors,
            "rejected": igblast_rejected,
        }
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(
            f"IgBLAST-only: {len(survivors):,}/{len(rows):,} pass; "
            f"wrote {args.output}"
        )
        if not survivors:
            # Degrade instead of aborting the epoch: write the (already empty)
            # pool and exit cleanly so the producer's next stage sees a marked
            # empty input rather than a killed subprocess.
            print(
                "LAYER2_EMPTY: no candidate passed IgBLAST; "
                "writing marked-empty pool",
                flush=True,
            )
            return
        return

    if survivors:
        concurrency = max(1, min(200, args.concurrency))
        dev_results = _cached_developability(survivors, args.cache_dir, concurrency)
        for row, result in zip(survivors, dev_results, strict=True):
            row["developability"] = result
            reason = developability_rejection(result)
            row["layer2_passed"] = not reason
            if reason:
                row["layer2_rejection"] = reason
                rejected[reason] = rejected.get(reason, 0) + 1

    passed = [row for row in survivors if row.get("layer2_passed")]
    failed_survivors = [row for row in survivors if not row.get("layer2_passed")]

    output = {
        "thresholds": {
            "min_nativeness_score": min_nativeness,
            "min_human_framework_score": min_humanness,
            "enforce_vhh_hallmarks": enforce_vhh,
        },
        "input_count": len(rows),
        "pass_count": len(passed),
        "rejection_counts": rejected,
        "sequences": [row["sequence"] for row in passed],
        "candidates": passed,
        "rejected": igblast_rejected + failed_survivors,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"TNP: {len(passed):,}/{len(survivors):,} pass; wrote {args.output}")
    if not passed:
        # Degrade instead of aborting the epoch: the (already written) pool is
        # marked-empty, and the downstream race handles empty input gracefully.
        print(
            "LAYER2_EMPTY: no candidate passed Layer-2; "
            "writing marked-empty pool",
            flush=True,
        )
        return


if __name__ == "__main__":
    main()
