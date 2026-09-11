"""Race Layer-2 nanobodies in publishable, validator-shaped screen tranches.

Each completed screen tranche is immediately visible to the independent
publisher. The first leader is repeated concurrently across isolated GPU
processes, then screening continues until the full candidate budget is used.
Later tranche leaders can replace the incumbent when their evidence is better.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nanobody_opt.mutation_network import (
    ranking_observations,
    update_observation_store,
)
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL


STRICT_RANK_ARGUMENTS = (
    "--minimum-observations", "8",
    "--competitive-score-threshold", "75",
    "--minimum-competitive-rate", "0.75",
    "--minimum-observation-competitive-rate", "0.75",
    "--maximum-p90-rank-sum", "90",
    "--maximum-p90-confidence-rank-sum", "70",
    "--maximum-p90-physical-interaction-rank-sum", "40",
    "--maximum-p90-developability-rank-sum", "15",
    "--minimum-structure-observations", "8",
    "--minimum-median-contact-jaccard", "0.35",
    "--maximum-cdr3-rmsd", "5",
    "--minimum-epitope-occupancy", "0",
)
DEFAULT_SCORER = Path("/root/nanobody_opt/score_boltzgen_batch.py")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
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


def _candidate_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    items = payload if isinstance(payload, list) else (
        payload.get("candidates") or payload.get("sequences") or []
    )
    if items and all(isinstance(item, dict) for item in items):
        items = sorted(items, key=lambda item: (
            0 if item.get("screen_priority") is not None else 1,
            int(item.get("screen_priority", 0)),
            0
            if item.get("proposal_type") == "exhaustive_single_substitution"
            else 1
        ))
    records = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("layer2_passed") is False:
            continue
        sequence = item.get("sequence") if isinstance(item, dict) else item
        if isinstance(sequence, str) and sequence.strip():
            normalized = sequence.strip().upper()
            if normalized in seen:
                continue
            record = dict(item) if isinstance(item, dict) else {}
            record["sequence"] = normalized
            records.append(record)
            seen.add(normalized)
    return records


def _candidate_sequences(path: Path, limit: int) -> list[str]:
    return [
        str(record["sequence"])
        for record in _candidate_records(path)[:limit]
    ]


def _batches(items: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


def _balanced_screen_tranche(
    pending: list[str],
    metadata: dict[str, dict[str, Any]],
    screened_parent_counts: Counter[str],
    size: int,
) -> list[str]:
    """Select the next tranche without oversampling one structural basin.

    The proposal generator normally interleaves parents already.  This second
    guard rebuilds the queue after every completed tranche, so a Layer-2 pass
    imbalance or an older grouped proposal file cannot spend most of the exact
    budget on one family.  It deliberately does not use neighbour total scores:
    the exact replay shows that binding/interaction outcomes are not smooth
    enough for graph-based arm elimination.
    """
    if size < 1:
        raise ValueError("tranche size must be positive")
    if not pending:
        return []
    def allocation_key(sequence: str) -> str:
        record = metadata.get(sequence, {})
        return str(
            record.get("parent_basin_core_id")
            or record.get("basin_core_id")
            or record.get("parent_sequence")
            or "__unattributed__"
        )

    if not any(allocation_key(sequence) != "__unattributed__" for sequence in pending):
        return pending[:size]

    available = list(enumerate(pending))
    counts = Counter(screened_parent_counts)
    selected: list[str] = []
    while available and len(selected) < size:
        choice = min(
            range(len(available)),
            key=lambda index: (
                counts[allocation_key(available[index][1])],
                int(
                    metadata.get(available[index][1], {}).get(
                        "parent_basin_rank", 10**9
                    )
                ),
                available[index][0],
            ),
        )
        _, sequence = available.pop(choice)
        parent = allocation_key(sequence)
        selected.append(sequence)
        counts[parent] += 1
    return selected


def _resumable_screen_tranche(
    pending: list[str],
    metadata: dict[str, dict[str, Any]],
    screened_parent_counts: Counter[str],
    size: int,
    *,
    source: Path,
    completed: Path,
) -> list[str]:
    """Preserve the actual candidate list of an already completed tranche."""
    if not completed.is_file():
        return _balanced_screen_tranche(
            pending, metadata, screened_parent_counts, size
        )
    if not source.is_file():
        raise ValueError(f"completed tranche lacks its candidate source: {source}")
    sequences = _candidate_sequences(source, size)
    if not sequences or any(sequence not in pending for sequence in sequences):
        raise ValueError(f"completed tranche source is inconsistent: {source}")
    return sequences


def _initial_repeat_candidate(
    ranked_sequences: list[str],
    candidates: dict[str, dict[str, Any]],
    *,
    repeated: set[str],
    maximum_screen_p90: float,
    allow_weak_fallback: bool,
) -> tuple[str, bool] | None:
    """Choose a strong one-shot candidate, or one bounded fallback candidate.

    A strict finalist may have at most a 90 p90 rank sum.  Repeating every first
    tranche leader above that ceiling consumed most of the early epoch without
    producing a qualified checkpoint.  Strong candidates remain immediately
    eligible; after enough screening, callers may authorize exactly one weak
    fallback to preserve a chance of first publication.
    """
    unseen = [sequence for sequence in ranked_sequences if sequence not in repeated]
    for sequence in unseen:
        score = float(
            candidates[sequence].get(
                "basin_adjusted_score", candidates[sequence]["score_p90"]
            )
        )
        if score <= maximum_screen_p90:
            return sequence, False
    if allow_weak_fallback and unseen:
        return unseen[0], True
    return None


def _screen_execution_shape(
    candidate_count: int,
    gpu_count: int,
    predict_batch_size: int | None = None,
) -> tuple[int, int]:
    """Keep one publishable tranche in one model-loading wave.

    The sharded scorer assigns candidates round-robin to GPUs.  Giving the
    worker its entire shard in one wrapper call lets BoltzGen load each model
    once for all candidates on that GPU.  The old ``wave_size=gpu_count`` and
    ``predict_batch_size=1`` combination reloaded the complete pipeline once
    per candidate even though publication only happened after the full tranche.

    Predict batch composition is load-bearing: ``score_boltzgen_batch`` scores a
    whole chunk in one ``predict`` call, and the wrapper's own measurements show
    the same molecule moving by more than a factor of two depending on what else
    shared that call.  The derived ``ceil(candidates / gpus)`` shape therefore
    silently changes scores when the tranche grows, which is exactly what a
    faster scorer invites.  Pin ``predict_batch_size`` whenever the tranche size
    changes so a throughput change cannot alter the measured metric.
    """
    if candidate_count < 1 or gpu_count < 1:
        raise ValueError("candidate and GPU counts must be positive")
    if predict_batch_size is not None:
        if predict_batch_size < 1:
            raise ValueError("predict batch size must be positive")
        return candidate_count, predict_batch_size
    return candidate_count, math.ceil(candidate_count / gpu_count)


def _chain_allows_work(path: Path | None, epoch: int, reserve_blocks: int) -> bool:
    if path is None:
        return True
    try:
        state = json.loads(path.read_text())
        current_epoch = int(state["epoch"])
        block = int(state["block"])
        epoch_length = int(state["epoch_length"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return current_epoch == epoch and block % epoch_length < epoch_length - reserve_blocks


def _append_timing(path: Path, record: dict[str, Any]) -> None:
    """Append one phase-timing record so the epoch budget comes from measurement.

    A faster scorer only moves the bottleneck; without per-phase wall time there
    is no way to tell whether an epoch is limited by folding, by ranking, or by
    replication.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _exact_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/exact_all.json"))


def _rank_command(
    *,
    exact: list[Path],
    archive: Path,
    config: Path,
    target: str,
    epoch: int,
    output: Path,
    ledger: Path | None,
    strict: bool,
) -> list[str]:
    command = [sys.executable, "-m", "nanobody_opt.rank_exact"]
    for path in exact:
        command.extend(["--exact", str(path)])
    command.extend([
        "--archive", str(archive),
        "--target", target,
        "--output", str(output),
        "--recent", "300",
        "--field-size", "49",
        # Provisional basin shrinkage needs the complete screened cohort, not
        # only the raw top 30. Strict publication files remain compact.
        "--limit", "30" if strict else "10000",
    ])
    if ledger is not None:
        command.extend(["--exclude-file", str(ledger)])
    if strict:
        command.extend([
            "--pose-config", str(config),
            "--before-epoch", str(epoch),
            "--api-field-count", "10",
            "--minimum-api-fields", "3",
        ])
        command.extend(STRICT_RANK_ARGUMENTS)
    else:
        command.extend(["--selection-order", "p90"])
    return command


def _screen_command(
    *,
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    candidate_limit: int,
    scorer: Path,
    predict_batch_size: int | None,
) -> list[str] | None:
    """Return the scorer command, or ``None`` when the tranche is already scored.

    An artifact from the previous scoring protocol is preserved beside the
    tranche and never resumed, so a protocol change re-scores rather than mixing
    observations from two different selection rules.
    """
    completed = output_dir / "exact_all.json"
    if completed.is_file():
        try:
            protocol = json.loads(completed.read_text()).get("scorer_protocol")
        except (OSError, json.JSONDecodeError):
            protocol = None
        if protocol == SCORER_PROTOCOL:
            return None
        archived = completed.with_name(
            "exact_all.legacy-pre-dual-sample.json"
        )
        suffix = 1
        while archived.exists():
            archived = completed.with_name(
                f"exact_all.legacy-pre-dual-sample-{suffix}.json"
            )
            suffix += 1
        os.replace(completed, archived)
        print(
            f"EXACT_SCREEN_PROTOCOL_REFRESH preserved={archived}",
            flush=True,
        )
    wave_size, batch_size = _screen_execution_shape(
        candidate_limit, len(gpus), predict_batch_size
    )
    return [
        sys.executable,
        "-m",
        "nanobody_opt.score_boltzgen_waves",
        "--sequences", str(source),
        "--config", str(config),
        "--target", target,
        "--output-dir", str(output_dir),
        "--gpus", ",".join(gpus),
        "--wave-size", str(wave_size),
        "--predict-batch-size", str(batch_size),
        "--limit", str(candidate_limit),
        "--scorer", str(scorer),
        "--capture-structures",
    ]


class ScreenJob:
    """One screen tranche whose scorer may still be folding on the GPUs.

    The scorer owns every GPU while it runs, so nothing else may be scheduled
    until ``wait`` returns.  Ranking, basin shrinkage and the mutation-database
    update are pure CPU work and are the only things overlapped with it.
    """

    def __init__(
        self,
        *,
        index: int,
        tranche: list[str],
        root: Path,
        process: subprocess.Popen | None,
        command: list[str] | None,
        started: float,
    ) -> None:
        self.index = index
        self.tranche = tranche
        self.root = root
        self.completed = root / "exact_all.json"
        self._process = process
        self._command = command
        self.started = started
        self.scoring_seconds = 0.0

    def wait(self) -> Path:
        if self._process is None:
            return self.completed
        return_code = self._process.wait()
        self.scoring_seconds = time.monotonic() - self.started
        self._process = None
        if return_code:
            raise subprocess.CalledProcessError(return_code, self._command or [])
        return self.completed


def _start_screen(
    *,
    index: int,
    tranche: list[str],
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    scorer: Path = DEFAULT_SCORER,
    predict_batch_size: int | None = None,
) -> ScreenJob:
    """Dispatch one tranche without waiting for its GPU work to finish."""
    command = _screen_command(
        source=source,
        config=config,
        target=target,
        output_dir=output_dir,
        gpus=gpus,
        candidate_limit=len(tranche),
        scorer=scorer,
        predict_batch_size=predict_batch_size,
    )
    started = time.monotonic()
    process = None if command is None else subprocess.Popen(command)
    return ScreenJob(
        index=index,
        tranche=tranche,
        root=output_dir,
        process=process,
        command=command,
        started=started,
    )


def _run_screen(
    *,
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    candidate_limit: int,
    scorer: Path = DEFAULT_SCORER,
    predict_batch_size: int | None = None,
) -> Path:
    command = _screen_command(
        source=source,
        config=config,
        target=target,
        output_dir=output_dir,
        gpus=gpus,
        candidate_limit=candidate_limit,
        scorer=scorer,
        predict_batch_size=predict_batch_size,
    )
    if command is not None:
        subprocess.run(command, check=True)
    return output_dir / "exact_all.json"


def _repeat_command(
    *,
    input_path: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpu: str,
    scorer: Path = DEFAULT_SCORER,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "nanobody_opt.score_boltzgen_waves",
        "--sequences", str(input_path),
        "--config", str(config),
        "--target", target,
        "--output-dir", str(output_dir),
        "--gpus", gpu,
        "--wave-size", "1",
        "--predict-batch-size", "1",
        "--limit", "1",
        "--scorer", str(scorer),
        "--capture-structures",
    ]


def _run_parallel(commands: list[list[str]]) -> None:
    processes = [subprocess.Popen(command) for command in commands]
    failures = []
    for command, process in zip(commands, processes, strict=True):
        return_code = process.wait()
        if return_code:
            failures.append((return_code, command))
    if failures:
        raise subprocess.CalledProcessError(failures[0][0], failures[0][1])


def _score_repeats(
    *,
    sequence: str,
    repeat_numbers: list[int],
    root: Path,
    config: Path,
    target: str,
    gpus: list[str],
    scorer: Path = DEFAULT_SCORER,
) -> None:
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    candidate_root = root / "candidates" / digest
    input_path = candidate_root / "sequence.json"
    _atomic_json(input_path, {"target": target, "sequences": [sequence]})
    for start in range(0, len(repeat_numbers), len(gpus)):
        group = repeat_numbers[start : start + len(gpus)]
        commands = [
            _repeat_command(
                input_path=input_path,
                config=config,
                target=target,
                output_dir=candidate_root / f"repeat_{repeat:02d}",
                gpu=gpus[index],
                scorer=scorer,
            )
            for index, repeat in enumerate(group)
        ]
        _run_parallel(commands)


def _try_strict_rank(
    *,
    root: Path,
    archive: Path,
    config: Path,
    target: str,
    epoch: int,
    output: Path,
    ledger: Path | None,
) -> bool:
    try:
        subprocess.run(
            _rank_command(
                exact=_exact_paths(root),
                archive=archive,
                config=config,
                target=target,
                epoch=epoch,
                output=output,
                ledger=ledger,
                strict=True,
            ),
            check=True,
        )
    except subprocess.CalledProcessError:
        return False
    return output.is_file()


def _ranking_by_sequence(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {
        str(candidate["sequence"]): candidate
        for candidate in payload.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("sequence")
    }


def _apply_basin_shrinkage(
    path: Path,
    metadata: dict[str, dict[str, Any]],
    *,
    prior_strength: float = 2.0,
    minimum_samples: int = 3,
) -> dict[str, Any]:
    """Stabilize one-shot terminal ranks while keeping every raw score intact.

    Only terminal candidates whose parent belongs to the same masked core are
    pooled.  Their one-shot p90 is shrunk toward the currently screened basin
    median.  This prevents a single lucky terminal sample from dominating a
    broad weak cohort; repeated exact evidence still decides qualification.
    """
    if prior_strength < 0 or minimum_samples < 1:
        raise ValueError("invalid basin shrinkage configuration")
    payload = json.loads(path.read_text())
    candidates = [
        candidate for candidate in payload.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("sequence")
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        sequence = str(candidate["sequence"])
        record = metadata.get(sequence, {})
        identifier = str(record.get("parent_basin_core_id") or "")
        if not identifier or record.get("proposal_lane") != "basin_terminal":
            continue
        grouped.setdefault(identifier, []).append(candidate)

    for identifier, items in grouped.items():
        if len(items) < minimum_samples:
            continue
        scores = sorted(float(item["score_p90"]) for item in items)
        midpoint = len(scores) // 2
        basin_median = (
            scores[midpoint]
            if len(scores) % 2
            else (scores[midpoint - 1] + scores[midpoint]) / 2
        )
        for item in items:
            raw = float(item["score_p90"])
            adjusted = (
                raw + prior_strength * basin_median
            ) / (1.0 + prior_strength)
            item["basin_adjusted_score"] = adjusted
            item["basin_screen_median"] = basin_median
            item["basin_screen_sample_count"] = len(items)
            item["basin_shrinkage_strength"] = prior_strength
            item["basin_core_id"] = identifier

    original_order = {
        str(candidate["sequence"]): index for index, candidate in enumerate(candidates)
    }
    candidates.sort(key=lambda item: (
        float(item.get("basin_adjusted_score", item["score_p90"])),
        float(item["score_p90"]),
        original_order[str(item["sequence"])],
    ))
    payload["candidates"] = candidates
    payload["sequences"] = [str(item["sequence"]) for item in candidates]
    payload["basin_ranking"] = {
        "enabled": True,
        "prior_strength": prior_strength,
        "minimum_samples": minimum_samples,
        "eligible_basin_count": sum(
            len(items) >= minimum_samples for items in grouped.values()
        ),
    }
    _atomic_json(path, payload)
    return payload


def _best_confirmed_p90(path: Path) -> float:
    scores = [
        float(candidate["score_p90"])
        for candidate in _ranking_by_sequence(path).values()
    ]
    if not scores:
        raise ValueError(f"qualified ranking {path} has no scored candidates")
    return min(scores)


def _challenger_can_overtake(
    *, screen_p90: float, incumbent_p90: float, margin: float
) -> bool:
    """Reject only challengers that clearly trail the confirmed incumbent."""
    return screen_p90 <= incumbent_p90 + margin


def _store_ranking(
    database: Path | None,
    ranking: Path,
    *,
    epoch: int,
    target: str,
    source: str,
    candidate_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    if database is None or not ranking.is_file():
        return
    try:
        observations = ranking_observations(
            ranking,
            epoch=epoch,
            target=target,
            source=source,
        )
        for observation in observations:
            metadata = (candidate_metadata or {}).get(str(observation["sequence"]), {})
            observation.update(
                {
                    key: metadata[key]
                    for key in (
                        "parent_sequence",
                        "proposal_type",
                        "mutations",
                        "proposal_lane",
                        "parent_selection_lane",
                        "parent_selection_score",
                        "screen_priority",
                        "parent_basin_core_id",
                        "parent_basin_rank",
                        "basin_core_id",
                        "basin_allocation_reserved",
                    )
                    if metadata.get(key) is not None
                }
            )
        count = update_observation_store(
            database,
            observations,
        )
    except Exception as error:
        # Historical learning must never suppress a publishable result.
        print(
            f"MUTATION_DB_UPDATE_FAILED source={source} "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
    else:
        print(
            f"MUTATION_DB_UPDATED source={source} observations={count}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer2", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--mutation-database", type=Path)
    parser.add_argument("--chain-state", type=Path)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--scorer",
        type=Path,
        default=DEFAULT_SCORER,
        help=(
            "validator-protocol-compatible scorer entry point; replacement "
            "scorers must emit the same dual-sample artifact contract"
        ),
    )
    parser.add_argument("--candidate-limit", type=int, default=320)
    parser.add_argument(
        "--screen-tranche-size",
        type=int,
        default=20,
        help="publish and reconsider the leader after this many screen candidates",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=3,
        help="maximum finalists; fewer are repeated when an incumbent qualifies",
    )
    parser.add_argument(
        "--challenger-p90-margin",
        type=float,
        default=20.0,
        help=(
            "repeat a challenger after qualification only when its one-shot "
            "screen p90 is within this rank-sum margin of the incumbent"
        ),
    )
    parser.add_argument(
        "--maximum-initial-screen-p90",
        type=float,
        default=90.0,
        help=(
            "immediately repeat an initial candidate only when its one-shot "
            "p90 is at or below this strict qualification ceiling"
        ),
    )
    parser.add_argument(
        "--weak-initial-fallback-after",
        type=int,
        default=80,
        help=(
            "after screening this many candidates, permit one initial repeat "
            "above the p90 ceiling so first publication remains possible"
        ),
    )
    parser.add_argument("--qualification-observations", type=int, default=8)
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=1,
        help=(
            "dispatch this many screen tranches ahead so ranking, basin "
            "shrinkage and the mutation-database update run while the GPUs "
            "fold the next tranche; 0 restores fully serial behaviour"
        ),
    )
    parser.add_argument(
        "--screen-predict-batch-size",
        type=int,
        default=0,
        help=(
            "pin how many candidates share one BoltzGen predict call; scores "
            "depend on that grouping, so a larger tranche must not be allowed "
            "to change it implicitly.  0 derives the historical "
            "ceil(candidates / GPUs) shape"
        ),
    )
    parser.add_argument("--basin-shrinkage-strength", type=float, default=2.0)
    parser.add_argument("--minimum-basin-samples", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.finalists <= args.candidate_limit:
        parser.error("require 1 <= --finalists <= --candidate-limit")
    if not 1 <= args.screen_tranche_size <= args.candidate_limit:
        parser.error("require 1 <= --screen-tranche-size <= --candidate-limit")
    if args.qualification_observations < 8:
        parser.error("qualification requires at least eight observations")
    if args.challenger_p90_margin < 0:
        parser.error("--challenger-p90-margin cannot be negative")
    if args.maximum_initial_screen_p90 < 0:
        parser.error("--maximum-initial-screen-p90 cannot be negative")
    if args.weak_initial_fallback_after < 1:
        parser.error("--weak-initial-fallback-after must be positive")
    if args.basin_shrinkage_strength < 0:
        parser.error("--basin-shrinkage-strength cannot be negative")
    if args.minimum_basin_samples < 1:
        parser.error("--minimum-basin-samples must be positive")
    if args.pipeline_depth not in (0, 1):
        parser.error("--pipeline-depth must be 0 or 1")
    if args.screen_predict_batch_size < 0:
        parser.error("--screen-predict-batch-size cannot be negative")
    args.screen_predict_batch_size = args.screen_predict_batch_size or None
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("at least one GPU is required")
    candidate_records = _candidate_records(args.layer2)[: args.candidate_limit]
    sequences = [str(record["sequence"]) for record in candidate_records]
    candidate_metadata = {
        str(record["sequence"]): record for record in candidate_records
    }
    if not sequences:
        raise SystemExit("Layer-2 input has no passing candidates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_dir / "screen_candidates.json",
        {"target": args.target, "sequences": sequences},
    )
    if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
        raise SystemExit("current chain epoch is inside the scoring reserve window")

    screen_ranking = args.output_dir / "screen_ranking.json"
    screen_exact_paths: list[Path] = []
    qualified_outputs: list[Path] = []
    repeated_finalists: list[str] = []
    skipped_finalists: list[str] = []
    deferred_weak_leaders: list[str] = []
    incumbent_p90: float | None = None
    weak_initial_fallback_spent = False
    screened_parent_counts: Counter[str] = Counter()
    screened = 0
    pending = list(sequences)
    tranche_index = 0
    timings_path = args.output_dir / "timings.jsonl"

    def _dispatch_next_tranche() -> ScreenJob | None:
        """Start the next tranche's GPU work, or return None when none may run.

        Tranche composition depends only on the pending queue and how often each
        structural basin has already been screened, never on the ranking of an
        earlier tranche, so dispatching one tranche ahead selects exactly the
        candidates a fully serial run would have selected.
        """
        nonlocal tranche_index
        if not pending:
            return None
        if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
            print("EXACT_RACE_RESERVE_REACHED before next screen tranche", flush=True)
            return None
        tranche_index += 1
        tranche_root = args.output_dir / "screen" / f"tranche_{tranche_index:02d}"
        source = tranche_root / "candidates.json"
        completed = tranche_root / "exact_all.json"
        tranche = _resumable_screen_tranche(
            pending,
            candidate_metadata,
            screened_parent_counts,
            args.screen_tranche_size,
            source=source,
            completed=completed,
        )
        if not completed.is_file():
            _atomic_json(source, {"target": args.target, "sequences": tranche})
        for sequence in tranche:
            pending.remove(sequence)
            record = candidate_metadata.get(sequence, {})
            parent = str(
                record.get("parent_basin_core_id")
                or record.get("basin_core_id")
                or record.get("parent_sequence")
                or "__unattributed__"
            )
            screened_parent_counts[parent] += 1
        return _start_screen(
            index=tranche_index,
            tranche=tranche,
            source=source,
            config=args.config,
            target=args.target,
            output_dir=tranche_root,
            gpus=gpus,
            scorer=args.scorer,
            predict_batch_size=args.screen_predict_batch_size,
        )

    in_flight = _dispatch_next_tranche()
    while in_flight is not None:
        job = in_flight
        in_flight = None
        job.wait()
        # The scorer owns every GPU, so the next tranche is dispatched first and
        # the CPU-only bookkeeping below runs while it folds.
        if args.pipeline_depth:
            in_flight = _dispatch_next_tranche()
        bookkeeping_started = time.monotonic()
        screen_exact_paths.append(job.completed)
        screened += len(job.tranche)
        subprocess.run(
            _rank_command(
                exact=screen_exact_paths,
                archive=args.archive,
                config=args.config,
                target=args.target,
                epoch=args.epoch,
                output=screen_ranking,
                ledger=args.ledger,
                strict=False,
            ),
            check=True,
        )
        _apply_basin_shrinkage(
            screen_ranking,
            candidate_metadata,
            prior_strength=args.basin_shrinkage_strength,
            minimum_samples=args.minimum_basin_samples,
        )
        _store_ranking(
            args.mutation_database,
            screen_ranking,
            epoch=args.epoch,
            target=args.target,
            source="live_exact_screen",
            candidate_metadata=candidate_metadata,
        )
        screen_candidates = _ranking_by_sequence(screen_ranking)
        ranked_sequences = json.loads(screen_ranking.read_text())["sequences"]
        print(
            f"EXACT_SCREEN_TRANCHE_COMPLETE tranche={job.index} "
            f"screened={screened}/{len(sequences)} leader={ranked_sequences[0]}",
            flush=True,
        )
        bookkeeping_seconds = time.monotonic() - bookkeeping_started
        _append_timing(
            timings_path,
            {
                "phase": "screen_tranche",
                "tranche": job.index,
                "candidates": len(job.tranche),
                "scoring_seconds": round(job.scoring_seconds, 1),
                "bookkeeping_seconds": round(bookkeeping_seconds, 1),
                "overlapped": bool(args.pipeline_depth),
                "screened": screened,
            },
        )
        print(
            f"EXACT_TRANCHE_TIMING tranche={job.index} "
            f"scoring={job.scoring_seconds:.0f}s "
            f"bookkeeping={bookkeeping_seconds:.0f}s "
            f"candidates={len(job.tranche)}",
            flush=True,
        )
        if len(repeated_finalists) >= args.finalists:
            continue
        weak_fallback_boundary = min(args.weak_initial_fallback_after, len(sequences))
        selection = _initial_repeat_candidate(
            ranked_sequences,
            screen_candidates,
            repeated=set(repeated_finalists),
            maximum_screen_p90=args.maximum_initial_screen_p90,
            allow_weak_fallback=(
                incumbent_p90 is None
                and not weak_initial_fallback_spent
                and screened >= weak_fallback_boundary
            ),
        )
        if selection is None:
            leader = next(
                (item for item in ranked_sequences if item not in repeated_finalists),
                None,
            )
            if leader is not None:
                if leader not in deferred_weak_leaders:
                    deferred_weak_leaders.append(leader)
                print(
                    f"EXACT_INITIAL_REPEAT_DEFERRED screen_p90="
                    f"{float(screen_candidates[leader]['score_p90']):.1f} "
                    f"ceiling={args.maximum_initial_screen_p90:.1f} "
                    f"screened={screened}/{weak_fallback_boundary}",
                    flush=True,
                )
            continue
        sequence, weak_fallback = selection
        if weak_fallback:
            weak_initial_fallback_spent = True
        finalist_index = len(repeated_finalists) + 1
        screen_p90 = float(
            screen_candidates[sequence].get(
                "basin_adjusted_score", screen_candidates[sequence]["score_p90"]
            )
        )
        if incumbent_p90 is not None and not _challenger_can_overtake(
            screen_p90=screen_p90,
            incumbent_p90=incumbent_p90,
            margin=args.challenger_p90_margin,
        ):
            if sequence not in skipped_finalists:
                skipped_finalists.append(sequence)
            print(
                f"EXACT_CHALLENGER_SKIPPED index={finalist_index} "
                f"screen_p90={screen_p90:.1f} incumbent_p90={incumbent_p90:.1f} "
                f"margin={args.challenger_p90_margin:.1f}",
                flush=True,
            )
            continue
        if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
            print("EXACT_RACE_RESERVE_REACHED before next finalist", flush=True)
            break
        if in_flight is not None:
            # Replication needs every GPU; finish the prefetched tranche first
            # and let the next loop iteration do its bookkeeping.
            in_flight.wait()
        repeated_finalists.append(sequence)
        repeats_started = time.monotonic()
        try:
            _score_repeats(
                sequence=sequence,
                repeat_numbers=list(range(2, args.qualification_observations + 1)),
                root=args.output_dir,
                config=args.config,
                target=args.target,
                gpus=gpus,
                scorer=args.scorer,
            )
        except subprocess.CalledProcessError as error:
            print(
                f"EXACT_REPEAT_FAILED index={finalist_index} "
                f"returncode={error.returncode}; continuing screen",
                flush=True,
            )
            continue
        _append_timing(
            timings_path,
            {
                "phase": "finalist_repeats",
                "finalist": finalist_index,
                "observations": args.qualification_observations - 1,
                "seconds": round(time.monotonic() - repeats_started, 1),
            },
        )
        output = args.output_dir / f"qualified_after_{finalist_index:02d}.json"
        if _try_strict_rank(
            root=args.output_dir,
            archive=args.archive,
            config=args.config,
            target=args.target,
            epoch=args.epoch,
            output=output,
            ledger=args.ledger,
        ):
            _store_ranking(
                args.mutation_database,
                output,
                epoch=args.epoch,
                target=args.target,
                source="live_exact_repeat",
                candidate_metadata=candidate_metadata,
            )
            qualified_outputs.append(output)
            incumbent_p90 = _best_confirmed_p90(output)
            print(
                f"EXACT_CANDIDATE_QUALIFIED index={finalist_index} "
                f"incumbent_p90={incumbent_p90:.1f} output={output}",
                flush=True,
            )
        else:
            print(f"EXACT_CANDIDATE_REJECTED index={finalist_index}", flush=True)

    _atomic_json(
        args.output_dir / "complete.json",
        {
            "epoch": args.epoch,
            "target": args.target,
            "screened": screened,
            "candidate_limit": len(sequences),
            "screen_tranche_size": args.screen_tranche_size,
            "maximum_finalists": args.finalists,
            "repeated_finalists": repeated_finalists,
            "skipped_finalists": skipped_finalists,
            "deferred_weak_leaders": deferred_weak_leaders,
            "challenger_p90_margin": args.challenger_p90_margin,
            "maximum_initial_screen_p90": args.maximum_initial_screen_p90,
            "weak_initial_fallback_after": args.weak_initial_fallback_after,
            "weak_initial_fallback_spent": weak_initial_fallback_spent,
            "basin_shrinkage_strength": args.basin_shrinkage_strength,
            "minimum_basin_samples": args.minimum_basin_samples,
            "incumbent_p90": incumbent_p90,
            "qualified_rankings": [str(path) for path in qualified_outputs],
            "exact_files": [str(path) for path in _exact_paths(args.output_dir)],
        },
    )
    print(
        f"EXACT_RACE_COMPLETE repeated={len(repeated_finalists)} "
        f"skipped={len(skipped_finalists)} "
        f"qualified={len(qualified_outputs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
