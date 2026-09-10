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
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


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


def _candidate_sequences(path: Path, limit: int) -> list[str]:
    payload = json.loads(path.read_text())
    items = payload if isinstance(payload, list) else (
        payload.get("candidates") or payload.get("sequences") or []
    )
    if items and all(isinstance(item, dict) for item in items):
        items = sorted(items, key=lambda item: (
            0
            if item.get("proposal_type") == "exhaustive_single_substitution"
            else 1
        ))
    sequences = []
    for item in items:
        if isinstance(item, dict) and item.get("layer2_passed") is False:
            continue
        sequence = item.get("sequence") if isinstance(item, dict) else item
        if isinstance(sequence, str) and sequence.strip():
            sequences.append(sequence.strip().upper())
    return list(dict.fromkeys(sequences))[:limit]


def _batches(items: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


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
        "--limit", "30",
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


def _run_screen(
    *,
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    candidate_limit: int,
) -> Path:
    completed = output_dir / "exact_all.json"
    if completed.is_file():
        return completed
    command = [
        sys.executable,
        "-m",
        "nanobody_opt.score_boltzgen_waves",
        "--sequences", str(source),
        "--config", str(config),
        "--target", target,
        "--output-dir", str(output_dir),
        "--gpus", ",".join(gpus),
        "--wave-size", str(len(gpus)),
        "--predict-batch-size", "1",
        "--limit", str(candidate_limit),
        "--capture-structures",
    ]
    subprocess.run(command, check=True)
    return completed


def _repeat_command(
    *,
    input_path: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpu: str,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer2", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--chain-state", type=Path)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--candidate-limit", type=int, default=160)
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
    parser.add_argument("--qualification-observations", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.finalists <= args.candidate_limit:
        parser.error("require 1 <= --finalists <= --candidate-limit")
    if not 1 <= args.screen_tranche_size <= args.candidate_limit:
        parser.error("require 1 <= --screen-tranche-size <= --candidate-limit")
    if args.qualification_observations < 8:
        parser.error("qualification requires at least eight observations")
    if args.challenger_p90_margin < 0:
        parser.error("--challenger-p90-margin cannot be negative")
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("at least one GPU is required")
    sequences = _candidate_sequences(args.layer2, args.candidate_limit)
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
    incumbent_p90: float | None = None
    screened = 0
    for tranche_index, tranche in enumerate(
        _batches(sequences, args.screen_tranche_size), start=1
    ):
        if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
            print("EXACT_RACE_RESERVE_REACHED before next screen tranche", flush=True)
            break
        tranche_root = args.output_dir / "screen" / f"tranche_{tranche_index:02d}"
        source = tranche_root / "candidates.json"
        _atomic_json(source, {"target": args.target, "sequences": tranche})
        screen_exact_paths.append(_run_screen(
            source=source,
            config=args.config,
            target=args.target,
            output_dir=tranche_root,
            gpus=gpus,
            candidate_limit=len(tranche),
        ))
        screened += len(tranche)
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
        screen_candidates = _ranking_by_sequence(screen_ranking)
        ranked_sequences = json.loads(screen_ranking.read_text())["sequences"]
        print(
            f"EXACT_SCREEN_TRANCHE_COMPLETE tranche={tranche_index} "
            f"screened={screened}/{len(sequences)} leader={ranked_sequences[0]}",
            flush=True,
        )
        if len(repeated_finalists) >= args.finalists:
            continue
        sequence = next(
            (item for item in ranked_sequences if item not in repeated_finalists),
            None,
        )
        if sequence is None:
            continue
        finalist_index = len(repeated_finalists) + 1
        screen_p90 = float(screen_candidates[sequence]["score_p90"])
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
        repeated_finalists.append(sequence)
        try:
            _score_repeats(
                sequence=sequence,
                repeat_numbers=list(range(2, args.qualification_observations + 1)),
                root=args.output_dir,
                config=args.config,
                target=args.target,
                gpus=gpus,
            )
        except subprocess.CalledProcessError as error:
            print(
                f"EXACT_REPEAT_FAILED index={finalist_index} "
                f"returncode={error.returncode}; continuing screen",
                flush=True,
            )
            continue
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
            "challenger_p90_margin": args.challenger_p90_margin,
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
