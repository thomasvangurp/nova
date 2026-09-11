"""Continuously generate, qualify, and exact-race live nanobody candidates.

The chain miner remains the source of current epoch/target state.  This process
owns only producer artifacts: it refreshes the public challenge inputs, builds
winner-informed local neighborhoods, runs locked Pixi Layer-2, and hands the
survivors to the candidate-major exact race.  Completed exact observations are
picked up atomically by the independent checkpoint publisher.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml


CONFIG_URL = "https://raw.githubusercontent.com/metanova-labs/nova/main/config/config.yaml"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/"
    "{target}_nanobodies.csv"
)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2) + "\n").encode())


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "nova-nanobody-producer/1"})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
    if len(payload) < 100:
        raise ValueError(f"download from {url} is unexpectedly small")
    _atomic_bytes(path, payload)


def _chain_state(path: Path, maximum_age: float) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    age = time.time() - float(payload["updated_at_unix"])
    if age < 0 or age > maximum_age:
        raise ValueError(f"chain state is stale by {age:.1f}s")
    block = int(payload["block"])
    epoch_length = int(payload["epoch_length"])
    epoch = int(payload["epoch"])
    if epoch_length < 1 or epoch != block // epoch_length:
        raise ValueError("chain state epoch is internally inconsistent")
    return {
        **payload,
        "block": block,
        "epoch_length": epoch_length,
        "epoch": epoch,
    }


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _prepare_epoch(
    *, root: Path, epoch: int, expected_target: str
) -> tuple[Path, Path, str]:
    epoch_root = root / "live" / f"epoch{epoch}" / "producer"
    config = epoch_root / "config.yaml"
    _download(CONFIG_URL, config)
    challenge = yaml.safe_load(config.read_text())["protein_selection"]["nanobody"]
    target = str(challenge["target"])
    if target != expected_target:
        raise ValueError(
            f"chain miner target {expected_target!r} disagrees with live config {target!r}"
        )
    archive = epoch_root / "archive.csv"
    _download(ARCHIVE_URL.format(target=target), archive)
    return config, archive, target


def _run_iteration(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    iteration: int,
) -> None:
    epoch = int(state["epoch"])
    target = str(state["nanobody_target"])
    config, archive, target = _prepare_epoch(
        root=args.runtime_root,
        epoch=epoch,
        expected_target=target,
    )
    producer_root = args.runtime_root / "live" / f"epoch{epoch}" / "producer"
    iteration_root = producer_root / f"iteration_{iteration:02d}"
    if (iteration_root / "race/complete.json").is_file():
        return
    winner_priors = producer_root / "winner_priors.json"
    if not winner_priors.is_file():
        _run([
            sys.executable,
            "-m", "nanobody_opt.analyze_epoch_winners",
            "--archive", str(archive),
            "--filename", f"{target}_nanobodies.csv",
            "--epochs", "40",
            "--output", str(winner_priors),
        ])
    elite_parents = producer_root / "elite_parents.json"
    if not elite_parents.is_file():
        _run([
            sys.executable,
            "-m", "nanobody_opt.select_elite_parents",
            "--target", target,
            "--before-epoch", str(epoch),
            "--epochs", str(args.parent_epochs),
            "--per-epoch", str(args.parents_per_epoch),
            "--limit", str(args.parent_limit),
            "--neighbor-per-epoch", str(args.neighbor_parents_per_epoch),
            "--neighbor-distance", str(args.neighbor_parent_distance),
            "--neighbor-limit", str(args.neighbor_parent_limit),
            "--terminal-variants-per-residue", str(
                args.terminal_variants_per_residue
            ),
            "--recent-epoch-window", str(args.recent_parent_epoch_window),
            "--recent-anchor-limit", str(args.recent_anchor_parent_limit),
            "--recent-neighbor-limit", str(args.recent_neighbor_parent_limit),
            "--scan-limit", str(args.parent_scan_limit),
            "--observation-database", str(args.mutation_database),
            "--output", str(elite_parents),
        ])
    seed = producer_root / "elite_parents_layer2.json"
    if not seed.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked",
            "--environment", "metanano",
            "filter-layer2", "--",
            "--input", str(elite_parents),
            "--config", str(config),
            "--output", str(seed),
            "--concurrency", str(args.layer2_concurrency),
            "--igblast-concurrency", str(args.layer2_concurrency),
            "--igblast-batch-size", "1",
            "--cache-dir", str(args.runtime_root / "layer2_cache"),
        ])
    candidates = iteration_root / "candidates.csv"
    if not candidates.is_file():
        command = [
            sys.executable,
            "-m", "nanobody_opt.local_neighborhood",
            "--archive", str(archive),
            "--config", str(config),
            "--layer2-seeds", str(seed),
            "--winner-priors", str(winner_priors),
            "--output", str(candidates),
            "--seed-count", str(args.seed_count),
            "--neighbor-seed-count", str(args.neighbor_seed_count),
            "--minimum-seeds", str(args.minimum_seeds),
            "--per-seed", str(args.per_seed),
            "--proposal-limit", str(args.proposal_limit),
            "--framework-fraction", str(args.framework_fraction),
            "--framework-per-seed", str(args.framework_per_seed),
            "--terminal-width", str(args.terminal_width),
            "--maximum-terminal-haplotype-mutations", str(
                args.maximum_terminal_haplotype_mutations
            ),
            "--mutation-observations", str(args.mutation_database),
            "--motif-fraction", str(args.motif_fraction),
            "--single-mutation-fraction", str(args.single_mutation_fraction),
            "--maximum-motif-mutations", str(args.maximum_motif_mutations),
            "--maximum-exhaustive-singles", str(
                args.maximum_exhaustive_singles
            ),
            "--random-seed", str(epoch * 100 + iteration),
        ]
        for previous in sorted(producer_root.glob("iteration_*/candidates.csv")):
            if previous != candidates:
                command.extend(["--exclude", str(previous)])
        _run(command)
    layer2 = iteration_root / "layer2.json"
    if not layer2.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked",
            "--environment", "metanano",
            "filter-layer2", "--",
            "--input", str(candidates),
            "--config", str(config),
            "--output", str(layer2),
            "--concurrency", str(args.layer2_concurrency),
            "--igblast-concurrency", str(args.layer2_concurrency),
            "--igblast-batch-size", "1",
            "--cache-dir", str(args.runtime_root / "layer2_cache"),
        ])
    _run([
        sys.executable,
        "-m", "nanobody_opt.sequential_exact_race",
        "--layer2", str(layer2),
        "--archive", str(archive),
        "--config", str(config),
        "--target", target,
        "--epoch", str(epoch),
        "--output-dir", str(iteration_root / "race"),
        "--ledger", str(args.ledger),
        "--mutation-database", str(args.mutation_database),
        "--chain-state", str(args.chain_state),
        "--reserve-blocks", str(args.reserve_blocks),
        "--gpus", args.gpus,
        "--candidate-limit", str(args.candidate_limit),
        "--screen-tranche-size", str(args.screen_tranche_size),
        "--finalists", str(args.finalists),
        "--challenger-p90-margin", str(args.challenger_p90_margin),
    ])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument(
        "--chain-state",
        type=Path,
        default=Path("/root/artifacts/chain_state.json"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("/root/artifacts/nanobody_published_sequences.txt"),
    )
    parser.add_argument(
        "--mutation-database",
        type=Path,
        default=Path("/root/artifacts/nanobody_mutations.sqlite"),
    )
    parser.add_argument("--maximum-chain-state-age", type=float, default=90)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--parent-epochs",
        type=int,
        default=0,
        help="zero keeps every historical epoch in the mutation database",
    )
    parser.add_argument("--parents-per-epoch", type=int, default=2)
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--parent-scan-limit", type=int, default=40)
    parser.add_argument("--neighbor-parents-per-epoch", type=int, default=8)
    parser.add_argument("--neighbor-parent-distance", type=int, default=6)
    parser.add_argument("--neighbor-parent-limit", type=int, default=32)
    parser.add_argument("--terminal-variants-per-residue", type=int, default=3)
    parser.add_argument("--recent-parent-epoch-window", type=int, default=16)
    parser.add_argument("--recent-anchor-parent-limit", type=int, default=16)
    parser.add_argument("--recent-neighbor-parent-limit", type=int, default=16)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--neighbor-seed-count", type=int, default=32)
    parser.add_argument("--minimum-seeds", type=int, default=8)
    parser.add_argument(
        "--per-seed",
        type=int,
        default=12,
        help="proposals generated per elite parent before Layer-2 filtering",
    )
    parser.add_argument("--proposal-limit", type=int, default=384)
    parser.add_argument("--framework-fraction", type=float, default=0.5)
    parser.add_argument("--framework-per-seed", type=int, default=64)
    parser.add_argument("--terminal-width", type=int, default=17)
    parser.add_argument(
        "--maximum-terminal-haplotype-mutations", type=int, default=6
    )
    parser.add_argument("--motif-fraction", type=float, default=0.25)
    parser.add_argument("--single-mutation-fraction", type=float, default=0.75)
    parser.add_argument("--maximum-motif-mutations", type=int, default=6)
    parser.add_argument("--maximum-exhaustive-singles", type=int, default=8192)
    parser.add_argument("--layer2-concurrency", type=int, default=40)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=320,
        help="maximum exact screens per iteration, published in small tranches",
    )
    parser.add_argument("--screen-tranche-size", type=int, default=20)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--challenger-p90-margin", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    completed_iterations: dict[int, int] = {}
    while True:
        try:
            state = _chain_state(args.chain_state, args.maximum_chain_state_age)
            epoch = int(state["epoch"])
            offset = int(state["block"]) % int(state["epoch_length"])
            if offset >= int(state["epoch_length"]) - args.reserve_blocks:
                print(
                    f"PRODUCER_RESERVE epoch={epoch} offset={offset}; waiting",
                    flush=True,
                )
            else:
                iteration = completed_iterations.get(epoch, 0) + 1
                print(
                    f"PRODUCER_ITERATION epoch={epoch} offset={offset} "
                    f"iteration={iteration}",
                    flush=True,
                )
                try:
                    _run_iteration(args=args, state=state, iteration=iteration)
                except subprocess.CalledProcessError as error:
                    failure = (
                        args.runtime_root / "live" / f"epoch{epoch}" / "producer"
                        / f"iteration_{iteration:02d}" / "failed.json"
                    )
                    _atomic_json(failure, {
                        "epoch": epoch,
                        "iteration": iteration,
                        "returncode": error.returncode,
                        "command": error.cmd,
                    })
                    completed_iterations[epoch] = iteration
                    print(f"PRODUCER_ITERATION_FAILED {error}", flush=True)
                else:
                    completed_iterations[epoch] = iteration
            if args.once:
                return
        except Exception as error:
            print(
                f"PRODUCER_RETRY {type(error).__name__}: {error}",
                flush=True,
            )
            if args.once:
                raise
        time.sleep(max(1.0, args.poll_interval))


if __name__ == "__main__":
    main()
