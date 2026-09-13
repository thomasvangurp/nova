"""Repeat-qualify a novel leader from a completed one-shot screen.

This is a recovery path for a screen that crossed an epoch boundary before the
candidate-major repeats ran.  The original screen remains observation one;
seven isolated scorer processes create observations two through eight.  The
independent publisher discovers the completed ``exact_all.json`` files and
applies every strict qualification and current-epoch novelty guard.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nanobody_opt.sequential_exact_race import (
    _atomic_json,
    _chain_allows_work,
    _score_repeats,
)


def visible_gpu_ids() -> str:
    """The GPU ids this machine actually has. See run_live_producer for why the
    old hardcoded ten-device list silently lost eight shards in ten."""
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return ",".join(str(i) for i in range(count)) if count >= 1 else "0"


def _known_sequences(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return {
                str(row.get("sequence", "")).strip().upper()
                for row in csv.DictReader(handle)
                if row.get("sequence")
            }
    return {
        line.strip().upper()
        for line in path.read_text().splitlines()
        if line.strip()
    }


def _leader(path: Path) -> str:
    payload = json.loads(path.read_text())
    sequences = payload.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError(f"ranking has no sequences: {path}")
    sequence = str(sequences[0]).strip().upper()
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("screen leader is not a canonical protein sequence")
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chain-state", required=True, type=Path)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated CUDA device ids; defaults to every device present",
    )
    parser.add_argument("--qualification-observations", type=int, default=8)
    args = parser.parse_args()
    if not args.gpus:
        args.gpus = visible_gpu_ids()
    if args.qualification_observations < 8:
        parser.error("qualification requires at least eight observations")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        parser.error("at least one GPU is required")
    if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
        raise SystemExit("current chain epoch is inside the scoring reserve window")

    sequence = _leader(args.ranking)
    if sequence in _known_sequences(args.archive):
        raise SystemExit("screen leader already exists in the public archive")
    if sequence in _known_sequences(args.ledger):
        raise SystemExit("screen leader already exists in the publication ledger")
    complete = args.output_dir / "complete.json"
    if complete.is_file():
        print(f"SCREEN_LEADER_RESCUE_RESUMED output={complete}")
        return
    _score_repeats(
        sequence=sequence,
        repeat_numbers=list(range(2, args.qualification_observations + 1)),
        root=args.output_dir,
        config=args.config,
        target=args.target,
        gpus=gpus,
    )
    _atomic_json(complete, {
        "epoch": args.epoch,
        "target": args.target,
        "sequence": sequence,
        "source_ranking": str(args.ranking),
        "observation_numbers": list(range(2, args.qualification_observations + 1)),
    })
    print(
        f"SCREEN_LEADER_RESCUE_READY epoch={args.epoch} "
        f"observations={args.qualification_observations - 1} output={complete}"
    )


if __name__ == "__main__":
    main()
