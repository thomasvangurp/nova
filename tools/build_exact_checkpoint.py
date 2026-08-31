#!/usr/bin/env python3
"""Run reduced Boltz acquisition, exact reranking, and atomic qualification."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORE_TOOL = ROOT / "tools" / "score_boltz_batch.py"
sys.path.insert(0, str(ROOT))

from compound_miner.archive import CandidateArchive
from compound_miner.release import (
    VALIDATOR_EXACT_SCORER,
    scored_candidates_from_payload,
    sha256_file,
)


def _run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def _write_exact_union(
    archive_path: Path,
    current_path: Path,
    destination: Path,
    reaction: int,
    target: str,
) -> int:
    """Preserve exact incumbents while adding the newly reranked shortlist."""
    current_payload = json.loads(current_path.read_text())
    if str(current_payload.get("target")) != target:
        raise ValueError("new exact score artifact target does not match challenge")
    current = scored_candidates_from_payload(current_payload)
    incumbents = CandidateArchive(archive_path).top(
        reaction,
        limit=1000,
        scorer=VALIDATOR_EXACT_SCORER,
    )
    by_name = {item.candidate.name: item for item in incumbents}
    by_name.update({item.candidate.name: item for item in current})
    combined = sorted(
        by_name.values(), key=lambda item: (-item.score, item.candidate.name)
    )
    payload = {
        "reaction": reaction,
        "target": target,
        "count": len(combined),
        "scorer": VALIDATOR_EXACT_SCORER,
        "sources": {
            "new_exact_sha256": sha256_file(current_path),
            "incumbent_archive": str(archive_path.resolve()),
            "incumbent_count": len(incumbents),
        },
        "scores": [item.as_dict() for item in combined],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name,
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--archive", type=Path, default=Path("archives/P40261.sqlite"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--baselines", type=Path, default=Path("config/compound_release_baselines.json")
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--shards-per-gpu", type=int, default=3)
    parser.add_argument("--preprocessing-threads", type=int, default=8)
    parser.add_argument("--reduced-limit", type=int, default=512)
    parser.add_argument("--exact-limit", type=int, default=128)
    parser.add_argument(
        "--exclude-records-json",
        action="append",
        type=Path,
        help="Do not spend reduced-scoring work on an earlier iteration's portfolio",
    )
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    reduced = args.work_dir / f"rxn{args.reaction}_reduced.json"
    exact_new = args.work_dir / f"rxn{args.reaction}_exact_new.json"
    exact = args.work_dir / f"rxn{args.reaction}_exact.json"
    common = [
        "--gpus",
        args.gpus,
        "--shards-per-gpu",
        str(args.shards_per_gpu),
        "--preprocessing-threads",
        str(args.preprocessing_threads),
    ]
    if args.scratch_root:
        common.extend(("--scratch-root", str(args.scratch_root)))

    reduced_arguments = [
        sys.executable,
        str(SCORE_TOOL),
        "--archive",
        str(args.archive),
        "--reaction",
        str(args.reaction),
        "--target",
        args.target,
        "--limit",
        str(args.reduced_limit),
        "--recycling-steps",
        "1",
        "--sampling-steps",
        "20",
        "--diffusion-samples",
        "1",
        "--recycling-steps-affinity",
        "5",
        "--sampling-steps-affinity",
        "20",
        "--diffusion-samples-affinity",
        "1",
        "--skip-affinity-structure",
        "--output",
        str(reduced),
        *common,
    ]
    for path in args.exclude_records_json or []:
        reduced_arguments.extend(("--exclude-records-json", str(path)))
    _run(reduced_arguments)
    _run(
        [
            sys.executable,
            str(SCORE_TOOL),
            "--archive",
            str(args.archive),
            "--reaction",
            str(args.reaction),
            "--target",
            args.target,
            "--records-json",
            str(reduced),
            "--limit",
            str(args.exact_limit),
            "--recycling-steps",
            "3",
            "--sampling-steps",
            "100",
            "--diffusion-samples",
            "1",
            "--recycling-steps-affinity",
            "5",
            "--sampling-steps-affinity",
            "100",
            "--diffusion-samples-affinity",
            "3",
            "--output",
            str(exact_new),
            *common,
        ]
    )
    exact_count = _write_exact_union(
        args.archive,
        exact_new,
        exact,
        args.reaction,
        args.target,
    )
    qualification = [
        sys.executable,
        "-m",
        "compound_miner.cli",
        "qualify-exact",
        "--scores",
        str(exact),
        "--reduced-scores",
        str(reduced),
        "--archive",
        str(args.archive),
        "--result",
        str(args.result),
        "--baselines",
        str(args.baselines),
        "--reaction",
        str(args.reaction),
        "--target",
        args.target,
        "--epoch",
        str(args.epoch),
    ]
    _run(qualification)
    print(
        json.dumps(
            {
                "reaction": args.reaction,
                "target": args.target,
                "epoch": args.epoch,
                "reduced": str(reduced.resolve()),
                "exact": str(exact.resolve()),
                "exact_new": str(exact_new.resolve()),
                "exact_count": exact_count,
                "result": str(args.result.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
