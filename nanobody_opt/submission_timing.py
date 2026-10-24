"""The Pi's submission clock, shared by scoring and publication.

Reserves are measured backwards from the end of the owning epoch. Scoring
must leave time for publication, and publication must leave time for the Pi
to fetch the artifact. None of these windows borrows time from another epoch.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

POLICY = "pi-handoff-v2"
BLOCK_SECONDS = 12
BLACKOUT_BLOCKS = 16
BLOCKS_PER_UID = 5
LATE_COMMIT_MARGIN_SECONDS = 60
PARALLEL_COMMIT_BLOCKS = 20
PARALLEL_BATCH_UIDS = 20
TRANSFER_SECONDS = 60
# Epoch 25914 took ~246 seconds from completed confirmation to final gates.
# Allow another publisher polling interval and margin; do not fit to one run.
PUBLICATION_SECONDS = 360
FLEET_SIZE = 13


def commit_reserve(uids: int, *, serial: bool = False) -> int:
    if serial:
        return BLACKOUT_BLOCKS + BLOCKS_PER_UID * max(1, uids)
    # Recent 12–13-ticket batches took 78–98 seconds (no repairs). Four
    # minutes leaves room for retries without paying a serial cost per UID.
    batches = math.ceil(max(1, uids) / PARALLEL_BATCH_UIDS)
    return BLACKOUT_BLOCKS + PARALLEL_COMMIT_BLOCKS * batches


def plan(crew_size: int = FLEET_SIZE, *, late_margin_seconds: float = LATE_COMMIT_MARGIN_SECONDS,
         publication_seconds: float = PUBLICATION_SECONDS,
         transfer_seconds: float = TRANSFER_SECONDS) -> dict:
    if crew_size < 1 or min(late_margin_seconds, publication_seconds, transfer_seconds) < 0:
        raise ValueError("submission timing requires positive fleet size and nonnegative margins")
    commit = commit_reserve(crew_size)
    pull = commit + math.ceil(late_margin_seconds / BLOCK_SECONDS)
    publish = pull + math.ceil(transfer_seconds / BLOCK_SECONDS)
    scoring = publish + math.ceil(publication_seconds / BLOCK_SECONDS)
    return {"policy": POLICY, "crew_size": crew_size,
            "commit_reserve_blocks": commit, "pi_pull_reserve_blocks": pull,
            "publication_reserve_blocks": publish, "scoring_reserve_blocks": scoring,
            "publication_prepare_blocks": scoring - publish,
            "late_commit_margin_seconds": late_margin_seconds,
            "transfer_seconds": transfer_seconds}


def add_arguments(parser) -> None:
    parser.add_argument("--pi-handoff", action="store_true",
                        help="finish scoring and publication before the Pi's final pull")
    parser.add_argument("--submission-fleet-size", type=int, default=FLEET_SIZE,
                        help="fleet capacity reserved even before registrations propagate")
    parser.add_argument("--submission-late-margin-seconds", type=float,
                        default=LATE_COMMIT_MARGIN_SECONDS)


def configured_plan(args, crew_size: int = 0) -> dict | None:
    if not getattr(args, "pi_handoff", False):
        return None
    size = max(crew_size, args.submission_fleet_size)
    # The publisher observes live registrations. A previously larger fleet is
    # a conservative capacity bound until it publishes the current view.
    if not crew_size:
        try:
            observed = json.loads((args.runtime_root / "artifacts/submission_handoff.json").read_text())
            if isinstance(observed, dict) and observed.get("policy") == POLICY:
                size = max(size, int(observed["crew_size"]))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return plan(size, late_margin_seconds=args.submission_late_margin_seconds,
                publication_seconds=max(PUBLICATION_SECONDS,
                    getattr(args, "publication_prepare_blocks", 0) * BLOCK_SECONDS))


def confirmation_digest(runtime_root: Path, epoch: int) -> str | None:
    path = runtime_root / "live" / f"epoch{epoch}" / "producer/full_scan_confirmation.json"
    try:
        raw = path.read_bytes()
        if int(json.loads(raw)["epoch"]) != epoch:
            return None
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, KeyError, TypeError):
        return None


def publication_receipt(*, epoch: int, timing: dict, confirmation: str | None,
                        block: int, epoch_length: int) -> dict:
    return {**timing, "epoch": epoch, "published_at_unix": time.time(),
            "published_block": block,
            "publish_by_block": (epoch + 1) * epoch_length - timing["publication_reserve_blocks"],
            "pi_pull_by_block": (epoch + 1) * epoch_length - timing["pi_pull_reserve_blocks"],
            "confirmation_sha256": confirmation,
            "confirmation_complete_at_start": confirmation is not None}


def artifact_receipt(path: Path, epoch: int) -> dict:
    """Allowlisted proof of the exact pulled input; never expose sequences."""
    try:
        raw = path.read_bytes()
        artifact = json.loads(raw)
        if not isinstance(artifact, dict):
            return {"confirmation_complete_at_start": False, "artifact_readable": False}
        receipt = artifact.get("submission_handoff") or {}
        if not isinstance(receipt, dict):
            receipt = {}
        valid = artifact.get("epoch") == epoch and receipt.get("epoch") == epoch
        return {"artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "artifact_epoch": artifact.get("epoch"),
                "handoff_policy": receipt.get("policy"),
                "confirmation_complete_at_start": bool(valid and receipt.get("confirmation_complete_at_start")),
                "confirmation_sha256": receipt.get("confirmation_sha256") if valid else None,
                "published_block": receipt.get("published_block") if valid else None,
                "pi_pull_by_block": receipt.get("pi_pull_by_block") if valid else None}
    except (OSError, ValueError, TypeError):
        return {"confirmation_complete_at_start": False, "artifact_readable": False}
