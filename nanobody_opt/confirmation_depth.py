"""Epoch-scoped evidence policy; eligibility and submission clocks stay hard."""
from __future__ import annotations

import json
from pathlib import Path

POLICY = "deadline-depth-v1"
TARGET = 20
FALLBACK = 10
COLD_GATE_SECONDS = 120.0
CACHED_GATE_SECONDS = 30.0


def add_arguments(parser):
    parser.add_argument("--adaptive-confirmation-depth", action="store_true",
                        help="prefer twenty observations; allow ten when the deadline cannot fit the target pool")


def decision(*, epoch, target, minimum, schedule):
    if minimum not in (TARGET, FALLBACK):
        raise ValueError("unsupported confirmation depth")
    return {"policy": POLICY, "epoch": epoch, "target": target,
            "minimum_observations": minimum, "preferred_observations": TARGET,
            "reason": "deadline" if minimum == FALLBACK else "target",
            "scheduled_repeat_rounds": sum(r for _, r in schedule)}


def read(root: Path, *, epoch: int, target: str) -> dict | None:
    try:
        value = json.loads((root / "confirmation_depth.json").read_text())
        return validate(value, epoch=epoch, target=target)
    except (OSError, ValueError, TypeError):
        return None


def validate(value, *, epoch: int, target: str) -> dict | None:
    try:
        minimum = value["minimum_observations"]
        if (value.get("policy") != POLICY or value.get("epoch") != epoch
                or value.get("target") != target or minimum not in (TARGET, FALLBACK)
                or value.get("preferred_observations") != TARGET
                or not isinstance(value.get("scheduled_repeat_rounds"), int)
                or value["scheduled_repeat_rounds"] < 0
                or value.get("reason") != ("deadline" if minimum == FALLBACK else "target")):
            return None
        return {k: value[k] for k in ("policy", "epoch", "target", "minimum_observations",
                                    "preferred_observations", "reason", "scheduled_repeat_rounds")}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
