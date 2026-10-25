"""Budget-preserving, broader confirmation of noisy first observations.

All counts here are repeat evaluations. Discovery retains its own NEW-only
ledger. The control schedule supplies both an evaluation cap and a wall-time
cap; widening must fit both, including whole-GPU rounding.
"""
from __future__ import annotations

import hashlib
import math

from nanobody_opt_v2.compute_budget import staged_schedule

POLICY = "broad-confirmation-v1"


def evaluation_count(schedule):
    return sum(width * rounds for width, rounds in schedule)


def seconds(schedule, model, gpus):
    return sum(rounds * model.round_seconds(width, gpus) for width, rounds in schedule)


def plan(*, candidates, rounds, rounds_per_stage, keep_fraction,
         minimum_keep, gpus, model):
    """Give a wider pool 3 total observations, then 8, then full depth.

    The middle width retains five eighths of the original shortlist. Both
    cuts occur later than a two-pass screen, which failed the replay audit.
    Small/short control schedules retain their original allocation.
    """
    if min(candidates, rounds, minimum_keep, gpus) < 1:
        raise ValueError("positive confirmation dimensions required")
    control = staged_schedule(candidates, rounds, keep_fraction, minimum_keep,
                              rounds_per_stage)
    budget = evaluation_count(control)
    wall = seconds(control, model, gpus)
    schedule = control
    floor = min(candidates, minimum_keep)
    if rounds > 7 and candidates > floor:
        middle = min(candidates, max(floor, int(candidates * .625) // gpus * gpus))
        tail = [(middle, 5), (floor, rounds - 7)]
        width = min(4 * candidates, (budget - evaluation_count(tail)) // 2)
        while width > candidates and seconds([(width, 2), *tail], model, gpus) > wall + 1e-7:
            width -= 1
        if width > candidates:
            schedule = [(width, 2), *tail]
    return {"policy": POLICY, "control_width": candidates,
            "control_schedule": control, "schedule": schedule,
            "evaluation_cap": budget, "planned_evaluations": evaluation_count(schedule),
            "seconds_cap": wall, "planned_seconds": seconds(schedule, model, gpus),
            "broadened": schedule[0][0] > candidates}


def cap_widths(schedule, available):
    return [(min(width, available), rounds) for width, rounds in schedule] if available else []


def fit(schedule, *, available, usable_seconds, model, gpus):
    """Narrow an existing plan after eligibility checks consume time.

    Remove breadth before depth; never lower the required repeat count.
    An empty result explicitly means even one complete finalist cannot fit.
    """
    result = cap_widths(schedule, available)
    while result and seconds(result, model, gpus) > usable_seconds:
        index = next((i for i in range(len(result) - 1)
                      if result[i][0] > result[i + 1][0]), len(result) - 1)
        width, rounds = result[index]
        if width == 1:
            return []
        result[index] = (width - 1, rounds)
    return result


def fit_depth(schedule, *, available, usable_seconds, model, gpus, minimum_keep):
    """Prefer full depth for the target pool, then ten total observations.

    Shortening only the tail retains the existing early pruning and never
    exceeds the original evaluation or time cap. Nine repeats plus the
    independent screen are ten observations; fewer are never a fallback.
    """
    from nanobody_opt.confirmation_depth import TARGET, FALLBACK
    preferred = fit(schedule, available=available, usable_seconds=usable_seconds,
                    model=model, gpus=gpus)
    if (preferred and sum(r for _, r in preferred) >= TARGET - 1
            and preferred[-1][0] >= min(minimum_keep, available)):
        return preferred, TARGET
    remaining = FALLBACK - 1
    shorter = []
    for width, rounds in schedule:
        take = min(remaining, rounds)
        if take:
            shorter.append((width, take))
            remaining -= take
    if remaining:
        return [], FALLBACK
    fallback = fit(shorter, available=available, usable_seconds=usable_seconds,
                   model=model, gpus=gpus)
    return fallback, FALLBACK


def shortlist(ordered, *, width, metadata, observations, epoch, broadened):
    """Keep the leading bulk plus small, deterministic second-chance samples.

    Only supplied eligible candidates can enter. One sixteenth of a wider
    pool is reserved: half by measured parent merit, half uniformly by an
    epoch-keyed hash. Parent evidence grants a repeat, never publication.
    """
    ordered = list(dict.fromkeys(ordered))
    width = min(width, len(ordered))
    rescue = width // 16 if broadened and len(ordered) > width else 0
    selected = ordered[:width - rescue]
    reasons = {sequence: "score" for sequence in selected}
    outside = ordered[width - rescue:]
    positions = {sequence: index for index, sequence in enumerate(ordered)}

    def sparse(sequence):
        return 0 < observations.get(sequence, 1) < 8

    def merit(sequence):
        try:
            value = float(metadata.get(sequence, {}).get("parent_multipass_mean"))
            return value if math.isfinite(value) else math.inf
        except (TypeError, ValueError):
            return math.inf

    parents = sorted((s for s in outside if sparse(s) and math.isfinite(merit(s))),
                     key=lambda s: (merit(s), positions[s]))
    for sequence in parents[:rescue // 2]:
        selected.append(sequence)
        reasons[sequence] = "parent_rescue"
    remaining = [s for s in outside if s not in reasons and sparse(s)]
    remaining.sort(key=lambda s: hashlib.sha256(f"{POLICY}:{epoch}:{s}".encode()).digest())
    for sequence in remaining[:width - len(selected)]:
        selected.append(sequence)
        reasons[sequence] = "random_audit"
    for sequence in ordered:
        if len(selected) >= width:
            break
        if sequence not in reasons:
            selected.append(sequence)
            reasons[sequence] = "score"
    return selected, reasons
