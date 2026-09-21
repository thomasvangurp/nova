"""Where the screen budget goes, once old folds stop being thrown away.

The screen currently spends ~1,300 folds an epoch re-measuring a pool it
then ranks on a single fold each. Two thirds of those folds land on
sequences already measured ten or more times -- we buy a 23rd observation
and rank on the 1st. Meanwhile brand new proposals are ranked on n=1, which
is precisely where the winner's curse is worst: our submitted entries rank
*worse* than the field median (133 against 116).

The fix is not to fold less. It is to spend the same budget on candidates
that need evidence, and to refuse to select on candidates that do not have
any.

TWO RULES.

1. NEVER SELECT ON n=1. A single fold has reliability 0.373; ranking on it
   is close to ranking on noise, and picking the maximum of such estimates
   systematically selects the largest error. A candidate is not eligible for
   the shortlist until it has MINIMUM_SELECTABLE observations.

2. KEEP THE POOL HEALTHY. Budget goes first to lifting candidates to that
   threshold, then to deepening the contenders. Sequences that can no longer
   be submitted -- the validator's novelty gate burns anything already
   published -- are retired rather than re-measured, because a fold spent on
   them buys nothing.

Deepening past a point is also waste: reliability is 0.960 at forty folds
and 0.963 at fifty. DEEP_ENOUGH stops the budget disappearing into
candidates that are already as well known as they will ever be.
"""
from __future__ import annotations

#: A candidate below this is not rankable; one fold is reliability 0.373.
MINIMUM_SELECTABLE = 3

#: Past this, another fold changes the estimate by less than a rounding error
#: (R(40)=0.960, R(50)=0.963), so the budget is better spent elsewhere.
DEEP_ENOUGH = 40


def eligible(counts: dict[str, int],
             minimum: int = MINIMUM_SELECTABLE) -> set[str]:
    """Candidates with enough evidence to be ranked at all."""
    return {key for key, count in counts.items() if count >= minimum}


def allocate(counts: dict[str, int], budget: int, *,
             promise: dict[str, float] | None = None,
             retired: set[str] | None = None,
             minimum: int = MINIMUM_SELECTABLE,
             deep_enough: int = DEEP_ENOUGH) -> dict[str, int]:
    """Plan this epoch's folds: {candidate: folds to run}.

    Budget is spent in two passes. The first lifts every live candidate to
    `minimum`, because a candidate below it cannot be selected and is
    therefore worth nothing until it gets there. Whatever remains deepens
    the most promising candidates that are not already deep enough, lowest
    count first so the shallowest contender gains the most reliability per
    fold.

    `promise` orders that second pass; without it the order is by current
    depth alone. `retired` names candidates that can no longer be submitted,
    which receive nothing.
    """
    if budget <= 0:
        return {}
    retired = retired or set()
    promise = promise or {}
    plan: dict[str, int] = {}
    remaining = int(budget)

    live = {k: c for k, c in counts.items() if k not in retired}

    # Pass 1 -- make candidates selectable at all.
    needy = sorted((c, k) for k, c in live.items() if c < minimum)
    for count, key in needy:
        if remaining <= 0:
            break
        take = min(minimum - count, remaining)
        if take > 0:
            plan[key] = plan.get(key, 0) + take
            remaining -= take

    # Pass 2 -- deepen contenders, shallowest first.
    if remaining > 0:
        def order(item):
            key, count = item
            return (-promise.get(key, 0.0), count + plan.get(key, 0))
        candidates = [(k, c) for k, c in live.items()
                      if c + plan.get(k, 0) < deep_enough]
        for key, count in sorted(candidates, key=order):
            if remaining <= 0:
                break
            have = count + plan.get(key, 0)
            take = min(deep_enough - have, remaining)
            if take > 0:
                plan[key] = plan.get(key, 0) + take
                remaining -= take
    return plan


def health(counts: dict[str, int], *, minimum: int = MINIMUM_SELECTABLE,
           retired: set[str] | None = None) -> dict[str, float]:
    """A pool's readiness to be selected from, for logging and alerting."""
    retired = retired or set()
    live = {k: c for k, c in counts.items() if k not in retired}
    if not live:
        return {"candidates": 0, "selectable": 0, "selectable_fraction": 0.0,
                "median_observations": 0.0, "wasted_on_retired": 0}
    values = sorted(live.values())
    selectable = sum(1 for c in values if c >= minimum)
    return {
        "candidates": len(values),
        "selectable": selectable,
        "selectable_fraction": selectable / len(values),
        "median_observations": float(values[len(values) // 2]),
        "wasted_on_retired": sum(counts[k] for k in retired if k in counts),
    }
