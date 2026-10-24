"""Fill confirmation seats with gate passes within a separate CPU/TNP window.

The caller still runs the full live gate, including for cache hits. A cached
profile is a way to avoid recomputing TNP, never permission to bypass a gate.
"""
from __future__ import annotations

import time

VERSION = "eligible-refill-v1"


def fill(*, ordered, initial, width, cached, gate, seconds_left,
         batch_size=32, fallback_seconds=30.0, batch_seconds=60.0):
    """Try ranked proposals, then refill from measured passes before the cutoff.

    ``seconds_left`` excludes the complete repeat schedule and its finish
    guard. Both uncached work and the final cached pass are hard-time-limited
    by the caller's process-group runner. Unavailable results never pass.
    """
    queue = list(dict.fromkeys([*initial, *ordered]))
    selected, attempted, batches = [], set(), []
    initial_set = set(initial)
    known_failures = {s for s in queue if cached(s) is False}
    queue = [s for s in queue if s not in known_failures]
    deadline_limited = False

    def check(batch, timeout, mode):
        start = time.monotonic()
        passed = set(gate(batch, timeout, len(batches), mode))
        accepted = [s for s in batch if s in passed and s not in selected]
        selected.extend(accepted)
        attempted.update(batch)
        elapsed = time.monotonic() - start
        batches.append({"mode": mode, "proposed": len(batch), "passed": len(accepted),
                        "seconds": elapsed, "timeout_seconds": timeout})
        return elapsed

    while len(selected) < width:
        remaining = [s for s in queue if s not in attempted]
        if not remaining:
            break
        batch = remaining[:min(batch_size, width - len(selected))]
        uncached = any(cached(s) is not True for s in batch)
        allowance = seconds_left() - (fallback_seconds if uncached else 0)
        if allowance < (batch_seconds if uncached else 1):
            deadline_limited = True
            break
        elapsed = check(batch, allowance, "ranked")
        if uncached:
            batch_seconds = max(batch_seconds, elapsed * 1.25)

    # Refresh the cache: a timed-out batch may have checkpointed measured
    # profiles even though it could not finish its output file. They still
    # have to pass a new full live gate here.
    fallback = [s for s in queue if s not in selected and cached(s) is True]
    used_fallback = set()
    while len(selected) < width and fallback:
        allowance = seconds_left()
        if allowance < 1:
            deadline_limited = True
            break
        batch = fallback[:width - len(selected)]
        fallback = fallback[len(batch):]
        check(batch, allowance, "cached_refill")
        used_fallback.update(batch)

    return selected, {"policy": VERSION, "target_candidates": width,
        "proposed_candidates": sum(b["proposed"] for b in batches),
        "unique_proposed_candidates": len(attempted),
        "known_measured_failures_skipped": len(known_failures),
        "gated_candidates": len(selected),
        "refilled_candidates": sum(s not in initial_set for s in selected),
        "cached_refill_candidates": sum(s in used_fallback for s in selected),
        "deadline_limited": deadline_limited, "batches": batches}
