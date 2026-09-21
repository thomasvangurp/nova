"""How many replication rounds to run at once, and on which GPUs.

A round is one scorer invocation over the whole shortlist. The scorer shards
its sequences round-robin across ``len(gpus) * shards_per_gpu`` shards, each a
separate process that loads BoltzGen before it scores anything. So a round
costs a model load whether it has sixty candidates or two, and a shortlist
smaller than the shard count leaves shards with nothing to do.

Measured on the live producer over 168 replication rounds:

    seconds = 32.9 + 1.02 * candidates          (4 GPUs)

    candidates   rounds   sec/round   overhead
             1        4        33.9        97%
             2       20        34.9        94%
            21        8        54.2        61%
            59        8        92.9        35%

    9,136 s total, 3,610 s of scoring, 5,526 s (60%) reloading the model.

Successive elimination makes this worse over an epoch, not better: the race
narrows to one or two survivors and then spends twelve full model loads
re-observing them. A sampled 150 s of the live box showed 2.5% mean GPU
utilisation with two of four GPUs at a flat zero -- the hardware was waiting
on ``import`` and checkpoint load.

WHY NOT BATCH THE REPLICATES INTO ONE CALL. Tried and measured: the scorer
deduplicates by sequence at merge, so twelve replicates of two sequences came
back as ``MERGE LOST SEQUENCES: submitted 12, merged 2. Refusing to write a
partial result``. Batch composition is load-bearing too -- the scorer's own
notes record the same molecule moving by more than a factor of two depending
on what shares its predict call -- so growing the batch would change the
metric even if dedup allowed it.

WHAT WORKS INSTEAD. Rounds are independent by construction, and the scorer is
not bit-reproducible even at a fixed seed, so nothing about them has to be
sequential. When the shortlist cannot fill the box, run several rounds at once
on disjoint GPU groups. Each round keeps its own invocation, its own shard
layout and its own predict composition, so the measured metric is untouched;
only the idle GPUs change.
"""
from __future__ import annotations

import os

#: Set to "0" to restore strictly sequential rounds.
ENABLED = os.environ.get("NANOBODY_PARALLEL_ROUNDS", "1") == "1"
#: Never run more than this many rounds at once, whatever the arithmetic says.
MAX_CONCURRENCY = int(os.environ.get("NANOBODY_PARALLEL_ROUNDS_MAX", "4"))


def plan(*, alive: int, gpus: list[str], rounds_left: int,
         shards_per_gpu: int = 1, enabled: bool | None = None,
         maximum: int | None = None) -> list[list[str]]:
    """GPU groups to run concurrently for the next wave of rounds.

    Returns one group per concurrent round. A single group containing every
    GPU is the sequential behaviour, which is what a full shortlist should
    still get: splitting a round that already fills the box only makes each
    round slower.
    """
    on = ENABLED if enabled is None else bool(enabled)
    cap = MAX_CONCURRENCY if maximum is None else int(maximum)
    rounds_left = max(0, int(rounds_left))
    if not gpus or rounds_left <= 0:
        return []
    if not on or rounds_left == 1 or alive <= 0:
        return [list(gpus)]

    shards = max(1, int(shards_per_gpu))
    # GPUs a single round can actually keep busy: the scorer fills shards
    # round-robin, so a shortlist of two occupies two shards and no more.
    useful = max(1, min(len(gpus), -(-int(alive) // shards)))
    if useful >= len(gpus):
        return [list(gpus)]

    concurrency = min(len(gpus) // useful, rounds_left, max(1, cap))
    if concurrency <= 1:
        return [list(gpus)]
    return [list(gpus[i * useful:(i + 1) * useful]) for i in range(concurrency)]


def speedup(*, alive: int, gpus: int, groups: int,
            load_seconds: float = 32.9, per_candidate: float = 1.02) -> float:
    """Rounds per second gained, for logging. 1.0 means no change."""
    if groups <= 1 or gpus <= 0:
        return 1.0
    return float(groups)
