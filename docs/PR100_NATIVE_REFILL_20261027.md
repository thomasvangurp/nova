# Bounded native slot refill prototype

The current `native_interleave_confirmations` scheduler fills spare slots with fresh work, but each wave still waits for its slowest call. An early completion cannot release a slot to the next confirmation or fresh candidate until `measure_batch` returns. The separate host dispatcher also waits for its slower host; this prototype changes only the within-host barrier.

## Observed opportunity

Retained epoch 25969 journal records for three complete 96-worker master waves show first-to-last completion gaps of 72.710, 66.367 and 86.456 seconds. Summing each completed slot's wait until the last completion yields 12,986.276 slot-seconds, equivalent to 135.274 seconds across all 96 slots. Four complete 32-worker slave waves similarly imply 154.984 seconds across all 32 slots. The partially recorded fourth master wave is excluded.

These are asynchronous journal persistence timestamps after completion callbacks, not exact native start/end measurements. They establish a substantial completion spread and motivate measurement; they do not establish how much production time refill will save. The original timing data and the derivation are retained under `docs/evidence/pr100_native_refill_20261027/`.

## Prototype behavior

`native_work_conserving` is a strict boolean, defaults to false, and requires the existing confirmation-interleaving opt-in. Experimental native screening remains incompatible. With the flag off, existing scheduling is unchanged.

The worker persists admission through the same path as batch profiling, then submits at most one original `pr100_filter.develop([sequence], 1)` call per free slot. All calls share one explicitly bounded event-loop thread executor; no implicit 32-thread ceiling and no queued profile beyond the admitted worker count are allowed. The unchanged adapter uses the original full TNP/aggregation computation and original exception representation. Its worker and subprocess cleanup finish before the slot is reused. No model residency, GPU path, scientific dependency, threshold or native resume identity is changed.

Completed observations go through the existing durable confirmation checkpoint and journal callbacks before scheduling advances. Pending confirmations are ordered by fewest missing observations, with stable ties. A candidate can have only one active call; another draw requires the preceding observations to pass. A genuine rejection or inconclusive call receives no further draw in the same invocation. Inconclusive confirmations retain earlier valid measurements for the established continuation policy. A fully measured resumed checkpoint is classified without new calls.

Fresh admission still uses the existing unique-candidate ledger and per-host partition of the shared 4,800 limit. Repeats do not consume fresh capacity. No host stealing or repartitioning is introduced.

Each actual adapter entry rechecks the absolute phase deadline after admission persistence and clamps its timeout to the remaining window minus cleanup reserve. If the minimum window has closed before invocation, the worker records a proved never-started result and restores fresh capacity; it does not invent a CPU observation, failure or terminal rejection. If the owner is interrupted before such proof is persisted, existing conservative admission evidence remains. At the admission cutoff, active calls finish their owned bounded work while new calls stop. The existing outer supervisor retains absolute deadlines, descendant ownership and fixed-point cleanup.

If scheduling, persistence or result validation raises, the executor cancels and awaits every active adapter task, then joins its private threads. The original adapter's cancellation path cleans up its worker. An inert full-worker test also exercises the real production supervisor killing a child and grandchild while retaining three already completed confirmation observations.

## Inert comparison

`scripts/pr100_refill_simulation.py` runs both actual worker policies with the same deterministic synthetic durations and numerical observations. It executes no native profile, model, accelerator or network call. The reports distinguish more attempted work from more completed confirmations and explicitly count deadline-censored calls.

| Scenario | Workers | Current fresh / fully confirmed | Refill fresh / fully confirmed | Current / refill wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Uniform 80-second calls, sparse passers | 96 | 600 / 17 | 600 / 17 | 880 / 720 |
| One 250-second straggler per initial 96, other calls 60 seconds | 96 | 285 / 6 | 600 / 17 | 990 / 750 |
| One 250-second straggler per initial 32, other calls 60 seconds | 32 | 95 / 2 | 416 / 10 | 990 / 990 |
| Deterministic 30–249-second calls, sparse passers | 96 | 189 / 6 | 600 / 11 | 936 / 990 |
| 430-second phase, fast confirmations and slower fresh calls | 32 | 95 / 2 | 99 / 3 | 360 / 360 |

In the heterogeneous case, refill uses more of the tail and produces 13 deadline-censored calls versus zero in the current schedule. These are inconclusive, not accepted observations. The failure scenario also shows more failures when more candidates are tried; neither count is hidden. With a sufficient window, a separate comparison produces identical call counts and scientific verdicts while eliminating only idle scheduling time. These are simulated outcomes, not a measured production speedup or a claim of identical stochastic native observations.

Focused coverage includes 32/96-slot stragglers, heterogeneous calls, terminal rejection, partial confirmation failure, fresh-cap exhaustion, resumed host identity, completed checkpoints, exact late cutoffs, admission-persistence delay, real thread cleanup, and actual supervisor ownership of an inert child tree. Existing native throughput, continuation, novelty and budget checks remain required.

## Proposed real qualification; not launched

After independent review and full shipping CI, first inventory the slave CPU topology, allocated affinity and the affinity of our existing pipeline owners. Only if physical CPUs 32–35 are allocated and unused by our pipeline, ask the rollout owner to approve a separate four-worker qualification there. Keep the already approved native-alt wrapper, scientific files, weights, thread settings and hidden accelerators unchanged. The run is declared research outside the epoch admission budget and cannot write publication or production continuation state.

Use the same deterministic, hash-bound cohort of eight previously recorded inputs for a current-scheduler arm and a refill arm, sequentially in new immutable qualification directories. Retain three required observations and early stop on rejection; each arm has a hard maximum of 24 original calls, 48 across both arms. Set a 600-second owned wall budget plus a 10-second cleanup allowance per arm, the existing 300-second call cap and 130-second admission minimum. Do not buy capacity or interrupt any existing job.

Record allowlisted candidate hashes/draw ordinals, adapter-entry and post-cleanup completion timestamps, checkpoint persistence times, in-flight maxima, fresh/repeat counts, finite native metrics and provenance, censored/error counts, and final fixed-point cleanup. Require exact approved native runtime identity, no more than four active calls, no overlap of a candidate's draws, no draw after a native rejection, no lost completed checkpoint, and zero descendants before starting the other arm. Distinguish legitimate scientific gate failures from runtime failures. Do not claim bit identity from stochastic profiles.

A four-worker result qualifies only that bounded path. Any subsequent 32-worker slave or 96-worker master rollout needs an additional approved qualification at the intended width and measurement of performance under that contention. Integration must preserve the qualified novelty runtime and pass shipping CI for the final committed source. This prototype does not alter the activation helper or authorize a production flag change. No native benchmark or production change was performed for this work.
