# Registered resident native execution

`native_reference.resident` defaults to `false`. Enabling it requires an approved
registration that binds `pr100_registered_resident.py`, the exact CUDA sidecar,
the original sealed scientific environment, and the main-CUDA policy. An enabled
flag alone does not register or qualify a runtime. Existing CPU registrations
and their receipt meanings remain intact.

The native phase selects `RegisteredResidentAdapter`. Each allowed GPU has one
ordinary cooperative lease and a bounded pool of `workers_per_gpu` independent
scientific processes. Those processes retain their four original NBB models
across successive tranches. Each draw executes the complete original TNP/NBB CLI
bodies, including fresh original numbering, prediction, ranked saves/refinement,
PSH and A3D. There is no prediction cache, NN batching, PCA change or H-GPU path.

The existing Observer is installed afresh after resetting the original refine
module globals. A scoped Context router underneath that Observer forces missing
atoms and hydrogen preparation to CPU Threads1 and main `minimize_energy` to CUDA
mixed precision with deterministic forces. Original systems, integrators,
restraints, tolerances, iteration limits, checks and fallback ordering are kept.
Trace/structure/aggregation receipts use the current registered verifier.

`resident_batch_draws` defaults to 32 and is bounded from 1 through 32 per GPU.
These are asynchronous queue slots, not model processes or CPU threads. Four
GPUs with four model workers each use 16 actual processes and at most 16 executor
workers while admitting up to 128 queued calls. The configured CPU allocation
must cover those actual processes. Each scientific worker checks its complete
model state at startup and tranche end; the coordinator checks the shared sealed
sources/weights before committing authoritative results. No full model-state
hash runs inside an individual draw.

The same sealed Python interpreter runs directly with user-site imports disabled
so the registered sidecar library paths survive; the legacy shell wrapper would
overwrite `LD_LIBRARY_PATH`. Model/thread settings and original CLI bytes remain
explicitly bound. Minimum free GPU admission memory is derived as
`2048 * (workers_per_gpu + 1)` MiB, or 10240 MiB for four workers.

Scientific completion has a per-batch cutoff, with a bounded reserve for state
checks, reply collection and cleanup. Nonblocking reply collection retains
checked responsive peers when another worker fails or hangs. A start marker is
written only after the per-draw lease/deadline guard. Clean owner exit recovers
committed measurements and distinguishes never-started inputs from attempts.
Late/failed work is inconclusive, never a biological rejection. Poisoned pools
retire; there is no automatic replay of their started draws.

The native phase's `finally` closes every session and requires the ordinary lease
owner's fixed-point descendant cleanup before novelty/publication handoff.
Cancellation uses that same owner; no unknown process is killed. Local fixtures
and lifecycle tests establish execution behavior, not scientific qualification.
