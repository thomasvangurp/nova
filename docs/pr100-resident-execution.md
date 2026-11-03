# Registered resident native execution

`native_reference.resident` defaults to `false`. Enabling it requires an approved
registration that binds `pr100_registered_resident.py`, the exact CUDA sidecar,
the original sealed scientific environment, and the main-CUDA policy. An enabled
flag alone does not register or qualify a runtime. Existing CPU registrations
and their receipt meanings remain intact.

An implementation repair appends a new registration and qualification report.
Safe activation selects it explicitly with `--native-registration <identity>`;
the transition refuses changes to the sealed runtime, scientific sources,
weights, policy, loader or allocation, and checks the installed Pi consumer
before stopping the old controller.

Replacing the slave's full CPU profiler requires an explicit
`--cuda-native-host-receipt <path>`. The receipt retains the actual requests,
results and ordinary lease cleanup from concurrent device canaries on all four
slave GPUs, and pins the unchanged scientific implementation. Activation keeps
the existing 32 CPU slots, enables four resident workers per GPU, selects the
queue depth that was actually tested, disables the old optional screen and
uses the same observed-fleet policy as the master. GPU admission failure stays
inconclusive; it does not call the full CPU profiler. Numbering, A3D and the
qualified CPU1 preparation remain CPU work.

The native phase selects `RegisteredResidentAdapter`. Each allowed GPU has one
ordinary cooperative lease and a bounded pool of `workers_per_gpu` independent
scientific processes. Those processes retain their four original NBB models
across successive tranches. Each draw executes the complete original TNP/NBB CLI
bodies, including fresh original numbering, prediction, ranked saves/refinement,
PSH and A3D. There is no prediction cache, NN batching, PCA change or H-GPU path.

The existing Observer is installed afresh after resetting the original refine
module globals. A scoped Context router underneath that Observer forces missing
atoms and hydrogen preparation to CPU Threads1 and both `minimize_energy` and
direct `sidechain_repair` Contexts to CUDA mixed precision with deterministic
forces. Repair also creates Simulations outside `minimize_energy`; both its
explicit-CPU and implicit-platform constructors must be routed. Original systems, integrators,
restraints, tolerances, iteration limits, checks and fallback ordering are kept.
Trace/structure/aggregation receipts use the current registered verifier.

`resident_batch_draws` remains a configured ceiling from 1 through 32 per GPU.
The deadline-admission successor caps each command at one draw per actual model
worker, so four GPUs with four workers each admit at most 16 outstanding calls.
A completed draw cannot wait behind a later slow refinement on the same worker.
The configured CPU allocation still covers the actual processes and executor
workers. Each scientific worker checks its complete model state at startup and
command end; the coordinator checks the shared sealed sources/weights before
committing authoritative results. The scientific prediction/refinement bodies,
Observer, native reference verifier and registered policy remain unchanged.

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

Queueing, waiting for an old owner to retire and cold model startup consume the
original call budget. They never reset the current 90-second cap or extend the
phase/lease deadline. Client, coordinator and worker recheck admission before a
draw starts: at least ten seconds of scientific window must remain after the
existing ten-second response/cleanup reserve. A never-submitted expired call
does not stop or retire a healthy or warming pool. That session remains owned
under its original lease and is closed by the normal phase lifecycle. Actually
started failures and uncertain workers still retire with fixed-point cleanup.

This successor changes the registered resident module's bytes. The historical
registration is not updated to bless it. A replacement row and real execution
canary are required before activation; lifecycle tests are not qualification.

The native phase's `finally` closes every session and requires the ordinary lease
owner's fixed-point descendant cleanup before novelty/publication handoff.
Cancellation uses that same owner; no unknown process is killed. Local fixtures
and lifecycle tests establish execution behavior, not scientific qualification.
