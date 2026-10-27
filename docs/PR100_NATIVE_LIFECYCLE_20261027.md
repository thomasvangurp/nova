# CPU-manager ownership for experimental native refill

The four-slot enriched refill trial created five GPU health-monitor tasks even though it performed only CPU-native profiles. All tasks were eventually cancelled by the enclosing runner, but the per-profile manager did not own the process-global scheduler safely. This patch removes that dependency from the experimental refill path. It does not enable refill in production, change the scientific runtime, or qualify a wider native deployment.

## Cause and bounded fix

The pinned `AsyncServiceManager.initialize()` calls the process-global `get_gpu_scheduler()` whenever GPU scheduling is enabled. `GPUScheduler.initialize()` tests its initialized flag before taking its lock but does not test again inside the lock. Concurrent first users can therefore create multiple health tasks. `shutdown()` cancels only the task stored in the latest handle and retains its loop-bound lock. A retained inert reproduction, using the exact pinned initialization/shutdown methods, demonstrates four health tasks after four simultaneous initializers, three still pending after scheduler shutdown, and failure when the same global lock is reused in a second runner. It performs no hardware probe or scientific profile.

The native `DevelopabilityService` never acquires a GPU. Its profiling path uses only the manager's local TNP semaphore, task timeout and batch size. The smallest repair is therefore a strict `cpu_only_manager` option on the existing `pr100_filter.develop` adapter. It defaults to false. Experimental slot refill passes true, which changes only `config.async_config.gpu_scheduler.enabled` in that call's fresh configuration. Each profile retains its original local manager; it never initializes, shuts down, or mutates any global GPU owner.

The original `AsyncServiceManager`, `DevelopabilityService`, `ProfileExecution`, TNP worker, aggregation implementation and scientific settings remain in use. There is no vendored source edit, production monkeypatch, shared mutable timeout, alternate profiler or suppressed initialization failure. The default batch path retains its original GPU configuration. This repair does not attempt to change the GPU scheduler used by other services.

## Failure ownership

A completed profile still reaches the original observation/checkpoint callback before its slot is released. The original service awaits its profile thread after cancellation, and the private refill runner joins its thread executor before returning.

For the CPU-only adapter, a primary failure and a subsequent manager shutdown failure are both retained in a `BaseExceptionGroup`. Refill likewise cancels and drains every peer, then retains any non-cancellation teardown failures together with the initial failure. It no longer discards peer teardown exceptions returned by `gather(return_exceptions=True)`. Each cleanup path tracks only a failure raised by its own operation; a caller’s previously handled exception is never included. An intentional cancellation alone retains its normal cancellation semantics. Scientific rejection dictionaries and native error observations are unchanged.

Budgets, the shared fresh-admission cap, the absolute admission-window recheck, per-call timeout clamps, rejection/no-redraw rules, native host binding and checkpoint persistence are unchanged.

## Checkpoint boundary

The actual adapter file digest remains part of native resume identity. This patch introduces no hash alias or retrospective compatibility assertion. A checkpoint created before the adapter change remains stored and is deferred as `native_runtime_identity_mismatch`; it is neither discarded nor redrawn. Regression cases cover both a partial one-observation checkpoint and a completed three-observation checkpoint.

A future production cutover must explicitly address any retained incompatible checkpoints. That migration and real-runtime qualification are separate review gates. The prior four-core measurements are evidence for the earlier implementation; they do not qualify this new adapter option or establish a new speedup.

## Inert validation

The new shipping suite executes the actual pinned configuration, scheduler, service manager, developability service and profile-execution classes. Only hardware probes and the scientific worker body are replaced with inert functions. Test-only Pydantic 2.10.6 is pinned in the shipping test requirements so the real configuration models can run; no production dependency is installed or changed.

At each width of 4, 32 and 96, tests run three waves in each of two separate runner lifetimes in the same interpreter. They verify the intended concurrency, zero global GPU-scheduler accesses, zero health tasks during completion callbacks, unchanged complete scientific configuration, unchanged async settings except the explicit monitor opt-out and existing per-call timeout/width settings, and no remaining private threads or initialized managers.

Additional cases exercise a foreign live GPU owner that must remain untouched, unchanged default behavior, strict option validation, cancellation while an inert profile thread is running, local initialization failure, manager teardown failure, simultaneous teardown failures at all three widths, combined callback and teardown failures, isolation from a caller’s previously handled exception, and preservation of old bound checkpoints. The new lifecycle suite contains 20 cases. The existing refill, native-worker, throughput, deadline, fresh-budget, supervisor and execution-identity suites remain required in shipping CI.

No new native profiles, GPU calls, live host writes, resource expansion or production deployment were performed for this lifecycle repair. Capacity is not yet demonstrated to be optimal: within-host scheduling, host allocation and real-runtime qualification remain distinct questions.
