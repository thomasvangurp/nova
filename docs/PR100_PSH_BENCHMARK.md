# Fixed-structure TNP stage benchmark

This diagnostic measures the installed, pinned TNP surface annotation before any optimization. It does not change inference, scoring, submission or production configuration.

Run the standalone `nanobody_opt/pr100_psh_profile.py` with the native TNP Python environment, an existing hydrogen-stripped IMGT-numbered H-chain final PDB, and `--pdb PATH --repeats 3`. The caller must enforce a process-group deadline and schedule this through Pi outside live resource allocations. Do not run instrumentation inside a shared worker process.

The script checks source hashes against TNP revision `29dcac72f1380e8538e8870f45a699d3c6156162`, creates a private PDB copy for every invocation, randomizes baseline/profiled order with a fixed seed, and emits JSON only. Native stdout and stderr are suppressed before being exposed to a caller.

Measured calls: PSA executable, both PSA parsing passes, PDB parsing, all-atom residue distances and salt-bridge distances. Original functions, argument order, surface selection, CDR neighborhood, accumulation and output writing remain unchanged. Unordered-pair counts describe repeated work; no caching is introduced.

Report baseline annotation time separately from instrumented time. Pair bookkeeping lies outside timed distance calls but inside annotation total, so residual includes instrumentation overhead. A negative residual is reported, not clamped. Exact metric and annotated-PDB hash agreement across all repetitions must be checked before using the timings.

This is the combined PSH/PPC/PNC stage on a fixed structure. It excludes model prediction and refinement, and process startup. Imported-module timing may include cached modules. Compare with the separately measured whole-TNP runtime before identifying a bottleneck. These timings alone cannot establish a GPU, KD-tree, persistent-worker or end-to-end throughput gain.
