# Fixed-structure TNP stage benchmark

This diagnostic measures the installed, pinned TNP surface annotation before any optimization. It does not change inference, scoring, submission or production configuration.

Run the standalone `nanobody_opt/pr100_psh_profile.py` with the native TNP Python environment, an existing hydrogen-stripped IMGT-numbered H-chain final PDB, and `--pdb PATH --repeats 3`. The caller must enforce a process-group deadline and schedule this through Pi outside live resource allocations. Do not run instrumentation inside a shared worker process.

The script checks source hashes against TNP revision `29dcac72f1380e8538e8870f45a699d3c6156162`, creates a private PDB copy for every invocation, randomizes baseline/profiled order with a fixed seed, and emits JSON only. Native stdout and stderr are suppressed before being exposed to a caller.

Measured calls: PSA executable, both PSA parsing passes, PDB parsing, all-atom residue distances and salt-bridge distances. Original functions, argument order, surface selection, CDR neighborhood, accumulation and output writing remain unchanged. Unordered-pair counts describe repeated work; no caching is introduced.

Report baseline annotation time separately from instrumented time. Pair bookkeeping lies outside timed distance calls but inside annotation total, so residual includes instrumentation overhead. A negative residual is reported, not clamped. Exact metric and annotated-PDB hash agreement across all repetitions must be checked before using the timings.

This is the combined PSH/PPC/PNC stage on a fixed structure. It excludes model prediction and refinement, and process startup. Imported-module timing may include cached modules. Compare with the separately measured whole-TNP runtime before identifying a bottleneck. These timings alone cannot establish a GPU, KD-tree, persistent-worker or end-to-end throughput gain.

## Experimental distance acceleration

`nanobody_opt/pr100_psh_geometry.py` provides an opt-in private-process context. No live entrypoint installs it. Source hashes must match the pinned TNP modules. The context restores all originals on exit and rejects competing installations.

- `reference_cache` computes a residue pair with the original function once and reuses the result for reverse/repeated calls.
- `numpy` builds a bounded all-atom distance table from the original PDB parser. It preserves explicit x/y/z arithmetic order, the original scalar `math.sqrt`, and the original distance sentinel. The original annotation loops, pair directions, cutoffs, accumulation order, salt-bridge behavior, surface selection and rounding remain untouched. It falls back to the reference cache for oversized or unsupported structures.
- `cuda` is an optional float64 geometry prototype, not qualified by the CPU result. It has not been run on a GPU. It uses no matrix multiplication, float32 or TF32, but numerical and speed equivalence still require separate GPU measurement.

The coordinate inputs must remain immutable within each context. Run a new context for each structure. The default atom cap is 2,048, limiting dense matrix allocation; this still needs memory/capacity measurement before any multiworker deployment.

`scripts/benchmark_pr100_psh_geometry.py` consumes a private fixture directory containing PDB and matching PSA outputs plus digest manifests. It runs the original `CreateAnnotation` with the same recorded PSA response for every mode, compares PSH/PPC/PNC and annotated-PDB hashes, and times randomized baseline/cache/NumPy runs. This excludes PSA execution, model inference and refinement. It is deliberately not an end-to-end benchmark.

Measured locally on six previously generated fixed structures, five repetitions per mode: all 85,116 ordered residue-pair distances matched bit-for-bit; all 90 annotation outputs matched in metrics and annotated-PDB bytes. Median component speedups were 1.31x for the cache and 1.92x for NumPy. The baseline after-PSA annotation took about 29–37ms and NumPy 16–19ms on the local CPU. Do not extrapolate these hardware-specific measurements into a whole-TNP or GPU speedup. The native measurements show this is a small stage of the whole pipeline.

The real fixture files remain outside the repository in the task evidence directory; no sequences or generated structural data are committed. Before deployment, run the same artifact comparison and bounded resource benchmark on the actual Linux runtime, include threshold-edge cases, and pass release CI.
