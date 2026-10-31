# Registered resident main-CUDA execution

The resident route is opt-in through `native_reference.resident: true`. Its
registration selects CUDA main refinement with unchanged original scientific
functions and CPU preparation. Existing CPU policy and historical receipts keep
their meaning. The published-validator trace check defaults to CPU refinement.

## Qualified policy and row

The new row's policy is exactly:

```json
{
  "prediction": "auto_cuda",
  "final_refinement": {
    "platform": "CUDA",
    "precision": "mixed",
    "deterministic_forces": true
  },
  "preparation_threads": 1
}
```

In addition to existing runtime, weights, NumPy, observer, implementation and
closure manifest pins, a CUDA row requires `qualification_status: "qualified"`,
`qualification_report_sha256`, `resident_module_sha256`, and `cuda_plugin`:

```json
{
  "distribution": "OpenMM-CUDA-12",
  "version": "8.5.0",
  "wheel_sha256": "5e9b1054820b177366113d954fa2885837b31333ea2f6daa489da23148469173",
  "manifest_sha256": "e3e4357e6e25936708ebebfbaf4b4fdaa36e0ded2b60c5fedecf5c192b776639"
}
```

The verified runtime identity adds the identical `cuda_plugin` descriptor **in
memory only**, after sidecar verification. The sealed bundle's original
`runtime-contract.json` and its manifest stay unchanged. The new CUDA runtime hash
is the digest of that extended dictionary; CPU runtime hashes remain unchanged. The
resident module is `nanobody_opt.pr100_registered_resident`. Pending registrations
remain outside the approved registry; a pending CUDA row is rejected even if its
other hashes have been filled. Root fills final rows only after qualification and
source freeze; this document does not register or enable them.

## Settings and sidecar

`resident` defaults to false. CUDA requires it to be true. The optional
`resident_batch_draws` defaults to 32 and must be an integer from 1 through 32;
`workers_per_gpu` still bounds actual worker processes. Existing GPU allowlists,
leases, native budgets and phase deadlines continue to apply.

`cuda_plugin_root` is the absolute sidecar root, currently staged as
`/root/pr100-validator-reference-20261030/cuda850-sidecar-v1`. Its fixed
`plugin-manifest.json` uses schema `pr100-openmm-cuda-plugin-v1`, the package/version/
wheel fields above, `plugin_directory`, `library_directories`, and a `files` map.
Directories and file paths are relative to that root. The loader helper verifies
all listed shared objects, refuses path escapes/symlinked libraries, and refuses
unlisted shared objects in any loaded directory. There is no automatic discovery.
CUDA library dependencies must belong to the qualified closure.

## Evidence and rollout

Every draw retains fresh original prediction, geometry, fallback and save events.
The unchanged observer records actual CUDA main Context properties, including
logical `DeviceIndex: "0"` inside the ordinary single-GPU lease mask. Missing-atom
and hydrogen Contexts must be CPU with `Threads: "1"`. The router is separate from
the observer; observer bytes and its no-mutation claim remain unchanged.

Resident requests bind `execution: "resident-v1"`, the registered module hash,
equal valid `model_state_start_sha256` / `model_state_end_sha256`, a 32-character
UUID-hex `resident_batch_id`, and `resident_batch_draws` from 1 through 32. These
are authentic batch-level state checks, not reused draw evidence. Complete ordinary
trace, structure, aggregation and current metric/selection checks still apply.

Preserve the historical CPU row unchanged. Editing the runner changes its byte
hash, so that old row continues validating historical receipts but cannot launch
the new runner. Root appends a current-runner CPU-compatible row for rollback and
the qualified GPU row, then selects the intended identity atomically. Do not change
GPU allocation as part of this rollout or relabel CPU/cache evidence as CUDA.

Focused tests cover old-default behavior, exact registered properties and closure,
pending/forged receipt rejection, bounded batching and historical CPU evidence.
Full shipping CI runs once after the shared implementation and final rows settle.
Scientific qualification uses observed distributions and real gate decisions,
not bit-identical final coordinates.
