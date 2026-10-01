# Boltz-2 NIM 1.9.0: initial isolated benchmark

Initial smoke-test conclusion: inference works; that test alone did not establish
speedup or validator parity. See the subsequent
[paired validator-contract comparison](NIM19_VALIDATOR_COMPARISON_20260917.md):
NIM is faster on one GPU/worker, but canonical-score parity fails.

One disposable RTX 6000 Ada (49,140 MB reported VRAM), driver 595.58.03.
GPU quote: $0.716667/hour, excluding any storage or other rental charges.
No production mining services, wallets or commitments were changed.
Test instance 51321895 was stopped after saving the results: the final Vast check
reported `cur_state=stopped`, `actual_status=exited`. Its disk is retained for
reproducibility; it was not destroyed, and storage charges may continue.

## Measured results

| Workload | First request | Warm median | Outcome |
| --- | ---: | ---: | --- |
| User GFP, r3 / 50 structure steps / 1 sample | 5.23 s | 1.76 s | 4/4 functional passes |
| P40261 + real MSA + aspirin, parallel cap 1 | 7.61 s | 7.01 s | 4/4 functional passes |
| Same P40261 control, parallel cap 3 | 6.83 s | 7.13 s | 4/4 functional passes |
| Hemoglobin + connected aspirin, separate control | 5.06 s | 4.69 s | 2/2 functional passes |

P40261 requests use structure recycling 3, structure steps 100, one structure
sample, affinity steps 100, three affinity samples, molecular-weight correction
enabled and the actual 991,788-byte P40261 A3M. Backend source initializes affinity
recycling at 5; the public API does not expose a separate affinity recycling
parameter. A paired reference run is still required to establish the full contract.

For the cap-1 control, warm throughput is approximately 514 requests/GPU-hour,
or approximately $0.00140 per request at the quoted GPU-only rate. This is an
aspirin timing control, not a legal mining portfolio or evidence of competitiveness.
Three warm repetitions are too few for a statistically reliable tuning conclusion.

Warm cap-1 requests spent approximately 3.0 s in structure inference, 3.7 s in
affinity prediction, and 0.2 s in input preparation. Raising the public parallel
cap did not improve this one-structure-sample workload. It must not be treated as
proof of cross-molecule batching or parallel affinity diffusion.

## Input failure, not a successful heme prediction

The exact user/NVIDIA-example SMILES returned HTTP 422. RDKit parses it but finds
two disconnected components. Changing `HEME` to `L1` did not fix the failure.
Connected organic porphyrin passed (4.99 s); connected aspirin passed; adding a
disconnected water molecule to aspirin caused HTTP 422. These controls support
disconnected-ligand rejection as the explanation for the supplied example.
Removing the iron changes the molecule: that control is **not** a heme prediction.
The original supplied example remains failed and is retained as such in the data.

## Fidelity limitations

- No paired open-source/validator scorer run has been performed on this GPU.
- Identical P40261 requests produced different scalar predictions across repeats,
  despite `NIM_DEFAULT_RANDOM_SEED=68`: cap-1 affinity values ranged 2.1875–2.25
  and probabilities 0.2294921875–0.26171875. Per-request RNG/MSA/ligand preparation
  needs examination before assuming deterministic agreement with our scorer.
- Returned model-1 and model-2 fields equal the aggregate in these observations.
  That requires checking the output adapter; it is not proof that only one head runs.
- GFP confidence was only 0.2664. A finite confidence and valid mmCIF establish
  functional execution, not biological accuracy.
- No leaderboard comparison, synthesis/novelty validation, six-round search,
  large-ligand throughput study or fleet scaling test was performed.
- A 3,023 MB between-request VRAM observation is not peak VRAM and does not prove
  that unsupported 24 GB cards can run the workload safely.

## Runtime and reproducibility

Image tag `nvcr.io/nim/mit/boltz2:1.9.0`:

- Index: `sha256:a2eb69458fd9505ce9fdaebb16e206da909666ec8d94a297d81e7b57a2f6c090`
- amd64: `sha256:1d2b8cd39962a7841908e5fff6ea9cd76eb76c000ca19690f8582277376d88d7`
- 108 layer descriptors, 11,502,437,377 compressed layer bytes.
- Layer transfer took 59.2 s; transfer plus unpack took 420.8 s.
- Direct extracted-runtime readiness took 98.08 s.
- PyTorch `2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2.

Four custom-checkpoint assets were verified by exact bytes and SHA-256, matching
the existing validated cache runbook:

| Asset | SHA-256 |
| --- | --- |
| boltz2_conf.ckpt | 090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1 |
| boltz2_aff.ckpt | dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e |
| mols.tar | 39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7 |
| ccd.pkl | 2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4 |

This was **not** a canonical Docker deployment. An advertised-VM rental failed
before receiving scientific data and was removed. The replacement standard
container received an isolated OCI bundle. PRoot could execute CUDA arithmetic
but did not reach service readiness within the tested window. Direct execution
of the extracted runtime succeeded. Stock NIM inference source and shipped GPU
kernels were not edited; only the runtime wrapper, library paths and environment
were adapted. No source-compilation benchmark or new kernel implementation was done.

Direct execution needs image-absolute symlinks resolved within the image,
relocated Python/package/library paths, OpenMPI and NVSHMEM libraries, and actual
host NVIDIA driver libraries ahead of image stubs/compatibility libraries.
CPU math thread pools were set to one. `/opt/nim` points only to this disposable
instance's extracted tree; model/cache/output directories are explicit symlinks.
Do not apply these changes to a production host or call this a validated fleet runbook.

Control helper: `tools/nim19_test_control.py`. Benchmark: `tools/nim19_benchmark.py`.
Runtime wrapper: `tools/nim19_native_runtime.py`. All SSH uses rediscovered direct
endpoints. Credentials are fetched in-process from Keychain and never stored in
test code, scientific artifacts, registry configuration, or remote credential files.
Remote stdout and stderr are captured; only allowlisted scientific scalars/events
are exposed. Ten network-free regression tests pass; the full local science suite
was not run.

For this already-provisioned disposable instance, the tested startup route is
`native-direct-probe`, then `native-direct-start`, then `native-status` and
`benchmark`. The legacy `native-start` uses PRoot and is not the successful route.
The helper's `stop` operation is guarded to this exact test instance ID and label.

## Saved evidence

Local directory: `data/nim19_benchmark/20260917/` (generated scientific artifacts).
Transferred copies were verified against remote SHA-256:

- `smoke-results.json`: `1d2d4a2fca526e09ad09c9a2e371a1848a58051ba4640a98dd4c795da934f0fe`
- `heme-control.json`: `8cb3b37cda174d65383e00c8c4e4e8f86d579e3d300847ec8c43087fb697ca3b`
- `hemoglobin-aspirin-control.json`: `63c726141ad66be6d06bfd6aa0068784453a885bd57fe1c94241ad6c6d0711a2`
- `fragment-controls.json`: `b94a7b390e3e4b21a5d60e034827f5872bbb1b058ea5700f70efd30e99929394`

Next gate: paired, pinned validator/optimized-OSS scoring of diverse legal
molecules using the same GPU and target/MSA, with repeatability, score deltas,
rank agreement and warmed batch throughput. Only then assess speedup or adoption.

References: [NVIDIA optimization options](https://docs.nvidia.com/nim/bionemo/boltz2/latest/optimization.html),
[custom checkpoints](https://docs.nvidia.com/nim/bionemo/boltz2/latest/custom-checkpoints.html),
[release notes](https://docs.nvidia.com/nim/bionemo/boltz2/latest/release-notes.html).
