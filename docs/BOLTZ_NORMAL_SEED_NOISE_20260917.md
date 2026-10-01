# Normal Boltz seed variability versus NIM

Status: all four normal passes completed, 80/80 predictions; final artifacts
fetched with remote/local SHA-256 and byte-count verification. Isolated instance
51332541 is verified `cur_state=stopped`, `actual_status=exited`; GPU compute has
stopped. Its disk is retained and storage charges may continue; it was not destroyed.
Label `nova-boltz2-seed-noise-test-20260917`, one RTX 6000 Ada. The original NIM
host 51321895 remains stopped and cannot currently resume because its host is
offline. Production instance 51125470 is untouched. No Blackwell fleet was rented.

## Test

Recompute the existing 20-molecule P40261 portfolio with normal OSS Boltz and
seeds 68, 69, 70 and 71. Preserve source, checkpoints, target/MSA, numeric record
IDs, batch ordering, precision, preprocessing, structure r3/s100/d1 and affinity
r5/s100/d3 with MW correction. Every seed gets a new output root; old scores
cannot satisfy a new pass. The seed-68 pass also checks cross-host reproducibility.

Compare all six normal ranking pairs across distinct seeds and all eight
normal-versus-corrected-NIM pairs. Pair comparisons share observations and are
not eight independent experiments. This is one target and 20 molecules, not a
fleet or mining competitiveness test. The prior identical-seed normal repetitions
tested reproducibility, not normal across-seed sampling variation.

Inputs and results: `data/nim19_benchmark/20260917/normal-seeds/`.
Harness: `nim19_seed_control.py`, `nim19_seed_runner.py`,
`nim19_paired_benchmark.py`, `analyze_boltz_seed_noise.py`.

Cross-host calibration completed: seed 68 reproduced all 20 original scores
exactly (maximum absolute difference 0; Spearman 1.000). Its new batch took
258.05 seconds. Do not compare this replacement-host timing directly with the
old host's NIM timing or revise the earlier same-host 2.48x speed measurement.

## Final result

| Comparison | Pairs | Mean Spearman | Range | Mean top-five overlap |
| --- | ---: | ---: | --- | ---: |
| Normal versus normal, different seeds | 6 | 0.91729 | 0.84662-0.96391 | 4.167/5 |
| MW-corrected NIM versus normal | 8 | 0.91880 | 0.87519-0.95338 | 4.375/5 |
| MW-corrected NIM versus NIM | 1 | 0.91128 | One pair | 4/5 |

All four normal passes and both prior NIM passes chose the same winning molecule.
All pairwise top-five overlaps were 4/5 or 5/5. Comparing four-seed normal and
two-pass NIM mean scores gives Spearman 0.96241. Normal seed 71 materially changes
middle/lower ordering, so the first three normal passes overestimated stability.
The final result supersedes preliminary updates.

Conclusion: on this fixed portfolio, corrected NIM-versus-normal ranking agreement
is comparable to normal Boltz's own different-seed agreement. These data do not
show substantially greater NIM ranking instability. They do not establish formal
statistical equivalence, exact numeric parity, or generalization to other targets,
near-tied elite candidates, a live validator revision or six mining iterations.

Normal pairwise mean absolute score difference is 0.005234; corrected NIM versus
normal is 0.005775. Mean-score NIM-minus-normal offset is +0.003911. The fixed-
portfolio centroid MSE is 2.6677e-5; estimated sampling-noise MSE is 1.8125e-5,
leaving a small-sample residual squared-difference estimate of 8.5513e-6.
With only two NIM passes this is uncertain, not a percentage of rank disagreement
or proof that all remaining disagreement is noise. Similar rank correlation can
coexist with a score offset. The missing NIM MW correction remains a concrete
behavioral discrepancy; correction here is an external float64 analysis adapter,
not a deployed NIM source fix or bitwise-parity claim.

Use corrected NIM for fast candidate ordering, then canonical-rescore a generous
final shortlist rather than requiring exact agreement with one noisy normal run.
This recommendation is supported only by the measured portfolio and should be
validated on additional targets and elite near-ties before production adoption.

Batch wall times for seeds 68/69/70/71: 258.05/245.46/243.80/241.24 seconds.
All checkpoint/chemical-asset digests were reverified at suite completion; all
new passes share the canonical model-source digest
`8b626489353b4cc923721d86da01f4aa9d288566cbbdd00d919d7f6190179863`.
The legacy reference artifact omitted that model-source field; new passes are
checked against seed 68, alongside original main-source/Torch/CUDA provenance
and the exact seed-68 cross-host score calibration.

Final numerical results: `data/nim19_benchmark/20260917/normal-seeds/comparison.json`.
Evidence file sizes/digests: `data/nim19_benchmark/20260917/normal-seeds/evidence-manifest.json`.

## NIM-compatible hosts: live search snapshot

Search on September 17, 2026: verified, rentable on-demand offers, at least 12
effective CPU cores, 64,000 MB host RAM, 150 GB available disk and direct SSH
ports. Reject drivers below 590.44.01. These are marketplace quotes, not
reservations or guaranteed all-in rates; chosen storage and transfer add charges.

| GPU | VRAM | Single-GPU quote/hour | Offer ID |
| --- | ---: | ---: | ---: |
| RTX 6000 Ada | 48 GB | $0.6691 | 40396208 |
| A100 SXM4 | 80 GB | $1.1504 | 39598325 |
| RTX PRO 6000 WS | 96 GB | $1.3356 | 51216015 |
| H100 SXM | 80 GB | $2.1436 | 47391250 |

The Ada offer was selected for the normal-seed test, with 100 GB allocated disk.
It matches the measured NIM GPU model and driver family. The others are candidates,
not tested runtime/cost improvements. Do not infer per-GPU VRAM solely from Vast
metadata: some RTX 4090 offers report 49,140 MB. Aftermarket 48 GB modifications
exist, so this is not necessarily a metadata error. Check actual per-GPU memory,
driver and sustained inference; modified 4090s are outside NVIDIA's tested list.

[NVIDIA's support matrix](https://docs.nvidia.com/nim/bionemo/boltz2/latest/support-matrix.html)
lists RTX 6000 Ada, L40S, A100 SXM4 80 GB, H100 80 GB and RTX PRO 6000 Blackwell
Workstation among tested hardware. It requires at least 48 GB per GPU, 12 CPU
cores, 64 GB host RAM, 80 GB storage and driver 590.44.01 or newer.

## Optimizations already used versus remaining

The unchanged NIM 1.9 inference path enables BF16/mixed precision, its optimized
backend and diffusion CUDA graphs. The test's short target is eligible for the
documented <=512-token graph path. This is source/configuration evidence, not a
new kernel-level proof of each dispatch.

The pinned source `lib/boltz_assist/boltz_pipeline.py`, SHA-256
`4e57666eef29ffdd39265512e1273720bd559888f6b16412d0b2f68d121fb4d9`,
sets affinity `max_parallel_samples` to 1 (line 427). Affinity request overrides
only change sampling steps and diffusion sample count (lines 554-564); the
public parallel cap does not override that affinity setting. The initial cap-1
versus cap-3 control therefore did not demonstrate parallel affinity execution.
Changing the affinity-specific setting is a contained source experiment, but
requires peak-memory, runtime and statistical ranking checks before adoption.

Other untested opportunities: concurrent different-molecule throughput and
multi-GPU scheduling; invariant preprocessing reuse; genuine cross-molecule
batching. The latter two require implementation and validation, not a documented
switch. Ligand-conditioned trunk features cannot simply be reused as if unchanged.
Input preparation is only about 0.31 seconds of the measured 7.4-second request,
so removing preprocessing alone would offer modest benefit.
Combining unrelated ligands into a single complex changes the task and
is not a batching workaround. Increasing samples to improve stability costs
compute rather than providing a free speedup.

[NVIDIA's optimization guide](https://docs.nvidia.com/nim/bionemo/boltz2/latest/optimization.html)
documents automatic optimized execution and sample-parallelism trade-offs.
Release 1.9 has no separate TensorRT engine path to enable. The public performance
tables are from older 1.6 TensorRT measurements, not an additional 1.9 speed gain.

Source audit: `data/nim19_benchmark/20260917/nim-optimization-source-audit.json`.
Only named scientific inference files were inspected from hash-verified image
layers; registry credentials were held in process memory and never stored.

## Multi-GPU search and modified 4090s

A second live query (2-16 GPUs per offer) found a 2x RTX 6000 Ada offer 26051040
at $1.4681/hour, but driver 570.211.01 fails NIM 1.9's minimum. It is not a
ready-to-use NIM host. Compatible-driver candidates include 8x RTX PRO 6000 WS
offers 50369726 and 49106941 at $10.1341/hour. Each physical machine is listed
separately; smaller bundles on the same machine are alternatives, not additive
independent capacity. Actual hardware, deployment and throughput remain unverified.

At 486 predictions/GPU-hour, ideal 8- and 16-GPU linear extrapolations are about
3,888 and 7,776 predictions/hour. These omit orchestration, uneven work and scaling
losses; no multi-GPU NIM workload was run in this experiment.

Stock 4090s have 24 GB according to
[NVIDIA](https://www.nvidia.com/en-ph/geforce/graphics-cards/compare/).
[A vendor's modified-card specification](https://millionminer.com/product/ai-hardware/nvidia-rtx-4090-48gb-blower)
describes aftermarket 48 GB variants. NVIDIA's NIM matrix says other GPUs meeting
the memory minimum may work but are not officially tested. This makes modified
48 GB 4090s experimental candidates, not proven NIM-compatible fleet choices.
A >=4-GPU search found no advertised 48 GB 4090 offer passing all current filters
and driver checks at the time of that query. No additional GPU fleet was rented.

Blackwell is supported by NIM's newer runtime, but do not assume our pinned
Torch 2.6/CUDA 12.6 reference environment can be moved unchanged onto it.
[PyTorch 2.7 introduced Blackwell support and CUDA 12.8 wheels](https://pytorch.org/blog/pytorch-2-7/).
Keep canonical validation on the proven Ada/4090 reference path unless a newer
reference runtime is independently validated. At the quoted prices, Blackwell
would need about 1.89x Ada's per-GPU throughput to improve cost per prediction;
that comparison has not been measured.

Bootstrap correction: the minimal public-source transfer did not create the
reaction-database parent directory, so initial setup ended with FileNotFoundError
before any scoring. The setup now creates that parent. A focused regression test
exercises a fresh minimal tree; the restarted setup completed successfully.
The failed state is preserved as `seed-runner-bootstrap-failure.json`. Twenty-five
focused network-free tests pass. Scientific model code/parameters were not changed.
