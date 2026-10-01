# NIM 1.9.0 versus the pinned NOVA molecule scoring contract

Status: completed. Vast confirms instance 51321895 is stopped (`actual_status=exited`,
`cur_state=stopped`). Its disk is retained; storage charges may continue.
Production mining is untouched. This is an isolated offline test, not a fleet
deployment or proof of a competitive mining entry.

## Measurements

| Pass (20 molecules) | Wall time | Seconds/molecule | Molecules/GPU-hour |
| --- | ---: | ---: | ---: |
| Reference, first batch | 428.19 s | 21.41 | 168.15 |
| Reference, warm filesystem/checkpoint caches | 367.38 s | 18.37 | 195.98 |
| Resident NIM, first portfolio pass | 148.34 s | 7.42 | 485.37 |
| Resident NIM, second portfolio pass | 148.11 s | 7.41 | 486.11 |
| Optimized OSS | 373.95 s | 18.70 | 192.54 |

The meaningful warmed-reference comparison is approximately **2.48×**, not the
2.89× ratio versus the first, colder reference batch. Actual warm reference
model callbacks total about 15.02 seconds/molecule. NIM reports mean structure
inference 3.25 s, affinity prediction 3.75 s and input preparation 0.31 s.
These phase counters are not a fresh kernel-level profile.

Reference repetition matches every final probability, affinity prediction and
normalized score numerically, with zero ranking displacement on all 20 records.
NIM is functional (40/40 successful predictions) but does **not** match:

| NIM versus reference | Score MAE | Maximum score error | Spearman | Top-5 overlap |
| --- | ---: | ---: | ---: | ---: |
| Raw NIM, pass 1 | 0.021657 | 0.032968 | 0.8992 | 4/5 |
| Raw NIM, pass 2 | 0.021864 | 0.036522 | 0.9038 | 5/5 |
| Experimental MW correction, pass 1 | 0.006068 | 0.021082 | 0.8872 | 4/5 |
| Experimental MW correction, pass 2 | 0.005914 | 0.018715 | 0.9534 | 5/5 |

All four selections retain the reference's highest-scoring molecule in this
small portfolio. Correction reduces average absolute error by approximately
72–73%, but does not restore exact values or consistent complete ordering.
Between the two raw NIM passes, score MAE is 0.005022, maximum delta 0.015869,
Spearman 0.8556 and top-5 overlap 4/5. Stable runtime is not stable scoring.
Twenty molecules and two passes do not establish general selection quality.

The optimized OSS batch preserves the complete reference ordering and all
probabilities. Its maximum affinity-value difference is 4.7684e-7 and maximum
normalized-score difference 1.9073e-8 (score MAE 8.6422e-9). These are tiny, but
fail strict zero-tolerance equality. At 373.95 seconds it is 1.79% slower than
the warmed reference in this measurement, so no speedup or strict-exact
acceptance is claimed. The requested preparation controls are enabled on the
models in all 40 callbacks; this is not proof of full-chunk dispatch in the
canonical sequential M=3 affinity path.

## Decision

NIM is approximately 2.48× faster than the warmed reference, and 2.52× faster
than this optimized variant, for one worker on this GPU/workload. It is **not a
canonical-score replacement**. The ignored MW flag is a concrete discrepancy;
fixing its post-processing alone does not resolve stochastic/numeric drift.
Keep the canonical exact path for release validation and exact-label storage.
NIM can be considered an approximate shortlist scorer only after broader
selection-quality testing and independent canonical rescoring.

This does not demonstrate superior fleet economics, 2–3-context/GPU throughput,
six-round epoch fit, leaderboard competitiveness, unsupported 24 GB GPU support,
or the validity of all ten hotkey submission sets. No mining deployment changed.

## Method

The same disposable RTX 6000 Ada 48 GB instance (51321895) runs every GPU pass
serially. NIM uses the unchanged extracted 1.9.0 image and custom checkpoints
from the [initial benchmark](NIM19_BENCHMARK_20260917.md). The reference uses
the local NOVA vendored Boltz 2.2.1 path from repository revision
`f79c4032e8e9dcc103c964fc27e75974e6a3e947`, Torch 2.6.0+cu126,
cuEquivariance 0.6.1 and Lightning 2.5.5. This pins our local validator-contract
reference; it does not establish the revision deployed by a live validator.

Twenty deterministic reaction-generated molecules cover reactions 1–5, four
per reaction, and 18–49 heavy atoms. The official reaction database SHA-256 is
`a84bd5216d38975b33670a137aaa20f2b3c2a29bc363db28d44f653d8ff2c617`.
Products pass BRENK, banned-atom, rotor, atom-name safety and single-component
checks, uniqueness and historical similarity below 0.5. History is frozen by
content hash `2090037f1c197e2b05d71e0c953b40a46af7e5e28bf3d28519132db54852d71d`.
No historical winner is used as a search seed. The mixed-reaction benchmark is
not a valid live submission set: epoch reaction, set entropy and fleet release
checks are separate.

Both paths use P40261 (258 residues), identical real A3M content (3,048 sequences,
991,788 bytes, SHA-256
`ce6c3d8ec8009c2107146c51825e2c826fee3bb79eff8ca0c60e8f7dc56a965b`),
structure r3/s100/d1, affinity r5/s100/d3, step scale 1.638, molecular-weight
correction requested, chain IDs A/B and stable numeric record IDs. Structure
and confidence remain enabled. Score is
`(affinity_probability_binary - affinity_pred_value) / heavy_atoms`; higher wins.

NIM keeps its loaded models resident. The reference repeats in a fresh process,
as the canonical worker does; checkpoint and file caches are warm on repetition.
Reference `predict_step` calls are timed with CUDA synchronization at callback
boundaries, without changing model arithmetic or RNG. Batch wall time includes
preprocessing/loading/inference/output but excludes initial package installation.
NIM time includes local HTTP input/output. First calls are not mislabeled as
warm GPU inference. Single-context timing does not establish throughput versus
the production 2–3-worker-per-GPU configuration.

The optimized comparison uses the current experimental source from
`nova-compound-fleet`, affinity-only output, uncompressed in-memory handoff,
prepared diffusion invariants and direct full-chunk dispatch. Inductor,
CUDA-graph experiments, packed projections, reduced token precision, indexed
routing and cross-record batching stay off. It is independently compared with
the reference; preceding experiments do not grant this run numeric acceptance.

## Concrete NIM contract discrepancy

The shipped `lib/boltz_assist/boltz_pipeline.py` initializes its affinity model
with `affinity_mw_correction=False` (line 487). Lines 457–459 explicitly note
that the bionemo_ir backend has no analogous correction and needs separate
post-processing. The public request model accepts the flag, but the inspected
pipeline has no read of the request's correction field and no corresponding
post-processing. Request overrides for affinity steps and samples are applied.
Source excerpts and digest are saved as `NIM-contract-audit.json`.

The pinned reference applies:

`corrected = 1.03525938 * raw - 0.59992683 * molecular_weight**0.3 + 2.83288489`

For `rxn:1:52872:39548`, reference affinity is 0.66881752 versus raw NIM
1.265625; the separately evaluated correction yields 0.68196527. This supports
the missing-correction diagnosis. An experimental Python float64 adapter lowers
absolute error but is not a fix to NIM, bitwise parity, or an accepted exact
scorer. Remaining differences are not fully isolated.

NIM's seed environment variable is applied at manager startup, not exposed as
a per-request seed. Identical requests vary across repetition. RNG/ligand
preparation and backend numeric differences require separate isolation; no
single remaining cause is claimed. The audited diffusion hyperparameter defaults
match the reference (including sigma_min, rho, gamma and noise_scale), with step
scale explicitly set to 1.638.

The API output adapter also falls back to aggregate predictions when per-model
fields are missing (lines 1023–1035). Duplicated model-1/model-2 fields cannot be
used as evidence that the two actual heads agree or that only one head executes.

## Evidence and reproducibility

Scientific inputs, scalar outputs, frozen history and source audit are in
`data/nim19_benchmark/20260917/paired/`. Transfers take stable snapshots and verify
every byte count and SHA-256. No raw HTTP error bodies, credentials, private
worker logs or model traces are stored or printed.

All four model/CCD assets were SHA-256 reverified after the final pass and remain
unchanged. Sixteen focused network-free regression tests pass. Full model
execution was verified remotely; the Mac's unrelated full science suite was
not used as evidence.

Programs: `nim19_pair_setup.py`, `nim19_paired_benchmark.py`,
`nim19_pair_runner.py`, `analyze_nim19_pair.py` and the guarded
`nim19_test_control.py`. Direct SSH endpoints are rediscovered through Vast;
source transfers do not preserve controller ownership/group/modes. Credentials
are retrieved in-process from Keychain; no wallets reach the test instance.

[NVIDIA's optimization documentation](https://docs.nvidia.com/nim/bionemo/boltz2/latest/optimization.html)
describes the 1.9 unified Torch/CUDA-graph backend. These measurements are not
the older H100/TensorRT performance table and cannot inherit its speedups.
