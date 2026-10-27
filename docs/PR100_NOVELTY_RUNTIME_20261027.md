# Dedicated PR100 novelty runtime

Epoch 25967 attempt 6 exposed a runtime error: native profiling used abnumber 0.2.7, which has no `Chain.batch`. The archive annotator caught that missing-API error as an ordinary batch failure. Every one of 8,575 archive entries therefore lacked CDR data. The fresh archive added 22 records; the old runtime changed 22 accepted candidates to 10, with all 12 removals caused by the 90% identity fallback. Missing CDRs also prevent CDR-specific duplicate checks, so neither the removed nor surviving candidates establish qualified validator parity.

`pr100_novelty_runtime` separates annotation and archive search from the native profiling interpreter. Both native-end checks and final fresh-archive rechecks use the configured child. It remains inside the existing supervisor's owned process tree, deadline and fixed-point cleanup. The profiling interpreter, scientific packages, observations and native checkpoint identity remain unchanged.

The child checks the pinned annotation package/tool/reference identities, verifies the installed public asset manifests, and executes `Chain.batch` on an already-native-validated positive input before indexing an archive. Missing or noncallable APIs, import failures and failed positive annotation produce an unsuccessful operation. The existing validator batch-failure semantics and duplicate thresholds remain unchanged after that preflight.

Each reply binds the exact input rows, archive bytes, algorithm sources and actual runtime identity. The parent reconstructs accepted rows from its unchanged input, validates every verdict and refuses changed inputs or archives. Library streams are captured without emitting raw diagnostics. Exported fleet evidence retains the qualified runtime hash.

## Configuration and qualification

Every native host and the final-check master requires a `novelty_runtime` object containing `python`, `path`, `python_path`, and `identity_sha256`. There is no fallback to the native interpreter. The dedicated prefix is `/root/pr100-novelty-runtime-20261027`; its interpreter wrapper is `bin/novelty-python`, executable path begins with `bin`, and its compatibility shim is `shim`.

The exact public annotation/dependency assets are copied from the existing qualified slave annotation installation and its scientific dependencies. Configuration dependencies come from the exact installed master packages. No native installation is upgraded. Both hosts verify the same three manifests:

- `assets-manifest.json`: `a7dfc8aa900ab17ae09d68fc174da463776eebed650527ada677d34089dd8daa`, 4,408 files.
- `pydantic-assets-manifest.json`: `69a292ead64e21d4bcb7f9f84501f6c75017ed1323b67138505e4a275f6c7ab8`, 157 files.
- `hmmer-assets-manifest.json`: `3c9277ee4b98b9f0e6939bc5afbbc0cc5713366cef741b33be7bd00a1fcae477`, four exact HMMER dynamic libraries and the interpreter wrapper.

The pinned hmmscan binary needs `libmpi.so.40`, `libopen-pal.so.40`, `libopen-rte.so.40` and `libz.so.1`. Their original qualified closure comes from the slave annotation installation. The wrapper sets only its private HMMER library directory before executing Python. Runtime identity also binds the actual numerical execution environment. A package list and executable hash alone did not establish a working runtime: the first isolated positive-batch probe correctly failed until this dynamic-library closure was supplied.

Annotation uses abnumber 0.4.4, ANARCI 2026.2.13.2, ANARCII metadata 2.0.5, BioPython 1.86, NumPy 2.0.2 and pandas 2.2.3. The pinned default is ANARCI (`use_anarcii=False`); this runtime does not provision the unused neural ANARCII/Torch path. Search configuration uses pydantic 2.13.5 and pydantic-core 2.46.5. Parasail is absent, so the pinned BioPython alignment fallback is explicit in the runtime identity. Qualification is not a claim of bit identity with an external validator.

Approval requires a complete replay of both original archive snapshots, an actual positive batch annotation, numerical verdicts, archive CDR coverage, measured elapsed time within the owned deadline, and zero remaining descendants. Python binary differences produce distinct host identities even when the scientific asset manifests agree. Do not approve an identity from dependency metadata alone.

## Cutover recovery

A one-time migration is keyed by the configured runtime identities. It polls and collects existing old-code requests without relaunching them, waits for verified cleanup, and restores reusable evidence from retained native results and dedicated confirmation checkpoints. Audit journals never become reusable evidence. Completed checkpoints are reclassified with zero fresh or repeat native draws; incomplete checkpoints retain their missing-confirmation requirement and original execution host.

Only old invalidations proved by a retained novelty input/result difference are reopened. Real native failures and unexplained invalidations remain blocked. Completed score results retain the original score identity. Existing fresh reservations are reconciled only from bound, cleanup-verified admission/non-start evidence. Same-archive proofs from the old runtime must be rechecked, and operation paths also bind runtime and input identities.

The existing publication stays unchanged while assessment is pending or fails. That preservation is not evidence that its old novelty proof is valid. The rollout owner must assess the consumer handoff before activating a new publication; this change does not silently certify the existing pool.

Tests cover missing batch APIs, positive-batch failure, reply/input/archive identity changes, host-specific proof binding, recovered complete checkpoints without redraw, preserved scoring, true native failures, uncertain ownership and reservations, and real supervisor termination of an inert child plus grandchild. Production activation requires the committed shipping checks and review of the actual replay receipts.
