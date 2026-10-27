# PR100 slave runtime repair and submission audit

Recorded 27 October 2026. The isolated slave CPU runtime completed all 32
qualification profiles with finite metrics and complete structure provenance.
The root orchestration agent reviewed and accepted its alternate Python and
Torch builds. This establishes a working, explicitly different runtime; it
does not establish bit identity, validator parity, candidate eligibility, or
whole-epoch throughput.

The [retained evidence bundle](evidence/pr100_slave_runtime_repair_20261027/README.md)
contains the exact E helpers, public scientific manifests, per-profile numeric
records, cleanup receipt, runtime-difference approval, and submission audit.
No sequence strings, raw child logs, credentials, or process-environment dumps
are included. The application controller code is unchanged by this retention.

## Runtime failure and repair

The alternate prefix `/root/pr100-native-alt-20261027` reused the slave's
existing Python and Torch base while retaining exact master scientific sources
and four model-weight files. Before E, qualification stopped at `torch_import`
with exit code 2. In-memory diagnostics exposed only error categories and the
missing library basename `libmkl_core.so.2`; after that library was installed,
the next import identified `libmkl_gnu_thread.so.2`.

The original provisioner followed ELF dependencies with `ldd`. That inventory
did not include the complete MKL family loaded dynamically at runtime.
Successful file copying, package metadata, or static dependency inspection
therefore did not establish an executable native pipeline.

Repair retained the isolated prefix and added fifteen exact master MKL CPU
libraries: core, scalar and AVX dispatch, LP64/ILP64 interfaces, thread
backends, and VML dispatch. It added no MPI or GPU libraries. Existing slave
locations were checked for exact source-hash matches before copying; none of
the fourteen remaining files could be reused locally. The verified core file
was retained. Four compressed streams copied the remaining files, followed by
complete hash verification. The family contained 456,479,536 bytes and its
repair completed in 278.67 seconds. Verified progress survived the earlier
uncompressed core-transfer timeout.

Every added library is bound by
[mkl-supplement-family-e.json](evidence/pr100_slave_runtime_repair_20261027/manifests/mkl-supplement-family-e.json),
SHA-256 `013428772117168e49e2cb3f2c82386b4f75892b70f788d11ebf903a3df6c5ca`.
Torch import, CPU matrix multiplication/SVD, and OpenMM CPU-platform import
then succeeded under the same wrapper and minimal environment used by E.
Those inert checks preceded the 32 actual profiles.

## Qualification and accepted differences

E used application source `51688c0a458f2aa401d7b1f569b2cf364f96aebc`,
32 distinct inputs, 32 physical CPU cores with affinity 0–31, hidden
accelerators, and one thread for each OpenMM/MKL/OMP/OpenBLAS context.
The Pi owned the launch and receipt collection. The guard allowed at most
345 seconds of child work and an eight-second descendant-cleanup budget.
The earlier qualification receipts were preserved under their original names.

| Evidence | Result |
| --- | --- |
| Completed profiles with finite metrics | 32 of 32 |
| Complete structure provenance | 32 of 32 |
| Runtime-inconclusive profiles | 0 |
| Native profile wall time | 69.91 seconds |
| Total guard wall time | 83.41 seconds |
| Worker exit code | 0 |
| Descendants after cleanup | 0; fixed point reached |
| Scientific and supplemental files after execution | Unchanged |

All 32 observations were rejected by existing scientific gates. A complete
numeric rejection qualifies execution, not the candidate. The receipt has
`authoritative=false` and `publication_enabled=false`; this experiment
published nothing.

The accepted runtime identity is
`6eaada274269a1e60b642240326d851830cb7dda262238ee2b9b8a3e6dc44e18`.
Four scientific source hashes and all four weight hashes match the reference.
The manifests also bind 5,369 original scientific files and 166 metadata files.
The actual runtime uses Torch `2.7.1+cu126`, NumPy `1.26.4`, OpenMM `8.5.1`,
ImmuneBuilder `1.2`, and theraprofnano `0.0.1`. Its Python/Torch base differs
from the master; the TNP and NanoBodyBuilder2 entrypoint shebangs were relocated.
The native wrapper pins the alternate plugin/library directories and thread
settings. Its SHA-256 is
`ab23f3e807a65753207d219a323d567924f9d00c8633c6cad811981dcd6e02e8`.

[qualification-approved.json](evidence/pr100_slave_runtime_repair_20261027/qualification-approved.json)
retains the root agent's explicit acceptance of those differences. The
approval makes no bit-identity claim. Actual validator-worker equivalence was
not measured by this qualification.

The root agent subsequently reported production source
`22407a4ae6324b8c051ef89f738eb74fddaad978` active at 10:02 UTC, epoch 25967,
attempt 5, with master 96 and slave 32 processing disjoint native work.
Its snapshot recorded 31 slave completions, two passing measurements and zero
inconclusive measurements. This dated
[root observation](evidence/pr100_slave_runtime_repair_20261027/activation-root-observation.json)
is separate from the E qualification and is not a final epoch result.

## Sequence availability and UID commitments

The [epoch audit](evidence/pr100_slave_runtime_repair_20261027/submission-yield-audit.json)
cross-checked worker completion counters, native journals, final continuation
sets, published pools, the final crew snapshot, and parsed Pi loop receipts.

| Count | Epoch 25965 | Epoch 25966 |
| --- | ---: | ---: |
| Completed fresh native evaluations | 1,396 | 1,167 |
| Completed repeat evaluations | 19 | 59 |
| First native measurements passing gates | 13 | 39 |
| Distinct scored candidates | 3 | 20 |
| Final eligible published pool | 3 | 18 |
| Published-pool yield per fresh evaluation | 0.21% | 1.54% |
| Registered crew at commit | 13 | 13 |
| Planned / committed UIDs | 3 / 3 | 13 / 13 |
| Failed commits in final receipt | 0 | 0 |

Epoch 25965's low UID count was limited by three available eligible sequences.
Epoch 25966 used the entire registered fleet. The final native hash set equalled
the published pool in both epochs, with no published hash marked invalidated.
The final epoch-25966 `upgrade_refused` outcome retained the same eighteen
eligible sequences; it did not hide missing UID submissions. A pool artifact's
`submission_count=0` is normal before the Pi binds its UID-independent sequences
to the current crew. It is not an on-chain commit count.

Across all completed measurements, the journals recorded:

| Native outcome | Epoch 25965 | Epoch 25966 |
| --- | ---: | ---: |
| PSH gate rejection | 1,176 | 443 |
| Aggrescan numeric-cap rejection | 73 | 618 |
| Developability red flags | 139 | 88 |
| Passed measurement | 26 | 76 |
| Timeout / inconclusive | 1 | 1 |

The worker label `aggrescan_unavailable_or_failed` alone is ambiguous. The
allowlisted observation journal distinguishes finite completed cap rejections
from execution failures. These epochs each had one timeout, not hundreds of
Aggrescan runtime errors. Their challenge requirements differed, so the two
yield rates are not a controlled comparison of generation quality.
Commit receipts establish the submit-host result, not validator acceptance,
ranking, or reward.

## Orchestration lessons retained

The native supervisor ran for 2,510.99 seconds in epoch 25965 and 2,233.88
seconds in 25966. Sequential scoring occupied another 287.14 and 464.22
seconds respectively. After the final attempt, 442.19 and 668.63 seconds
remained before publication cutoff. Those tails were below the configured
695-second native-continuation and 755-second new-sequence admission floors;
they were conservative admission stops, not evidence of stuck workers.

One avoidable retry did occur in each epoch. A fully measured candidate rejected
by archive novelty was absent from the finished candidate set, so its completed
confirmation checkpoint was queued again. Attempt 2 performed zero new native
measurements, consuming 52.56 seconds in 25965 and 26.42 seconds in 25966.

Fix `52abf73703a3c06262b384f9839dae24c799a88e`, integrated as
`22407a4ae6324b8c051ef89f738eb74fddaad978`, retires such checkpoints only after
complete classification and novelty checks bound to the exact challenge,
epoch, target and archive. Incomplete, failed, unresolved, or binding-mismatched
work retains recovery. A multi-host result claims full classification only
when every participating peer is complete and cleaned up. Ninety-one focused
tests passed for that fix. Admission reserves and scientific gates were unchanged.

Future throughput work should measure native pauses during GPU scoring and
the CPU occupancy of small confirmation batches. Neither the 32-profile
qualification nor a configured 96+32 fleet proves 4,800 fresh completions per
epoch. Keep fresh work, repeats, unstarted inputs, gate rejection, execution
failure, publication, UID commitments and validator outcomes separate.
