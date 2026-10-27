# PR100 novelty runtime qualification, 2026-10-27

`novelty-qualification.json` is the complete sanitized qualification receipt. Both hosts replayed the same 22 already-native-measured inputs against the original epoch 25967 attempt-6 archive snapshots. Both annotated all 8,553 old and 8,575 fresh entries, and retained all 22 inputs under unchanged novelty thresholds. Full numerical verdict arrays agree between hosts. Each runtime executed a known-positive `Chain.batch` before archive classification. Both outer supervisors exited zero and reached a cleanup fixed point with zero descendants.

| Host | Runtime identity SHA-256 | Old archive seconds | Fresh archive seconds | Supervisor seconds |
| --- | --- | ---: | ---: | ---: |
| master | `f6f27ad1bec8edaa47d61dbd1f444eb3e4fb9a3107589f5d903e1248f544ea30` | 105.8773 | 106.0850 | 212.0917 |
| slave | `9851767c681498521c7e86cc0f46159d32e1115acf674c71f1f9ada863d22fb9` | 159.5018 | 156.4803 | 317.1613 |

The dedicated runtime uses the same exact public asset manifests on both hosts. Host Python binary hashes differ, so their identities differ. Native profiling, Torch installations, scoring and checkpoint execution identities remain unchanged. This is qualification of the recorded algorithm/runtime combination, not a claim of bit identity or parity with an external validator.

## Receipt binding

`artifact_hash_encoding` is `canonical-json-v1`. Every `hosts.<host>.artifact_sha256.<key>` is the SHA-256 of the corresponding **embedded object**, encoded with `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()`; Python's default `ensure_ascii=True` and UTF-8 apply. These are not hashes of the remote JSON files' original formatting. The same encoding hashes the embedded runtime `identity` into `runtime.identity_sha256`.

The identity's `sources` map binds the exact novelty algorithm/helper source files, which must match the candidate release. The aggregate's `candidate_source_commit` is provenance, not a requirement that unrelated activation/documentation changes retain that commit ID. `qualification_helper_sha256.worker` binds the byte-identical archived worker. `result.input_sha256` binds the original native input rows without exposing them. Every verdict has only a sequence hash, numerical comparison result and boolean verdict. Archive byte hashes are retained independently from row counts.

`novelty-runtime-config-overlay.json` contains exactly the four approved `hosts.<host>.novelty_runtime` fields. The interpreter is the private `bin/novelty-python` wrapper; directly using `bin/python` omits its qualified HMMER library closure. This overlay is evidence for the rollout owner, not an automatic deployment operation.

Remote originals on both hosts are `/root/pr100-novelty-runtime-20261027/replay-25967-v2/result.json` and `supervisor-receipt.json`. The aggregate retains only the allowlisted supervisor fields. The original snapshots and input remain on the controller and qualification hosts; no sequences or raw streams are copied into this repository. Runtime identity binds the numerical environment by hash, not by storing the full environment.

## Exact helper archive and dependency lessons

`helpers/` contains byte-identical one-off sources used during provisioning, qualification and readback, with hashes in `manifest.json`. They are historical evidence, not a supported general-purpose deployment interface. Their absolute paths, fixed release, strict host aliases and existing trust references document the original run. The scripts must not be executed blindly against a new fleet. Copying or reinstalling assets alone does not renew qualification.

The initial bulk provision script is retained because later HMMER provisioning extracts its exact safe installer. Its first transfer did not finish; the resume helper used verified public archives and existing host trust. The pydantic builder's inherited opening comment incorrectly names the slave interpreter; its actual configuration dependencies came from the master installation, as recorded by the verified runtime identities. This archive preserves that original file unchanged.

The first isolated positive annotation correctly failed because a package list and hmmscan binary did not include its dynamic-library closure. The repair copied only the exact public `libmpi.so.40`, `libopen-pal.so.40`, `libopen-rte.so.40` and `libz.so.1` assets from the previously qualified annotation installation, plus a private interpreter wrapper. The resulting third asset manifest and actual execution environment are part of both approved identities. No native scientific installation was upgraded.

Some transport calls timed out after the remote operation had started or completed. `already_exists` prevented duplicate qualification ownership, and final success was established from the actual bounded supervisor and output artifacts. A transport exit alone is insufficient proof that a remote job never started. The retained v2 launch used eight CPU slots, a 550-second total deadline for two full snapshots, hidden accelerators, and no native profiles or publication. The production supervisor owned the worker, annotation children and cleanup. Each individual novelty check finished within the existing 300-second phase reserve.

`shipping-validation.json` records the two committed source builds, each passing 1,591 shipping tests with zero failures, errors or skips. The later root integration and activation helper require their own full shipping checks; this evidence does not substitute for them. The manifest records the archived source and evidence file bytes. Preservation of these files made no production changes.
