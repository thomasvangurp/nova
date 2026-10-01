# Nanobody submission-integrity incident — 2026-09-23

## Symptom

The dashboard reported epoch 25294 as `9 / 11 at epoch close · 13 now` even
though the Pi reported successful commitments. This was not a dashboard count
error: only nine owned UIDs appeared in the scored public field.

## Reconstruction

- Pi planned and successfully sent 12 commitments for epoch 25294.
- UID 52 left our roster before epoch close, so the correct held-at-close
  denominator was 11.
- UIDs 6 and 59 were held at close but did not score.
- The GPU publisher was still launched with `--commit-command`, despite the Pi
  having become the sole intended writer. In epoch 25295 it demonstrably found
  eight matching commitments and rewrote three more at blocks 9131830–9131840,
  at the edge of the validator blackout. This is the matching mechanism for a
  Pi-success/GPU-success epoch that nevertheless loses scored UIDs.
- The live producer was also refusing work at 16 GiB free because its reserve
  was 20 GiB. `/root/live` held 137 GiB of historical epoch trees.
- The handoff pool had a floor of 12 while the live roster had grown to 13.

## Root fixes

1. The GPU publisher is artifact-only. Canonical Ansible launch arguments no
   longer include `--commit-command`; Pi remains the sole commitment writer.
2. The watchdog now captures the argv of the running artifact-only publisher
   and refuses to persist it if a commit command is present or the sequence
   reserve is below 16. A restart therefore cannot silently restore the old
   dual-writer configuration.
3. The producer exports a 16-sequence UID-independent pool. Pi binds that pool
   to its fresh roster.
4. Pi prefers the provider's discovered direct SSH endpoint, retaining the SSH
   proxy only as fallback. The observed artifact pull fell to eight seconds.
5. Pi refreshes the metagraph after the artifact pull and immediately before
   planning. A no-write epoch-25296 rehearsal showed why: 13 UIDs were planned,
   but one UID turned over before the ownership check and only 12 remained.
6. Slot purchases freeze for the final 90 blocks of every validator epoch, 15
   blocks before Pi opens its end-75 submission window. This prevents our own
   opportunistic registration from changing the roster underneath a commit.
7. Historical epoch directories are retained for 16 epochs. Older directories
   are removed only if every `exact_all.json` is byte-matched to a complete,
   sealed database run and every expected observation has an audit-artifact
   row. The first guarded pass removed 148 epoch trees (141.8 GB); free space
   increased from 16 GB to 149 GB. The normalized observation database was
   backed up on Pi immediately before cleanup.

## Verification

- 123 focused publisher/producer/submit tests passed before GPU deployment.
- 114 focused submit/slot/producer tests passed after the roster fixes.
- Live GPU publisher: artifact-only, minimum pool 16.
- Live watchdog argv: artifact-only, minimum pool 16.
- Live artifact: epoch 25296, 16 distinct pool entries.
- Pi no-write plan: 13/13 sequences available, no missing-sequence UIDs.
- Producer and publisher sessions are active; all four GPUs resumed work.
- First epoch protected by the complete fix: **25296**.

