# Exact archive transport caching

The controller previously resent the same public archive to each phase. The
epoch 25969 archive is 2,316,063 bytes; its compressed request is about 1.17 MB.
Recorded Pi-to-slave launch preparation took 43–47 seconds, compared with about
9 seconds to master. Repeated chain-clock calls measured about 1.2 seconds.
These timings distinguish input transfer from native scientific computation.

The [retained memory-only transfer benchmark](evidence/pr100_archive_cache_20261027/wire-benchmark.json)
used that exact public archive: the compressed full request was 1,171,384 bytes
and the reference 516 bytes. One paired transfer measured 37.53 versus 9.17
seconds on slave, and 6.02 versus 4.06 seconds on master. These are SSH transfer
measurements, not a claim about full-epoch throughput; native input and source
verification costs remain.

`pr100_archive_cache` changes only transport. The durable phase intent still
contains the full archive and its original file, runtime, source and challenge
bindings. The first upload populates `<remote_root>/archive-cache-v1/SHA256.csv`.
Only an acknowledged host/digest can subsequently receive a reference instead
of the full bytes. Each hit verifies size, SHA256, the owning epoch path, and
`job.archive_sha256`, then copies the identical bytes into the isolated attempt.
Cache files are atomically published read-only; attempt files are independent.

Missing or corrupt entries trigger one full upload under the same launch
identity and absolute transport deadline. Clock refresh, stale-clock checks,
source verification, launch ownership and cleanup requirements remain intact.
An existing launch or uncertain launch gap is checked before cache hydration.
Failure to write an optional cache entry preserves the verified original full
payload and does not acknowledge a cache hit. A cache may be removed between
uses without losing scientific work; the next call uploads its bound bytes.

No candidate measurement, refined structure, novelty verdict or scoring result
is cached by this feature. Fresh evaluation accounting is unchanged. Set
`archive_cache_enabled=false` to retain the original full-upload path.

The activation helper now allows the pending-attempt novelty migration exception
only when the proposed qualified novelty runtime IDs actually change. A code
upgrade preserving those IDs requires `active=None`, even if old receipts are
clean. The existing qualified native/scoring settings are preserved.

Mandatory shipping tests cover cache hit/miss, corrupt and unsafe entries,
atomic concurrent writers, per-host acknowledgement, immutable attempt copies,
same-identity fallback, deadline exhaustion, existing owner/crash-gap protection,
and the unchanged-novelty handoff fence. Inert transfer measurements use the
actual public archive and a memory-only SSH receiver; no scientific calls or
production cache writes are part of that benchmark.
