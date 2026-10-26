# PR100 bounded live trial, epoch 25956

The first bounded live artifact was published on Pi for human PD-L1 (Q9NZQ7),
IGHV3-23, CDR3 length 18. Publication means the artifact was ready for the
submission loop; it does not establish that a commitment or validator reward
occurred. The final integrated release was `3dc498ea1c71d65e9a9da00d4344bd1faebc3c26`,
with 1,094 passing shipping checks.

## Measured supply and checks

- Sixteen existing candidates with earlier passing native measurements received
  fresh baseline CPU profiles. Four passed; twelve failed the PSH threshold.
  The batch took 104.94 seconds. Earlier cache acceptance was not substituted
  for these new measurements, and this result alone does not identify the
  cause of the changed native measurements.
- The unchanged conditional generator emitted 64 fallback proposals for the
  actual challenge. Sixty-three passed the pinned annotation/basic rules;
  one failed basic sequence checks. This took 4.31 seconds.
- All 79 annotated candidates passed the unchanged all-archive novelty check
  against 8,469 current Q9NZQ7 records. The full check took 234.15 seconds.
- The four native passers received one fresh target-scoring observation with
  five diffusion samples and retained structures. The existing fast baseline
  uses 35 folding steps, while the upstream default uses 200. Its scoring
  equivalence has not been established.

## Publication semantics

The artifact and its checkpoints use `live_trial`. They explicitly set
`exact`, `runtime_matched`, `validator_matched`, and
`scoring_equivalence_qualified` to false. Raw finite metrics, actual settings,
source hashes, epoch, target, and measured validity evidence remain attached.
The selection diagnostic is a one-observation dense rank sum within this
four-entry trial. It is not a ten-epoch or multipass comparison.

No complete-fleet minimum, twenty-pass confirmation, 4,800-candidate capacity
proof, or warm-model research qualification is required for this bounded
trial. Actual sequence, challenge, native developability, archive novelty,
and submission-window constraints remain enforced.

## Runtime correction and handoff

`annotation_identity()` previously depended on interpreter-specific
`ast.dump()` output. Python 3.12 includes an empty `type_params` field that
Python 3.11 lacks. The same scientific annotation could therefore be rejected
by Pi. Canonicalizing this empty field preserves the existing 3.12 identity
and allows the actual Pi 3.11 verifier to accept the same evidence. Biological
thresholds and annotation semantics are unchanged.

The annotation-only environment lacked a search dependency. A private overlay
received the existing native runtime's public Pydantic dependency files and
the already pinned logging shim. Annotation package versions were unchanged.
Archive fetching used the pinned fetch implementation from a working system
Python, then bound the downloaded bytes before the annotation/search process.

Pi collected hash-bound input snapshots before the master became temporarily
unreachable. A detached, bounded Pi exporter rechecked source identities,
input hashes, the live epoch-start hash, archive hash, and all four checkpoints
under the actual Pi interpreter. It published the reviewed bytes atomically
with 150 blocks remaining. Mac or master availability is not needed for this
already prepared artifact to reach the submission loop.

The artifact SHA-256 is
`b5a6584fd409b1bd312e6d54672dc384fb7ab8350754403c988e6af5736c2dd3`.
Pi receipts and snapshots are under `/home/thomasg/pr100-live-25956`.
Local receipts are under
`/Users/tvgdeena/nova-reports/pr100-readiness-20261026/live25956-validity-export`.

Next decisions should use actual validator acceptance and quality outcomes.
This trial establishes an operational partial-fleet path, not near-optimality
or a promised score.
