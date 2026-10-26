# Molecule conformer gate: isolated cutover correction

The new upstream molecule gate tries `ETKDGv3` once, then retries the same
hydrogenated molecule with `useRandomCoords=True` only when the first result is
`-1`. Two failures reject the molecule; exceptions also reject it. No seed,
thread count, embedding parameter, threshold or retry count is changed locally.

The immutable public source is Git blob
`7d309072c82c8ba72ddeb61b1fdc1f3b292f00e1` from `metanova-labs/nova`.
Its Git object digest was verified including the blob header. Full file SHA-256:
`830136d3f4c87ecca304c7599e5878c10699274272f9b66792bb3ac72ac88b0b`.
The exact `is_boltz_safe_smiles` source-segment SHA-256 is
`bf66ddbdb94b668cebf5ecb75544567bdd752009f68f8b3371d1e5fedd240f04`.
The source is bound to the immutable blob, not assumed to belong to the PR merge
commit: main had additional commits when this change was observed.

`utils/molecules.py` contains the exact updated function while retaining unrelated
local file contents. `tools/molecule_gate.py` extracts only three pinned pure
functions from the selected validator source. It does not import `utils`, execute
its dotenv/application setup, or trust an already-loaded `utils` package.
The gate requires RDKit 2024.9.4 as specified in validator requirements and
records the distribution version and RDKit/AllChem/rdBase/rdDistGeom module-file
digests. These records are partial runtime identity, not a complete native
dependency closure or proof of conformer success.

Bank release requires the bound gate from epoch 25955 onwards, including offline
`--no-archive` runs. An unavailable source/runtime refuses the release. Chemical
checks run before pool selection; the release records gate identity and counts.
Builder/rescore paths inherit the same helper when their existing validator-tree
gate is available. Historical/archive checks and portfolio-size rules remain
independent requirements. The separate Pi commit gate remains necessary because
bank artifacts can predate the current validator rule.

The prepare-only `scripts/patch_pi_molecule_gate.py` accepts only the audited Pi
`validator_rules.py` SHA-256
`0833c725daefa191a9f97cb18d63759b834df5f4cedcf7cbdf21b10683cf8c15`.
It also checks the existing `boltz_safe` function SHA-256
`5d47685526fbba5a7c36930403e7c763211fadc32ceeeaa72e974a9dc6b15dc1`.
It replaces exactly that AST byte span with the upstream function renamed to
`boltz_safe`, keeping the old local RDKit import pattern. All other source bytes
are preserved. CLI output is a new private candidate file; the script never
installs it, changes the consumer/signer, or signs/submits anything. Changed old
source and double replacement are refused.

Mandatory CI exercises first success, retry success, double failure, exceptions,
parse/name rejection before embedding, source/runtime mismatch, stale ambient
imports, exact Pi span preservation and the post-activation release tripwire.
These use explicit fake RDKit boundaries. The CI interpreter has no RDKit
distribution, so they do not establish real chemistry acceptance or timing.

Before deployment, independently verify the actual Pi/bank helper and RDKit
origins/version, then run the exact gate on the frozen proposed portfolios under
a finite CPU/time budget. Measure real wall/CPU cost and every first/retry/failure
outcome. Prior scoring success and mocked tests cannot replace that evidence.
The added upstream function has no explicit random seed or timeout, so deadline
cost and repeat variation remain empirical questions. No native gate execution,
package installation or remote deployment was performed for this candidate.

This is a separate branch based on d766fa8. The native N0/N1 experiment continues
to use its original d766 source/runtime manifest and its existing review.

## Preparation process control (separate operational delta)

The Pi helper is also called by bank release and portfolio preflight. The live
bank has thousands of eligible rows in some reactions; the measured 20-row
portfolio is not a full-bank throughput qualification. Existing nested
`subprocess.run` timeouts kill immediate children only and can orphan preparation
work. The separate pinned loop patch routes its three `make_fleet` launches
through the existing `epoch_deadline.run` process-group cleanup. The first uses
the epoch's commit-reserve cutoff; both background branches use their existing
`collect_at`. Two seconds are reserved for the one-second TERM grace and KILL.
Expired work returns 124 and cannot publish a successful result. All selection
arguments and the rest of `handle_epoch` are unchanged (AST regression checked).

The installed loop contains an additional late-compound branch absent from the
tracked loop; patching its exact audited bytes preserves that branch. Do not
replace the live loop with the entire tracked file. A loop restart is required
to load the change and must occur only after a fresh idle-phase/no-child audit,
never during preparation or commit. Consumer/signer hold files stay unchanged.
