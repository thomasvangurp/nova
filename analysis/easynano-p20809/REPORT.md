# P20809 EasyNano basin experiment

Completed 2026-09-22. This was an isolated, non-production experiment.

## Question and result

We tested whether EasyNano recognizes the public 5EvpaL11 basin anchor as
better than our anchor, and whether its local differentiable search improves
either anchor. It does not reproduce the validator preference for the rival
anchor.

Every sequence was evaluated with full ESMFold2 on the same diffusion seeds
0, 1, and 2. Unchanged anchor sequences were excluded from the generated-design
comparison.

| Basin | Anchor mean ipTM | Best mutated mean ipTM | Change | Distance |
|---|---:|---:|---:|---:|
| Ours, epoch 25242 | 0.6395 | 0.6957 | +0.0562 | 1 |
| 5EvpaL11, epoch 25109 | 0.5954 | 0.5616 | -0.0338 | 1 |

The best mutation from our anchor improved two of three paired seeds
(+0.0157, +0.1688, -0.0159). The best mutation from the rival anchor improved
one of three (+0.0022, -0.0532, -0.0505). EasyNano therefore offers one local
proposal worth passing to the actual validator-shaped scorer, but it neither
explains nor reproduces the rival's public advantage.

## Why this differs from the validator

EasyNano reports full-ESMFold2 complex ipTM and pTM after optimizing a
distogram/epitope-proximity proxy. The public validator uses BoltzGen and ranks
the complete field independently on ten metrics using dense ranks:

- confidence: `design_iiptm`, `design_ptm`, `design_to_target_iptm`,
  `min_design_to_target_pae`, and `interaction_pae`;
- physical interaction: refolded PLIP H-bonds, refolded PLIP salt bridges, and
  refolded delta SASA;
- developability: liability score and liability violation count.

It sums those ten ranks without fitted weights and minimizes the total.
EasyNano ipTM is therefore neither the validator's `design_iiptm` nor its final
score. In the retained five-pass BoltzGen replay, the rival has the reproducible
edge specifically on validator `design_iiptm`: 17/20 pairs and +0.01039 on
average. EasyNano's reversal is evidence that it is a proposal generator, not
a replacement ranker for this competition.

## Target and search

- Target: UniProt P20809; structure 6O4O; chain A; 168 residues.
- Configured epitope PDB residue IDs: 31–35, 37–38, 40, 88, 135–138, 142.
- Corresponding zero-based EasyNano chain-token indices: 29–33, 35–36, 38,
  86, 133–136, 140.
- Search: two public anchors, three design seeds each, 60 steps per run.
- Fair rerank: 18 unique anchor/candidate sequences × three common seeds = 54
  full-model evaluations.

The result is small-sample directional evidence, not biological validation.
The proposed Hamming-1 sequence should be evaluated with the exact fast
BoltzGen path across the full ten-metric objective before submission.

## Implementation and infrastructure

The private fork is `thomasvangurp/EasyNano-NOVA`, revision
`ac4a53de53e53a879799867da113c7f25674d263`. It adds custom-anchor handling,
P20809 metadata, current Transformers/ANARCI compatibility, native CDR cysteine
preservation, deterministic evaluation-safe gradient checkpointing, and GPU
memory release between runs.

Provisioning and execution used Ansible plus the locked Pixi environment on an
isolated A100 80 GB host. Production was untouched. All artifacts were
checksum-verified locally before the host was stopped; provider state is
`exited` / `stopped`.

## Reproducible outputs

- `results/artifacts/monte_carlo_summary.json`: final shared-seed comparison.
- `results/artifacts/monte_carlo_evaluation.json`: all 54 full evaluations.
- `results/artifacts/design_runs.json`: six EasyNano optimization runs.
- `experiment.json`: anchors, target mapping, epitope mapping, seeds, and fork
  revision.
- `rerank_mc.py`: shared-seed, mutation-only aggregation.

