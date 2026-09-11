# Claude handoff: validator-faithful fast nanobody screening

## Objective

Integrate the incoming approximately 10x faster BoltzGen scorer without losing
validator fidelity, then use the added budget to search more structural basins
and supported terminal/framework haplotypes. The live miner is nanobody-only and
publishes as UID 8. Do not destroy the GPU box.

The repository is `/Users/tvgdeena/nova-nanobody`, branch `nanobody-opt`. Deploy
only committed work with `ansible/nanobody.yml`; both scientific environments
must continue to be installed and invoked through the committed Pixi lockfile.

## Current live state

- GPU host: `ssh -p 20868 root@ssh9.vast.ai`.
- Live producer tmux session: `nova-nanobody-producer`.
- The normal live profile is 320 candidates in tranches of 20, five finalists,
  and five structural basins.
- Recent confirmed results for UID 8 were rank 3 / score 79 in epoch 25052 and
  rank 4 / score 85 in epoch 25053.
- Keep the current scorer live until the replacement passes the canaries below.
- Never combine an old-protocol observation with a new-protocol observation.

Relevant commits, oldest to newest:

- `e8a0cb2`: cap exact-scorer CPU thread pools in Pixi.
- `0e9769f`: rank terminal routes with historical priors.
- `78a1906`: match validator diffusion-sample selection and retain both poses.
- `c03a91c`: make the exact scorer work as a direct script entry point.

## Validator fidelity correction

One BoltzGen observation contains five diffusion samples. It is not the mean of
those samples. Current upstream validator code applies two independent choices:

1. The writer chooses `argmax(0.8 * iptm + 0.2 * ptm)` and writes that full
   complex into `refold_cif/`.
2. Analysis chooses
   `argmax(0.8 * design_to_target_iptm + 0.2 * design_ptm)` for the five reported
   confidence metrics and its in-memory contact analysis.

These choices can disagree. The repository now records both selections from
`fold_out_npz`, writes the analysis-selected full complex to
`refold_analyze_cif/`, and treats the analysis-selected CIF as the confidence
pose. The writer-selected pose remains a separate diagnostic. The expected
protocol is `validator-dual-sample-v1`, defined in
`nanobody_opt/diffusion_sample_selection.py`.

Every replacement scorer must emit the existing CLI/output contract used by
`score_boltzgen_waves.py`:

- accept `--sequences`, `--config`, `--target`, `--gpus`, `--shards-per-gpu`,
  `--batch-size`, `--work-root`, `--out`, and, when requested,
  `--structure-dir --require-structures`;
- set top-level `scorer_protocol` to `validator-dual-sample-v1`;
- return all ten validator metrics for every requested sequence;
- retain, for each sequence and target, `path` (analysis-selected pose),
  `writer_path`, hashes, and `sample_selection` with both indices and all five
  sample score vectors;
- prove the aggregate five confidence metrics equal the selected analysis
  sample exactly;
- fail without promoting partial output on a missing sequence or artifact.

`run_live_producer.py --exact-scorer PATH` now carries a replacement scorer
through both one-shot tranches and repeat waves. Ansible exposes the path as
`nanobody_exact_scorer`. Do not bypass `score_boltzgen_waves.py`; its atomic
wave validation and resume behavior are part of the publication safety model.

## How to spend the 10x budget

Start with the disabled profile
`ansible/vars/nanobody-high-throughput.yml`. It budgets 2,400 unique one-shot
observations over 40 basins, retains 20-candidate publication checkpoints, and
allows repeat evidence for eight finalists. This is intentionally breadth-first:

- roughly 80% of proposals should explore historically strong masked cores and
  their supported terminal/framework haplotypes;
- keep a bounded novel-core/CDR lane so the search can escape an old basin;
- do not spend the multiplier by scoring 320 close neighbors ten times;
- after each tranche, update the mutation database and reallocate subsequent
  tranches across under-sampled competitive basins;
- preserve eight independent full observations for anything published as
  qualified. A one-observation provisional publication may still be replaced
  only by better evidence under the current publisher rule.

The 2,400 count is a starting envelope, not a promise. Measure median and p90
tranche duration with the new scorer on all ten GPUs, reserve at least 55 blocks
for commitment, and calculate the safe per-epoch budget from observed wall time.

## TNP/Layer-2 work: required, and not a serial post-pass

The faster exact scorer changes the bottleneck. Do not run full TNP serially on
thousands of proposals before GPU screening, and do not wait for the whole GPU
screen to finish before validating a leader. Build an overlapped pipeline:

1. Run Layer 1 and cached, batched IgBLAST over the complete proposal pool.
2. Exact-score IgBLAST survivors in 20-candidate tranches.
3. Immediately enqueue the leading candidates from each completed tranche for
   full TNP while the next exact tranche uses the GPUs.
4. Cache full TNP by sequence and merge results atomically into a durable
   Layer-2 checkpoint.
5. Never provisionally or strictly publish a sequence until that exact sequence
   has `layer2_passed: true`; TNP failure removes it from finalist contention
   and the next ranked candidate is tried.

TNP performance TODOs and acceptance criteria:

- Keep ImmuneBuilder/NanoBodyBuilder2 and supporting worker processes warm
  across candidates; do not reload models per sequence.
- Verify ImmuneBuilder is actually using an assigned GPU. Partition it from the
  exact-scoring workers or schedule it only when GPU headroom exists.
- Preserve and verify the committed `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` Pixi `metanano`
  activation settings so ANARCI, DSSP, OpenMM, NumPy, and subprocesses cannot
  oversubscribe 128 cores.
- Retain the existing rolling queue: start a replacement as soon as a TNP task
  finishes instead of waiting for rigid waves.
- Reuse the existing sequence-addressed IgBLAST and TNP caches and verify cache
  entries are measured results, never execution failures.
- Profile the phases separately: IgBLAST/ANARCI, ImmuneBuilder prediction,
  OpenMM refinement, DSSP, surface hydrophobicity, and charge patches. Record
  median/p90 wall time, CPU time, GPU utilization, and cache hit rate.
- Investigate safe persistent worker pools for the deterministic CPU analysis.
  Confirm that outputs are byte- or metric-equivalent before enabling them.
- Use full TNP on tranche leaders and a small reserve shortlist, not all 2,400,
  unless measured overlap proves it fits comfortably inside an epoch.
- Add an integration test in which the first exact leader fails TNP and the race
  advances to the second leader without publishing the failed candidate.
- Add a resume test proving a completed TNP cache entry survives producer and
  epoch-worker restarts.

`filter_layer2.py` already supports `--igblast-only`, a rolling asynchronous TNP
queue, and sequence-addressed caches. The missing work is orchestration: overlap
the two stages and provide a durable full-TNP result file that the independent
publisher can validate. Do not weaken `publish_checkpoint.py` to accept an
IgBLAST-only record.

## Activation gates

Before enabling the high-throughput profile:

1. Run the full local test suite and Ruff on touched Python files.
2. Run `ansible-playbook --syntax-check ansible/nanobody.yml`.
3. On the GPU box, run the existing dual-sample Ansible canary.
4. Score a fixed multi-sequence cohort with old and fast scorers from the same
   locked Pixi environment. Compare all ten metrics, both selected sample
   indices, both retained structure hashes/coordinates, and failure behavior.
5. Require identical selection indices and metrics within an explicitly stored
   numerical tolerance. Investigate any mismatch; do not average it away.
6. Run at least one end-to-end dry iteration with publication pointed at a
   temporary checkpoint and prove Layer 1, full Layer 2, novelty, dual-pose
   exact evidence, and atomic replacement all remain enforced.
7. Commit, then deploy. Verify `/root/artifacts/nanobody_deployed_commit.txt`
   equals local `HEAD` and the producer tmux session stays alive.

Example dormant deployment after all gates pass:

```bash
ansible-playbook -i ansible/inventory.vast.yml ansible/nanobody.yml \
  -e @ansible/vars/nanobody-high-throughput.yml \
  -e nanobody_producer_activate=true
```

Confirm the actual inventory filename before running the example. Do not encode
the current Vast host, key, epoch, target, or molecule in committed strategy.

## Verification commands

```bash
git status --short
git log -5 --oneline
python -m pytest -q
ruff check nanobody_opt tests
ansible-playbook --syntax-check ansible/nanobody.yml
ssh -p 20868 root@ssh9.vast.ai \
  'cat /root/artifacts/nanobody_deployed_commit.txt; tmux has-session -t nova-nanobody-producer'
```

After a new wave completes, inspect `exact_all.json`. It must have
`scorer_protocol == "validator-dual-sample-v1"`, and every retained structure
record must contain `path`, `writer_path`, and `sample_selection`. Report the
fraction where writer and analysis sample indices disagree; that is diagnostic,
not an error.
