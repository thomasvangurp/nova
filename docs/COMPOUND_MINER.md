# Competitive mechanism-0 molecule miner

The experimental diffusion-free Boltz cascade is documented in
[DISTILLED_ORACLE.md](DISTILLED_ORACLE.md). It is not promoted until it passes
the frozen 1% teacher-budget and constrained-decision gates described there.

The miner separates broad exploration from expensive structural scoring:

1. Public P40261 leaderboard observations train a target-conditioned proxy on molecular fingerprints, descriptors, reagent-role effects, and reagent-pair effects.
2. Proposal generation mixes substitutions around successful public structures, joint draws from learned component priors, and a persistent exploration fraction.
3. Exact validator gates remove invalid and historically non-novel candidates.
4. A broad acquisition shortlist is ranked by the local public Boltz-2 fork in
   reduced-step mode, then its leaders are rescored with validator-matched
   Boltz-2 settings.
5. The best chemically unique, entropy-valid 20 are written atomically. The chain miner reads only this validated checkpoint.

This avoids using exact Boltz-2 as the broad search engine. NVIDIA NIM 1.8.0
requires at least 48 GB VRAM per GPU, but its service output did not reproduce
the competition's pinned public Boltz scorer closely enough to be the final
oracle. Upstream Boltz-2 runs on 24 GB RTX 4090 cards and is the authoritative
local implementation.

The fork exposes `skip_affinity_structure`: the first checkpoint's persisted
coordinates are passed directly to the affinity checkpoint, avoiding its second
diffusion generator. This and reduced diffusion steps are surrogate-only
optimizations. Defaults preserve upstream behavior, and the final shortlist is
always rescored without either approximation.

## Setup

Use the official environment and point all local tools at the supplied database without copying or mutating it:

```bash
export NOVA_COMBINATORIAL_DB=/absolute/path/to/molecules.sqlite
python -m compound_miner.cli sync-history
python -m compound_miner.cli train --trees 256
```

Download the P40261 novelty archive from `Metanova/Submission-Archive` and run a search for each reaction.

```bash
python -m compound_miner.cli search \
  --reaction 5 \
  --historical-molecules /absolute/path/P40261_molecules.csv \
  --proxy-only
```

On a prepared GPU worker, rank a fixed acquisition portfolio and rescore its
leaders exactly. The integrated command uses reduced Boltz only for acquisition,
uses the validator-matched configuration for the final shortlist, and writes the
submission checkpoint only if its exact constrained score reaches the frozen
per-reaction baseline:

```bash
python tools/build_exact_checkpoint.py \
  --reaction 5 \
  --archive archives/P40261.sqlite \
  --result artifacts/result_rxn5.json \
  --work-dir artifacts/qualification/rxn5 \
  --gpus 0,1 --shards-per-gpu 3 --scratch-root /dev/shm
```

The two underlying stages are equivalent to:

```bash
python tools/score_boltz_batch.py \
  --archive archives/P40261.sqlite --reaction 5 --limit 256 \
  --gpus 0,1 --shards-per-gpu 4 \
  --recycling-steps 1 --sampling-steps 20 \
  --diffusion-samples-affinity 1 --sampling-steps-affinity 20 \
  --skip-affinity-structure --output artifacts/fast_scores.json

python tools/score_boltz_batch.py \
  --archive archives/P40261.sqlite --reaction 5 --limit 64 \
  --records-json artifacts/fast_scores.json \
  --gpus 0,1 --shards-per-gpu 3 \
  --output artifacts/exact_scores.json

python -m compound_miner.cli qualify-exact \
  --reaction 5 --scores artifacts/exact_scores.json \
  --reduced-scores artifacts/fast_scores.json \
  --archive archives/P40261.sqlite \
  --result artifacts/result_rxn5.json
```

`artifacts/result_rxnN.json` is a release artifact, not a search scratch file.
It embeds the exact scorer identity, hashes of the reduced and exact score
artifacts, exact and selected counts, achieved score, and the frozen baseline.
The provider rejects legacy files, proxy/reduced scores, incomplete exact
shortlists, duplicate molecules, reaction mismatches, and baseline misses.
Proxy search may still write an isolated diagnostic result, but that file cannot
be submitted.

Reaction 5 uses context-conditioned substitutions. A replacement component is
ranked from its own role evidence and its observed interactions with each
retained component; a strong fixed B-C pair cannot falsely credit an unrelated A.
Only validator-matched exact rows may become exact-parent neighborhoods.

Run all five reaction archives before going live because the allowed reaction is derived from the epoch boundary hash. Validate every checkpoint independently:

```bash
python -m compound_miner.cli validate-result \
  --reaction 5 \
  --result artifacts/result_rxn5.json \
  --historical-molecules /absolute/path/P40261_molecules.csv
```

The live miner reads `artifacts/result_rxn{reaction}.json` by default. Override this with `NOVA_CANDIDATE_RESULT`, retaining the `{reaction}` placeholder. If a qualified exact 20-member checkpoint is unavailable, it fails closed instead of submitting proxy-ranked or random molecules.

## Performance measurements

Measure three layers separately:

- proxy quality: chronological top-100 recall and achieved oracle mean;
- candidate throughput: constructed, valid, and historically novel candidates per second;
- exact scoring: molecules per second at one through four resident workers per
  GPU;
- fast scoring: rank correlation, top-k overlap, and recall against the exact
  score on the identical fixed molecule set.

On 2x RTX 4090, three and four workers per GPU both reached about 0.103 exact
molecules/second. A reduced-step 18-molecule test reached Spearman 0.959 and
retained all exact top five, but it remains a surrogate pending broader
cross-reaction validation.

The first production reaction-5 qualification exposed why exact reranking is a
mandatory release boundary. On 1,825 fixed candidates, reduced versus exact
Spearman was only 0.328 near the frontier. Exact-scoring reduced ranks 1-64 gave
an entropy-valid top-20 sum of 2.198937; adding ranks 65-128 replaced three
members and raised it to 2.219774. The latter beats the frozen 2.203281
reaction-5 baseline by 0.016493. Each 64-molecule exact pass took 185-196
seconds on 8x RTX 4090 with three resident workers per GPU.
