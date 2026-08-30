# Competitive mechanism-0 molecule miner

The miner separates broad exploration from expensive structural scoring:

1. Public P40261 leaderboard observations train a target-conditioned proxy on molecular fingerprints, descriptors, reagent-role effects, and reagent-pair effects.
2. Proposal generation mixes substitutions around successful public structures, joint draws from learned component priors, and a persistent exploration fraction.
3. Exact validator gates remove invalid and historically non-novel candidates.
4. A small acquisition shortlist is scored by Boltz-2 NIM with validator-matched sampling settings.
5. The best chemically unique, entropy-valid 20 are written atomically. The chain miner reads only this validated checkpoint.

This avoids using Boltz-2 as the broad search engine. NVIDIA NIM 1.8.0 requires at least 48 GB VRAM per GPU; supported 2/4/8-GPU systems improve throughput through concurrent single-ligand affinity requests, not by splitting one request across GPUs.

## Setup

Use the official environment and point all local tools at the supplied database without copying or mutating it:

```bash
export NOVA_COMBINATORIAL_DB=/absolute/path/to/molecules.sqlite
python -m compound_miner.cli sync-history
python -m compound_miner.cli train --trees 256
```

Download the P40261 novelty archive from `Metanova/Submission-Archive` and run a search for each reaction. A private/local NIM endpoint is preferred; omit `--proxy-only` only when an exact endpoint is available.

```bash
export NOVA_BOLTZ_NIM_URL=http://gpu-host:8000/biology/mit/boltz2/predict
python -m compound_miner.cli search \
  --reaction 5 \
  --historical-molecules /absolute/path/P40261_molecules.csv \
  --nim-concurrency 8
```

Run all five reaction archives before going live because the allowed reaction is derived from the epoch boundary hash. Validate every checkpoint independently:

```bash
python -m compound_miner.cli validate-result \
  --reaction 5 \
  --result artifacts/result_rxn5.json \
  --historical-molecules /absolute/path/P40261_molecules.csv
```

The live miner reads `artifacts/result_rxn{reaction}.json` by default. Override this with `NOVA_CANDIDATE_RESULT`, retaining the `{reaction}` placeholder. If a valid 20-member checkpoint is unavailable, it fails closed instead of submitting random molecules.

## Performance measurements

Measure three layers separately:

- proxy quality: chronological top-100 recall and achieved oracle mean;
- candidate throughput: constructed, valid, and historically novel candidates per second;
- exact scoring: warm NIM requests per second at concurrency 1, 2, 4, 8, and 16.

Use the smallest concurrency before latency or error rate rises. One supported 48 GB GPU is the capacity floor. Multiple GPUs are useful only when the request queue keeps them busy.

