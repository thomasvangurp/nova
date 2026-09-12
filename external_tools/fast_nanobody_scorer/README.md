# Fast nanobody exact scorer

Drop-in replacement for `score_boltzgen_batch.py`. Same CLI, same output JSON,
same ten scored metrics.

**176 -> 6.2 GPU-seconds per sequence (~28x).**

Nothing here touches `/root/external_tools/boltzgen`. The live miner is
unaffected until you point it at this directory; rollback is one path.

---

## What changed, and what each is worth

| change | measured | fidelity evidence |
|---|---|---|
| **Resident models** -- load checkpoints once per worker, not once per chunk | folding step 52.9s -> 11.4s per design | none needed: same weights, same `predict_args`, same forward. `Predict.run(cfg, run_prediction=False)` is the production loader. |
| **Design stage replaced by side-chain grafting** | 24s -> **94ms** per sequence | 100 folds, 2 disjoint-seed rounds: every same-backbone shift below the same-seed reproducibility floor |
| **Folding sampler 200 -> 50 steps** | 10.8s -> 4.6s per design | at the production-vs-production agreement ceiling on all five confidence metrics; mean diffs smaller than 200 steps gives |
| **TF32 on the folding step** (`matmul_precision: high`; upstream leaves it `null`) | 44.3 -> 34.4 ms per denoising step | perturbation equals rerunning the identical config at the identical seed |
| **Analysis trimmed** | 23.3s -> 9.9s per 1-design chunk | **all ten scored columns bit-identical**; the 14 dropped columns are all unread |
| **Thread caps** | load 299 on 128 cores -> bounded | none needed |
| **Overhead removal** -- configure hoisted to once per worker; analysis moved to a persistent overlapped child process; folding dataloader `num_workers` 1 -> 0 | 8.3 -> 6.2 s/seq | pure scheduling: validated at the production-agreement ceiling on all five metrics, sign splits 12+/12- on three of five |

### Why the design stage can go

Folding consumes exactly two things from a design output: both chains'
**sequences**, and the **target chain's geometry** as an SE(3)-invariant
template. The designed chain's coordinates reach no module the folding
checkpoint contains -- `TokenDistanceModule` is the only module reading
coordinates directly and it is not instantiated (zero `token_distance*` keys,
`strict=True` load). Verified empirically: displacing every chain-B atom by
N(0, 5 A) left every tensor that reaches the model bit-identical.

So a real design output is retargeted to any same-length sequence by copying
chain A verbatim and grafting chain B's side chains onto its backbone.

### Why 50 sampling steps is not a fidelity trade

The scorer is **not reproducible even at a fixed seed**, so the yardstick is
not "does the number change" but "does it change more than rerunning the same
config". Two independent measurements say no:

* 20 designs x 6 repeats, all three settings interleaved: every sign split is a
  coin flip (`design_iiptm` 10+/10- at 100 steps, 11+/9- at 50), every bias is
  below the reproducibility floor, and standard error **per GPU-second** is
  better at 50 steps on all five metrics.
* End-to-end against production's own scoring history (24 sequences production
  had already scored 5-26 times each): at the production-vs-production
  agreement ceiling on all five metrics, four of five exceeding it.

Pass `--fold-sampling-steps 200` to reproduce upstream exactly.

**Do NOT lower `--fold-diffusion-samples`.** The score is a MAX over samples
(`analyze_utils.get_best_folding_sample`), so fewer samples biases it *down*
rather than merely adding noise: 5 -> 2 moved `min_design_to_target_pae` +0.61.

---

## Usage

```bash
cd /root/fastscore/release

./release_python.sh fast_score_batch.py \
  --sequences <seqs.json> \
  --config /root/config/config.yaml \
  --target P20809 \
  --gpus 0,1,2,3,4,5,6,7,8,9 \
  --shards-per-gpu 1 \
  --batch-size 20 \
  --synth-reference refdesign/<a real design>.cif \
  --shard-root /root/bgen_shards_fast \
  --out result.json
```

`--synth-reference` needs a real design `.cif` (with its `.npz` alongside) whose
**designed chain has the same length as the candidates**. Length mismatch fails
loudly rather than guessing. Nearly all live candidates are 127 residues;
`refdesign/` holds four 127-residue references. For a new target or length,
generate one reference the normal way (one design-stage run) and keep it.

Omit `--synth-reference` to run the real diffusion design stage instead.

### Size waves large

The only remaining non-GPU cost is a **one-time ~80 s model load per worker**
(77% of a worker's life is GPU-forward work). At 24 sequences per worker that
load is 12.9% of the time; at 100 sequences it is negligible. This matters more
than any other tuning available: measured GPU-s/sequence was 26.0 at 24
sequences/worker on one GPU.

### Knobs

| flag | default | note |
|---|---|---|
| `--fold-sampling-steps` | **50** | 200 reproduces upstream. See above. |
| `--fold-diffusion-samples` | config (5) | do not lower; the score is a MAX over samples |
| `--fold-matmul-precision` | `high` | TF32. `null` reproduces upstream. |
| `--design-sampling-steps` | 100 | only used without `--synth-reference` |
| `--shards-per-gpu` | 1 | **leave at 1**; K=3 buys 1.21x for triple the memory |
| `--batch-size` | 24 | sequences per chunk; `data.cfg.batch_size` stays 1 |

`recycling_steps` is deliberately not exposed: 3 -> 1 buys 18% and costs a
7x-the-floor systematic shift; at 0 the ranking collapses (rho 0.20).

---

## Rollback

The live path is untouched. To revert, point the caller back at
`/root/nanobody_opt/score_boltzgen_batch.py` under `/root/bgen_python.sh`.
Nothing here is on the live `PYTHONPATH`.

`boltzgen_src/` is the vendored tree plus exactly four files
(`task/analyze/analyze.py`, `task/analyze/analyze_utils.py`, `data/data.py`,
`data/parse/mmcif.py`), all analysis-side and all verified column-identical.
**The model path is byte-identical to the vendored source.**

---

## Measured dead ends -- do not re-try these

| candidate | result |
|---|---|
| `batch_size > 1` (folding several designs per forward) | **1.00x / 0.98x / 0.99x / 1.02x** at bs 1/2/4/8. The 5-sample multiplicity axis already saturates the sampler; `diffusion_samples` 1->5 costs only +13%. Also hides five real correctness bugs (see below). |
| Multiple resident workers per GPU | K=3 gives **1.21x**, ceiling is 1.30x: GPU-forward work is 77% of worker life and the idle part is a one-time load that amortises away |
| `compile_structure=true` | **6.6x slower** (291.5 vs 44.3 ms/step) -- dynamic-shape recompiles across the 200-step loop |
| Hoisting loop-invariant conditioning tensors | no gain (45.98 vs 44.20 ms) |
| `recycling_steps` 3 -> 1 | 18% faster, `design_iiptm` bias -0.0339 = 7x the floor |
| `diffusion_samples` 5 -> 2 | biases the score DOWN (max over samples) |
| Backbone-only design synthesis | -0.092 `design_iiptm`, +5.73 PAE -- fixed by full-atom grafting |
| Batched attention-mask rework | slower at the real shapes (0.380 vs 0.287 ms) |

---

## Structure capture: the contract this scorer honours

BoltzGen uses **two different argmax rules** in one refolding run:

```
writer.py:72          0.8*iptm                  + 0.2*ptm            -> refold_cif
analyze_utils.py:98   0.8*design_to_target_iptm + 0.2*design_ptm     -> the reported confidence metrics
```

They disagree on roughly 12-15% of designs, so the three physical metrics and
the five confidence metrics can describe two different diffusion samples.

**This is already handled by the surrounding pipeline, and this scorer does not
change it.** The vendored BoltzGen is patched to emit `refold_analyze_cif`
beside `refold_cif` (`writer.py:55`), and `nanobody_opt` keeps both indices
explicit in `diffusion_sample_selection.py`. `--capture-structures` retains both
poses per observation, as `confidence_path` (the pose whose metrics are
reported, also exposed as the backward-compatible `path`) and `writer_path`.

Two guards are reproduced exactly, not reimplemented -- this module imports
`nanobody_opt.diffusion_sample_selection`, so the fast path and the production
path cannot drift in how they identify the poses:

* every reported CSV metric is checked against the analyze-selected sample's
  value (atol 5.01e-6, admitting the CSV's five-decimal serialization). A
  scorer reporting a different sample's number cannot pass.
* `sample_count` is recorded so `require_expected_sample_count` can reject an
  observation folded with fewer than five diffusion samples. That guard exists
  because the score is a MAX over samples, so a faster scorer cutting them
  biases every metric rather than merely adding noise. **This package keeps
  `diffusion_samples` at 5 and passes.**

The live path requires this: `score_boltzgen_waves.py` rejects a wave whose
sequences lack captured structures. Run with `--capture-structures <dir>
--require-structures` to match it.

---

## One thing worth knowing regardless of this package

`plip_hbonds_refolded` is close to noise. Its production-measured sd is 2.33
median (up to 5.91) on values ranging 3-8, yet it carries one tenth of the
rank_sum weight.
