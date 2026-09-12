# Validation

## Equivalence against the live production scorer

The reference arm is not a re-run: it is harvested from production's own
history, `/root/live/epoch*/producer/**/wave_*_exact.json` (452 files, epochs
25035-25055, 954 unique sequences). 24 sequences that production scored **5-26
times each** were selected, stratified across `design_iiptm` 0.198-0.788. All
195 records were confirmed to be distinct observations (distinct
`observation_id`, distinct values) with `diagnostic_seed: None` throughout, so
they are genuine independent draws and their spread is a legitimate yardstick.

Fast path: 4 independent repeats of the same 24 sequences.

| metric | rho | mean diff | median abs(diff)/prod_sd | sign split |
|---|---|---|---|---|
| design_iiptm | 0.847 | -0.0011 | 0.49 | 12+/12- |
| design_ptm | 0.880 | +0.0010 | 0.36 | 14+/10- |
| design_to_target_iptm | 0.910 | +0.0012 | 0.43 | 17+/7- |
| min_design_to_target_pae | 0.912 | -0.0110 | 0.42 | 9+/15- |
| interaction_pae | 0.969 | -0.0133 | 0.33 | 11+/13- |
| plip_hbonds_refolded | 0.528 | +0.288 | 0.42 | 15+/9- |
| plip_saltbridge_refolded | 0.507 | +0.011 | 0.29 | 8+/16- |
| delta_sasa_refolded | 0.697 | +2.16 | 0.39 | 13+/11- |
| liability_score | 1.000 | 0.000 | exact | exact |
| liability_num_violations | 1.000 | 0.000 | exact | exact |

Every figure improved from 3 repeats to 4 (rho up, abs(diff)/prod_sd down),
which is what a residual made of noise rather than bias does.

Reading it:

* **No bias.** Signs are balanced on every confidence metric. `design_iiptm`'s
  mean difference (-0.0022) is below the same-seed reproducibility floor
  (0.0048), and `interaction_pae`'s (-0.0054) is 4% of production's own sd.
* **Right scale.** Each sequence's fast-path mean sits within 0.4-0.5 of
  production's own standard deviation -- what an equivalent scorer looks like.
* **The low rho on the three physical metrics is an artefact of ties**, not
  disagreement: `plip_hbonds_refolded` takes values 4-8 and
  `plip_saltbridge_refolded` 0-2, so rank correlation is dominated by
  near-ties. Their abs(diff)/prod_sd is 0.40-0.48, i.e. well inside production's
  spread.
* The `liability_*` "possible bias" flag in `cmp_prod.py` is a false positive:
  the script counts an exact 0.0 difference as negative. Those two metrics are
  sequence-only and match exactly, which is the intended check that the
  sequence plumbing is right.

## Noise: the fast path is not noisier than production

Measured on the SAME 24 sequences (an earlier comparison used a different
sequence set and was meaningless -- PAE variance scales hard with PAE
magnitude: production sd is 0.041 at mean 2.3 and 0.929 at mean 15.0):

| metric | fast sd | production sd | ratio |
|---|---|---|---|
| design_iiptm | 0.0157 | 0.0147 | 1.07x |
| design_ptm | 0.0024 | 0.0036 | 0.68x |
| design_to_target_iptm | 0.0043 | 0.0052 | 0.82x |
| min_design_to_target_pae | 0.0684 | 0.0948 | 0.72x |
| interaction_pae | 0.0900 | 0.1253 | 0.72x |

## Speed

| configuration | per sequence |
|---|---|
| production (measured from its own wave logs) | **176 GPU-s** |
| fast, 1 GPU, 8-sequence chunk | 14.5 GPU-s (**12.2x**) |
| fast, 4 GPUs, 24 sequences, 6 per shard | 8.5 s wall / 34 GPU-s |

The 4-GPU figure is worse per GPU because each shard paid its own 33 s model
load for only 6 sequences. Load amortises with wave size; size waves so each
shard gets >= 15 sequences.

## Component evidence

| change | evidence |
|---|---|
| resident models | folding step 52.9 s -> 11.4 s per design; same weights, same code path |
| design stage -> grafting | 24 s -> 94 ms; 100 folds over 2 seed-disjoint rounds, all shifts below the same-seed floor |
| TF32 | 44.3 -> 34.4 ms/step; perturbation equals a same-seed rerun |
| analysis trim | 23.3 -> 9.9 s; all ten scored columns bit-identical, 14 dropped columns all unread |
| design 500 -> 100 steps | target template within its own run-to-run spread (0.122 A vs 0.145 A floor, 4 designs); downstream scores move < 0.5 sd |

## Rejected on evidence

| candidate | why |
|---|---|
| `compile_structure=true` | **6.6x slower** (291.5 vs 44.3 ms/step): dynamic-shape recompiles across the 200-step loop |
| loop-invariant hoisting of the conditioning tensors | no gain (45.98 vs 44.20 ms) |
| `recycling_steps` 3 -> 1 | 18% faster, `design_iiptm` bias -0.0339 = 7x the floor. At 0, rho collapses to 0.20 |
| `diffusion_samples` 5 -> 2 | the score is a MAX over samples, so this biases it DOWN, not just noisier: `min_design_to_target_pae` +0.61 |
| folding 50 steps | `design_iiptm` bias +0.0049, at the floor; 100 steps gives +0.0003 |
| backbone-only design synthesis | -0.092 `design_iiptm`, +5.73 PAE. `parse_polymer` marks unfound CCD atoms absent, so ~40% of chain B went unresolved |
| batched attention-mask rework | slower at the real shapes (0.380 vs 0.287 ms) |

---

# The 50-step default: why it is not a fidelity trade

## The control that decides it

Rank correlation against production is meaningless without knowing what rho
production scores against ITSELF. This scorer is not reproducible even at a
fixed seed, so that ceiling is well below 1.0. Estimated by splitting each
sequence's production observations into disjoint halves, 200 random splits, at
matched averaging depth (rho rises with depth, so an unmatched comparison
flatters whichever side averaged more):

| metric | production self-agreement | synth+200x5 | synth+100x5 | **synth+50x5** |
|---|---|---|---|---|
| design_iiptm | 0.813 +- 0.055 | 0.758 | 0.771 | 0.776 |
| design_ptm | 0.757 +- 0.068 | 0.775 | 0.726 | **0.847** |
| design_to_target_iptm | 0.711 +- 0.065 | 0.764 | 0.690 | **0.765** |
| min_design_to_target_pae | 0.745 +- 0.069 | 0.768 | 0.796 | **0.777** |
| interaction_pae | 0.898 +- 0.034 | 0.916 | 0.915 | 0.891 |

All fifteen are at ceiling. This is the control I initially omitted, and
omitting it led me to read "rho 0.847 -> 0.799" as degradation when the ceiling
was 0.81.

## Bias vs production, with sign splits (24 sequences)

| metric | synth+200x5 | synth+50x5 | production sd |
|---|---|---|---|
| design_iiptm | -0.0011 (12+/12-) | **-0.0007** (10+/14-) | 0.0147 |
| design_ptm | +0.0010 (14+/10-) | +0.0007 (18+/6-) | 0.0036 |
| design_to_target_iptm | +0.0012 (17+/7-) | **+0.0002** (12+/12-) | 0.0052 |
| min_design_to_target_pae | -0.0110 (9+/15-) | -0.0111 (11+/13-) | 0.0948 |
| interaction_pae | -0.0133 (11+/13-) | **-0.0042** (11+/13-) | 0.1253 |

50 steps is no worse than 200 on any metric and better on three. The one
nominal flag (`design_ptm` 18+/6-) has magnitude +0.0007 against a production
sd of 0.0036 -- 0.2 sd -- and does not survive correction across ten tests.

## Independent confirmation: 20 designs x 6 repeats, 360 observations

Settings interleaved within each repeat with rotating order, so box-load drift
cannot be confounded with setting.

Sign splits vs 200x5 -- every one a coin flip:

| metric | 100x5 | 50x5 |
|---|---|---|
| design_iiptm | 10+/10- | 11+/9- |
| design_ptm | 8+/12- | 12+/8- |
| design_to_target_iptm | 11+/9- | 11+/9- |
| min_design_to_target_pae | 7+/13- | 5+/15- |
| interaction_pae | 8+/12- | 7+/13- |

Standard error per GPU-second (`sd*sqrt(s)`, lower is better) -- 50x5 wins on
all five:

| metric | 200x5 | 100x5 | 50x5 |
|---|---|---|---|
| design_iiptm | 0.0534 | 0.0325 | **0.0270** |
| design_ptm | 0.0157 | 0.0137 | **0.0105** |
| design_to_target_iptm | 0.0423 | 0.0270 | **0.0244** |
| min_design_to_target_pae | 1.956 | 1.266 | **0.937** |
| interaction_pae | 1.420 | 0.896 | **0.734** |

## A "saturating bias" that turned out to be an artefact

Deltas at 100 and 50 steps correlated strongly (r 0.74-0.88), which looked like
a shared per-design effect saturating below 200 steps. But both deltas subtract
the SAME 200x5 mean, which induces correlation. Re-graded against disjoint
reference repeats, r collapses to 0.07-0.26. There is no shared effect, and
therefore no bias to explain -- which also reconciles the +0.0008 measured on
synthesized inputs with the +0.0031 measured on real design outputs: both are
draws from zero.

## The consequence worth acting on

The dominant error term is the scorer's own irreproducibility, not step count:
even 200x5 mis-ranks against itself at rho ~0.81-0.90. So the right use of the
speedup is MORE REPEATS at 50 steps, not more steps -- which is exactly what
standard error per GPU-second says.

## Final speed

| configuration | per sequence |
|---|---|
| production (from its own wave logs) | **176 GPU-s** |
| fast, synth + 200x5, 1 GPU, 8-seq chunk | 14.5 GPU-s (12.2x) |
| **fast, synth + 50x5, 1 GPU, 8-seq chunk** | **8.3 GPU-s (21.2x)** |


---

# Overhead removal (8.3 -> 6.2 s/seq)

Measured split before the change, per sequence (synth + 50x5 + TF32):

    folding_predict 5.707   (of which 4.54 model compute, ~1.17 data/IO)
    analysis        1.796   CPU only; the GPU idles through all of it
    configure       1.017   re-materialises identical YAML every chunk
    synthesize      0.089

Three changes, none of which alter what is computed:

* `configure` runs once per worker. The YAML it emits differs between chunks
  only in paths, and those are already overridden at instantiate time.
* analysis runs in a persistent child process and is collected after the GPU
  loop, so a chunk's analysis overlaps the NEXT chunk's folding. A fresh
  subprocess per chunk would pay ~5.5 s of imports for ~1.8 s of work.
* the folding dataloader no longer spawns a worker process per `predict()`
  call to featurise a handful of designs.

Result: 24 sequences, one GPU, six chunks -> 6.23 s/seq, from 8.3.

Validated against production history, ceiling-controlled:

| metric | production self-agreement | pipelined fast path |
|---|---|---|
| design_iiptm | 0.813 +- 0.055 | 0.813 |
| design_ptm | 0.763 +- 0.062 | 0.822 |
| design_to_target_iptm | 0.701 +- 0.067 | 0.695 |
| min_design_to_target_pae | 0.749 +- 0.068 | 0.783 |
| interaction_pae | 0.895 +- 0.035 | 0.926 |

All at ceiling. Mean diffs -0.0029 / +0.0003 / -0.0009 / -0.0017 / +0.0041,
sign splits 12+/12- on three of five. 24/24 structures captured.

## Rejected: uncompressing the folding npz

Compressing an npz that is read back seconds later and then deleted looked like
free CPU on the critical path. The A/B could not resolve it:

    uncompressed  190.2 s at load 20.3
    compressed    235.6 s at load 32.8
    compressed    184.2 s at load 29.0   (earlier run, same config)

The live miner shares this box and its load moved 60% between arms. The same
configuration spans 184-236 s, so this box's timing noise is ~28% -- far larger
than the ~0.2 s/seq effect under test. Reverted: unproven changes do not ship.

**This bounds further micro-optimisation here.** Any change worth less than
roughly 1.5 s/seq cannot be honestly validated on this box while it is mining;
it would need an idle box or a many-repeat paired design.

## What is left

Of 6.2 s/seq, **4.54 s is model compute** -- sampler 2.03, trunk 1.55,
confidence 0.57, encoders 0.39. Removing every remaining overhead would reach
~4.5 s/seq. Past that the floor only moves by changing what is computed:
fewer recycles (measured harmful), fewer than 50 sampling steps (unvalidated),
a distilled model, or not running the exact scorer on most candidates.
