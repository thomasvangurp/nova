# Nanobody track — design

## Why this is a different problem from molecules

For molecules we could not tell which component combinations mattered; the proxy
ranked 0 of 19 field winners into its top-2000 of 8.26M. Nanobodies hand us the
prior for free: the config names the paratope
(`structure_binding_site`, `structure_res_index`), and VHH architecture tells us
the CDR loops are the binding determinant.

## 1. Constrain generation to the CDRs (implemented: `regions.py`)

The reference miner mutates random positions across all 118 residues
(`miner.py:281`). Framework mutations **cannot** improve binding and **do** erode
the two gates (`min_nativeness_score 0.45`, `min_human_framework_score 0.85`).
Pure downside.

Anchor-based parse of the base VHH (no anarci dependency):

    CDR1 [24:35]  SGFTFSTYAMS
    CDR2 [49:66]  RVNQNGGTTTYADAMKG
    CDR3 [96:108] ARWDGGSWSTDP
    -> 40 mutable of 118 (34%); 78 framework residues frozen

Consequences: search space shrinks ~3x, and both quality gates pass by
construction. Note **no UID currently fails those gates**, so they are a floor to
clear, not an edge — the edge has to come from binding.

## 2. Weight positions by contact, not uniformly

CDR3 dominates antigen contact in VHHs and is the most variable loop. Expected
priority: CDR3 > CDR2 > CDR1. The config's `structure_binding_site` residues are
on the TARGET side; mapping them to paratope positions needs the complex
structure, which is what boltzgen produces — so contact weights should be
*learned* from our own scored mutants rather than assumed.

## 3. The proxy (the part that decides whether this works)

Same shape as the molecule problem: cheap generation, expensive oracle
(boltzgen, GPU), and a ranking we cannot shortcut.

**One structural advantage over molecules.** Our molecule proxy failed from
covariate shift — trained on other miners' *submissions* (rho 0.2768) it scored
rho 0.0599 on randomly sampled molecules, the distribution we actually search.
Here we generate our own training data from the start, so it is in-distribution
by construction. Do not repeat the mistake of training on someone else's
distribution.

- **Target:** boltzgen metrics. Scoring is `final_nanobody_score` (mode=min),
  tie-broken by confidence / physical-interaction / developability
  `{category}_rank_sum` (`ranking.py:115-175`).
- **Features:** start with one-hot over the 40 CDR positions (760 dims); add an
  ESM embedding only if one-hot plateaus. Resist feature engineering first —
  on molecules that was exhausted while the objective was the real lever.
- **Data:** our own boltzgen scores, persisted across epochs exactly like
  `own_exact_scores.csv`.
- **Validation:** hold-out rho on RANDOM CDR mutants, never on a curated set.

## 4. Deployment (option 2)

One hotkey commits both tracks: `message = f"{molecules}|{nanobodies}"`
(`miner.py:426`). So no second registration — the nanobody pipeline runs on a
SEPARATE box and ships a result file the miner reads at commit time, exactly as
it reads `result_rxn{N}.json`. No GPU contention with the molecule search, and if
the nanobody side produces nothing the miner commits `~` for that half while the
molecule submission is untouched.

## 5. Blocker before any of this counts

`config/config.yaml` nanobody target is **stale**: ours is `P05231`/`4O9H`,
upstream is `P20809`/`6O4O`. Optimising against the wrong protein wastes the
whole effort. Sync first, and re-check it every epoch the way `upstream_config.py`
does for the compound side.

## 6. Honest EV

|            | competing/total | emission share | per competitor |
|------------|-----------------|----------------|----------------|
| molecules  | 32 / 54         | 15.8%          | 0.49%          |
| nanobodies | 52 / 55         | 10.5%          | 0.20%          |

Nanobodies is smaller, 1.6x more contested, and **no one fails the gates** — the
field is already past the easy wins. Molecules remains ~2.4x better EV per
competitor. Proceed here only as diversification, not as the primary bet.

---

## 7. CDR-constraint VALIDATED AGAINST DATA — and it is insufficient

Source: `P20809_nanobodies.csv` (HuggingFace `Metanova/Submission-Archive`),
**28,166 unique modal-length sequences with all 10 scored metrics**. This is the
public archive for the UPSTREAM target, so it is the right protein.

Method (no model, no GPU): rank every sequence by `rank_sum` exactly as the
validator does, take top-1% and bottom-1%, and compute per-position Shannon
entropy in each. The entropy DROP from losers to winners measures how much a
position discriminates.

```
mean per-position entropy   ALL 0.611   TOP1% 0.301   BOTTOM1% 0.407
positions entropy<0.5       ALL 73      TOP1% 100     BOTTOM1% 86
```

**Winners are twice as converged as the field.** There is a discoverable optimum
and most competitors have not found it: pos61 `Y` in 100% of winners, pos48 `F`
(100%) vs `W` in losers, pos36 `S` vs `A`, pos112 `W` vs `E`.

```
CDRs = 37% of sequence but hold 60% of top-20 signal
discriminative power: CDR 0.140 vs framework 0.085 (1.65x per position)
top-20 positions: 12 in CDRs, 8 in FRAMEWORK (2,12,45,48,81,122,125,126)
```

**Verdict: do NOT hard-restrict to CDRs.** They are enriched but carry only 60%
of the signal. The framework hits cluster at the termini, where developability
lives -- and `liability_score` + `liability_num_violations` are 2 of the 10
ranked metrics. CDRs drive binding, framework drives developability, `rank_sum`
weights both equally.

**Revised approach:** weight mutation probability by MEASURED entropy drop per
position rather than by region membership. The archive gives that for free.

## 8. Corrections to earlier sections

- `liability.py` does NOT reproduce the validator's formula. Mine scores the
  reference base at 4.5; the archive's `liability_score` averages 199 (110-280).
  Treat it as illustrative only -- learn the real one from the archive columns.
- The field does broad redesign, not scaffold mutation: 98 of 127 positions vary
  in >5% of sequences. The "everyone mutates one base" assumption was wrong.

---

## 9. The proxy is a DISCRIMINATOR, not an objective (measured)

Trained on 22,533 archive sequences (top 5% held out), one-hot Ridge:

```
held-out rho                     0.8824
top-1% enrichment                23.4x
true top-10 -> proxy ranks       140, 83, 12, 2262, 170, 119, 23, 270, 275, 144
budget 500 of 5,633 captures     9/10 of the true best
transfer near-consensus -> far   rho 0.8270   (NO covariate-shift collapse)
```

That is a far better proxy than the molecule one (rho 0.2768, ~2x enrichment,
0 of 19 winners in top-2000). Ridge beats ExtraTrees here (0.8824 vs 0.8546).
Adding motif features changed rho by 0.0001 -- one-hot already encodes them.

**But it cannot be used as a search objective.**

```
unconstrained hill-climb   best predicted rank_sum  -1,080,668  (impossible)
                           Hamming 111-113 vs archive 6-101 -> off-manifold
constrained (per-position observed alphabet, Hamming cap 37)
                           best predicted           -131,972   (still impossible)
                           Hamming pinned 37..37    -> saturates the cap
                           physically plausible      0/8
```

A per-position additive model scores a sequence as a sum of independent residue
contributions. Assembling the best residue at every position yields a combination
that has never existed, and the model cannot know it is implausible -- real
sequences respect epistasis, synthetic ones do not. Constraining the alphabet does
not help because the violation is in the COMBINATION, not the parts.

This is the same failure as APEX on the molecule side (additive over synthons,
cannot represent interactions).

**Consequence for the pipeline:**
  generate with boltzgen inverse folding  (on-manifold by construction)
  -> filter with the proxy                (rho 0.88, 23x enrichment, cheap)
  -> score the shortlist with boltzgen    (the real oracle)

Never invert that order. The proxy ranks what exists; it does not invent.

## 10. "Learn from nature" -- checked, mostly closed for this target

Target is human **Interleukin-11** (P20809); `6O4O` is IL-11 alone. The signalling
complex IS solved (`8DPS`/`8DPT`, IL-11 + IL-11RA + gp130), and residue numbering
between 6O4O and 8DPS was verified consistent (**100.0% identity at shared
numbers**, chain B).

```
config binding_site residues that are NATURAL receptor contacts: 2/14  [33, 40]
not natural contacts: 31,32,34,35,37,38,88,135,136,137,138,142
```

The nanobody must bind a largely NON-natural epitope, so copying how gp130 or
IL-11RA grip IL-11 does not transfer.

Evolutionary models (ESM-style) are also the wrong tool here, and we have the
precedent in-house: PlantCaduceus on eIF4E recovered CONSERVATION, not the
functional property (p=0.077). VHH frameworks are germline-encoded (every miner
already passes those gates) while CDRs are somatically generated -- an
evolutionary prior would pull CDRs toward germline, away from any specific
antigen. It would help on the saturated half and hurt on the decisive half.
