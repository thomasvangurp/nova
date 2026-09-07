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
