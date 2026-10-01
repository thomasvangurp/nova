# Nanobody competitor-gap report

Data cut: public leaderboard through epoch **25260**; mutation-graph snapshot
through local exact-screen epoch **25262**.  This analysis is offline and made
no production changes.

## Executive conclusion

The top miners' sustained lead is real, but it is not explained by discovering
a different part of sequence space.  We had already exact-screened a sequence
within two substitutions of **85.3% of their winning entries** before those
entries were submitted.  The main failure is local proposal completion and
selection: we reach the right neighborhood, but do not consistently enumerate
and select the winning 1–3-residue haplotype.

There is also a concrete obsolete-policy problem.  On 40 exact sequences
remeasured five times, the old `historical-drivers` weighted score has Spearman
correlation **−0.20** with the public score.  It prefers our sequence in 13 of
20 same-epoch pairs even though the competitor sequence was the public winner.
This is not a blanket failure of every component: five-pass `design_iiptm`
correlates **0.645** with its public measurement and favors the competitor in
17 of 20 pairs.  Most other stochastic structural/interface components have
near-zero or negative local-to-public rank correlation.  That weighted policy
is allowing noisy components—and deterministic liability advantages that did
not produce the public wins—to overwhelm the one reproducible competitor edge.

This result does **not** establish drift in the fast BoltzGen implementation.
The fast path was previously validated at the production-versus-production
reproducibility ceiling.  It also does not describe the current live default,
which is the validator's unweighted dense-rank sum.  The concrete fix is to
remove the obsolete weighted alternatives so they cannot be selected again,
then compare strategies by held-out Monte Carlo draws of the validator-shaped
objective.  That arbitrary-molecule experiment is now complete: empirical
fresh-draw selection improves on aggregate-mean selection when it has enough
independent observations, without changing the ten-metric objective.

## Validator source of truth

The validator metric is explicit in source. `config/boltzgen_config.yaml`
defines five confidence, three physical-interaction, and two developability
metrics. `BoltzgenWrapper._rank_metrics_dataframe` applies pandas dense rank to
each metric and sums all ten rank columns. `config/config.yaml` selects
`rank_sum` with `min` mode. The three category sums are dashboard display
breakdowns, not separately weighted scoring terms.

The exact repeated discriminator in the matched replay is `design_iiptm`, not
an unspecified validator metric: the rival is better in 17/20 pairs, with a
mean advantage of +0.01039. EasyNano full-ESMFold2 ipTM is a different model's
interface score and must not be labelled validator `design_iiptm`.

## Temporal interpretation through epoch 25260

The lead is sustained rather than monotonically accelerating. Dividing the 193
complete public epochs 25068–25260 into four contiguous windows, 5EvpaL11's
mean within-epoch rank fraction on `design_iiptm` improves from 0.501 in the
first window to 0.387 in the last (lower is better). Its
`design_to_target_iptm` improves 0.595→0.457 and `design_ptm` 0.466→0.431.
Over the same windows ours moves 0.475→0.543, 0.636→0.521, and 0.561→0.494.

This localizes the cumulative performance gap to structural-confidence ranks.
It is not a developability story: the competitor's liability-score rank moves
0.536→0.528 while ours moves 0.495→0.467. Nor is it simply fleet size. The
graph and exact sequence join show that 85.3% of top-miner wins were within two
edits of a sequence we had already screened, while only 2.7% were exact. The
most defensible causal hypothesis is therefore iterative retention of coherent
local haplotypes plus better confidence-oriented selection; the available data
does not reveal the competitors' private generator or prove which algorithm
they use.

## Leaderboard gap

| Miner | Entries | Wins | Wins / entry | Top-10 / entry | Median fleet |
|---|---:|---:|---:|---:|---:|
| 5EvpaL11 | 757 | 33 | 4.36% | 26.3% | 4 |
| 5F1RWsGe | 569 | 16 | 2.81% | 23.4% | 3 |
| 5Fq7TBKW | 954 | 26 | 2.73% | 31.3% | 5 |
| **US** | **1,330** | **23** | **1.73%** | **13.9%** | **10** |

Fleet size is therefore not the explanation.  The top three obtain more wins
and top-10 placements per molecule with materially smaller fleets.

## What the graph says

The graph snapshot is current and internally consistent:

- 55,678 nodes, 316,865 neighborhood edges, and 212,044 lineage edges.
- 417,297 observations, including 388,188 `live_exact_screen` observations
  through epoch 25262 and 22,190 public-validator observations through 25260.
- `proposal_scores` is empty; selection history lives in observations and
  producer artifacts, not that table.

For the top three miners' 631 top-10 entries, direct comparison against every
sequence screened before the corresponding epoch gives median Hamming distance
2, p90 distance 3, 34.2% within one edit, 81.5% within two, and 100% within six.
For their 75 wins the corresponding figures are median 2, p90 3, 37.3%, 85.3%,
and 100%.  The persisted graph stores only 16 bounded nearest neighbors, so the
direct all-screened comparison—not unresolved graph edges—is the decisive test.

This rules out a broad "we never find their basins" explanation.  It supports
quota-based enumeration of combinations around the best screened nodes,
especially at under-generated positions 113/115 and recurrent CDR3/tail sites,
rather than another global residue-frequency reweighting.

## Sequence and haplotype signatures

The top three use fewer non-consensus framework edits: **4.04 per molecule vs
4.51 for us**.  Their CDR3 search burden is almost the same (1.32 vs 1.29 edits),
as are CDR3 Y/G/S/R/V/I fraction (0.259 vs 0.257) and R/S fraction (0.0337 vs
0.0336).  Thus the paper's target-specific R/S enrichment does not explain the
leaderboard gap.

The largest top-three marginal enrichments are 126Q, 0K, 72S, 83G, 30R, 45E,
113H, 122V, 120T, 99G, and 121G/P.  The leading joint signatures—such as
126Q+72S, 126Q+0K, 0K+83G, and 72S+83G—mostly look like preservation of coherent
framework/tail haplotypes.  These are associations, not causal single-residue
rules; nearest-sequence edits and recent folding literature both show why the
combinations must be tested intact.

## Which public metrics separate them

Positive values below mean a better mean within-epoch normalized rank for the
top three than for us:

| Metric | Competitor advantage |
|---|---:|
| design ipTM | +0.127 |
| design-to-target ipTM | +0.123 |
| design pTM | +0.080 |
| interaction PAE | +0.067 |
| salt bridges | +0.057 |
| minimum target PAE | +0.032 |
| H-bonds | +0.002 |
| delta SASA | +0.001 |
| liability count | −0.033 |
| liability score | −0.099 |

Their public edge is structural confidence/interface geometry, not generic
developability.  They actually tolerate somewhat worse liability scores.

## Five-pass exact replay and scorer diagnosis

The retained replay contains 20 same-epoch pairs: a realized public winner and
our producer's best submitted sequence, each scored in five independent passes.

- Averaging all five passes, our aggregate historical-driver score is better by
  13.11 points on average (median 15.75), and the rival is preferred in only
  7/20 pairs.  Across all 100 individual pass-pairs, the rival is preferred 46%.
- `design_iiptm` is the exception: rival advantage averages +0.0104 and occurs
  in 17/20 pairs.  Its local/public Spearman is 0.645.
- `design_to_target_iptm` has weaker local/public Spearman 0.319.  `design_ptm`,
  interface PAE, minimum PAE, H-bonds, salt bridges, and delta SASA range from
  −0.105 to 0.093—effectively no reliable ordering in this cohort.
- Liability metrics reproduce perfectly because they are deterministic, but
  favor us; they cannot explain the competitors' public wins.

So the correct statement is: **the historical weighted surrogate is not a valid
proxy for the public ordering in this cohort**.  This is a selector-objective
bug, not evidence that the fast scorer differs from BoltzGen.  Leave-one-pass-out
Monte Carlo using the untouched pass as the validator-shaped outcome gives 59%
correct pair choices for confidence-only, 56% for the live-style plain objective,
54% for design-ipTM-only, and 57% for target-ipTM-only.  The confidence-only
lead is too small to deploy as a new weighting; it is a hypothesis for a larger
arbitrary-molecule replay.

The replay is retrospective and subject to winner's curse.  It diagnoses the
instrument; it is not proof of biological affinity or prospective win rate.

## Arbitrary-molecule Monte Carlo experiment

An isolated 8×RTX 4090 host scored a preregistered cohort of 96 molecules in 12
independent passes, for 1,152 complete observations.  The cohort contains 12
recent rival/ours basins.  Each basin includes both anchors and three generated
variants around each anchor; 72/96 molecules had never appeared in the graph.
Every result uses the `validator-dual-sample-v1` protocol and contains all ten
finite validator metrics.

For each of the 12 passes, selection used only the other 11 and evaluation used
the untouched pass.  Each policy therefore made 144 held-out basin decisions:

| Selector | Held-out winner picks | Pick rate | Mean regret | Median regret |
|---|---:|---:|---:|---:|
| target ipTM only | 42/144 | 29.2% | 76.9 | 58.0 |
| empirical fresh-draw probability | 38/144 | 26.4% | 83.5 | 65.0 |
| confidence only | 33/144 | 22.9% | 92.8 | 80.5 |
| validator plain aggregate | 32/144 | 22.2% | 93.3 | 83.5 |
| design ipTM only | 25/144 | 17.4% | 126.5 | 127.0 |

The fresh-draw policy beats plain aggregate selection on regret in 24 decisions,
loses in 13, and ties in 107; it gains 12 held-out winner picks while losing 6.
It also chooses the rival side in 86/144 decisions versus 74/144 for plain and
chooses a graph-unseen variant in 74/144 versus 58/144.  This directly corrects
the earlier failure mode in which the aggregate selector preferred our anchor
despite a reproducible rival-side structural advantage.

Target-ipTM-only is numerically best in this cohort, but it is not adopted as
the objective.  It was one of several alternatives compared on the same cohort,
would discard nine validator metrics, and the older winner-pair replay pointed
to a different single metric (`design_iiptm`).  That reversal is exactly the
kind of cohort-specific metric chasing the arbitrary replay was intended to
detect.  Target ipTM remains a shadow diagnostic.

Depth matters.  With 1, 2, 4, 8, and 11 training observations, fresh-draw mean
regret is respectively 94.3, 102.9, 93.5, 92.0, and 83.5, versus 94.3, 91.7,
95.7, 89.9, and 93.3 for plain aggregate selection.  The advantage is clear
only at depth 11.  The dormant fast-GPU profile is therefore changed from four
to eleven repeat rounds, with four-round staged pruning and a 12-observation
publication floor.  It remains inactive and has not been deployed.

The live selector implementation now calculates the exact empirical probability
that each qualified molecule wins when every molecule receives one independent
fresh draw of the complete validator rank sum.  It integrates the observed
discrete distributions exactly, including fair tie splitting; it does not fit
independent metric weights, optimize a fixed score threshold, or use a noisy
bootstrap RNG.  P90 and threshold rates remain qualification diagnostics, and
the existing configured settled-site consensus tier remains an outer policy.

## Isolated scorer/deployment finding

The first clean-host run failed before scoring because the generic `models/`
ignore rule omitted two required vendored BoltzGen model files from a detached
worktree.  The files in the developer tree were byte-identical to the canonical
BoltzGen copies; the fix is a narrow `.gitignore` allowlist so clean Ansible
deployments contain them.  No scoring arithmetic, checkpoint, or protocol was
changed.  This was a packaging defect, not evidence that fast BoltzGen differs
from the validator.

Provisioning used `ansible/nanobody.yml` and the locked Pixi `bgen` environment.
The unrelated optional MetaNano import failed later in the full play, after the
scorer runtime and all eight GPUs had already validated.  Scorer-only execution
then completed all 12 passes.  Results were downloaded and checksum-verified
before instance 51991448 was stopped; its final provider state is `exited` /
`stopped`.  Production was not used or modified.

## Literature interpretation

[Wang et al.](https://academic.oup.com/bib/article/27/2/bbag180/8660411) use a
fixed/humanized framework, natural position-frequency priors, Metropolis
sequence search, solubility rejection, and aggressive structural/interface
filtering.  That is compatible with the top miners' more conservative framework
haplotypes and confidence advantage.  It does not identify any named miner or
justify copying its target-specific motifs.  Only 2 of 1,500 candidates tested
as far as binding were binders, so it is a search/filter recipe, not an oracle.

The case for haplotypes is strengthened by deep loop profiling showing explicit
CDR epistasis ([Wan et al.](https://www.nature.com/articles/s41594-026-01804-9))
and by natural-prior library design
([McMahon et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5839991/)).

The case for multi-seed evaluation is also strong.  AFsample sometimes found
only five good nanobody poses in 6,000 samples
([Wallner](https://academic.oup.com/bioinformatics/article/39/9/btad573/7274860));
an AF3 antibody/nanobody benchmark reports substantial single-seed failure and
uses 20 seeds
([Hitawala and Gray](https://pmc.ncbi.nlm.nih.gov/articles/PMC12360200/)); and
successful RFdiffusion antibodies were enriched for high maximum AF3 ipTM over
ten seeds ([Bennett et al.](https://www.nature.com/articles/s41586-025-09721-5)).

## Attribution answer

Sequence notes/provenance do **not** contain the proposing miner.  Graph nodes
are sequence identities; public observation provenance contains the API source
but no coldkey, hotkey, or UID.  The analysis attributes 5EvpaL11 and the other
miners by exact `(epoch, sequence)` join to the leaderboard database, where
coldkey/hotkey/UID are present.

The ingestion schema should copy those three identifiers into public
observation provenance.  That would make miner-local trajectories, nearest
same-miner ancestors, and signature queries first-class rather than an external
join.

## Recommended next changes

1. **Keep the selector objective identical to the validator.** This is now
   implemented: obsolete weighted policy paths are gone and independent intact
   observations estimate the probability of winning a fresh draw.  Continue to
   shadow target ipTM and confidence-only; do not turn this cohort into a new
   hand-weighted objective.
2. **Complete local haplotypes.** From each strong screened node, reserve budget
   for exhaustive/tiered 1–3-edit combinations at empirically enriched sites,
   with explicit quotas for positions 113/115.  Preserve canonical
   framework/tail haplotypes unless an exact measurement justifies breaking
   them.
3. **Use paired multi-seed replay as an evaluation, not a max-score contest.**
   Start at 20 common seeds for preregistered rival/ours/missed-candidate trios;
   retain medians, lower quantiles, threshold hit rates, pose clusters, and
   epitope consistency.  Escalate finalists to 100 seeds.  Maxima alone reward
   rare lucky poses.
4. **Add miner metadata to graph provenance** and store proposal-stage state so
   "screened but not submitted" regret can be queried without producer-artifact
   joins.
5. **Do not call stochastic replay proof.** It can demonstrate run-by-run
   dominance or robustness.  Biological superiority still requires experiment.

## Reproducible outputs

- `analyze.py`: leaderboard/graph join, signatures, coverage, and metric gaps.
- `analyze_remeasure.py`: retained five-pass replay and local/public calibration.
- `monte_carlo_selector.py`: leave-one-observation-out strategy comparison for
  arbitrary fast-scorer cohorts.
- `monte-carlo-20260921/manifest.json`: 96-molecule, 12-basin cohort definition.
- `monte-carlo-20260921/pass_*/exact_all.json`: 12 independent complete passes.
- `monte-carlo-20260921/selector_results.json`: exact held-out decisions and
  policy summary; `selector_depth_*.json` contains the depth curve.
- `monte-carlo-20260921/remote_results.tar.gz`: retrieved raw result archive,
  SHA-256 `d57c26cecdd2a7398ddcec4ddca6e6aad2ca9d1b388d4fc97f275ad458312194`.
- `results/report.json`: machine-readable main report.
- `results/remeasure_report.json`: replay and scorer-calibration report.
- CSVs in `results/`: miner, residue, haplotype, regional, graph-distance,
  missing-edit, per-metric, and replay detail.
