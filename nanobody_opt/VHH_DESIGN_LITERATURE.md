# VHH design methods relevant to NOVA

Updated 2026-09-09. This is an implementation-oriented review of primary
papers and official code, not a claim that computational confidence equals
experimental affinity.

## What the first campaign was missing

Local sequence mutation started from an archive VHH and asked a sequence model
to predict small improvements. That does not solve the central problem: jointly
choosing the global VHH pose, CDR backbone geometry, CDR sequence, and target
hotspot. Modern successful workflows explicitly search those variables and use
an independent complex refold as a rejection test.

The target site cannot be treated as one fixed residue list without checking
its geometry. RFantibody's official guidance says target-site choice and
hotspot choice are major determinants of success and recommends sites with
multiple hydrophobic contacts. A 2026 multi-method VHH campaign went further:
it swept eight epitope regions, three VHH frameworks, multiple target
structures, and RFantibody, IgGM, and mBER, then used multi-objective Pareto
filtering. This is much closer to the search space NOVA requires than local
mutagenesis.

## Methods worth using

| Method | What it adds | Evidence and practical status | NOVA role |
|---|---|---|---|
| [BoltzGen](https://pmc.ncbi.nlm.nih.gov/articles/PMC12697729/) | Joint target-conditioned all-atom backbone/pose generation, inverse folding, Boltz-2 refolding, and analysis | Open code; reported nanomolar VHHs for 6/9 novel targets with at most 15 tested per target. Its [official workflow](https://github.com/HannesStark/boltzgen) recommends a 50-design pilot followed by 10,000–60,000 designs. Already locked in our Pixi environment. | Primary generator now, but only after hotspot/scaffold pilots prove the pose distribution is competitive. |
| [RFantibody](https://www.nature.com/articles/s41586-025-09721-5) | Antibody-tuned RFdiffusion for dock/backbone generation, ProteinMPNN for sequence, antibody-tuned RF2 for filtering | Open [pipeline](https://github.com/RosettaCommons/RFantibody) and experimentally validated atomically accurate VHH designs. Exposes hotspot and CDR-length sweeps directly. | Second independent generator and a useful comparison if BoltzGen remains site-blind. |
| [Germinal](https://www.nature.com/articles/s41587-026-03187-0) | AlphaFold-Multimer hallucination regularized by an antibody language model, AbMPNN redesign, and cofolding | Functional antibodies across four targets with only 43–101 tested per antigen; open [repository](https://github.com/SantiagoMille/germinal). The official implementation strongly recommends AF3-calibrated filtering and substantially more VRAM than a 4090 for normal runs. | Strong method, but not the first deployment on the current 24 GB cards. |
| [mBER](https://github.com/manifoldbio/mber-open) | Backpropagation through AlphaFold-Multimer with VHH templates, explicit hotspots, and sequence conditioning | Open VHH CLI; the authors say VHH/small-target work can fit below 16 GB. Experimental design was million-scale, so its hit-rate economics differ from an epoch. | High-priority refinement lane starting from a competitive teacher pose. |
| [IgGM](https://github.com/TencentAI4S/IgGM) | Joint antigen-conditioned sequence/structure diffusion, including epitope-specific VHH generation | Open ICLR 2025 implementation; also one of three generators in the 2026 multi-method experimental campaign. | Independent proposal diversity and cross-method consensus. |
| [EasyNano](https://arxiv.org/abs/2606.12772) | Differentiable CDR-logit optimization against an ESMFold2 distogram with epitope proximity and a framework-pose prior | Open [code](https://github.com/YueHuLab/EasyNano); reports 10–20 minute optimization and large predicted ipTM gains, but no wet-lab validation in the paper. | Very attractive fast exploitation experiment once a credible dock/framework exists; validate every output with BoltzGen. |
| [TiDE-Ab](https://pubmed.ncbi.nlm.nih.gov/42635242/) | Time-dependent guided SE(3) flow matching that first establishes pose, then relaxes guidance for CDR refinement | Open [code](https://github.com/SNU-CSSB/TiDE-Ab); benchmark reports better epitope recall and far fewer clashes than RFantibody, but no comparable wet-lab campaign yet. | Research lane, after the experimentally validated/open methods above. |
| [AntiFold](https://pmc.ncbi.nlm.nih.gov/articles/PMC11961221/) | Antibody/nanobody-specific inverse folding, optionally antigen-aware | Open [code](https://github.com/oxpig/AntiFold); samples CDR sequences on an existing IMGT-numbered complex. | Sequence redesign after a strong pose; it cannot rescue a wrong dock by itself. |

The strongest recent scale lesson comes from the [2026 agent-guided VHH
study](https://www.biorxiv.org/content/10.64898/2026.04.13.717816v1.full):
multiple generators, target conformations, frameworks, CDR3 lengths, and
hotspot sets were searched, and candidates were Pareto-filtered instead of
collapsed early to one surrogate score. It generated 288,000 designs across
eight epitopes and three frameworks. That scale is not necessary for our first
diagnostic, but the factorial design is.

## Immediate experimental design

1. Refold strong recent public submissions with the exact NOVA scorer and keep
   their predicted complexes. They are structural teachers, never submission
   candidates.
2. Infer consensus target contacts using canonical mmCIF residue indices. Build
   generation specs from that target-matched artifact rather than hard-coding
   the current challenge in source.
3. Run small BoltzGen cells across hotspot subsets, four VHH scaffolds, both
   design checkpoints, and CDR geometry. Reject cells whose exact-scored tail
   has no credible interface signal before scaling.
4. Add EasyNano/mBER exploitation around the best exact-scored pose and an
   RFantibody or IgGM exploration lane. Require agreement from a model that did
   not generate the candidate where practical.
5. Preserve the eight structural/interaction values plus two developability
   values and select a Pareto set. A sequence surrogate may reject only the
   obvious bottom tail until held-parent top-tail recall is demonstrated.

Mining must remain disabled until a novel candidate has repeated exact scores
competitive with the live reference vector, not merely a favorable archive
bootstrap rank.
