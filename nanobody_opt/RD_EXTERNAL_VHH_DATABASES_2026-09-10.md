# External VHH databases for P20809 preselection

Date: 2026-09-10 (live endpoints rechecked 2026-09-11 UTC)

Scope: R&D assessment only. This note does not change the live miner, graph, or
ranking code. Licensing observations are operational gates, not legal advice.

## Decision

Use external databases first as **structural-interface, VHH-naturalness, and
novelty priors**, not as direct P20809 binding labels. No P20809, IL11, IL-11,
or interleukin-11 match was found in the four catalogs inspected at record level
(PLAbDab-nano VHH, INDI2 AgAb, ANDD v2 VHH, and the current SAbDab2
single-domain export). OAS has no antigen-specificity field. The full SNAC-DB
archive was not downloaded because its license expressly restricts commercial,
text/data-mining, and AI-training use.

The recommended order is:

1. **SAbDab2-nano** for versioned, CC-BY structural/interface priors and
   antigen-aware leakage-safe splits.
2. **ANDD v2, experimental-only and source-allowlisted** for a small affinity
   proof of value. Do not use predicted ANTIPASTI values as labels.
3. **OAS/OAS-nano, sampled rather than bulk-ingested**, for camelid/VHH
   naturalness and out-of-distribution penalties only.
4. **PLAbDab-nano** for patent/literature provenance, target annotations, and
   known-sequence novelty checks. Clarify database reuse terms before putting
   its records in a commercial production artifact.
5. **INDI2 only after a commercial license**; keep it in an isolated R&D store
   until then.
6. **SNAC-DB only with written Sanofi permission** covering the intended TDM,
   AI, and commercial use.

This ordering treats the sources as heavily overlapping evidence, not six
independent votes.

## Source matrix

| Source | Access and license status | Useful fields | Missing / principal risk | Recommended use |
|---|---|---|---|---|
| [PLAbDab-nano](https://opig.stats.ox.ac.uk/webapps/plabdab-nano/) | Direct `all_sequences.csv.gz`, `vhh_sequences.csv.gz`, and `vnar_sequences.csv.gz` downloads; local Python search package, not a documented hosted REST API. Repository code is BSD-3-Clause and the paper is CC BY, but a separate license for the downloaded database was not visible in the export or landing page. Confirm reuse terms for commercial ingestion. | Raw sequence, VHH/VNAR type, CDR sequences/lengths, source/model IDs, definition, authors/title, organism, update date, free-text targets, and precomputed NanoBodyBuilder2 model reference. | No standardized affinity or assay; free-text targets; patents and duplicate records; model IDs may be shared by redundant sequences. | Novelty/IP-proximity flags, provenance, target-name search, VHH quality cross-check. |
| [INDI2](https://naturalantibody.com/indi2/) | Google Drive bulk folders in Parquet/Delta format; no documented supported query REST API. Official site permits non-commercial research by non-commercial entities and directs commercial users to contact the provider. Downloaded folder licenses are CC-BY-NC-4.0. | About 30M records across NGS, structures, patents, therapeutics, GenBank, and AgAb; sequence, source metadata, targets, antigen sequence, structures, and heterogeneous binding fields. | Commercial-use gate; cross-source redundancy is explicitly acknowledged; heuristic VHH classification can include false positives and miss synthetic VHHs. Binding values are not one common scale. | Quarantined R&D only unless licensed; potentially broad sequence/antigen coverage after modality and source QC. |
| [SNAC-DB](https://github.com/Sanofi-Public/SNAC-DB) | Versioned [Zenodo archive](https://doi.org/10.5281/zenodo.15870002) plus local curation pipelines; no hosted query API. The repository [license](https://github.com/Sanofi-Public/SNAC-DB/blob/main/LICENSE.txt) allows academic/non-profit non-commercial research only, requires contact for commercial benefit, and reserves TDM/AI-training rights. | Cleaned PDB/mmCIF-derived complexes, full sequences including missing residues, atom37 coordinates, B-factors, IMGT numbering, chain/pairing metadata, summary experimental metadata, and redundancy/test-split utilities. | No affinity labels; roughly 11.5 GB archive; license blocks the proposed live use without written permission. | Do not ingest now. Reassess only if permission is obtained; SAbDab2 provides a clearer-licensed structural starting point. |
| [ANDD v2](https://zenodo.org/records/18151718) | Zenodo workbook (13.3 MB), structures archive (2.2 GB), data dictionary, and QC report; Zenodo API/file URLs support automated download, but no record-query API. Landing record is CC BY 4.0. | Source/release/PDB/method, VHH/antibody type, heavy/light and antigen accessions/names/sequences, CDRs, Kd in M, binding free energy, assay method, mutation, real/predicted indicator, provenance. | Aggregates 15 upstream sources, so row counts are not independent evidence. Some upstream sources have narrower terms than the aggregate record; use a source allowlist. Heterogeneous assays and predicted affinity create label and model-circularity risks. | Fast experimental-affinity PoV after `Predicted_or_Not=real`, unit/assay QC, deduplication, and source-license review. |
| [SAbDab2-nano](https://sabdab.opig.stats.ox.ac.uk/search-nanobodies) | Live UI/API, custom/bulk mmCIF and CSV downloads, updated weekly; UI 2.1.0/API 2.0.10 at review. Current site states CC BY 4.0. A versioned [ML dataset](https://zenodo.org/records/20083995) supplies antigen-aware splits. | Stable antibody/instance/PDB IDs, IMGT-numbered VH/CDRs, chain mappings, antigen chain/type/name, dates, construct/species, method, resolution and other structure QC. Downloaded complexes supply antigen sequence/coordinates. | No affinity values. Multiple instances/assemblies of the same antibody can overcount evidence. Antigen names require accession/sequence normalization. | First ingest: interface contacts, paratope geometry, CDR conformational clusters, antigen-aware nearest-neighbor priors, and structural QC. |
| [OAS](https://opig.stats.ox.ac.uk/webapps/oas/) | CC BY 4.0; metadata-filtered bulk `.csv.gz` download via generated `wget` script, not a documented REST API. Complete OAS can exceed 700 GB. The VHH-specific [OAS-nano](https://oas-nano.opig.stats.ox.ac.uk/documentation/) view reports over 72M unique VHH sequences from 18 studies. | Raw/translated sequence, productivity, V/D/J calls and identities, framework/CDR/junction fields, ANARCI numbering, redundancy, and study/run/subject/species/isotype/disease/vaccine/B-cell metadata. | No target, binding, affinity, or complex structure. Read abundance is not binding evidence. Study/run/subject leakage and overlap with INDI NGS or model pretraining are likely. Capture a terms snapshot for the exact OAS-nano download path. | Sample camelid/VHH data for naturalness, liability, and OOD priors; never use as a positive binder set. |

### Record-level observations

The current files/endpoints were sampled or downloaded into a temporary R&D
directory and not committed:

- PLAbDab-nano's current VHH CSV contained 4,457 rows and 1,969 unique cleaned
  80-160 aa VHH sequences. The paper describes about 4,913 annotated VHH/VNAR
  sequences across the database; row counts and unique sequences are therefore
  different quantities.
- INDI2 reports 29,224,937 AbNGS sequences, 2,768 nanobody-antigen complexes,
  4,160 unbound structures, 67,479 patent sequences, 4,169 GenBank sequences,
  and 110,935 AgAb records. In the downloaded AgAb Parquet, 67,058/110,935 rows
  used an AlphaSeq-like measurement type, 43,754 were Boolean, and only 123
  were typed as Kd. These fields must not be pooled as if all were molar Kd.
- ANDD v2's landing page reports 48,683 sequences, 24,941 structures, 12,575
  antigen sequences, and 9,557 affinity values. The workbook contained 48,800
  source rows, of which 30,119 were labeled Nanobody/VHH; those collapsed to
  24,797 unique raw heavy sequences. Among VHH rows, 3,178 had a PDB ID and
  1,724 had a non-null Kd field. Only 1,738 VHH rows were marked real, 79 were
  marked predicted, and most had no real/predicted flag, so filtering cannot be
  inferred from presence of a numeric field alone.
- The current SAbDab2 single-domain summary endpoint
  (`/api/download/all-single-domain-summary`) returned 4,788 instances from
  2,485 PDB IDs and 1,407 stable SAbDab2 IDs. It included 4,602 SD-H, 95 SD-L,
  and 91 VNAR instances; 4,438 instances listed an antigen chain. This is an
  instance-level structural catalog, not 4,788 independent VHH sequences.

## Leakage and deduplication audit against NOVA

The audit used the public snapshot
`artifacts/rd/mutation_ranker_v1_20260910/api_field_snapshot.csv.gz` (10,092
rows and 10,092 unique sequences). External sequences were uppercased, stripped
to the 20 canonical amino acids, restricted to 80-160 aa, and deduplicated by
exact normalized sequence before comparison.

Results for the directly downloaded sequence catalogs:

- Combined ANDD-VHH plus PLAbDab-VHH: 22,907 unique external sequences.
- Exact overlap with NOVA: **0**.
- MMseqs2 search against NOVA, requiring at least 70% alignment coverage, found
  no best hit at or above 90% sequence identity. There were 61 best hits at or
  above 80%, two at or above 85%, and none at or above 90%. With at least 95%
  coverage on both sequences, 30 hits reached 80% identity. The best
  full-coverage identity was 81.3%, which is not evidence of a duplicated NOVA
  candidate.
- PLAbDab contributed little independent sequence mass beyond ANDD: 1,953 of
  its 1,969 unique cleaned VHH sequences (99.2%) occurred exactly in ANDD.
- Of 1,376 unique normalized VH strings from the current SAbDab2 single-domain
  export, 915 (66.5%) occurred in ANDD and 671 (48.8%) occurred in PLAbDab.

This is a useful initial screen, not proof that leakage is absent. Full OAS,
INDI2 NGS/patent/structure folders, and SNAC-DB were not exhaustively aligned.
Further, sequence-disjoint evaluation can still leak through the same antigen,
PDB assembly, patent family, lineage, donor, study, or a nearly identical CDR3.

Minimum split and provenance rules:

1. Deduplicate VHHs on normalized IMGT variable-domain SHA-256 and retain every
   raw source record; also cluster by CDR1+CDR2+CDR3 and by sequence identity.
2. Deduplicate antigens on UniProt accession + isoform + sequence version and
   mature-chain SHA-256. Do not silently collapse the P20809 precursor and
   mature cytokine sequence.
3. Group complexes by PDB biological assembly, antibody chains, antigen chains,
   and stable antibody ID. Treat symmetry mates/conformations as related
   instances, not independent examples.
4. For antigen-aware evaluation, hold out antibody sequence clusters **and**
   antigen sequence/family clusters. SAbDab2's `abag_split_sd.csv` is the better
   starting split than an antibody-only split.
5. Group patents by patent family and repertoires by study/run/subject. Never
   random-split rows from the same group.
6. Attach `released_at`, `retrieved_at`, source release/version, file hash, and
   license/terms snapshot to every record. Only sources released before a NOVA
   evaluation epoch may contribute features to that epoch.
7. Keep experimental, predicted, Boolean, enrichment, and assay-specific scores
   as different evidence kinds. Preserve raw value/unit/conditions alongside
   any normalized molar Kd.

## Graph ingestion contract

`mutation_network.py` currently stores scored NOVA sequences in `nodes`, their
numeric evidence in `observations`, Hamming edges, mutation effects, proposal
scores, and lineage edges. External unscored sequences must **not** be inserted
as observations with synthetic scores: doing so would contaminate mutation
effects and the current score semantics.

Add a separate external-evidence layer, with foreign keys into NOVA only after
explicit matching:

```sql
external_datasets(
  dataset_id TEXT PRIMARY KEY, name TEXT, release TEXT, retrieved_at TEXT,
  source_url TEXT, license_id TEXT, commercial_use_allowed INTEGER,
  tdm_ai_allowed INTEGER, redistribution_allowed INTEGER,
  artifact_sha256 TEXT, terms_snapshot_uri TEXT
)

external_records(
  record_id TEXT PRIMARY KEY, dataset_id TEXT, source_accession TEXT,
  source_url TEXT, released_at TEXT, record_sha256 TEXT, raw_json TEXT,
  UNIQUE(dataset_id, source_accession)
)

external_vhh(
  vhh_id TEXT PRIMARY KEY, sequence_sha256 TEXT UNIQUE, raw_sequence TEXT,
  variable_sequence TEXT, imgt_sequence TEXT, length INTEGER, modality TEXT,
  source_species TEXT, cdr1 TEXT, cdr2 TEXT, cdr3 TEXT,
  numbering_scheme TEXT, quality_flags_json TEXT
)

external_antigens(
  antigen_id TEXT PRIMARY KEY, sequence_sha256 TEXT,
  uniprot_accession TEXT, isoform TEXT, sequence_version TEXT, name TEXT,
  organism_taxid TEXT, raw_sequence TEXT, mature_sequence TEXT,
  quality_flags_json TEXT
)

external_interactions(
  interaction_id TEXT PRIMARY KEY, vhh_id TEXT, antigen_id TEXT,
  record_id TEXT, evidence_kind TEXT, raw_value TEXT, value REAL, unit TEXT,
  assay TEXT, experimental_or_predicted TEXT, confidence TEXT,
  conditions_json TEXT
)

external_complexes(
  complex_id TEXT PRIMARY KEY, interaction_id TEXT, pdb_id TEXT,
  assembly_id TEXT, vhh_chain_ids TEXT, antigen_chain_ids TEXT,
  method TEXT, resolution REAL, structure_sha256 TEXT, structure_uri TEXT,
  release_date TEXT
)

external_record_links(
  record_id TEXT, entity_kind TEXT, entity_id TEXT,
  PRIMARY KEY(record_id, entity_kind, entity_id)
)

external_nova_links(
  external_vhh_id TEXT, nova_node_id INTEGER, alignment_method TEXT,
  sequence_identity REAL, query_coverage REAL, target_coverage REAL,
  cdr1_identity REAL, cdr2_identity REAL, cdr3_identity REAL,
  is_exact INTEGER, cutoff_snapshot TEXT,
  PRIMARY KEY(external_vhh_id, nova_node_id, alignment_method, cutoff_snapshot)
)

proposal_external_features(
  epoch INTEGER, node_id INTEGER, feature_version TEXT, features_json TEXT,
  PRIMARY KEY(epoch, node_id, feature_version)
)
```

Foreign-key `nova_node_id` and `node_id` to the existing `nodes(id)`. Store one
interaction row per measurement rather than overwriting measurements. Generate
`proposal_external_features` only from a license- and date-allowlisted view.
Feature generation should be reproducible from dataset/file hashes and a
feature-version string.

## Small proof-of-value experiment

Freeze the existing 200-training-epoch / 20-holdout-epoch ranker split and its
current metrics before ingesting anything. The experiment changes feature
generation only; it does not modify live candidate generation or overwrite
NOVA observations.

### Inputs

1. A pinned SAbDab2 ML release, restricted to SD-H/VHH-like structures and the
   antigen-aware split.
2. ANDD v2 VHH rows with `Predicted_or_Not=real`, valid experimental units and
   assay metadata, plus an approved upstream-source allowlist. Run an ablation
   excluding every INDI-derived record.
3. Optionally, a reproducible 50k-100k camelid/VHH sample from OAS-nano for a
   naturalness-only feature. Do not ingest the whole corpus.

### Candidate features

- nearest external VHH whole-domain identity and CDR3 identity;
- distance to SAbDab2 VHH CDR length/conformation clusters;
- paratope/interface-contact propensity by IMGT position from high-quality
  VHH-antigen complexes;
- nearest experimental-affinity neighborhood median and support, gated on both
  VHH similarity and antigen sequence/family similarity;
- OAS-nano naturalness percentile / out-of-distribution distance;
- nearest patent/literature sequence identity as a novelty/IP-review flag, not
  a binding score.

Because none of the inspected sources supplies a P20809 positive, target-family
or structural similarity must be an explicit gate. A high-affinity VHH against
an unrelated antigen is not a positive training label for IL-11.

### Evaluation and promotion rule

Compare the frozen baseline with `+SAbDab2 structure`, `+ANDD experimental`, and
`+OAS naturalness` ablations using the existing epoch-grouped walk-forward
evaluation. Report median field Spearman, winner top-10 rate, winner rank, and
top-160/proposal recall with bootstrap confidence intervals. Also stratify by
distance from the NOVA graph and by CDR3 novelty.

A reasonable small-experiment promotion gate is a positive bootstrap interval
for the primary ranking metric and at least a 10 percentage-point improvement
in winner-top-10 rate, with no meaningful regression in liability filtering,
inference cost, or proposal recall. If the gate is not met, retain external
databases only for QC, naturalness, novelty, and audit—not for ranking.

Given the lack of target-specific positives and the low external/NOVA sequence
overlap, the expected benefit is modest regularization and better tie-breaking,
not dramatic direct binding transfer.

## Primary sources

- P20809/IL-11: [UniProt P20809](https://www.uniprot.org/uniprotkb/P20809/entry)
- PLAbDab-nano: [database/downloads](https://opig.stats.ox.ac.uk/webapps/plabdab-nano/), [paper](https://doi.org/10.1093/nar/gkae881), [software repository](https://github.com/oxpig/PLAbDab_nano)
- INDI2: [current database/access page](https://naturalantibody.com/indi2/), [INDI paper](https://doi.org/10.1093/nar/gkab1021)
- SNAC-DB: [repository and pipeline](https://github.com/Sanofi-Public/SNAC-DB), [license](https://github.com/Sanofi-Public/SNAC-DB/blob/main/LICENSE.txt), [dataset DOI](https://doi.org/10.5281/zenodo.15870002), [2026 paper](https://doi.org/10.1002/pro.70655)
- ANDD: [v2 dataset and license](https://zenodo.org/records/18151718), [Scientific Data paper](https://doi.org/10.1038/s41597-026-06878-0)
- SAbDab2/SAbDab-nano: [current site and license](https://sabdab.opig.stats.ox.ac.uk/), [data/API details](https://sabdab.opig.stats.ox.ac.uk/about), [versioned ML dataset](https://zenodo.org/records/20083995), [SAbDab-nano paper](https://doi.org/10.1093/nar/gkab1050)
- OAS: [current database and license](https://opig.stats.ox.ac.uk/webapps/oas/), [download/data format documentation](https://opig.stats.ox.ac.uk/webapps/oas/documentation), [VHH-specific OAS-nano documentation](https://oas-nano.opig.stats.ox.ac.uk/documentation/)
