"""Developability liabilities: 2 of the 10 scored metrics, free on CPU.

`boltzgen_config.yaml` ranks every miner on `liability_score` (min) and
`liability_num_violations` (min). Both are sequence-motif properties -- no fold,
no GPU, microseconds. Since scoring is rank_sum across all 10 metrics, carrying
avoidable liabilities forfeits rank on two of them regardless of how good the
binding is.

Standard antibody-developability motifs. Weights reflect relative severity in
common liability schemes; the exact weighting matters less than being at zero.
"""
import re

MOTIFS = [
    ("deamidation_NG",    r"NG",        1.0),
    ("deamidation_NS",    r"NS",        0.5),
    ("deamidation_NT",    r"NT",        0.3),
    ("isomerisation_DG",  r"DG",        1.0),
    ("isomerisation_DS",  r"DS",        0.5),
    ("hydrolysis_DP",     r"DP",        0.5),
    ("glycosylation_NxST", r"N[^P][ST]", 1.5),
    ("oxidation_M",       r"M",         0.3),
    ("oxidation_W",       r"W",         0.2),
    ("free_cysteine",     r"C",         2.0),
    ("tryptic_KK_RR",     r"(KK|RR)",   0.2),
]


def liabilities(seq: str, region: tuple[int, int] | None = None):
    """Return (score, violations, detail). `region` limits the scan to CDRs."""
    window = seq if region is None else seq[region[0]:region[1]]
    offset = 0 if region is None else region[0]
    detail, score, count = [], 0.0, 0
    for name, pattern, weight in MOTIFS:
        for m in re.finditer(pattern, window):
            # the two canonical VHH cysteines are structural, not liabilities
            if name == "free_cysteine":
                continue
            detail.append((name, offset + m.start(), m.group()))
            score += weight
            count += 1
    return score, count, detail


def is_clean(seq: str, cdr_regions) -> bool:
    """True if no liability motif falls inside any CDR."""
    for start, end in cdr_regions:
        if liabilities(seq, (start, end))[1]:
            return False
    return True
