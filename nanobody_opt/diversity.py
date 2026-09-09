"""Diversity gate + top-50 selection, ported from upstream.

Two DIFFERENT rankings are in play and conflating them is a live hazard:

  * What is SCORED (who wins the epoch): boltzgen_config.rank_by = rank_sum,
    the SUM of ranks across all 10 metrics.
  * What defines the TOP 50 used by the diversity gate: utils/minmax_weighted_rank
    -- BoltzGen paper Algorithm 2 -- the WORST (max) weighted rank across only
    6 metrics, with weights. A different set, a different aggregator.

Our proxy targets rank_sum (correct for scoring). The top-50 must use Algorithm 2.
"""
from __future__ import annotations
import pandas as pd

# utils/minmax_weighted_rank.py: METRIC_DEFS / PROTEIN_WEIGHTS
ALGO2_WEIGHTS = {
    "design_iiptm": 1.0,
    "design_ptm": 2.0,
    "min_design_to_target_pae": 1.0,
    "plip_hbonds_refolded": 2.0,
    "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 2.0,
}
ALGO2_DIRECTION = {
    "design_iiptm": "max",
    "design_ptm": "max",
    "min_design_to_target_pae": "min",
    "plip_hbonds_refolded": "max",
    "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max",
}


def minmax_weighted_rank(df: pd.DataFrame, k: int = 50) -> pd.DataFrame:
    """Top-k by worst weighted rank (BoltzGen Algorithm 2). Lower is better."""
    work = df.copy()
    ranks = {}
    for metric, w in ALGO2_WEIGHTS.items():
        ascending = ALGO2_DIRECTION[metric] == "min"
        ranks[metric] = work[metric].rank(ascending=ascending, method="average") / w
    work["algo2_score"] = pd.DataFrame(ranks, index=work.index).max(axis=1)
    return work.sort_values("algo2_score", ascending=True).head(k)


def kmers(seq: str, k: int = 5) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else set()


def kmer_jaccard(a: str, b: str, k: int = 5) -> float:
    """Exact Jaccard over k-mer SETS -- what the service's MinHash estimates."""
    ka, kb = kmers(a, k), kmers(b, k)
    if not ka or not kb:
        return 0.0
    union = len(ka | kb)
    return (len(ka & kb) / union) if union else 0.0


def passes_diversity(seq: str, top_seqs, threshold: float = 0.9,
                     k: int = 5) -> tuple[bool, float]:
    """Reject if Jaccard >= threshold against ANY of the target's top sequences."""
    query_kmers = kmers(seq, k)
    worst = 0.0
    for t in top_seqs:
        target_kmers = t if isinstance(t, set) else kmers(t, k)
        union = len(query_kmers | target_kmers)
        j = (len(query_kmers & target_kmers) / union) if union else 0.0
        if j > worst:
            worst = j
        if worst >= threshold:
            return False, worst
    return True, worst
