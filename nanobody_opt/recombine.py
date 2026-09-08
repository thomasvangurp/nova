"""Generate candidates by RECOMBINING residues seen in high-scoring sequences.

Why this and not mutation. Measured on the real field: our 4-5 random CDR
mutations produced candidates ranking #38 of 50, worse than randomly chosen
archive sequences -- because the field is converged, so a random substitution
off an optimum is almost always downhill, and a proxy at rho 0.345 cannot pick
the rare good one out of thousands of bad ones.

Recombination changes the sampling distribution rather than the ranker. Every
residue placed at a position is one that ALREADY appears at that position in a
high-scoring sequence, so each individual choice is field-validated; only the
COMBINATION is new. That is also what keeps it novel: the archive constrains
residues, not their joint assignment.

Elites are drawn from the recent scoring era only. The archive spans eras --
oldest rows average design_iiptm 0.385 vs 0.758 for recent -- so "high scoring"
computed archive-wide would promote sequences that were strong in a weaker era.
"""
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd

SCORED = {"design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
          "min_design_to_target_pae": "min", "interaction_pae": "min",
          "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
          "delta_sasa_refolded": "max", "liability_score": "min",
          "liability_num_violations": "min"}
AAS = set("ACDEFGHIKLMNPQRSTVWY")


def elite_pools(elites: np.ndarray, weights: np.ndarray, L: int):
    """Per position: the residues elites use there, and how often (score-weighted).

    Weighting by elite rank means a residue used by the very best sequences is
    sampled more often than one used by a marginal elite.
    """
    pools = []
    for pos in range(L):
        col = elites[:, pos]
        counts = {}
        for res, w in zip(col, weights):
            if res in AAS:
                counts[res] = counts.get(res, 0.0) + w
        residues = sorted(counts, key=lambda r: -counts[r])
        probs = np.array([counts[r] for r in residues], dtype=float)
        pools.append((residues, probs / probs.sum()))
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recent-only", type=int, default=6000,
                    help="restrict to the current scoring era")
    ap.add_argument("--elites", type=int, default=200,
                    help="how many top sequences donate their residues")
    ap.add_argument("--n-candidates", type=int, default=40000)
    ap.add_argument("--shortlist", type=int, default=200)
    ap.add_argument("--cdr-only", action="store_true", default=False,
                    help="recombine only inside CDRs (space is just 768 -- see --exhaustive)")
    ap.add_argument("--exhaustive-cdr", action="store_true",
                    help="enumerate ALL CDR combinations (768) before sampling the rest")
    a = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nanobody_opt import gates, diversity
    from nanobody_opt.regions import find_regions
    import yaml
    cfg = yaml.safe_load(open(a.config))["nanobody_requirements"]

    df = pd.read_csv(a.archive).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    if a.recent_only:
        df = df.tail(min(a.recent_only, len(df))).reset_index(drop=True)
    L = int(df.sequence.str.len().mode()[0])
    df = df[df.sequence.str.len() == L].reset_index(drop=True)
    rank_sum = sum(df[m].rank(ascending=(d == "min")).values for m, d in SCORED.items())
    order = np.argsort(rank_sum)
    print(f"pool {len(df):,} sequences (recent era), length {L}")

    elite_idx = order[: a.elites]
    elites = np.array([list(s) for s in df.sequence.values[elite_idx]])
    # weight: best elite gets weight ~1, worst elite ~1/elites
    w = np.linspace(1.0, 1.0 / a.elites, len(elite_idx))
    pools = elite_pools(elites, w, L)
    varied = [p for p in range(L) if len(pools[p][0]) > 1]
    print(f"elites: {len(elite_idx)}; positions where elites differ: {len(varied)}")

    top50 = diversity.minmax_weighted_rank(df, k=cfg.get("num_top_sequences", 50))
    top50_seqs = top50.sequence.tolist()
    existing = set(df.sequence.values)

    rng = np.random.default_rng(0)
    parents = df.sequence.values[elite_idx]
    cand, seen = [], set()

    # Elites vary at only 9 CDR positions -> 768 CDR combinations total. That is
    # small enough to enumerate COMPLETELY, so nothing in that subspace is missed
    # by sampling luck. The 18 framework variable positions are what make the
    # full space large (1.1e14); recombining them is safe because every residue
    # comes from an elite that already passed nativeness, humanness, VHH
    # classification and TNP -- unlike a random mutation, which passes nothing.
    if a.exhaustive_cdr:
        import itertools
        best = parents[0]
        reg0 = find_regions(best)
        cdr_pos = [p for p in reg0.cdr_positions() if len(pools[p][0]) > 1]
        combos = list(itertools.product(*[pools[p][0] for p in cdr_pos]))
        print(f"exhaustive CDR enumeration: {len(cdr_pos)} varying CDR positions "
              f"-> {len(combos):,} combinations")
        for combo in combos:
            s2 = list(best)
            for p, r in zip(cdr_pos, combo):
                s2[p] = r
            s2 = "".join(s2)
            if s2 in existing or s2 in seen:
                continue
            seen.add(s2); cand.append(s2)
        print(f"  {len(cand):,} of them are novel")
    for _ in range(a.n_candidates):
        base = parents[rng.integers(len(parents))]
        try:
            reg = find_regions(base)
        except Exception:
            continue
        positions = reg.cdr_positions() if a.cdr_only else list(range(L))
        s = list(base)
        for p in positions:
            residues, probs = pools[p]
            if len(residues) > 1:
                s[p] = residues[rng.choice(len(residues), p=probs)]
        s = "".join(s)
        if s in existing or s in seen:
            continue
        seen.add(s)
        cand.append(s)
    print(f"generated {len(cand):,} unique novel recombinants")

    rows, rej = [], {}
    for s in cand:
        f = gates.check_sequence(s, cfg)
        if f:
            k = "layer1:" + f[0].split("=")[0]; rej[k] = rej.get(k, 0) + 1; continue
        ok, j = diversity.passes_diversity(s, top50_seqs, threshold=0.9, k=5)
        if not ok:
            rej["diversity"] = rej.get("diversity", 0) + 1; continue
        rows.append((s, j))
    print(f"gate gauntlet: {len(rows):,}/{len(cand):,} survive")
    for k, v in sorted(rej.items(), key=lambda x: -x[1]):
        print(f"  rejected by {k}: {v:,}")
    if not rows:
        raise SystemExit("nothing survived the gates")

    # how far are recombinants from their nearest elite? (sanity: novel but on-manifold)
    el = set(df.sequence.values[elite_idx])
    sample = [r[0] for r in rows[:200]]
    dmin = []
    for s in sample:
        dmin.append(min(sum(1 for x, y in zip(s, e) if x != y) for e in list(el)[:100]))
    print(f"mutations from the nearest elite: median {int(np.median(dmin))}, "
          f"min {min(dmin)}, max {max(dmin)}")

    # Rank the survivors with the proxy so the framework space can be sampled by
    # merit rather than at random. The proxy is trained WITHOUT the top 1% so its
    # ordering is not memorisation, and only on the recent era.
    from sklearn.linear_model import Ridge
    AA = "ACDEFGHIKLMNPQRSTVWY"; IX = {c: i for i, c in enumerate(AA)}

    def onehot(seqs):
        X = np.zeros((len(seqs), L * 20), dtype=np.float32)
        for r, sq in enumerate(seqs):
            for pos, c in enumerate(sq):
                j = IX.get(c)
                if j is not None:
                    X[r, pos * 20 + j] = 1.0
        return X

    keep = order[int(0.01 * len(df)):]
    proxy = Ridge(alpha=1.0).fit(onehot(df.sequence.values[keep]), rank_sum[keep])
    seqs = [r[0] for r in rows]
    pred = proxy.predict(onehot(seqs))
    exhaustive = set(cand[:len(combos)]) if a.exhaustive_cdr else set()
    out = pd.DataFrame({
        "sequence": seqs,
        "max_jaccard_to_top50": [r[1] for r in rows],
        "proxy_rank_sum": pred,
        "exhaustive_cdr": [s in exhaustive for s in seqs],
    }).sort_values("proxy_rank_sum")
    n_exh = int(out.exhaustive_cdr.sum())
    print(f"proxy: best {pred.min():.0f}, archive best {rank_sum.min():.0f}, "
          f"median {np.median(rank_sum):.0f}")
    if (pred < rank_sum.min()).sum():
        print(f"  WARNING {(pred < rank_sum.min()).sum()} predict below the archive "
              f"minimum -- extrapolating, do not trust the ordering there")
    out.to_csv(a.out, index=False)
    print(f"wrote ALL {len(out):,} gate-passing candidates to {a.out} "
          f"({n_exh} of them the exhaustive CDR set)")


if __name__ == "__main__":
    main()
