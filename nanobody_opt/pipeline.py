"""Experimental nanobody candidate pipeline: generate -> gate -> proxy rank.

Every gate is enforced locally before anything is committed. We submit
num_sequences=1 and a single failed gate rejects the WHOLE submission for the
epoch, so an unchecked commit risks the entire epoch.

The search operates in a thin shell around known-good sequences:
  * FAR enough    -- k-mer Jaccard < 0.9 vs the target's top 50 (hard gate)
  * CLOSE enough  -- on the natural VHH manifold, where framework/nativeness
                     filters are more likely to pass.

The archive proxy is a diagnostic ordering only. Its Spearman correlation was
0.045 on the 2026-09-09 exact-scored synthetic probe, so output from this script
must not be treated as submission-ready without Layer-2 and exact scoring.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanobody_opt import gates, diversity
from nanobody_opt.simulate_round import dense_rank_sums

SCORED = {  # boltzgen rank_sum: all 10 metrics, this is what WINS the epoch
    "design_iiptm": "max", "design_ptm": "max", "design_to_target_iptm": "max",
    "min_design_to_target_pae": "min", "interaction_pae": "min",
    "plip_hbonds_refolded": "max", "plip_saltbridge_refolded": "max",
    "delta_sasa_refolded": "max", "liability_score": "min",
    "liability_num_violations": "min",
}
AAS = "ACDEFGHIKLMNPQRSTVWY"
IX = {a: i for i, a in enumerate(AAS)}


def onehot(seqs, L):
    X = np.zeros((len(seqs), L * 20), dtype=np.float32)
    for r, s in enumerate(seqs):
        for p, c in enumerate(s):
            j = IX.get(c)
            if j is not None:
                X[r, p * 20 + j] = 1.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--config", required=True, help="upstream config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-candidates", type=int, default=40000)
    ap.add_argument("--muts", type=int, default=4,
                    help="CDR mutations per candidate; 1 always fails diversity")
    ap.add_argument("--seed-pool", type=int, default=400,
                    help="how many high-ranked archive sequences to perturb")
    ap.add_argument("--shortlist", type=int, default=200)
    ap.add_argument("--recent-only", type=int, default=0,
                    help="use only the most recent N archive rows (current scoring era)")
    ap.add_argument("--min-support", type=int, default=25,
                    help="a residue must appear this often at a position to be usable")
    a = ap.parse_args()

    import yaml
    a_min_support = a.min_support
    cfg = yaml.safe_load(open(a.config))["nanobody_requirements"]

    df = pd.read_csv(a.archive).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    # ERA MATTERS, and ignoring it is why the first attempt placed #38 of 50.
    # The archive spans multiple scoring eras: the oldest rows average
    # design_iiptm 0.385 / delta_sasa 1223, the most recent 0.758 / 717. Ranking
    # across the whole archive therefore promotes sequences that were strong in
    # a WEAKER era, and seeding from them produces candidates that lose to the
    # current field. Competition is against this epoch's ~50 rivals, so only the
    # recent era is informative -- for the seeds AND for the proxy's training.
    if a.recent_only:
        keep_n = min(a.recent_only, len(df))
        df = df.tail(keep_n).reset_index(drop=True)
        print(f"restricted to the most recent {keep_n:,} submissions (current era)")
    L = int(df.sequence.str.len().mode()[0])
    df = df[df.sequence.str.len() == L].reset_index(drop=True)
    print(f"archive: {len(df):,} sequences at the modal length {L}")

    # what is SCORED
    rank_sum = dense_rank_sums(df, SCORED)
    # what defines the TOP 50 for the diversity gate -- a DIFFERENT algorithm
    top50 = diversity.minmax_weighted_rank(df, k=cfg.get("num_top_sequences", 50))
    top50_seqs = top50.sequence.tolist()
    overlap = len(set(top50_seqs) & set(df.sequence.values[np.argsort(rank_sum)[:50]]))
    print(f"top-50 by Algorithm 2 vs by rank_sum: {overlap}/50 overlap "
          f"-- using Algorithm 2 for the gate, rank_sum for the proxy target")

    # proxy trained WITHOUT the top 1%, so ranking is not memorisation
    keep = np.argsort(rank_sum)[int(0.01 * len(df)):]
    proxy = Ridge(alpha=1.0).fit(onehot(df.sequence.values[keep], L), rank_sum[keep])

    from nanobody_opt.regions import find_regions
    existing = set(df.sequence.values)
    rng = np.random.default_rng(0)

    # Support-constrained alphabet. An unconstrained argmin over the proxy is
    # ADVERSARIAL to it: a one-hot Ridge is additive, so mutations to residues
    # the archive never places at a position carry unregularised weights that
    # simply stack. Measured: selecting the top 200 that way produced 200/200
    # predictions of a NEGATIVE rank_sum, which cannot exist (the floor is ~10).
    # Restricting each position to residues actually observed there keeps every
    # query inside the region where the proxy measured rho 0.88.
    # This does not cost novelty: novelty comes from the COMBINATION of
    # observed residues, which is still combinatorially vast.
    arr_arch = np.array([list(x) for x in df.sequence.values])
    allowed = []
    for pos in range(L):
        col = pd.Series(arr_arch[:, pos]).value_counts()
        allowed.append([a for a, c in col.items()
                        if c >= a_min_support and a in AAS])

    # perturb the best-scoring archive sequences inside their CDRs only
    seeds = df.sequence.values[np.argsort(rank_sum)[:a.seed_pool]]
    cand, seen = [], set()
    for _ in range(a.n_candidates):
        base = seeds[rng.integers(len(seeds))]
        try:
            reg = find_regions(base)
        except Exception:
            continue  # not VHH-shaped by our anchors; skip rather than crash
        cdr_pos = reg.cdr_positions()
        cdr3_pos = list(range(*reg.cdr3))
        s = list(base)
        # >=1 mutation in CDR3 is required by the CDR-mutation rule
        picks = [int(rng.choice(cdr3_pos))]
        picks += [int(p) for p in rng.choice(
            [p for p in cdr_pos if p != picks[0]], a.muts - 1, replace=False)]
        for p in picks:
            opts = [c for c in allowed[p] if c != s[p]]
            if not opts:
                continue  # nothing else was ever observed here; leave it alone
            s[p] = opts[rng.integers(len(opts))]
        s = "".join(s)
        if s == base:
            continue
        if s in existing or s in seen:
            continue
        seen.add(s)
        cand.append(s)
    print(f"generated {len(cand):,} unique novel candidates ({a.muts} CDR mutations each)")

    # ---- gate gauntlet, in the validator's own order ----
    rows, reasons = [], {}
    for s in cand:
        f = gates.check_sequence(s, cfg)
        if f:
            reasons.setdefault("layer1:" + f[0].split("=")[0], 0)
            reasons["layer1:" + f[0].split("=")[0]] += 1
            continue
        ok, j = diversity.passes_diversity(
            s, top50_seqs, threshold=0.9, k=5)
        if not ok:
            reasons["diversity_jaccard>=0.9"] = reasons.get("diversity_jaccard>=0.9", 0) + 1
            continue
        rows.append((s, j))
    print(f"\ngate gauntlet: {len(rows):,} / {len(cand):,} survive "
          f"({100*len(rows)/max(1,len(cand)):.1f}%)")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  rejected by {k}: {v:,}")
    if not rows:
        sys.exit("no candidate survived the gates")

    seqs = [r[0] for r in rows]
    pred = proxy.predict(onehot(seqs, L))
    out = pd.DataFrame({"sequence": seqs,
                        "max_jaccard_to_top50": [r[1] for r in rows],
                        "proxy_rank_sum": pred}).sort_values("proxy_rank_sum")
    out.head(a.shortlist).to_csv(a.out, index=False)
    print(f"\nwrote {min(a.shortlist, len(out))} shortlisted candidates to {a.out}")
    print(f"proxy_rank_sum (lower=better): best {pred.min():.0f}, "
          f"archive best {rank_sum.min():.0f}, archive median {np.median(rank_sum):.0f}")
    n_impossible = int((pred < rank_sum.min()).sum())
    if n_impossible:
        print(f"  WARNING: {n_impossible:,} candidates predict below the archive "
              f"minimum -- the proxy is extrapolating and MUST NOT be trusted there.")
    else:
        print("  all predictions lie inside the archive's observed range "
              "(the proxy is interpolating, not extrapolating)")
    print("\nNOT yet checked locally (need external tools):")
    for g in gates.EXTERNAL_GATES:
        if g != "diversity_vs_top50":
            print(f"  - {g}")


if __name__ == "__main__":
    main()
