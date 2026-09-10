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
    ap.add_argument(
        "--shortlist", type=int, default=1300,
        help="proxy-ranked full recombinants to retain in addition to every "
             "surviving candidate from the bounded CDR sweep",
    )
    ap.add_argument("--cdr-only", action="store_true", default=False,
                    help="recombine only inside CDRs")
    ap.add_argument("--exhaustive-cdr", action="store_true",
                    help="enumerate the CDR product when it fits --cdr-budget; "
                         "otherwise draw a deterministic weighted CDR sweep")
    ap.add_argument("--cdr-budget", type=int, default=768,
                    help="maximum retained CDR-only candidates (default: 768)")
    ap.add_argument(
        "--cdr-template", default="",
        help="verified current-gate archive sequence whose framework is held "
             "fixed during the CDR sweep",
    )
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
    from nanobody_opt.simulate_round import dense_rank_sums
    rank_sum = dense_rank_sums(df, SCORED)
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
    top50_kmers = [diversity.kmers(seq, 5) for seq in top50_seqs]
    existing = set(df.sequence.values)

    rng = np.random.default_rng(0)
    parents = df.sequence.values[elite_idx]
    cand, seen = [], set()

    # The archive changes over time, so never assume its CDR Cartesian product
    # is still the 768 combinations observed during the original analysis. A
    # list(product(...)) on a later, more diverse archive consumed hundreds of
    # GB before producing its first candidate. Enumerate lazily when the whole
    # product fits the explicit budget; otherwise retain a deterministic,
    # score-weighted CDR-only sweep of that size.
    cdr_sweep: set[str] = set()
    cdr_enumeration_complete = False
    cdr_template = ""
    if a.exhaustive_cdr:
        import itertools, math
        if a.cdr_budget < 0:
            raise SystemExit("--cdr-budget must be non-negative")
        if not a.cdr_template.strip():
            raise SystemExit(
                "--exhaustive-cdr requires an explicit --cdr-template that "
                "has passed the current Layer-2 gates"
            )
        cdr_template = a.cdr_template.strip().upper()
        if len(cdr_template) != L or set(cdr_template) - AAS:
            raise SystemExit("--cdr-template must be a valid archive-length sequence")
        if cdr_template not in existing:
            raise SystemExit("--cdr-template is not present in the selected archive era")
        reg0 = find_regions(cdr_template)
        cdr_pos = [p for p in reg0.cdr_positions() if len(pools[p][0]) > 1]
        total_cdr = math.prod(len(pools[p][0]) for p in cdr_pos)
        cdr_enumeration_complete = total_cdr <= a.cdr_budget
        if cdr_enumeration_complete:
            combos = itertools.product(*[pools[p][0] for p in cdr_pos])
            mode = "complete lazy enumeration"
        else:
            cdr_rng = np.random.default_rng(1)

            def sampled_combos():
                attempts = 0
                while len(cdr_sweep) < a.cdr_budget and attempts < a.cdr_budget * 100:
                    attempts += 1
                    yield tuple(
                        pools[p][0][cdr_rng.choice(len(pools[p][0]), p=pools[p][1])]
                        for p in cdr_pos
                    )

            combos = sampled_combos()
            mode = f"weighted sweep capped at {a.cdr_budget:,}"
        print(f"CDR space: {len(cdr_pos)} varying positions -> "
              f"{total_cdr:,} combinations; {mode}")
        before = len(cand)
        for combo in combos:
            s2 = list(cdr_template)
            for p, r in zip(cdr_pos, combo):
                s2[p] = r
            s2 = "".join(s2)
            if s2 in existing or s2 in seen:
                continue
            seen.add(s2); cand.append(s2); cdr_sweep.add(s2)
        print(f"  {len(cand) - before:,} CDR-only candidates are novel")
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
        ok, j = diversity.passes_diversity(s, top50_kmers, threshold=0.9, k=5)
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
    out = pd.DataFrame({
        "sequence": seqs,
        "max_jaccard_to_top50": [r[1] for r in rows],
        "proxy_rank_sum": pred,
        "cdr_sweep": [s in cdr_sweep for s in seqs],
        "exhaustive_cdr": [cdr_enumeration_complete and s in cdr_sweep for s in seqs],
        "cdr_template": [cdr_template if s in cdr_sweep else "" for s in seqs],
    }).sort_values("proxy_rank_sum")
    if a.shortlist < 0:
        raise SystemExit("--shortlist must be non-negative")
    cdr_out = out[out.cdr_sweep]
    sampled_out = out[~out.cdr_sweep].head(a.shortlist)
    out = pd.concat([cdr_out, sampled_out]).sort_values("proxy_rank_sum")
    n_cdr = int(out.cdr_sweep.sum())
    print(f"proxy: best {pred.min():.0f}, archive best {rank_sum.min():.0f}, "
          f"median {np.median(rank_sum):.0f}")
    if (pred < rank_sum.min()).sum():
        print(f"  WARNING {(pred < rank_sum.min()).sum()} predict below the archive "
              f"minimum -- extrapolating, do not trust the ordering there")
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out):,} candidates to {a.out}: all {n_cdr:,} "
          f"gate-passing CDR-sweep candidates plus "
          f"{len(out) - n_cdr:,} proxy-ranked full recombinants")


if __name__ == "__main__":
    main()
