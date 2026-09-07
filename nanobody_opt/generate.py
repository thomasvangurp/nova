"""Generator that satisfies every nanobody gate, ranked by the proxy.

Constraints, all from the validator:
  1. exact uniqueness      -- cannot resubmit an existing sequence
  2. identity < 0.95 to any of the target's TOP 50   (nanobody_validity.py:109-140)
  3. VHH hallmarks preserved -- framework must stay intact
  4. rank_sum across 10 metrics is what is scored

Strategy forced by the evidence: the proxy CANNOT generate (exploited to impossible
values at every constraint level), so candidates come from perturbing real
gate-passing sequences -- on-manifold by construction -- and the proxy only RANKS
them, which is where it measures rho 0.88 / 23x enrichment.
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0, "/root")
from compound_miner.vhh_regions import find_regions

P="/root/P20809_nanobodies.csv"
df = pd.read_csv(P).drop_duplicates(subset=["sequence"]).reset_index(drop=True)
L = df.sequence.str.len().mode()[0]; df = df[df.sequence.str.len()==L].reset_index(drop=True)
DIR = {"design_iiptm":"max","design_ptm":"max","design_to_target_iptm":"max",
       "min_design_to_target_pae":"min","interaction_pae":"min","plip_hbonds_refolded":"max",
       "plip_saltbridge_refolded":"max","delta_sasa_refolded":"max",
       "liability_score":"min","liability_num_violations":"min"}
y = sum(df[m].rank(ascending=(d=="min")).values for m,d in DIR.items())
arr = np.array([list(s) for s in df.sequence]); order = np.argsort(y)
existing = set(df.sequence.values)
top50 = arr[order[:50]]
AAS="ACDEFGHIKLMNPQRSTVWY"; ix={a:i for i,a in enumerate(AAS)}
def oh(seqs):
    X=np.zeros((len(seqs),L*20),dtype=np.float32)
    for r,s in enumerate(seqs):
        for p,c in enumerate(s):
            j=ix.get(c)
            if j is not None: X[r,p*20+j]=1.0
    return X

# proxy trained WITHOUT the top 1% so ranking is not memorisation
hold=set(order[:len(df)//100].tolist()); tr=[i for i in range(len(df)) if i not in hold]
model=Ridge(alpha=1.0).fit(oh(df.sequence.values[tr]), y[tr])

def max_identity_to_top50(seq):
    a=np.array(list(seq))
    return (top50 == a).sum(axis=1).max() / L

# seeds: best sequences that already pass the gate
ident=np.array([ (top50==arr[i]).sum(axis=1).max()/L for i in range(len(df)) ])
passing=np.where(ident < 0.95)[0]
seeds=[df.sequence.values[i] for i in passing[np.argsort(y[passing])[:25]]]
print(f"seeds: {len(seeds)} gate-passing sequences, true rank_sum "
      f"{y[passing][np.argsort(y[passing])[0]]:,.0f}..{y[passing][np.argsort(y[passing])[24]]:,.0f}")

# freeze framework: mutate CDR positions only, so VHH hallmarks survive
try:
    cons="".join(pd.Series(arr[:,p]).mode()[0] for p in range(L))
    cdr=set(find_regions(cons).cdr_positions())
except Exception:
    cdr=set(range(L))
print(f"mutable CDR positions: {len(cdr)} of {L}\n")

rng=np.random.default_rng(0)
cands=[]
for s0 in seeds:
    for _ in range(400):
        t=list(s0)
        for p in rng.choice(sorted(cdr), size=rng.integers(1,4), replace=False):
            t[p]=AAS[rng.integers(20)]
        s="".join(t)
        if s in existing:                      # gate 1: uniqueness
            continue
        if max_identity_to_top50(s) >= 0.95:   # gate 2: novelty vs top-50
            continue
        cands.append(s)
cands=list(dict.fromkeys(cands))
print(f"generated {len(cands):,} candidates passing uniqueness + top-50 novelty")
pred=model.predict(oh(cands))
arch_pred=model.predict(oh(df.sequence.values))
print(f"predicted rank_sum  ours: best {pred.min():,.0f} median {np.median(pred):,.0f}")
print(f"                 archive: best {arch_pred.min():,.0f} median {np.median(arch_pred):,.0f}")
print(f"  plausible (>0): {(pred>0).sum()}/{len(pred)}")
better=(pred < np.percentile(arch_pred,1)).sum()
print(f"  predicted better than archive's top 1%: {better} ({better/len(pred)*100:.1f}%)")
best=np.argsort(pred)[:20]
print(f"\n  top-20 shortlist identity to top-50: "
      f"{min(max_identity_to_top50(cands[i]) for i in best):.3f}"
      f"..{max(max_identity_to_top50(cands[i]) for i in best):.3f}  (must stay < 0.95)")
