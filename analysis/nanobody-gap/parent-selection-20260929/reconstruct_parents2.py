import sqlite3, json
from collections import defaultdict
import numpy as np
import statistics

DB = '/Users/tvgdeena/nova-nanobody/data/nanobody_dash.sqlite'
TARGET_COLD = '5GhREcsKiqXyafTNd7HsVDm3y22bp5HMnZYMjtLccQWmnLpK'

c = sqlite3.connect(DB); cur = c.cursor()
cur.execute("SELECT epoch, rank, coldkey, sequence FROM entries ORDER BY epoch ASC, rank ASC")
rows = cur.fetchall()
seqs = [r[3] for r in rows]
L = len(seqs[0])
arr = np.frombuffer(''.join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)

target = [ (i, r[0], r[1], r[2]) for i, r in enumerate(rows) if r[2] == TARGET_COLD ]

# For each target, among all PRIOR public sequences, find:
#  - the most RECENT sequence at distance exactly 1 (causal parent candidate)
#  - the count of distinct distance-1 neighbors
#  - also distance 0 (exact prior)
rec_gaps, rec_ranks, rec_self = [], [], 0
rec_other = 0
n_d1_neighbors = []
exact_prior = 0
rec_d0_gaps = []

for idx, ep, rank, ck in target:
    prior_idx = np.where(np.array([rows[i][0] < ep for i in range(len(rows))]))[0]
    t = arr[idx]
    sub = arr[prior_idx]
    dist = (sub != t).sum(axis=1)
    d1 = prior_idx[dist == 1]
    d0 = prior_idx[dist == 0]
    if len(d0) > 0:
        exact_prior += 1
        # most recent exact
        r_ep = max(rows[i][0] for i in d0)
        rec_d0_gaps.append(ep - r_ep)
    n_d1_neighbors.append(len(d1))
    if len(d1) == 0:
        continue
    # most recent distance-1 neighbor
    best = max(d1, key=lambda i: rows[i][0])
    gap = ep - rows[best][0]
    p_rank = rows[best][1]
    p_ck = rows[best][2]
    rec_gaps.append(gap); rec_ranks.append(p_rank)
    if p_ck == TARGET_COLD: rec_self += 1
    else: rec_other += 1

print(f"target submissions: {len(target)}")
print(f"exact prior (d=0) present: {exact_prior}  (median rec gap {statistics.median(rec_d0_gaps) if rec_d0_gaps else 'na'})")
print(f"\ndistinct distance-1 prior neighbors per submission:")
print(f"  median={statistics.median(n_d1_neighbors)}  mean={statistics.mean(n_d1_neighbors):.1f}  min={min(n_d1_neighbors)}  max={max(n_d1_neighbors)}")

print(f"\nMOST RECENT distance-1 parent (causal candidate), n={len(rec_gaps)}:")
print(f"  epoch gap: median={statistics.median(rec_gaps)}  mean={statistics.mean(rec_gaps):.1f}  max={max(rec_gaps)}")
print(f"  gap<=1: {sum(1 for g in rec_gaps if g<=1)}  gap<=3: {sum(1 for g in rec_gaps if g<=3)}  gap<=10: {sum(1 for g in rec_gaps if g<=10)}  gap<=30: {sum(1 for g in rec_gaps if g<=30)}")
print(f"  parent rank: median={statistics.median(rec_ranks)}  mean={statistics.mean(rec_ranks):.1f}")
print(f"  parent SELF: {rec_self}  OTHER: {rec_other}  ({100*rec_other/max(1,rec_self+rec_other):.1f}% other)")

# histogram of gaps
hist = defaultdict(int)
for g in rec_gaps:
    if g<=1: hist['<=1']+=1
    elif g<=3: hist['2-3']+=1
    elif g<=10: hist['4-10']+=1
    elif g<=30: hist['11-30']+=1
    elif g<=100: hist['31-100']+=1
    else: hist['>100']+=1
print("gap histogram:", dict(hist))

json.dump(dict(n=len(target), rec_gaps=rec_gaps, rec_ranks=rec_ranks, rec_self=rec_self, rec_other=rec_other,
               n_d1_neighbors=n_d1_neighbors, exact_prior=exact_prior),
          open('/Users/tvgdeena/nova-nanobody/analysis/nanobody-gap/parent-selection-20260929/reconstruct2.json','w'), indent=1)
print("saved reconstruct2.json")
