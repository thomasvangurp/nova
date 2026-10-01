import sqlite3, json, sys
from collections import defaultdict
import numpy as np

DB = '/Users/tvgdeena/nova-nanobody/data/nanobody_dash.sqlite'
TARGET_COLD = '5GhREcsKiqXyafTNd7HsVDm3y22bp5HMnZYMjtLccQWmnLpK'

c = sqlite3.connect(DB)
cur = c.cursor()
cur.execute("SELECT epoch, rank, coldkey, sequence FROM entries ORDER BY epoch ASC, rank ASC")
rows = cur.fetchall()

# map sequence string -> char array index
seqs = [r[3] for r in rows]
L = len(seqs[0])
# numpy matrix of sequences as byte arrays
arr = np.frombuffer(''.join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)

# build public universe: all sequences seen strictly before each epoch
# process target miner's submissions
target = [ (i, r[0], r[1], r[2]) for i, r in enumerate(rows) if r[2] == TARGET_COLD ]
print(f"target submissions: {len(target)}")

# precompute: for each target submission, compare to all prior sequences
# We'll iterate epochs and maintain an index of prior public sequences
epochs_sorted = sorted(set(r[0] for r in rows))

# For speed, do it per target epoch with a vectorized Hamming over prior slice
results = []
for idx, ep, rank, ck in target:
    prior_mask = np.array([rows[i][0] < ep for i in range(len(rows))])
    prior_idx = np.where(prior_mask)[0]
    if len(prior_idx) == 0:
        results.append(dict(epoch=ep, rank=rank, seq=seqs[idx], dmin=None, parent_idx=None))
        continue
    t = arr[idx]
    sub = arr[prior_idx]
    dist = (sub != t).sum(axis=1)  # Hamming distance to each prior public seq
    dmin = int(dist.min())
    p_i = int(prior_idx[int(dist.argmin())])
    results.append(dict(epoch=ep, rank=rank, seq=seqs[idx], dmin=dmin, parent_idx=p_i))

# Characterize
same_self = 0
other = 0
d0 = d1 = d2 = 0
parent_ranks = []
parent_epoch_gaps = []
edit_positions = defaultdict(int)
positions_dist = defaultdict(int)

for r in results:
    if r['dmin'] is None: continue
    d = r['dmin']
    if d == 0: d0 += 1
    elif d == 1: d1 += 1
    elif d == 2: d2 += 1
    if d <= 1 and r['parent_idx'] is not None:
        pi = r['parent_idx']
        p_ck = rows[pi][2]
        p_ep = rows[pi][0]
        p_rank = rows[pi][1]
        if p_ck == TARGET_COLD: same_self += 1
        else: other += 1
        parent_ranks.append(p_rank)
        parent_epoch_gaps.append(r['epoch'] - p_ep)
        if d == 1:
            # find edited position
            ts = r['seq']; ps = rows[pi][3]
            pos = [k for k in range(L) if ts[k] != ps[k]]
            for p in pos: edit_positions[p] += 1

print(f"\n=== 5GhREcsK parent reconstruction ({len(results)} submissions) ===")
print(f"dmin=0 (exact resubmit): {d0}")
print(f"dmin=1 (point mutation):  {d1}")
print(f"dmin=2:                    {d2}")
print(f"dmin>2 or none:            {len(results)-d0-d1-d2}")
print(f"\nAmong d<=1 with a parent:")
print(f"  parent is SELF: {same_self}")
print(f"  parent is OTHER: {other}  ({100*other/max(1,same_self+other):.1f}%)")

import statistics
if parent_ranks:
    parent_ranks_sorted = sorted(parent_ranks)
    print(f"\nparent rank distribution (n={len(parent_ranks)}):")
    print(f"  median={statistics.median(parent_ranks)}  mean={statistics.mean(parent_ranks):.1f}  p25={parent_ranks_sorted[len(parent_ranks_sorted)//4]}  p75={parent_ranks_sorted[3*len(parent_ranks_sorted)//4]}")
    print(f"  top-3: {sum(1 for x in parent_ranks if x<=3)}  top-10: {sum(1 for x in parent_ranks if x<=10)}  >10: {sum(1 for x in parent_ranks if x>10)}")
if parent_epoch_gaps:
    print(f"\nparent epoch gap (how many epochs back the parent was public):")
    print(f"  median={statistics.median(parent_epoch_gaps)}  mean={statistics.mean(parent_epoch_gaps):.1f}  max={max(parent_epoch_gaps)}")
    print(f"  gap<=1: {sum(1 for x in parent_epoch_gaps if x<=1)}  gap<=3: {sum(1 for x in parent_epoch_gaps if x<=3)}  gap<=10: {sum(1 for x in parent_epoch_gaps if x<=10)}")

print(f"\nedit position distribution (for d=1 parents):")
for p in sorted(edit_positions, key=lambda x:-edit_positions[x])[:15]:
    print(f"  pos {p}: {edit_positions[p]}")

# save full results
with open('/Users/tvgdeena/nova-nanobody/analysis/nanobody-gap/parent-selection-20260929/reconstruct_results.json','w') as f:
    json.dump(dict(
        target=TARGET_COLD,
        n=len(results),
        d0=d0, d1=d1, d2=d2,
        parent_self=same_self, parent_other=other,
        parent_ranks=parent_ranks,
        parent_epoch_gaps=parent_epoch_gaps,
        edit_positions={str(k):v for k,v in edit_positions.items()},
    ), f, indent=1)
print("\nsaved reconstruct_results.json")
