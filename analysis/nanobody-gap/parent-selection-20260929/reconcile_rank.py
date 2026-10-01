import sqlite3, numpy as np, statistics
from collections import Counter
DB = '/Users/tvgdeena/nova-nanobody/data/nanobody_dash.sqlite'
TARGET = '5GhREcsKiqXyafTNd7HsVDm3y22bp5HMnZYMjtLccQWmnLpK'
c = sqlite3.connect(DB); cur = c.cursor()
cur.execute("SELECT epoch, rank, coldkey, sequence FROM entries ORDER BY epoch ASC, rank ASC")
rows = cur.fetchall()
seqs = [r[3] for r in rows]; L = len(seqs[0])
arr = np.frombuffer(''.join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)
epochs = np.array([r[0] for r in rows])
ranks = np.array([r[1] for r in rows])
coldkeys = np.array([r[2] for r in rows])

tgt = [(i,r[0],r[1]) for i,r in enumerate(rows) if r[2]==TARGET]
print(f"target submissions: {len(tgt)}")

# For each target, ALL distance-1 prior neighbors (full 351-epoch history)
all_nb_ranks = []
all_nb_gaps = []
all_nb_other = 0; all_nb_self = 0
most_recent_ranks = []
for idx, ep, rank in tgt:
    prior = np.where(epochs < ep)[0]
    d = (arr[prior] != arr[idx]).sum(axis=1)
    d1 = prior[d==1]
    if len(d1)==0: continue
    all_nb_ranks.extend(ranks[d1].tolist())
    all_nb_gaps.extend((ep - epochs[d1]).tolist())
    for j in d1:
        if coldkeys[j]==TARGET: all_nb_self+=1
        else: all_nb_other+=1
    best = d1[np.argmax(epochs[d1])]
    most_recent_ranks.append(int(ranks[best]))

print(f"\n=== ALL distance-1 neighbors (full history), n={len(all_nb_ranks)} ===")
print(f"rank: median={statistics.median(all_nb_ranks)}  mean={statistics.mean(all_nb_ranks):.1f}")
for lo,hi in [(1,2),(3,5),(6,10),(11,20),(21,30),(31,53)]:
    n=sum(1 for x in all_nb_ranks if lo<=x<=hi)
    print(f"  rank {lo:>2}-{hi:<2}: {n:4d} ({100*n/len(all_nb_ranks):.0f}%)")
print(f"neighbor coldkey: other={all_nb_other} self={all_nb_self} ({100*all_nb_other/(all_nb_other+all_nb_self):.0f}% other)")
print(f"neighbor epoch gap: median={statistics.median(all_nb_gaps)} mean={statistics.mean(all_nb_gaps):.1f}")
print(f"\n=== most-recent distance-1 neighbor rank, n={len(most_recent_ranks)} ===")
print(f"median={statistics.median(most_recent_ranks)} mean={statistics.mean(most_recent_ranks):.1f}")
for lo,hi in [(1,2),(3,10),(11,20),(21,53)]:
    n=sum(1 for x in most_recent_ranks if lo<=x<=hi)
    print(f"  rank {lo:>2}-{hi:<2}: {n:4d} ({100*n/len(most_recent_ranks):.0f}%)")
