import sqlite3, json
from collections import defaultdict
import numpy as np, statistics

DB = '/Users/tvgdeena/nova-nanobody/data/nanobody_dash.sqlite'
TARGET = '5GhREcsKiqXyafTNd7HsVDm3y22bp5HMnZYMjtLccQWmnLpK'
c = sqlite3.connect(DB); cur = c.cursor()
cur.execute("SELECT epoch, rank, coldkey, sequence FROM entries ORDER BY epoch ASC, rank ASC")
rows = cur.fetchall()
seqs = [r[3] for r in rows]; L=len(seqs[0])
arr = np.frombuffer(''.join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)

def parent_stats(rank_filter):
    tgt = [(i,r[0],r[1],r[2]) for i,r in enumerate(rows) if r[2]==TARGET and rank_filter(r[1])]
    gaps, ranks, other_n, self_n = [], [], 0, 0
    n_d1 = 0; n_any = 0
    for idx, ep, rank, ck in tgt:
        prior = np.where(np.array([rows[i][0] < ep for i in range(len(rows))]))[0]
        dist = (arr[prior] != arr[idx]).sum(axis=1)
        d1 = prior[dist==1]
        if len(d1)==0: continue
        n_d1 += 1
        best = max(d1, key=lambda i: rows[i][0])
        gaps.append(ep - rows[best][0]); ranks.append(rows[best][1])
        if rows[best][2]==TARGET: self_n+=1
        else: other_n+=1
    return tgt, gaps, ranks, other_n, self_n

for label, rf in [("ALL submissions", lambda r: True), ("TOP-10 (wins)", lambda r: r<=10), ("TOP-3", lambda r: r<=3), ("WINNER rank1", lambda r: r==1)]:
    tgt, gaps, ranks, o, s = parent_stats(rf)
    if not tgt:
        print(f"{label}: n={len(tgt)}"); continue
    has_parent = len(gaps)
    print(f"\n=== {label}: n={len(tgt)} submissions ===")
    print(f"  have a distance-1 prior parent: {has_parent} ({100*has_parent/len(tgt):.0f}%)")
    if gaps:
        print(f"  parent OTHER-miner share: {100*o/(o+s):.0f}%  (other={o}, self={s})")
        print(f"  parent epoch gap: median={statistics.median(gaps)}  mean={statistics.mean(gaps):.1f}")
        print(f"    <=3: {sum(1 for g in gaps if g<=3)}  <=10: {sum(1 for g in gaps if g<=10)}  <=30: {sum(1 for g in gaps if g<=30)}  <=100: {sum(1 for g in gaps if g<=100)}")
        print(f"  parent rank: median={statistics.median(ranks)}  mean={statistics.mean(ranks):.1f}")
