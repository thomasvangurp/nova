import json
import sys
import time

sys.path.insert(0, "/root")
from nanobody_opt import rank_exact as R

d = json.load(open("/root/live/epoch25574/qualified_rankings.json"))
cands = d["candidates"]
t116A = "KWVLVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQAPGKEREWVSLISSSGSTWLDYPDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCGVFLGAHPCPLEGEHDAFGAGATVTVGQ"


def label(x):
    s = x["sequence"]
    return f"{s[115]}@116/{s[122]}@123"


t0 = time.time()
# real production path: win-probability + tier consensus, then residue sort inside.
ordered = R._order_candidates(
    cands, "win-probability", 60.0, "tier",
    rank_workers=8, rank_workers_minimum_candidates=64,
)
print(f"_order_candidates done in {time.time()-t0:.1f}s  n={len(ordered)}")
print("top 10 after fix:")
for i, x in enumerate(ordered[:10]):
    print(f"  rank {i}: {label(x)}  win={x['win_probability']:.4f} sel={x['selection_score']} viol={R._residue_violations(x)}")
rk = next(i for i, x in enumerate(ordered) if x["sequence"] == t116A)
print(f"116A single-edit rank = {rk}  win={ordered[rk]['win_probability']:.2e} viol={R._residue_violations(ordered[rk])}")
# also count how many trap (S@116/A@123) are still in top 20
trap_top = sum(1 for x in ordered[:20] if x["sequence"][115] == "S" and x["sequence"][122] == "A")
print(f"trap (S@116/A@123) in top 20: {trap_top}")
