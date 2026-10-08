import json

PREF = {116: "A", 123: "V"}


def order_key(item):
    p = float(item.get("win_probability") or 0.0)
    return (0 if p > 0.0 else 1, int(item.get("settled_deviations", 0)), -p)


def violations(item):
    seq = str(item.get("sequence") or "")
    if len(seq) != 127:
        return 0
    return sum(1 for pos, pref in PREF.items() if seq[pos - 1] != pref)


def label(item):
    seq = item["sequence"]
    r116 = seq[115] if len(seq) >= 116 else "?"
    r123 = seq[122] if len(seq) >= 123 else "?"
    return f"{r116}@{116}/{r123}@{123}"


d = json.load(open("/root/live/epoch25574/qualified_rankings.json"))
cands = d["candidates"]

# markers
t116A = "KWVLVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQAPGKEREWVSLISSSGSTWLDYPDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCGVFLGAHPCPLEGEHDAFGAGATVTVGQ"

# before fix: reproduce consensus_policy.ordered_candidates(..., "tier")
before = sorted(cands, key=order_key)
# after fix: one stable sort by violations on top of the existing order
after = sorted(before, key=violations)


def rank_of(seq, ordered):
    return next(i for i, x in enumerate(ordered) if x["sequence"] == seq)


print("total qualified:", len(cands))
print("=== BEFORE fix (win-probability tier order) ===")
for i, x in enumerate(before[:12]):
    print(f"  rank {i}: {label(x)}  win={x['win_probability']:.4f} dev={x.get('settled_deviations')} sel={x['selection_score']}")
print(f"  ... 116A single-edit rank = {rank_of(t116A, before)}  ({label(before[rank_of(t116A, before)])}, win={before[rank_of(t116A, before)]['win_probability']:.2e})")

print()
print("=== AFTER fix (residue-preference primary) ===")
for i, x in enumerate(after[:12]):
    print(f"  rank {i}: {label(x)}  viol={violations(x)} win={x['win_probability']:.4f} dev={x.get('settled_deviations')} sel={x['selection_score']}")
print(f"  ... 116A single-edit rank = {rank_of(t116A, after)}  ({label(after[rank_of(t116A, after)])}, viol={violations(after[rank_of(t116A, after)])})")

# how many 0-violation candidates (A@116 + V@123) are now ahead of the trap?
zero_v = [x for x in after if violations(x) == 0]
one_v = [x for x in after if violations(x) == 1]
two_v = [x for x in after if violations(x) == 2]
print()
print(f"violation groups: 0={len(zero_v)} 1={len(one_v)} 2={len(two_v)}")
# among 0-violation, what are the top candidates?
print("top of 0-violation group:")
for x in zero_v[:8]:
    print(f"   {label(x)} win={x['win_probability']:.4f} sel={x['selection_score']} {x['sequence'][:16]}")
