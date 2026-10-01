import json, collections, statistics, math
from pathlib import Path
OUT = Path("/private/tmp/claude-502/-Users-tvgdeena/6a0daf20-2ab0-4cb4-bf0b-7bcb06048b44/scratchpad/epochs")
rows = []
for path in sorted(OUT.glob("[0-9]*.json")):
    data = json.loads(path.read_text()).get("data") or []
    rc, scored = collections.Counter(), []
    for sub in data:
        for m in (sub.get("molecules") or []):
            n = m.get("name") or ""
            if n.startswith("rxn:"): rc[n.split(":")[1]] += 1
        s = sub.get("final_score")
        if s is not None and s > -900: scored.append(float(s))
    if scored and rc:
        rows.append((int(path.stem), rc.most_common(1)[0][0], sorted(scored, reverse=True), len(data)))

print("=== every rxn 3 epoch, newest first ===")
print(f"{'epoch':>7} {'scored':>7} {'winner':>8} {'2nd':>8} {'median':>8}")
r3 = [r for r in rows if r[1] == '3']
for epoch, _, s, listed in sorted(r3, reverse=True)[:20]:
    print(f"{epoch:>7} {len(s):>7} {s[0]:>8.4f} {s[1] if len(s)>1 else 0:>8.4f} "
          f"{statistics.median(s):>8.4f}")

A, B = 0.8344, 0.0818
ours_naive = B*math.log(19799) + A
print(f"\nour bank 19,799 -> naive best20 {ours_naive:.4f}, curse-corrected {ours_naive-0.056:.4f}")

for label, subset in (("all 39 rxn3 epochs", r3),
                      ("last 15 rxn3 epochs", sorted(r3, reverse=True)[:15]),
                      ("last 10 rxn3 epochs", sorted(r3, reverse=True)[:10])):
    wins = [s[0] for _, _, s, _ in subset]
    beat = sum(1 for w in wins if ours_naive - 0.056 > w)
    print(f"  {label:<22} winner med {statistics.median(wins):.4f}  "
          f"we beat {beat}/{len(wins)} = {beat/len(wins):.0%}")

print("\nbank size to beat a given rxn3 winner score (curse-corrected):")
wins3 = sorted(s[0] for _, _, s, _ in sorted(r3, reverse=True)[:15])
for pct, idx in (("median", len(wins3)//2), ("p75", int(len(wins3)*0.75)), ("max", -1)):
    t = wins3[idx]
    print(f"  {pct:<7} {t:.4f}  ->  {math.exp((t + 0.056 - A)/B):>12,.0f} molecules")
