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
        rows.append((int(path.stem), rc.most_common(1)[0][0], max(scored), len(scored)))

A, B, CURSE = 0.8344, 0.0818, 0.056
RATE, DPH, CUR = 20.5, 6.039, 19799       # molecules/min, $/hr combined, current bank

recent = [r for r in rows if r[0] >= 25176]
print(f"regime break at epoch 25176: participation "
      f"{statistics.median(r[3] for r in rows if r[0] < 25176):.0f} -> "
      f"{statistics.median(r[3] for r in recent):.0f} scored UIDs per epoch\n")

w3 = [r[2] for r in recent if r[1] == '3']
print(f"rxn 3 winners since 25176 (n={len(w3)}): "
      f"{', '.join(f'{x:.3f}' for x in sorted(w3, reverse=True))}\n")

print(f"{'bank':>9} {'best20':>8} {'realised':>9} {'beats':>7} {'extra h':>8} {'extra $':>9}")
for n in (19799, 25000, 40000, 60000, 100000, 150000, 215000):
    naive = B*math.log(n) + A
    real = naive - CURSE
    beat = sum(1 for w in w3 if real > w)
    hours = max(0, (n - CUR)) / RATE / 60
    print(f"{n:>9,} {naive:>8.4f} {real:>9.4f} {beat:>4}/{len(w3):<2} "
          f"{hours:>8.1f} {hours*DPH:>9,.0f}")

print("\nper-reaction winner medians since 25176 (score SCALES differ, not difficulty):")
for k in sorted({r[1] for r in recent}):
    w = [r[2] for r in recent if r[1] == k]
    f = [r[3] for r in recent if r[1] == k]
    print(f"  rxn {k}: n={len(w):>2}  winner med {statistics.median(w):.4f}  "
          f"median field size {statistics.median(f):.0f}")
