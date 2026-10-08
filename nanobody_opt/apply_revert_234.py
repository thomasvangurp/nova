#!/usr/bin/env python3
"""Apply the basin-convergence revert #2-4 (three one-line edits)."""
import ast

edits = {
    "/root/nanobody_opt/local_neighborhood.py": [
        ("follow_field: bool = True", "follow_field: bool = False"),
        ("EXPLORATION_SEED_SHARE = 0.35", "EXPLORATION_SEED_SHARE = 0.5"),
    ],
    "/root/nanobody_opt/run_live_producer.py": [
        ('"--basin-fraction", type=float, default=0.70',
         '"--basin-fraction", type=float, default=0.35'),
    ],
}

for path, pairs in edits.items():
    s = open(path).read()
    for old, new in pairs:
        if old not in s:
            print(f"MISS in {path}: {old!r}")
            continue
        s = s.replace(old, new, 1)
        print(f"OK {path}: {old!r} -> {new!r}")
    ast.parse(s)  # syntax check
    open(path, "w").write(s)
    print(f"  {path}: written, syntax OK")
