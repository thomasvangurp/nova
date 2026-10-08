#!/usr/bin/env python3
"""Skip the slow score-selection (select_elite_parents) — the frontier
selection below already replaces its output, so the score selection is wasted
(reads the 120K-row DB + basin allocation, ~3 min)."""
import re
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

# match the multi-line call: selected = select_elite_parents( ... )
pat = re.compile(r"    selected = select_elite_parents\(.*?\n    \)\n", re.DOTALL)
m = pat.search(s)
if not m:
    print("MISS: select_elite_parents call not found")
else:
    s = s[: m.start()] + "    selected = []  # score selection skipped; frontier selection below\n" + s[m.end():]
    ast.parse(s)
    open(P, "w").write(s)
    print("PATCHED: score selection skipped")
