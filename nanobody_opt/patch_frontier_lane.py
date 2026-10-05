#!/usr/bin/env python3
"""Teach local_neighborhood's core-seed selection to recognize the frontier lane.

select_elite_parents now emits parents tagged parent_selection_lane="frontier"
(score>=100). local_neighborhood.py only seeded core_seeds from "epoch_elite" +
reserved, so the frontier parents were silently dropped and every children-pool
prescreen failed with "only 0 elite seeds passed Layer-2 and diversity".
"""
import ast

P = "/root/nanobody_opt/local_neighborhood.py"
s = open(P).read()

old = """    legacy_seeds = [
        item for item in eligible
        if item[5] == "epoch_elite" and item[1] not in reserved_sequences
    ]"""
new = """    legacy_seeds = [
        item for item in eligible
        if item[5] in ("epoch_elite", "frontier") and item[1] not in reserved_sequences
    ]"""

if old not in s:
    print("MISS: legacy_seeds epoch_elite filter not found")
    raise SystemExit(1)
s = s.replace(old, new, 1)
ast.parse(s)
open(P, "w").write(s)
print("PATCHED: legacy_seeds recognizes (epoch_elite, frontier)")
