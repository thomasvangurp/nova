#!/usr/bin/env python3
"""Switch the frontier from edit-distance (dmin>=6, too far -> fails layer2) to
the competitor's actual signal: the low-scoring tail (final_score >= 100)."""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

old = """        _frontier = sorted(
            (c for c in _pool.values() if _dmin(c["sequence"]) >= 6),
            key=lambda c: (c["validator_final_score"], c["sequence"]),
        )"""
new = """        _frontier = sorted(
            (c for c in _pool.values() if c["validator_final_score"] >= 100),
            key=lambda c: (c["validator_final_score"], c["sequence"]),
        )"""

if old not in s:
    print("MISS: dmin filter not found")
else:
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(P, "w").write(s)
    print("PATCHED: score-based frontier (final_score >= 100)")
