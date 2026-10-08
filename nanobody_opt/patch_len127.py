#!/usr/bin/env python3
"""Filter the frontier pool to 127-residue sequences (some API rows are not 127,
which breaks the numpy top-50 matrix)."""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

old_pool = """        for _cand in _epoch_candidates(payloads[_epoch], target=args.target):
            _pool.setdefault(_cand["sequence"], _cand)"""
new_pool = """        for _cand in _epoch_candidates(payloads[_epoch], target=args.target):
            if len(_cand["sequence"]) == 127:
                _pool.setdefault(_cand["sequence"], _cand)"""

old_top = """    _top = np.array([[ord(c) for c in c["sequence"]] for c in _all[:50]],
                    dtype=np.uint8)"""
new_top = """    _top = np.array([[ord(c) for c in c["sequence"]] for c in _all[:50]
                     if len(c["sequence"]) == 127], dtype=np.uint8)"""

changed = 0
for old, new in ((old_pool, new_pool), (old_top, new_top)):
    if old in s:
        s = s.replace(old, new, 1)
        changed += 1
    else:
        print(f"MISS: {old[:40]!r}")

if changed:
    ast.parse(s)
    open(P, "w").write(s)
    print(f"PATCHED length-127 filter ({changed} edits)")
