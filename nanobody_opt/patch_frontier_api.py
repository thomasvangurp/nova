#!/usr/bin/env python3
"""Frontier selection from the API data already fetched: compute the top-50
from the payloads' final_score (not a stale local index), then keep >=6-edit
parents."""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

old = '''    import numpy as np
    _idx = json.load(open("/root/artifacts/top50_cdr3_index.json"))
    _top = np.array([[ord(c) for c in seq] for seq in _idx.get("sequences", [])
                     if len(seq) == 127], dtype=np.uint8)
    if _top.size:
        def _dmin(sequence):
            a = np.frombuffer(sequence.encode(), dtype=np.uint8)
            return int((_top != a).sum(axis=1).min())
        _pool = {}
        for _epoch in complete_epochs:
            for _cand in _epoch_candidates(payloads[_epoch], target=args.target):
                _pool.setdefault(_cand["sequence"], _cand)'''

new = '''    import numpy as np
    _pool = {}
    for _epoch in complete_epochs:
        for _cand in _epoch_candidates(payloads[_epoch], target=args.target):
            _pool.setdefault(_cand["sequence"], _cand)
    _all = sorted(_pool.values(),
                  key=lambda c: (c["validator_final_score"], c["sequence"]))
    _top = np.array([[ord(c) for c in c["sequence"]] for c in _all[:50]],
                    dtype=np.uint8)
    if _top.size:
        def _dmin(sequence):
            a = np.frombuffer(sequence.encode(), dtype=np.uint8)
            return int((_top != a).sum(axis=1).min())'''

if old not in s:
    print("MISS: stale-index block not found")
else:
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(P, "w").write(s)
    print("PATCHED frontier-from-API")
