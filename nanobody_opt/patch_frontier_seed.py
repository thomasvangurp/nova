#!/usr/bin/env python3
"""Fix the frontier selection: select >=6-edit parents from the FULL candidate
pool (not post-filter the score-selected basin 64)."""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

old = '''    # Frontier redirect: keep only parents >=6 edits from the fresh top-50,
    # so the producer mines the competitive frontier instead of the basin.
    import numpy as np
    _idx = json.load(open("/root/artifacts/top50_cdr3_index.json"))
    _top = np.array([[ord(c) for c in seq] for seq in _idx.get("sequences", [])
                     if len(seq) == 127], dtype=np.uint8)
    if _top.size:
        def _dmin(sequence):
            a = np.frombuffer(sequence.encode(), dtype=np.uint8)
            return int((_top != a).sum(axis=1).min())
        _before = len(selected)
        selected = [p for p in selected if _dmin(p["sequence"]) >= 6]
        print(f"frontier redirect: {_before} -> {len(selected)} parents (dmin>=6)",
              flush=True)'''

new = '''    # Frontier selection: build the FULL candidate pool, keep the >=6-edit
    # parents (the competitive frontier), score-rank them.  The score selection
    # above picks the dedup-locked basin, so select from the full pool instead.
    import numpy as np
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
                _pool.setdefault(_cand["sequence"], _cand)
        _frontier = sorted(
            (c for c in _pool.values() if _dmin(c["sequence"]) >= 6),
            key=lambda c: (c["validator_final_score"], c["sequence"]),
        )
        selected = _frontier[: args.limit]
        for _cand in selected:
            _cand.setdefault("parent_source", "public_leaderboard_api")
            _cand["parent_selection_lane"] = "frontier"
            _cand.setdefault("nearest_elite_distance", 0)
            _cand.setdefault("public_observations", [{
                "epoch": 0, "epoch_rank": 0,
                "validator_final_score": _cand["validator_final_score"],
            }])
        print(f"frontier selection: pool={len(_pool)} frontier={len(_frontier)} "
              f"selected={len(selected)}", flush=True)'''

if old not in s:
    print("MISS: old frontier redirect not found")
else:
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(P, "w").write(s)
    print("FIXED frontier selection")
