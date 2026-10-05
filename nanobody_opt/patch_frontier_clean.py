#!/usr/bin/env python3
"""One-shot clean frontier path for select_elite_parents.py (SN68 nanobody).

Reconstructs the PROVEN frontier seed selector (won 25458 + 25459) in a single
deterministic patch instead of the prior chain of order-dependent patches:

1. Skip the DB read (_database_payloads) -- the frontier only needs the API's
   validator_final_score, and the DB's replay scores corrupt the ranking.
2. Replace the (already-skipped) score-selection with the frontier selection:
   full API pool, validator_final_score >= 100 (the competitor's mid/low band,
   NOT the dedup-locked top basin), score-ranked, capped at --limit.

The `dmin>=6` edit-distance filter is deliberately NOT used: it proved too far
out (0 parents passed the producer's layer2/nativeness gate).
"""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

# 1. Skip the DB read.
db_block = """    payloads: dict[int, Any] = (
        _database_payloads(
            args.observation_database,
            target=args.target,
            before_epoch=args.before_epoch,
        )
        if args.observation_database is not None
        and args.observation_database.is_file()
        else {}
    )"""
if db_block not in s:
    print("MISS: _database_payloads block not found")
    raise SystemExit(1)
s = s.replace(
    db_block,
    "    payloads: dict[int, Any] = {}  # DB read skipped: frontier uses the API only",
    1,
)

# 2. Replace the skipped score-selection with the frontier selection.
skip_marker = "    selected = []  # score selection skipped; frontier selection below\n"
if skip_marker not in s:
    print("MISS: score-selection skip marker not found")
    raise SystemExit(1)
frontier = """    # Frontier selection: the competitive frontier (validator_final_score >= 100)
    # instead of the slow dedup-locked basin. The score selection above is
    # skipped (it read the DB + basin allocation, ~3 min); select from the full
    # API pool instead.
    _pool = {}
    for _epoch in complete_epochs:
        for _cand in _epoch_candidates(payloads[_epoch], target=args.target):
            _pool.setdefault(_cand["sequence"], _cand)
    _frontier = sorted(
        (c for c in _pool.values() if c["validator_final_score"] >= 100),
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
          f"selected={len(selected)}", flush=True)
"""
s = s.replace(skip_marker, frontier, 1)

ast.parse(s)
open(P, "w").write(s)
print("PATCHED frontier (score>=100, API-only, DB read skipped)")
