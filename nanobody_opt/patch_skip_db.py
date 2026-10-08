#!/usr/bin/env python3
"""Skip the DB read (_database_payloads) in the frontier path — the frontier
only needs the API's final_score, and the DB's replay scores corrupt the
top-50 sorting + edit-distance filter."""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

old = """    payloads: dict[int, Any] = (
        _database_payloads(
            args.observation_database,
            target=args.target,
            before_epoch=args.before_epoch,
        )
        if args.observation_database is not None
        and args.observation_database.is_file()
        else {}
    )"""
new = """    payloads: dict[int, Any] = {}  # DB read skipped: the frontier uses the API only"""

if old not in s:
    print("MISS: _database_payloads block not found")
else:
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(P, "w").write(s)
    print("PATCHED: DB read skipped")
