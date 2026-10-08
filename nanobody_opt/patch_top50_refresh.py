#!/usr/bin/env python3
"""Add a fresh top-50 rebuild to select_elite_parents.py before the frontier
filter. The rebuild uses the FULL HF archive, not the 15-epoch API window."""
import ast

P = "/root/nanobody_opt/select_elite_parents.py"
s = open(P).read()

old = "    import numpy as np\n    _idx = json.load(open(\"/root/artifacts/top50_cdr3_index.json\"))"
new = """    import sys, subprocess
    subprocess.run(
        [sys.executable, "/root/nanobody_opt/fetch_hf_archive.py",
         "/root/P20809_nanobodies.csv"],
        check=False,
    )
    subprocess.run(
        [sys.executable, "/root/nanobody_opt/build_top50_index.py",
         "/root/P20809_nanobodies.csv", "/root/artifacts/top50_cdr3_index.json"],
        check=False,
    )
    import numpy as np
    _idx = json.load(open("/root/artifacts/top50_cdr3_index.json"))"""

if old not in s:
    print("MISS: import numpy anchor not found")
else:
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(P, "w").write(s)
    print("PATCHED top-50 refresh")
