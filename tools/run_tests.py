#!/usr/bin/env python3
"""Run each test module in its own process.

tools/test_rowcolumn_lane.py installs a stub compound_miner.adaptive_strategy in
sys.modules so it can run without rdkit. In a shared process that stub leaks:
it broke joblib pickling in test_bank_proxy and the imports in the older
nim19 test modules, depending purely on command-line order. Process isolation
removes the whole class of problem instead of sequencing around it.

Modules needing rdkit run under the analysis pixi environment when present, and
are skipped with a clear note otherwise.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIXI = ROOT / "analysis/.pixi/envs/default/bin/python"

PLAIN = ["test_compound_adaptive_wiring", "test_nim19_schedule_audit_repair",
         "test_nim19_leader_extension", "test_nim19_schedule_comparison",
         "test_nim19_offline_playbook", "test_nim19_native_runtime",
         "test_nim19_benchmark", "test_rowcolumn_lane"]
CHEMISTRY = ["test_bank_proxy", "test_residual_base_contract"]


def run(module, interpreter):
    # Some older test modules do a bare `import nim19_native_runtime`, which only
    # worked because a sibling test module had already put tools/ on sys.path.
    # In isolation that assumption fails, so make the path explicit.
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "tools"),
                                         env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run([str(interpreter), "-m", "unittest",
                             "tools/%s.py" % module, "-v"],
                            cwd=ROOT, capture_output=True, text=True, env=env)
    tail = [l for l in result.stderr.splitlines() if l.startswith("Ran ")]
    count = int(tail[0].split()[1]) if tail else 0
    return result.returncode == 0, count, result.stderr.strip().splitlines()[-1:]


def main() -> int:
    total = failed = 0
    for module in PLAIN:
        ok, n, tail = run(module, sys.executable)
        total += n
        print("  %-42s %-4s %d tests" % (module, "OK" if ok else "FAIL", n))
        if not ok:
            failed += 1
            print("      ", *tail)
    if PIXI.is_file():
        for module in CHEMISTRY:
            ok, n, tail = run(module, PIXI)
            total += n
            print("  %-42s %-4s %d tests" % (module, "OK" if ok else "FAIL", n))
            if not ok:
                failed += 1
                print("      ", *tail)
    else:
        print("  (analysis pixi environment absent; skipped %s)" % ", ".join(CHEMISTRY))
    print("\n%d tests across %d modules, %d module(s) failing"
          % (total, len(PLAIN) + (len(CHEMISTRY) if PIXI.is_file() else 0), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
