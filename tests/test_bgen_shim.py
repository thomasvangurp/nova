"""The scorer's minimal `utils` must win over the real package of that name.

boltzgen_wrapper does `from utils import get_sequence_from_protein_code,
seq_hash`. nova's real utils/__init__ eagerly imports molecules (rdkit),
files (bittensor), nanobodies (the whole metanano/TNP/IgBLAST stack) and btdr
(timelock) -- none of it reachable from those two functions, and none of it
present in the scorer's environment.

Consolidating the deployment onto a single runtime root put the real utils at
/root, and `python -m` places the working directory at sys.path[0] ahead of every
PYTHONPATH entry. Every scorer shard then died three imports deep on
`ModuleNotFoundError: No module named 'metanano'`, having written nothing, and
the producer could only report "shard 0 exited 1 without writing out0.json".
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIM = REPO / "ansible" / "files" / "root" / "bgen_shim"


def test_the_shim_supplies_exactly_the_two_symbols():
    source = (SHIM / "utils" / "__init__.py").read_text()
    assert "get_sequence_from_protein_code" in source
    assert "seq_hash" in source


def test_the_shim_wins_against_a_shadowing_utils(tmp_path):
    """The decisive test: a real `utils` in the working directory that raises,
    exactly as the deployed one does in this environment."""
    root = tmp_path / "root"
    (root / "utils").mkdir(parents=True)
    (root / "bgen_shim" / "utils").mkdir(parents=True)
    (root / "utils" / "__init__.py").write_text(
        'raise ImportError("the real utils drags in metanano")\n'
    )
    (root / "bgen_shim" / "utils" / "__init__.py").write_text(
        'def get_sequence_from_protein_code(*a): return "SEQ"\n'
        'def seq_hash(*a): return "HASH"\n'
    )
    (root / "bgen_shim" / "sitecustomize.py").write_text(
        (SHIM / "sitecustomize.py").read_text()
    )

    done = subprocess.run(
        [sys.executable, "-c",
         "from utils import get_sequence_from_protein_code, seq_hash;"
         "print(get_sequence_from_protein_code(), seq_hash())"],
        cwd=root,
        env={"PYTHONPATH": str(root / "bgen_shim"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "SEQ HASH" in done.stdout


def test_reordering_sys_path_would_not_have_worked():
    """Pins the reasoning, because the obvious fix is wrong.

    For `python -m` and `-c` the working directory is re-inserted at sys.path[0]
    AFTER site initialisation, so anything sitecustomize does to the path is
    undone. Registering in sys.modules is what survives.
    """
    source = (SHIM / "sitecustomize.py").read_text()
    assert 'sys.modules["utils"] = _module' in source
    assert "sys.path.insert(0, _shim)" not in source


def test_a_broken_shim_never_breaks_the_interpreter(tmp_path):
    """The seeding hook this file also carries must survive a bad shim utils."""
    root = tmp_path / "root"
    (root / "bgen_shim" / "utils").mkdir(parents=True)
    (root / "bgen_shim" / "utils" / "__init__.py").write_text("raise RuntimeError('bad shim')\n")
    (root / "bgen_shim" / "sitecustomize.py").write_text(
        (SHIM / "sitecustomize.py").read_text()
    )
    done = subprocess.run(
        [sys.executable, "-c", "print('interpreter usable')"],
        cwd=root,
        env={"PYTHONPATH": str(root / "bgen_shim"), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "interpreter usable" in done.stdout
