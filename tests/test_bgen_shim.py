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


def _load_shim_utils(monkeypatch, responses):
    """Import the shim's utils with `requests.get` replaced by a scripted fake."""
    import importlib.util
    import requests

    spec = importlib.util.spec_from_file_location(
        "bgen_shim_utils_under_test", SHIM / "utils" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class _Resp:
        def __init__(self, status, text=""):
            self.status_code, self.text = status, text

    def fake_get(url, timeout=None):
        calls.append(url)
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return _Resp(*item)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    return module, calls


def test_transient_uniprot_failures_are_retried_not_lost(monkeypatch):
    """RETRY_SHORT_SHARDS root cause: one SSL EOF used to lose the sequence."""
    import requests
    module, calls = _load_shim_utils(monkeypatch, [
        requests.exceptions.SSLError("UNEXPECTED_EOF_WHILE_READING"),
        (503,),
        (200, ">sp|X\nMKT\nLLA\n"),
    ])
    assert module.get_sequence_from_protein_code("X00000") == "MKTLLA"
    assert len(calls) == 3


def test_a_good_answer_is_cached_so_replicate_chunks_do_not_refetch(monkeypatch):
    module, calls = _load_shim_utils(monkeypatch, [(200, ">sp|X\nMKTLLA\n")])
    for _ in range(5):
        assert module.get_sequence_from_protein_code("X00000", (1, 4)) == "KTL"
    assert len(calls) == 1


def test_a_definitive_answer_still_fails_closed_without_retrying(monkeypatch):
    import pytest
    module, calls = _load_shim_utils(monkeypatch, [(404,)])
    with pytest.raises(RuntimeError, match="Refusing to guess"):
        module.get_sequence_from_protein_code("NOPE")
    assert len(calls) == 1


def test_persistent_outage_still_raises_after_bounded_attempts(monkeypatch):
    import pytest
    import requests
    module, calls = _load_shim_utils(
        monkeypatch, [requests.exceptions.ConnectionError("dns")])
    with pytest.raises(RuntimeError, match="unreachable"):
        module.get_sequence_from_protein_code("X00000")
    assert len(calls) == module._FETCH_ATTEMPTS
