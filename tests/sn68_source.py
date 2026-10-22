"""Offline test support: real shipping imports, plus pinned validator AST only."""
import ast
import importlib
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_REF = 'b53923caab62d9ad6990ba453a7e17d18b8444c0'


def import_shipping(name, monkeypatch):
    """Load the actual module; only the optional database driver is isolated.

    CI installs mysqlclient. On hosts without it, its import stub prevents
    connection attempts and does not replace any miner module/function.
    """
    if 'MySQLdb' not in sys.modules and importlib.util.find_spec('MySQLdb') is None:
        driver = ModuleType('MySQLdb')
        cursors = ModuleType('MySQLdb.cursors')
        def forbidden(*args, **kwargs):
            raise AssertionError('database access forbidden in CPU guardrails')
        driver.connect = forbidden
        driver.OperationalError = type('OperationalError', (Exception,), {})
        cursors.SSCursor = type('SSCursor', (), {})
        driver.cursors = cursors
        monkeypatch.setitem(sys.modules, 'MySQLdb', driver)
        monkeypatch.setitem(sys.modules, 'MySQLdb.cursors', cursors)
    module = importlib.import_module(name)
    expected = ROOT / (name.replace('.', '/') + '.py')
    assert Path(module.__file__).resolve() == expected.resolve(), f'wrong shipping module: {name}'
    return module


def pinned_validator_predicate():
    # This exact predicate is packaged with the tested release so the guard
    # runs on the installed tree without a git database or a network call.
    source = (ROOT / 'tests/fixtures/validator_is_duplicate.py').read_bytes()
    assert hashlib.sha256(source).hexdigest() == 'c7dd627b6c98619f2512520e3c0dd039795dac64e4be80752295b5c78e8792ad'
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'is_duplicate')
    from types import SimpleNamespace
    namespace = {'bt': SimpleNamespace(logging=SimpleNamespace(debug=lambda *a, **k: None))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), f'{VALIDATOR_REF}:utils/nanobodies.py', 'exec'), namespace)
    return namespace['is_duplicate']
