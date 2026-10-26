"""Exact upstream molecule gate, loaded without utils import side effects.

Source/version binding is evidence about this gate, not a complete RDKit runtime
qualification. No seed or embedding parameter is changed here.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

ACTIVATION_EPOCH = 25955
UPSTREAM_BLOB_SHA1 = '7d309072c82c8ba72ddeb61b1fdc1f3b292f00e1'
UPSTREAM_FILE_SHA256 = '830136d3f4c87ecca304c7599e5878c10699274272f9b66792bb3ac72ac88b0b'
FUNCTION_SHA256 = {
    'is_boltz_safe_smiles': 'bf66ddbdb94b668cebf5ecb75544567bdd752009f68f8b3371d1e5fedd240f04',
    'get_heavy_atom_count': '8b8f580fb5724fc9ca953e314478e45da691d288e9bce425fbb3d7a109891bb7',
    'contains_atom_type': '761284e6df6eea9615267fad2c85a97cfd9f4596088388b7dfc559de86c8d3b0',
}
RDKIT_VERSION = (2024, 9, 4)  # Exact validator requirements: rdkit==2024.9.4.


class MoleculeGateUnavailable(ValueError):
    """A fixed reason code, never native logs or credential-bearing exceptions."""


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_functions(source):
    """Extract only the three pinned pure functions; never execute module setup."""
    text = source.decode('utf-8') if isinstance(source, bytes) else source
    tree = ast.parse(text)
    functions = {}
    for name, expected in FUNCTION_SHA256.items():
        found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(found) != 1:
            raise MoleculeGateUnavailable('required_upstream_function_missing')
        segment = ast.get_source_segment(text, found[0])
        if hashlib.sha256(segment.encode()).hexdigest() != expected:
            raise MoleculeGateUnavailable('upstream_function_source_mismatch')
        functions[name] = found[0]
    return functions


def rdkit_identity():
    try:
        import rdkit
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem, rdDistGeom
        version = str(rdBase.rdkitVersion)
        if tuple(int(x) for x in version.split('.')) != RDKIT_VERSION:
            raise MoleculeGateUnavailable('rdkit_version_mismatch')
        package_version = importlib.metadata.version('rdkit')
        if tuple(int(x) for x in package_version.split('.')) != RDKIT_VERSION:
            raise MoleculeGateUnavailable('rdkit_distribution_version_mismatch')
        files = {str(Path(module.__file__).resolve()): digest_file(module.__file__)
                 for module in (rdkit, Chem, AllChem, rdBase, rdDistGeom)}
    except MoleculeGateUnavailable:
        raise
    except Exception as error:
        raise MoleculeGateUnavailable('rdkit_runtime_identity_unavailable') from error
    return Chem, AllChem, {'rdkit_version': version, 'distribution_version': package_version,
        'module_file_sha256': files, 'complete_native_dependency_closure': False}


def load_helpers(source_path):
    source_path = Path(source_path)
    try:
        raw = source_path.read_bytes()
        functions = source_functions(raw)
        Chem, AllChem, runtime = rdkit_identity()
    except MoleculeGateUnavailable:
        raise
    except Exception as error:
        raise MoleculeGateUnavailable('molecule_gate_source_unavailable') from error
    # Suppress the helper's optional parse-failure warning only. No logging or
    # dotenv/application module is imported, and the scientific code is exact.
    namespace = {'Chem': Chem, 'AllChem': AllChem,
                 'bt': SimpleNamespace(logging=SimpleNamespace(warning=lambda *args: None))}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              *functions.values()], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), 'exec'), namespace)
    identity = {'schema': 'pr100-molecule-gate-v1', 'upstream_blob_sha1': UPSTREAM_BLOB_SHA1,
                'upstream_file_sha256': UPSTREAM_FILE_SHA256,
                'source_path': str(source_path.resolve()), 'source_file_sha256': hashlib.sha256(raw).hexdigest(),
                'function_sha256': dict(FUNCTION_SHA256), 'runtime': runtime,
                'runtime_qualified': False, 'seed_override': False, 'embedding_attempts_maximum': 2}
    return {'heavy': namespace['get_heavy_atom_count'], 'banned': namespace['contains_atom_type'],
            'boltz': namespace['is_boltz_safe_smiles'], 'molecule_gate_identity': identity}
