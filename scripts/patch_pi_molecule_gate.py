#!/usr/bin/env python3
"""Prepare a hash-bound single-function Pi correction; never install it."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PI_SOURCE_SHA256 = '0833c725daefa191a9f97cb18d63759b834df5f4cedcf7cbdf21b10683cf8c15'
PI_FUNCTION_SHA256 = '5d47685526fbba5a7c36930403e7c763211fadc32ceeeaa72e974a9dc6b15dc1'


def replacement_function():
    spec = importlib.util.spec_from_file_location('_pi_patch_molecule_gate', ROOT/'tools/molecule_gate.py')
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    text = (ROOT/'utils/molecules.py').read_text()
    node = gate.source_functions(text)['is_boltz_safe_smiles']
    source = ast.get_source_segment(text, node)
    source = source.replace('def is_boltz_safe_smiles(', 'def boltz_safe(', 1)
    lines = source.splitlines(keepends=True)
    # Keep the exact scientific body. Pi's independent helper imports RDKit
    # locally, as its existing function does; no utils namespace is imported.
    lines.insert(node.body[0].end_lineno-node.lineno+1, '    from rdkit.Chem import AllChem\n')
    return ''.join(lines).encode('utf-8')


def patch_pi_boltz_safe(source: bytes, *, expected_sha256=PI_SOURCE_SHA256):
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('Pi molecule helper source changed')
    text = source.decode('utf-8'); tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'boltz_safe']
    if len(nodes) != 1:
        raise ValueError('Pi boltz_safe function missing or ambiguous')
    node = nodes[0]
    segment = ast.get_source_segment(text, node)
    if hashlib.sha256(segment.encode()).hexdigest() != PI_FUNCTION_SHA256:
        raise ValueError('Pi boltz_safe function changed or already patched')
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    replacement = replacement_function()
    result = source[:start] + replacement + source[end:]
    ast.parse(result.decode('utf-8'))
    # The only mutation is the exact AST span, including when unrelated source
    # contains non-ASCII text, comments, unusual whitespace or audit history.
    assert result[:start] == source[:start]
    assert result[start+len(replacement):] == source[end:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        # CLI always uses the exact audited installed-file hash; no override.
        source = args.source.read_bytes()
        candidate = patch_pi_boltz_safe(source)
        fd = os.open(args.output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(candidate)
        print(json.dumps({'status':'candidate_prepared_not_installed',
            'original_sha256':PI_SOURCE_SHA256,'candidate_sha256':hashlib.sha256(candidate).hexdigest(),
            'replacement_function_sha256':hashlib.sha256(replacement_function()).hexdigest()}))
        return 0
    except Exception as error:
        print(json.dumps({'status':'refused','error_class':type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
