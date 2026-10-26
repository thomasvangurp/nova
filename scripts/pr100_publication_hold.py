"""Build a minimal, reviewable nanobody-only hold for an unqualified cutover.

This creates patched source; it does not install it, restart services, read
wallets or sign submissions. A qualified release must explicitly replace the
hold. No environment variable or stale-artifact option bypasses it.
"""
from __future__ import annotations

import ast
import hashlib

MARKER = '# PR100_QUALIFICATION_HOLD_V1'


def _allow_molecule_only_plan(text):
    tree = ast.parse(text)
    plans = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan']
    if not plans:
        return text
    function = plans[0]
    original = ast.get_source_segment(text, function)
    skipped = [n for n in ast.walk(function) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'not seq']
    if skipped:
        if len(skipped) != 1 or 'skipped_no_sequence' not in ast.get_source_segment(text, skipped[0]):
            raise ValueError('unrecognized molecule-only planning restriction')
        lines = original.splitlines(keepends=True)
        node = skipped[0]
        source = ''.join(lines[:node.lineno-function.lineno] + lines[node.end_lineno-function.lineno+1:])
        source = source.replace('plaintext=f"{mols or NULL}|{seq}",',
                                'plaintext=f"{mols or NULL}|{seq or NULL}",', 1)
        source = source.replace('has_molecules=bool(mols), has_sequence=True)',
                                'has_molecules=bool(mols), has_sequence=bool(seq))', 1)
        text = text.replace(original, source, 1)
    return text


def patch_consumer(source: bytes, *, expected_sha256: str, activation_epoch=25955) -> bytes:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('installed consumer changed; inspect the new source first')
    if type(activation_epoch) is not int or activation_epoch != 25955:
        raise ValueError('unexpected activation epoch')
    text = source.decode('utf-8')
    if MARKER in text:
        raise ValueError('hold already present')
    text = _allow_molecule_only_plan(text)
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'sequences_by_uid']
    if len(functions) != 1 or [a.arg for a in functions[0].args.args[:2]] != ['artifact', 'epoch']:
        raise ValueError('unsupported consumer allocation interface')
    function = functions[0]
    first = function.body[0]
    after_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    line = first.end_lineno if after_docstring else first.lineno-1
    lines = text.splitlines(keepends=True)
    guard = (f'    {MARKER}\n'
             f'    if epoch >= {activation_epoch}:\n'
             '        print(json.dumps({"event": "pr100_qualification_hold", "epoch": epoch,\n'
             '                          "action": "withhold unqualified nanobodies"}), flush=True)\n'
             '        return {}\n')
    result = ''.join(lines[:line]) + guard + ''.join(lines[line:])
    compile(result, '<held-pi-consumer>', 'exec')
    # Apart from the bounded molecule-only planning repair above, this is one
    # insertion; allocation and all other statements are preserved bytewise.
    assert result.replace(guard, '', 1) == text
    return result.encode('utf-8')


def patch_signer(source: bytes, *, expected_sha256: str) -> bytes:
    """Permit the null sequence half while preserving strict molecule checks."""
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('installed signer changed; inspect the new source first')
    text = source.decode('utf-8')
    tree = ast.parse(text)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_plaintext')
    original = ast.get_source_segment(text, function)
    condition = 'if not sequence or set(sequence) - CANONICAL_AMINO_ACIDS:'
    branch = 'if molecule_part == NULL_TOKEN:\n        return plaintext'
    if original.count(condition) != 1 or original.count(branch) != 1:
        raise ValueError('unsupported signer validation implementation')
    replacement = original.replace(condition,
        'if not sequence or (sequence != NULL_TOKEN and set(sequence) - CANONICAL_AMINO_ACIDS):')
    replacement = replacement.replace(branch,
        'if molecule_part == NULL_TOKEN:\n'
        '        if sequence == NULL_TOKEN:\n'
        '            raise ValueError("refusing two empty competition halves")\n'
        '        return plaintext')
    result = text.replace(original, replacement, 1)
    compile(result, '<molecule-only-signer>', 'exec')
    return result.encode('utf-8')
