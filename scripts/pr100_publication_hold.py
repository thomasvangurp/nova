"""Build a minimal, reviewable nanobody-only hold for an unqualified cutover.

This creates patched source; it does not install it, restart services, read
wallets or sign submissions. A qualified release must explicitly replace the
hold. No environment variable or stale-artifact option bypasses it.
"""
from __future__ import annotations

import ast
import hashlib

MARKER = '# PR100_QUALIFICATION_HOLD_V1'


def _empty_batch_noop(text):
    """An empty molecule plan must not access chain, wallets or upload APIs."""
    functions = [n for n in ast.parse(text).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'commit_batch']
    if len(functions) != 1:
        raise ValueError('exactly one parallel commit function is required')
    function = functions[0]
    if [a.arg for a in function.args.args] != ['rows', 'arguments', 'environment']:
        raise ValueError('unsupported parallel commit interface')
    first = function.body[0]
    has_docstring = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                     and isinstance(first.value.value, str))
    line = first.end_lineno if has_docstring else first.lineno - 1
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:line]) + '    if not rows:\n        return 0, 0, []\n' + ''.join(lines[line:])


def _hold_nanobody_artifact_io(text, activation_epoch):
    """Keep the independent molecule path free of held nanobody I/O."""
    functions = [n for n in ast.parse(text).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(functions) != 1:
        raise ValueError('exactly one submission main function is required')
    function = functions[0]
    original = ast.get_source_segment(text, function)
    pulls = [n for n in ast.walk(function) if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name)
             and n.value.func.id == 'pull_producer']
    load = '_load(Path(arguments.nanobody))'
    loads = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_load'
             and ast.unparse(n) == load]
    if len(pulls) != 1 or len(loads) != 1 or original.count(load) != 1:
        raise ValueError('unsupported nanobody artifact I/O')
    node = pulls[0]
    lines = original.splitlines(keepends=True)
    start, end = node.lineno-function.lineno, node.end_lineno-function.lineno+1
    indent = ' ' * node.col_offset
    replacement = (indent + f'if arguments.epoch < {activation_epoch}:\n'
                   + ''.join('    ' + line for line in lines[start:end]))
    updated = ''.join(lines[:start]) + replacement + ''.join(lines[end:])
    updated = updated.replace(load,
        f'(None if arguments.epoch >= {activation_epoch} else {load})', 1)
    return text.replace(original, updated, 1)


def _allow_molecule_only_plan(text):
    tree = ast.parse(text)
    plans = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan']
    if len(plans) != 1:
        raise ValueError('exactly one molecule planning function is required')
    function = plans[0]
    original = ast.get_source_segment(text, function)
    skipped = [n for n in ast.walk(function) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'not seq']
    if len(skipped) != 1 or 'skipped_no_sequence' not in ast.get_source_segment(text, skipped[0]):
        raise ValueError('unrecognized molecule-only planning restriction')
    lines = original.splitlines(keepends=True)
    node = skipped[0]
    source = ''.join(lines[:node.lineno-function.lineno] + lines[node.end_lineno-function.lineno+1:])
    replacements = {
        'plaintext=f"{mols or NULL}|{seq}",': 'plaintext=f"{mols or NULL}|{seq or NULL}",',
        'has_molecules=bool(mols), has_sequence=True)': 'has_molecules=bool(mols), has_sequence=bool(seq))',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('required molecule planning replacement did not match exactly once')
        source = source.replace(before, after, 1)
    return text.replace(original, source, 1)


def publication_routes(source: bytes):
    """Refuse a different installed publication dataflow; report code locations."""
    tree = ast.parse(source)
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(mains) != 1:
        raise ValueError('exactly one submission main function is required')
    main = mains[0]
    plan_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'plan']
    sequence_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == 'sequences_by_uid']
    assignments = [n for n in ast.walk(main) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                   and any(isinstance(t, ast.Name) and t.id == 'sequences'
                           for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if (len(plan_calls) != 1 or len(sequence_calls) != 1 or len(assignments) != 1
            or assignments[0].value is not sequence_calls[0]
            or plan_calls[0] not in list(ast.walk(main))
            or len(plan_calls[0].args) < 3
            or not isinstance(plan_calls[0].args[2], ast.Name)
            or plan_calls[0].args[2].id != 'sequences'):
        raise ValueError('unsupported sequence publication route')
    return {'sequence_producer': 'sequences_by_uid', 'producer_call_line': sequence_calls[0].lineno,
            'plan_call_line': plan_calls[0].lineno, 'sequence_assignments': 1,
            'plan_calls': 1, 'scope': 'installed consumer main only; not arbitrary external callers'}


def patch_consumer(source: bytes, *, expected_sha256: str, activation_epoch=25955) -> bytes:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('installed consumer changed; inspect the new source first')
    if type(activation_epoch) is not int or activation_epoch != 25955:
        raise ValueError('unexpected activation epoch')
    text = source.decode('utf-8')
    if MARKER in text:
        raise ValueError('hold already present')
    if not any(isinstance(n, ast.Import) and any(a.name == 'json' and a.asname in (None, 'json')
               for a in n.names) for n in ast.parse(text).body):
        raise ValueError('consumer must import json at module scope')
    publication_routes(source)
    text = _allow_molecule_only_plan(text)
    text = _hold_nanobody_artifact_io(text, activation_epoch)
    text = _empty_batch_noop(text)
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
    # The installer additionally pins and reviews the complete transformed
    # bytes: planning, guarded artifact I/O, empty-batch no-op and this guard.
    return result.encode('utf-8')


def verified_restore(current: bytes, backup: bytes, *, expected_current_sha256: str,
                     expected_backup_sha256: str) -> bytes:
    """Pure rollback rehearsal. Installation policy must gate any live restore."""
    if (hashlib.sha256(current).hexdigest() != expected_current_sha256 or
            hashlib.sha256(backup).hexdigest() != expected_backup_sha256):
        raise ValueError('rollback source or backup identity mismatch')
    return backup


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
