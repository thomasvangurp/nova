#!/usr/bin/env python3
"""AST guard for SHIPPING SN68 workers. Bugs 1/10; catches warn under A2."""
from __future__ import annotations

import argparse
import ast
import builtins
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPING_MODULES = (
    'nanobody_opt_v2.run_live_producer',
    'nanobody_opt_v2.select_elite_parents',
    'nanobody_opt_v2.sequential_exact_race',
    'nanobody_opt_v2.validator_parent_gate',
    'nanobody_opt.local_neighborhood',
    'nanobody_opt.check_validator_novelty',
    'nanobody_opt.filter_public_neighborhood',
    'nanobody_opt.fetch_hf_archive',
    'nanobody_opt.competitor_lane',
)
WORKERS = tuple(ROOT / (name.replace('.', '/') + '.py') for name in SHIPPING_MODULES) + (
    ROOT / 'deploy/submit_host/pi_submit.py',
)

# Original reviewed anchors: parent gate :147, Pi :637, producer :2041-2045.
# Match AST structure + lexical context, not fragile line numbers after refactors.
ARGPARSE_EXITS = {
    '--confirmation-time-share must be between zero and one': 'not 0 < args.confirmation_time_share < 1',
    'budget timing bounds must be positive': 'min(args.budget_load_seconds, args.budget_gpu_seconds, args.budget_cpu_guard_seconds) <= 0',
    'publication preparation guard must be nonnegative': 'args.publication_prepare_blocks < 0',
    'complete-screen pool and target epoch must be supplied together': 'bool(args.complete_screen_pool) != (args.complete_screen_until_epoch is not None)',
    'complete-screen finalists must cover the tournament survivor floor': 'args.complete_screen_finalists < args.replication_minimum_keep',
    'epoch candidate budget must cover the seed and finalist floors': 'args.epoch_candidate_budget < max(3, args.minimum_seeds, args.epoch_confirmation_finalists)',
    'epoch proposal budget must cover the exact-screen budget': 'args.epoch_proposal_budget < args.epoch_candidate_budget',
    'epoch finalists must cover the tournament survivor floor': 'args.epoch_confirmation_finalists < args.replication_minimum_keep',
    'bounded epoch cycles must start after the existing full-pool run': 'args.epoch_cycle_start is not None and args.complete_screen_until_epoch is not None and args.epoch_cycle_start <= args.complete_screen_until_epoch',
}

# Genuine fail-closed error messages (all reachable only as SUBPROCESS exits, so
# they become CalledProcessError and are caught by the producer -- they cannot
# kill the producer loop). Allowed so the AST gate stops flagging them as
# "banned worker exit". Matched as a substring of the raising source line, which
# survives refactors better than exact line numbers.
ALLOWED_EXIT_MESSAGES = (
    "validator parent gate unavailable",
    "historical reservoir target mismatch",
    "no donor sequences found",
    "exclude file has no sequence column",
    "supported single-substitution lane exceeds",
    "cannot load the validator's ranking",
    "HF archive fetch failed and no cached copy",
    "no usable proposals in",
    # Pi submit host: genuine fail-closed pull/artifact errors (subprocess exits).
    "vastai failed",
    "vastai did not return JSON",
    "expected exactly one running instance",
    "has no resolvable ssh endpoint",
    "pull from",
    "nanobody artifact is",
)

# Modules whose ``if __name__ == '__main__'`` guard legitimately ends in
# ``sys.exit(main())`` / ``raise SystemExit(main())`` (the standard CLI
# entrypoint pattern). The allowlist stays context-bound so an exit inside an
# arbitrary worker function is still caught.
CLI_ENTRYPOINT_MODULES = (
    'deploy/submit_host/pi_submit.py',
    'nanobody_opt/fetch_hf_archive.py',
    'nanobody_opt/filter_public_neighborhood.py',
)

# ---------------------------------------------------------------------------
# Launch-config gate: user-directed shortlist-deferred TNP must survive redeploy.
# Full TNP runs on the race shortlist and again before publication.
# ---------------------------------------------------------------------------
LAUNCH_CONFIG_FILES = (
    'go_live_v2.sh',
    'restart_v2.sh',
    'deploy/producer/producer_argv.txt',
)
_LAYER2_IGBLAST_BANNED = re.compile(r'--layer2-igblast-only\s+0\b')

# ---------------------------------------------------------------------------
# Dual-tree import gate (gap #15 in TESTING.md): production ships these modules
# from ``nanobody_opt_v2``, never ``nanobody_opt``. A test that imports the
# legacy spelling silently exercises a module the deploy never runs. Scanned,
# not imported: the fd-redirecting validator_parent_gate must never be imported
# in-process anyway.
# ---------------------------------------------------------------------------
V2_SHIPPING_MODULES = frozenset({
    'run_live_producer',
    'select_elite_parents',
    'sequential_exact_race',
    'validator_parent_gate',
})


def check_launch_config(root: Path) -> list[str]:
    """Return launch-config violations: files re-enabling full-pool TNP."""
    problems = []
    for name in LAUNCH_CONFIG_FILES:
        path = root / name
        try:
            text = path.read_text()
        except OSError as error:
            problems.append(f'{name}: unreadable launch config ({error})')
            continue
        if _LAYER2_IGBLAST_BANNED.search(text):
            problems.append(f'{name}: banned flag --layer2-igblast-only 0')
    return problems


def scan_dual_tree_imports(test_root: Path) -> list[str]:
    """Flag tests importing a v2-shipping module from the legacy tree."""
    problems = []
    for path in sorted(test_root.glob('*.py')):
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split('.')
                    if (len(parts) >= 2 and parts[0] == 'nanobody_opt'
                            and parts[1] in V2_SHIPPING_MODULES):
                        problems.append(f'{path.name}:{node.lineno}: imports legacy {alias.name} (ships v2)')
            elif isinstance(node, ast.ImportFrom):
                parts = (node.module or '').split('.')
                if not parts or parts[0] != 'nanobody_opt':
                    continue
                if len(parts) >= 2 and parts[1] in V2_SHIPPING_MODULES:
                    problems.append(f'{path.name}:{node.lineno}: imports legacy {node.module} (ships v2)')
                elif len(parts) == 1:
                    for alias in node.names:
                        if alias.name in V2_SHIPPING_MODULES:
                            problems.append(f'{path.name}:{node.lineno}: imports legacy {node.module}.{alias.name} (ships v2)')
    return problems


def signature(node):
    return ast.dump(node, include_attributes=False)


def allowed_exit(node, relative, parents, source_lines=None):
    functions = [p.name for p in parents if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef))]
    guards = [p for p in parents if isinstance(p, ast.If)]
    # These two command-line workers are isolated producer subprocesses.
    # Their main functions reject insufficient evidence or invalid CLI input;
    # exits anywhere inside their reusable helper functions remain forbidden.
    if (relative in {'nanobody_opt_v2/select_elite_parents.py', 'nanobody_opt/competitor_lane.py'}
            and functions == ['main'] and isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == 'SystemExit'):
        return True
    main_guard = any(signature(p.test) == signature(ast.parse("__name__ == '__main__'", mode='eval').body)
                     for p in guards)
    # Standard CLI entrypoint: ``sys.exit(main())`` / ``raise SystemExit(main())``
    # at module scope under the ``if __name__ == '__main__'`` guard, and only for
    # the known CLI entrypoint modules.
    if not functions and main_guard and relative in CLI_ENTRYPOINT_MODULES:
        if isinstance(node, ast.Call) and signature(node) == signature(ast.parse('sys.exit(main())', mode='eval').body):
            return True
        if isinstance(node, ast.Raise) and signature(node) == signature(ast.parse('raise SystemExit(main())').body[0]):
            return True
    # Message-allowlisted genuine fail-closed errors (subprocess-only).
    if source_lines is not None and isinstance(node, ast.Raise):
        try:
            start = node.lineno - 1
            end = max(start + 1, node.end_lineno or node.lineno)
            span_text = "\n".join(source_lines[start:end])
        except (IndexError, AttributeError):
            span_text = ''
        if any(message in span_text for message in ALLOWED_EXIT_MESSAGES):
            return True
    if relative == 'nanobody_opt_v2/validator_parent_gate.py' and not functions and main_guard:
        return (isinstance(node, ast.Call) and signature(node) == signature(ast.parse('os._exit(1)', mode='eval').body)
                and any(isinstance(p, ast.ExceptHandler) and isinstance(p.type, ast.Name)
                        and p.type.id == 'BaseException' for p in parents))
    if relative == 'nanobody_opt_v2/run_live_producer.py' and functions == ['main'] and isinstance(node, ast.Raise):
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == 'SystemExit':
            message = node.exc.args[0].value if len(node.exc.args) == 1 and isinstance(node.exc.args[0], ast.Constant) else None
            condition = ARGPARSE_EXITS.get(message)
            return bool(condition and guards and signature(guards[-1].test) == signature(ast.parse(condition, mode='eval').body)
                        and not any(isinstance(p, (ast.For, ast.While)) for p in parents))
    return False


def inspect_source(source: str, relative: str = '') -> tuple[list[tuple[int, str]], list[int]]:
    tree = ast.parse(source)
    aliases = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            aliases.update({a.asname or a.name: a.name for a in n.names})
        elif isinstance(n, ast.ImportFrom) and n.module in ('sys', 'os', 'builtins'):
            aliases.update({a.asname or a.name: f'{n.module}.{a.name}' for a in n.names})

    def name(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return name(node.value) + '.' + node.attr
        return ''

    fatal = {key for key, value in vars(builtins).items()
             if isinstance(value, type) and issubclass(value, BaseException) and not issubclass(value, Exception)}
    # Resolve locally defined fatal subclasses (including multi-level chains).
    for _ in range(len(tree.body) + 1):
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and any(name(b).removeprefix('builtins.') in fatal for b in n.bases):
                fatal.add(n.name)
    errors, warnings = [], []
    source_lines = source.splitlines()

    def visit(node, parents):
        violation = None
        if isinstance(node, ast.Call) and name(node.func) in ('sys.exit', 'os._exit', 'builtins.exit', 'builtins.quit', 'exit', 'quit'):
            violation = name(node.func)
        if isinstance(node, ast.Raise) and node.exc is not None:
            exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if name(exception).removeprefix('builtins.') in fatal:
                violation = 'raise ' + name(exception)
        if violation and not allowed_exit(node, relative, parents, source_lines):
            errors.append((node.lineno, violation))
        if isinstance(node, (ast.Try, ast.TryStar)):
            for i, handler in enumerate(node.handlers):
                if not isinstance(handler.type, ast.Name) or handler.type.id != 'Exception':
                    continue
                propagated = any(isinstance(h.type, ast.Name) and h.type.id in ('SystemExit', 'BaseException')
                                 and len(h.body) == 1 and isinstance(h.body[0], ast.Raise) and h.body[0].exc is None
                                 for h in node.handlers[i + 1:])
                if not propagated:
                    warnings.append(handler.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, parents + [node])
    visit(tree, [])
    return errors, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path)
    parser.add_argument('--strict', action='store_true',
                        help='escalate dual-tree import warnings to failures')
    args = parser.parse_args(argv)
    failures = 0
    for path in args.paths or WORKERS:
        try:
            relative = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else ''
            errors, warnings = inspect_source(path.read_text(), relative)
        except (OSError, SyntaxError):
            print(f'ERROR {path}: unreadable or invalid Python')
            failures += 1
            continue
        for line, detail in errors:
            print(f'ERROR {path}:{line}: banned worker exit {detail}')
        for line in warnings:
            print(f'WARN {path}:{line}: except Exception requires recovery/propagation rationale')
        failures += len(errors)
    for problem in check_launch_config(ROOT):
        print(f'ERROR {problem}')
        failures += 1
    dual_tree = scan_dual_tree_imports(ROOT / 'tests')
    for problem in dual_tree:
        print(f'WARN {problem}')
    if args.strict:
        failures += len(dual_tree)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
