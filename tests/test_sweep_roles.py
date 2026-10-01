"""Which role the sweep is allowed to substitute.

The sweep has only ever varied the LAST reagent, holding components[:-1]
fixed. The epoch-25226 winner submitted `rxn:3:61524:*:98582` -- the first and
third reagents fixed, the MIDDLE one varying. That is a portfolio this builder
could not have generated at any budget, on any bank size, because the axis was
not searched. Reaction 3 has three roles and we were searching one of them.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "bank_builder.py").read_text()
TREE = ast.parse(SOURCE)


def _main():
    return next(n for n in TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == "main")


def test_the_substituted_position_is_a_variable_not_the_last_index():
    """`prefix + (identifier,)` can only ever append to the end."""
    source = ast.get_source_segment(SOURCE, _main()) or SOURCE
    assert "prefix, swept = tuple(components[:-1]), components[-1]" not in source, (
        "the sweep still hardcodes the last position")
    assert "components[:index]" in source and "components[index + 1:]" in source


def test_a_ranker_is_built_for_every_swept_role():
    """The ranker fingerprints ONE role. Reusing the last role's ranker for
    role 0 returns [] -- the reference id is absent from its table -- which
    disables the lane silently rather than raising, exactly as it did for
    reactions 1, 2, 4 and 5 before --reaction existed."""
    source = ast.get_source_segment(SOURCE, _main()) or SOURCE
    assert "rankers = {" in source
    assert "role_index=index" in source


def test_last_remains_the_default():
    """`all` triples proposals per anchor and builds a ranker per role. It
    earns its way in against a control, not by being switched on silently."""
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        if not node.args or getattr(node.args[0], "value", None) != "--sweep-roles":
            continue
        defaults = {k.arg: getattr(k.value, "value", None) for k in node.keywords}
        assert defaults.get("default") == "last"
        choices = next((k.value for k in node.keywords if k.arg == "choices"), None)
        assert choices is not None
        assert {e.value for e in choices.elts} == {"last", "all"}
        return
    raise AssertionError("--sweep-roles is not declared")


def test_only_the_enabled_roles_are_fingerprinted():
    """Building a ranker per role costs a full fingerprint pass over that
    role's reagents. `last` must still pay for exactly one."""
    source = ast.get_source_segment(SOURCE, _main()) or SOURCE
    assert "swept_roles = (tuple(range(len(roles)))" in source
    assert "else (len(roles) - 1,)" in source


def test_an_anchor_is_retired_only_when_no_role_yielded():
    """Role 0 still having ground left is a reason to come back even if role 2
    is mined out. Retiring on the last role alone would throw that away."""
    source = ast.get_source_segment(SOURCE, _main()) or SOURCE
    marker = source.index("Retire anchors that yielded nothing")
    arm = source[marker:marker + 700]
    assert "if not any(" in arm and "proposals.values()" in arm


def test_the_round_record_says_how_many_roles_were_swept():
    """A configuration that silently reverted would otherwise be invisible in
    the per-round accounting, which is where the last two regressions showed."""
    assert "swept_roles=len(swept_roles)" in SOURCE


def test_the_module_compiles():
    compile(SOURCE, "tools/bank_builder.py", "exec")
