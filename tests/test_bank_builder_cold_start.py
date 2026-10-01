"""Building a bank for a reaction we have never mined before.

The sweep lane proposes analogues of molecules that already scored well, so on
an empty bank it has nothing to anchor on: `pairs` is empty, `fresh` is empty,
and `floor = fresh[-1][1]` raised IndexError before the first batch was ever
proposed. It never showed up on the first two bank boxes because both were
seeded from an existing bank, so `fresh` was never empty -- and it is exactly
the cold start that opening a second reaction requires.

Checked structurally rather than by running main(), which wants rdkit, a
combinatorial database and four GPUs. The structure is the invariant: every
read of `fresh[-1]` must sit under a branch that has established `fresh` is
non-empty.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools" / "bank_builder.py").read_text()
TREE = ast.parse(SOURCE)


def _is_not_fresh(test: ast.expr) -> bool:
    return (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Name) and test.operand.id == "fresh")


def _guarded_regions() -> list[list[ast.stmt]]:
    """The `else:` bodies of every `if not fresh:` in the module."""
    return [node.orelse for node in ast.walk(TREE)
            if isinstance(node, ast.If) and _is_not_fresh(node.test) and node.orelse]


def _subscripts_of_fresh() -> list[ast.Subscript]:
    return [node for node in ast.walk(TREE)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == "fresh"
            and isinstance(node.slice, ast.UnaryOp)]


def test_the_module_has_the_subscript_this_is_about():
    """If the sweep is rewritten and `fresh[-1]` disappears, this file should
    be deleted deliberately rather than passing vacuously."""
    assert _subscripts_of_fresh(), "no negative subscript of `fresh` remains"


def test_every_negative_subscript_of_fresh_is_inside_a_non_empty_branch():
    guarded = _guarded_regions()
    assert guarded, "`if not fresh:` no longer has an else branch to guard it"
    protected = {id(node) for region in guarded
                 for statement in region for node in ast.walk(statement)}
    for subscript in _subscripts_of_fresh():
        assert id(subscript) in protected, (
            f"fresh[-1] at line {subscript.lineno} is not guarded by "
            "`if not fresh:` / `else:`")


def test_a_cold_round_still_proposes_from_the_coverage_lane():
    """Bailing out of the round would mean a new reaction never starts:
    coverage puts the first molecules in, and the sweep needs those to exist."""
    cold = [node for node in ast.walk(TREE)
            if isinstance(node, ast.If) and _is_not_fresh(node.test) and node.orelse]
    assert cold, "expected the guarded form"
    arm = cold[-1].body
    assigns_empty_anchors = any(
        isinstance(statement, ast.Assign)
        and any(getattr(t, "id", None) == "anchors" for t in statement.targets)
        and isinstance(statement.value, ast.List) and not statement.value.elts
        for statement in arm)
    assert assigns_empty_anchors, "the cold arm must fall through with no anchors"
    for statement in arm:
        for node in ast.walk(statement):
            assert not isinstance(node, (ast.Return, ast.Continue, ast.Raise)), (
                "a cold bank must still run the round, not abandon it")


def test_the_module_compiles():
    compile(SOURCE, "tools/bank_builder.py", "exec")


def test_no_reaction_is_hardcoded_in_the_generalised_builder():
    """The literal 3, three times over.

    bank_rescore.py carried it in two places and bank_builder.py in a third:
    the records handed to score_boltz_batch were labelled reaction=3 whatever
    was being built, and the scorer -- which filters its input by reaction --
    saw an empty shortlist. It reported "archive shortlist is empty", the
    builder counted a batch failure, and a reaction-2 run died having scored
    nothing. Every one of these tools takes --reaction now; a literal is a
    tool that silently only works for the bank it was written against.
    """
    for name in ("bank_builder.py", "bank_rescore.py", "bank_release.py"):
        source = (ROOT / "tools" / name).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "reaction":
                assert not (isinstance(node.value, ast.Constant)
                            and node.value.value in (1, 2, 3, 4, 5)), (
                    f"{name}:{node.value.lineno} passes reaction={node.value.value} "
                    "as a literal")
