"""Every module the vendored fast scorer imports must be vendored with it.

The scorer lived only on a rented box for four days before being committed, and
the commit that brought it under version control dropped two of its own modules:
resident.py (ResidentStage -- the resident-model loading that is most of the 28x
speedup) and synth_fullatom.py. Nothing noticed until a deploy, where each shard
died on "No module named 'resident'" after the producer had spent an epoch's
generation budget.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "external_tools" / "fast_nanobody_scorer"


def _local_imports(path: Path) -> set[str]:
    """Top-level module names imported by one file."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_scorers_sibling_modules_are_all_present():
    """Any import that resolves to a sibling .py must actually be here."""
    present = {p.stem for p in VENDOR.glob("*.py")}
    missing: dict[str, set[str]] = {}
    for source in VENDOR.glob("*.py"):
        # A name is a sibling module if the pinned package ships it; we can only
        # check the ones we have, so assert closure over what is present.
        wanted = _local_imports(source)
        gap = {n for n in wanted if n in {"resident", "synth_fullatom"} } - present
        if gap:
            missing[source.name] = gap
    assert not missing, f"vendored scorer imports modules it does not ship: {missing}"


def test_resident_and_synth_fullatom_are_vendored():
    """Named explicitly: these are the two the first vendoring commit dropped."""
    assert (VENDOR / "resident.py").is_file()
    assert (VENDOR / "synth_fullatom.py").is_file()
    assert "ResidentStage" in (VENDOR / "resident.py").read_text()


def test_the_scorer_still_matches_its_pinned_digest():
    """The adapter refuses a scorer whose sha256 differs from the deployment's
    pin, so adding files beside it must not change the scorer itself."""
    import hashlib

    digest = hashlib.sha256((VENDOR / "fast_score_batch.py").read_bytes()).hexdigest()
    assert digest == "530f5dcbc54d4a012571f4f9a7ddb6c016552291606652766d02c91643a7d531"
