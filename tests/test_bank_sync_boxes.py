"""Which control module drives which bank box.

`box()` used to read `'bank_box_control' if name == 'box1' else
'bank_box2_control'`. Every name that was not box1 therefore resolved to box2:
a third box would have had its bank merged into box2's, pulled from box2's
manifest, and reported success the whole way. That is the same failure shape as
a default argument that silently disables a lane, and it is only visible as
molecules that never appear.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "tools"))
    return module


@pytest.fixture
def sync():
    return _module("_bank_sync", "bank_sync.py")


def test_every_box_has_its_own_control_module(sync):
    assert sync.BOXES == {
        "box1": "bank_box_control",
        "box2": "bank_box2_control",
        "box3": "submit_box_control",
    }
    assert len(set(sync.BOXES.values())) == len(sync.BOXES)


def test_an_unknown_box_is_an_error_and_not_a_default(sync):
    """The whole point: box3 must not silently become box2."""
    with pytest.raises(SystemExit, match="unknown box"):
        sync.box("box4")


def test_the_named_control_modules_all_exist(sync):
    for filename in sync.BOXES.values():
        assert (ROOT / "tools" / f"{filename}.py").is_file(), filename


def test_each_control_module_exposes_what_sync_calls(sync):
    """`box()` uses instance() and ssh(); `pull`/`push_and_merge` use the
    public address out of the route. A module missing one of these fails at
    rsync time, mid-sync, having already pulled from the others."""
    for filename in sync.BOXES.values():
        module = _module(f"_ctl_{filename}", f"{filename}.py")
        assert callable(getattr(module, "instance", None)), filename
        assert callable(getattr(module, "ssh", None)), filename


def test_the_boxes_guard_each_other_out(sync):
    """Each control module's FORBIDDEN set must name the OTHER boxes, so a
    stale manifest can never aim a stop or a destroy at the wrong bank."""
    identifiers = {}
    for name, filename in sync.BOXES.items():
        module = _module(f"_guard_{filename}", f"{filename}.py")
        manifest = getattr(module, "MANIFEST", None)
        if manifest is None or not manifest.is_file():
            continue
        import json
        identifiers[name] = json.loads(manifest.read_text())["id"]
    for name, filename in sync.BOXES.items():
        module = _module(f"_fb_{filename}", f"{filename}.py")
        forbidden = getattr(module, "FORBIDDEN", set())
        for other, identifier in identifiers.items():
            if other != name:
                assert identifier in forbidden, (
                    f"{filename} does not guard {other} ({identifier})")
