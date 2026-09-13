"""The scorer must dispatch to GPUs this machine actually has.

The default was the literal "0,1,2,3,4,5,6,7,8,9" -- the old ten-GPU box written
down. On the two-GPU box that replaced it, eight of every ten shards were
dispatched to devices that do not exist. Each chunk raised, the shard worker
caught it and continued, and the run ended with "MERGE LOST SEQUENCES: submitted
350, merged 70": two surviving shards of ten, 35 sequences each. Nothing in that
message names CUDA, a device, or the missing GPUs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import run_live_producer as rlp
from nanobody_opt import sequential_exact_race as race

REPO = Path(__file__).resolve().parents[1]


def test_no_module_hardcodes_the_old_boxs_gpu_count():
    # Checks the argparse default, not prose: the docstrings deliberately quote
    # the old value to explain what went wrong.
    for name in ("run_live_producer.py", "sequential_exact_race.py",
                 "qualify_screen_leader.py"):
        source = (REPO / "nanobody_opt" / name).read_text()
        assert 'default="0,1,2,3,4,5,6,7,8,9"' not in source, name


def test_the_discovered_list_matches_the_device_count(monkeypatch):
    import types

    for module in (rlp, race):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 2))
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert module.visible_gpu_ids() == "0,1"
        fake.cuda.device_count = lambda: 8
        assert module.visible_gpu_ids() == "0,1,2,3,4,5,6,7"


def test_a_machine_with_no_cuda_still_yields_one_id(monkeypatch):
    """Returning an empty list would produce zero shards and score nothing,
    which is a worse failure than trying and reporting."""
    import types

    for module in (rlp, race):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0))
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert module.visible_gpu_ids() == "0"


def test_an_unimportable_torch_does_not_break_discovery(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert rlp.visible_gpu_ids() == "0"


def test_help_still_works_without_torch():
    """The deploy gate runs --help where torch may be absent, which is why the
    default is resolved after parsing rather than as an argparse default."""
    import subprocess

    done = subprocess.run(
        [sys.executable, "-m", "nanobody_opt.run_live_producer", "--help"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0
    assert "--gpus" in done.stdout
