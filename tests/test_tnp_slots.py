import os
from pathlib import Path
import subprocess
import sys
import time

from nanobody_opt import tnp_slots


def test_gpu_visibility_is_preserved(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert tnp_slots.gpu_devices() == ["2", "5"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert tnp_slots.gpu_devices() == []


def test_slot_survives_exec_and_is_released_on_kill(tmp_path):
    ready = tmp_path / "ready"
    claimed = tmp_path / "claimed"
    child = f"import pathlib,time;pathlib.Path({str(ready)!r}).touch();time.sleep(30)"
    first_code = ("from pathlib import Path;import os,sys;from nanobody_opt.tnp_slots import acquire;"
        f"fd,device=acquire(Path({str(tmp_path)!r}),['0'],slots_per_device=1);"
        f"os.execv(sys.executable,[sys.executable,'-c',{child!r}])")
    second_code = ("from pathlib import Path;from nanobody_opt.tnp_slots import acquire;"
        f"fd,device=acquire(Path({str(tmp_path)!r}),['0'],slots_per_device=1);"
        f"Path({str(claimed)!r}).touch()")
    first = subprocess.Popen([sys.executable, "-c", first_code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    second = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), "slot holder did not reach exec"
        second = subprocess.Popen([sys.executable, "-c", second_code], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(.3)
        assert not claimed.exists()
        first.kill()
        first.wait(timeout=5)
        assert second.wait(timeout=5) == 0
        assert claimed.exists()
    finally:
        for process in [first, second]:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)


def test_shipping_shim_invokes_the_bounded_wrapper():
    root = Path(__file__).resolve().parents[1]
    assert '/root/nanobody_opt/tnp_slots.py "$@"' in (root / "ansible/files/root/metanano_shim/bin/TNP").read_text()
