"""Host-wide TNP GPU slots, shared by producer and publisher subprocesses.

The lock stays open across exec and is released by the OS on exit or kill.
This changes scheduling only; the validator's executable and arguments remain
unchanged. No sequence or child output is logged by this wrapper.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

SLOTS_PER_GPU = 2


def acquire(directory: Path, devices: list[str], *, slots_per_device=SLOTS_PER_GPU):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    slots = [(device, slot) for slot in range(slots_per_device) for device in devices]
    offset = os.getpid() % len(slots)
    slots = slots[offset:] + slots[:offset]
    while True:
        for device, slot in slots:
            key = hashlib.sha256(device.encode()).hexdigest()[:16]
            fd = os.open(directory / f"device-{key}-slot-{slot}.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            os.set_inheritable(fd, True)
            return fd, device
        time.sleep(.1)


def gpu_devices():
    # Respect an explicit allocation instead of escaping CUDA visibility.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [s.strip() for s in visible.split(",") if s.strip() and s.strip() != "-1"]
    count = int(os.environ.get("NOVA_TNP_GPU_COUNT", "0"))
    if count > 0:
        return [str(i) for i in range(count)]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    except (OSError, subprocess.SubprocessError):
        pass
    return []


def main():
    devices = gpu_devices()
    fd, device = acquire(Path("/tmp/nova-tnp-slots"), devices or ["cpu"])
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    executable = "/root/.pixi/envs/metanano/bin/TNP"
    # Keep fd live/inheritable until exec; a child context manager would close
    # the slot just before TNP starts, silently defeating the global limit.
    os.execv(executable, [executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
