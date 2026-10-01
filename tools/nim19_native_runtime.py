#!/usr/bin/env python3
"""Run an unmodified extracted NIM image on a disposable Vast GPU container.

PRoot supplies the image filesystem; only NVIDIA driver files and Linux kernel
interfaces come from the rental. Never extract image layers over the host root.
Server output is drained and classified, never persisted or printed verbatim.
"""
import argparse
import glob
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import sys
import threading
import time
import urllib.request

BASE = Path("/root/nim19-test")
ROOT = BASE / "bundle/rootfs"
STATUS = BASE / "server-status.json"
DIRECT = False
EXPECTED_GPUS = 1


def guest_path(path):
    """Resolve image-absolute symlinks inside the image, never through the host."""
    parts = Path(path).parts[1:]
    for _ in range(32):
        current = ROOT
        for index, part in enumerate(parts):
            current = current / part
            if current.is_symlink():
                link = os.readlink(current)
                parent = "/" if link.startswith("/") else "/"+"/".join(parts[:index])
                target = posixpath.normpath(posixpath.join(parent, link, *parts[index+1:]))
                parts = Path(target).parts[1:]
                break
        else:
            return current
    raise RuntimeError("symlink_cycle")


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def command(argv):
    if DIRECT:
        return [str(guest_path(argv[0])), *argv[1:]]
    args = ["proot", "-r", str(ROOT), "-w", "/opt/nim"]
    for path in ["/dev", "/proc", "/sys"]:
        args += ["-b", path]
    for source, dest in [
        (BASE / "models", "/opt/nim/custom-model"),
        (BASE / "cache", "/opt/nim/.cache"),
        (BASE / "output", "/opt/nim/output"),
    ]:
        args += ["-b", str(source) + ":" + dest]
    driver_files = set()
    for pattern in ["/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/lib/x86_64-linux-gnu/libnvidia*.so*", "/usr/local/nvidia/lib64/libcuda.so*", "/usr/local/nvidia/lib64/libnvidia*.so*"]:
        driver_files.update(glob.glob(pattern))
    if os.path.isfile("/usr/bin/nvidia-smi"):
        driver_files.add("/usr/bin/nvidia-smi")
    for path in sorted(driver_files):
        if os.path.isfile(path):
            args += ["-b", path + ":" + path]
    return args + argv


def environment():
    config = json.loads((BASE / "image-runtime.json").read_text())
    # Do not forward unrelated rental/application credentials into NIM.
    env = {"HOME": "/root", "LANG": "C.UTF-8"}
    public = {"PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "PYTHONPATH", "PYTHONHOME",
              "ENV", "BASH_ENV", "SHELL", "SERVER_START_SCRIPT_PATH", "BACKEND_TYPE",
              "PYTHONIOENCODING", "PYTHONDONTWRITEBYTECODE", "PYTORCH_HOME",
              "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "CUDA_MODULE_LOADING",
              "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUDA_BINARY_LOADER_THREAD_COUNT"}
    public.add("OPAL_PREFIX")
    for value in config.get("Env", []):
        name, _, content = value.partition("=")
        public_nim = name.startswith("NIM_") and not any(x in name for x in ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "TELEMETRY_ENDPOINT"])
        if name in public or public_nim or name == "NIMTOOLS_VERSION" or name.startswith("TRITON_"):
            env[name] = content
    # Prefer the actual host driver, not forward-compatibility libraries.
    env["LD_LIBRARY_PATH"] = ":".join(x for x in env.get("LD_LIBRARY_PATH", "").split(":") if x and "/compat" not in x)
    env["LD_LIBRARY_PATH"] += ":/usr/lib/x86_64-linux-gnu"
    env.update(MODEL_PATH="/opt/nim/custom-model", NIM_DISABLE_MODEL_DOWNLOAD="1",
               NIM_DEFAULT_RANDOM_SEED="68", NIM_TELEMETRY_MODE="0",
               NIM_LOG="WARNING", NIM_LOG_LEVEL="WARNING",
               NIM_HTTP_API_PORT="8000", NIM_OUTPUT_PATH="/opt/nim/output",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if DIRECT:
        dirs = ["/usr/lib/x86_64-linux-gnu", "/usr/local/lib",
                "/usr/local/lib/python3.12/dist-packages/torch/lib",
                "/usr/local/cuda-13.2/targets/x86_64-linux/lib",
                "/usr/local/mpi/lib", "/usr/local/ucx/lib", "/opt/amazon/efa/lib",
                "/usr/lib/x86_64-linux-gnu/nvshmem/13"]
        for conf in (ROOT / "etc/ld.so.conf.d").glob("*.conf"):
            for line in conf.read_text().splitlines():
                line = line.strip()
                if re.fullmatch(r"/[A-Za-z0-9_/.-]{1,160}", line) and "/compat" not in line:
                    dirs.append(line)
        dirs.extend("/"+str(p.relative_to(ROOT)) for p in (ROOT / "usr/local/lib/python3.12/dist-packages/nvidia").glob("*/lib"))
        # Host driver libraries must win over image compatibility/stub libraries.
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:"+":".join(dict.fromkeys(str(guest_path(p)) for p in dirs if guest_path(p).is_dir()))
        env["PYTHONHOME"] = str(ROOT / "usr")
        env["PYTHONPATH"] = str(ROOT / "usr/local/lib/python3.12/dist-packages")
        env["CUDA_HOME"] = str(guest_path("/usr/local/cuda"))
        env["PATH"] = ":".join(str(guest_path(p)) for p in env["PATH"].split(":") if p.startswith("/"))+":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        if env.get("OPAL_PREFIX"):
            env["OPAL_PREFIX"] = str(guest_path(env["OPAL_PREFIX"]))
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
    return env


def prepare():
    for path in [BASE / "cache", BASE / "output"]:
        path.mkdir(exist_ok=True)
    if not (ROOT / "opt/nim/start_server.sh").is_file():
        raise RuntimeError("missing_image")
    missing = [name for name in ["boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols.tar"] if not (BASE / "models" / name).is_file()]
    if missing:
        raise RuntimeError("missing_checkpoints")
    if DIRECT:
        nim = Path('/opt/nim')
        if nim.exists() or nim.is_symlink():
            if not nim.is_symlink() or nim.resolve() != ROOT / 'opt/nim':
                raise RuntimeError('host_nim_collision')
        else:
            nim.symlink_to(ROOT / 'opt/nim', target_is_directory=True)
        for name, target in [('custom-model', BASE/'models'), ('.cache', BASE/'cache'), ('output', BASE/'output')]:
            link = ROOT / 'opt/nim' / name
            if link.is_symlink():
                if link.resolve() != target:
                    raise RuntimeError('runtime_path_collision')
            elif link.exists():
                if not link.is_dir() or any(link.iterdir()):
                    raise RuntimeError('runtime_path_collision')
                link.rmdir()  # Empty image mount point only; no scientific data.
                link.symlink_to(target, target_is_directory=True)
            else:
                link.symlink_to(target, target_is_directory=True)


def probe():
    prepare()
    script = (
        "import json,time,torch;start=time.perf_counter();available=torch.cuda.is_available();"
        "x=torch.ones((64,64),device='cuda') if available else None;"
        "y=x@x if x is not None else None;"
        "torch.cuda.synchronize() if available else None;"
        "print('__NIM_PROBE__'+json.dumps({'passed':bool(y is not None and y[0,0].item()==64),'elapsed_s':time.perf_counter()-start,"
        "'torch_version':torch.__version__,'cuda_version':torch.version.cuda,"
        "'gpu_visible':torch.cuda.is_available(),'gpus':torch.cuda.device_count()}))"
    )
    result = subprocess.run(command(["/usr/bin/python3.12", "-c", script]), env=environment(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    for line in result.stdout.splitlines():
        if line.startswith(b"__NIM_PROBE__"):
            data = json.loads(line[len(b"__NIM_PROBE__"):])
            safe = {key: value for key, value in data.items() if key in ["torch_version", "cuda_version", "gpu_visible", "gpus", "passed", "elapsed_s"]}
            emit("native_runtime", **safe, exit_code=result.returncode)
            return 0 if data.get("passed") is True and data.get("gpu_visible") and data.get("gpus") == EXPECTED_GPUS else 1
    classes = [name for name in ["ModuleNotFoundError", "ImportError", "RuntimeError", "PermissionError"] if name.encode() in result.stderr]
    emit("native_runtime", runtime_available=False, exit_code=result.returncode, error_classes=classes)
    return 1


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def process_start_ticks(pid):
    try:
        # Numeric identity only. Never inspect or expose command-line arguments.
        return int(Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/health/ready", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def supervise():
    prepare()
    started = time.monotonic()
    process = subprocess.Popen(command(["/bin/bash", "/opt/nim/start_server.sh"]), env=environment(),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    errors = set()
    stages = set()
    lock = threading.Lock()

    def drain():
        for line in process.stdout:
            # Only static classification labels survive. No raw traces/URLs.
            with lock:
                for name in ["ModuleNotFoundError", "ImportError", "RuntimeError", "PermissionError", "ValueError", "TypeError", "CUDA out of memory", "NGC_API_KEY", "No space left on device", "FileNotFoundError", "unbound variable", "command not found", "No such file or directory"]:
                    if name.encode() in line:
                        errors.add(name)
                for name in ['checkpoint', 'mols.tar', 'warmup', 'startup', 'ready', 'Downloading', 'profile', 'custom', 'license']:
                    if name.lower().encode() in line.lower():
                        stages.add(name)

    threading.Thread(target=drain, daemon=True).start()
    first_ready = None
    while True:
        ready = health()
        if ready and first_ready is None:
            first_ready = time.monotonic()-started
        with lock:
            classified = sorted(errors)
            progress = sorted(stages)
        status = {"supervisor_pid": os.getpid(), "server_pid": process.pid, "boot_id": boot_id(), "supervisor_start_ticks": process_start_ticks(os.getpid()),
                  "ready": ready, "elapsed_seconds": round(time.monotonic()-started, 2),
                  "cold_start_seconds": first_ready, "exit_code": process.poll(),
                  "error_classes": classified, "startup_stages": progress,
                  "execution_mode": "direct" if DIRECT else "proot"}
        temporary = STATUS.with_suffix(".partial")
        temporary.write_text(json.dumps(status))
        temporary.replace(STATUS)
        if process.poll() is not None:
            return process.returncode
        time.sleep(2)


def main():
    global DIRECT, EXPECTED_GPUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["probe", "direct-probe", "start", "direct-start", "status", "supervise", "direct-supervise"])
    parser.add_argument("--expected-gpus", type=int, choices=[1,8], default=1)
    args = parser.parse_args()
    EXPECTED_GPUS = args.expected_gpus
    DIRECT = args.operation.startswith('direct-')
    operation = args.operation.removeprefix('direct-')
    try:
        if operation == "probe":
            return probe()
        if operation == "supervise":
            return supervise()
        if operation == "start":
            if STATUS.exists():
                state = json.loads(STATUS.read_text())
                if state.get("boot_id") == boot_id() and state.get("supervisor_start_ticks") is not None and state.get("supervisor_start_ticks") == process_start_ticks(state["supervisor_pid"]) and state.get("exit_code") is None and alive(state["supervisor_pid"]):
                    emit("started", already_running=True)
                    return 0
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "direct-supervise" if DIRECT else "supervise"],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            emit("started", supervisor_pid=process.pid)
            return 0
        if operation == "status":
            state = json.loads(STATUS.read_text()) if STATUS.exists() else {}
            emit("service", **state, ready_now=health())
            return 0
    except Exception as exc:
        emit("operation_failed", error_class=type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
