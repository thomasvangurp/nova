"""Pi preparation and explicit runtime capture for the bounded replay worker.

Capture runs under the scoring interpreter on the execution host. Prepare runs
on Pi using its frozen input bundle and that captured protocol JSON.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.pr100_replay import SETTINGS, TARGET_CONTEXT, atomic, prepare
from tools.pr100_replay_worker import artifact_hash

PACKAGES = ("torch", "numpy", "scipy", "biopython", "rdkit",
            "pytorch-lightning", "omegaconf", "hydra-core")


def capture(spec):
    def stamp(items):
        return {key: {**value, "sha256": artifact_hash(value)} for key, value in items.items()}
    return {"settings": dict(SETTINGS), "subnet_config": dict(TARGET_CONTEXT),
            "seed_policy": "independent_process_stage_seed_v1", "asset_root": spec["asset_root"],
            "source_identity": {"files": stamp(spec["files"])},
            "model_identity": {"artifacts": stamp(spec["artifacts"])},
            "runtime_identity": {"python": platform.python_version(),
                "distributions": {name: importlib.metadata.version(name) for name in PACKAGES}},
            "validator_worker_runtime_parity": "unattested"}


def lease_job(*, python, release, manifest, output, gpu, scorer, deadline, pythonpath, now=None):
    now = time.time() if now is None else now
    ttl = deadline - now
    if not 15 <= ttl <= 1200:
        raise ValueError("lease_must_end_within_1200_seconds")
    return {"gpu": gpu, "ttl_seconds": ttl, "minimum_free_mib": 8192,
            "lock_directory": "/root/pr100-controller/gpu-locks", "scorer": scorer,
            "receipt": str(Path(output) / "lease.json"), "cwd": str(release),
            "command": [str(python), str(Path(release) / "tools/pr100_replay_worker.py"),
                        "--manifest", str(manifest), "--output", str(output),
                        "--deadline-unix", str(deadline - 5)],
            "env": {"PATH": str(Path(python).parent) + ":/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/root", "PYTHONPATH": pythonpath, "PYTHONDONTWRITEBYTECODE": "1",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "NANOBODY_SCORER_DAEMON": "0"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-spec", type=Path)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.capture_spec:
        value = capture(json.loads(a.capture_spec.read_text()))
    else:
        inputs = json.loads(a.inputs.read_text())
        value = prepare(inputs["cohort"], inputs["commitments"], inputs["leaderboard"],
                        inputs["epoch"], json.loads(a.protocol.read_text()))
    atomic(a.out, value)
    print(json.dumps({"status": "prepared", "artifact": str(a.out), "job_id": value.get("job_id")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error_class": type(error).__name__}))
        raise SystemExit(1)
