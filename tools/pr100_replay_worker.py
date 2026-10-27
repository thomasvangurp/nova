"""Bounded full-wrapper replay. Run only inside pr100_live_gpu's owned lease.

One fresh process per pass, with all submitted candidates in that pass. Five
internal diffusion samples remain ONE observation, never five passes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

# Preserve the explicitly configured bgen shim ahead of repository utilities.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tools.pr100_replay import (SETTINGS, atomic, digest, file_hash, metrics,
                               observation, pass_seed, validate_manifest)


def artifact_hash(spec):
    path = Path(spec["path"])
    forbidden = re.compile(r"(?:^|[_-])(?:credentials?|secrets?|passwords?|cookies?|tokens?|sessions?|login|authorization)(?:[_.-]|$)|application.state", re.I)
    if any(part in {".env", ".netrc", ".npmrc", ".aws", ".ssh", ".git"}
           or part.startswith(".env.") or forbidden.search(part) for part in path.parts):
        raise ValueError("sensitive_artifact_path_refused")
    if spec.get("kind", "file") == "file":
        return file_hash(path)
    if spec.get("kind") != "tree" or not path.is_dir():
        raise ValueError("invalid_artifact_type")
    # Prune sensitive/hidden directories before traversal, and allowlist source
    # and scientific formats before reading bytes. Never inventory HOME/state.
    suffixes = {".py", ".yaml", ".yml", ".json", ".npz", ".npy", ".cif", ".pdb",
                ".csv", ".a3m", ".fasta", ".fa", ".ckpt", ".pt", ".pth", ".zip", ".pkl", ".bin"}
    files = []
    for directory, names, filenames in os.walk(path, topdown=True, followlinks=False):
        names[:] = [n for n in names if not n.startswith(".") and n not in
                    {"__pycache__", "node_modules", "state", "secrets", "credentials"}
                    and not forbidden.search(n)]
        for name in filenames:
            item = Path(directory) / name
            if not name.startswith(".") and item.suffix in suffixes and not forbidden.search(name):
                files.append(item)
    files.sort()
    if not files or any(p.is_symlink() for p in files):
        raise ValueError("invalid_artifact_tree")
    return digest({str(p.relative_to(path)): file_hash(p) for p in files})


def verify_runtime(m):
    validate_manifest(m)
    p = m["protocol"]
    runtime = p["runtime_identity"]
    if runtime["python"] != platform.python_version():
        raise ValueError("python_identity_mismatch")
    required_packages = {"torch", "numpy", "scipy", "biopython", "rdkit",
                         "pytorch-lightning", "omegaconf", "hydra-core"}
    if not required_packages <= set(runtime["distributions"]):
        raise ValueError("runtime_package_identity_incomplete")
    for name, version in runtime["distributions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("package_identity_mismatch")
    files = p["source_identity"]["files"]
    required = {"wrapper", "cli", "main", "fold", "design", "boltzgen_config",
                "target_structure", "target_msa", "boltzgen_source", "replay_worker", "replay_records"}
    if not required <= set(files) or not p["model_identity"]["artifacts"]:
        raise ValueError("runtime_artifact_identity_incomplete")
    for spec in [*files.values(), *p["model_identity"]["artifacts"].values()]:
        if artifact_hash(spec) != spec["sha256"]:
            raise ValueError("runtime_artifact_hash_mismatch")
    if (file_hash(__file__) != files["replay_worker"]["sha256"] or
            file_hash(sys.modules[observation.__module__].__file__) != files["replay_records"]["sha256"]):
        raise ValueError("replay_source_identity_mismatch")
    import yaml
    fold = yaml.safe_load(Path(files["fold"]["path"]).read_text())
    design = yaml.safe_load(Path(files["design"]["path"]).read_text())
    bg = yaml.safe_load(Path(files["boltzgen_config"]["path"]).read_text())
    if fold.get("sampling_steps") != 200 or fold.get("diffusion_samples") != 5 or design.get("sampling_steps") != 500:
        raise ValueError("fast_settings_rejected")
    for key in ("protocol", "step_scale", "noise_scale", "num_designs", "skip_inverse_folding"):
        if bg.get(key) != SETTINGS[key]:
            raise ValueError("wrapper_settings_mismatch")
    if bg.get("no_subprocess") or bg.get("execute_steps") != ["design", "folding", "design_folding", "analysis"]:
        raise ValueError("wrapper_execution_policy_mismatch")
    return files


def check_generated(output, model_paths):
    """Verify effective configuration after configure, before any inference."""
    import yaml
    output = Path(output)
    manifest = yaml.safe_load((output / "steps.yaml").read_text())
    steps = {s["name"]: yaml.safe_load((output / s["config_file"]).read_text())
             for s in manifest["steps"]}
    if not {"design", "folding", "analysis"} <= set(steps):
        raise ValueError("missing_full_pipeline_stage")
    for name, count, samples in (("design", 500, 1), ("folding", 200, 5)):
        config = steps[name]
        if config.get("sampling_steps") != count or config.get("diffusion_samples") != samples:
            raise ValueError("generated_fast_settings_rejected")
        if str(Path(config["checkpoint"]).resolve()) not in model_paths:
            raise ValueError("unregistered_generated_checkpoint")
    return {name: file_hash(output / s["config_file"])
            for s in manifest["steps"] for name in [s["name"]]}


def child(m, index, work):
    """In-memory path/seed hooks only; installed scientific files stay unchanged."""
    validate_manifest(m)
    p = m["protocol"]; files = p["source_identity"]["files"]
    from boltzgen import boltzgen_wrapper as wrapper_module
    from boltzgen.cli import boltzgen as cli
    for module, role in ((wrapper_module, "wrapper"), (cli, "cli")):
        if (Path(module.__file__).resolve() != Path(files[role]["path"]).resolve()
                or file_hash(module.__file__) != files[role]["sha256"]):
            raise ValueError("loaded_source_identity_mismatch")
    assets = p["model_identity"]["artifacts"]
    def pinned_artifact(args, key, repo_type="model", verbose=False):
        key = next((name for name, (uri, _) in cli.ARTIFACTS.items() if uri == key), key)
        if key not in assets:
            raise ValueError("unregistered_model_or_molecule_asset")
        return Path(assets[key]["path"])
    cli.get_artifact_path = pinned_artifact
    wrapper_module.BOLTZGEN_TMP_FILES_DIR = str(work / "wrapper")
    wrapper_module.BOLTZGEN_CONFIG_FILE = files["boltzgen_config"]["path"]
    wrapper_module.NOVA_DIR = p["asset_root"]
    expected_structure = Path(p["asset_root"]) / "data/structures/8AOK.cif"
    expected_msa = Path(p["asset_root"]) / "data/msa_files/Q9NZQ7.a3m"
    if (Path(files["target_structure"]["path"]).resolve() != expected_structure.resolve()
            or Path(files["target_msa"]["path"]).resolve() != expected_msa.resolve()):
        raise ValueError("target_asset_path_mismatch")
    # Each actual stage subprocess seeds Python/NumPy/Torch before invoking the
    # unmodified upstream main. Separate pass and stage seeds prevent inherited
    # RNG state from creating duplicate draws. Numerical determinism is NOT claimed.
    base_seed = pass_seed(m, index)
    seed_launcher = work / "seeded_stage.py"
    seed_launcher.write_text(
        "import hashlib,json,pathlib,random,runpy,sys\n"
        "import numpy as np\nimport torch\n"
        f"base={base_seed!r}\n"
        "stage=pathlib.Path(sys.argv[1]).stem\n"
        "seed=int(hashlib.sha256(f'{base}:{stage}'.encode()).hexdigest()[:8],16)\n"
        "random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)\n"
        "pathlib.Path(sys.argv[1]+'.seed.json').write_text(json.dumps({'stage':stage,'seed':seed}))\n"
        f"runpy.run_path({files['main']['path']!r},run_name='__main__')\n")
    cli.main_script = seed_launcher
    original = wrapper_module.configure_command
    configured = {}
    def configure(args):
        result = original(args)
        configured.update(check_generated(args.output, {
            str(Path(s["path"]).resolve()) for s in assets.values() if s.get("kind", "file") == "file"}))
        return result
    wrapper_module.configure_command = configure
    config = p["subnet_config"]
    if config["nanobody_target"] != [m["target"]]:
        raise ValueError("target_config_mismatch")
    instance = wrapper_module.BoltzgenWrapper()
    by_uid = {c["uid"]: {"sequences": [c["sequence"]]} for c in m["cohort"]}
    components = instance.run_nanobody_inference(by_uid, config)
    output_rows = []
    for candidate in m["cohort"]:
        values = metrics(components[candidate["uid"]][candidate["sequence"]][m["target"]])
        output_rows.append({"uid": candidate["uid"], "metrics": values})
    # All structures/configurations stay in this unique pass directory.
    scientific_files = {str(f.relative_to(work)): file_hash(f) for f in (work / "wrapper").rglob("*")
                        if f.is_file() and f.suffix in {".cif", ".npz", ".yaml", ".json", ".csv"}}
    if not {"design", "folding", "analysis"} <= set(configured) or not any(
            name.endswith(".cif") for name in scientific_files):
        raise ValueError("full_pipeline_artifact_receipts_missing")
    stage_seeds = {}
    for path in (work / "wrapper").rglob("*.seed.json"):
        value = json.loads(path.read_text()); stage = value["stage"]
        expected = int(hashlib.sha256(f"{base_seed}:{stage}".encode()).hexdigest()[:8], 16)
        if value["seed"] != expected:
            raise ValueError("stage_seed_receipt_mismatch")
        stage_seeds[stage] = expected
    if not {"design", "folding", "analysis"} <= set(stage_seeds):
        raise ValueError("independent_stage_seed_receipts_missing")
    artifact_identity = digest(scientific_files)
    result = {"pass_index": index, "seed": base_seed, "stage_seeds": stage_seeds, "generated_configs": configured,
              "artifact_files": scientific_files, "artifact_sha256": artifact_identity,
              "observations": [observation(m, row["uid"], index, row["metrics"],
                                artifact_sha256=artifact_identity) for row in output_rows]}
    atomic(work / "result.json", result)
    return result


def run(m, manifest_path, output, deadline, *, launch=subprocess.Popen, clock=time.time):
    """Checkpoint per complete pass; an expired/failed pass never supplies rows.

    The existing GPU lease owns the full process tree and provides SIGKILL
    cleanup even if a scientific child ignores termination.
    """
    validate_manifest(m)
    if deadline <= clock() or deadline - clock() > 1800:
        raise ValueError("bounded_deadline_required")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    verify_runtime(m)
    outcomes = []
    for index in range(1, m["passes_required"] + 1):
        work = output / f"pass-{index:02d}"
        result_path = work / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            expected_uids = {c["uid"] for c in m["cohort"]}
            if {r["uid"] for r in result["observations"]} != expected_uids or len(result["observations"]) != len(expected_uids):
                raise ValueError("partial_or_duplicate_cached_pass")
            for row in result["observations"]:
                if row != observation(m, row["uid"], index, row["metrics"], artifact_sha256=result["artifact_sha256"]):
                    raise ValueError("incompatible_cached_pass")
            outcomes.append({"pass_index": index, "status": "already_complete"}); continue
        if clock() >= deadline - 10:
            break
        # Never reuse partial scientific output. A retry has a new work path.
        work = output / f"pass-{index:02d}-attempt-{time.time_ns()}"
        work.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(manifest_path),
                   "--output", str(work), "--child-pass", str(index)]
        process = launch(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            code = process.wait(timeout=max(.01, deadline - clock() - 5))
        except BaseException:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait(timeout=2)
            raise
        if code or not (work / "result.json").is_file():
            outcomes.append({"pass_index": index, "status": "failed", "returncode": code}); break
        work.rename(output / f"pass-{index:02d}")
        outcomes.append({"pass_index": index, "status": "complete"})
        atomic(output / "progress.json", {"job_id": m["job_id"], "passes": outcomes})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--child-pass", type=int)
    a = parser.parse_args()
    m = json.loads(a.manifest.read_text())
    if a.child_pass is not None:
        child(m, a.child_pass, a.output)
    else:
        run(m, a.manifest.resolve(), a.output.resolve(), a.deadline_unix)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        # Scientific tool errors/streams may contain arbitrary runtime state.
        # Keep the orchestration interface explicitly allowlisted.
        print(json.dumps({"status": "failed", "error_class": type(error).__name__}), flush=True)
        raise SystemExit(1)
