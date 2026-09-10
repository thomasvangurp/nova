import csv
import hashlib
import os

from nanobody_opt.score_boltzgen_batch import (
    _configure_diagnostic_determinism,
    _copy_refolded_complexes,
)


def test_diagnostic_seed_configures_pipeline_subprocesses(monkeypatch):
    for name in (
        "BOLTZGEN_DIAGNOSTIC_SEED",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)

    _configure_diagnostic_determinism(25037)

    assert os.environ["BOLTZGEN_DIAGNOSTIC_SEED"] == "25037"
    assert os.environ["PYTHONHASHSEED"] == "25037"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"


def test_refolded_complex_is_copied_before_work_tree_cleanup(tmp_path):
    sequence = "QWQLVESGG"
    target = "P20809"
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    record_id = f"{digest}_{target}_input"
    design_dir = tmp_path / "chunk/outputs/intermediate_designs"
    refolds = design_dir / "refold_cif"
    refolds.mkdir(parents=True)
    source = refolds / f"{record_id}.cif"
    source.write_bytes(b"complex-cif")
    with (design_dir / "aggregate_metrics_analyze.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "file_name"])
        writer.writeheader()
        writer.writerow({"id": record_id, "file_name": source.name})

    copied = _copy_refolded_complexes(
        [sequence], [target], str(tmp_path / "chunk"),
        str(tmp_path / "structures"), "observation-1",
    )

    record = copied[sequence][target]
    assert record["sha256"] == hashlib.sha256(b"complex-cif").hexdigest()
    assert (tmp_path / "structures/observation-1" / f"{digest}_{target}.cif").read_bytes() == b"complex-cif"
