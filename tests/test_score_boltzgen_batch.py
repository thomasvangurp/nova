import csv
import hashlib
import os

import numpy as np

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
    analyze_refolds = design_dir / "refold_analyze_cif"
    folds = design_dir / "fold_out_npz"
    refolds.mkdir(parents=True)
    analyze_refolds.mkdir(parents=True)
    folds.mkdir(parents=True)
    writer_source = refolds / f"{record_id}.cif"
    analyze_source = analyze_refolds / f"{record_id}.cif"
    writer_source.write_bytes(b"writer-complex-cif")
    analyze_source.write_bytes(b"confidence-complex-cif")
    sample_metrics = {
        "iptm": np.asarray([0.90, 0.50]),
        "ptm": np.asarray([0.90, 0.50]),
        "design_iiptm": np.asarray([0.31, 0.42]),
        "design_ptm": np.asarray([0.40, 0.90]),
        "design_to_target_iptm": np.asarray([0.40, 0.90]),
        "min_design_to_target_pae": np.asarray([4.0, 2.0]),
        "interaction_pae": np.asarray([14.0, 10.0]),
    }
    np.savez_compressed(folds / f"{record_id}.npz", **sample_metrics)
    confidence_metrics = {
        name: values[1]
        for name, values in sample_metrics.items()
        if name in {
            "design_iiptm",
            "design_ptm",
            "design_to_target_iptm",
            "min_design_to_target_pae",
            "interaction_pae",
        }
    }
    with (design_dir / "aggregate_metrics_analyze.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "file_name", *confidence_metrics],
        )
        writer.writeheader()
        writer.writerow({
            "id": record_id,
            "file_name": writer_source.name,
            **confidence_metrics,
        })

    copied = _copy_refolded_complexes(
        [sequence], [target], str(tmp_path / "chunk"),
        str(tmp_path / "structures"), "observation-1",
    )

    record = copied[sequence][target]
    assert record["sha256"] == hashlib.sha256(
        b"confidence-complex-cif"
    ).hexdigest()
    assert record["writer_sha256"] == hashlib.sha256(
        b"writer-complex-cif"
    ).hexdigest()
    assert (
        tmp_path / "structures/observation-1" / f"{digest}_{target}.cif"
    ).read_bytes() == b"confidence-complex-cif"
    assert (
        tmp_path
        / "structures/observation-1"
        / f"{digest}_{target}_writer.cif"
    ).read_bytes() == b"writer-complex-cif"
    assert record["sample_selection"]["writer_sample_index"] == 0
    assert record["sample_selection"]["analyze_sample_index"] == 1
    assert record["sample_selection"]["sample_selection_disagrees"] is True
