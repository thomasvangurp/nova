import csv
import hashlib

from nanobody_opt.score_boltzgen_batch import _copy_refolded_complexes


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
