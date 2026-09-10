import json

import numpy as np
import pytest

from nanobody_opt import pose_stability


SEQUENCE = (
    "QWQLVESGGGLVKPGGSLRLSCAASGFTMSQGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPMEGEMESFGAGTAVTVAP"
)


def _feature(translation=0.0, cdr3_shift=0.0, contacts=None):
    target = {
        (1, " "): np.asarray([translation, 0.0, 0.0]),
        (2, " "): np.asarray([translation + 1.0, 0.0, 0.0]),
        (3, " "): np.asarray([translation, 1.0, 0.0]),
    }
    cdr3 = {
        index: np.asarray([translation + index, cdr3_shift, 1.0])
        for index in range(100, 117)
    }
    pairs = set(contacts or {(100, 1), (101, 2)})
    return {
        "target_ca": target,
        "cdr3_ca": cdr3,
        "contacts": pairs,
        "contacted_target": {target for _, target in pairs},
        "contacted_design": {design for design, _ in pairs},
        "epitope_occupancy": 0.5,
    }


def test_pose_summary_aligns_on_target_and_measures_contact_stability(
    tmp_path, monkeypatch
):
    paths = [tmp_path / f"pose-{index}.cif" for index in range(3)]
    for path in paths:
        path.write_text("placeholder")
    features = {
        paths[0].resolve(): _feature(),
        paths[1].resolve(): _feature(translation=10.0),
        paths[2].resolve(): _feature(
            translation=-4.0,
            cdr3_shift=2.0,
            contacts={(100, 1), (102, 3)},
        ),
    }
    monkeypatch.setattr(
        pose_stability,
        "structure_features",
        lambda path, **kwargs: features[path.resolve()],
    )

    summary = pose_stability.summarize_sequence_structures(
        SEQUENCE,
        paths,
        target_chain="A",
        nanobody_chain="B",
        epitope={1, 2, 3, 4},
    )

    assert summary["structure_observation_count"] == 3
    assert summary["cdr3_rmsd_max_angstrom"] == pytest.approx(2.0)
    assert summary["contact_jaccard_min"] == pytest.approx(1 / 3)
    assert summary["epitope_occupancy_min"] == 0.5
    assert summary["consensus_nanobody_contact_positions"] == [100, 101]


def test_exact_structure_paths_deduplicates_one_sweep(tmp_path):
    structure = tmp_path / "complex.cif"
    structure.write_text("complex")
    payload = {
        "target": "P20809",
        "observation_id": "one-sweep",
        "structures": {
            SEQUENCE: {"P20809": {"path": str(structure), "sha256": "unused"}}
        },
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))

    found = pose_stability.exact_structure_paths([first, second], "P20809")

    assert found == {SEQUENCE: [structure]}
