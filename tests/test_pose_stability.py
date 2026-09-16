import json
from types import SimpleNamespace

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


def _atom(x, y, z, element="C"):
    return SimpleNamespace(
        pos=SimpleNamespace(x=float(x), y=float(y), z=float(z)),
        element=SimpleNamespace(name=element),
    )


class _Residue(list):
    def __init__(self, number, atoms):
        super().__init__(atoms)
        self.seqid = SimpleNamespace(num=number)


def test_vectorized_contacts_match_the_validator_distance_rule():
    target = [
        _Residue(11, [_atom(0, 0, 0), _atom(100, 100, 100, "H")]),
        _Residue(12, [_atom(10, 0, 0)]),
        _Residue(13, [_atom(20, 0, 0)]),
    ]
    design = [
        _Residue(1, [_atom(3, 4, 0)]),       # exactly at the 5A cutoff from 11
        _Residue(2, [_atom(10, 0, 5.001)]),  # just outside the cutoff from 12
        _Residue(3, [_atom(14.9, 0, 0)]),    # contacts 12 but not 13
    ]

    contacts, target_hits, design_hits = pose_stability._contact_sets(
        target, design, 25.0
    )

    assert contacts == {(1, 11), (3, 12)}
    assert target_hits == {11, 12}
    assert design_hits == {1, 3}


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
    structure = tmp_path / "confidence.cif"
    writer_structure = tmp_path / "writer.cif"
    structure.write_text("confidence")
    writer_structure.write_text("writer")
    selection = {
        "writer_sample_index": 2,
        "analyze_sample_index": 4,
        "sample_selection_disagrees": True,
    }
    payload = {
        "target": "P20809",
        "observation_id": "one-sweep",
        "structures": {
            SEQUENCE: {"P20809": {
                "path": str(structure),
                "confidence_path": str(structure),
                "writer_path": str(writer_structure),
                "sha256": "unused",
                "sample_selection": selection,
            }}
        },
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(payload))
    second.write_text(json.dumps(payload))

    found = pose_stability.exact_structure_paths([first, second], "P20809")
    writer_found = pose_stability.exact_structure_paths(
        [first, second], "P20809", pose_kind="writer"
    )
    selections = pose_stability.exact_sample_selections(
        [first, second], "P20809"
    )

    assert found == {SEQUENCE: [structure]}
    assert writer_found == {SEQUENCE: [writer_structure]}
    assert selections == {SEQUENCE: [selection]}
