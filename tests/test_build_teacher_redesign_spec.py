import json
from pathlib import Path

import pytest

from nanobody_opt.build_teacher_redesign_spec import (
    build_spec,
    cdr_ranges,
    contact_positions,
    locate_retained_complex,
    redesign_ranges,
)


def test_cdr_ranges_are_one_based_inclusive():
    sequence = "AAAACCCCGGGGTTTT"
    assert cdr_ranges(sequence, ["CCCC", "GG", "TTT"]) == "5..8,9..10,13..15"


def test_cdr_ranges_can_preserve_observed_contacts():
    sequence = "AAAACCCCGGGGTTTT"
    assert cdr_ranges(
        sequence,
        ["CCCC", "GG", "TTT"],
        excluded_positions={6, 9, 14},
    ) == "5,7..8,10,13,15"


def test_loads_contact_sequence_indices(tmp_path: Path):
    report = tmp_path / "contacts.json"
    report.write_text(json.dumps({
        "interface_residues": [
            {"sequence_index": 33},
            {"sequence_index": 102},
        ]
    }))
    assert contact_positions(report) == {33, 102}


def test_locates_exactly_one_retained_teacher(tmp_path: Path):
    sequence = "ACDE"
    (tmp_path / "out0.json").write_text(json.dumps({
        "shard": 3,
        "components": {"0": {sequence: {"P": {}}}},
    }))
    refolds = (
        tmp_path / "shard3" / "chunk0" / "outputs"
        / "intermediate_designs" / "refold_cif"
    )
    refolds.mkdir(parents=True)
    expected = refolds / "teacher.cif"
    expected.write_text("data_teacher\n")

    assert locate_retained_complex(tmp_path, sequence) == (expected, 3)


def test_rejects_missing_teacher(tmp_path: Path):
    with pytest.raises(ValueError, match="maps to 0"):
        locate_retained_complex(tmp_path, "ACDE")


def test_builds_cdr_only_design_mask():
    got = build_spec(Path("/teacher.cif"), "A", "B", "26..35,53..61,100..116")
    assert got == {
        "entities": [{
            "file": {
                "path": "/teacher.cif",
                "include": [
                    {"chain": {"id": "A"}},
                    {"chain": {"id": "B"}},
                ],
                "design": [{
                    "chain": {
                        "id": "B",
                        "res_index": "26..35,53..61,100..116",
                    }
                }],
            }
        }]
    }


def test_framework_scope_preserves_cdrs_contacts_and_cysteines():
    sequence = "AACDEFGHCIKLMN"
    assert redesign_ranges(
        sequence,
        ["DEF", "H", "LM"],
        scope="framework",
        excluded_positions={2, 11},
    ) == "1,7,10,14"


def test_cdr_scope_preserves_observed_contacts():
    assert redesign_ranges(
        "AAAACCCCGGGGTTTT",
        ["CCCC", "GG", "TTT"],
        scope="cdr",
        excluded_positions={6, 9, 14},
    ) == "5,7..8,10,13,15"
