from pathlib import Path

import pytest

from nanobody_opt.build_design_spec import binding_site_from_summary, build_spec


def _config():
    return {
        "protein_selection": {
            "nanobody": {
                "target": "P20809",
                "structure": "6O4O",
                "structure_chain": "A",
                "structure_res_index": "2..169",
                "structure_binding_site": "31..35,37..38,40",
            }
        }
    }


def test_build_spec_uses_every_live_challenge_field():
    target, spec = build_spec(
        _config(),
        structure_dir=Path("/structures"),
        msa_dir=Path("/msa"),
        scaffold_paths=[Path("/scaffolds/a.yaml"), Path("/scaffolds/b.yaml")],
    )

    assert target == "P20809"
    target_file, scaffold_file = [item["file"] for item in spec["entities"]]
    assert target_file == {
        "path": "/structures/6O4O.cif",
        "include": [
            {
                "chain": {
                    "id": "A",
                    "res_index": "2..169",
                    "msa": "/msa/P20809.a3m",
                }
            }
        ],
        "binding_types": [
            {"chain": {"id": "A", "binding": "31..35,37..38,40"}}
        ],
    }
    assert scaffold_file["path"] == [
        "/scaffolds/a.yaml",
        "/scaffolds/b.yaml",
    ]


def test_build_spec_rejects_an_incomplete_live_challenge():
    config = _config()
    del config["protein_selection"]["nanobody"]["structure_binding_site"]

    with pytest.raises(ValueError, match="structure_binding_site"):
        build_spec(
            config,
            structure_dir=Path("/structures"),
            msa_dir=Path("/msa"),
            scaffold_paths=[Path("/scaffolds/a.yaml")],
        )


def test_teacher_consensus_can_replace_only_the_generation_epitope():
    config = _config()
    summary = {
        "target": "P20809",
        "structure": "6O4O",
        "teacher_count": 10,
        "recommended_binding_site": "11,14..15,94..95",
    }
    target, spec = build_spec(
        config,
        structure_dir=Path("/structures"),
        msa_dir=Path("/msa"),
        scaffold_paths=[Path("/scaffolds/a.yaml")],
        binding_site=binding_site_from_summary(config, summary),
    )

    assert target == "P20809"
    assert spec["entities"][0]["file"]["binding_types"] == [
        {"chain": {"id": "A", "binding": "11,14..15,94..95"}}
    ]


def test_teacher_consensus_must_match_the_live_target():
    summary = {
        "target": "WRONG",
        "structure": "6O4O",
        "teacher_count": 10,
        "recommended_binding_site": "11",
    }
    with pytest.raises(ValueError, match="does not match live challenge"):
        binding_site_from_summary(_config(), summary)
