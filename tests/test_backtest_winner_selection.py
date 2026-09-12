import json

import pytest

from nanobody_opt import backtest_winner_selection as bw
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL

METRICS = {
    "design_iiptm": 0.78, "design_ptm": 0.88, "design_to_target_iptm": 0.68,
    "min_design_to_target_pae": 2.3, "interaction_pae": 10.5,
    "plip_hbonds_refolded": 4.0, "plip_saltbridge_refolded": 2.0,
    "delta_sasa_refolded": 730.0, "liability_score": 265.0,
    "liability_num_violations": 29.0,
}


def _nanobody(sequence, score, **overrides):
    record = {"protein_name": "P20809", "sequence": sequence,
              "final_score": score, **METRICS}
    record.update(overrides)
    return record


def _field_payload():
    return {"success": True, "data": [
        {"uid": 1, "nanobodies": [_nanobody("WINNER", 51.0, design_iiptm=0.80)]},
        {"uid": 8, "nanobodies": [_nanobody("OURS", 190.0, design_iiptm=0.77)]},
        {"uid": 2, "nanobodies": [_nanobody("OTHER", 120.0, design_iiptm=0.78)]},
    ]}


def test_winner_is_the_lowest_score(tmp_path):
    sequence, score, uid = bw.winner_of(_field_payload(), "P20809")
    assert (sequence, score, uid) == ("WINNER", 51.0, 1)


def test_our_own_entry_is_excluded_from_the_field():
    field = bw.competitor_field(_field_payload(), "P20809", exclude_uid=8)
    # two competitors remain, not three
    assert len(field) == 2


def test_prepare_writes_the_winner_sequence(tmp_path, capsys):
    import sys

    fields = tmp_path / "fields"
    fields.mkdir()
    (fields / "epoch25052.json").write_text(json.dumps(_field_payload()))
    out = tmp_path / "winner.json"
    argv = ["prog", "prepare", "--epoch", "25052", "--field-dir", str(fields),
            "--output", str(out)]
    old = sys.argv
    sys.argv = argv
    try:
        bw.main()
    finally:
        sys.argv = old

    assert json.loads(out.read_text())["sequences"] == ["WINNER"]
    assert "published_score=51" in capsys.readouterr().out


def test_observations_are_read_from_exact_artifacts(tmp_path):
    path = tmp_path / "exact_all.json"
    path.write_text(json.dumps({
        "target": "P20809", "scorer_protocol": SCORER_PROTOCOL, "n": 1,
        "per_nanobody_components": {"0": {"WINNER": {"P20809": METRICS}}},
    }))

    found = bw.observations_from([path], "P20809")

    assert list(found) == ["WINNER"]
    assert found["WINNER"][0]["design_iiptm"] == 0.78


def test_a_missing_field_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no cached field"):
        bw._payload(tmp_path, 25052)
