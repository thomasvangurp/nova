"""De novo recombination bank converter: pure, no annotation/network at import."""
import json

import pytest

from nanobody_opt import pr100_reference_donors as donors
from nanobody_opt import rules
from tools import de_novo_bank


def anno(gene, length):
    return {"cdr3_length": length, "germline": {"V": {"genes": [gene]}}}


def anno_genes(genes, length):
    return {"cdr3_length": length, "germline": {"V": {"genes": genes}}}


def annotator(mapping):
    """Return an annotation stub keyed by sequence -> annotation dict or Exception."""
    def annotate(sequence):
        value = mapping[sequence.strip().upper()]
        if isinstance(value, Exception):
            raise value
        return value
    return annotate


def challenge(gene="IGHV3-48", length=17, epoch=26000):
    return {"epoch": epoch, "target": rules.TARGET,
            "allowed_v": gene, "allowed_cdr3_length": length}


def test_convert_filters_annotates_and_dedups():
    mapping = {
        "S1": anno("IGHV3-23", 16),
        "S2": anno("IGHV3-66", 18),
        "S3": anno("IGHV1-2", 16),       # outside the 9-gene set
        "S4": anno("IGHV3-23", 24),      # CDR3 too long
        "S5": anno("IGHV3-23", 4),       # CDR3 too short
        "S6": ValueError("no domain"),
    }
    sequences = ["S1", "S1", "S2", "S3", "S4", "S5", "S6"]
    rows, summary = de_novo_bank.convert(
        sequences, source="de_novo_recombine_test", annotate=annotator(mapping))

    assert summary["input"] == 7
    assert summary["kept"] == 2
    assert summary["dedup_dropped"] == 1
    assert summary["annotation_failed"] == 1
    assert summary["no_allowed_v_gene"] == 1
    assert summary["cdr3_length_out_of_range"] == 2
    assert dict(summary["v_gene_distribution"]) == {"IGHV3-23": 1, "IGHV3-66": 1}
    assert dict(summary["cdr3_length_distribution"]) == {16: 1, 18: 1}

    assert [r["sequence"] for r in rows] == ["S1", "S2"]
    assert rows[0] == {"cells": ["IGHV3-23:16"], "sequence": "S1",
                       "sequence_hash": rules.sequence_hash("S1"),
                       "source": "de_novo_recombine_test"}
    assert rows[1] == {"cells": ["IGHV3-66:18"], "sequence": "S2",
                       "sequence_hash": rules.sequence_hash("S2"),
                       "source": "de_novo_recombine_test"}
    assert de_novo_bank.validate(rows) is None


def test_ambiguous_v_genes_become_multiple_sorted_cells():
    rows, summary = de_novo_bank.convert(
        ["AMB"], source="s", annotate=annotator({"AMB": anno_genes(["IGHV3-7", "IGHV3-23"], 16)}))
    assert rows[0]["cells"] == ["IGHV3-23:16", "IGHV3-7:16"]
    assert summary["kept"] == 1


def test_every_kept_row_is_accepted_by_reference_donor_select():
    mapping = {
        "A1": anno("IGHV3-23", 16),
        "A2": anno("IGHV3-66", 18),
        "A3": anno("IGHV3-7", 5),
        "A4": anno("IGHV3-74", 21),
        "A5": anno("IGHV3-23", 23),
        "A6": anno("IGHV3-64", 12),
    }
    rows, _ = de_novo_bank.convert(
        list(mapping), source="s", annotate=annotator(mapping))
    # A challenge with no exact cell forces the fallback path, which validates
    # every row's identity and cells before selecting.
    exact, chosen, evidence = donors.select(
        rows, challenge(gene="IGHV3-48", length=17), 16, fallback=True)
    assert exact == [] and evidence["fallback_used"]
    assert {r["sequence_hash"] for r, _ in chosen} == {r["sequence_hash"] for r in rows}
    # And an exact cell match still selects the right rows by their cells.
    exact, _, _ = donors.select(rows, challenge(gene="IGHV3-23", length=16), 16)
    assert {r["sequence"] for r in exact} == {"A1"}


@pytest.mark.parametrize("change", [
    {"sequence_hash": "f" * 64},
    {"cells": []},
    {"cells": ["IGHV3-23:99"]},
    {"cells": ["IGHV1-2:16"]},
    {"source": ""},
])
def test_validate_rejects_invalid_rows(change):
    good = {"cells": ["IGHV3-23:16"], "sequence": "X",
            "sequence_hash": rules.sequence_hash("X"), "source": "s"}
    good.update(change)
    with pytest.raises(ValueError):
        de_novo_bank.validate([good])


def test_cli_writes_schema_exact_bank_and_summary(tmp_path, monkeypatch):
    candidates = tmp_path / "recombinants.csv"
    candidates.write_text(
        "sequence,max_jaccard_to_top50,proxy_rank_sum,cdr_sweep,exhaustive_cdr,cdr_template\n"
        "S1,0.1,12.5,False,False,\n"
        "S2,0.2,13.0,True,False,\n")
    mapping = {"S1": anno("IGHV3-23", 16), "S2": anno("IGHV3-66", 18)}
    monkeypatch.setattr(de_novo_bank, "default_annotate", annotator(mapping))
    out = tmp_path / "bank.json"
    summary = tmp_path / "summary.json"
    assert de_novo_bank.main(
        ["--candidates", str(candidates), "--out", str(out),
         "--summary", str(summary), "--source", "de_novo_recombine_ts"]) == 0

    bank = json.loads(out.read_text())
    assert set(bank) == {"rows"}
    assert bank["rows"] == [
        {"cells": ["IGHV3-23:16"], "sequence": "S1",
         "sequence_hash": rules.sequence_hash("S1"), "source": "de_novo_recombine_ts"},
        {"cells": ["IGHV3-66:18"], "sequence": "S2",
         "sequence_hash": rules.sequence_hash("S2"), "source": "de_novo_recombine_ts"},
    ]
    assert json.loads(summary.read_text())["kept"] == 2
