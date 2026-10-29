"""Rival-parent bank builder: pure, no annotation/network at import."""
import json

import pytest

from nanobody_opt import pr100_reference_donors as donors
from nanobody_opt import rules
from tools import build_rival_bank, de_novo_bank


def anno(gene, length):
    return {"cdr3_length": length, "germline": {"V": {"genes": [gene]}}}


def annotator(mapping):
    """Sequence -> annotation dict (or Exception to fail annotation)."""
    def annotate(sequence):
        value = mapping[sequence.strip().upper()]
        if isinstance(value, Exception):
            raise value
        return value
    return annotate


def challenge(gene="IGHV3-48", length=17, epoch=26000):
    return {"epoch": epoch, "target": rules.TARGET,
            "allowed_v": gene, "allowed_cdr3_length": length}


def test_build_emits_donor_accepted_rows_with_correct_cells_and_dedup(monkeypatch):
    fetched = ["S1", "S1", "S2", "S3", "S4", "S5"]
    monkeypatch.setattr(build_rival_bank, "fetch_rival_sequences",
                        lambda epoch, prefix, n_epochs=15: list(fetched))
    mapping = {
        "S1": anno("IGHV3-23", 16),
        "S2": anno("IGHV3-66", 18),
        "S3": anno("IGHV1-2", 16),       # outside the 9-gene IGHV3 set
        "S4": anno("IGHV3-23", 24),      # CDR3 too long
        "S5": ValueError("no domain"),   # unannotatable
    }
    bank, summary, sequences = build_rival_bank.build(
        26000, "5GhR", source="rival_bank_test", annotate=annotator(mapping))

    assert sequences == fetched
    # S1 is duplicated in the rival fetch; kept once, best-first order preserved.
    assert summary["input"] == 6 and summary["kept"] == 2
    assert summary["dedup_dropped"] == 1
    assert summary["no_allowed_v_gene"] == 1
    assert summary["cdr3_length_out_of_range"] == 1
    assert summary["annotation_failed"] == 1
    assert [r["sequence"] for r in bank["rows"]] == ["S1", "S2"]
    assert bank["rows"][0] == {"cells": ["IGHV3-23:16"], "sequence": "S1",
                               "sequence_hash": rules.sequence_hash("S1"),
                               "source": "rival_bank_test"}
    assert bank["rows"][1] == {"cells": ["IGHV3-66:18"], "sequence": "S2",
                               "sequence_hash": rules.sequence_hash("S2"),
                               "source": "rival_bank_test"}

    # Every kept row is accepted by the live donor selector's fallback path,
    # which validates identity and cells before selecting.
    exact, chosen, evidence = donors.select(
        bank["rows"], challenge(gene="IGHV3-48", length=17), 16, fallback=True)
    assert exact == [] and evidence["fallback_used"]
    assert {r["sequence_hash"] for r, _ in chosen} == {r["sequence_hash"] for r in bank["rows"]}


def test_build_rejects_invalid_epoch_and_coldkey(monkeypatch):
    monkeypatch.setattr(build_rival_bank, "fetch_rival_sequences", lambda *a, **k: [])
    with pytest.raises(ValueError, match="epoch"):
        build_rival_bank.build(-1, "5GhR", annotate=annotator({}))
    with pytest.raises(ValueError, match="coldkey"):
        build_rival_bank.build(26000, "", annotate=annotator({}))
    with pytest.raises(ValueError, match="n_epochs"):
        build_rival_bank.build(26000, "5GhR", n_epochs=0, annotate=annotator({}))


def test_cli_writes_schema_exact_bank_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(build_rival_bank, "fetch_rival_sequences",
                        lambda epoch, prefix, n_epochs=15: ["S1", "S2"])
    mapping = {"S1": anno("IGHV3-23", 16), "S2": anno("IGHV3-66", 18)}
    monkeypatch.setattr(de_novo_bank, "default_annotate", annotator(mapping))
    out = tmp_path / "rival_bank.json"
    summary = tmp_path / "summary.json"
    assert build_rival_bank.main(
        ["--epoch", "26000", "--rival-coldkey", "5GhR", "--out", str(out),
         "--summary", str(summary), "--source", "rival_bank_ts"]) == 0

    bank = json.loads(out.read_text())
    assert set(bank) == {"rows"}
    assert bank["rows"] == [
        {"cells": ["IGHV3-23:16"], "sequence": "S1",
         "sequence_hash": rules.sequence_hash("S1"), "source": "rival_bank_ts"},
        {"cells": ["IGHV3-66:18"], "sequence": "S2",
         "sequence_hash": rules.sequence_hash("S2"), "source": "rival_bank_ts"},
    ]
    assert json.loads(summary.read_text())["kept"] == 2
