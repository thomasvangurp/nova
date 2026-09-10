import json

import pytest

from nanobody_opt.qualify_screen_leader import _known_sequences, _leader


def test_leader_requires_ranked_canonical_sequence(tmp_path):
    ranking = tmp_path / "ranking.json"
    ranking.write_text(json.dumps({"sequences": [" acdef ", "AAAA"]}))
    assert _leader(ranking) == "ACDEF"

    ranking.write_text(json.dumps({"sequences": ["ACDX"]}))
    with pytest.raises(ValueError, match="canonical"):
        _leader(ranking)


def test_known_sequences_reads_archive_and_ledger(tmp_path):
    archive = tmp_path / "archive.csv"
    archive.write_text("sequence,score\nAAA,1\nBBB,2\n")
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("CCC\nDDD\n")
    assert _known_sequences(archive) == {"AAA", "BBB"}
    assert _known_sequences(ledger) == {"CCC", "DDD"}
