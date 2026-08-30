from __future__ import annotations

import json
import random
import sqlite3

import pytest

from compound_miner.archive import CandidateArchive
from compound_miner.database import ReactionDatabase, readonly_connection
from compound_miner.provider import load_submission
from compound_miner.selection import select_submission
from compound_miner.types import Candidate, ScoredCandidate, parse_reaction_name
from compound_miner.validity import HistoricalIndex, ValidityGate, maccs_entropy


def test_database_is_immutable(monkeypatch):
    connection = readonly_connection()
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("DELETE FROM molecules")
    connection.close()


@pytest.mark.parametrize("reaction", range(1, 6))
def test_constructs_every_reaction(reaction):
    database = ReactionDatabase()
    rng = random.Random(68 + reaction)
    candidates = [database.candidate(reaction, components) for components in database.sample_component_sets(reaction, 20, rng)]
    assert any(candidate is not None for candidate in candidates)
    for candidate in (value for value in candidates if value is not None):
        parsed_reaction, components = parse_reaction_name(candidate.name)
        assert parsed_reaction == reaction
        assert components == candidate.components


def test_gate_rejects_banned_atoms():
    molecule = Candidate("rxn:1:1:2", "CCBr", 1, (1, 2), "fake", 3, 1)
    assert not ValidityGate(HistoricalIndex()).accepts(molecule)


def test_archive_and_provider_are_atomic(tmp_path, monkeypatch):
    archive = CandidateArchive(tmp_path / "archive.sqlite")
    scores = []
    database = ReactionDatabase()
    generated = []
    for components in database.sample_component_sets(2, 80, random.Random(123)):
        candidate = database.candidate(2, components)
        if candidate is not None and candidate.inchikey not in {item.inchikey for item in generated}:
            generated.append(candidate)
        if len(generated) == 20:
            break
    for index, candidate in enumerate(generated):
        scores.append(ScoredCandidate(candidate, 0.5, -1.0, 1.0 - index / 100.0, "test", 0.1))
    archive.add_candidates(item.candidate for item in scores)
    archive.add_scores(scores)
    selected = select_submission(archive.top(2))
    result = tmp_path / "result_rxn2.json"
    archive.atomic_write_result(result, selected)
    monkeypatch.setenv("NOVA_CANDIDATE_RESULT", str(tmp_path / "result_rxn{reaction}.json"))
    names = load_submission(2).split(",")
    assert len(names) == len(set(names)) == 20
    assert json.loads(result.read_text())["molecules"][0]["score"] == 1.0
