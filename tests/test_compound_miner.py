from __future__ import annotations

import json
import random
import sqlite3

import numpy as np
import pytest
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold

from compound_miner.archive import CandidateArchive
from compound_miner.database import ReactionDatabase, readonly_connection
from compound_miner.distilled import (
    DistilledOracle,
    StudentExample,
    collate_examples,
    examples_from_bindingdb_tsv,
    examples_from_public_csv,
    prepare_bindingdb_balanced,
    protein_features,
    scaffold_split,
    target_split,
)
from compound_miner.interaction_student import RidgeInteractionEnsemble, scaffold_indices
from compound_miner.provider import load_submission
from compound_miner.selection import select_submission
from compound_miner.types import Candidate, ScoredCandidate, parse_reaction_name
from compound_miner.validity import HistoricalIndex, ValidityGate, maccs_entropy
from tools.score_boltz_batch import balanced_partitions


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


def test_boltz_partitions_are_complete_and_deterministic():
    records = [
        {"name": f"molecule-{index}", "heavy_atoms": 10 + index}
        for index in range(17)
    ]
    first = balanced_partitions(records, 4)
    second = balanced_partitions(list(reversed(records)), 4)
    assert first == second
    assert sorted(item["name"] for part in first for item in part) == sorted(
        item["name"] for item in records
    )
    costs = [
        sum((258 + item["heavy_atoms"]) ** 2 + (200 + item["heavy_atoms"]) ** 2 for item in part)
        for part in first
    ]
    assert max(costs) / min(costs) < 1.3


def test_distilled_oracle_batches_variable_molecules():
    examples = [
        StudentExample(name="ethanol", smiles="CCO", target="P40261", objective=0.1),
        StudentExample(name="benzene", smiles="c1ccccc1", target="P40261", objective=0.2),
    ]
    batch = collate_examples(examples)
    model = DistilledOracle(hidden=32, layers=2)
    output = model(batch["graphs"], batch["protein"], batch["reaction"], batch["components"])
    assert output["objective_mean"].shape == (2,)
    assert torch.isfinite(output["objective_mean"]).all()
    assert batch["protein"].shape == (2, 544)


def test_protein_features_include_cached_esm():
    embedding = np.arange(480, dtype=np.float32)
    features = protein_features("ACDEFG", esm_embedding=embedding)
    assert features.shape == (544,)
    np.testing.assert_array_equal(features[64:], embedding)


def test_public_affinity_rows_are_aggregated(tmp_path):
    public = tmp_path / "public.csv"
    public.write_text(
        "target_accession,protein_sequence,molecule_chembl_id,canonical_smiles,pchembl_value\n"
        "T1,ACDE,m1,CCO,6.0\n"
        "T1,ACDE,m1,CCO,8.0\n"
    )
    examples = examples_from_public_csv(public)
    assert len(examples) == 1
    assert examples[0].experimental_affinity == 7.0
    assert examples[0].protein_sequence == "ACDE"


def test_scaffold_split_has_no_scaffold_overlap():
    examples = [
        StudentExample(name=f"a{index}", smiles=smiles, target="P40261", objective=float(index))
        for index, smiles in enumerate(("c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CC1CCCCC1", "CCO", "CCN"))
    ]
    train, validation = scaffold_split(examples, validation_fraction=0.5)
    assert train
    assert validation


def test_target_split_has_no_protein_overlap():
    examples = [
        StudentExample(name=f"{target}-{index}", smiles="CCO", target=target, protein_sequence="ACDE", experimental_affinity=6.0)
        for target in ("T1", "T2", "T3", "T4", "T5")
        for index in range(3)
    ]
    train, validation = target_split(examples, validation_fraction=0.4)
    assert train and validation
    assert {value.target for value in train}.isdisjoint(value.target for value in validation)


def test_bindingdb_balancer_keeps_single_protein_numeric_rows(tmp_path):
    source = tmp_path / "bindingdb.tsv"
    fields = [
        "Ligand SMILES", "Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "BindingDB Target Chain Sequence 1",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
    ]
    source.write_text(
        "\t".join(fields) + "\n"
        + "\t".join(("CCO", "100", "", "", "", "1", "ACDE", "T1", "")) + "\n"
        + "\t".join(("CCN", ">100", "", "", "", "1", "ACDE", "T1", "")) + "\n"
        + "\t".join(("CCC", "10", "", "", "", "2", "ACDE", "T1", "")) + "\n"
    )
    output = tmp_path / "balanced.csv.gz"
    metrics = prepare_bindingdb_balanced(source, output, maximum_per_target=4, minimum_per_target=1)
    examples = examples_from_public_csv(output)
    direct_examples = examples_from_bindingdb_tsv(source)
    assert metrics["interactions"] == 1
    assert len(examples) == 1
    assert examples[0].target == "T1"
    assert examples[0].experimental_affinity == 7.0
    assert len(direct_examples) == 1
    assert direct_examples[0].target == "T1"


def test_interaction_adapter_roundtrip_and_scaffold_split(tmp_path):
    features = np.asarray(
        [[0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.1, 0.0], [0.5, 0.5], [0.6, 0.4]],
        dtype=np.float32,
    )
    labels = features[:, 0] - features[:, 1]
    model = RidgeInteractionEnsemble.fit(features, labels, members=4, ridge=0.1)
    before, uncertainty = model.predict(features)
    loaded = RidgeInteractionEnsemble.load(model.save(tmp_path / "adapter.npz"))
    after, loaded_uncertainty = loaded.predict(features)
    np.testing.assert_allclose(before, after)
    np.testing.assert_allclose(uncertainty, loaded_uncertainty)

    smiles = ["c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CC1CCCCC1", "CCO", "CCN"]
    train, validation = scaffold_indices(smiles, validation_fraction=0.4)
    assert set(train).isdisjoint(validation)
    train_scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(smiles[index]) for index in train}
    validation_scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(smiles[index]) for index in validation}
    assert train_scaffolds.isdisjoint(validation_scaffolds)
