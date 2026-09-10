import json
import sqlite3

from nanobody_opt.mutation_network import (
    MutationNetwork,
    load_mutation_effects,
    load_observations,
    rebuild_nearest_edges,
    update_observation_store,
)


def test_virtual_node_uses_all_nearby_scored_predecessors(tmp_path):
    network = MutationNetwork([
        {"sequence": "AAAA", "score": 40, "epoch": 1, "source": "winner"},
        {"sequence": "AAAC", "score": 60, "epoch": 2, "source": "elite"},
        {"sequence": "AAAC", "score": 50, "epoch": 3, "source": "winner"},
        {"sequence": "CCCC", "score": 100, "epoch": 3, "source": "elite"},
    ])

    evidence = network.evidence("AAAS", distance_penalty=5, consensus_size=2)

    assert evidence["predecessor_min_distance"] == 1
    assert evidence["predecessor_support_within_six"] == 3
    assert evidence["predecessor_potential_score"] == 45
    assert evidence["predecessor_neighbours"][0]["sequence"] == "AAAA"

    output = tmp_path / "network.json"
    network.save(output, edge_distance=4)
    payload = json.loads(output.read_text())
    assert payload["node_count"] == 3
    assert payload["edge_count"] >= 2


def test_observation_store_updates_existing_identity_atomically(tmp_path):
    path = tmp_path / "observations.sqlite"
    first = {
        "observation_id": "epoch1:AAA",
        "sequence": "AAA",
        "score": 80,
        "epoch": 1,
        "target": "P20809",
    }
    update_observation_store(path, [first])
    update_observation_store(path, [{**first, "score": 70}])

    observations = load_observations(path, target="P20809")
    assert len(observations) == 1
    assert observations[0]["score"] == 70


def test_public_validator_score_has_priority_over_local_proxy():
    network = MutationNetwork([
        {
            "sequence": "AAAA",
            "score": 130,
            "score_kind": "local_validator_replay_rank_sum",
        },
        {
            "sequence": "AAAA",
            "score": 40,
            "score_kind": "validator_final_rank_sum",
        },
    ])

    assert network.nodes[0].score == 40


def test_contextual_directional_effect_changes_virtual_node_prospect():
    network = MutationNetwork(
        [
            {"sequence": "AAAA", "score": 80},
            {"sequence": "CCCC", "score": 50},
        ],
        mutation_effects=[{
            "position": 4,
            "from_residue": "A",
            "to_residue": "S",
            "region": "framework",
            "contact_class": "unknown",
            "support": 9,
            "median_score_delta": -30,
            "uncertainty": 0,
        }],
    )

    evidence = network.evidence("AAAS")

    assert evidence["predecessor_neighbours"][0]["sequence"] == "AAAA"
    assert evidence["prospective_effect_support"] == 1
    assert evidence["prospective_score"] == 57.5


def test_parallel_edge_rebuild_learns_directional_single_mutation_effects(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [
        {
            "observation_id": "epoch1:AAAA",
            "sequence": "AAAA",
            "score": 80,
            "score_kind": "validator_final_rank_sum",
            "epoch": 1,
            "target": "P20809",
        },
        {
            "observation_id": "epoch2:AAAC",
            "sequence": "AAAC",
            "score": 50,
            "score_kind": "validator_final_rank_sum",
            "epoch": 2,
            "target": "P20809",
        },
        {
            "observation_id": "epoch2:CCCCC",
            "sequence": "CCCCC",
            "score": 60,
            "score_kind": "validator_final_rank_sum",
            "epoch": 2,
            "target": "P20809",
        },
    ])

    summary = rebuild_nearest_edges(path, target="P20809", workers=2)

    assert summary["node_count"] == 3
    assert summary["edge_count"] == 1
    assert summary["mutation_effect_count"] == 2
    assert len(load_mutation_effects(path, target="P20809")) == 2
    with sqlite3.connect(path) as connection:
        effects = connection.execute(
            """
            SELECT from_residue, to_residue, median_score_delta
            FROM mutation_effects
            ORDER BY from_residue, to_residue
            """
        ).fetchall()
    assert effects == [("A", "C", -30.0), ("C", "A", 30.0)]
