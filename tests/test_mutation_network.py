import json
import sqlite3

from nanobody_opt.mutation_network import (
    MutationNetwork,
    exact_run_is_complete,
    load_exact_artifacts,
    load_lineage_edges,
    load_mutation_effects,
    load_observations,
    record_exact_run,
    rebuild_nearest_edges,
    update_observation_store,
)


def test_raw_exact_run_is_sealed_only_with_all_target_rows_and_artifacts(tmp_path):
    path = tmp_path / "observations.sqlite"
    observation_id = "local-exact:run-one:0:AAAA"
    update_observation_store(path, [{
        "observation_id": observation_id,
        "sequence": "AAAA",
        "score": 42,
        "source": "local_exact_raw",
        "target": "P20809",
        "metrics": {"design_iiptm": 0.7},
        "structure": {
            "confidence_path": "confidence.cif",
            "writer_path": "writer.cif",
            "sample_selection": {"confidence_sample": 1, "writer_sample": 0},
        },
    }])

    sealed = record_exact_run(
        path,
        target="P20809",
        run_id="run-one",
        epoch=25_275,
        source_path=tmp_path / "exact_all.json",
        file_sha256="abc123",
        scorer_protocol="validator-fast-v1",
        observation_ids=[observation_id],
    )

    assert sealed["complete"] is True
    assert exact_run_is_complete(
        path,
        target="P20809",
        run_id="run-one",
        file_sha256="abc123",
        expected_rows=1,
    )
    assert load_exact_artifacts(path, target="P20809")["AAAA"] == [{
        "observation_id": observation_id,
        "run_id": "run-one",
        "confidence_path": "confidence.cif",
        "writer_path": "writer.cif",
        "structure": {
            "confidence_path": "confidence.cif",
            "writer_path": "writer.cif",
            "sample_selection": {"confidence_sample": 1, "writer_sample": 0},
        },
        "sample_selection": {"confidence_sample": 1, "writer_sample": 0},
    }]


def test_incomplete_raw_exact_run_never_becomes_authoritative(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [{
        "observation_id": "present",
        "sequence": "AAAA",
        "score": 42,
        "source": "local_exact_raw",
        "target": "P20809",
    }])

    sealed = record_exact_run(
        path,
        target="P20809",
        run_id="partial",
        epoch=1,
        source_path=tmp_path / "exact_all.json",
        file_sha256="partial-sha",
        scorer_protocol=None,
        observation_ids=["present", "missing"],
    )

    assert sealed == {
        "target": "P20809",
        "run_id": "partial",
        "expected_rows": 2,
        "ingested_rows": 1,
        "complete": False,
    }
    assert not exact_run_is_complete(
        path,
        target="P20809",
        run_id="partial",
        file_sha256="partial-sha",
        expected_rows=2,
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


def test_observation_store_loads_an_unknown_observation_epoch(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [{
        "observation_id": "pilot:AAA",
        "sequence": "AAA",
        "score": 80,
        "epoch": None,
        "target": "P20809",
    }])

    assert load_observations(path, target="P20809")[0]["epoch"] is None


def test_observation_store_preserves_observed_parent_child_lineage(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [{
        "observation_id": "epoch1:AAAC",
        "sequence": "AAAC",
        "parent_sequence": "AAAA",
        "proposal_type": "supported_single",
        "mutations": "A4C",
        "score": 50,
        "epoch": 1,
        "target": "P20809",
    }])

    assert load_lineage_edges(path, target="P20809") == [{
        "parent_sequence": "AAAA",
        "sequence": "AAAC",
        "epoch": 1,
        "source": "unknown",
        "proposal_type": "supported_single",
        "mutations": ["A4C"],
        "metadata": {},
    }]


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
            "epoch": 1,
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


def _reference_neighbour_order(network, sequence, *, limit, maximum_distance,
                               distance_penalty):
    """The ranking exactly as it was computed before it was vectorised.

    Sorts every node with a Python key, then filters -- the O(nodes log nodes)
    shape the fast path replaces.  Kept here so the fast path is pinned to an
    independent implementation rather than to its own output.
    """
    import numpy as np
    from nanobody_opt.mutation_network import AA_INDEX

    vector = np.asarray([AA_INDEX[r] for r in sequence], dtype=np.uint8)
    distances = np.count_nonzero(network._vectors != vector, axis=1)
    order = sorted(
        range(len(network.nodes)),
        key=lambda index: (
            network.nodes[index].score + distance_penalty * int(distances[index]),
            int(distances[index]),
            network.nodes[index].sequence,
        ),
    )
    compatible = [
        index for index in order
        if maximum_distance is None or int(distances[index]) <= maximum_distance
    ]
    return compatible[: max(256, limit * 16)]


def _reference_adjustment(network, source, candidate, *, distance_penalty):
    """The directional-effect refinement exactly as it was before batching.

    Walks every position of the pair through the public ``mutation_contexts``
    and recomputes each effect's shrinkage and uncertainty terms per lookup --
    the shape the batched path replaces.  Kept here so the fast path is pinned
    to an independent implementation rather than to its own output.
    """
    import math

    from nanobody_opt.mutation_network import mutation_contexts

    adjustment = 0.0
    supported = 0
    uncertainty_squared = 0.0
    for context in mutation_contexts(
        source,
        candidate,
        contact_positions=network._contacts.get(source, set()),
    ):
        key = (
            int(context["position"]),
            str(context["from_residue"]),
            str(context["to_residue"]),
            str(context["region"]),
            str(context["contact_class"]),
        )
        effect = network._effects.get(key)
        if effect is None:
            adjustment += distance_penalty
            continue
        support = int(effect["support"])
        shrinkage = support / (support + 3.0)
        uncertainty = float(effect["uncertainty"])
        adjustment += (
            shrinkage * float(effect["median_score_delta"])
            + uncertainty / math.sqrt(max(support, 1))
        )
        supported += 1
        uncertainty_squared += uncertainty**2
    return adjustment, supported, math.sqrt(uncertainty_squared)


def _refine(network, shortlist, query, *, limit, distance_penalty):
    """Refine a reference shortlist with the reference refinement."""
    refined = []
    import numpy as np
    from nanobody_opt.mutation_network import AA_INDEX

    vector = np.asarray([AA_INDEX[r] for r in query], dtype=np.uint8)
    distances = np.count_nonzero(network._vectors != vector, axis=1)
    for index in shortlist:
        node = network.nodes[index]
        adjustment, _, _ = _reference_adjustment(
            network, node.sequence, query, distance_penalty=distance_penalty
        )
        refined.append((
            node.score + adjustment, int(distances[index]), node.sequence,
        ))
    refined.sort()
    return [
        (sequence, distance, round(adjusted, 12))
        for adjusted, distance, sequence in refined[:limit]
    ]


def test_vectorised_neighbour_ranking_matches_the_original_ordering():
    # Randomised networks, including heavy score ties -- ties are where a
    # partial selection can silently disagree with a full sort.
    import random

    import numpy as np
    from nanobody_opt.mutation_network import AA_INDEX

    alphabet = sorted(AA_INDEX)[:6]
    rng = random.Random(20260915)
    for trial in range(12):
        length = rng.choice([4, 6])
        count = rng.choice([5, 40, 400])
        records = []
        for _ in range(count):
            seq = "".join(rng.choice(alphabet) for _ in range(length))
            # A small score vocabulary forces many exact ties.
            records.append({
                "sequence": seq,
                "score": float(rng.choice([10, 20, 30])),
                "epoch": 1,
                "source": "elite",
            })
        network = MutationNetwork(records)
        if not network.nodes:
            continue
        for _ in range(4):
            # Novel queries, like the producer's: a node querying itself sits at
            # distance zero and never exercises the shortlist boundary.
            query = "".join(rng.choice(alphabet) for _ in range(length))
            for maximum_distance in (None, 1, 6):
                for limit in (1, 16):
                    expected = _reference_neighbour_order(
                        network, query, limit=limit,
                        maximum_distance=maximum_distance,
                        distance_penalty=5.0,
                    )
                    actual = network.neighbours(
                        query, limit=limit,
                        maximum_distance=maximum_distance,
                        distance_penalty=5.0,
                    )
                    # Compare the refined result the fast path produces against
                    # the one the full-sort shortlist produces -- exact order
                    # and exact adjusted scores, not mere membership.  A
                    # membership check passes for a shortlist that silently
                    # dropped a node it should have ranked.
                    reference = _refine(network, expected, query,
                                        limit=limit, distance_penalty=5.0)
                    assert [
                        (n.sequence, n.distance, round(n.adjusted_score, 12))
                        for n in actual
                    ] == reference


def test_the_neighbour_shortlist_boundary_keeps_ties():
    # Every node scores identically, so the cut value is the only value: a
    # partial selection that dropped ties at the boundary would lose nodes the
    # exact sort must still be allowed to order.
    records = [
        {"sequence": f"{a}{b}{c}{d}", "score": 10.0, "epoch": 1, "source": "elite"}
        for a in "ACD" for b in "ACD" for c in "ACD" for d in "ACD"
    ]
    network = MutationNetwork(records)
    neighbours = network.neighbours("AAAA", limit=8, maximum_distance=None)
    assert neighbours
    assert len({item.sequence for item in neighbours}) == len(neighbours)


# A real 127-residue VHH, so ``find_regions`` actually resolves cdr1/cdr2/cdr3
# rather than falling back to an all-framework map.  Short synthetic sequences
# never parse, which would leave the region and contact key components -- the
# parts the batched refinement caches -- completely unexercised.
_VHH_SCAFFOLD = (
    "KWQLVESGGGLVKPGGSLRLSCAASGFTMSQGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPMEGEMESFGAGTTVTVAP"
)


def _mutate(rng, sequence, count, alphabet):
    residues = list(sequence)
    for position in rng.sample(range(len(residues)), count):
        residues[position] = rng.choice(alphabet)
    return "".join(residues)


def _vhh_effect_network(rng, *, node_count=300, query_count=3, effect_count=140):
    """A VHH network with learned effects that genuinely fire.

    Effects are harvested from substitutions the queries actually make against
    the nodes, so the refinement reorders instead of degenerating into
    ``distance_penalty * distance`` for every candidate -- which is what a
    network with no effects silently does.
    """
    from nanobody_opt.mutation_network import AMINO_ACIDS, mutation_contexts

    alphabet = AMINO_ACIDS
    sequences = {
        _mutate(rng, _VHH_SCAFFOLD, rng.randint(1, 8), alphabet)
        for _ in range(node_count)
    }
    # One node whose FR1 cysteine anchor is gone: ``find_regions`` raises for it
    # and every position must fall back to "framework".  The cached parse has to
    # reproduce that, not just the happy path.
    unparseable = _VHH_SCAFFOLD[:21] + "A" + _VHH_SCAFFOLD[22:]
    sequences.add(unparseable)
    records = []
    for order, sequence in enumerate(sorted(sequences)):
        record = {
            "sequence": sequence,
            # A small score vocabulary forces exact ties in the primary key.
            "score": float(rng.choice([10.0, 20.0, 30.0])),
            "epoch": 1,
            "source": "elite",
        }
        if order % 5 == 0:
            record["pose_stability"] = {
                "consensus_nanobody_contact_positions": sorted(
                    rng.sample(range(1, len(sequence) + 1), 12)
                )
            }
        records.append(record)

    queries = []
    while len(queries) < query_count:
        candidate = _mutate(rng, _VHH_SCAFFOLD, rng.randint(1, 5), alphabet)
        if candidate not in sequences and candidate not in queries:
            queries.append(candidate)

    probe = MutationNetwork(records)
    harvest = []
    for query in queries:
        for node in rng.sample(probe.nodes, min(40, len(probe.nodes))):
            harvest.extend(
                (
                    int(context["position"]),
                    str(context["from_residue"]),
                    str(context["to_residue"]),
                    str(context["region"]),
                    str(context["contact_class"]),
                )
                for context in mutation_contexts(
                    node.sequence,
                    query,
                    contact_positions=probe._contacts.get(node.sequence, set()),
                )
            )
    chosen = rng.sample(sorted(set(harvest)), min(effect_count, len(set(harvest))))
    effects = [
        {
            "position": position,
            "from_residue": before,
            "to_residue": after,
            "region": region,
            "contact_class": contact_class,
            "support": rng.randint(1, 40),
            "median_score_delta": rng.uniform(-37.3, 37.3),
            "uncertainty": rng.uniform(0.0, 19.7),
        }
        for position, before, after, region, contact_class in chosen
    ]
    return MutationNetwork(records, mutation_effects=effects), queries, unparseable


def _key_coverage(network, queries):
    """Which region / contact-class branches the effects in play actually hit."""
    from nanobody_opt.mutation_network import mutation_contexts

    regions = set()
    contact_classes = set()
    hits = 0
    for query in queries:
        for node in network.nodes:
            for context in mutation_contexts(
                node.sequence,
                query,
                contact_positions=network._contacts.get(node.sequence, set()),
            ):
                key = (
                    int(context["position"]),
                    str(context["from_residue"]),
                    str(context["to_residue"]),
                    str(context["region"]),
                    str(context["contact_class"]),
                )
                if key in network._effects:
                    hits += 1
                    regions.add(key[3])
                    contact_classes.add(key[4])
    return hits, regions, contact_classes


def test_batched_refinement_reproduces_the_original_effect_arithmetic():
    # The refinement result is a sort key AND a returned field
    # (``adjusted_score``), so "close enough" is not enough: compare the exact
    # float, via repr so a -0.0/0.0 swap cannot slip through ``==``.
    import random

    import numpy as np
    from nanobody_opt.mutation_network import AA_INDEX

    rng = random.Random(20260916)
    total_supported = 0
    total_hits = 0
    seen_regions = set()
    seen_contact_classes = set()
    for _ in range(2):
        network, queries, unparseable = _vhh_effect_network(rng)
        indices = list(range(len(network.nodes)))
        for query in queries:
            vector = np.asarray([AA_INDEX[r] for r in query], dtype=np.uint8)
            batched = network._prospective_adjustments(
                indices, query, vector, distance_penalty=5.0
            )
            reference = [
                _reference_adjustment(
                    network, network.nodes[index].sequence, query,
                    distance_penalty=5.0,
                )
                for index in indices
            ]
            assert [repr(item) for item in batched] == [
                repr(item) for item in reference
            ]
            assert batched == reference
            total_supported += sum(item[1] for item in batched)
        hits, regions, contact_classes = _key_coverage(network, queries)
        total_hits += hits
        seen_regions |= regions
        seen_contact_classes |= contact_classes
        # The node whose anchors are gone must take the failed-parse path.
        broken = network._by_sequence[unparseable]
        assert network._region_bounds(network.nodes.index(broken)) is None

    # Guard the guard: if the effects never fired, the comparison above would
    # be comparing distance_penalty * distance against itself.
    assert total_supported > 0
    assert total_hits > 0
    assert {"cdr1", "cdr2", "cdr3", "framework"} <= seen_regions
    assert {"contact", "unknown"} <= seen_contact_classes


def test_cached_region_parse_matches_a_fresh_parse_for_every_node():
    import random

    from nanobody_opt.regions import find_regions

    rng = random.Random(20260918)
    network, _, _ = _vhh_effect_network(rng, node_count=120)
    parsed = 0
    failed = 0
    for index, node in enumerate(network.nodes):
        try:
            regions = find_regions(node.sequence)
        except ValueError:
            expected = None
            failed += 1
        else:
            expected = (regions.cdr1, regions.cdr2, regions.cdr3)
            parsed += 1
        assert network._region_bounds(index) == expected
        # Second call must come from the cache and still agree.
        assert network._region_bounds(index) == expected
    assert parsed and failed


def test_neighbours_with_learned_effects_match_the_full_sort_reference():
    # The same contract as the unrefined ranking test, but on a network whose
    # directional effects actually reorder the shortlist -- the case where the
    # refinement, not the primary key, decides what is returned.
    import random

    rng = random.Random(20260917)
    reordered = 0
    for _ in range(2):
        network, queries, _ = _vhh_effect_network(rng, node_count=320)
        for query in queries:
            for maximum_distance in (None, 6):
                for limit in (1, 16):
                    expected = _reference_neighbour_order(
                        network, query, limit=limit,
                        maximum_distance=maximum_distance,
                        distance_penalty=5.0,
                    )
                    reference = _refine(
                        network, expected, query, limit=limit,
                        distance_penalty=5.0,
                    )
                    actual = network.neighbours(
                        query, limit=limit,
                        maximum_distance=maximum_distance,
                        distance_penalty=5.0,
                    )
                    assert [
                        (item.sequence, item.distance,
                         round(item.adjusted_score, 12))
                        for item in actual
                    ] == reference
                    if actual and expected and (
                        actual[0].sequence
                        != network.nodes[expected[0]].sequence
                    ):
                        reordered += 1
    # If the refinement never moved anything, every assertion above would hold
    # for a refinement that did nothing at all.
    assert reordered > 0


def test_supported_effect_count_and_uncertainty_survive_the_batched_path():
    import random

    import numpy as np
    from nanobody_opt.mutation_network import AA_INDEX

    rng = random.Random(20260919)
    network, queries, _ = _vhh_effect_network(rng, node_count=200)
    checked = 0
    for query in queries:
        vector = np.asarray([AA_INDEX[r] for r in query], dtype=np.uint8)
        for neighbour in network.neighbours(query, limit=24,
                                            maximum_distance=None):
            _, supported, uncertainty = _reference_adjustment(
                network, neighbour.sequence, query, distance_penalty=5.0
            )
            assert neighbour.supported_effect_count == supported
            assert repr(neighbour.effect_uncertainty) == repr(uncertainty)
            checked += 1
        assert network._prospective_adjustments(
            [], query, vector, distance_penalty=5.0
        ) == []
    assert checked >= 24


def test_load_observations_serves_a_cache_hit_without_reopening_the_database(tmp_path, monkeypatch):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [{
        "observation_id": "epoch1:AAA", "sequence": "AAA", "score": 80,
        "epoch": 1, "target": "P20809",
    }])
    first = load_observations(path, target="P20809")
    assert len(first) == 1 and first[0]["score"] == 80

    import nanobody_opt.mutation_network as mn
    def fail_if_called(*a, **k):
        raise AssertionError("cache hit should not touch the database")
    monkeypatch.setattr(mn, "_load_observations_uncached", fail_if_called)
    cached = load_observations(path, target="P20809")
    assert cached == first


def test_load_observations_cache_invalidates_when_the_database_changes(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [{
        "observation_id": "epoch1:AAA", "sequence": "AAA", "score": 80,
        "epoch": 1, "target": "P20809",
    }])
    load_observations(path, target="P20809")  # populate the cache
    update_observation_store(path, [{
        "observation_id": "epoch1:CCC", "sequence": "CCC", "score": 90,
        "epoch": 1, "target": "P20809",
    }])
    refreshed = load_observations(path, target="P20809")
    assert {item["sequence"] for item in refreshed} == {"AAA", "CCC"}


def test_load_observations_cache_is_specific_to_the_target(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [
        {"observation_id": "epoch1:AAA", "sequence": "AAA", "score": 80,
         "epoch": 1, "target": "P20809"},
        {"observation_id": "epoch1:CCC", "sequence": "CCC", "score": 50,
         "epoch": 1, "target": "OTHER"},
    ])
    assert {i["sequence"] for i in load_observations(path, target="P20809")} == {"AAA"}
    assert {i["sequence"] for i in load_observations(path, target="OTHER")} == {"CCC"}


def test_load_observations_survives_a_corrupt_cache_file(tmp_path):
    path = tmp_path / "observations.sqlite"
    update_observation_store(path, [{
        "observation_id": "epoch1:AAA", "sequence": "AAA", "score": 80,
        "epoch": 1, "target": "P20809",
    }])
    cache_path = path.with_name(f"{path.name}.obs_cache.P20809.pkl")
    cache_path.write_bytes(b"not a pickle")
    result = load_observations(path, target="P20809")
    assert len(result) == 1 and result[0]["sequence"] == "AAA"
