import json

import pytest

from nanobody_opt import fleet


OUR_COLDKEY = "5EkCwcNTfxeNqmQ4ZQqaMGRVJmJmJmJmJmJmJmJmJmJmJmJm"
OTHER_COLDKEY = "5FOtherColdkeyOtherColdkeyOtherColdkeyOtherCold"
HOTKEYS = [f"5H{'abcdefghij'[index] * 46}" for index in range(4)]


def _rows(*pairs):
    return [
        {"uid": uid, "hotkey": hotkey, "coldkey": coldkey}
        for uid, hotkey, coldkey in pairs
    ]


class _Metagraph:
    def __init__(self, uids, hotkeys, coldkeys):
        self.uids = uids
        self.hotkeys = hotkeys
        self.coldkeys = coldkeys


def test_an_empty_fleet_fails_loudly_instead_of_skipping():
    rows = _rows((3, HOTKEYS[0], OTHER_COLDKEY), (4, HOTKEYS[1], OTHER_COLDKEY))

    with pytest.raises(fleet.EmptyFleetError) as error:
        fleet.discover_fleet(rows, OUR_COLDKEY, netuid=68)

    message = str(error.value)
    # 21 days of silence is what this banner exists to prevent.
    assert "NANOBODY_FLEET_EMPTY" in message
    assert "recycled" in message
    assert OUR_COLDKEY in message
    assert "scanned 2 neuron rows" in message


def test_uids_are_discovered_by_coldkey_not_by_remembered_number():
    # uid 8 is the number this miner published into for 21 days after losing it.
    rows = _rows(
        (8, HOTKEYS[0], OTHER_COLDKEY),
        (12, HOTKEYS[1], OUR_COLDKEY),
        (5, HOTKEYS[2], OUR_COLDKEY),
    )

    crew = fleet.discover_fleet(rows, OUR_COLDKEY)

    assert [member.uid for member in crew] == [5, 12]
    assert [member.hotkey for member in crew] == [HOTKEYS[2], HOTKEYS[1]]


def test_a_metagraph_object_is_an_accepted_chain_view():
    metagraph = _Metagraph(
        uids=[0, 1], hotkeys=HOTKEYS[:2], coldkeys=[OTHER_COLDKEY, OUR_COLDKEY]
    )

    crew = fleet.discover_fleet(metagraph, OUR_COLDKEY)

    assert crew == [fleet.FleetMember(uid=1, hotkey=HOTKEYS[1])]


def test_a_truncated_coldkey_is_a_configuration_error_not_an_empty_fleet():
    rows = _rows((1, HOTKEYS[0], OUR_COLDKEY))

    with pytest.raises(ValueError, match="truncated"):
        fleet.discover_fleet(rows, "5EkCwcNTfxeN...")


def test_the_coldkey_can_come_from_the_wallets_public_coldkeypub(tmp_path):
    wallet = tmp_path / "mining_ops"
    wallet.mkdir()
    (wallet / "coldkeypub.txt").write_text(
        json.dumps({"ss58Address": OUR_COLDKEY, "publicKey": "0xdead"})
    )

    assert fleet.resolve_coldkey(
        None, wallet_root=tmp_path, wallet_name="mining_ops"
    ) == OUR_COLDKEY


def _wallet_with_hotkeys(root, names_to_addresses):
    hotkeys = root / "mining_ops" / "hotkeys"
    hotkeys.mkdir(parents=True)
    for name, address in names_to_addresses.items():
        (hotkeys / name).write_text(
            json.dumps({"ss58Address": address, "secretPhrase": "NEVER READ"})
        )
    return root


def test_an_onchain_hotkey_with_no_local_key_fails_separately(tmp_path):
    _wallet_with_hotkeys(tmp_path, {"nova68": HOTKEYS[0]})
    crew = [
        fleet.FleetMember(uid=1, hotkey=HOTKEYS[0]),
        fleet.FleetMember(uid=2, hotkey=HOTKEYS[1]),
    ]

    with pytest.raises(fleet.MissingHotkeyError) as error:
        fleet.require_local_hotkeys(crew, tmp_path, "mining_ops")

    assert "NANOBODY_FLEET_KEYFILE_MISSING" in str(error.value)
    assert "uid 2" in str(error.value)


def test_local_hotkeys_resolve_by_public_address_only(tmp_path):
    _wallet_with_hotkeys(tmp_path, {"nova68": HOTKEYS[0], "nova68-v2": HOTKEYS[1]})
    crew = [
        fleet.FleetMember(uid=1, hotkey=HOTKEYS[0]),
        fleet.FleetMember(uid=2, hotkey=HOTKEYS[1]),
    ]

    resolved = fleet.require_local_hotkeys(crew, tmp_path, "mining_ops")

    assert resolved == {HOTKEYS[0]: "nova68", HOTKEYS[1]: "nova68-v2"}
    # Only the public address is ever surfaced; nothing else from the key file is.
    assert "NEVER READ" not in json.dumps(resolved)


def _ranking(sequences, probabilities=None, order="win-probability"):
    probabilities = probabilities or [
        1.0 - index * 0.1 for index in range(len(sequences))
    ]
    return {
        "selection_order": order,
        "candidates": [
            {"sequence": sequence, "win_probability": probability}
            for sequence, probability in zip(sequences, probabilities)
        ],
    }


def test_k_sequences_are_taken_in_win_probability_order():
    ranking = _ranking(["AAA", "CCC", "DDD", "EEE"])

    assert fleet.distinct_sequences(ranking, 3) == ["AAA", "CCC", "DDD"]


def test_a_ranking_ordered_any_other_way_is_refused():
    ranking = _ranking(["AAA", "CCC"], order="p90")

    with pytest.raises(fleet.SelectionOrderError, match="win-probability"):
        fleet.distinct_sequences(ranking, 2)


def test_a_misordered_win_probability_file_is_refused():
    ranking = _ranking(["AAA", "CCC", "DDD"], probabilities=[0.4, 0.9, 0.2])

    with pytest.raises(fleet.SelectionOrderError, match="not ordered"):
        fleet.distinct_sequences(ranking, 3)


def test_repeated_sequences_are_dropped_because_the_validator_dedupes():
    ranking = _ranking(
        ["AAA", "aaa", "CCC"], probabilities=[0.9, 0.9, 0.5]
    )

    assert fleet.distinct_sequences(ranking, 3) == ["AAA", "CCC"]


def test_assignment_is_pairwise_distinct_one_sequence_per_uid():
    crew = [fleet.FleetMember(uid=index, hotkey=f"5H{index}") for index in (4, 9, 11)]
    blocks = [100, 101, 102]

    assignments = fleet.assign_sequences(crew, ["AAA", "CCC", "DDD"], blocks)

    assert [item.uid for item in assignments] == [4, 9, 11]
    assert [item.sequence for item in assignments] == ["AAA", "CCC", "DDD"]
    assert len({item.sequence for item in assignments}) == 3
    assert [item.commit_block for item in assignments] == blocks


def test_assigning_one_molecule_twice_is_refused():
    crew = [fleet.FleetMember(uid=index, hotkey=f"5H{index}") for index in (1, 2)]

    with pytest.raises(ValueError, match="same sequence"):
        fleet.assign_sequences(crew, ["AAA", "AAA"], [10, 11])


def test_uncovered_uids_are_reported_when_candidates_run_out():
    crew = [fleet.FleetMember(uid=index, hotkey=f"5H{index}") for index in (1, 2, 3)]
    assignments = fleet.assign_sequences(crew, ["AAA", "CCC"], [10, 11])

    assert [member.uid for member in fleet.uncovered(crew, assignments)] == [3]


def test_commitments_are_one_per_block_inside_the_reserve():
    blocks = fleet.commitment_blocks(
        epoch=25062, epoch_length=361, reserve_blocks=55, count=9, block=361 * 25062 + 10
    )

    epoch_end = 361 * 25063
    assert blocks == list(range(epoch_end - 55, epoch_end - 55 + 9))
    assert all(epoch_end - 55 <= block < epoch_end for block in blocks)
    assert len(set(blocks)) == 9


def test_commitments_start_after_the_current_block_inside_the_reserve():
    epoch_end = 361 * 25063
    blocks = fleet.commitment_blocks(
        epoch=25062,
        epoch_length=361,
        reserve_blocks=55,
        count=3,
        block=epoch_end - 20,
    )

    assert blocks == [epoch_end - 19, epoch_end - 18, epoch_end - 17]


def test_a_fleet_too_large_for_the_remaining_reserve_is_refused():
    epoch_end = 361 * 25063
    with pytest.raises(ValueError, match="do not fit"):
        fleet.commitment_blocks(
            epoch=25062,
            epoch_length=361,
            reserve_blocks=55,
            count=9,
            block=epoch_end - 3,
        )


def test_the_ledger_records_every_submission_once(tmp_path):
    ledger = tmp_path / "published.txt"

    assert fleet.record_published(ledger, ["AAA", "CCC", "DDD"]) == ["AAA", "CCC", "DDD"]
    assert fleet.record_published(ledger, ["CCC", "eee"]) == ["EEE"]
    assert fleet.read_ledger(ledger) == ["AAA", "CCC", "DDD", "EEE"]


def test_the_live_epochs_own_sequences_stay_selectable_this_epoch(tmp_path):
    ledger = tmp_path / "published.txt"
    fleet.record_published(ledger, ["AAA", "CCC", "DDD"])

    exclusion = fleet.write_exclusion_file(
        tmp_path / "excluded.txt", ledger, ["CCC"]
    )

    assert fleet.read_ledger(exclusion) == ["AAA", "DDD"]
    assert fleet.read_ledger(ledger) == ["AAA", "CCC", "DDD"]


def _checkpoint(sequence, epoch=25062):
    return {
        "schema_version": 3,
        "status": "qualified",
        "publication_stage": "qualified",
        "target": "P20809",
        "epoch": epoch,
        "sequences": [sequence],
        "validation": {"boltzgen": {"metrics": {"design_ptm": 0.5}}},
        "selection": {"score": 120.0, "score_median": 110.0, "score_p90": 130.0},
        "exact": {"observation_count": 4},
    }


def test_the_artifact_is_one_file_with_one_entry_per_uid():
    crew = [fleet.FleetMember(uid=uid, hotkey=f"5H{uid}") for uid in (5, 8)]
    assignments = fleet.assign_sequences(crew, ["AAA", "CCC"], [100, 101])
    checkpoints = {
        "AAA": _checkpoint("AAA"),
        "CCC": _checkpoint("CCC"),
    }

    artifact = fleet.fleet_artifact(
        target="P20809",
        epoch=25062,
        stage="qualified",
        assignments=assignments,
        checkpoints=checkpoints,
    )

    assert artifact["schema_version"] == fleet.FLEET_ARTIFACT_SCHEMA_VERSION
    assert artifact["artifact_kind"] == "nanobody-fleet"
    assert artifact["submission_count"] == 2
    assert [entry["uid"] for entry in artifact["submissions"]] == [5, 8]
    assert [entry["sequence"] for entry in artifact["submissions"]] == ["AAA", "CCC"]
    assert artifact["submissions"][0]["metrics"] == {"design_ptm": 0.5}
    assert artifact["submissions"][0]["checkpoint"]["sequences"] == ["AAA"]
    assert fleet.artifact_sequences(artifact) == ["AAA", "CCC"]
    assert fleet.assignments_from_artifact(artifact) == assignments


def test_a_single_sequence_artifact_is_still_retired_after_the_upgrade():
    assert fleet.artifact_sequences({"sequences": ["AAA"]}) == ["AAA"]


def test_the_artifact_is_replaced_in_one_rename(tmp_path):
    destination = tmp_path / "nested" / "nanobody_result.json"
    fleet.write_atomic_json(destination, {"epoch": 1})
    fleet.write_atomic_json(destination, {"epoch": 2})

    assert json.loads(destination.read_text()) == {"epoch": 2}
    assert [path.name for path in destination.parent.iterdir()] == [destination.name]


def test_the_fleet_fingerprint_changes_when_the_uid_set_changes():
    one = [fleet.FleetMember(uid=1, hotkey="5H1")]
    two = [*one, fleet.FleetMember(uid=2, hotkey="5H2")]

    assert fleet.fleet_fingerprint(one) != fleet.fleet_fingerprint(two)


def _pool(entries):
    """entries: (sequence, win_probability, score_p90) in win-probability order."""
    return {
        "selection_order": fleet.REQUIRED_SELECTION_ORDER,
        "candidates": [
            {"sequence": s, "win_probability": w, "score_p90": p}
            for s, w, p in entries
        ],
    }


def test_the_value_share_knob_buys_expected_score_when_asked():
    """Off by default -- variance seeking is the right objective -- but the
    knob must work, so an estimator that goes wrong can be sidestepped."""
    pool = _pool([
        ("AAA", 0.30, 200.0),
        ("BBB", 0.20, 190.0),
        ("CCC", 0.10, 100.0),   # best expected score, worst draw odds
        ("DDD", 0.05, 110.0),
    ])
    chosen = fleet.distinct_sequences(pool, 4, value_share=0.5)
    assert set(chosen) == {"AAA", "BBB", "CCC", "DDD"}
    # The two best expected scores are picked first, then the tail seekers.
    assert chosen[:2] == ["CCC", "DDD"]


def test_the_fleet_splits_between_expected_score_and_the_tail():
    """Predicted p90 correlated +0.807 with the actual score across our five
    scored tickets in epoch 25099; win probability managed only -0.526. Half
    the fleet buys expected quality, half still buys the tail."""
    assert 0.0 < fleet.DEFAULT_VALUE_SHARE < 1.0
    pool = _pool([("AAA", 0.3, 200.0), ("BBB", 0.2, 100.0)])
    assert set(fleet.distinct_sequences(pool, 2)) == {"AAA", "BBB"}


def test_a_zero_value_share_is_pure_win_probability_order():
    pool = _pool([("AAA", 0.3, 200.0), ("BBB", 0.2, 100.0)])
    assert fleet.distinct_sequences(pool, 2, value_share=0.0) == ["AAA", "BBB"]


def test_the_fleet_never_repeats_a_sequence_across_the_two_halves():
    """A molecule that is both the best expected score and the best draw must
    not consume two uids: the validator scores one sequence once."""
    pool = _pool([("AAA", 0.9, 100.0), ("BBB", 0.2, 200.0), ("CCC", 0.1, 300.0)])
    chosen = fleet.distinct_sequences(pool, 3, value_share=0.5)
    assert len(chosen) == len(set(chosen)) == 3


def test_value_share_must_be_a_fraction():
    pool = _pool([("AAA", 0.3, 200.0)])
    with pytest.raises(ValueError, match="fraction"):
        fleet.distinct_sequences(pool, 1, value_share=1.5)


REAL = "KWALVESGGGLVKPGGSLRLSCAASGFTASRGWYGMSWFRQAPGKEREWVSLISSSGSTWMDYPDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCGVFLGAHPCPLEGEMDAFGAGTTVVVGQ"


def _variant(base, swaps):
    """A near-neighbour: the kind of candidate local search actually produces."""
    out = list(base)
    for position, residue in swaps:
        out[position] = residue
    return "".join(out)


def test_a_near_duplicate_yields_to_a_separated_candidate():
    """Two tickets closer than the limit are one ticket to the validator.

    Over epochs 25093-25095 every dropped uid was 0.9685-0.9843 identical to a
    uid that scored, while every surviving pair sat at 0.9449-0.9606: we
    committed seven submissions and played three.
    """
    twin = _variant(REAL, [(2, "M")])          # one residue apart
    far = _variant(REAL, [(i, "G") for i in range(30, 50)])
    # The twin outranks the separated candidate on win probability.
    pool = _pool([(REAL, 0.30, 100.0), (twin, 0.29, 101.0), (far, 0.10, 150.0)])

    chosen = fleet.distinct_sequences(pool, 2)

    assert REAL in chosen
    assert far in chosen, "a one-residue neighbour outranked a real second ticket"
    assert twin not in chosen


def test_a_degenerate_pool_still_fills_the_fleet_with_its_most_separated():
    """A hard floor is the wrong tool: on the epoch-25095 pool it left one
    assignable candidate out of thirty, because the search produces one
    molecule thirty times. Fewer tickets is worse than close tickets."""
    twins = [(_variant(REAL, [(2, aa)]), 0.3 - i * 0.01, 100.0 + i)
             for i, aa in enumerate("MVQTA")]
    chosen = fleet.distinct_sequences(_pool(twins), 5)
    assert len(chosen) == 5, "the fleet went unspent over similarity"
    assert len(set(chosen)) == 5


def test_sequence_identity_is_conservative_against_the_validators_rule():
    twin = _variant(REAL, [(2, "M")])
    assert fleet.sequence_identity(REAL, twin) > 0.99
    assert fleet.sequence_identity(REAL, REAL) == 1.0
    assert fleet.MAXIMUM_FLEET_IDENTITY < 0.95, (
        "the proxy needs margin under the rule it stands in for"
    )


def test_a_candidate_the_field_is_sitting_on_is_avoided():
    """Separation among our own tickets does not see the competition.

    In epoch 25099 our uid 12 was 0.9528 identical to another miner's
    submission in the same epoch: the validator scored theirs, dropped ours,
    and we paid for a registration that produced nothing.
    """
    contested = _variant(REAL, [(2, "M")])
    clear = _variant(REAL, [(i, "G") for i in range(30, 50)])
    pool = _pool([(contested, 0.30, 100.0), (clear, 0.10, 150.0)])

    chosen = fleet.distinct_sequences(pool, 1, crowd=[REAL])

    assert chosen == [clear], "took a molecule the field already occupies"


def test_crowding_never_leaves_a_uid_unspent():
    """A registration held is paid for; a contested ticket still beats none."""
    contested = [_variant(REAL, [(2, aa)]) for aa in "MVQ"]
    pool = _pool([(s, 0.3 - i * 0.01, 100.0 + i) for i, s in enumerate(contested)])
    chosen = fleet.distinct_sequences(pool, 2, crowd=[REAL])
    assert len(chosen) == 2


def test_no_crowd_is_the_previous_behaviour():
    far = _variant(REAL, [(i, "G") for i in range(30, 50)])
    pool = _pool([(REAL, 0.3, 100.0), (far, 0.2, 110.0)])
    assert fleet.distinct_sequences(pool, 2) == fleet.distinct_sequences(
        pool, 2, crowd=[]
    )
