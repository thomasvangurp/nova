from nanobody_opt.select_elite_parents import select_elite_parents
from nanobody_opt.simulate_round import METRICS


TARGET = "P20809"
PARENT_25022 = (
    "KWQLVESGGGLVKPGGSLRLSCAASGFTMSQGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPMEGEMESFGAGTTVTVAP"
)
PARENT_25026 = (
    "QWQLVESGGGLVKPGGSLRLSCAASGFTMSQGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPMEGEMEAFGVGAVVTVAP"
)
UID2_25037 = (
    "KWNLVESGGGLVKPGGSLRLSCAASGFTMSRGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPLEGEMDAFSAGTAVTVGS"
)
UID40_25033 = UID2_25037[:-1] + "D"
UID63_25032 = (
    "KWNLVESGGGLVKPGGSLRLSCAASGFTMSRGWYGMSWFRQAPGKEREWVSLISSSGSTWMDY"
    "PDSVKGRFTSSRDNAKNSLYGQMNSLRAEDTAVYYCAVSLGAHPCPLEGEMDAFSAGTTVTVAP"
)


def _nanobody(sequence, score):
    return {
        "protein_name": TARGET,
        "sequence": sequence,
        "final_score": score,
        "confidence_rank": 20,
        "physical_interaction_rank": 20,
        "developability_rank": 10,
        **{name: float(index + 1) for index, name in enumerate(METRICS)},
    }


def _payload(*nanobodies):
    return {"success": True, "data": [{"nanobodies": list(nanobodies)}]}


def test_top_two_per_epoch_retains_both_proven_successful_parents():
    payloads = {
        25019: _payload(
            _nanobody("A" * 127, 35),
            _nanobody(PARENT_25022, 44),
            _nanobody("C" * 127, 70),
        ),
        25016: _payload(
            _nanobody(PARENT_25026, 72),
            _nanobody("D" * 127, 79),
        ),
    }

    selected = select_elite_parents(
        payloads, target=TARGET, per_epoch=2, limit=4, basin_core_limit=0
    )

    sequences = [item["sequence"] for item in selected]
    assert PARENT_25022 in sequences
    assert PARENT_25026 in sequences
    assert sequences.index(PARENT_25022) < sequences.index(PARENT_25026)
    assert next(
        item for item in selected if item["sequence"] == PARENT_25022
    )["public_observations"] == [
        {"epoch": 25019, "epoch_rank": 2, "validator_final_score": 44.0}
    ]


def test_terminally_diverse_neighbor_lane_retains_uid2_direct_predecessor():
    payloads = {
        25032: _payload(
            _nanobody(UID63_25032, 85),
            _nanobody("A" * 127, 86),
        ),
        25033: _payload(
            _nanobody("C" * 127, 40),
            _nanobody("D" * 127, 41),
            _nanobody(UID2_25037[:-1] + "P", 70),
            _nanobody(UID2_25037[:-1] + "G", 75),
            _nanobody(UID2_25037[:-1] + "A", 80),
            _nanobody(UID2_25037[:-1] + "Q", 85),
            _nanobody(UID2_25037[:-1] + "E", 90),
            _nanobody(UID2_25037[:-1] + "K", 91),
            _nanobody(UID40_25033, 93),
            _nanobody(UID2_25037[:-1] + "T", 95),
        ),
    }

    selected = select_elite_parents(
        payloads,
        target=TARGET,
        per_epoch=2,
        limit=12,
        neighbor_per_epoch=8,
        neighbor_distance=6,
        neighbor_limit=8,
        terminal_variants_per_residue=1,
        basin_core_limit=0,
    )

    predecessor = next(
        item for item in selected if item["sequence"] == UID40_25033
    )
    assert predecessor["parent_selection_lane"] == "family_neighbor"
    assert predecessor["nearest_elite_distance"] == 3
    assert predecessor["public_observations"] == [
        {"epoch": 25033, "epoch_rank": 9, "validator_final_score": 93.0}
    ]


def test_neighbor_capacity_is_reserved_even_when_elite_bank_fills_limit():
    payloads = {
        25030: _payload(
            _nanobody(UID63_25032, 50),
            _nanobody("A" * 127, 51),
            _nanobody(UID40_25033, 70),
        ),
        25031: _payload(
            _nanobody("C" * 127, 52),
            _nanobody("D" * 127, 53),
        ),
        25032: _payload(
            _nanobody("E" * 127, 54),
            _nanobody("F" * 127, 55),
        ),
    }

    selected = select_elite_parents(
        payloads,
        target=TARGET,
        per_epoch=2,
        limit=4,
        neighbor_per_epoch=1,
        neighbor_distance=6,
        neighbor_limit=2,
        terminal_variants_per_residue=2,
        basin_core_limit=0,
    )

    assert len(selected) == 4
    predecessor = next(
        item for item in selected if item["sequence"] == UID40_25033
    )
    assert predecessor["parent_selection_lane"] == "family_neighbor"


def test_older_better_elite_is_not_replaced_by_newer_mediocre_parent():
    old_elite = PARENT_25022
    payloads = {
        24000: _payload(_nanobody(old_elite, 30)),
        25039: _payload(_nanobody("A" * 127, 100)),
    }

    selected = select_elite_parents(
        payloads, target=TARGET, per_epoch=1, limit=1, basin_core_limit=0
    )

    assert selected[0]["sequence"] == old_elite
    assert selected[0]["validator_final_score"] == 30


def test_active_bank_balances_recent_lineage_and_global_quality():
    old_elite = PARENT_25022
    recent_elite = PARENT_25026
    payloads = {
        24000: _payload(
            _nanobody(old_elite, 30),
            _nanobody("A" * 127, 31),
        ),
        25039: _payload(
            _nanobody(recent_elite, 90),
            _nanobody("C" * 127, 91),
        ),
    }

    selected = select_elite_parents(
        payloads,
        target=TARGET,
        per_epoch=1,
        limit=2,
        neighbor_per_epoch=1,
        neighbor_limit=1,
        recent_epoch_window=1,
        recent_anchor_limit=1,
        recent_neighbor_limit=0,
        basin_core_limit=0,
    )

    assert [item["sequence"] for item in selected] == [recent_elite, old_elite]
    assert selected[0]["parent_history_scope"] == "recent"
    assert selected[1]["parent_history_scope"] == "global"


def test_parent_capacity_is_reserved_across_strong_masked_cores():
    core_a = "A" * 100
    core_b = "C" * 100
    a1 = "A" * 10 + core_a + "D" * 17
    a2 = "C" * 10 + core_a + "E" * 17
    b1 = "D" * 10 + core_b + "F" * 17
    payloads = {
        25040: _payload(_nanobody(a1, 45), _nanobody(b1, 60)),
        25041: _payload(_nanobody(a2, 50), _nanobody("G" * 127, 90)),
    }

    selected = select_elite_parents(
        payloads,
        target=TARGET,
        per_epoch=1,
        limit=3,
        basin_core_limit=2,
        basin_parent_limit=3,
    )

    assert [item["sequence"] for item in selected] == [a1, b1, a2]
    assert [item["basin_rank"] for item in selected] == [1, 2, 1]
    assert all(item["basin_allocation_reserved"] for item in selected)
