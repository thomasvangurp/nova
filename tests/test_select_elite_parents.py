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
        payloads, target=TARGET, per_epoch=2, limit=4
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
    )

    predecessor = next(
        item for item in selected if item["sequence"] == UID40_25033
    )
    assert predecessor["parent_selection_lane"] == "family_neighbor"
    assert predecessor["nearest_elite_distance"] == 3
    assert predecessor["public_observations"] == [
        {"epoch": 25033, "epoch_rank": 9, "validator_final_score": 93.0}
    ]
