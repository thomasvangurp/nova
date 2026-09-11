import numpy as np
import pandas as pd

from nanobody_opt.audit_graph_discrimination import (
    distance_matrix,
    relationship_discrimination,
    relationship_masks,
)


def test_relationship_audit_exposes_bad_neighbour_precision():
    frame = pd.DataFrame({
        "sequence": ["AAAA", "AAAC", "ACCC", "CCCC"],
        "parent_sequence": ["AAAA", "AAAA", "CCCC", "CCCC"],
        "mutations": ["", "A4C", "A2C;A3C;A4C", "A1C"],
    })
    distances = distance_matrix(frame.sequence.tolist())
    masks = relationship_masks(frame, distances)

    result = relationship_discrimination(
        np.asarray([10.0, 20.0, 90.0, 100.0]), masks
    )

    assert masks["distance_1"].sum() == 4
    assert result["baseline_bad_rate"] == 0.5
    assert (
        result["relationships"]["distance_1"]
        ["p_bad_neighbor_given_bad_source"]
        == 1.0
    )


def test_distance_matrix_rejects_mixed_lengths():
    try:
        distance_matrix(["AAA", "AAAA"])
    except ValueError as error:
        assert "one sequence length" in str(error)
    else:
        raise AssertionError("mixed sequence lengths must be rejected")
