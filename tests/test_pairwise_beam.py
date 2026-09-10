import numpy as np

from nanobody_opt.backtest_mutation_ranker import Parent, Proposal
from nanobody_opt.backtest_pairwise_beam import (
    model_allocation,
    model_beam_proposals,
    quality_round_robin,
)


class _MutationCountModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        changed = np.asarray(matrix[:, 2 * 4 * 20 : 2 * 4 * 20 + 4].sum(axis=1)).ravel()
        probability = np.clip(0.45 + 0.1 * changed, 0.0, 0.99)
        return np.column_stack([1.0 - probability, probability])


def _parent(sequence="AAAA", score=20.0):
    return Parent(sequence, 1, score, (0.0,) * 10)


def test_model_beam_uses_only_supported_residues_and_respects_depth():
    parent = _parent()
    proposals = model_beam_proposals(
        _MutationCountModel(),
        [parent],
        ["CAAA", "ACAA", "AADA", "AAAE"],
        historical={parent.sequence},
        length=4,
        beam_width=3,
        maximum_edits=2,
    )
    assert proposals
    assert max(item.edit_distance for item, _ in proposals) == 2
    assert all(set(item.sequence) <= set("ACDE") for item, _ in proposals)


def test_model_allocation_is_unique_and_keeps_parent_coverage():
    parents = [_parent("AAAA", 10), _parent("CCCC", 20)]
    proposals = []
    for parent in parents:
        for index, sequence in enumerate(("DAAA", "EAAA", "FCCC", "GCCC")):
            if sequence == parent.sequence:
                continue
            proposals.append((Proposal(sequence, parent, "beam", 1, "x"), 1 - index / 10))
    selected = model_allocation(
        proposals, budget=3, parent_floor=2, exploration=0, seed=1
    )
    assert len({item.sequence for item in selected}) == 3
    assert {item.parent.sequence for item in selected} == {"AAAA", "CCCC"}


def test_quality_round_robin_balances_strong_parents():
    first, second = _parent("AAAA", 10), _parent("CCCC", 20)
    proposals = [
        Proposal("DAAA", first, "exhaustive_single", 1, "x"),
        Proposal("EAAA", first, "exhaustive_single", 1, "x"),
        Proposal("FCCC", second, "exhaustive_single", 1, "x"),
        Proposal("GCCC", second, "exhaustive_single", 1, "x"),
    ]
    selected = quality_round_robin(proposals, budget=2)
    assert [item.parent.sequence for item in selected] == ["AAAA", "CCCC"]
