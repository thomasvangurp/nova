import numpy as np
import pandas as pd

from nanobody_opt.backtest_mutation_ranker import (
    Parent,
    Proposal,
    allocate_budget,
    generate_proposals,
    mutation_context_diagnostics,
    percentile_targets,
    select_parents,
)
from nanobody_opt.simulate_round import METRICS


def _frame(epoch, rows):
    records = []
    for index, sequence in enumerate(rows):
        record = {
            "epoch": epoch,
            "sequence": sequence,
            "final_score": float(index + 10),
        }
        for metric, direction in METRICS.items():
            record[metric] = float(index if direction == "min" else -index)
        records.append(record)
    return pd.DataFrame(records)


def _parent(sequence, score=10.0, epoch=1):
    return Parent(sequence, epoch, score, tuple(0.0 for _ in METRICS))


def test_percentile_targets_respect_metric_directions():
    frame = pd.DataFrame({
        metric: ([3.0, 2.0, 1.0] if direction == "max" else [1.0, 2.0, 3.0])
        for metric, direction in METRICS.items()
    })
    targets = percentile_targets(frame)
    assert np.all(targets[0] == 0.0)
    assert np.all(targets[-1] == 1.0)


def test_select_parents_deduplicates_before_per_epoch_limit():
    a = "A" * 127
    b = "A" * 126 + "C"
    frame = pd.concat([_frame(1, [a]), _frame(1, [a, b])], ignore_index=True)
    frame.loc[:, "final_score"] = [10.0, 10.0, 11.0]
    parents = select_parents([(1, frame)], recent_epochs=1, per_epoch=2, limit=2)
    assert [item.sequence for item in parents] == [a, b]


def test_generate_proposals_uses_donor_residues_and_excludes_history():
    parent = _parent("A" * 127)
    donor = "C" + "A" * 125 + "D"
    proposals = generate_proposals(
        [parent],
        [parent.sequence, donor],
        historical={parent.sequence, donor},
        length=127,
        random_candidates_per_parent=4,
        seed=7,
    )
    assert proposals
    assert all(item.sequence not in {parent.sequence, donor} for item in proposals)
    assert all(set(item.sequence) <= {"A", "C", "D"} for item in proposals)


def test_allocate_budget_is_exact_unique_and_covers_parents():
    parents = [_parent("A" * 126 + residue, score=index) for index, residue in enumerate("CDE")]
    proposals = []
    scores = []
    for parent_index, parent in enumerate(parents):
        for item_index in range(5):
            residue = "FGHIK"[item_index]
            sequence = residue + parent.sequence[1:]
            proposals.append(Proposal(sequence, parent, "test", 1, f"A1{residue}"))
            scores.append(float(parent_index * 10 + item_index))
    selected = allocate_budget(
        proposals,
        np.asarray(scores),
        budget=10,
        parent_floor=3,
        exploration=2,
        seed=1,
    )
    assert len(selected) == 10
    assert len({item.sequence for item, _, _ in selected}) == 10
    assert {item.parent.sequence for item, _, lane in selected if lane == "parent_floor"} == {
        item.sequence for item in parents
    }


def test_mutation_context_diagnostic_checks_repeated_ordering():
    evidence = pd.DataFrame([
        {"epoch": 1, "sequence": "A", "mutation": "M29L", "validator_final_score": 80.0},
        {"epoch": 2, "sequence": "B", "mutation": "M29L", "validator_final_score": 50.0},
    ])
    predictions = pd.DataFrame([
        {"epoch": 1, "sequence": "A", "predicted_score": 4.0, "predicted_field_rank": 8, "actual_field_rank": 5},
        {"epoch": 2, "sequence": "B", "predicted_score": 2.0, "predicted_field_rank": 2, "actual_field_rank": 1},
    ])
    merged, summary = mutation_context_diagnostics(evidence, predictions)
    assert len(merged) == 2
    assert np.isclose(
        summary["recurring_mutations"]["M29L"]["ordering_spearman"], 1.0
    )
