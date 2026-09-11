import json

import numpy as np
import pandas as pd

from nanobody_opt.backtest_mutation_ranker import Parent, Proposal
from nanobody_opt.backtest_pairwise_beam import (
    model_allocation,
    model_beam_proposals,
    quality_round_robin,
    summarize,
)
from nanobody_opt.run_pairwise_beam_shards import heldout_epochs, shard_is_complete


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


def test_summary_reports_high_scorer_epoch_recall():
    frame = pd.DataFrame([
        {
            "epoch": 10,
            "budget": 160,
            "leakage_check": True,
            "baseline_winner_generated": False,
            "model_winner_generated": True,
            "baseline_winner_selected": False,
            "model_winner_selected": True,
            "baseline_field_matches": 1,
            "model_field_matches": 2,
            "baseline_best_matched_field_rank": 8,
            "model_best_matched_field_rank": 1,
        },
        {
            "epoch": 11,
            "budget": 160,
            "leakage_check": True,
            "baseline_winner_generated": False,
            "model_winner_generated": False,
            "baseline_winner_selected": False,
            "model_winner_selected": False,
            "baseline_field_matches": 1,
            "model_field_matches": 1,
            "baseline_best_matched_field_rank": 20,
            "model_best_matched_field_rank": 5,
        },
    ])
    result = summarize(frame)["budgets"]["160"]
    assert result["model_epochs_with_top1_match"] == 1
    assert result["model_epochs_with_top5_match"] == 2
    assert result["model_epochs_with_top10_match"] == 2
    assert result["baseline_epochs_with_top10_match"] == 1


def test_resumable_shard_requires_matching_hash_epoch_and_budgets(tmp_path):
    snapshot = tmp_path / "snapshot.csv.gz"
    pd.DataFrame([
        {"epoch": 1, "sequence": "AAAA"},
        {"epoch": 2, "sequence": "CCCC"},
        {"epoch": 3, "sequence": "DDDD"},
    ]).to_csv(snapshot, index=False)
    assert heldout_epochs(snapshot, holdout_epochs=2, length=4) == [2, 3]

    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "summary.json").write_text(json.dumps({
        "config": {
            "api_snapshot_sha256": "snapshot",
            "model_sha256": "model",
            "target_epochs": [3],
        }
    }))
    pd.DataFrame([
        {"epoch": 3, "budget": 160, "leakage_check": True},
        {"epoch": 3, "budget": 320, "leakage_check": True},
    ]).to_csv(shard / "placements.csv", index=False)
    assert shard_is_complete(
        shard,
        epoch=3,
        budgets=(160, 320),
        snapshot_sha256="snapshot",
        model_sha256="model",
    )
    assert not shard_is_complete(
        shard,
        epoch=3,
        budgets=(160, 320),
        snapshot_sha256="different",
        model_sha256="model",
    )
