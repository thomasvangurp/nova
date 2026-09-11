from nanobody_opt.terminal_policy import (
    basin_parent_recency_ranks,
    learn_terminal_priors,
    terminal_route_priority,
)


def _record(epoch: int, sequence: str, score: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "sequence": sequence,
        "score": score,
        "score_kind": "validator_final_rank_sum",
    }


def test_terminal_priors_learn_winner_step_without_current_epoch():
    original = "A" * 30
    position_zero = "C" + original[1:]
    position_one = "AC" + original[2:]
    records = [
        _record(1, original, 50),
        _record(1, "D" + original[1:], 100),
        _record(2, position_zero, 40),
        _record(2, position_one, 80),
    ]

    priors = learn_terminal_priors(
        records,
        length=30,
        exploration_residues_per_position=2,
    )

    assert priors.winner_step_count[0] == 1
    assert priors.position_rank[0] == 1
    assert "C" in priors.exploration_residues[0]
    assert priors.public_epoch_count == 2


def test_route_priority_uses_position_residue_then_parent_recency():
    old = "A" * 30
    recent = "D" + old[1:]
    records = [
        _record(1, old, 50),
        _record(2, recent, 60),
        _record(3, "C" + old[1:], 40),
        _record(4, "C" + old[1:], 45),
    ]
    priors = learn_terminal_priors(records, length=30)
    parent_rank = basin_parent_recency_ranks([old, recent], priors)

    assert parent_rank[recent] < parent_rank[old]
    priority = terminal_route_priority(
        "C" + old[1:], old, priors=priors, base_rank=parent_rank
    )
    assert priority[:3] == (1, 1, parent_rank[old])
