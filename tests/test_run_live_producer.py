import json
import time

import pytest

from nanobody_opt.run_live_producer import _chain_state, _parser


def test_live_producer_uses_fresh_miner_chain_state(tmp_path):
    path = tmp_path / "chain.json"
    path.write_text(json.dumps({
        "epoch": 25034,
        "block": 25034 * 361 + 20,
        "epoch_length": 361,
        "nanobody_target": "P20809",
        "updated_at_unix": time.time(),
    }))

    state = _chain_state(path, maximum_age=90)

    assert state["epoch"] == 25034
    assert state["nanobody_target"] == "P20809"


def test_live_producer_rejects_stale_chain_state(tmp_path):
    path = tmp_path / "chain.json"
    path.write_text(json.dumps({
        "epoch": 25034,
        "block": 25034 * 361,
        "epoch_length": 361,
        "updated_at_unix": time.time() - 1000,
    }))

    with pytest.raises(ValueError, match="stale"):
        _chain_state(path, maximum_age=90)


def test_live_producer_uses_elite_multi_epoch_parent_defaults():
    args = _parser().parse_args([])

    assert args.parent_epochs == 16
    assert args.parents_per_epoch == 2
    assert args.parent_limit == 32
    assert args.seed_count == 32
    assert args.minimum_seeds == 8
    assert args.per_seed == 6
    assert args.maximum_exhaustive_singles == 160
    assert args.candidate_limit == 160
    assert args.screen_tranche_size == 20
    assert args.finalists == 3
    assert args.challenger_p90_margin == 20.0
    assert args.single_mutation_fraction == 0.75
