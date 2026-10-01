"""Where the sweep spends its anchor budget.

The original rule weighted each neighbourhood by `best_score - floor`, linear
above the anchor pool's worst score. Linear does not concentrate when the
low-mean neighbourhoods outnumber the high-mean ones a hundred to one: measured
on the live reaction-3 bank at 45,731 molecules, 1.4% of the anchor weight
reached the fifteen prefixes that held the top 20, while 67% of the bank sat in
13,044 prefixes at median depth 1 whose best score ever (0.0539) was below the
top-20 cut (0.0802). Those draws could not enter the top 20 at all, which is
why best-20 stayed bit-identical across 17,400 molecules.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder():
    spec = importlib.util.spec_from_file_location(
        "_bank_builder", ROOT / "tools" / "bank_builder.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bank_builder"] = module
    spec.loader.exec_module(module)
    return module


def _bank(prefixes, reaction=3):
    """An in-memory bank: {prefix_id: [scores]}."""
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE candidates(name TEXT PRIMARY KEY, components TEXT, "
        "reaction INT, score REAL)")
    index = 0
    for prefix, scores in prefixes.items():
        for score in scores:
            connection.execute(
                "INSERT INTO candidates VALUES(?,?,?,?)",
                (f"m{index}", json.dumps([prefix, 900 + index]), reaction, score))
            index += 1
    connection.commit()
    return connection


def _population(good_mean, bad_mean, good=3, bad=300, depth=12):
    """A few rich neighbourhoods swamped in count by many poor ones."""
    generator = __import__("random").Random(11)
    prefixes = {}
    for identifier in range(good):
        prefixes[identifier] = [generator.gauss(good_mean, 0.010) for _ in range(depth)]
    for identifier in range(good, good + bad):
        prefixes[identifier] = [generator.gauss(bad_mean, 0.010) for _ in range(depth)]
    return prefixes


def test_the_budget_follows_the_neighbourhoods_that_can_reach_the_cut(builder):
    prefixes = _population(good_mean=0.075, bad_mean=0.030)
    connection = _bank(prefixes)
    cut = 0.080
    keys = list(prefixes)
    weights = builder.expected_gain_weights(connection, 3, [(k,) for k in keys], cut)
    assert weights is not None
    total = sum(weights)
    rich = sum(weights[index] for index, key in enumerate(keys) if key < 3)
    assert rich / total > 0.5, f"only {rich / total:.3f} of the budget reached the rich 1%"


def test_the_linear_rule_it_replaces_does_not(builder):
    """The comparison is the whole point: the fix has to beat the thing it
    replaces on the same population, or it is just a different arbitrary rule."""
    prefixes = _population(good_mean=0.075, bad_mean=0.030)
    best = {key: max(scores) for key, scores in prefixes.items()}
    floor = min(best.values())
    linear = {key: max(1e-6, value - floor + 1e-3) for key, value in best.items()}
    total = sum(linear.values())
    rich = sum(value for key, value in linear.items() if key < 3)
    assert rich / total < 0.10, "the linear rule was not actually diffuse here"


def test_a_lucky_singleton_loses_to_a_genuinely_rich_neighbourhood(builder):
    """The max is the wrong statistic; the shrunk mean is the point.

    A prefix seen once carries mostly seed noise -- per-molecule sd is 0.00297
    and selecting a top-20 on single passes costs 0.1335 against a three-pass
    mean -- so weighting on `best_score` sent budget to whichever neighbourhood
    got the luckiest draw. Here the singleton has the HIGHER maximum and must
    still lose to the neighbourhood that has shown twelve good draws.
    """
    generator = __import__("random").Random(5)
    prefixes = {
        0: [0.092],                                                   # lucky once
        1: [generator.gauss(0.075, 0.010) for _ in range(12)],        # genuinely rich
    }
    prefixes.update({identifier: [generator.gauss(0.030, 0.010) for _ in range(12)]
                     for identifier in range(2, 60)})
    assert max(prefixes[0]) > max(prefixes[1]), "the fixture must favour the singleton on max"
    connection = _bank(prefixes)
    weights = builder.expected_gain_weights(connection, 3, [(0,), (1,)], 0.080)
    assert weights is not None
    assert weights[1] > weights[0] * 5, (
        f"the rich neighbourhood only got {weights[1] / weights[0]:.1f}x the "
        "lucky singleton's weight")


def test_a_cold_bank_falls_back_rather_than_guessing(builder):
    """Estimating the spread needs deep prefixes. A new reaction has none, and
    inventing one would misdirect the whole budget from round one."""
    connection = _bank({identifier: [0.03] for identifier in range(50)})
    assert builder.expected_gain_weights(
        connection, 3, [(0,), (1,)], 0.080) is None


def test_the_weighting_is_selectable_and_defaults_to_the_measured_one(builder):
    parsed = builder.__dict__["argparse"].ArgumentParser  # module imported argparse
    assert "expected-gain" in (ROOT / "tools" / "bank_builder.py").read_text()
    import subprocess
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "bank_builder.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0
    assert "--anchor-weighting" in done.stdout


def test_every_weight_is_finite_and_positive(builder):
    """A zero weight makes a neighbourhood unreachable forever; random.choices
    raises on a zero total."""
    prefixes = _population(good_mean=0.075, bad_mean=0.005)
    connection = _bank(prefixes)
    weights = builder.expected_gain_weights(
        connection, 3, [(key,) for key in prefixes], 0.30)  # cut far above everything
    assert weights is not None
    assert all(w > 0.0 and math.isfinite(w) for w in weights)
    assert sum(weights) > 0.0
