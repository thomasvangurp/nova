import json
import subprocess
import sys

import pytest

from nanobody_opt import publication_gates as pg


def test_the_module_has_no_third_party_imports():
    # The live publisher runs in a different Pixi environment from the scorer.
    # A shared source of truth that only one side can import is not shared.
    source = open(pg.__file__).read()
    for forbidden in ("import numpy", "import pandas", "from numpy", "from pandas"):
        assert forbidden not in source


def test_it_imports_with_no_site_packages():
    completed = subprocess.run(
        [sys.executable, "-S", "-c",
         "import sys; sys.path.insert(0, '.'); "
         "from nanobody_opt import publication_gates as p; "
         "print(len(p.gate_arguments()))"],
        capture_output=True, text=True, cwd="/Users/tvgdeena/nova-nanobody",
    )
    assert completed.returncode == 0, completed.stderr


def test_the_evidence_floor_stays_at_one_until_the_producer_replicates():
    # Measured on the live score bank, a sequence's observation count equals the
    # number of epochs it appears in, exactly: the producer screens each
    # candidate once per epoch and never repeats one within an epoch.  Depth is
    # age, not replication, so any floor above one is a survivorship filter that
    # excludes every freshly generated candidate -- all 200 of epoch 25134's
    # were at depth 1.  Raise this only together with a producer that screens a
    # candidate more than once inside a single epoch.
    gates = pg.gate_defaults()
    assert gates["--minimum-observations"] == "1"
    assert gates["--minimum-structure-observations"] == "1"


def test_integrity_gates_are_not_relaxed():
    gates = pg.gate_defaults()
    assert gates["--minimum-median-contact-jaccard"] == "0.35"
    assert gates["--maximum-cdr3-rmsd"] == "5"


def test_overrides_accept_either_spelling():
    args = pg.gate_arguments({"maximum_p90_rank_sum": 250})
    assert args[args.index("--maximum-p90-rank-sum") + 1] == "250"
    args = pg.gate_arguments({"--maximum-p90-rank-sum": 251})
    assert args[args.index("--maximum-p90-rank-sum") + 1] == "251"


def test_an_unknown_gate_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown publication gate"):
        pg.gate_arguments({"maximum_p90_invented": 1})


def test_overrides_load_from_either_shape(tmp_path):
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"maximum_p90_rank_sum": 104}))
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(json.dumps({"gates": {"maximum_p90_rank_sum": 104}}))

    assert pg.load_overrides(flat) == {"maximum_p90_rank_sum": 104}
    assert pg.load_overrides(wrapped) == {"maximum_p90_rank_sum": 104}
    assert pg.load_overrides(None) == {}
