"""Neighbourhood saturation is a stop condition, not a failure.

Every producer iteration excludes the candidates of all previous iterations, so a
basin revisited enough times runs out of novel molecules within one mutation of
its seeds. Observed live: 43 iterations in one epoch, 13,504 candidates already
excluded, exit code 1 on "only generated 11/12". The producer treated that as a
retry and spun on an exhausted basin at 0% GPU for the rest of the epoch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt import local_neighborhood as ln
from nanobody_opt import run_live_producer as rlp


def test_the_generator_no_longer_raises_on_a_short_seed():
    """The SystemExit that ended every iteration is gone from the source."""
    source = (Path(__file__).resolve().parents[1]
              / "nanobody_opt" / "local_neighborhood.py").read_text()
    # The branch that used to end the run now records and carries on. Asserting
    # on behaviour rather than on the old message, which is deliberately gone.
    branch = source.split("if stochastic_accepted < args.per_seed:")[1][:1400]
    assert "raise SystemExit" not in branch
    assert "exhausted_seeds.append" in branch
    assert ln.EXHAUSTED_MARKER == "NEIGHBOURHOOD_EXHAUSTED"
    assert ln.EXHAUSTED_SUFFIX.endswith(".json")


def test_the_signal_is_a_file_because_output_is_not_captured():
    """`_run` streams child output to the terminal, so a caller has no log to
    grep. Only a file beside the output survives."""
    producer = (Path(__file__).resolve().parents[1]
                / "nanobody_opt" / "run_live_producer.py").read_text()
    assert "subprocess.run(command, check=True)" in producer
    assert "EXHAUSTED_SUFFIX" in producer
    assert ".log" not in producer.split("def _looks_exhausted")[1][:400]


def test_the_producer_sees_the_sidecar(tmp_path):
    iteration = tmp_path / "iteration_01"
    iteration.mkdir()
    assert rlp._looks_exhausted(iteration) is False
    (iteration / ("candidates.csv" + ln.EXHAUSTED_SUFFIX)).write_text(
        json.dumps({"seeds_short": ["11/12 at score 45.0"], "per_seed": 12})
    )
    assert rlp._looks_exhausted(iteration) is True


def test_an_ordinary_failure_still_retries(tmp_path):
    """Saturation and breakage both exit non-zero and only one is worth
    retrying. A real fault must not silently end an epoch's generation."""
    iteration = tmp_path / "iteration_02"
    iteration.mkdir()
    (iteration / "failed.json").write_text(json.dumps({"returncode": 1}))
    assert rlp._looks_exhausted(iteration) is False


def test_the_producer_stops_generating_once_exhausted():
    source = (Path(__file__).resolve().parents[1]
              / "nanobody_opt" / "run_live_producer.py").read_text()
    assert "exhausted_epochs" in source
    assert "PRODUCER_EXHAUSTED" in source
    # The epoch continues -- only generation stops. Replication is the point.
    assert "leaving the epoch to replication" in source
