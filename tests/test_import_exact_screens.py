import json

import numpy as np
from nanobody_opt.import_exact_screens import (
    discover_exact_files,
    epoch_from_path,
    exact_observations,
    validator_replay_distribution,
)
from nanobody_opt.simulate_round import METRICS


def _metrics(value):
    return {metric: float(value) for metric in METRICS}


def test_discovery_deduplicates_copies_and_ignores_wave_fragments(tmp_path):
    payload = {
        "target": "P20809",
        "per_nanobody_components": {"0": {"AAAA": {"P20809": _metrics(1)}}},
    }
    first = tmp_path / "epoch25000" / "exact_all.json"
    first.parent.mkdir()
    first.write_text(json.dumps(payload))
    copy = tmp_path / "legacy_exact.json"
    copy.write_text(json.dumps(payload))
    wave = first.parent / "wave_000_exact.json"
    wave.write_text(json.dumps(payload))

    assert discover_exact_files([tmp_path]) == [first]
    assert epoch_from_path(first) == 25000


def test_exact_import_preserves_raw_metrics_parent_and_repeat_identity(tmp_path):
    exact = tmp_path / "epoch25001" / "exact_all.json"
    exact.parent.mkdir()
    exact.write_text(json.dumps({
        "target": "P20809",
        "observation_id": "run-one",
        "per_nanobody_components": {
            "0": {"AAAC": {"P20809": _metrics(5)}}
        },
        "structures": {
            "AAAC": {"P20809": {"path": "candidate.cif", "sha256": "abc"}}
        },
    }))
    fields = [{metric: np.asarray([1.0, 10.0]) for metric in METRICS}]
    observations, summary = exact_observations(
        [exact],
        target="P20809",
        fields=fields,
        metadata={(25001, "AAAC"): {
            "parent_sequence": "AAAA",
            "proposal_type": "single",
            "mutations": "A4C",
        }},
        archive_sha256="archive",
    )

    assert summary["complete_observations"] == 1
    assert summary["with_parent_metadata"] == 1
    assert observations[0]["observation_id"] == "local-exact:run-one:0:AAAC"
    assert observations[0]["metrics"] == _metrics(5)
    assert observations[0]["parent_sequence"] == "AAAA"
    assert observations[0]["structure"]["sha256"] == "abc"


def test_fast_replay_matches_dense_rank_direction():
    fields = [{
        metric: np.asarray([1.0, 5.0, 10.0]) for metric in METRICS
    }]
    # Five maximize metrics score rank two; five minimize metrics also rank two.
    assert validator_replay_distribution(_metrics(5), fields) == [20.0]
