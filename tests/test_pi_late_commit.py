from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _loop():
    path = Path(__file__).resolve().parents[1] / "tools" / "pi_epoch_loop.py"
    spec = importlib.util.spec_from_file_location("pi_epoch_loop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_confirmation_counts_deep_submissions_for_this_epoch(tmp_path):
    loop = _loop()
    artifact = tmp_path / "nanobody_result.json"
    artifact.write_text(json.dumps({"epoch": 25400, "submissions": [
        {"sequence": "A", "observation_count": 12},
        {"sequence": "B", "observation_count": 8},
        {"sequence": "C", "observation_count": 3},
    ]}))
    assert loop.artifact_confirmation(artifact, 25400, 8) == (2, 3)
    # a stale or missing artifact never reads as confirmed
    assert loop.artifact_confirmation(artifact, 25401, 8) == (0, 0)
    assert loop.artifact_confirmation(tmp_path / "missing.json", 25400, 8) == (0, 0)


def test_publication_floor_matches_the_producers_publishable_depth():
    from nanobody_opt import publication_gates
    assert publication_gates.gate_defaults()["--minimum-observations"] == "8"
