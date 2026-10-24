import json

import pytest

from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL
from nanobody_opt.score_boltzgen_waves import (
    _archive_legacy_wave,
    _validated_wave,
)


def _payload(sequence="AAAA"):
    return {
        "target": "P20809",
        "n": 1,
        "scorer_protocol": SCORER_PROTOCOL,
        "per_nanobody_components": {
            "0": {sequence: {"P20809": {"design_iiptm": 0.8}}}
        },
        "structures": {
            sequence: {
                "P20809": {
                    "path": "/confidence.cif",
                    "writer_path": "/writer.cif",
                    "sample_selection": {
                        "sample_count": 5,
                        "writer_sample_index": 0,
                        "analyze_sample_index": 1,
                    },
                }
            }
        },
    }


def test_validated_wave_requires_dual_sample_protocol(tmp_path):
    path = tmp_path / "wave.json"
    payload = _payload()
    payload.pop("scorer_protocol")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="dual-sample"):
        _validated_wave(path, ["AAAA"], "P20809", require_structures=True)


def test_validated_wave_accepts_both_retained_poses(tmp_path):
    path = tmp_path / "wave.json"
    payload = _payload()
    path.write_text(json.dumps(payload))

    assert _validated_wave(
        path, ["AAAA"], "P20809", require_structures=True
    ) == payload


def test_legacy_wave_is_preserved_before_refresh(tmp_path):
    path = tmp_path / "wave.json"
    path.write_text("legacy")

    archived = _archive_legacy_wave(path)

    assert not path.exists()
    assert archived.read_text() == "legacy"


def test_wave_with_fewer_diffusion_samples_is_refused(tmp_path):
    path = tmp_path / "wave.json"
    payload = _payload()
    payload["structures"]["AAAA"]["P20809"]["sample_selection"]["sample_count"] = 2
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="maximum over samples"):
        _validated_wave(path, ["AAAA"], "P20809", require_structures=True)


def test_wave_with_the_upstream_diffusion_sample_count_is_accepted(tmp_path):
    path = tmp_path / "wave.json"
    payload = _payload()
    payload["structures"]["AAAA"]["P20809"]["sample_selection"]["sample_count"] = 5
    path.write_text(json.dumps(payload))

    assert _validated_wave(path, ["AAAA"], "P20809", require_structures=True)


def test_later_wave_failure_preserves_completed_evidence_without_finishing_dispatch(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    from nanobody_opt import score_boltzgen_waves as waves
    from nanobody_opt.exact_checkpoints import exact_paths
    from nanobody_opt_v2.sequential_exact_race import _pooled_exact_paths
    from tools.publish_current_nanobody import _current_epoch_exact_inputs
    source = tmp_path / 'candidates.json'
    source.write_text(json.dumps({'sequences': ['AAAA', 'CCCCC']}))
    out = tmp_path / 'live/epoch100/producer/iteration_01/race/screen/tranche_01'
    monkeypatch.setattr(sys, 'argv', ['waves', '--sequences', str(source), '--config', 'unused',
        '--target', 'P20809', '--output-dir', str(out)])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, ['fake-scorer'])
        Path(command[command.index('--out') + 1]).write_text(json.dumps(_payload()))
    monkeypatch.setattr(waves.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError): waves.main()
    partial = out / 'exact_partial_000.json'
    data = json.loads(partial.read_text())
    assert data['n'] == 1 and data['complete'] is False
    assert not (out / 'exact_all.json').exists()
    assert _current_epoch_exact_inputs(tmp_path, 100) == (partial,)
    assert _pooled_exact_paths([], tmp_path / 'live/epoch100/producer', tmp_path / 'later') == [partial]
    # Finishing the same run never counts the checkpoint as another pass.
    complete = out / 'exact_all.json'
    complete.write_text(json.dumps({**data, 'complete': True}))
    assert exact_paths(out) == [complete]
