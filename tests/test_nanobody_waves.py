import json
import sys
from pathlib import Path

import pytest

from nanobody_opt import score_boltzgen_waves as waves
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL


def test_waves_are_promoted_and_resumed(tmp_path, monkeypatch):
    sequences = [f"SEQUENCE{index:02d}" for index in range(17)]
    source = tmp_path / "sequences.json"
    source.write_text(json.dumps({"sequences": sequences}))
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    output_dir = tmp_path / "waves"
    calls = []

    def fake_run(command, check):
        assert check is True
        calls.append(command)
        wave_input = Path(command[command.index("--sequences") + 1])
        output = Path(command[command.index("--out") + 1])
        target = command[command.index("--target") + 1]
        assert Path(command[command.index("--work-root") + 1]) == output_dir / "work"
        items = json.loads(wave_input.read_text())["sequences"]
        output.write_text(
            json.dumps(
                {
                    "target": target,
                    "n": len(items),
                    "scorer_protocol": SCORER_PROTOCOL,
                    "per_nanobody_components": {
                        "0": {item: {target: {}} for item in items}
                    },
                }
            )
        )

    monkeypatch.setattr(waves.subprocess, "run", fake_run)
    arguments = [
        "score_boltzgen_waves.py",
        "--sequences",
        str(source),
        "--config",
        str(config),
        "--target",
        "P20809",
        "--output-dir",
        str(output_dir),
        "--wave-size",
        "8",
        "--predict-batch-size",
        "8",
        "--limit",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", arguments)

    waves.main()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    combined = json.loads((output_dir / "exact_all.json").read_text())
    assert len(calls) == 2
    assert manifest["completed_count"] == 10
    assert manifest["observation_id"] == combined["observation_id"]
    assert [item["count"] for item in manifest["completed_waves"]] == [8, 2]
    assert combined["n"] == 10
    assert len(combined["per_nanobody_components"]["0"]) == 10

    waves.main()

    assert len(calls) == 2
    resumed = json.loads((output_dir / "exact_all.json").read_text())
    assert resumed["observation_id"] == combined["observation_id"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_mixed_length_admission_scores_every_candidate_and_resumes(
    tmp_path, monkeypatch, interrupted,
):
    from types import SimpleNamespace
    from nanobody_opt import score_boltzgen_fast_adapter as adapter
    from nanobody_opt.diffusion_sample_selection import EXPECTED_DIFFUSION_SAMPLES

    sequences = ["AAAA", "CCCCCC", "DDDD", "EEEEEE", "FFFF", "GGGGGG", "HHHHHH", "IIII"]
    source = tmp_path / "sequences.json"
    source.write_text(json.dumps({"sequences": sequences}))
    output_dir = tmp_path / "waves"
    calls, completed = [], []

    def run(command, check):
        def value(flag):
            return command[command.index(flag) + 1]
        items = json.loads(Path(value("--sequences")).read_text())["sequences"]
        calls.append(items)
        # Exercise the adapter's actual one-length precondition, not a copy.
        adapter.build_command(SimpleNamespace(
            sequences=Path(value("--sequences")), config=Path(value("--config")),
            target=value("--target"), gpus=value("--gpus"), shards_per_gpu=1,
            batch_size=2, work_root=Path(value("--work-root")), out=Path(value("--out")),
            structure_dir=Path(value("--structure-dir")), require_structures=True,
        ), {"synth_reference": "/synthetic/reference"})
        if interrupted and len(calls) == 2:
            raise RuntimeError("interrupted after a completed wave")
        completed.extend(items)
        Path(value("--out")).write_text(json.dumps({
            "target": "P20809", "n": len(items), "scorer_protocol": SCORER_PROTOCOL,
            "per_nanobody_components": {"0": {s: {"P20809": {}} for s in items}},
            "structures": {s: {"P20809": {
                "path": "/synthetic/confidence.cif", "writer_path": "/synthetic/writer.cif",
                "sample_selection": {"sample_count": EXPECTED_DIFFUSION_SAMPLES,
                                     "writer_sample_index": 0, "analyze_sample_index": 1},
            }} for s in items},
        }))

    monkeypatch.setattr(waves.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [
        "waves", "--sequences", str(source), "--config", str(tmp_path / "config.yaml"),
        "--target", "P20809", "--output-dir", str(output_dir), "--wave-size", "2",
        "--predict-batch-size", "2", "--limit", "7", "--capture-structures",
    ])
    if interrupted:
        with pytest.raises(RuntimeError, match="interrupted"):
            waves.main()
        partial = json.loads((output_dir / "manifest.json").read_text())
        assert partial["completed_count"] == 2
    waves.main()
    combined = json.loads((output_dir / "exact_all.json").read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert combined["n"] == 7
    assert combined["wave_count"] == 4
    assert len(completed) == 7 and set(completed) == set(sequences[:7])
    assert set(combined["structures"]) == set(sequences[:7])
    assert [row["start"] for row in manifest["completed_waves"]] == [0, 2, 3, 5]
    assert manifest["completed_count"] == 7
    if interrupted:
        assert manifest["observation_id"] == partial["observation_id"]
    previous_calls = len(calls)
    waves.main()
    assert len(calls) == previous_calls
