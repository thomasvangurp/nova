import json
import sys
from pathlib import Path

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
