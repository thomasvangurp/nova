import json

import pytest

from nanobody_opt import score_boltzgen_fast_adapter as adapter


def _args(tmp_path, **overrides):
    import argparse

    sequences = tmp_path / "wave.json"
    if not sequences.exists():
        sequences.write_text(
            json.dumps({"target": "P20809", "sequences": ["AAAA", "BBBB"]})
        )
    defaults = dict(
        sequences=sequences,
        config=tmp_path / "config.yaml",
        target="P20809",
        gpus="0,1",
        shards_per_gpu=1,
        batch_size=2,
        work_root=tmp_path / "work",
        out=tmp_path / "out.json",
        structure_dir=tmp_path / "structures",
        require_structures=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_contract_flags_are_translated_to_the_fast_scorer(tmp_path):
    command = adapter.build_command(_args(tmp_path), {})

    assert "--work-root" not in command
    assert "--structure-dir" not in command
    assert command[command.index("--shard-root") + 1] == str(tmp_path / "work")
    assert command[command.index("--capture-structures") + 1] == str(
        tmp_path / "structures"
    )
    assert "--require-structures" in command
    # everything the contract fixes is passed through untouched
    assert command[command.index("--batch-size") + 1] == "2"
    assert command[command.index("--gpus") + 1] == "0,1"
    assert command[command.index("--out") + 1] == str(tmp_path / "out.json")


def test_structures_are_not_requested_when_the_contract_omits_them(tmp_path):
    command = adapter.build_command(
        _args(tmp_path, structure_dir=None, require_structures=False), {}
    )

    assert "--capture-structures" not in command
    assert "--require-structures" not in command


def test_grafting_reference_is_used_only_for_one_candidate_length(tmp_path):
    settings = {"synth_reference": "/ref/design.cif"}

    same = adapter.build_command(_args(tmp_path), settings)
    assert same[same.index("--synth-reference") + 1] == "/ref/design.cif"

    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({"sequences": ["AAAA", "BBB"]}))
    command = adapter.build_command(_args(tmp_path, sequences=mixed), settings)
    assert "--synth-reference" not in command


def test_tuning_knobs_come_from_box_settings(tmp_path):
    command = adapter.build_command(
        _args(tmp_path),
        {"fold_sampling_steps": 200, "fold_diffusion_samples": 5},
    )

    assert command[command.index("--fold-sampling-steps") + 1] == "200"
    assert command[command.index("--fold-diffusion-samples") + 1] == "5"


def test_absent_settings_file_is_not_an_error(tmp_path):
    assert adapter.load_settings(tmp_path / "missing.json") == {}


def test_settings_file_must_be_an_object(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(["not", "an", "object"]))

    with pytest.raises(ValueError, match="must be an object"):
        adapter.load_settings(path)


def test_a_modified_scorer_is_refused(tmp_path):
    scorer = tmp_path / "fast_score_batch.py"
    scorer.write_text("print('validated')")
    import hashlib

    digest = hashlib.sha256(scorer.read_bytes()).hexdigest()

    adapter.require_pinned_scorer(scorer, digest)  # unchanged: allowed

    scorer.write_text("print('edited mid-epoch')")
    with pytest.raises(SystemExit, match="has been modified"):
        adapter.require_pinned_scorer(scorer, digest)


def test_an_unpinned_scorer_is_allowed(tmp_path):
    scorer = tmp_path / "fast_score_batch.py"
    scorer.write_text("print('dev tree')")

    assert adapter.require_pinned_scorer(scorer, None) is None
