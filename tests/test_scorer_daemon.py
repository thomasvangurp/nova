"""job_from_command's --sequences parsing must match fast_score_batch.py's own.

Regression for the 2026-10-10 bug: job_from_command used to hand
{"sequence": ...} dict items straight through as "sequences" instead of
extracting the string, same as fast_score_batch.py's own main() does. The
worker's _deduplicate_nanobodies then tried to hash a dict, raised inside
run_chunk, logged "continuing" and silently dropped the sequence -- the
daemon answered "ok" with nothing scored, and _merge correctly reported
"MERGE LOST SEQUENCES: submitted N, merged 0". Every real caller
(score_boltzgen_waves._wave_command, the live race) writes dict items, so
this made every daemon-routed round fail and silently fall back to the slow
subprocess path -- the daemon provided zero speedup despite being enabled.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobody_opt import scorer_daemon as sd


def _command(sequences_path: Path, **extra: str) -> list[str]:
    command = [
        "/root/release_python.sh",
        "/root/fastscore/pinned-x/fast_score_batch.py",
        "--sequences", str(sequences_path),
        "--config", "/root/config.yaml",
        "--target", "P20809",
        "--gpus", "0,1",
        "--out", "/root/out.json",
    ]
    for flag, value in extra.items():
        command.extend([flag, value])
    return command


def test_job_from_command_extracts_dict_items(tmp_path: Path) -> None:
    """{"sequences": [{"sequence": "..."}]} -- score_boltzgen_waves's own shape."""
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": [
        {"sequence": "AAAA"}, {"sequence": "CCCC"},
    ]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA", "CCCC"]
    assert all(isinstance(s, str) for s in job["sequences"])


def test_job_from_command_accepts_bare_strings(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": ["AAAA", "CCCC"]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA", "CCCC"]


def test_job_from_command_accepts_bare_list(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps([{"sequence": "AAAA"}]))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA"]


def test_job_from_command_accepts_cases_key(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"cases": [{"sequence": "AAAA"}]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA"]


def test_job_from_command_rejects_empty_sequences(tmp_path: Path) -> None:
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": []}))
    with pytest.raises(ValueError, match="no sequences parsed"):
        sd.job_from_command(_command(path))


def test_job_from_command_drops_empty_items(tmp_path: Path) -> None:
    """A blank string item is dropped, matching fast_score_batch.py's own
    `isinstance(s, str) and s` filter (a dict item missing "sequence" is a
    KeyError in both, by design -- an outer CLI's argv must fail loudly, not
    parse into a plausible-looking job with silently wrong data)."""
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"sequences": [{"sequence": "AAAA"}, ""]}))
    job = sd.job_from_command(_command(path))
    assert job["sequences"] == ["AAAA"]
