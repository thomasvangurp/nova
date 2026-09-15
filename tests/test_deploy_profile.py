"""Deploy-time defaults that are load-bearing if a profile is forgotten."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _vars_block(path: Path) -> dict:
    text = path.read_text()
    match = re.search(r"^  vars:\n((?:    .*\n|\n)+)", text, re.M)
    assert match, f"no vars: block in {path.name}"
    return yaml.safe_load(match.group(1)) or {}


def test_the_default_exact_scorer_is_the_fast_adapter():
    """
    Both scorers implement the same validator protocol and produce the same
    metric vector, but score_boltzgen_batch.py runs the full BoltzGen pipeline
    at roughly 84 seconds of GPU per observation against the adapter's 0.62.
    The live profile has always overridden this; leaving the slow one as the
    DEFAULT meant a deploy that forgot the profile came up looking healthy while
    screening a hundredth of the candidates, which is indistinguishable from bad
    luck on the leaderboard.
    """
    defaults = _vars_block(ROOT / "ansible" / "nanobody.yml")
    assert defaults["nanobody_exact_scorer"].endswith(
        "score_boltzgen_fast_adapter.py"
    )


def test_the_profile_and_the_default_agree_on_the_scorer():
    """A profile that silently disagrees with the default is the same trap."""
    defaults = _vars_block(ROOT / "ansible" / "nanobody.yml")
    profile = yaml.safe_load(
        (ROOT / "ansible" / "vars" / "nanobody-high-throughput.yml").read_text()
    )
    assert profile["nanobody_exact_scorer"] == defaults["nanobody_exact_scorer"]
