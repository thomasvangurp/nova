"""Every operator tool must be able to print its own help.

`bank_builder.py --help` crashed with `ValueError: unsupported format
character ','` -- a bare `%` in a help string, which argparse expands as
`help % params`. It went unnoticed because the check for it was piped
(`--help | head`), and `$?` then reports the exit status of `head`.

That matters beyond tidiness: an operator reaching for --help is usually
mid-incident, and a tool that cannot describe its own arguments at that moment
is a tool whose arguments get guessed. The deploy playbooks run --help as a
gate for exactly this reason.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Operator entry points. Anything with an argparse parser belongs here; the
#: cost of one subprocess each is trivial next to a broken --help on the box.
TOOLS = [
    "bank_builder.py",
    "bank_release.py",
    "bank_rescore.py",
    "bank_sync.py",
    "bank_backup.py",
    "bank_box_control.py",
    "bank_box2_control.py",
    "submit_box_control.py",
    "check_staged_hotkeys.py",
]


@pytest.mark.parametrize("tool", TOOLS)
def test_help_parses_and_exits_zero(tool):
    path = ROOT / "tools" / tool
    if not path.is_file():
        pytest.skip(f"{tool} is not present in this tree")
    done = subprocess.run([sys.executable, str(path), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (
        f"{tool} --help exited {done.returncode}: {done.stderr.strip()[-400:]}")
    assert "usage:" in done.stdout


@pytest.mark.parametrize("tool", TOOLS)
def test_no_unescaped_percent_in_a_help_string(tool):
    """The specific defect, caught at source so it cannot come back quietly.

    argparse only expands `%` when it formats that action, so a help string
    added today breaks --help for the whole parser the first time someone runs
    it -- typically on the box, not here.
    """
    import ast

    path = ROOT / "tools" / tool
    if not path.is_file():
        pytest.skip(f"{tool} is not present in this tree")
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", None) != "add_argument":
            continue
        for keyword in node.keywords:
            if keyword.arg != "help":
                continue
            try:
                text = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                continue
            if not isinstance(text, str):
                continue
            # "%%" is the escape; "%(default)s" is argparse's own substitution.
            probe = text.replace("%%", "").replace("%(default)s", "")
            probe = probe.replace("%(prog)s", "").replace("%(type)s", "")
            if "%" in probe:
                offenders.append(text)
    assert not offenders, f"{tool} has an unescaped % in: {offenders}"
