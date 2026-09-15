"""Every entry point must build its parser.

argparse formats help strings with %-substitution, so a literal percent in help
text raises at parser-construction time -- `~27% that` becomes an unsupported
format character.  That is invisible until something calls --help, which for the
producer is the deploy's verify_entrypoint probe, i.e. after a push and a sync.
This catches it in the suite instead.
"""
import subprocess
import sys

import pytest

ENTRY_POINTS = (
    "nanobody_opt.sequential_exact_race",
    "nanobody_opt.local_neighborhood",
    "nanobody_opt.run_live_producer",
    "nanobody_opt.rank_exact",
    "nanobody_opt.slot_manager",
    "nanobody_opt.score_boltzgen_waves",
)


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_the_parser_builds_and_help_renders(module):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"{module} --help failed:\n{completed.stderr[-600:]}"
    )
    assert completed.stdout.strip(), f"{module} --help printed nothing"
