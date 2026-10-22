"""Bugs 1/10: test the gate itself so changes cannot silently disable it."""
import importlib.util

import pytest

from sn68_source import ROOT

spec = importlib.util.spec_from_file_location("sn68_ci_guardrails", ROOT / "scripts/ci_guardrails.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("statement", ["raise SystemExit(1)", "raise KeyboardInterrupt",
                                      "raise GeneratorExit()", "raise BaseException()",
                                      "sys.exit(1)", "os._exit(1)", "builtins.exit(1)",
                                      "exit(1)", "quit(1)"])
def test_bug1_bug10_all_exit_forms_fail(statement):
    assert gate.inspect_source(f"def worker():\n    {statement}\n")[0]


def test_bug1_comments_strings_and_other_methods_do_not_fail():
    assert gate.inspect_source('# SystemExit\nmessage = "sys.exit(1)"\nobj.exit()\n') == ([], [])


@pytest.mark.parametrize("suffix,warns", [
    ("", True),
    ("except SystemExit:\n    raise\n", False),
    ("except BaseException:\n    raise\n", False),
    ("except BaseException:\n    pass\n", True),
])
def test_bug1_narrow_catch_warns_unless_followed_by_reraise(suffix, warns):
    source = "try:\n    work()\nexcept Exception as error:\n    recover(error)\n" + suffix
    assert bool(gate.inspect_source(source)[1]) is warns


def test_bug1_gate_cli_returns_failure_and_warnings_are_nonfatal(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("raise SystemExit(1)\n")
    assert gate.main([str(worker)]) == 1
    worker.write_text("try:\n    work()\nexcept Exception:\n    pass\n")
    assert gate.main([str(worker)]) == 0


def test_bug1_ast_distinguishes_handlers_and_ordinary_raises():
    source = 'try:\n    work()\nexcept SystemExit:\n    recover()\nraise ValueError("retry")\n'
    assert gate.inspect_source(source) == ([], [])


def test_bug1_ast_catches_aliases_and_custom_fatal_subclasses():
    source = ('import sys as system\nfrom builtins import exit as die\n'
              'class WorkerDeath(BaseException):\n    pass\n'
              'system.exit(1)\ndie(1)\nraise WorkerDeath()\n')
    assert len(gate.inspect_source(source)[0]) == 3


@pytest.mark.parametrize('relative,source', [
    ('deploy/submit_host/pi_submit.py', 'if __name__ == "__main__":\n    raise SystemExit(main())\n'),
    ('nanobody_opt_v2/validator_parent_gate.py', 'if __name__ == "__main__":\n    try:\n        main()\n    except BaseException:\n        os._exit(1)\n'),
    ('nanobody_opt_v2/run_live_producer.py', 'def main():\n    if args.publication_prepare_blocks < 0:\n        raise SystemExit("publication preparation guard must be nonnegative")\n'),
])
def test_bug1_reviewed_exit_allowlist_is_context_bound(relative, source):
    assert not gate.inspect_source(source, relative)[0]
    assert gate.inspect_source(source, 'another_worker.py')[0]


def test_bug1_allowlist_cannot_hide_exit_in_worker():
    assert gate.inspect_source('def worker():\n    raise SystemExit(main())\n', 'deploy/submit_host/pi_submit.py')[0]


def test_bug6_scan_targets_shipping_module_split():
    paths = {p.relative_to(ROOT).as_posix() for p in gate.WORKERS}
    assert 'nanobody_opt_v2/run_live_producer.py' in paths
    assert 'nanobody_opt/run_live_producer.py' not in paths
    for worker in ('local_neighborhood', 'check_validator_novelty', 'filter_public_neighborhood', 'fetch_hf_archive'):
        assert f'nanobody_opt/{worker}.py' in paths
