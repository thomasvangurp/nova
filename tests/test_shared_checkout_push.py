"""The submission push must not re-clone the results repo for every uid.

Cloning per uid cost ~73s each; at ~6 blocks apiece a twelve-uid fleet ran off
the end of the epoch and the tail committed inside the validator's blackout,
silently. These tests use a real local git remote -- the mechanics are the
whole point, so mocking them would test nothing.
"""
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The git mechanics under test need none of these, and requiring a bittensor
# install to test `git reset --hard` would mean never running this locally.
for absent in ("bittensor", "requests", "dotenv"):
    if absent not in sys.modules:
        try:
            __import__(absent)
        except ImportError:
            stub = types.ModuleType(absent)
            stub.logging = types.SimpleNamespace(error=lambda *a, **k: None)
            stub.load_dotenv = lambda *a, **k: None
            sys.modules[absent] = stub

spec = importlib.util.spec_from_file_location("files_under_test", ROOT / "utils/files.py")
files = importlib.util.module_from_spec(spec)
sys.modules["files_under_test"] = files
try:
    spec.loader.exec_module(files)
except Exception as error:                                   # pragma: no cover
    pytest.skip(f"utils/files.py needs bittensor: {error}", allow_module_level=True)

ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0",
           GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
           GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


def _remote(tmp_path):
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", "-b", "main", str(bare)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "--quiet", str(bare), str(seed)], check=True, env=ENV)
    (seed / "README").write_text("seed\n")
    for argv in (["git", "add", "-A"], ["git", "commit", "--quiet", "-m", "seed"],
                 ["git", "push", "--quiet", "origin", "HEAD:refs/heads/main"]):
        subprocess.run(argv, cwd=seed, check=True, env=ENV)
    return str(bare)


def test_the_checkout_is_created_once_and_then_reused(tmp_path):
    remote = _remote(tmp_path)
    work = tmp_path / "work"; work.mkdir()
    first = files._shared_checkout(remote, "main", work, ENV)
    assert first is not None and (first / ".git").is_dir()
    marker = first / ".git" / "created-by-this-test"
    marker.write_text("x")
    second = files._shared_checkout(remote, "main", work, ENV)
    # Same directory, not a new clone: that is the entire saving.
    assert second == first and marker.is_file()


def test_two_submissions_reuse_one_checkout_and_both_land(tmp_path):
    remote = _remote(tmp_path)
    work = tmp_path / "work"; work.mkdir()
    for name, body in (("aaa", b"one"), ("bbb", b"two")):
        checkout = files._shared_checkout(remote, "main", work, ENV)
        assert files._write_and_push(checkout, Path("data/results"), name, body, "main", ENV)
    verify = tmp_path / "verify"
    subprocess.run(["git", "clone", "--quiet", remote, str(verify)], check=True, env=ENV)
    assert (verify / "data/results/aaa.txt").read_bytes() == b"one"
    assert (verify / "data/results/bbb.txt").read_bytes() == b"two"


def test_a_refresh_discards_local_drift(tmp_path):
    # reset --hard is what makes reuse equivalent to a fresh clone. Without it
    # one uid's leftovers could ride along into the next uid's commit.
    remote = _remote(tmp_path)
    work = tmp_path / "work"; work.mkdir()
    checkout = files._shared_checkout(remote, "main", work, ENV)
    (checkout / "STRAY").write_text("should not survive")
    (checkout / "README").write_text("locally mangled")
    checkout = files._shared_checkout(remote, "main", work, ENV)
    assert not (checkout / "STRAY").exists()
    assert (checkout / "README").read_text() == "seed\n"


def test_an_unusable_remote_reports_none_rather_than_raising(tmp_path):
    # The caller falls back to the fresh-clone path on None; an exception here
    # would take the whole commit down instead.
    work = tmp_path / "work"; work.mkdir()
    assert files._shared_checkout(str(tmp_path / "nope.git"), "main", work, ENV) is None


def test_rewriting_identical_content_is_a_no_op(tmp_path):
    remote = _remote(tmp_path)
    work = tmp_path / "work"; work.mkdir()
    checkout = files._shared_checkout(remote, "main", work, ENV)
    assert files._write_and_push(checkout, Path("data/results"), "same", b"x", "main", ENV)
    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                            capture_output=True, text=True, env=ENV).stdout
    checkout = files._shared_checkout(remote, "main", work, ENV)
    assert files._write_and_push(checkout, Path("data/results"), "same", b"x", "main", ENV)
    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                           capture_output=True, text=True, env=ENV).stdout
    assert before == after
