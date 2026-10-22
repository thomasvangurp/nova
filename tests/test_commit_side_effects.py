"""Highest-value regressions: bugs 3/4 (false success and stale replay), 9.

Import the real Pi CLI, never the producer. All inputs are synthetic and all
chain/git/subprocess seams are replaced before invoking main().
"""
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOW = 2_000_000_000


def fresh_artifact():
    return {"epoch": 100, "updated_at_unix": NOW, "sequence_pool": ["K" * 120],
            "submissions": [{"uid": 7, "sequence": "K" * 120}]}


@pytest.fixture
def submit(monkeypatch):
    spec = importlib.util.spec_from_file_location("_sn68_pi_submit", ROOT / "deploy/submit_host/pi_submit.py")
    module = importlib.util.module_from_spec(spec)
    # pi_submit inserts search paths at import; restore them after the test.
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "time", lambda: NOW)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    def forbidden(*args, **kwargs):
        pytest.fail("unexpected subprocess/network/git operation")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    monkeypatch.setattr(module, "pull_producer", forbidden)
    monkeypatch.setattr(module, "blocks_left", lambda *a, **k: 100)
    module._real_commit_batch = module.commit_batch
    monkeypatch.setattr(module, "commit_batch", forbidden)
    return module


@pytest.fixture
def isolated_preflight(monkeypatch):
    # The current checkout's Pi expects validate_plaintext, but the local
    # commit helper only exposes submission_message. Isolate that API drift
    # so it cannot mask the zero-side-effect regression; test it separately.
    from nanobody_opt import commit_submission
    def validate(message):
        molecules, sequence = message.split("|")
        assert commit_submission.submission_message(sequence, molecules) == message
    monkeypatch.setattr(commit_submission, "validate_plaintext", validate, raising=False)


@pytest.fixture
def arguments(tmp_path):
    crew = tmp_path / "crew.json"
    artifact = tmp_path / "nanobody.json"
    crew.write_text(json.dumps([[7, "synthetic-public-hotkey"]]))
    artifact.write_text(json.dumps(fresh_artifact()))
    return ["--epoch", "100", "--reaction", "1", "--crew", str(crew),
            "--nanobody", str(artifact), "--fleet", str(tmp_path / "absent.json")]


def test_bug3_parallel_zero_commits_is_failure(submit, arguments, monkeypatch, isolated_preflight):
    """Bug 3: failures==0 does not imply a single intended commit landed."""
    def batch(rows, *args):
        assert len(rows) == 1  # planned > 0, not an empty-input test
        return 0, 0, [7]
    monkeypatch.setattr(submit, "commit_batch", batch)
    assert submit.main(arguments) != 0


def test_bug3_serial_blackout_zero_commits_is_failure(submit, arguments, monkeypatch, isolated_preflight):
    """Bug 3: blackout before the first commit must return nonzero too."""
    monkeypatch.setattr(submit, "blocks_left", lambda *a, **k: 0)
    assert submit.main(arguments + ["--serial"]) != 0


def test_bug3_success_requires_a_commit(submit, arguments, monkeypatch, isolated_preflight):
    """Bug 3 positive control: a landed batch remains successful."""
    monkeypatch.setattr(submit, "commit_batch", lambda *a: (1, 0, []))
    assert submit.main(arguments) == 0


def test_bug3_real_preflight_api_exists():
    """Bug 3 integration prerequisite: real CLI preflight must be callable."""
    from nanobody_opt import commit_submission
    assert callable(getattr(commit_submission, "validate_plaintext", None)), "Pi main imports missing validate_plaintext"


@pytest.mark.parametrize("epoch", [99, 1, None])
def test_bug4_stale_or_unversioned_artifact_rejected(submit, epoch):
    """Bug 4: ordinary submission cannot replay old/missing-epoch artifacts."""
    artifact = fresh_artifact()
    artifact.pop("epoch")
    if epoch is not None:
        artifact["epoch"] = epoch
    with pytest.raises(ValueError):
        submit.sequences_by_uid(artifact, 100)


def test_bug9_current_crew_replaces_stale_uid_assignment(submit):
    """Bug 9: an artifact cannot bind a sequence to a deregistered UID."""
    artifact = fresh_artifact()
    assert submit.sequences_by_uid(artifact, 100, crew=[(8, "new-public-hotkey")]) == {8: "K" * 120}


def test_bug3_all_rows_stale_is_failure(submit, arguments, monkeypatch, isolated_preflight):
    """Bugs 3/9: real commit_batch sees every UID owned by a different hotkey."""
    from nanobody_opt import fleet
    def forbidden(*a, **k):
        pytest.fail('encryption/git/chain write must not occur for stale rows')
    bt = ModuleType('bittensor')
    bt.Subtensor = lambda **k: SimpleNamespace(metagraph=lambda netuid: SimpleNamespace(hotkeys=['other-public-hotkey'] * 8))
    core = ModuleType('bittensor.core')
    errors = ModuleType('bittensor.core.errors')
    errors.MetadataError = type('MetadataError', (Exception,), {})
    btdr = ModuleType('btdr')
    btdr.QuicknetBittensorDrandTimelock = forbidden
    files = ModuleType('files')
    files.load_github_path = files.push_submissions = forbidden
    for name, module in [('bittensor', bt), ('bittensor.core', core),
                         ('bittensor.core.errors', errors), ('btdr', btdr), ('files', files)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(fleet, 'local_hotkey_addresses', lambda *a: {})
    batches = []
    def real_batch(rows, args, environment):
        result = submit._real_commit_batch(rows, args, environment)
        batches.append(result)
        return result
    monkeypatch.setattr(submit, 'commit_batch', real_batch)
    assert submit.main(arguments) != 0
    assert batches == [(0, 0, [7])]


@pytest.mark.parametrize('pool', [[], [''], [{'sequence': None}]])
def test_bug4_right_epoch_but_empty_pool_rejected(submit, pool):
    """Bugs 4/10: current epoch is insufficient without usable pool evidence."""
    artifact = fresh_artifact()
    artifact['sequence_pool'] = pool
    with pytest.raises(ValueError):
        submit.sequences_by_uid(artifact, 100)


@pytest.mark.parametrize('timestamp', [NOW - 86400, NOW + 86400])
def test_bug4_freshness_age_bound_even_with_epoch_override(submit, timestamp):
    """Bug 4: a VALID-but-out-of-window stamp still rejects even on override."""
    artifact = fresh_artifact()
    artifact['updated_at_unix'] = timestamp
    with pytest.raises(ValueError):
        submit.sequences_by_uid(artifact, 100, allow_stale=True)


@pytest.mark.parametrize('timestamp', [None, float('nan')])
def test_bug4_missing_stamp_falls_through_to_epoch_field(submit, timestamp):
    """Bug 4 (format-robust): a missing/non-finite stamp does NOT hard-fail;
    it falls through to the always-present epoch field, and only a genuinely
    stale epoch (age > STALE_AGE_LIMIT) rejects -- not the absent stamp."""
    artifact = fresh_artifact()
    artifact['updated_at_unix'] = timestamp
    # Current epoch with a missing stamp is accepted: epoch == target.
    assert submit.sequences_by_uid(artifact, 100, allow_stale=True) == {7: 'K' * 120}
    # Only a genuinely stale epoch rejects, via the epoch field alone.
    artifact['epoch'] = 100 - (submit.STALE_AGE_LIMIT + 1)
    with pytest.raises(ValueError):
        submit.sequences_by_uid(artifact, 100, allow_stale=True)


@pytest.mark.parametrize('skew', [-600, 600])
def test_bug4_generous_writer_clock_skew_allowed(submit, skew):
    """Bug 4 positive control: two-host clock drift is not stale evidence."""
    artifact = fresh_artifact()
    artifact['updated_at_unix'] += skew
    assert submit.sequences_by_uid(artifact, 100, crew=[(7, 'public-hotkey')]) == {7: 'K' * 120}


def test_bug4_writer_stamps_artifact_on_writing_host(monkeypatch):
    """Bug 4: real shipping publisher artifact builder supplies freshness."""
    from nanobody_opt import fleet
    monkeypatch.setattr(fleet.time, 'time', lambda: NOW)
    artifact = fleet.fleet_artifact(target='fixture', epoch=100, stage='qualified',
                                   assignments=[], checkpoints={}, sequence_pool=['K' * 120])
    assert artifact['updated_at_unix'] == NOW


# ---------------------------------------------------------------------------
# Cross-boundary schema contract (bug 4 / C7): every field the Pi reader reads
# from the artifact must be produced by the fleet writer.
# ---------------------------------------------------------------------------
def reader_required_keys() -> set[str]:
    """AST-extract every top-level artifact key ``sequences_by_uid`` reads.

    Scans ``deploy/submit_host/pi_submit.py`` for ``artifact.get("...")`` and
    ``artifact["..."]`` string-literal keys inside ``sequences_by_uid``.
    """
    tree = ast.parse((ROOT / "deploy/submit_host/pi_submit.py").read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "sequences_by_uid")
    arg = node.args.args[0].arg  # "artifact"
    keys: set[str] = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == arg
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            keys.add(n.args[0].value)
        elif (isinstance(n, ast.Subscript)
                and isinstance(n.value, ast.Name) and n.value.id == arg
                and isinstance(n.slice, ast.Constant)
                and isinstance(n.slice.value, str)):
            keys.add(n.slice.value)
    return keys


def writer_produced_keys() -> set[str]:
    """AST-extract the top-level keys the fleet writer emits into the artifact.

    ``fleet_artifact`` returns a literal dict; its keys are the writer's output
    schema. (The producer ``nanobody_opt_v2/run_live_producer.py`` does not
    write ``nanobody_result.json`` -- the publisher calls
    ``nanobody_opt/fleet.py:fleet_artifact``.)
    """
    tree = ast.parse((ROOT / "nanobody_opt/fleet.py").read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "fleet_artifact")
    for n in ast.walk(node):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
            return {
                k.value for k in n.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    return set()


def test_bug4_reader_requires_only_writer_produced_fields():
    """Bug 4/C7 contract: reader-required fields ⊆ writer-produced fields.

    A future reader check that reads a field the writer never emits fails here
    with the exact diff, instead of surfacing as a runtime stale/missing
    artifact only after deploy.
    """
    required = reader_required_keys()
    produced = writer_produced_keys()
    missing = required - produced
    assert not missing, (
        f"reader requires fields the writer does not produce: {sorted(missing)}; "
        f"reader={sorted(required)} writer={sorted(produced)}")


def test_bug4_writer_artifact_feeds_reader_without_raising(submit, monkeypatch):
    """Bug 4/C7 dynamic round-trip: the real fleet writer output (epoch present,
    no hand-written fixture) passes the Pi reader unchanged."""
    from nanobody_opt import fleet
    monkeypatch.setattr(fleet.time, 'time', lambda: NOW)
    artifact = fleet.fleet_artifact(
        target='fixture', epoch=100, stage='qualified',
        assignments=[], checkpoints={}, sequence_pool=['K' * 120],
    )
    assert submit.sequences_by_uid(artifact, 100, crew=[(7, 'public-hotkey')]) == {7: 'K' * 120}
