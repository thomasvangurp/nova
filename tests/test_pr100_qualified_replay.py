"""Submitted-cohort coverage and scientific protocol isolation regressions."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import types
import subprocess

import pytest

from tools import pr100_replay as replay
from tools import pr100_replay_worker as worker


def fixture():
    sequences = ["ACDEFGHIKLMNPQRSTVWY" + c for c in "ACDEF"]
    cohort = {"epoch": 25959, "target": "Q9NZQ7", "sequence_pool": sequences}
    proof = {"epoch": 25959, "verified": True, "source": "fixture-validator-consumed",
             "submitted": [{"uid": uid, "sequence_sha256": hashlib.sha256(s.encode()).hexdigest()}
                           for uid, s in zip([1, 7, 16, 17, 20], sequences)]}
    rows = [{"uid": uid, "nanobodies": [{"protein_name": "Q9NZQ7", "sequence": seq,
                                        **{m: float(uid) for m in replay.METRICS}}]}
            for uid, seq in [(16, sequences[2]), *[(i, "RIVAL") for i in range(30, 37)]]]
    protocol = {"settings": dict(replay.SETTINGS), "seed_policy": "independent_process_stage_seed_v1",
                "subnet_config": copy.deepcopy(replay.TARGET_CONTEXT),
                "source_identity": {"fixture": "source"}, "model_identity": {"fixture": "model"},
                "runtime_identity": {"fixture": "runtime"}}
    return cohort, proof, rows, protocol


def manifest():
    cohort, proof, board, protocol = fixture()
    return replay.prepare(cohort, proof, board, 25959, protocol)


def observations(m, passes=10):
    return [replay.observation(m, c["uid"], n, {k: float(n + c["uid"]) for k in replay.METRICS},
                               artifact_sha256=replay.digest(["artifact", n]))
            for n in range(1, passes + 1) for c in m["cohort"]]


def test_five_actual_submissions_survive_single_accepted_leaderboard_row():
    m = manifest()
    assert [c["uid"] for c in m["cohort"]] == [1, 7, 16, 17, 20]
    assert [c["uid"] for c in m["cohort"] if c["accepted_on_board"]] == [16]
    assert len(m["field"]["rows"]) == 7
    assert m["measurement"] == "full_protocol_frozen_same_epoch_rivals"


@pytest.mark.parametrize("mutation", ["missing", "wrong_hash", "wrong_target"])
def test_bad_submission_proof_or_target_rejected(mutation):
    cohort, proof, board, protocol = fixture()
    if mutation == "missing": proof["submitted"].pop()
    elif mutation == "wrong_hash": proof["submitted"][0]["sequence_sha256"] = "a" * 64
    else: cohort["target"] = "P20809"
    with pytest.raises(ValueError): replay.prepare(cohort, proof, board, 25959, protocol)


@pytest.mark.parametrize("key,value", [("fold_sampling_steps", 35), ("fold_diffusion_samples", 1),
                                      ("design_sampling_steps", 50), ("synthetic_design", True)])
def test_fast_protocol_is_not_qualified(key, value):
    cohort, proof, board, protocol = fixture()
    protocol["settings"][key] = value
    with pytest.raises(ValueError, match="full_protocol"):
        replay.prepare(cohort, proof, board, 25959, protocol)


def test_idempotent_individual_pass_ingestion_complete_five_member_mean(tmp_path):
    m = manifest(); db = tmp_path / "observations.sqlite"
    rows = observations(m)
    replay.ingest(db, m, rows[:49]); replay.ingest(db, m, rows[:49])
    partial = replay.summary(db, m)
    assert partial["observations"] == 49
    assert partial["fleet_mean"] is None and not partial["coverage_complete"]
    assert partial["candidates"]["20"]["passes"] == 9
    replay.ingest(db, m, rows[49:]); replay.ingest(db, m, rows)
    complete = replay.summary(db, m)
    assert complete["observations"] == 50 and complete["coverage_complete"]
    assert complete["fleet_mean"] == pytest.approx(sum(v["mean"] for v in complete["candidates"].values()) / 5)
    assert complete["historical_ten_epoch_comparable"] is False
    assert len({r["observation_id"] for r in rows}) == 50
    assert len({r["seed"] for r in rows}) == 10  # one independent batch per pass


def test_conflicting_retry_cannot_overwrite_pass_and_batch_rolls_back(tmp_path):
    m = manifest(); db = tmp_path / "observations.sqlite"
    rows = observations(m, 1); replay.ingest(db, m, rows[:1])
    conflict = replay.observation(m, rows[0]["uid"], 1, {k: 999. for k in replay.METRICS},
                                 artifact_sha256="f" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        replay.ingest(db, m, [rows[1], conflict])
    assert replay.summary(db, m)["observations"] == 1


def test_other_runtime_field_or_legacy_observations_do_not_count(tmp_path):
    m = manifest(); row = observations(m, 1)[0]; db = tmp_path / "observations.sqlite"
    for key in ("protocol_sha256", "field_sha256", "cohort_sha256", "measurement", "seed"):
        changed = dict(row); changed[key] = "incompatible"
        with pytest.raises(ValueError): replay.ingest(db, m, [changed])
    replay.ingest(db, m, [row])
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE observations (observation_id TEXT, score_kind TEXT)")
        c.executemany("INSERT INTO observations VALUES (?,?)", [(str(i), "multipass_sum10_pass") for i in range(99)])
    assert replay.summary(db, m)["observations"] == 1


def test_dense_ranks_match_pinned_validator_ties():
    field = {"rows": [{"metrics": {m: v for m in replay.METRICS}} for v in [1., 1., 2., 3.]]}
    # Six maximized metrics: one better distinct value. Four minimized: one.
    assert replay.rank_sum({m: 2. for m in replay.METRICS}, field) == 20.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, None])
def test_invalid_metrics_do_not_become_observations(bad):
    values = {k: 1. for k in replay.METRICS}; values["design_ptm"] = bad
    with pytest.raises(ValueError): replay.observation(manifest(), 1, 1, values, artifact_sha256="a" * 64)


def test_generated_pipeline_must_use_full_steps_and_registered_checkpoint(tmp_path):
    import yaml
    configs = {"design": {"sampling_steps": 500, "diffusion_samples": 1, "checkpoint": "/model/design"},
               "folding": {"sampling_steps": 200, "diffusion_samples": 5, "checkpoint": "/model/fold"},
               "analysis": {"fixture": True}}
    (tmp_path / "steps.yaml").write_text(yaml.safe_dump({"steps": [
        {"name": k, "config_file": k + ".yaml"} for k in configs]}))
    for k, v in configs.items(): (tmp_path / (k + ".yaml")).write_text(yaml.safe_dump(v))
    assert set(worker.check_generated(tmp_path, {"/model/design", "/model/fold"})) == set(configs)
    with pytest.raises(ValueError, match="unregistered"): worker.check_generated(tmp_path, {"/model/design"})
    configs["folding"]["sampling_steps"] = 35
    (tmp_path / "folding.yaml").write_text(yaml.safe_dump(configs["folding"]))
    with pytest.raises(ValueError, match="fast_settings"): worker.check_generated(tmp_path, {"/model/design", "/model/fold"})


def test_resume_rejects_old_protocol_completed_pass(tmp_path, monkeypatch):
    m = manifest(); monkeypatch.setattr(worker, "verify_runtime", lambda m: {})
    path = tmp_path / "pass-01"; path.mkdir()
    rows = observations(m, 1); rows[0]["protocol_sha256"] = "wrong"
    (path / "result.json").write_text(json.dumps({"artifact_sha256": rows[0]["artifact_sha256"], "observations": rows}))
    with pytest.raises(ValueError, match="incompatible_cached"):
        worker.run(m, tmp_path / "manifest.json", tmp_path, 100, clock=lambda: 1)


def test_deadline_stops_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "verify_runtime", lambda m: {})
    def forbidden(*a, **kw): raise AssertionError("must not launch")
    assert worker.run(manifest(), tmp_path / "manifest.json", tmp_path, 10,
                      clock=lambda: 1, launch=forbidden) == []


def test_child_uses_real_wrapper_contract_and_pinned_asset_resolver(tmp_path, monkeypatch):
    """Integration boundary caught the old scorer's invalid {uid: sequences} shape."""
    m = manifest(); wrapper = types.ModuleType("boltzgen.boltzgen_wrapper")
    cli = types.ModuleType("boltzgen.cli.boltzgen")
    source = {}
    for name in ("wrapper", "cli", "main", "boltzgen_config"):
        p = tmp_path / (name + ".py"); p.write_text("# fixture")
        source[name] = {"path": str(p), "sha256": replay.file_hash(p)}
    for role, relative in (("target_structure", "data/structures/8AOK.cif"),
                           ("target_msa", "data/msa_files/Q9NZQ7.a3m")):
        source[role] = {"path": str(tmp_path / relative)}
    m["protocol"]["source_identity"] = {"files": source}
    m["protocol"]["asset_root"] = str(tmp_path)
    m["protocol"]["model_identity"] = {"artifacts": {"moldir": {"path": "/model/mols.zip"},
        "design-diverse": {"path": "/model/design"}, "folding": {"path": "/model/fold"}}}
    m["protocol_sha256"] = replay.digest(m["protocol"])
    m["job_id"] = replay.digest({k: m[k] for k in ("epoch", "target", "field_sha256", "cohort_sha256", "protocol_sha256")})
    wrapper.__file__ = source["wrapper"]["path"]; cli.__file__ = source["cli"]["path"]
    cli.ARTIFACTS = {"moldir": ("huggingface:fixture:mols.zip", "dataset")}
    wrapper.configure_command = lambda args: None
    captured = {}
    class FakeWrapper:
        def run_nanobody_inference(self, by_uid, context):
            import yaml
            captured.update(cohort=by_uid, context=context)
            assert cli.get_artifact_path(None, "huggingface:fixture:mols.zip", repo_type="dataset") == Path("/model/mols.zip")
            assert Path(cli.main_script).is_file()
            output = Path(wrapper.BOLTZGEN_TMP_FILES_DIR) / "outputs"; output.mkdir(parents=True)
            configs = {"design": {"sampling_steps": 500, "diffusion_samples": 1, "checkpoint": "/model/design"},
                       "folding": {"sampling_steps": 200, "diffusion_samples": 5, "checkpoint": "/model/fold"},
                       "analysis": {"fixture": True}}
            (output / "steps.yaml").write_text(yaml.safe_dump({"steps": [
                {"name": k, "config_file": k + ".yaml"} for k in configs]}))
            for k, v in configs.items():
                (output / (k + ".yaml")).write_text(yaml.safe_dump(v))
                seed = int(hashlib.sha256(f"{replay.pass_seed(m, 1)}:{k}".encode()).hexdigest()[:8], 16)
                (output / (k + ".yaml.seed.json")).write_text(json.dumps({"stage": k, "seed": seed}))
            (output / "fixture.cif").write_text("# retained fixture structure")
            wrapper.configure_command(types.SimpleNamespace(output=output))
            return {uid: {r["sequences"][0]: {"Q9NZQ7": {k: float(uid) for k in replay.METRICS}}}
                    for uid, r in by_uid.items()}
    wrapper.BoltzgenWrapper = FakeWrapper
    package = types.ModuleType("boltzgen"); package.boltzgen_wrapper = wrapper
    cli_package = types.ModuleType("boltzgen.cli"); cli_package.boltzgen = cli
    for name, module in [("boltzgen", package), ("boltzgen.boltzgen_wrapper", wrapper),
                         ("boltzgen.cli", cli_package), ("boltzgen.cli.boltzgen", cli)]:
        monkeypatch.setitem(sys.modules, name, module)
    result = worker.child(m, 1, tmp_path)
    assert set(captured["cohort"]) == {1, 7, 16, 17, 20}
    assert captured["context"] == replay.TARGET_CONTEXT
    assert len(result["observations"]) == 5
    assert all(row["observation_count"] == 1 for row in result["observations"])


def test_source_inventory_prunes_sensitive_paths_before_reading(tmp_path, monkeypatch):
    for relative in [".env", ".git/config", ".aws/credentials", "session.json", "secrets/client.json",
                     "tokenizer.py", "task/predict.py"]:
        p = tmp_path / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("fixture")
    read = []
    original = worker.file_hash
    def guarded(path):
        relative = str(path.relative_to(tmp_path)); read.append(relative)
        assert relative in {"tokenizer.py", "task/predict.py"}
        return original(path)
    monkeypatch.setattr(worker, "file_hash", guarded)
    assert len(worker.artifact_hash({"kind": "tree", "path": str(tmp_path)})) == 64
    assert set(read) == {"tokenizer.py", "task/predict.py"}


def test_deadline_terminates_own_real_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "verify_runtime", lambda m: {})
    child = []
    def launch(*args, **kwargs):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        child.append(process); return process
    moments = iter([0., 0., 0., 20.])
    with pytest.raises(subprocess.TimeoutExpired):
        worker.run(manifest(), tmp_path / "manifest.json", tmp_path, 12.,
                   launch=launch, clock=lambda: next(moments))
    assert child and child[0].poll() is not None
    assert not list(tmp_path.glob("pass-*/result.json"))


def test_pi_lease_job_is_bounded_and_uses_private_full_worker():
    from tools.pr100_replay_prepare import lease_job
    kwargs = dict(python="/runtime/bin/python", release="/release", manifest="/job/manifest.json",
                  output="/job/results", gpu=0, scorer="/known/scorer.py", pythonpath="/release:/source")
    job = lease_job(**kwargs, deadline=1100, now=100)
    assert job["ttl_seconds"] == 1000
    assert job["command"][1] == "/release/tools/pr100_replay_worker.py"
    assert job["command"][-1] == "1095"
    assert job["env"]["NANOBODY_SCORER_DAEMON"] == "0"
    for deadline in (99, 110, 1301):
        with pytest.raises(ValueError): lease_job(**kwargs, deadline=deadline, now=100)
