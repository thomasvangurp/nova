"""Local tests for the fleet publisher. No chain, no wallet, no network.

Every subprocess the publisher shells out to is replaced by a fake that writes
the file the real program would have written, so the whole publication path --
discovery, ordering, distinctness, the single artifact, the ledger -- is
exercised without a host.
"""
import argparse
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from compound_miner.nanobody_provider import (
    fleet_submission,
    validator_boltzgen_metrics,
)
from nanobody_opt import fleet, publication_gates
from tools import publish_current_nanobody
from tools.publish_current_nanobody import (
    COMMIT_FLEET_FAILED,
    PUBLISH_FLEET_REJECTED,
    COMMIT_FLEET_INCOMPLETE,
    RANK_QUALIFICATION_ARGUMENTS,
    SELECTION_ORDER,
    _archive_and_audit_previous,
    _chain_view,
    _checkpoint_sequence,
    _default_exact_inputs,
    _exact_bank_fingerprints,
    _fleet_is_current,
    _has_new_exact_data,
    _is_current_qualified,
    _parser,
    _commit_fleet,
    _publish_once,
    _publisher_state_matches,
    _retire_completed_epoch,
    _strict_state_matches,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PUBLISHER = REPOSITORY / "tools" / "publish_current_nanobody.py"
TARGET = "P20809"
EPOCH = 25062
EPOCH_LENGTH = 361
RESERVE_BLOCKS = 55
OUR_COLDKEY = "5EkCwcNTfxeNqmQ4ZQqaMGRVJmJmJmJmJmJmJmJmJmJmJmJm"
OTHER_COLDKEY = "5FOtherColdkeyOtherColdkeyOtherColdkeyOtherCold"
HOTKEYS = [f"5H{letter * 46}" for letter in "abcdef"]
METRICS = {name: 1.0 for name in validator_boltzgen_metrics()}
REQUIREMENTS = {
    "min_sequence_length": 90,
    "max_sequence_length": 150,
    "num_sequences": 1,
    "num_top_sequences": 50,
    "max_similarity_score": 0.7,
}


def _sequence(index: int) -> str:
    """A distinct, canonical, correctly sized nanobody sequence."""
    body = ("ACDEFGHIKLMNPQRSTVWY" * 6)[:116]
    return "ACDEFGHIKLMNPQRSTVWY"[index] + body


SEQUENCE = _sequence(0)


def _checkpoint(epoch=EPOCH, sequence=SEQUENCE, stage="qualified"):
    return {
        "schema_version": 3,
        "status": stage,
        "publication_stage": stage,
        "target": TARGET,
        "epoch": epoch,
        "sequences": [sequence],
        "validation": {
            "layer_1": {"passed": True},
            "layer_2": {"passed": True},
            "boltzgen": {
                "exact": True,
                "runtime_matched": True,
                "metrics": METRICS,
            },
        },
        "exact": {
            "observation_count": 8,
            "pose_stability": {
                "structure_observation_count": 8,
                "contact_jaccard_median": 0.5,
                "cdr3_rmsd_max_angstrom": 4.0,
                "epitope_occupancy_min": 0.5,
            },
        },
        "selection": {
            "score": 120.0,
            "competitive_rate": 0.75,
            "observation_competitive_rate": 0.75,
            "score_median": 110.0,
            "score_p90": 130.0,
            "p90_category_rank_sums": {
                "confidence": 70.0,
                "physical_interaction": 40.0,
                "developability": 15.0,
            },
        },
    }


# --------------------------------------------------------------------------
# Gate drift
# --------------------------------------------------------------------------


def test_publication_gates_are_not_restated_in_this_file():
    assert RANK_QUALIFICATION_ARGUMENTS == tuple(publication_gates.gate_arguments())
    arguments = list(RANK_QUALIFICATION_ARGUMENTS)
    # The relaxed values, which the hardcoded copy used to contradict for hours.
    assert arguments[arguments.index("--minimum-observations") + 1] == "1"
    assert arguments[arguments.index("--maximum-p90-rank-sum") + 1] == "400"
    # ... and the integrity gates, which are never relaxed for score.
    assert arguments[arguments.index("--minimum-median-contact-jaccard") + 1] == "0.35"


def test_the_publisher_source_holds_no_threshold_of_its_own():
    source = PUBLISHER.read_text()
    for stale in ('"75"', '"90"', '"0.75"', '"8"'):
        assert stale not in source, f"a hardcoded gate value {stale} is back"


def test_the_shortlist_is_ordered_by_win_probability():
    assert SELECTION_ORDER == "win-probability"
    assert SELECTION_ORDER == fleet.REQUIRED_SELECTION_ORDER


def test_help_works_with_no_chain_no_wallet_and_no_keychain():
    # The deploy gate runs this before it stops the producer.
    completed = subprocess.run(
        [sys.executable, str(PUBLISHER), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPOSITORY),
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--coldkey" in completed.stdout


# --------------------------------------------------------------------------
# The chain view
# --------------------------------------------------------------------------


def _chain_state(path, *, epoch=EPOCH, age=0.0, neurons=None, block=None):
    block = EPOCH_LENGTH * epoch + 17 if block is None else block
    payload = {
        "schema_version": 2,
        "netuid": 68,
        "block": block,
        "epoch_length": EPOCH_LENGTH,
        "epoch": epoch,
        "updated_at_unix": time.time() - age,
    }
    if neurons is not None:
        payload["neurons"] = neurons
    path.write_text(json.dumps(payload))
    return path


def _neurons(count=3, ours=(0, 1, 2)):
    return [
        {
            "uid": index,
            "hotkey": HOTKEYS[index],
            "coldkey": OUR_COLDKEY if index in ours else OTHER_COLDKEY,
        }
        for index in range(count)
    ]


def test_chain_view_carries_identity_with_the_block(tmp_path):
    state = _chain_state(tmp_path / "chain.json", neurons=_neurons())

    view = _chain_view(state, 68, 90)

    assert view.epoch == EPOCH
    assert view.epoch_length == EPOCH_LENGTH
    assert [row["uid"] for row in view.neurons] == [0, 1, 2]


def test_chain_view_rejects_stale_state_without_opening_a_websocket(tmp_path):
    state = _chain_state(tmp_path / "chain.json", age=1000, neurons=_neurons())

    with pytest.raises(RuntimeError, match="stale"):
        _chain_view(state, 68, 90)


def test_chain_view_refuses_a_state_with_no_neuron_rows(tmp_path):
    # Without neuron rows the only way to name a uid is to remember one, which
    # is the failure mode that cost 21 days.
    state = _chain_state(tmp_path / "chain.json")

    with pytest.raises(RuntimeError, match="no neuron rows"):
        _chain_view(state, 68, 90)


# --------------------------------------------------------------------------
# The exact-bank caches
# --------------------------------------------------------------------------


def _write_exact(path, *, observation, sequence=SEQUENCE, metric_value=1.0):
    path.write_text(json.dumps({
        "target": TARGET,
        "observation_id": observation,
        "per_nanobody_components": {
            "0": {
                sequence: {TARGET: {name: metric_value for name in METRICS}}
            }
        },
        "structures": {sequence: {TARGET: {"path": f"/{observation}.pdb"}}},
    }))


def test_one_shot_exact_data_invalidates_provisional_but_not_strict_cache(tmp_path):
    exact = []
    for index in range(8):
        path = tmp_path / f"repeat_{index}.json"
        _write_exact(path, observation=f"repeat-{index}")
        exact.append(path)
    before_exact, before_strict = _exact_bank_fingerprints(
        tuple(exact), TARGET, minimum=8
    )
    one_shot = tmp_path / "one_shot.json"
    _write_exact(one_shot, observation="one-shot", sequence=_sequence(1))

    after_exact, after_strict = _exact_bank_fingerprints(
        (*exact, one_shot), TARGET, minimum=8
    )

    assert after_exact != before_exact
    assert after_strict == before_strict


def test_the_strict_eligibility_floor_comes_from_publication_gates(tmp_path):
    # One observation is the floor today, so one new observation must invalidate
    # the strict cache: the cache may never be more patient than the gate.
    first = tmp_path / "first.json"
    _write_exact(first, observation="first")
    _, before = _exact_bank_fingerprints((first,), TARGET)
    second = tmp_path / "second.json"
    _write_exact(second, observation="second")

    _, after = _exact_bank_fingerprints((first, second), TARGET)

    assert int(publication_gates.gate_defaults()["--minimum-observations"]) == 1
    assert after != before


def test_default_exact_inputs_include_aggregate_but_not_wave_shards(tmp_path):
    aggregate = tmp_path / "live/epoch25024/exact_round1/exact_all.json"
    shard = tmp_path / "live/epoch25024/exact_round1/wave_000_exact.json"
    confirmation = tmp_path / "live/epoch25024/finalists_confirm.json"
    for path in (aggregate, shard, confirmation):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    discovered = _default_exact_inputs(tmp_path)

    assert aggregate in discovered
    assert confirmation in discovered
    assert shard not in discovered
    assert len(discovered) == len(set(discovered))


def test_new_atomic_exact_result_triggers_same_epoch_refresh(tmp_path):
    artifact = tmp_path / "checkpoint.json"
    exact = tmp_path / "exact_all.json"
    artifact.write_text("{}")
    exact.write_text("{}")
    artifact.touch()

    assert not _has_new_exact_data(artifact, (exact,))

    exact.touch()
    exact_stat = exact.stat()
    artifact_stat = artifact.stat()
    if exact_stat.st_mtime_ns <= artifact_stat.st_mtime_ns:
        import os

        os.utime(exact, ns=(artifact_stat.st_atime_ns, artifact_stat.st_mtime_ns + 1))
    assert _has_new_exact_data(artifact, (exact,))


def test_processed_state_is_bound_to_the_uid_set_as_well(tmp_path):
    common = {
        "epoch": EPOCH,
        "target": TARGET,
        "exact_fingerprint": "exact",
        "ledger_fingerprint": "ledger",
        "fleet_fingerprint": "[[1,\"5H\"]]",
    }
    state = {"schema_version": 2, **common}

    assert _publisher_state_matches(state, **common)
    assert not _publisher_state_matches(state, **{**common, "exact_fingerprint": "new"})
    assert not _publisher_state_matches(state, **{**common, "epoch": EPOCH + 1})
    # A registration or deregistration changes how many distinct sequences are
    # needed, so it must never be served from cache.
    assert not _publisher_state_matches(
        state, **{**common, "fleet_fingerprint": "[[1,\"5H\"],[2,\"5I\"]]"}
    )


def test_strict_cache_survives_an_unrelated_one_shot_but_not_a_new_uid():
    state = {
        "schema_version": 2,
        "epoch": EPOCH,
        "target": TARGET,
        "exact_fingerprint": "old-exact",
        "strict_fingerprint": "strict",
        "ledger_fingerprint": "ledger",
        "fleet_fingerprint": "fleet",
    }
    common = {
        "epoch": EPOCH,
        "target": TARGET,
        "strict_fingerprint": "strict",
        "ledger_fingerprint": "ledger",
        "fleet_fingerprint": "fleet",
    }

    assert _strict_state_matches(state, **common)
    assert not _strict_state_matches(
        state, **{**common, "strict_fingerprint": "new-strict"}
    )
    assert not _strict_state_matches(state, **{**common, "fleet_fingerprint": "other"})


# --------------------------------------------------------------------------
# Checkpoint and artifact validation
# --------------------------------------------------------------------------


def test_current_checkpoint_requires_full_consumer_validation():
    assert _is_current_qualified(
        _checkpoint(), target=TARGET, epoch=EPOCH, requirements=REQUIREMENTS
    )
    partial = _checkpoint()
    partial["validation"]["layer_2"] = {"passed": False}
    assert not _is_current_qualified(
        partial, target=TARGET, epoch=EPOCH, requirements=REQUIREMENTS
    )
    assert not _is_current_qualified(
        _checkpoint(), target=TARGET, epoch=EPOCH + 1, requirements=REQUIREMENTS
    )


def _artifact(sequences, hotkeys, epoch=EPOCH, stage="qualified"):
    crew = [
        fleet.FleetMember(uid=index, hotkey=hotkey)
        for index, hotkey in enumerate(hotkeys)
    ]
    blocks = list(range(100, 100 + len(sequences)))
    assignments = fleet.assign_sequences(crew, sequences, blocks)
    return fleet.fleet_artifact(
        target=TARGET,
        epoch=epoch,
        stage=stage,
        assignments=assignments,
        checkpoints={
            sequence: _checkpoint(epoch=epoch, sequence=sequence, stage=stage)
            for sequence in sequences
        },
    )


def test_a_live_fleet_must_cover_every_uid_with_a_distinct_sequence():
    sequences = [_sequence(index) for index in range(3)]
    hotkeys = HOTKEYS[:3]
    artifact = _artifact(sequences, hotkeys)

    assert _fleet_is_current(
        artifact,
        target=TARGET,
        epoch=EPOCH,
        requirements=REQUIREMENTS,
        expected_hotkeys=hotkeys,
    )
    # A newly registered hotkey is not covered by yesterday's artifact.
    assert not _fleet_is_current(
        artifact,
        target=TARGET,
        epoch=EPOCH,
        requirements=REQUIREMENTS,
        expected_hotkeys=[*hotkeys, HOTKEYS[3]],
    )
    duplicated = json.loads(json.dumps(artifact))
    duplicated["submissions"][1]["sequence"] = sequences[0]
    assert not _fleet_is_current(
        duplicated,
        target=TARGET,
        epoch=EPOCH,
        requirements=REQUIREMENTS,
        expected_hotkeys=hotkeys,
    )


def test_a_miner_selects_its_own_submission_by_hotkey():
    sequences = [_sequence(index) for index in range(3)]
    artifact = _artifact(sequences, HOTKEYS[:3])

    checkpoint = fleet_submission(artifact, HOTKEYS[1])

    assert checkpoint["sequences"] == [sequences[1]]
    with pytest.raises(ValueError, match="0 submissions for this hotkey"):
        fleet_submission(artifact, HOTKEYS[4])


def test_every_sequence_of_a_completed_epoch_is_retired(tmp_path):
    sequences = [_sequence(index) for index in range(3)]
    ledger = tmp_path / "published.txt"
    artifact = _artifact(sequences, HOTKEYS[:3], epoch=EPOCH - 1)

    assert _retire_completed_epoch(artifact, ledger, EPOCH) == sequences
    # Idempotent, and the live epoch is never retired early.
    assert _retire_completed_epoch(artifact, ledger, EPOCH) == []
    assert _retire_completed_epoch(_artifact(sequences, HOTKEYS[:3]), ledger, EPOCH) == []
    assert fleet.read_ledger(ledger) == sequences


def test_completed_fleet_is_archived_and_audited_per_hotkey(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        "tools.publish_current_nanobody._run", lambda command: commands.append(command)
    )
    sequences = [_sequence(index) for index in range(2)]
    artifact = _artifact(sequences, HOTKEYS[:2], epoch=EPOCH - 1)

    _archive_and_audit_previous(
        artifact,
        current_epoch=EPOCH,
        history_dir=tmp_path / "history",
        audit_dir=tmp_path / "audits",
        bgen_python=tmp_path / "bgen-python",
        nanobody_opt_root=tmp_path / "nanobody_opt",
    )

    archived = tmp_path / "history" / f"epoch{EPOCH - 1}.json"
    assert json.loads(archived.read_text())["epoch"] == EPOCH - 1
    audited = [command[command.index("--hotkey") + 1] for command in commands]
    assert audited == HOTKEYS[:2]


def test_a_malformed_previous_checkpoint_is_not_remembered():
    assert _checkpoint_sequence({"sequences": ["lowercase"]}) is None
    assert _checkpoint_sequence({"sequences": []}) is None


# --------------------------------------------------------------------------
# The whole publication, with every subprocess faked
# --------------------------------------------------------------------------


class _Host:
    """A fake nanobody host: no GPUs, no chain, no network, no wallet keys."""

    def __init__(self, root: Path, monkeypatch, *, uids=(0, 1, 2), candidates=5):
        self.root = root
        self.commands: list[list[str]] = []
        self.candidate_sequences = [_sequence(index) for index in range(candidates)]
        # Sequences publish_checkpoint.py refuses, so a test can reproduce
        # the live "winner lacks a current Layer-2 pass" exit for one uid.
        self.unpublishable: set[str] = set()
        # Ranked candidates this epoch's Layer-2 run does not pass.
        self.layer2_failures: set[str] = set()
        # Candidates the validator's own duplicate gates reject: >=0.95
        # whole-sequence identity, or >=0.85 CDR3, to a top sequence.
        self.non_novel: set[str] = set()
        # The publisher now waits for each uid's scheduled block inside the
        # end-of-epoch reserve, so the fake clock advances the chain instead of
        # sleeping: the waiting logic runs for real, the test does not.
        self.blocks_waited = 0

        def advance(seconds):
            self.blocks_waited += 1
            state = json.loads(self.chain_state.read_text())
            state["block"] += max(1, int(seconds // 12) or 1)
            self.chain_state.write_text(json.dumps(state))

        monkeypatch.setattr(publish_current_nanobody, "_sleep", advance)
        runtime = root / "runtime"
        (runtime / ".pixi/envs/bgen/bin").mkdir(parents=True)
        (runtime / ".pixi/envs/bgen/bin/python").write_text("#!/bin/sh\n")
        (runtime / "pixi.toml").write_text("[project]\n")
        self.runtime = runtime
        self.wallet_root = root / "wallets"
        hotkeys = self.wallet_root / "mining_ops" / "hotkeys"
        hotkeys.mkdir(parents=True)
        for index, hotkey in enumerate(HOTKEYS):
            (hotkeys / f"nova68-{index}").write_text(
                json.dumps({"ss58Address": hotkey, "secretPhrase": "NEVER READ"})
            )
        self.chain_state = _chain_state(
            root / "chain.json", neurons=_neurons(count=4, ours=uids)
        )
        self.exact = root / "exact_all.json"
        _write_exact(self.exact, observation="wave-1")
        self.layer2_history = root / "historical_layer2.json"
        self.layer2_history.write_text(json.dumps({"candidates": []}))
        self.artifact = root / "artifacts" / "nanobody_result.json"
        self.ledger = root / "artifacts" / "nanobody_published_sequences.txt"
        monkeypatch.setattr(
            "tools.publish_current_nanobody._run", self.run
        )
        monkeypatch.setattr(
            "tools.publish_current_nanobody._atomic_download", self.download
        )
        monkeypatch.setattr(
            "tools.publish_current_nanobody._gpu_count", lambda: 0
        )

    def args(self, extra_exact=(), strict_qualification=True, **overrides):
        arguments = [
            "--expected-gpus", "0",
            "--netuid", "68",
            "--runtime-root", str(self.runtime),
            "--score-bank-root", str(self.root),
            "--nanobody-opt-root", str(self.root / "nanobody_opt"),
            "--pixi", str(self.root / "pixi"),
            "--chain-state", str(self.chain_state),
            "--artifact", str(self.artifact),
            "--history-dir", str(self.root / "history"),
            "--audit-dir", str(self.root / "audits"),
            "--coldkey", OUR_COLDKEY,
            "--wallet-name", "mining_ops",
            "--wallet-path", str(self.wallet_root),
            "--ledger", str(self.ledger),
            "--lock", str(self.root / "publisher.lock"),
            "--state", str(self.root / "publisher_state.json"),
            "--exact", str(self.exact),
            "--historical-layer2", str(self.layer2_history),
            "--layer2-cache-dir", str(self.root / "layer2_cache"),
            "--commit-pause-seconds", "0",
        ]
        # The strict upgrade is off in production; these tests cover it, so the
        # harness turns it on unless a test is specifically about the default.
        if strict_qualification:
            arguments.append("--strict-qualification")
        for path in extra_exact:
            arguments.extend(["--exact", str(path)])
        for flag, value in overrides.items():
            arguments.extend([f"--{flag.replace('_', '-')}", str(value)])
        return _parser().parse_args(arguments)

    def download(self, url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if url.endswith("config.yaml"):
            destination.write_text(yaml.safe_dump({
                "protein_selection": {"nanobody": {
                    "target": TARGET,
                    "structure": "7XDS",
                    "structure_chain": "A",
                    "structure_res_index": "1",
                    "structure_binding_site": "10",
                }},
                "nanobody_requirements": REQUIREMENTS,
            }))
        else:
            destination.write_text("sequence\nAAAA\n")

    def run(self, command):
        self.commands.append(list(command))
        joined = " ".join(command)

        def value(flag):
            return command[command.index(flag) + 1]

        if "rank_exact.py" in joined:
            output = Path(value("--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "target": TARGET,
                "selection_order": SELECTION_ORDER,
                "candidates": [
                    {
                        "sequence": sequence,
                        "win_probability": 0.9 - 0.1 * index,
                        "score_median": 100.0 + index,
                    }
                    for index, sequence in enumerate(self.candidate_sequences)
                ],
            }))
        elif "filter-layer2" in joined:
            output = Path(value("--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"candidates": [
                {
                    "sequence": sequence,
                    "layer2_passed": sequence not in self.layer2_failures,
                }
                for sequence in self.candidate_sequences
            ]}))
        elif "check-validator-novelty" in joined:
            output = Path(value("--output"))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"sequences": [
                sequence for sequence in self.candidate_sequences
                if sequence not in self.non_novel
            ]}))
        elif "publish_checkpoint.py" in joined:
            if value("--sequence") in self.unpublishable:
                raise subprocess.CalledProcessError(1, command)
            output = Path(value("--output"))
            stage = "provisional" if "--provisional" in command else "qualified"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(_checkpoint(
                epoch=int(value("--epoch")),
                sequence=value("--sequence"),
                stage=stage,
            )))
        elif "audit_validator_score.py" in joined:
            Path(value("--output")).parent.mkdir(parents=True, exist_ok=True)
            Path(value("--output")).write_text("{}")

    def ranked(self, strict):
        return [
            command for command in self.commands
            if "rank_exact.py" in " ".join(command)
            and (("--pose-config" in command) == strict)
        ]


def test_every_held_uid_publishes_one_distinct_sequence(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))

    _publish_once(host.args())

    artifact = json.loads(host.artifact.read_text())
    assert artifact["artifact_kind"] == "nanobody-fleet"
    assert artifact["publication_stage"] == "qualified"
    assert artifact["submission_count"] == 3
    sequences = [entry["sequence"] for entry in artifact["submissions"]]
    assert len(set(sequences)) == 3, "the validator dedupes by sequence"
    assert sequences == host.candidate_sequences[:3], "best three by win probability"
    assert [entry["uid"] for entry in artifact["submissions"]] == [0, 1, 2]
    assert [entry["hotkey"] for entry in artifact["submissions"]] == HOTKEYS[:3]
    assert not artifact["unassigned_uids"]
    for entry in artifact["submissions"]:
        assert entry["metrics"] == METRICS


def test_all_k_submissions_are_recorded_in_the_ledger(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))

    _publish_once(host.args())

    assert fleet.read_ledger(host.ledger) == host.candidate_sequences[:3]


def test_the_strict_pass_uses_the_shared_gates_and_win_probability(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))

    _publish_once(host.args())

    strict = host.ranked(strict=True)
    assert len(strict) == 1
    command = strict[0]
    for flag, expected in zip(
        RANK_QUALIFICATION_ARGUMENTS[::2], RANK_QUALIFICATION_ARGUMENTS[1::2]
    ):
        assert command[command.index(flag) + 1] == expected
    assert command[command.index("--selection-order") + 1] == SELECTION_ORDER
    assert command[command.index("--limit") + 1] == "30"
    # The provisional net applies the evidence floor only, never the quality gates.
    provisional = host.ranked(strict=False)[0]
    assert "--maximum-p90-rank-sum" not in provisional
    assert provisional[provisional.index("--selection-order") + 1] == SELECTION_ORDER
    # The winning threshold is what win probability is measured against, so both
    # passes must order against the same one rather than against a CLI default.
    threshold = publication_gates.gate_defaults()["--competitive-score-threshold"]
    for ranked in (command, provisional):
        assert ranked[ranked.index("--competitive-score-threshold") + 1] == threshold


def test_every_commitment_lands_in_the_end_of_epoch_reserve(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))

    _publish_once(host.args())

    artifact = json.loads(host.artifact.read_text())
    blocks = [entry["commit_block"] for entry in artifact["submissions"]]
    epoch_end = EPOCH_LENGTH * (EPOCH + 1)
    assert blocks == list(range(epoch_end - RESERVE_BLOCKS, epoch_end - RESERVE_BLOCKS + 3))
    assert len(set(blocks)) == 3, "max_regs_per_block is 1"


def test_one_commit_extrinsic_is_sent_per_uid(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))

    _publish_once(host.args(commit_command="/root/commit.sh {hotkey} {sequence} {block}"))

    commits = [
        command for command in host.commands if command[0].endswith("commit.sh")
    ]
    # Two passes over the fleet: the provisional commit that guarantees every
    # uid lands inside the epoch, then the qualified upgrade. The chain's own
    # rate limit decides whether the second one takes.
    assert [command[1] for command in commits] == HOTKEYS[:3] * 2
    assert len({command[2] for command in commits[:3]}) == 3
    assert len({command[2] for command in commits[3:]}) == 3


def test_a_registration_is_picked_up_without_touching_configuration(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1))
    _publish_once(host.args())
    assert json.loads(host.artifact.read_text())["submission_count"] == 2

    # A third hotkey registers. Nothing is reconfigured; the chain is re-read.
    _chain_state(host.chain_state, neurons=_neurons(count=4, ours=(0, 1, 2)))
    _publish_once(host.args())

    artifact = json.loads(host.artifact.read_text())
    assert artifact["submission_count"] == 3
    assert [entry["hotkey"] for entry in artifact["submissions"]] == HOTKEYS[:3]
    assert len({entry["sequence"] for entry in artifact["submissions"]}) == 3


def test_a_deregistered_uid_is_dropped_rather_than_published_into(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    _publish_once(host.args())

    # uid 2 is deregistered and its slot is recycled to another coldkey.
    _chain_state(host.chain_state, neurons=_neurons(count=4, ours=(0, 1)))
    _publish_once(host.args())

    artifact = json.loads(host.artifact.read_text())
    assert [entry["hotkey"] for entry in artifact["submissions"]] == HOTKEYS[:2]


def test_an_empty_fleet_stops_the_publication_loudly(tmp_path, monkeypatch, capsys):
    host = _Host(tmp_path, monkeypatch, uids=())

    with pytest.raises(fleet.EmptyFleetError) as error:
        _publish_once(host.args())

    assert "NANOBODY_FLEET_EMPTY" in str(error.value)
    assert not host.artifact.exists()
    assert not host.commands


def test_the_watch_loop_reports_an_empty_fleet_on_every_pass(tmp_path, monkeypatch, capsys):
    host = _Host(tmp_path, monkeypatch, uids=())
    import tools.publish_current_nanobody as publisher

    passes = {"count": 0}

    def _stop(_seconds):
        passes["count"] += 1
        if passes["count"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(publisher.time, "sleep", _stop)
    monkeypatch.setattr(
        publisher, "_parser", lambda: _FixedParser(host.args(watch_interval=30))
    )

    with pytest.raises(KeyboardInterrupt):
        publisher.main()

    captured = capsys.readouterr()
    assert captured.err.count("NANOBODY_FLEET_EMPTY") == 2


def test_a_registered_uid_with_no_local_key_fails_loudly(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    for keyfile in (host.wallet_root / "mining_ops" / "hotkeys").iterdir():
        keyfile.unlink()

    with pytest.raises(fleet.MissingHotkeyError, match="KEYFILE_MISSING"):
        _publish_once(host.args())


def test_more_uids_than_qualified_sequences_reports_the_shortfall(
    tmp_path, monkeypatch, capsys
):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2), candidates=2)

    _publish_once(host.args())

    artifact = json.loads(host.artifact.read_text())
    assert artifact["submission_count"] == 2
    assert [entry["uid"] for entry in artifact["unassigned_uids"]] == [2]
    assert "NANOBODY_FLEET_SHORTFALL" in capsys.readouterr().err


def test_a_qualified_fleet_is_not_republished_when_nothing_changed(tmp_path, monkeypatch):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    _publish_once(host.args())
    first = len(host.commands)

    _publish_once(host.args())

    assert len(host.commands) == first, "an unchanged epoch must be a no-op"


def test_a_republished_epoch_keeps_the_sequences_it_already_committed(
    tmp_path, monkeypatch
):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    _publish_once(host.args())
    published = [
        entry["sequence"]
        for entry in json.loads(host.artifact.read_text())["submissions"]
    ]

    # New exact evidence arrives mid-epoch: the ledger already holds this epoch's
    # own sequences, and they must not be excluded from their own re-ranking.
    second = host.root / "exact_all_2.json"
    _write_exact(second, observation="wave-2")
    before = len(host.commands)
    _publish_once(host.args(extra_exact=[second]))
    assert len(host.commands) > before, "new exact evidence must trigger a refresh"

    assert [
        entry["sequence"]
        for entry in json.loads(host.artifact.read_text())["submissions"]
    ] == published


class _FixedParser:
    """Return a prepared namespace, so main() can run without a command line."""

    def __init__(self, namespace):
        self.namespace = namespace

    def parse_args(self):
        return self.namespace


def test_a_missing_legacy_default_does_not_defer_publication(tmp_path):
    """DEFAULT_EXACT_INPUTS are paths from epoch 25021 on a destroyed machine.

    Requiring them meant a rebuilt box deferred publication for ever while
    sitting on a score bank the glob had already discovered -- observed live:
    five exact_all.json files present, four uids held, and every epoch deferred
    on /root/local_neighborhood_round1_exact.json.
    """
    import tools.publish_current_nanobody as pub

    root = tmp_path
    found = root / "live" / "epoch25085" / "race" / "screen"
    found.mkdir(parents=True)
    (found / "exact_all.json").write_text("[]")

    discovered = pub._default_exact_inputs(root)
    # The legacy names are offered...
    assert any("local_neighborhood_round1_exact.json" in str(p) for p in discovered)
    # ...and the real file is discovered alongside them.
    assert any(p.name == "exact_all.json" for p in discovered)
    # Only the real one exists, which must be enough to proceed.
    assert [p for p in discovered if p.is_file()] == [found / "exact_all.json"]


def test_publication_still_refuses_when_there_is_no_evidence_at_all(tmp_path):
    """Relaxing the legacy requirement must not let us publish on nothing."""
    import tools.publish_current_nanobody as pub

    discovered = pub._default_exact_inputs(tmp_path)
    assert [p for p in discovered if p.is_file()] == []
    source = (Path(__file__).resolve().parents[1]
              / "tools" / "publish_current_nanobody.py").read_text()
    assert "no exact score-bank inputs found under" in source



COMMIT_EPOCH = 25_087
RESERVE_START = EPOCH_LENGTH * COMMIT_EPOCH + EPOCH_LENGTH - 55


def _commit_args(tmp_path, block):
    """Minimal namespace for a direct _commit_fleet call: it reads the chain
    view to know when each uid's reserve block has arrived."""
    state = _chain_state(
        tmp_path / "chain_state.json", epoch=COMMIT_EPOCH, block=block,
        neurons=_neurons(),
    )
    return argparse.Namespace(
        chain_state=state, netuid=68, maximum_chain_state_age=600.0
    )


def test_one_uid_s_commit_failure_does_not_cost_the_rest_of_the_fleet(
    monkeypatch, capsys, tmp_path
):
    """A chain error on the first uid must not skip the other three.

    ``_run`` raises on a non-zero exit, so an unguarded loop would abandon
    every later uid -- the exact way a four-uid fleet lands one commitment.
    """
    assignments = [
        fleet.Assignment(uid=uid, hotkey=f"5Hotkey{uid}", sequence="Q" * 120,
                         commit_block=RESERVE_START + index)
        for index, uid in enumerate((1, 40, 46, 51))
    ]
    attempted: list[str] = []

    def fake_run(command):
        attempted.append(command[command.index("--uid") + 1])
        if command[command.index("--uid") + 1] == "1":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("tools.publish_current_nanobody._run", fake_run)
    _commit_fleet(
        assignments,
        args=_commit_args(tmp_path, RESERVE_START + 10),
        template="commit --uid {uid} --hotkey {hotkey} --sequence {sequence}",
        artifact=tmp_path / "artifact.json",
        epoch=COMMIT_EPOCH,
        pause_seconds=0,
    )

    assert attempted == ["1", "40", "46", "51"]
    captured = capsys.readouterr().err
    assert f"{COMMIT_FLEET_FAILED} uid=1" in captured
    assert f"{COMMIT_FLEET_INCOMPLETE} 1 of 4" in captured


def test_a_fully_successful_fleet_commit_reports_no_failure(
    monkeypatch, capsys, tmp_path
):
    assignments = [
        fleet.Assignment(uid=uid, hotkey=f"5Hotkey{uid}", sequence="Q" * 120,
                         commit_block=RESERVE_START + index)
        for index, uid in enumerate((1, 40))
    ]
    monkeypatch.setattr(
        "tools.publish_current_nanobody._run", lambda command: None
    )
    _commit_fleet(
        assignments,
        args=_commit_args(tmp_path, RESERVE_START + 10),
        template="commit --uid {uid}",
        artifact=tmp_path / "artifact.json",
        epoch=COMMIT_EPOCH,
        pause_seconds=0,
    )
    assert COMMIT_FLEET_INCOMPLETE not in capsys.readouterr().err


def test_the_provisional_fleet_commits_before_the_strict_rank_runs():
    """The strict rank has run over twenty minutes; the epoch is 72.

    Committing only after it means a slow rank silently costs every uid its
    commitment for the epoch, so the provisional fleet must reach the chain
    first and let the qualified upgrade re-commit if the rate limit allows.
    """
    source = inspect.getsource(publish_current_nanobody._publish_once)
    provisional = source.index('_report(artifact, "provisional")')
    strict_rank = source.index("strict=True")
    commit = source.index("_commit_fleet(", provisional)
    assert provisional < commit < strict_rank


def test_strict_qualification_is_off_unless_asked_for():
    assert _parser().parse_args([]).strict_qualification is False


def test_the_default_run_commits_the_provisional_fleet_and_skips_the_strict_rank(
    tmp_path, monkeypatch, capsys
):
    """Six live epochs produced no qualified upgrade for ~20 minutes each.

    With the upgrade cut, an epoch must still reach the chain once per uid --
    and must not shell out to the strict rank at all.
    """
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    _publish_once(
        host.args(
            strict_qualification=False,
            commit_command="/root/commit.sh {hotkey} {sequence} {block}",
        )
    )

    commits = [c for c in host.commands if c[0].endswith("commit.sh")]
    assert [c[1] for c in commits] == HOTKEYS[:3]
    assert len({c[2] for c in commits}) == 3
    # --pose-config is passed by the strict pass alone; the provisional pass
    # shares every other rank flag, so it is the only honest marker here.
    assert not [c for c in host.commands if "--pose-config" in c]
    assert "strict qualification is disabled" in capsys.readouterr().out


def test_one_unpublishable_uid_does_not_cost_the_rest_of_the_fleet(
    tmp_path, monkeypatch, capsys
):
    """publish_checkpoint.py exiting non-zero used to abort the whole fleet.

    Epochs 25089 and 25091 published two uids of six that way: the failure came
    from inside the per-uid loop, so the uids already staged never reached an
    artifact and nothing committed at all.
    """
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    host.unpublishable = {_sequence(1)}

    _publish_once(
        host.args(
            strict_qualification=False,
            commit_command="/root/commit.sh {hotkey} {sequence} {block}",
        )
    )

    artifact = json.loads(host.artifact.read_text())
    published = [entry["uid"] for entry in artifact["submissions"]]
    assert len(published) == 2
    commits = [c for c in host.commands if c[0].endswith("commit.sh")]
    assert len(commits) == 2
    assert PUBLISH_FLEET_REJECTED in capsys.readouterr().err


def test_a_fleet_with_no_publishable_uid_stops_rather_than_writing_nothing(
    tmp_path, monkeypatch
):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    host.unpublishable = set(host.candidate_sequences)
    with pytest.raises(SystemExit, match="no uid could be published"):
        _publish_once(host.args(strict_qualification=False))


def test_uids_are_assigned_only_sequences_that_passed_current_layer_2(
    tmp_path, monkeypatch
):
    """The ranking is ordered by win probability and ignores Layer-2.

    In epoch 25092 only three of the top six ranked sequences held a current
    pass, so the fleet assigned three uids a sequence publish_checkpoint.py
    then refused outright. The assignable pool must be the passing subsequence.
    """
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    # Fail the top two: a naive top-k slice would hand these to uid 0 and uid 1.
    host.layer2_failures = {_sequence(0), _sequence(1)}

    _publish_once(host.args(strict_qualification=False))

    artifact = json.loads(host.artifact.read_text())
    published = {entry["sequence"] for entry in artifact["submissions"]}
    assert len(published) == 3
    assert not published & host.layer2_failures


def test_publication_stops_when_no_ranked_candidate_passed_layer_2(
    tmp_path, monkeypatch
):
    host = _Host(tmp_path, monkeypatch, uids=(0, 1))
    host.layer2_failures = set(host.candidate_sequences)
    with pytest.raises(SystemExit, match="no ranked candidate passed Layer-2"):
        _publish_once(host.args(strict_qualification=False))


def test_a_uid_is_never_assigned_a_sequence_the_validator_calls_a_duplicate(
    tmp_path, monkeypatch
):
    """Distinctness is not novelty, and this is what zero scores looked like.

    utils/nanobodies.py::is_duplicate drops a UID outright at >=0.95 whole
    sequence identity or >=0.85 CDR3 to a top sequence. Our epoch-25093 fleet
    was 0.95-0.98 identical to the epoch-25091 rank-1 sequence, so all six
    tickets were discarded before scoring and we appeared nowhere on the
    leaderboard while the plumbing reported six healthy commits.
    """
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    host.non_novel = {_sequence(0), _sequence(2)}

    _publish_once(host.args(strict_qualification=False))

    published = {
        entry["sequence"]
        for entry in json.loads(host.artifact.read_text())["submissions"]
    }
    assert not published & host.non_novel
    assert len(published) == 3


def test_publication_stops_when_every_candidate_is_a_validator_duplicate(
    tmp_path, monkeypatch
):
    """Committing a fleet the validator will discard is worse than not
    committing: it spends the epoch and returns nothing."""
    host = _Host(tmp_path, monkeypatch, uids=(0, 1))
    host.non_novel = set(host.candidate_sequences)
    with pytest.raises(SystemExit, match="validator-novelty"):
        _publish_once(host.args(strict_qualification=False))


def test_the_novelty_gate_runs_against_the_target_archive(tmp_path, monkeypatch):
    """It replays the validator's gates, so it needs the same archive."""
    host = _Host(tmp_path, monkeypatch, uids=(0, 1))
    _publish_once(host.args(strict_qualification=False))

    checks = [c for c in host.commands if "check-validator-novelty" in " ".join(c)]
    assert checks, "the novelty gate never ran"
    assert "--archive" in checks[0]
    assert "--target" in checks[0]


def test_no_uid_commits_before_its_scheduled_reserve_block(tmp_path, monkeypatch):
    """Every scored submission in the field commits inside the reserve.

    Epoch 25093's leaderboard ran blocks_elapsed 306-344 of 361 across the
    whole field; our two scored uids sat at 336 and 344 and the four we
    committed early were not scored at all. fleet.commitment_blocks() already
    computed that schedule -- _commit_fleet fired immediately and ignored it.
    """
    assignments = [
        fleet.Assignment(uid=uid, hotkey=f"5Hotkey{uid}", sequence="Q" * 120,
                         commit_block=RESERVE_START + index)
        for index, uid in enumerate((1, 40, 46))
    ]
    # Chain sits well before the reserve, as it does when the cycle finishes early.
    args = _commit_args(tmp_path, EPOCH_LENGTH * COMMIT_EPOCH + 17)
    seen = []

    def advance(seconds):
        state = json.loads(args.chain_state.read_text())
        state["block"] += max(1, int(seconds // 12))
        args.chain_state.write_text(json.dumps(state))

    def record(command):
        seen.append(json.loads(args.chain_state.read_text())["block"])

    monkeypatch.setattr(publish_current_nanobody, "_sleep", advance)
    monkeypatch.setattr(publish_current_nanobody, "_run", record)

    _commit_fleet(assignments, args=args, template="commit --uid {uid}",
                  artifact=tmp_path / "a.json", epoch=COMMIT_EPOCH,
                  pause_seconds=0)

    assert len(seen) == 3
    for block, assignment in zip(seen, assignments):
        assert block >= assignment.commit_block, "committed before the reserve"
        assert block < EPOCH_LENGTH * (COMMIT_EPOCH + 1), "committed after rollover"


def test_a_rolled_epoch_stops_the_wait_rather_than_committing_late(
    tmp_path, monkeypatch
):
    """A commitment landing after the rollover is worthless, not merely late."""
    assignments = [
        fleet.Assignment(uid=1, hotkey="5Hotkey1", sequence="Q" * 120,
                         commit_block=RESERVE_START)
    ]
    args = _commit_args(tmp_path, EPOCH_LENGTH * COMMIT_EPOCH + 17)

    def jump(seconds):
        state = json.loads(args.chain_state.read_text())
        state["block"] = EPOCH_LENGTH * (COMMIT_EPOCH + 1) + 5
        state["epoch"] = COMMIT_EPOCH + 1
        args.chain_state.write_text(json.dumps(state))

    monkeypatch.setattr(publish_current_nanobody, "_sleep", jump)
    monkeypatch.setattr(publish_current_nanobody, "_run",
                        lambda c: pytest.fail("committed into the wrong epoch"))

    with pytest.raises(SystemExit, match="epoch rolled"):
        _commit_fleet(assignments, args=args, template="commit --uid {uid}",
                      artifact=tmp_path / "a.json", epoch=COMMIT_EPOCH,
                      pause_seconds=0)


def test_the_fleet_does_not_pace_itself_out_of_a_tie_by_default():
    """Ties are broken by block_submitted, earlier first.

    Epoch 25096: our uid 12 scored 43.0 and so did uid 44, and we lost the
    epoch because our fleet landed at elapsed 328-350 against their 326. The
    old 12-second pace cost roughly eight blocks across eight uids. That pace
    was justified by max_regs_per_block, which caps registrations, not
    commitments -- in that same epoch uid 60 and uid 61 committed in one block.
    """
    assert _parser().parse_args([]).commit_pause_seconds == 0.0


def test_the_best_ranked_sequence_gets_the_earliest_commit_block(
    tmp_path, monkeypatch
):
    """Any of our tickets can be the one that ties, but the best-ranked has the
    best chance of it, so it must not be the one that commits last."""
    host = _Host(tmp_path, monkeypatch, uids=(0, 1, 2))
    _publish_once(host.args(strict_qualification=False))

    entries = json.loads(host.artifact.read_text())["submissions"]
    blocks = [entry["commit_block"] for entry in entries]
    assert blocks == sorted(blocks), "commit blocks are out of schedule order"
    ranked = [c["sequence"] for c in json.loads(
        (host.runtime / "live" / f"epoch{EPOCH}" / "provisional_rankings.json"
         ).read_text())["candidates"]]
    first = min(entries, key=lambda e: e["commit_block"])["sequence"]
    assert first == ranked[0], "the earliest block did not go to the best sequence"


def test_the_reserve_opening_is_noticed_within_a_block():
    """The miner rewrites the chain view about every 11 seconds; polling more
    slowly than that spends blocks discovering a reserve that already opened.
    Epoch 25098 committed first at elapsed 313 against a schedule of 306, and
    epoch 25097's winner committed at 306."""
    assert publish_current_nanobody.COMMIT_WAIT_POLL_SECONDS <= 6.0
