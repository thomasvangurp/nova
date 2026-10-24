"""Current cheap gates for stored folds, independently of fresh discovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import threading
from pathlib import Path

_JOBS: dict[tuple[str, int], dict] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_scope(root: Path, *, epoch: int, target: str, archive: Path,
               config: Path, opt_root: Path) -> tuple[set[str], dict]:
    """Never admit an unchecked entry or a stale epoch/target/gate proof."""
    try:
        from tools.publish_current_nanobody import PUBLISHER_STATE_SCHEMA_VERSION
        proof = json.loads((root / "comparison_precheck.json").read_text())
        key = proof["key"]
        if (key.get("schema_version") != PUBLISHER_STATE_SCHEMA_VERSION
                or key.get("policy") != "incremental-layer2-v1"
                or key.get("epoch") != epoch or key.get("target") != target
                or key.get("confirmation_only") is not True
                or key.get("minimum_observations") != 1
                or key.get("archive") != digest(archive)
                or key.get("config") != digest(config)
                or key.get("gates") != {name: digest(opt_root / name) for name in (
                    "check_validator_novelty.py", "filter_layer2.py")}):
            return set(), {"ready": False, "reason": "scope_changed"}
        allowed = set(proof["eligible_sequences"]) & set(proof["checked_sequences"])
        return allowed, {"ready": True, "eligible": len(allowed),
                         "pending": proof["pending_count"]}
    except (OSError, ValueError, KeyError, TypeError):
        return set(), {"ready": False, "reason": "not_prepared"}


def prepare(*, args, state, archive: Path, config: Path, root: Path, runner) -> None:
    """Checkpoint the entire stored bank; full Layer 2 remains a finalist gate."""
    from tools.publish_current_nanobody import _prechecked_comparison_scope
    root.mkdir(parents=True, exist_ok=True)
    snapshot_archive, snapshot_config = root / "archive.csv", root / "config.yaml"
    shutil.copy2(archive, snapshot_archive)
    shutil.copy2(config, snapshot_config)
    exclusion = root / "excluded_sequences.txt"
    ledger = getattr(args, "ledger", None)
    exclusion.write_text(Path(ledger).read_text() if ledger and Path(ledger).is_file() else "")
    gate_args = argparse.Namespace(
        nanobody_opt_root=args.runtime_root / "nanobody_opt",
        layer2_cache_dir=args.runtime_root / "layer2_cache",
        layer2_concurrency=args.layer2_concurrency, pixi=Path("/usr/local/bin/pixi"),
    )
    while True:
        _prechecked_comparison_scope(
            args=gate_args, epoch_root=root, epoch=int(state["epoch"]),
            target=state["nanobody_target"], database=args.mutation_database,
            archive=snapshot_archive, config=snapshot_config, exclusion=exclusion,
            pixi_manifest=args.runtime_root / "pixi.toml", ready_artifact=True,
            minimum_observations=1, cheap_only=True, runner=runner,
        )
        proof = json.loads((root / "comparison_precheck.json").read_text())
        print(f"CACHED_CONFIRMATION_PRECHECK epoch={state['epoch']} "
              f"eligible={proof['eligible_count']} pending={proof['pending_count']}", flush=True)
        if not proof["pending_count"]:
            return


def start(*, args, state, archive: Path, config: Path, producer_root: Path) -> None:
    """Run beside fresh scoring, bounded by its owning epoch; never join it."""
    from nanobody_opt_v2 import run_live_producer as producer
    from nanobody_opt.epoch_deadline import EpochDeadline
    root = producer_root / "cached_confirmation"
    key = (str(root.resolve()), int(state["epoch"]))
    record = _JOBS.get(key)
    if record and record["thread"].is_alive():
        return
    _, status = read_scope(root, epoch=state["epoch"], target=state["nanobody_target"],
                           archive=archive, config=config,
                           opt_root=args.runtime_root / "nanobody_opt")
    if status.get("ready") and not status["pending"]:
        return
    record = {}
    def work():
        producer._STAGED.deadline = EpochDeadline(
            args.chain_state, int(state["epoch"]), producer._handoff_reserve(args),
            args.maximum_chain_state_age)
        producer._STAGED.nice = True
        try:
            producer._STAGED.deadline.remaining()
            prepare(args=args, state=state, archive=archive, config=config,
                    root=root, runner=producer._run)
            record["complete"] = True
        except (Exception, SystemExit) as error:
            record["error_type"] = type(error).__name__
            print(f"CACHED_CONFIRMATION_RETRY epoch={state['epoch']} "
                  f"error={type(error).__name__}", flush=True)
    thread = threading.Thread(target=work, daemon=True)
    record["thread"] = thread
    _JOBS[key] = record
    thread.start()


def write_ranking_scope(*, root: Path, epoch: int, target: str, archive: Path,
                        config: Path, fresh_sequences: set[str], output: Path) -> Path | None:
    from nanobody_opt_v2 import sequential_exact_race as race
    cached, status = read_scope(root, epoch=epoch, target=target, archive=archive,
                               config=config, opt_root=Path(race.__file__).resolve().parents[1] / "nanobody_opt")
    print(f"CACHED_CONFIRMATION_SCOPE epoch={epoch} ready={status['ready']} "
          f"cached={len(cached)} pending={status.get('pending', 0)}", flush=True)
    if not cached:
        return None
    race._atomic_json(output.with_suffix(".json"), {"epoch": epoch, "cached": len(cached),
                       "fresh": len(fresh_sequences), "pending": status.get("pending", 0)})
    output.write_text("".join(s + "\n" for s in sorted(cached | fresh_sequences)))
    return output
