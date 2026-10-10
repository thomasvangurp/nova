"""Continuously generate, qualify, and exact-race live nanobody candidates.

The chain miner remains the source of current epoch/target state.  This process
owns only producer artifacts: it refreshes the public challenge inputs, builds
winner-informed local neighborhoods, runs locked Pixi Layer-2, and hands the
survivors to the candidate-major exact race.  Completed exact observations are
picked up atomically by the independent checkpoint publisher.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

from nanobody_opt import db


CONFIG_URL = "https://raw.githubusercontent.com/metanova-labs/nova/main/config/config.yaml"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/"
    "{target}_nanobodies.csv"
)


#: How many previous epochs seed the timing fit. Enough that the very first
#: plan of an epoch has real replication rows to fit; short enough that a
#: genuine change in cost is reflected within an epoch or two.
BUDGET_SEED_EPOCHS = 3
DEFAULT_MINIMUM_FREE_GIB = 20.0
DEFAULT_MAXIMUM_PERSISTED_RANKING_MIB = 64.0
DEFAULT_RETAINED_EPOCHS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed_exact_files(database: Path, paths: list[Path]) -> bool:
    """Prove every raw exact file is losslessly represented in SQLite.

    An epoch is eligible for removal only when every raw run is byte-matched,
    sealed with all expected observations, and has the structure/sample audit
    row extracted by the importer. Any ambiguity fails closed.
    """
    if not paths:
        return False
    try:
        connection = db.connect()
        rows = connection.execute(
            """
            SELECT r.source_path, r.file_sha256, r.expected_rows,
                   r.ingested_rows, r.complete,
                   COUNT(DISTINCT x.observation_id),
                   COUNT(DISTINCT a.observation_id)
            FROM exact_runs r
            LEFT JOIN exact_run_observations x
              ON x.target=r.target AND x.run_id=r.run_id
            LEFT JOIN observation_artifacts a
              ON a.observation_id=x.observation_id
            WHERE r.target='P20809'
            GROUP BY r.target, r.run_id
            """
        ).fetchall()
    except Exception:
        return False
    finally:
        if "connection" in locals():
            connection.close()
    sealed = {str(row[0]): row[1:] for row in rows}
    for path in paths:
        row = sealed.get(str(path))
        if row is None:
            return False
        digest, expected, ingested, complete, linked, artifacts = row
        expected = int(expected)
        if not (
            expected > 0
            and int(ingested) == expected
            and int(complete) == 1
            and int(linked) == expected
            and int(artifacts) == expected
            and str(digest) == _sha256(path)
        ):
            return False
    return True


def maintain_runtime_storage(
    runtime_root: Path,
    current_epoch: int,
    *,
    minimum_free_gib: float = DEFAULT_MINIMUM_FREE_GIB,
    maximum_persisted_ranking_mib: float = DEFAULT_MAXIMUM_PERSISTED_RANKING_MIB,
    observation_database: Path | None = None,
    retained_epochs: int = DEFAULT_RETAINED_EPOCHS,
) -> dict[str, int]:
    """Prune regenerable legacy rank matrices and enforce a disk reserve.

    Only the two internal race rankings are eligible.  They are derived from
    retained ``exact_all.json`` observations and are not publisher inputs once
    their epoch has completed. Older epoch trees are removed only after every
    raw exact file and its extracted audit metadata are byte-verified in the
    normalized observation database. Publication artifacts are never touched.
    """
    live_root = runtime_root / "live"
    maximum_bytes = int(maximum_persisted_ranking_mib * 1024**2)
    removed_files = 0
    removed_bytes = 0
    removed_epoch_dirs = 0
    if live_root.is_dir():
        for epoch_root in live_root.glob("epoch[0-9]*"):
            suffix = epoch_root.name.removeprefix("epoch")
            if not suffix.isdigit() or int(suffix) >= current_epoch:
                continue
            for pattern in (
                "producer/iteration_*/race/screen_ranking.json",
                "producer/iteration_*/race/rounds/stage_ranking.json",
            ):
                for path in epoch_root.glob(pattern):
                    try:
                        size = path.stat().st_size
                    except FileNotFoundError:
                        continue
                    if size <= maximum_bytes:
                        continue
                    path.unlink()
                    removed_files += 1
                    removed_bytes += size
        if observation_database is not None and retained_epochs >= 1:
            cutoff = current_epoch - retained_epochs
            for epoch_root in sorted(live_root.glob("epoch[0-9]*")):
                suffix = epoch_root.name.removeprefix("epoch")
                if (
                    not suffix.isdigit()
                    or int(suffix) >= cutoff
                    or epoch_root.is_symlink()
                ):
                    continue
                exact_files = sorted(epoch_root.glob("**/exact_all.json"))
                if not _sealed_exact_files(observation_database, exact_files):
                    continue
                # Preserve per-residue structures (.cif) before deleting the
                # epoch tree: they are the only source of per-residue contact
                # topology (the observations DB stores only aggregate metrics)
                # and are not cheaply regenerable. The .cif sync loop ships
                # these to Hetzner for persistence.
                structures_root = live_root / "structures"
                structures_root.mkdir(parents=True, exist_ok=True)
                for cif in epoch_root.rglob("*.cif"):
                    if not cif.is_file() or cif.is_symlink():
                        continue
                    try:
                        shutil.move(str(cif), str(structures_root / cif.name))
                    except (FileNotFoundError, shutil.Error):
                        pass
                for path in epoch_root.rglob("*"):
                    try:
                        if path.is_file() and not path.is_symlink():
                            removed_bytes += path.stat().st_size
                            removed_files += 1
                    except FileNotFoundError:
                        pass
                shutil.rmtree(epoch_root)
                removed_epoch_dirs += 1
    free_bytes = shutil.disk_usage(runtime_root).free
    minimum_bytes = int(minimum_free_gib * 1024**3)
    if free_bytes < minimum_bytes:
        raise RuntimeError(
            "producer disk reserve violated: "
            f"free={free_bytes} required={minimum_bytes}; refusing new GPU work"
        )
    if removed_files or removed_epoch_dirs:
        print(
            "PRODUCER_STORAGE_PRUNED "
            f"epoch_dirs={removed_epoch_dirs} files={removed_files} "
            f"bytes={removed_bytes} free={free_bytes}",
            flush=True,
        )
    return {
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "removed_epoch_dirs": removed_epoch_dirs,
        "free_bytes": free_bytes,
    }


def _budget_plan(args: argparse.Namespace, state: dict, producer_root: Path) -> dict:
    if args.budget_mode != "adaptive" or not args.replication_rounds:
        return {"replication_reserve_blocks": args.replication_reserve_blocks,
                "confirmation_candidates": args.finalists}
    from nanobody_opt.compute_budget import TimingModel, confirmation_plan

    # SEED FROM RECENT EPOCHS, not this one alone. Replication rows only
    # exist after confirmation has run, but the width that governs
    # confirmation is decided at iteration 1 -- so with this epoch's records
    # alone the fitted model sees zero replication rounds every time and
    # falls back to the raw CLI defaults. Measured on the live box:
    # iterations 1-3 carry 0 replication records, so the fit had never once
    # run in the decision it exists for. The previous epochs supply ~119.
    #
    # Oldest first: the fit keeps the most recent window, so this epoch's
    # rows must come last and displace older ones as they arrive.
    epoch = int(state["epoch"])
    live_root = producer_root.parent.parent
    roots = [live_root / f"epoch{past}" / "producer"
             for past in range(epoch - BUDGET_SEED_EPOCHS, epoch)]
    records = []
    for root in [*roots, producer_root]:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("iteration_[0-9][0-9]/race/timings.jsonl")):
            try:
                text = path.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    model = TimingModel(args.budget_load_seconds, args.budget_gpu_seconds,
                        args.budget_cpu_guard_seconds).fit_from_measurements(
                            records, len(args.gpus.split(",")))
    # The run prunes between stages, so the plan must price it that way or it
    # reserves blocks it will never spend.
    plan = confirmation_plan(epoch_length=int(state["epoch_length"]),
                             commitment_blocks=_handoff_reserve(args),
                             rounds=args.replication_rounds,
                             maximum_candidates=args.finalists,
                             gpus=len(args.gpus.split(",")), model=model,
                             time_share=args.confirmation_time_share,
                             rounds_per_stage=args.replication_rounds_per_stage,
                             keep_fraction=args.replication_keep_fraction)
    _atomic_json(producer_root / "budget.json", {"epoch": int(state["epoch"]),
                 "commitment_reserve_blocks": args.reserve_blocks,
                 "publication_prepare_blocks": args.publication_prepare_blocks, **plan})
    return plan


def _closing(args: argparse.Namespace, state: dict, producer_root: Path) -> bool:
    remaining = int(state["epoch_length"]) - int(state["block"]) % int(state["epoch_length"])
    plan = _budget_plan(args, state, producer_root)
    return remaining <= _handoff_reserve(args) + plan["replication_reserve_blocks"]


def _handoff_reserve(args: argparse.Namespace) -> int:
    return args.reserve_blocks + (args.publication_prepare_blocks if args.budget_mode == "adaptive" else 0)


def visible_gpu_ids() -> str:
    """The GPU ids this machine actually has, as the scorer's comma list.

    The default used to be the literal "0,1,2,3,4,5,6,7,8,9", which was the old
    ten-GPU box written down. On a two-GPU box that dispatches eight of every ten
    shards to devices that do not exist; each chunk raises, the shard worker
    catches it and continues, and the run ends with "MERGE LOST SEQUENCES:
    submitted 350, merged 70" -- two surviving shards of ten, which is exactly
    the 70. Nothing in the failure names CUDA or the missing devices.

    Counting the devices costs one import and cannot be wrong about the hardware
    it is running on.
    """
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    if count < 1:
        return "0"
    return ",".join(str(index) for index in range(count))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2) + "\n").encode())


def _download(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "nova-nanobody-producer/1"})
    with urlopen(request, timeout=60) as response:  # noqa: S310
        payload = response.read()
    if len(payload) < 100:
        raise ValueError(f"download from {url} is unexpectedly small")
    _atomic_bytes(path, payload)


def _chain_state(path: Path, maximum_age: float) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    age = time.time() - float(payload["updated_at_unix"])
    if age < 0 or age > maximum_age:
        raise ValueError(f"chain state is stale by {age:.1f}s")
    block = int(payload["block"])
    epoch_length = int(payload["epoch_length"])
    epoch = int(payload["epoch"])
    if epoch_length < 1 or epoch != block // epoch_length:
        raise ValueError("chain state epoch is internally inconsistent")
    return {
        **payload,
        "block": block,
        "epoch_length": epoch_length,
        "epoch": epoch,
    }


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _prepare_epoch(
    *, root: Path, epoch: int, expected_target: str
) -> tuple[Path, Path, str]:
    epoch_root = root / "live" / f"epoch{epoch}" / "producer"
    config = epoch_root / "config.yaml"
    _download(CONFIG_URL, config)
    challenge = yaml.safe_load(config.read_text())["protein_selection"]["nanobody"]
    target = str(challenge["target"])
    if target != expected_target:
        raise ValueError(
            f"chain miner target {expected_target!r} disagrees with live config {target!r}"
        )
    archive = epoch_root / "archive.csv"
    _download(ARCHIVE_URL.format(target=target), archive)
    return config, archive, target


def refresh_donor_grafts(
    *,
    epoch_root: Path,
    archive: Path,
    config: Path,
    pixi: Path,
    manifest: Path,
    teacher_rank: int,
    limit: int,
    donors: int,
) -> Path | None:
    """Rebuild the graft pool from this epoch's archive, or keep the last one.

    Generated per epoch because the teacher is the archive's best measured
    sequence and the donors are public frameworks that keep arriving: a graft
    pool pinned at deploy time goes stale exactly the way a remembered uid did.

    A failure here is not fatal. Grafts widen the pool; losing them costs
    diversity for one epoch, while stopping the producer costs the epoch
    itself.
    """
    teachers = epoch_root / "structural_teachers.json"
    grafts = epoch_root / "donor_grafts.json"
    if grafts.is_file():
        return grafts
    base = [
        str(pixi), "run", "--manifest-path", str(manifest), "-e", "metanano",
        "python", "-m",
    ]
    try:
        _run(base + [
            "nanobody_opt.select_structural_teachers",
            "--archive", str(archive),
            "--config", str(config),
            "--output", str(teachers),
        ])
        _run(base + [
            "nanobody_opt.generate_donor_grafts",
            "--teachers", str(teachers),
            "--archive", str(archive),
            "--config", str(config),
            "--teacher-rank", str(teacher_rank),
            "--limit", str(limit),
            "--donors", str(donors),
            "--output", str(grafts),
        ])
    except (subprocess.CalledProcessError, OSError) as error:
        print(
            f"donor grafts unavailable this epoch ({error}); continuing with "
            "neighbourhood proposals alone",
            file=sys.stderr,
            flush=True,
        )
        return None
    return grafts


def _looks_exhausted(iteration_root: Path) -> bool:
    """Whether this iteration failed because the neighbourhood is used up.

    Saturation and breakage both surface as a non-zero exit, and only one of them
    is worth retrying. The generator prints a marker for the first; anything else
    keeps the existing retry behaviour, because a real fault should not silently
    end an epoch's generation.
    """
    from nanobody_opt.local_neighborhood import EXHAUSTED_SUFFIX

    # A sidecar file, because the producer streams child output to its own
    # terminal instead of capturing it -- there is no log here to grep.
    return (iteration_root / ("candidates.csv" + EXHAUSTED_SUFFIX)).is_file()


def _run_iteration(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    iteration: int,
) -> None:
    preparation_started = time.monotonic()
    epoch = int(state["epoch"])
    target = str(state["nanobody_target"])
    config, archive, target = _prepare_epoch(
        root=args.runtime_root,
        epoch=epoch,
        expected_target=target,
    )
    producer_root = args.runtime_root / "live" / f"epoch{epoch}" / "producer"
    iteration_root = producer_root / f"iteration_{iteration:02d}"
    if (iteration_root / "race/complete.json").is_file():
        return
    from nanobody_opt.consensus_trial import write_immutable
    write_immutable(iteration_root / "scientific_configuration.json", {
        "epoch": epoch, "iteration": iteration,
        "parameters": {key: getattr(args, key, None) for key in (
            "selection_metric_policy", "consensus_ranking_policy",
            "settled_deviation_policy", "minimum_selection_observations",
            "parent_limit", "neighbor_parent_limit", "proposal_limit",
            "candidate_limit", "screen_tranche_size", "finalists",
            "replication_rounds", "replication_rounds_per_stage",
            "replication_keep_fraction", "minimum_publishable_observations",
            "qualification_observations", "basin_core_limit",
            "basin_parent_limit", "basin_fraction", "basin_evidence_fraction",
            "public_neighbourhood_distance", "scout_iterations",
            "exploit_basin_fraction", "exploit_basin_core_limit",
            "screen_predict_batch_size", "budget_mode", "confirmation_time_share",
            "publication_prepare_blocks", "reserve_blocks",
        )},
    })
    preparations = [json.loads(p.read_text())["seconds"] for p in
                    sorted(producer_root.glob("iteration_[0-9][0-9]/producer_preparation.json"))[-2:]]
    preparation_guard = max(preparations, default=60.0) * 1.15
    plan = _budget_plan(args, state, producer_root)
    remaining = int(state["epoch_length"]) - int(state["block"]) % int(state["epoch_length"])
    confirmation_due = (
        remaining * 12 <= (_handoff_reserve(args) + plan["replication_reserve_blocks"]) * 12 + preparation_guard
    )
    if args.budget_mode == "adaptive" and args.replication_rounds and confirmation_due:
        retained = [p for p in sorted(producer_root.glob("iteration_[0-9][0-9]/layer2.json"))
                    if (d := json.loads(p.read_text())).get("candidates") or d.get("sequences")]
        exact = list(producer_root.glob("iteration_[0-9][0-9]/race/**/exact_all.json"))
        if retained and exact:
            _atomic_json(iteration_root / "replication_only.json", {
                "epoch": epoch, "layer2_source": str(retained[-1].relative_to(producer_root)),
                "reason": "remaining time is reserved for confirmation",
            })
            _run_race(args=args, state=state, iteration=iteration,
                      config=config, archive=archive, target=target,
                      producer_root=producer_root, iteration_root=iteration_root,
                      layer2=retained[-1], force_confirmation=True)
            return
    winner_priors = producer_root / "winner_priors.json"
    if not winner_priors.is_file():
        _run([
            sys.executable,
            "-m", "nanobody_opt.analyze_epoch_winners",
            "--archive", str(archive),
            "--filename", f"{target}_nanobodies.csv",
            "--epochs", "40",
            "--output", str(winner_priors),
        ])
    # Broad single-edit parent lane (the winning competitor's own strategy:
    # single framework edits across the unfiltered public pool) REPLACES the
    # elite-centric select_elite_parents bank. The single-edits flow through
    # filter_layer2 and local_neighborhood exactly as the elite parents did.
    top50 = iteration_root / "top50_index.json"
    # The dedup top-50 is rank_binders over the FULL HF archive (all history),
    # not the live leaderboard -- a 15-epoch API window indexes a different
    # basin and over-filters (the 25454/25455 wholesale rejections). Fetch the
    # archive once here; the layer-2 novelty gate below reuses the same file.
    hf_archive = iteration_root / "hf_archive.csv"
    if not hf_archive.is_file():
        _run([sys.executable, "-m", "nanobody_opt.fetch_hf_archive", str(hf_archive)])
    if not top50.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked", "--environment", "metanano",
            "python", "-m", "nanobody_opt.build_top50_index",
            str(hf_archive), str(top50), str(args.mutation_database),
        ])
    # Children pool (breadth): the multi-edit elite parents, generated alongside
    # the single-edit competitor lane. Both are scored and the source column
    # tracks which pool each observation came from.
    children_parents = producer_root / "elite_parents.json"
    if not children_parents.is_file():
        _run([
            sys.executable,
            "-m", "nanobody_opt.select_elite_parents",
            "--target", target,
            "--before-epoch", str(epoch),
            "--epochs", str(args.parent_epochs),
            "--per-epoch", str(args.parents_per_epoch),
            "--limit", str(args.parent_limit),
            "--neighbor-per-epoch", str(args.neighbor_parents_per_epoch),
            "--neighbor-distance", str(args.neighbor_parent_distance),
            "--neighbor-limit", str(args.neighbor_parent_limit),
            "--terminal-variants-per-residue", str(
                args.terminal_variants_per_residue
            ),
            "--recent-epoch-window", str(args.recent_parent_epoch_window),
            "--recent-anchor-limit", str(args.recent_anchor_parent_limit),
            "--recent-neighbor-limit", str(args.recent_neighbor_parent_limit),
            "--basin-core-limit", str(args.basin_core_limit),
            "--basin-parent-limit", str(args.basin_parent_limit),
            "--priority-coldkeys", str(args.priority_coldkeys),
            "--scan-limit", str(args.parent_scan_limit),
            "--observation-database", str(args.mutation_database),
            "--output", str(children_parents),
        ])
    elite_parents = producer_root / "competitor_parents.json"
    if not elite_parents.is_file():
        lane_argv = [
            sys.executable,
            "-m", "nanobody_opt.competitor_lane",
            "--archive", str(archive),
            "--priority-parents", str(top50),
            "--rival-coldkey", args.rival_coldkey,
            "--limit", "2000",
            "--mutation-database", str(args.mutation_database),
            "--output", str(elite_parents),
        ]
        if args.ranker_model.is_file():
            lane_argv += ["--ranker-model", str(args.ranker_model)]
        _run(lane_argv)
    seed = producer_root / "competitor_parents_layer2.json"
    if not seed.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked",
            "--environment", "metanano",
            "filter-layer2", "--",
            "--input", str(elite_parents),
            "--config", str(config),
            "--output", str(seed),
            "--concurrency", str(args.layer2_concurrency),
            "--igblast-concurrency", str(args.layer2_concurrency),
            "--igblast-batch-size", str(args.igblast_batch_size),
            "--cache-dir", str(args.runtime_root / "layer2_cache"),
        ])
    children_seed = producer_root / "elite_parents_layer2.json"
    if not children_seed.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked",
            "--environment", "metanano",
            "filter-layer2", "--",
            "--input", str(children_parents),
            "--config", str(config),
            "--output", str(children_seed),
            "--concurrency", str(args.layer2_concurrency),
            "--igblast-concurrency", str(args.layer2_concurrency),
            "--igblast-batch-size", str(args.igblast_batch_size),
            "--cache-dir", str(args.runtime_root / "layer2_cache"),
        ])
    # Scout, then exploit.  The first generation spreads across the strongest
    # historical cores; later generations narrow onto the cores that scored best
    # in this epoch's own observations, which the race has already written to
    # the mutation database.  Held-out replay puts the eventual winner's core in
    # the top five 6/6 times, so the epoch's remaining budget is better spent
    # deepening those than re-spreading across cores already measured as weak.
    exploiting = iteration > args.scout_iterations
    basin_fraction = (
        args.exploit_basin_fraction if exploiting else args.basin_fraction
    )
    basin_evidence_core_limit = (
        args.exploit_basin_core_limit
        if exploiting
        else args.basin_evidence_core_limit
    )
    print(
        f"PRODUCER_LANE iteration={iteration} "
        f"mode={'exploit' if exploiting else 'scout'} "
        f"basin_fraction={basin_fraction} "
        f"evidence_core_limit={basin_evidence_core_limit}",
        flush=True,
    )
    candidates = iteration_root / "candidates.csv"
    if not candidates.is_file():
        # Two pools, each tagged with its source so the race's observations can
        # be attributed per pool: single-edits (competitor lane, exploitation)
        # and multi-edit children (elite parents + neighbourhood, breadth).
        single_rows = json.loads(seed.read_text()).get("candidates", [])
        with candidates.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sequence", "source", "parent_sequence"]
            )
            writer.writeheader()
            for row in single_rows:
                seq = str(row.get("sequence", "")).strip().upper()
                if seq:
                    writer.writerow({
                        "sequence": seq,
                        "source": "single_edit",
                        "parent_sequence": str(row.get("parent", "")).strip().upper(),
                    })

        children_candidates = iteration_root / "children_candidates.csv"
        if not children_candidates.is_file():
            command = [
                sys.executable,
                "-m", "nanobody_opt.local_neighborhood",
                "--archive", str(archive),
                "--config", str(config),
                "--layer2-seeds", str(children_seed),
                "--winner-priors", str(winner_priors),
                "--output", str(children_candidates),
                "--seed-count", str(args.seed_count),
                "--neighbor-seed-count", str(args.neighbor_seed_count),
                "--minimum-seeds", str(args.minimum_seeds),
                "--per-seed", str(args.per_seed),
                "--generation-workers", str(args.generation_workers),
                "--generation-workers-minimum-candidates",
                str(args.generation_workers_minimum_candidates),
                "--proposal-limit", str(args.proposal_limit),
                "--basin-fraction", str(basin_fraction),
                "--basin-routes-per-core", str(args.basin_routes_per_core),
                "--basin-evidence-fraction", str(args.basin_evidence_fraction),
                "--basin-evidence-core-limit", str(basin_evidence_core_limit),
                "--basin-top-core-weight", str(args.basin_top_core_weight),
                "--basin-exploration-residues-per-position", str(
                    args.basin_exploration_residues_per_position
                ),
                "--framework-fraction", str(args.framework_fraction),
                "--haplotype-lane-fraction", str(args.haplotype_lane_fraction),
                "--framework-per-seed", str(args.framework_per_seed),
                "--terminal-width", str(args.terminal_width),
                "--maximum-terminal-haplotype-mutations", str(
                    args.maximum_terminal_haplotype_mutations
                ),
                "--mutation-observations", str(args.mutation_database),
                "--motif-fraction", str(args.motif_fraction),
                "--single-mutation-fraction", str(args.single_mutation_fraction),
                "--maximum-motif-mutations", str(args.maximum_motif_mutations),
                *(
                    ["--maximum-exhaustive-singles", str(args.maximum_exhaustive_singles)]
                    if args.maximum_exhaustive_singles is not None
                    else []
                ),
                "--settled-deviation-policy", str(args.settled_deviation_policy),
                "--random-seed", str(epoch * 100 + iteration),
            ]
            for previous in sorted(producer_root.glob("iteration_*/children_candidates.csv")):
                if previous != children_candidates:
                    command.extend(["--exclude", str(previous)])
            _run(command)
        # Append the children pool to candidates.csv with its source tag.
        with candidates.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sequence", "source", "parent_sequence"]
            )
            with children_candidates.open(newline="") as ch:
                for row in csv.DictReader(ch):
                    seq = str(row.get("sequence", "")).strip().upper()
                    if seq:
                        writer.writerow({
                            "sequence": seq,
                            "source": "children",
                            "parent_sequence": str(
                                row.get("parent_sequence", "")
                            ).strip().upper(),
                        })

        if args.public_neighbourhood_distance:
            # Every winner in the leakage-free replay sat one substitution from
            # a sequence another hotkey had already published, and about forty
            # per cent of each generation sits further out than that.  A
            # 320-candidate budget had to gamble beyond the shell; a
            # several-thousand-observation budget can cover it instead.
            filtered = iteration_root / "candidates_public.csv"
            _run([
                sys.executable,
                "-m", "nanobody_opt.filter_public_neighborhood",
                "--input", str(candidates),
                "--archive", str(archive),
                "--output", str(filtered),
                "--maximum-distance", str(args.public_neighbourhood_distance),
                "--minimum-keep", str(args.candidate_limit),
            ])
            os.replace(filtered, candidates)
        # AFTER the public-neighbourhood filter, never before it. That filter
        # keeps proposals within --public-neighbourhood-distance substitutions
        # of a published sequence, which is 2 in production, and a graft is
        # 16-18 mutations from its teacher by construction. Merging first meant
        # every graft was added and then stripped on the next line: epochs
        # 25097 and 25098 generated sixty cases each and screened none of them.
        # The filter is an exploitation heuristic; grafts are the deliberate
        # exception to it.
        grafts = args.graft_candidates
        if grafts is None and args.donor_grafts:
            grafts = refresh_donor_grafts(
                epoch_root=args.runtime_root / "live" / f"epoch{epoch}",
                archive=archive,
                config=config,
                pixi=Path("/usr/local/bin/pixi"),
                manifest=args.runtime_root / "pixi.toml",
                teacher_rank=args.graft_teacher_rank,
                limit=args.graft_limit,
                donors=args.graft_donors,
            )
        if grafts is not None:
            merge_graft_candidates(candidates, grafts)
    layer2 = iteration_root / "layer2.json"
    if not layer2.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked",
            "--environment", "metanano",
            "filter-layer2", "--",
            "--input", str(candidates),
            "--config", str(config),
            "--output", str(layer2),
            "--concurrency", str(args.layer2_concurrency),
            "--igblast-concurrency", str(args.layer2_concurrency),
            "--igblast-batch-size", str(args.igblast_batch_size),
            "--cache-dir", str(args.runtime_root / "layer2_cache"),
        ] + (["--igblast-only"] if args.layer2_igblast_only else []))
    # The validator's EXACT novelty gate (Algorithm-2 top-50 + SearchEngine fine
    # alignment + is_duplicate), indexed from the FULL HF archive (all history),
    # not the live leaderboard. Our old gates ranked the last 15 API epochs, a
    # different basin ~30 edits from the validator's reference, so they over-
    # filtered molecules the validator accepts (5GhREcsK wins with such edits).
    hf_archive = iteration_root / "hf_archive.csv"
    if not hf_archive.is_file():
        _run([
            sys.executable,
            "-m", "nanobody_opt.fetch_hf_archive",
            str(hf_archive),
        ])
    layer2_dedup = iteration_root / "layer2_dedup.json"
    if not layer2_dedup.is_file():
        _run([
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked", "--environment", "metanano",
            "python", "-m", "nanobody_opt.check_validator_novelty",
            "--archive", str(hf_archive),
            "--target", target,
            "--input", str(layer2),
            "--output", str(layer2_dedup),
        ])
    _atomic_json(iteration_root / "producer_preparation.json", {"seconds": round(time.monotonic() - preparation_started, 1)})
    _run_race(args=args, state=state, iteration=iteration,
              config=config, archive=archive, target=target,
              producer_root=producer_root, iteration_root=iteration_root,
              layer2=layer2_dedup)


def _run_race(*, args: argparse.Namespace, state: dict, iteration: int,
              config: Path, archive: Path, target: str, producer_root: Path,
              iteration_root: Path, layer2: Path, force_confirmation: bool = False) -> None:
    epoch = int(state["epoch"])
    # Generate-score-regenerate: every iteration screens a fresh generation
    # steered by the observations the previous ones wrote to the mutation
    # database.  Replication is spent once, in the iteration that opens inside
    # the closing window, over the pooled best of the whole epoch -- doing it
    # per iteration would burn the budget on the first generation's leader.
    # Re-read the chain rather than trusting the snapshot main() took: the work
    # between then and here -- downloads, parent selection, elite Layer 2,
    # proposal generation over thousands of candidates, pool IgBLAST -- is
    # minutes long, and a stale snapshot decides the closing generation wrongly.
    #
    # Through _chain_state, not a bare read: the point of re-reading is freshness,
    # and a bare read accepts a file whose miner died an hour ago as long as its
    # epoch field still matches -- which it does for up to a whole epoch.  That
    # substitutes an older block for a newer one and turns the closing generation
    # into a non-closing one, spending no replication budget at all.
    try:
        current = _chain_state(Path(args.chain_state), args.maximum_chain_state_age)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"PRODUCER_CHAIN_STATE_REREAD_REFUSED {error}", flush=True)
    else:
        if int(current["epoch"]) == epoch:
            state = current
    remaining_blocks = int(state["epoch_length"]) - (
        int(state["block"]) % int(state["epoch_length"])
    )
    # A non-closing generation stops screening at the same boundary and then
    # exits immediately, because two-stage mode spends no budget on greedy
    # per-tranche repeats.  The closing generation therefore starts with the
    # whole reserve intact, which is what makes room for enough rounds to reach
    # the observation count strict qualification requires.
    plan = _budget_plan(args, state, producer_root)
    closing = force_confirmation or remaining_blocks <= plan["replication_reserve_blocks"] + _handoff_reserve(args)
    replication_rounds = args.replication_rounds if closing else 0
    print(
        f"PRODUCER_GENERATION iteration={iteration} "
        f"remaining_blocks={remaining_blocks} closing={closing} "
        f"replication_rounds={replication_rounds}",
        flush=True,
    )
    race_command = [
        sys.executable,
        "-m", "nanobody_opt.sequential_exact_race",
        "--layer2", str(layer2),
        "--archive", str(archive),
        "--config", str(config),
        "--target", target,
        "--epoch", str(epoch),
        "--output-dir", str(iteration_root / "race"),
        "--ledger", str(args.ledger),
        "--mutation-database", str(args.mutation_database),
        "--chain-state", str(args.chain_state),
        "--reserve-blocks", str(_handoff_reserve(args)),
        "--gpus", args.gpus,
        "--scorer", str(args.exact_scorer),
        "--candidate-limit", str(args.candidate_limit),
        "--screen-tranche-size", str(args.screen_tranche_size),
        "--maximum-initial-screen-p90", str(args.maximum_initial_screen_p90),
        "--weak-initial-fallback-after", str(args.weak_initial_fallback_after),
        "--finalists", str(plan["confirmation_candidates"] or 1),
        "--challenger-p90-margin", str(args.challenger_p90_margin),
        "--basin-shrinkage-strength", str(args.basin_shrinkage_strength),
        "--minimum-basin-samples", str(args.minimum_basin_samples),
        "--qualification-observations", str(args.qualification_observations),
        "--pipeline-depth", str(args.pipeline_depth),
        "--screen-predict-batch-size", str(args.screen_predict_batch_size),
        "--two-stage", str(1 if args.replication_rounds else 0),
        "--minimum-publishable-observations", str(args.minimum_publishable_observations),
        "--replication-rounds", str(replication_rounds),
        "--pooled-exact-root", str(producer_root),
        "--replication-reserve-blocks", str(plan["replication_reserve_blocks"]),
        "--replication-rounds-per-stage", str(args.replication_rounds_per_stage),
        "--replication-keep-fraction", str(args.replication_keep_fraction),
        "--strict-gates", str(args.strict_gates or ""),
        "--selection-metric-policy", getattr(args, "selection_metric_policy", "plain"),
        "--consensus-ranking-policy", getattr(args, "consensus_ranking_policy", "tier"),
    ]
    if getattr(args, "minimum_selection_observations", None) is not None:
        race_command.extend(["--minimum-selection-observations", str(args.minimum_selection_observations)])
    if args.budget_mode == "adaptive":
        race_command.extend([
            "--carry-pending-candidates", "--maximum-screen-tranches",
            str(getattr(args, "maximum_screen_tranches_override", 1)),
            "--replication-load-seconds", str(plan["load_seconds"]),
            "--replication-seconds-per-observation", str(plan["gpu_seconds_per_candidate"]),
            "--screen-gpu-seconds-per-candidate", str(plan["gpu_seconds_per_candidate"]),
            "--replication-finish-guard-seconds", "30",
            "--adaptive-finish-guard",
            "--maximum-current-epoch-observations", str(args.replication_rounds + 1),
        ])
        if force_confirmation:
            race_command.append("--replication-only")
    if args.layer2_igblast_only:
        # The pool was screened on IgBLAST alone, so the shortlist must earn
        # full Layer-2 evidence before anything can be published.
        gate = [
            "/usr/local/bin/pixi", "run",
            "--manifest-path", str(args.runtime_root / "pixi.toml"),
            "--locked", "--environment", "metanano", "filter-layer2", "--",
            "--input", "{input}",
            "--config", str(config),
            "--output", "{output}",
            "--concurrency", str(args.layer2_concurrency),
            "--igblast-concurrency", str(args.layer2_concurrency),
            "--igblast-batch-size", str(args.igblast_batch_size),
            "--cache-dir", str(args.runtime_root / "layer2_cache"),
        ]
        race_command.extend(
            ["--layer2-gate-command", " ".join(shlex.quote(item) for item in gate)]
        )
    _run(race_command)


GRAFT_PROPOSAL_TYPE = "donor_graft"


def merge_graft_candidates(candidates: Path, grafts: Path) -> int:
    """Add donor-graft sequences to an iteration's proposals, in place.

    Neighbourhood search explores one molecule's surroundings, so the pool it
    produces is one molecule: across epochs 25093-25095 every candidate sat
    within 0.93 identity of every other, we committed seven submissions and the
    validator scored three, and a hard separation rule would have left ONE
    assignable candidate out of thirty.

    Donor grafts are the diversity that pool lacks -- the measured paratope on
    a different validator-proven VHH framework, 0.85-0.89 from our own lineage
    rather than 0.98. They enter here, before Layer-2 and screening, so a graft
    earns its exact score on exactly the same evidence as anything else and
    nothing downstream needs to know where it came from.
    """
    if not grafts.is_file():
        raise SystemExit(f"graft candidates file is missing: {grafts}")
    payload = json.loads(grafts.read_text())
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit(f"{grafts} holds no graft cases")

    with candidates.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "sequence" not in fieldnames:
        raise SystemExit(f"{candidates} has no sequence column")

    known = {str(row["sequence"]).strip().upper() for row in rows}
    teacher = str(payload.get("teacher_sequence") or "").strip().upper()
    added = 0
    for case in cases:
        sequence = str(case.get("sequence") or "").strip().upper()
        if not sequence or sequence in known:
            continue
        known.add(sequence)
        row = {name: "" for name in fieldnames}
        row["sequence"] = sequence
        # Provenance the screen already understands: the teacher is the parent
        # this hybrid was built from, and the lane says why it exists.
        if "parent_sequence" in row:
            row["parent_sequence"] = teacher
        if "proposal_type" in row:
            row["proposal_type"] = GRAFT_PROPOSAL_TYPE
        if "proposal_lane" in row:
            row["proposal_lane"] = GRAFT_PROPOSAL_TYPE
        if "edit_distance" in row:
            row["edit_distance"] = case.get("mutation_count_from_teacher", "")
        if "parent_max_jaccard_to_top50" in row:
            row["parent_max_jaccard_to_top50"] = case.get("max_jaccard_to_top50", "")
        rows.append(row)
        added += 1

    with candidates.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"donor grafts: merged {added} of {len(cases)} case(s) into "
        f"{candidates.name}; {len(rows)} proposals total",
        flush=True,
    )
    return added


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-metric-policy",
        choices=("plain",),
        default="plain",
        help="compatibility flag; screen and confirmation use validator arithmetic",
    )
    parser.add_argument("--consensus-ranking-policy", choices=("tier", "off"), default="tier")
    parser.add_argument("--minimum-selection-observations", type=int)
    parser.add_argument("--budget-mode", choices=("static", "adaptive"), default="static")
    parser.add_argument("--confirmation-time-share", type=float, default=.25,
                        help="fraction of the usable epoch reserved for repeat folds; CPU and commitments have separate guards")
    parser.add_argument("--budget-load-seconds", type=float, default=45.0)
    parser.add_argument("--budget-gpu-seconds", type=float, default=7.5)
    parser.add_argument("--budget-cpu-guard-seconds", type=float, default=120.0)
    parser.add_argument("--publication-prepare-blocks", type=int, default=0,
                        help="adaptive mode stops GPU work before the publisher's preparation window")
    parser.add_argument("--runtime-root", type=Path, default=Path("/root"))
    parser.add_argument(
        "--chain-state",
        type=Path,
        default=Path("/root/artifacts/chain_state.json"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("/root/artifacts/nanobody_published_sequences.txt"),
    )
    parser.add_argument(
        "--mutation-database",
        type=Path,
        default=Path("/root/artifacts/nanobody_mutations.sqlite"),
    )
    parser.add_argument(
        "--ranker-model",
        type=Path,
        default=Path("/root/artifacts/parent_relative_ranker/model.joblib"),
        help="model.joblib from train_parent_relative_ranker; when present, the "
             "competitor_lane single-edits are ordered by predicted improvement",
    )
    parser.add_argument(
        "--rival-coldkey",
        type=str,
        default="5GhREcsK",
        help="coldkey prefix of the top competitor; their recent submissions "
             "become the competitor_lane parent pool (proven-good source)",
    )
    parser.add_argument("--maximum-chain-state-age", type=float, default=90)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument("--poll-interval", type=float, default=30)
    parser.add_argument(
        "--prescreen-next-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="once this epoch's confirmation is done, run the next epoch's first "
             "generate-and-screen iterations in the otherwise idle tail",
    )
    parser.add_argument(
        "--prescreen-screen-tranches",
        type=int,
        default=2,
        help="screen tranches per prescreen iteration: one generation, several "
             "tranches from its pool, pipelined, to fill the idle tail",
    )
    parser.add_argument(
        "--prescreen-minimum-blocks",
        type=int,
        default=60,
        help="start another prescreen iteration only with at least this many "
             "blocks of the current epoch left",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated CUDA device ids; defaults to every device present",
    )
    parser.add_argument(
        "--exact-scorer",
        type=Path,
        default=Path("/root/nanobody_opt/score_boltzgen_batch.py"),
        help="validator-protocol-compatible exact scorer entry point",
    )
    parser.add_argument(
        "--parent-epochs",
        type=int,
        default=0,
        help="zero keeps every historical epoch in the mutation database",
    )
    parser.add_argument("--parents-per-epoch", type=int, default=2)
    parser.add_argument("--parent-limit", type=int, default=64)
    parser.add_argument("--parent-scan-limit", type=int, default=40)
    parser.add_argument("--neighbor-parents-per-epoch", type=int, default=8)
    parser.add_argument("--neighbor-parent-distance", type=int, default=6)
    parser.add_argument("--neighbor-parent-limit", type=int, default=32)
    parser.add_argument("--terminal-variants-per-residue", type=int, default=3)
    parser.add_argument("--recent-parent-epoch-window", type=int, default=16)
    parser.add_argument("--recent-anchor-parent-limit", type=int, default=16)
    parser.add_argument("--recent-neighbor-parent-limit", type=int, default=16)
    parser.add_argument("--basin-core-limit", type=int, default=5)
    parser.add_argument("--basin-parent-limit", type=int, default=32)
    parser.add_argument(
        "--priority-coldkeys",
        default="",
        help="forwarded to select_elite_parents: comma-separated coldkeys "
             "each guaranteed a parent slot for their best recent submission",
    )
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--neighbor-seed-count", type=int, default=32)
    parser.add_argument("--minimum-seeds", type=int, default=8)
    parser.add_argument(
        "--per-seed",
        type=int,
        default=12,
        help="proposals generated per elite parent before Layer-2 filtering",
    )
    parser.add_argument("--generation-workers", type=int,
                        default=min(32, os.cpu_count() or 1),
                        help="forwarded to local_neighborhood")
    parser.add_argument("--generation-workers-minimum-candidates", type=int,
                        default=64, help="forwarded to local_neighborhood")
    parser.add_argument(
        "--settled-deviation-policy",
        choices=("repair-only", "prefer", "off"),
        default="repair-only",
        help="forwarded to local_neighborhood; 'off' is the rollback lever",
    )
    parser.add_argument("--proposal-limit", type=int, default=384)
    parser.add_argument("--basin-fraction", type=float, default=0.70)
    parser.add_argument("--basin-routes-per-core", type=int, default=192)
    parser.add_argument("--basin-evidence-fraction", type=float, default=0.40)
    parser.add_argument("--basin-evidence-core-limit", type=int, default=1)
    parser.add_argument("--basin-top-core-weight", type=int, default=2)
    parser.add_argument(
        "--basin-exploration-residues-per-position", type=int, default=5
    )
    parser.add_argument("--framework-fraction", type=float, default=0.5)
    parser.add_argument("--framework-per-seed", type=int, default=64)
    parser.add_argument("--haplotype-lane-fraction", type=float, default=0.5,
                        help="forwarded to local_neighborhood; share of the framework_terminal/lineage budget reserved for haplotype candidates")
    parser.add_argument(
        "--donor-grafts",
        action="store_true",
        help=(
            "rebuild a donor-graft pool from each epoch's archive and add it "
            "to the proposals. Neighbourhood search alone produces one "
            "molecule many times: across epochs 25093-25095 every candidate "
            "sat within 0.93 identity of every other and the validator scored "
            "three of our seven submissions"
        ),
    )
    parser.add_argument("--graft-teacher-rank", type=int, default=1)
    # Measured on epoch 25098's archive: the default 200-donor, 60-case pool
    # yields 3 mutually separated molecules -- no better than the neighbourhood
    # pool it is meant to widen -- while 534 donors and 300 cases yield 27.
    # Framework diversity comes from the donors, not from the teachers: all ten
    # structural teachers sit in one family and only two of them are separated.
    parser.add_argument("--graft-limit", type=int, default=300)
    parser.add_argument("--graft-donors", type=int, default=600)
    parser.add_argument(
        "--graft-candidates",
        type=Path,
        help=(
            "generate_donor_grafts output whose sequences join each "
            "iteration's proposals. Neighbourhood search alone produces one "
            "molecule many times; this is the scaffold diversity it cannot "
            "reach on its own"
        ),
    )
    parser.add_argument("--terminal-width", type=int, default=17)
    parser.add_argument(
        "--maximum-terminal-haplotype-mutations", type=int, default=6
    )
    parser.add_argument("--motif-fraction", type=float, default=0.25)
    parser.add_argument("--single-mutation-fraction", type=float, default=0.75)
    parser.add_argument("--maximum-motif-mutations", type=int, default=6)
    parser.add_argument(
        "--maximum-exhaustive-singles",
        type=int,
        default=None,
        help="passed through; None lets the generator scale it to the proposal budget",
    )
    parser.add_argument(
        "--scout-iterations",
        type=int,
        default=1,
        help=(
            "generations that spread across the strongest historical cores "
            "before the epoch narrows onto what it has actually scored"
        ),
    )
    parser.add_argument(
        "--exploit-basin-fraction",
        type=float,
        default=0.95,
        help="basin lane share once the epoch switches from scouting",
    )
    parser.add_argument(
        "--exploit-basin-core-limit",
        type=int,
        default=5,
        help="cores the evidence lane concentrates on once exploiting",
    )
    parser.add_argument(
        "--public-neighbourhood-distance",
        type=int,
        default=0,
        help=(
            "keep only proposals within this Hamming distance of a public "
            "sequence; 0 disables the filter"
        ),
    )
    parser.add_argument("--layer2-concurrency", type=int, default=128)
    parser.add_argument(
        "--igblast-batch-size",
        type=int,
        default=8,
        help=(
            "sequences per IgBLAST call; measured 42 ms/seq at one and 21 ms/seq "
            "at eight, because each call reloads the germline databases"
        ),
    )
    parser.add_argument(
        "--layer2-igblast-only",
        type=int,
        default=0,
        help=(
            "screen the pool on IgBLAST alone and spend full TNP only on the "
            "replication shortlist; uncached TNP is about two seconds per "
            "sequence, so a breadth-first pool would cost more than an epoch"
        ),
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=320,
        help="maximum exact screens per iteration, published in small tranches",
    )
    parser.add_argument("--screen-tranche-size", type=int, default=20)
    parser.add_argument("--maximum-initial-screen-p90", type=float, default=90.0)
    parser.add_argument("--weak-initial-fallback-after", type=int, default=80)
    parser.add_argument(
        "--finalists",
        type=int,
        default=5,
        help="repeat capacity; five permits evidence from each leading basin",
    )
    parser.add_argument("--challenger-p90-margin", type=float, default=20.0)
    parser.add_argument("--basin-shrinkage-strength", type=float, default=2.0)
    parser.add_argument("--minimum-basin-samples", type=int, default=3)
    parser.add_argument(
        "--qualification-observations",
        type=int,
        default=8,
        help=(
            "independent observations per finalist; one observation of a fixed "
            "sequence is a wide draw, so this is the main accuracy control"
        ),
    )
    parser.add_argument(
        "--strict-gates",
        default="",
        help="re-derived strict qualification thresholds for the active scorer",
    )
    parser.add_argument(
        "--minimum-publishable-observations",
        type=int,
        default=5,
        help=(
            "forwarded to the race, which sizes the replication shortlist by "
            "solving for this minus one rounds; keep it in step with "
            "--replication-rounds or the shortlist is sized for rounds that "
            "never run"
        ),
    )
    parser.add_argument(
        "--replication-rounds",
        type=int,
        default=0,
        help=(
            "re-observe the best --finalists screened candidates this many "
            "times after screening instead of repeating one tranche leader"
        ),
    )
    parser.add_argument(
        "--replication-rounds-per-stage",
        type=int,
        default=0,
        help="rounds between prunes in the replication tournament",
    )
    parser.add_argument(
        "--replication-keep-fraction",
        type=float,
        default=1.0,
        help="fraction of candidates carried into each next stage",
    )
    parser.add_argument(
        "--replication-reserve-blocks",
        type=int,
        default=0,
        help="stop screening this many blocks early so replication still fits",
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=1,
        help="overlap tranche bookkeeping with the next tranche's GPU work",
    )
    parser.add_argument(
        "--screen-predict-batch-size",
        type=int,
        default=0,
        help=(
            "pin candidates per BoltzGen predict call so a larger tranche "
            "cannot change the grouping that scores depend on"
        ),
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=DEFAULT_MINIMUM_FREE_GIB,
        help="refuse new GPU work when the runtime filesystem falls below this reserve",
    )
    parser.add_argument(
        "--maximum-persisted-ranking-mib",
        type=float,
        default=DEFAULT_MAXIMUM_PERSISTED_RANKING_MIB,
        help="prune older regenerable screen/stage rank files above this size",
    )
    parser.add_argument(
        "--retained-epoch-directories",
        type=int,
        default=DEFAULT_RETAINED_EPOCHS,
        help=(
            "keep this many recent raw epoch trees; older trees are removed "
            "only after byte-matched sealed ingestion into the observation DB"
        ),
    )
    parser.add_argument("--once", action="store_true")
    return parser


#: Per-epoch inputs computed from the leaderboard before the epoch. A prescreen
#: builds them one epoch early, before the previous epoch is scored, so they
#: are rebuilt once the epoch really starts.
PRESCREEN_STALE_INPUTS = ("winner_priors.json", "competitor_parents.json",
                          "competitor_parents_layer2.json")


def run_prescreen(args: argparse.Namespace, state: dict[str, Any]) -> bool:
    """Run one generate-and-screen iteration for the NEXT epoch, early.

    After confirmation the GPUs sat idle for the rest of the epoch (~25-30
    min). Screening does not depend on the epoch: the target is fixed and raw
    metrics are only ranked against the field later. The iteration is written
    into the next epoch's own producer root, so when that epoch starts its
    ranking pools these folds like any other iteration's and the loop simply
    continues at the next iteration number.

    The race reads a synthetic chain state placing it at the start of the next
    epoch; the real one would report zero blocks left for an epoch that has
    not begun and the race would screen nothing.
    """
    epoch_length = int(state["epoch_length"])
    remaining = epoch_length - int(state["block"]) % epoch_length
    if remaining < args.prescreen_minimum_blocks:
        return False
    following = int(state["epoch"]) + 1
    producer_root = args.runtime_root / "live" / f"epoch{following}" / "producer"
    producer_root.mkdir(parents=True, exist_ok=True)
    synthetic_path = producer_root / "prescreen_chain_state.json"
    synthetic = {**state, "epoch": following, "block": following * epoch_length,
                 "updated_at_unix": time.time()}
    _atomic_json(synthetic_path, synthetic)
    retained = [int(path.name.split("_")[-1]) for path in producer_root.glob("iteration_[0-9]*")
                if path.is_dir() and path.name.split("_")[-1].isdigit()]
    iteration = max(retained, default=0) + 1
    marker = producer_root / "prescreen.json"
    record = json.loads(marker.read_text()) if marker.is_file() else {"iterations": []}
    record["iterations"].append(iteration)
    record["refreshed"] = False
    _atomic_json(marker, record)
    print(f"PRODUCER_PRESCREEN epoch={following} iteration={iteration} "
          f"current_epoch_blocks_left={remaining}", flush=True)
    early = argparse.Namespace(**{**vars(args), "chain_state": synthetic_path,
                                  "maximum_screen_tranches_override": args.prescreen_screen_tranches})
    try:
        _run_iteration(args=early, state=synthetic, iteration=iteration)
    except subprocess.CalledProcessError as error:
        _atomic_json(producer_root / f"iteration_{iteration:02d}" / "failed.json", {
            "epoch": following, "iteration": iteration, "prescreen": True,
            "returncode": error.returncode, "command": error.cmd,
        })
        print(f"PRODUCER_PRESCREEN_FAILED {error}", flush=True)
    return True


def refresh_prescreen_inputs(producer_root: Path) -> None:
    """Drop the leaderboard-derived inputs a prescreen built one epoch early."""
    marker = producer_root / "prescreen.json"
    if not marker.is_file():
        return
    record = json.loads(marker.read_text())
    if record.get("refreshed"):
        return
    for name in PRESCREEN_STALE_INPUTS:
        (producer_root / name).unlink(missing_ok=True)
    record["refreshed"] = True
    _atomic_json(marker, record)
    print(f"PRODUCER_PRESCREEN_REFRESHED {producer_root}", flush=True)


def main() -> None:
    args = _parser().parse_args()
    if not 0 < args.confirmation_time_share < 1:
        raise SystemExit("--confirmation-time-share must be between zero and one")
    if min(args.budget_load_seconds, args.budget_gpu_seconds, args.budget_cpu_guard_seconds) <= 0:
        raise SystemExit("budget timing bounds must be positive")
    if args.publication_prepare_blocks < 0:
        raise SystemExit("publication preparation guard must be nonnegative")
    if not args.gpus:
        # Resolved here rather than as an argparse default so --help stays
        # importable without torch, which the deploy gate depends on.
        args.gpus = visible_gpu_ids()
    completed_iterations: dict[int, int] = {}
    #: Epochs whose neighbourhood is used up; generation stops, replication continues.
    exhausted_epochs: set[int] = set()
    finished_epochs: set[int] = set()
    while True:
        try:
            state = _chain_state(args.chain_state, args.maximum_chain_state_age)
            epoch = int(state["epoch"])
            maintain_runtime_storage(
                args.runtime_root,
                epoch,
                minimum_free_gib=args.minimum_free_gib,
                maximum_persisted_ranking_mib=args.maximum_persisted_ranking_mib,
                observation_database=args.mutation_database,
                retained_epochs=args.retained_epoch_directories,
            )
            offset = int(state["block"]) % int(state["epoch_length"])
            if epoch in finished_epochs or offset >= int(state["epoch_length"]) - _handoff_reserve(args):
                if not (args.prescreen_next_epoch and run_prescreen(args, state)):
                    label = ("PRODUCER_CONFIRMATION_FINISHED" if epoch in finished_epochs
                             else "PRODUCER_RESERVE")
                    print(f"{label} epoch={epoch} offset={offset}; waiting", flush=True)
            else:
                producer_root = args.runtime_root / "live" / f"epoch{epoch}" / "producer"
                refresh_prescreen_inputs(producer_root)
                if epoch in exhausted_epochs and not (
                    args.budget_mode == "adaptive" and args.replication_rounds
                    and _closing(args, state, producer_root)
                ):
                    # The neighbourhood is used up for this epoch. Generating
                    # again would exclude everything already proposed and come
                    # back empty, which is how this span was spent at 0% GPU
                    # before: 43 iterations into a basin with 13,504 candidates
                    # already excluded. Wait for the reserve and let replication
                    # have the rest of the epoch -- eight observations take
                    # reliability from 0.16 to 0.61, and a forty-fourth pass
                    # takes it nowhere.
                    print(
                        f"PRODUCER_EXHAUSTED epoch={epoch} offset={offset}; "
                        "generation complete, leaving the epoch to replication",
                        flush=True,
                    )
                    time.sleep(args.poll_interval)
                    continue
                retained_iterations = [int(path.name.split("_")[-1])
                    for path in producer_root.glob("iteration_[0-9]*")
                    if path.is_dir() and path.name.split("_")[-1].isdigit()]
                iteration = max([completed_iterations.get(epoch, 0), *retained_iterations]) + 1
                print(
                    f"PRODUCER_ITERATION epoch={epoch} offset={offset} "
                    f"iteration={iteration}",
                    flush=True,
                )
                try:
                    _run_iteration(args=args, state=state, iteration=iteration)
                except subprocess.CalledProcessError as error:
                    failure = (
                        args.runtime_root / "live" / f"epoch{epoch}" / "producer"
                        / f"iteration_{iteration:02d}" / "failed.json"
                    )
                    _atomic_json(failure, {
                        "epoch": epoch,
                        "iteration": iteration,
                        "returncode": error.returncode,
                        "command": error.cmd,
                    })
                    completed_iterations[epoch] = iteration
                    print(f"PRODUCER_ITERATION_FAILED {error}", flush=True)
                    if _looks_exhausted(failure.parent):
                        exhausted_epochs.add(epoch)
                else:
                    completed_iterations[epoch] = iteration
                    root = producer_root / f"iteration_{iteration:02d}"
                    complete = root / "race/complete.json"
                    if args.budget_mode == "adaptive" and (root / "replication_only.json").exists() and complete.exists():
                        done = json.loads(complete.read_text())
                        if not done.get("screened") and not done.get("replication_rounds_completed"):
                            finished_epochs.add(epoch)
            if args.once:
                return
        except Exception as error:
            print(
                f"PRODUCER_RETRY {type(error).__name__}: {error}",
                flush=True,
            )
            if args.once:
                raise
        time.sleep(max(1.0, args.poll_interval))


if __name__ == "__main__":
    main()
