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
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml


CONFIG_URL = "https://raw.githubusercontent.com/metanova-labs/nova/main/config/config.yaml"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/Metanova/Submission-Archive/resolve/main/"
    "{target}_nanobodies.csv"
)


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
    elite_parents = producer_root / "elite_parents.json"
    if not elite_parents.is_file():
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
            "--scan-limit", str(args.parent_scan_limit),
            "--observation-database", str(args.mutation_database),
            "--output", str(elite_parents),
        ])
    seed = producer_root / "elite_parents_layer2.json"
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
        command = [
            sys.executable,
            "-m", "nanobody_opt.local_neighborhood",
            "--archive", str(archive),
            "--config", str(config),
            "--layer2-seeds", str(seed),
            "--winner-priors", str(winner_priors),
            "--output", str(candidates),
            "--seed-count", str(args.seed_count),
            "--neighbor-seed-count", str(args.neighbor_seed_count),
            "--minimum-seeds", str(args.minimum_seeds),
            "--per-seed", str(args.per_seed),
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
            "--random-seed", str(epoch * 100 + iteration),
        ]
        for previous in sorted(producer_root.glob("iteration_*/candidates.csv")):
            if previous != candidates:
                command.extend(["--exclude", str(previous)])
        _run(command)
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
    closing = remaining_blocks <= args.replication_reserve_blocks + args.reserve_blocks
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
        "--reserve-blocks", str(args.reserve_blocks),
        "--gpus", args.gpus,
        "--scorer", str(args.exact_scorer),
        "--candidate-limit", str(args.candidate_limit),
        "--screen-tranche-size", str(args.screen_tranche_size),
        "--maximum-initial-screen-p90", str(args.maximum_initial_screen_p90),
        "--weak-initial-fallback-after", str(args.weak_initial_fallback_after),
        "--finalists", str(args.finalists),
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
        "--replication-reserve-blocks", str(args.replication_reserve_blocks),
        "--replication-rounds-per-stage", str(args.replication_rounds_per_stage),
        "--replication-keep-fraction", str(args.replication_keep_fraction),
        "--strict-gates", str(args.strict_gates or ""),
    ]
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
    parser.add_argument("--maximum-chain-state-age", type=float, default=90)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument("--poll-interval", type=float, default=30)
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
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--neighbor-seed-count", type=int, default=32)
    parser.add_argument("--minimum-seeds", type=int, default=8)
    parser.add_argument(
        "--per-seed",
        type=int,
        default=12,
        help="proposals generated per elite parent before Layer-2 filtering",
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
    parser.add_argument("--layer2-concurrency", type=int, default=40)
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
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.gpus:
        # Resolved here rather than as an argparse default so --help stays
        # importable without torch, which the deploy gate depends on.
        args.gpus = visible_gpu_ids()
    completed_iterations: dict[int, int] = {}
    #: Epochs whose neighbourhood is used up; generation stops, replication continues.
    exhausted_epochs: set[int] = set()
    while True:
        try:
            state = _chain_state(args.chain_state, args.maximum_chain_state_age)
            epoch = int(state["epoch"])
            offset = int(state["block"]) % int(state["epoch_length"])
            if offset >= int(state["epoch_length"]) - args.reserve_blocks:
                print(
                    f"PRODUCER_RESERVE epoch={epoch} offset={offset}; waiting",
                    flush=True,
                )
            else:
                if epoch in exhausted_epochs:
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
                    time.sleep(args.poll_seconds)
                    continue
                iteration = completed_iterations.get(epoch, 0) + 1
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
