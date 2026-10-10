"""Race Layer-2 nanobodies in publishable, validator-shaped screen tranches.

Each completed screen tranche is immediately visible to the independent
publisher. The first leader is repeated concurrently across isolated GPU
processes, then screening continues until the full candidate budget is used.
Later tranche leaders can replace the incumbent when their evidence is better.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import csv
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from nanobody_opt.mutation_network import (
    ranking_observations,
    update_observation_store,
)
from nanobody_opt import publication_gates
from nanobody_opt.diffusion_sample_selection import SCORER_PROTOCOL
from nanobody_opt.metric_outliers import (
    field_distribution,
    without_catastrophic_outliers,
)


# The gates live in nanobody_opt.publication_gates, which the live publisher
# imports as well.  Two hardcoded copies in two repositories is what let a
# relaxation be deployed here while the program that actually publishes kept
# refusing, so this module keeps no copy of its own.
STRICT_RANK_GATES = publication_gates.gate_defaults()

try:  # package import when run as -m nanobody_opt.sequential_exact_race
    from nanobody_opt import round_parallelism as _round_parallelism
    from nanobody_opt import scorer_daemon as _scorer_daemon
except ImportError:  # direct script execution
    import round_parallelism as _round_parallelism
    import scorer_daemon as _scorer_daemon


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


def strict_rank_arguments(overrides: dict[str, Any] | None = None) -> list[str]:
    """Flatten the publication gates, applying calibrated overrides by name."""
    return publication_gates.gate_arguments(overrides)


def load_strict_gate_overrides(path: Path | None) -> dict[str, Any]:
    """Read re-derived gate thresholds, refusing anything but a flat object."""
    return publication_gates.load_overrides(path)


DEFAULT_SCORER = Path("/root/nanobody_opt/score_boltzgen_batch.py")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _candidate_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    items = payload if isinstance(payload, list) else (
        payload.get("candidates") or payload.get("sequences") or []
    )
    return _ordered_candidate_records(items)


def _ordered_candidate_records(items: list) -> list[dict[str, Any]]:
    if items and all(isinstance(item, dict) for item in items):
        items = sorted(items, key=lambda item: (
            0 if item.get("screen_priority") is not None else 1,
            int(item.get("screen_priority", 0)),
            0
            if item.get("proposal_type") == "exhaustive_single_substitution"
            else 1
        ))
    records = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict) and item.get("layer2_passed") is False:
            continue
        sequence = item.get("sequence") if isinstance(item, dict) else item
        if isinstance(sequence, str) and sequence.strip():
            normalized = sequence.strip().upper()
            if normalized in seen:
                continue
            record = dict(item) if isinstance(item, dict) else {}
            record["sequence"] = normalized
            records.append(record)
            seen.add(normalized)
    return records


def _carry_pending_records(local: list[dict], pooled_root: Path | None,
                          output_dir: Path, target: str) -> list[dict]:
    """Keep unscored proposals; only a complete exact vector consumes a case."""
    if pooled_root is None:
        return local
    from nanobody_opt.rank_exact import _observations
    records = list(local)
    for path in sorted(pooled_root.glob("iteration_[0-9][0-9]/layer2.json")):
        records.extend(_candidate_records(path))
    scored = set(_observations(_pooled_exact_paths([], pooled_root, output_dir), target))
    return [r for r in _ordered_candidate_records(records) if r["sequence"] not in scored]


def _candidate_sequences(path: Path, limit: int) -> list[str]:
    return [
        str(record["sequence"])
        for record in _candidate_records(path)[:limit]
    ]


def _batches(items: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [items[start : start + size] for start in range(0, len(items), size)]


def _tranche_source_key(metadata: dict[str, dict[str, Any]], sequence: str) -> str:
    """Which generation lane proposed this candidate (local_neighborhood's
    ``parent_source``: ``exact_shortlist``/``recent_archive`` from
    ``elite_parents.json``, or ``historical_basin_member`` from its own
    internal basin exploration -- see local_neighborhood.py)."""
    return str(metadata.get(sequence, {}).get("parent_source") or "__unattributed__")


def _balanced_screen_tranche(
    pending: list[str],
    metadata: dict[str, dict[str, Any]],
    screened_parent_counts: Counter[str],
    size: int,
    *,
    screened_source_counts: "Counter[str] | None" = None,
) -> list[str]:
    """Select the next tranche without oversampling one structural basin.

    The proposal generator normally interleaves parents already.  This second
    guard rebuilds the queue after every completed tranche, so a Layer-2 pass
    imbalance or an older grouped proposal file cannot spend most of the exact
    budget on one family.  It deliberately does not use neighbour total scores:
    the exact replay shows that binding/interaction outcomes are not smooth
    enough for graph-based arm elimination.

    Balancing by individual parent alone is not balancing by LANE: a lane
    with many distinct parents (e.g. elite_parents.json's exact_shortlist,
    often ~400 distinct basin cores) dilutes across them while a lane with
    few parents concentrates its candidates on each one, so equal per-parent
    treatment still starves the many-parent lane of tranche slots overall --
    measured live at 23-32/epoch for exact_shortlist against 270-800 for the
    other lane despite exact_shortlist's children having the BEST median
    screened score of any lane (ARCHITECTURE_REDESIGN new finding 6). The
    source key below is checked FIRST, so no lane can be starved regardless
    of how many distinct parents it has; the existing per-parent key still
    breaks ties within a lane exactly as before.
    """
    if size < 1:
        raise ValueError("tranche size must be positive")
    if not pending:
        return []
    def allocation_key(sequence: str) -> str:
        record = metadata.get(sequence, {})
        return str(
            record.get("parent_basin_core_id")
            or record.get("basin_core_id")
            or record.get("parent_sequence")
            or "__unattributed__"
        )

    if not any(allocation_key(sequence) != "__unattributed__" for sequence in pending):
        return pending[:size]

    available = list(enumerate(pending))
    counts = Counter(screened_parent_counts)
    source_counts = Counter(screened_source_counts or {})
    selected: list[str] = []
    while available and len(selected) < size:
        choice = min(
            range(len(available)),
            key=lambda index: (
                source_counts[_tranche_source_key(metadata, available[index][1])],
                counts[allocation_key(available[index][1])],
                int(
                    metadata.get(available[index][1], {}).get(
                        "parent_basin_rank", 10**9
                    )
                ),
                available[index][0],
            ),
        )
        _, sequence = available.pop(choice)
        parent = allocation_key(sequence)
        source = _tranche_source_key(metadata, sequence)
        selected.append(sequence)
        counts[parent] += 1
        source_counts[source] += 1
    return selected


def _resumable_screen_tranche(
    pending: list[str],
    metadata: dict[str, dict[str, Any]],
    screened_parent_counts: Counter[str],
    size: int,
    *,
    source: Path,
    completed: Path,
    screened_source_counts: "Counter[str] | None" = None,
) -> list[str]:
    """Preserve the actual candidate list of an already completed tranche."""
    if not completed.is_file():
        return _balanced_screen_tranche(
            pending, metadata, screened_parent_counts, size,
            screened_source_counts=screened_source_counts,
        )
    if not source.is_file():
        raise ValueError(f"completed tranche lacks its candidate source: {source}")
    sequences = _candidate_sequences(source, size)
    if not sequences or any(sequence not in pending for sequence in sequences):
        raise ValueError(f"completed tranche source is inconsistent: {source}")
    return sequences


def _initial_repeat_candidate(
    ranked_sequences: list[str],
    candidates: dict[str, dict[str, Any]],
    *,
    repeated: set[str],
    maximum_screen_p90: float,
    allow_weak_fallback: bool,
) -> tuple[str, bool] | None:
    """Choose a strong one-shot candidate, or one bounded fallback candidate.

    A strict finalist may have at most a 90 p90 rank sum.  Repeating every first
    tranche leader above that ceiling consumed most of the early epoch without
    producing a qualified checkpoint.  Strong candidates remain immediately
    eligible; after enough screening, callers may authorize exactly one weak
    fallback to preserve a chance of first publication.
    """
    unseen = [sequence for sequence in ranked_sequences if sequence not in repeated]
    for sequence in unseen:
        score = float(
            candidates[sequence].get(
                "basin_adjusted_score", candidates[sequence]["score_p90"]
            )
        )
        if score <= maximum_screen_p90:
            return sequence, False
    if allow_weak_fallback and unseen:
        return unseen[0], True
    return None


def _screen_execution_shape(
    candidate_count: int,
    gpu_count: int,
    predict_batch_size: int | None = None,
) -> tuple[int, int]:
    """Keep one publishable tranche in one model-loading wave.

    The sharded scorer assigns candidates round-robin to GPUs.  Giving the
    worker its entire shard in one wrapper call lets BoltzGen load each model
    once for all candidates on that GPU.  The old ``wave_size=gpu_count`` and
    ``predict_batch_size=1`` combination reloaded the complete pipeline once
    per candidate even though publication only happened after the full tranche.

    Predict batch composition is load-bearing: ``score_boltzgen_batch`` scores a
    whole chunk in one ``predict`` call, and the wrapper's own measurements show
    the same molecule moving by more than a factor of two depending on what else
    shared that call.  The derived ``ceil(candidates / gpus)`` shape therefore
    silently changes scores when the tranche grows, which is exactly what a
    faster scorer invites.  Pin ``predict_batch_size`` whenever the tranche size
    changes so a throughput change cannot alter the measured metric.
    """
    if candidate_count < 1 or gpu_count < 1:
        raise ValueError("candidate and GPU counts must be positive")
    if predict_batch_size is not None:
        if predict_batch_size < 1:
            raise ValueError("predict batch size must be positive")
        return candidate_count, predict_batch_size
    return candidate_count, math.ceil(candidate_count / gpu_count)


def _remaining_blocks(path: Path | None, epoch: int) -> float:
    """Blocks left in ``epoch``, or infinity when the chain state is unusable."""
    if path is None:
        return float("inf")
    try:
        state = json.loads(path.read_text())
        if int(state["epoch"]) != epoch:
            return 0.0
        epoch_length = int(state["epoch_length"])
        return float(epoch_length - (int(state["block"]) % epoch_length))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return float("inf")


def _chain_allows_work(path: Path | None, epoch: int, reserve_blocks: int) -> bool:
    if path is None:
        return True
    try:
        state = json.loads(path.read_text())
        current_epoch = int(state["epoch"])
        block = int(state["block"])
        epoch_length = int(state["epoch_length"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return current_epoch == epoch and block % epoch_length < epoch_length - reserve_blocks


def _pooled_candidate_metadata(
    local: dict[str, dict[str, Any]], pooled_root: Path | None
) -> dict[str, dict[str, Any]]:
    """Proposal metadata for every generation this epoch has screened.

    Basin shrinkage keys on a candidate's parent core and proposal lane, which
    live in the Layer-2 file of the generation that proposed it.  Loading only
    this generation's file makes shrinkage a silent no-op on pooled candidates,
    so the closing shortlist fills with un-shrunk lucky outliers from earlier
    generations -- the opposite of the intent.
    """
    merged: dict[str, dict[str, Any]] = {}
    if pooled_root is not None:
        for path in sorted(pooled_root.glob("iteration_*/layer2.json")):
            try:
                for record in _candidate_records(path):
                    merged.setdefault(str(record["sequence"]), record)
            except (OSError, ValueError, KeyError):
                continue
    merged.update(local)
    return merged


def _pooled_exact_paths(
    local: list[Path], pooled_root: Path | None, output_dir: Path
) -> list[Path]:
    """Every exact observation this epoch has produced, not just this iteration.

    A generate-score-regenerate epoch screens several generations.  Ranking and
    replication must see all of them: the generator deliberately does not
    re-propose candidates it has already screened, so a strong hit from the
    first generation can never be confirmed later unless the closing iteration
    can still reach its observations.
    """
    paths = list(local)
    if pooled_root is None:
        return paths
    seen = {path.resolve() for path in paths}
    own = output_dir.resolve()
    for path in sorted(pooled_root.glob("iteration_*/race/**/exact_all.json")):
        resolved = path.resolve()
        if resolved in seen or own in resolved.parents:
            continue
        seen.add(resolved)
        paths.append(path)
    return paths


def _layer2_gate(
    shortlist: list[str],
    *,
    command: list[str],
    root: Path,
    target: str,
) -> list[str]:
    """Run full Layer-2 on the shortlist and keep only what actually passes.

    Full TNP costs about two seconds per uncached sequence and is GPU-resident,
    so running it over a breadth-first proposal pool would cost more than an
    epoch: 3,200 novel candidates is roughly 107 minutes.  The pool is therefore
    screened on IgBLAST alone, and full TNP is spent only on the candidates that
    could actually be published.  A sequence that fails here is removed from
    contention rather than published on IgBLAST evidence.

    ``command`` is a template containing ``{input}`` and ``{output}``; the caller
    owns how Layer 2 is invoked, so this stays free of environment specifics.
    """
    if not shortlist or not command:
        return shortlist
    gate_root = root / "layer2_gate"
    source = gate_root / "shortlist.json"
    output = gate_root / "layer2.json"
    _atomic_json(source, {"target": target, "sequences": shortlist})
    resolved = [
        item.format(input=str(source), output=str(output)) for item in command
    ]
    try:
        subprocess.run(resolved, check=True)
        payload = json.loads(output.read_text())
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError) as error:
        # Failing open would publish on IgBLAST evidence alone, which the
        # publisher is right to refuse; failing closed costs this epoch's
        # replication instead.
        print(f"EXACT_LAYER2_GATE_FAILED error={type(error).__name__}", flush=True)
        return []
    passed = {
        str(record["sequence"])
        for record in payload.get("candidates", [])
        if isinstance(record, dict)
        and record.get("sequence")
        and record.get("layer2_passed") is True
    }
    kept = [sequence for sequence in shortlist if sequence in passed]
    print(
        f"EXACT_LAYER2_GATE kept={len(kept)}/{len(shortlist)}",
        flush=True,
    )
    return kept


def _replication_shortlist(
    ranking: Path, *, exclude: set[str], limit: int
) -> list[str]:
    """Best screened candidates to re-observe, screen order preserved."""
    if limit < 1 or not ranking.is_file():
        return []
    payload = json.loads(ranking.read_text())
    sequences = [
        str(sequence)
        for sequence in payload.get("sequences", [])
        if str(sequence) not in exclude
    ]
    return sequences[:limit]


def _drop_metric_outliers(
    shortlist: list[str],
    *,
    ranking: Path,
    archive: Path,
    maximum_percentile: float,
    recent: int = 300,
) -> list[str]:
    """Remove candidates whose weakest metric sits near the bottom of the field.

    The rank sum already charges for a weak metric; what it does not express is
    that one catastrophic metric is disqualifying when only rank one pays.  Epoch
    25061 published a candidate ranked 47th of 48 on min_design_to_target_pae
    with two hydrogen bonds against a field median of four.
    """
    if maximum_percentile >= 1.0 or not shortlist or not ranking.is_file():
        return shortlist
    try:
        payload = json.loads(ranking.read_text())
        by_sequence = {
            str(item["sequence"]): item.get("metrics") or {}
            for item in payload.get("candidates", [])
            if isinstance(item, dict) and item.get("sequence")
        }
        with archive.open() as handle:
            rows = list(csv.DictReader(handle))[-recent:]
    except (OSError, ValueError, KeyError) as error:
        print(
            f"EXACT_OUTLIER_GUARD_SKIPPED {type(error).__name__}",
            flush=True,
        )
        return shortlist
    candidates = [
        (sequence, by_sequence[sequence])
        for sequence in shortlist
        if by_sequence.get(sequence)
    ]
    if not candidates:
        return shortlist
    kept, dropped = without_catastrophic_outliers(
        candidates,
        field_distribution(rows),
        maximum_percentile=maximum_percentile,
        minimum_keep=max(len(shortlist) // 4, 1),
    )
    unscored = [item for item in shortlist if not by_sequence.get(item)]
    for sequence, fraction, culprit in dropped[:5]:
        print(
            f"  dropped {sequence[:12]} worst_metric={culprit} "
            f"field_fraction_better={fraction:.2f}",
            flush=True,
        )
    print(
        f"EXACT_OUTLIER_GUARD kept={len(kept)}/{len(candidates)} "
        f"threshold={maximum_percentile}",
        flush=True,
    )
    order = {sequence: index for index, sequence in enumerate(shortlist)}
    return sorted(kept + unscored, key=lambda item: order.get(item, 10**9))


def _affordable_tranche_size(
    size: int,
    *,
    remaining_blocks: float,
    load_seconds: float,
    gpu_seconds_per_candidate: float,
    gpu_count: int,
    block_seconds: float = 12.0,
) -> int:
    """Largest screen tranche that still FINISHES inside the window.

    The reserve check answers "is there room to start", which is not the
    question.  A tranche runs for as long as it runs: epoch 25131 dispatched 700
    candidates at block 154 of 361, folded for 1,899 seconds -- 158 blocks --
    finished at block 312 and left 49, under the 55-block commitment reserve.
    The replication shortlist was then sized against a window of zero, narrowed
    to a single candidate, and none of the twelve configured rounds ran.  That
    is the whole reason the score bank holds no within-epoch replicates.
    """
    if size < 1 or remaining_blocks <= 0 or gpu_seconds_per_candidate <= 0:
        return 0
    budget = remaining_blocks * block_seconds
    if load_seconds >= budget:
        return 0
    width = int(
        (budget - load_seconds) * max(gpu_count, 1) / gpu_seconds_per_candidate
    )
    return max(0, min(size, width))


def _affordable_shortlist(
    shortlist: list[str],
    *,
    rounds: int,
    minimum_observations: int,
    remaining_blocks: float,
    load_seconds: float,
    seconds_per_observation: float,
    gpu_count: int,
    block_seconds: float = 12.0,
) -> list[str]:
    """Trade shortlist width for depth when the closing window is short.

    A round costs one model load plus the shortlist's folding, so a narrower
    shortlist buys more rounds in the same time.  Publishing nothing is the
    worst outcome available -- strict qualification needs a fixed number of
    independent observations, and an epoch that cannot reach it scores nothing
    at all -- so when the window is tight the shortlist shrinks rather than the
    observation count.
    """
    if not shortlist or rounds < 1:
        return shortlist
    if remaining_blocks <= 0:
        # No window left at all (the epoch rolled over mid-screen).  Narrow to
        # one candidate rather than failing open to the widest shortlist, which
        # is the inverse of this function's purpose.
        print("EXACT_SHORTLIST_NARROWED remaining_blocks=0 to=1", flush=True)
        return shortlist[:1]
    required_rounds = max(minimum_observations - 1, 1)
    budget = remaining_blocks * block_seconds

    def affordable(width: int) -> bool:
        per_round = load_seconds + seconds_per_observation * width / gpu_count
        return per_round * required_rounds <= budget

    if affordable(len(shortlist)):
        return shortlist
    width = len(shortlist)
    while width > 1 and not affordable(width):
        width -= 1
    print(
        f"EXACT_SHORTLIST_NARROWED from={len(shortlist)} to={width} "
        f"remaining_blocks={remaining_blocks:.0f} "
        f"required_rounds={required_rounds}",
        flush=True,
    )
    return shortlist[:width]


#: Two submissions closer than this are one submission: the validator dedupes
#: them and only one is scored.
REPLICATION_FAMILY_IDENTITY = 0.95


def _sequence_identity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    shared = sum(1 for a, b in zip(left, right) if a == b)
    return shared / max(len(left), len(right))


def _keep_a_family_slate(ranked: list[str], keep: int) -> list[str]:
    """Keep the best of each family first, then fill by rank.

    Pruning purely by score spends every observation on one family: the leader
    reached twelve to thirteen observations while the separated molecules the
    fleet actually commits sat at two to six. That penalises them twice, once
    because they are less optimised and once because a low-observation estimate
    is shrunk toward the cohort mean, so they look worse than they are.

    A fleet of eight commits eight distinct molecules. Observations spent
    beyond what makes an estimate trustworthy are better spent covering the
    molecules that will actually be submitted.
    """
    if keep >= len(ranked):
        return list(ranked)
    slate: list[str] = []
    for sequence in ranked:
        if len(slate) >= keep:
            break
        if all(
            _sequence_identity(sequence, chosen) < REPLICATION_FAMILY_IDENTITY
            for chosen in slate
        ):
            slate.append(sequence)
    if len(slate) < keep:
        taken = set(slate)
        slate.extend(item for item in ranked if item not in taken)
    return slate[:keep]


def _run_replication_rounds(
    *,
    shortlist: list[str],
    rounds: int,
    root: Path,
    config: Path,
    target: str,
    gpus: list[str],
    scorer: Path,
    chain_state: Path | None,
    epoch: int,
    reserve_blocks: int,
    predict_batch_size: int | None,
    timings: Path,
    rounds_per_stage: int = 0,
    keep_fraction: float = 1.0,
    rerank: Any = None,
    finish_model: Any = None,
    finish_guard_seconds: float = 0.0,
) -> int:
    """Re-observe one shortlist as whole rounds of independent observations.

    A single observation of a fixed sequence is a wide draw, so a shortlist
    chosen on one observation each is largely chosen by luck.  Giving every
    shortlist member the same number of independent observations lets selection
    compare pooled means instead.  One round is one scorer invocation over the
    whole shortlist, which is also what amortises the scorer's model load.
    """
    if not shortlist or rounds < 1:
        return 0
    # Successive elimination.  A flat allocation spends the same depth on
    # candidates that were hopeless after four observations as on the eventual
    # pick; re-ranking between stages and dropping the trailing fraction reached
    # the same outcome on 43% of the observations in simulation.  The budget
    # freed pays for a far wider tournament, which is what actually raises the
    # chance of holding a winning candidate at all.
    alive = list(shortlist)
    completed = 0
    stage = 0
    while completed < rounds and alive:
        stage += 1
        source = root / "rounds" / f"stage_{stage:02d}.json"
        if not source.is_file():
            _atomic_json(source, {"target": target, "sequences": alive})
        else:
            # Resume reuses the set this stage actually observed.
            alive = [
                str(item)
                for item in json.loads(source.read_text()).get("sequences", [])
            ] or alive
        stage_rounds = min(rounds_per_stage or rounds, rounds - completed)
        # A round costs a model load whether it scores sixty candidates or
        # two, and a shortlist smaller than the shard count leaves shards
        # idle. When the shortlist cannot fill the box, run several rounds at
        # once on disjoint GPU groups: rounds are independent, and each keeps
        # its own invocation and predict composition, so the metric is
        # untouched. See round_parallelism for the measurements.
        groups = _round_parallelism.plan(
            alive=len(alive), gpus=gpus, rounds_left=stage_rounds)
        if len(groups) > 1:
            done = _run_rounds_concurrently(
                groups=groups, source=source, config=config, target=target,
                root=root, first_index=completed + 1, alive=alive,
                scorer=scorer, predict_batch_size=predict_batch_size,
                timings=timings, stage=stage, all_gpus=gpus,
                chain_state=chain_state, epoch=epoch,
                reserve_blocks=reserve_blocks)
            completed += done
            if done:
                continue
        for _ in range(stage_rounds):
            index = completed + 1
            already = root / "rounds" / f"round_{index:02d}" / "exact_all.json"
            if already.is_file():
                # Resume must count work already on disk before consulting the
                # clock: turning completed observations into a qualified ranking
                # is pure CPU work, and discarding them publishes nothing.
                completed += 1
                continue
            if not _chain_allows_work(chain_state, epoch, reserve_blocks):
                print(
                    f"EXACT_REPLICATION_RESERVE_REACHED rounds={completed}/{rounds}",
                    flush=True,
                )
                return completed
            if finish_model is not None:
                usable = max(_remaining_blocks(chain_state, epoch) - reserve_blocks, 0) * 12.0
                if finish_model.round_seconds(len(alive), len(gpus)) + finish_guard_seconds > usable:
                    print(f"EXACT_REPLICATION_FINISH_GUARD rounds={completed}/{rounds}", flush=True)
                    return completed
            started = time.monotonic()
            _run_screen(
                source=source,
                config=config,
                target=target,
                output_dir=root / "rounds" / f"round_{index:02d}",
                gpus=gpus,
                candidate_limit=len(alive),
                scorer=scorer,
                predict_batch_size=predict_batch_size,
            )
            completed += 1
            seconds = time.monotonic() - started
            _append_timing(
                timings,
                {
                    "phase": "replication_round",
                    "round": index,
                    "stage": stage,
                    "shortlist": len(alive),
                    "candidates": len(alive),
                    "gpu_count": len(gpus),
                    "seconds": round(seconds, 1),
                },
            )
            print(
                f"EXACT_REPLICATION_ROUND_COMPLETE round={index}/{rounds} "
                f"stage={stage} alive={len(alive)} seconds={seconds:.0f}",
                flush=True,
            )
        if keep_fraction < 1.0 and len(alive) > 1 and rerank is not None:
            keep = max(int(len(alive) * keep_fraction), 1)
            pruned = rerank(alive, keep)
            if pruned:
                print(
                    f"EXACT_REPLICATION_PRUNED stage={stage} "
                    f"from={len(alive)} to={len(pruned)}",
                    flush=True,
                )
                alive = pruned
    return completed


def _append_timing(path: Path, record: dict[str, Any]) -> None:
    """Append one phase-timing record so the epoch budget comes from measurement.

    A faster scorer only moves the bottleneck; without per-phase wall time there
    is no way to tell whether an epoch is limited by folding, by ranking, or by
    replication.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _exact_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/exact_all.json"))


def _rank_command(
    *,
    exact: list[Path],
    archive: Path,
    config: Path,
    target: str,
    epoch: int,
    output: Path,
    ledger: Path | None,
    strict: bool,
    strict_gate_overrides: dict[str, Any] | None = None,
    selection_metric_policy: str = "plain",
    consensus_ranking_policy: str = "tier",
    minimum_selection_observations: int | None = None,
    observation_history: Path | None = None,
    only_sequences: Path | None = None,
) -> list[str]:
    command = [sys.executable, "-m", "nanobody_opt.rank_exact",
               "--selection-metric-policy", selection_metric_policy,
               "--consensus-ranking-policy", consensus_ranking_policy]
    # NOVA_FULL_ARCHIVE_FIELD switches BOTH the ranking field and the strict
    # qualification gates together, atomically. The two must never be enabled
    # independently: the small-field gates (60/400/300/100/30) reject almost
    # nothing on the archive's ~50k-260k scale, so a field change without the
    # matching gate change would let everything through (ARCHITECTURE_REDESIGN
    # new finding 7).
    if os.environ.get("NOVA_FULL_ARCHIVE_FIELD") == "1":
        command.append("--full-archive-field")
        if strict and not strict_gate_overrides:
            default_overrides_path = (
                Path(__file__).resolve().parent / "strict_gates_archive_field.json"
            )
            strict_gate_overrides = load_strict_gate_overrides(default_overrides_path)
    # Earlier-epoch folds are still valid evidence: the challenge is identical
    # across epochs, and the database holds ~40 observations per sequence
    # where this epoch produced one. Out of sample over 22 epochs this lifts
    # agreement with the validator from +0.165 to +0.365.
    if observation_history is not None:
        command.extend(["--observation-history", str(observation_history)])
    for path in exact:
        command.extend(["--exact", str(path)])
    command.extend([
        "--archive", str(archive),
        "--target", target,
        "--output", str(output),
        "--recent", "300",
        "--field-size", "49",
        # Provisional basin shrinkage needs the complete screened cohort, not
        # only the raw top 30. Strict publication files remain compact.
        "--limit", "30" if strict else "10000",
    ])
    if ledger is not None:
        command.extend(["--exclude-file", str(ledger)])
    if strict:
        command.extend([
            "--pose-config", str(config),
            "--before-epoch", str(epoch),
            "--api-field-count", "10",
            "--minimum-api-fields", "3",
        ])
        command.extend(strict_rank_arguments(strict_gate_overrides))
        # Strict qualification decides what actually gets published, so it is the
        # one ordering that must target a winning draw rather than a good average.
        command.extend(["--selection-order", "win-probability"])
    else:
        command.extend(["--selection-order", "p90"])
        if minimum_selection_observations is not None:
            command.extend(["--minimum-observations", str(minimum_selection_observations)])
    if only_sequences is not None:
        # A prune reorders the alive finalists only. They already passed the
        # pool-level construction constraints, which would otherwise re-cut
        # them against a pool of tens rather than thousands.
        command.extend(["--only-sequences", str(only_sequences),
                        "--maximum-public-edits", "0", "--liability-quantile", "0"])
    return command


def _screen_command(
    *,
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    candidate_limit: int,
    scorer: Path,
    predict_batch_size: int | None,
) -> list[str] | None:
    """Return the scorer command, or ``None`` when the tranche is already scored.

    An artifact from the previous scoring protocol is preserved beside the
    tranche and never resumed, so a protocol change re-scores rather than mixing
    observations from two different selection rules.
    """
    completed = output_dir / "exact_all.json"
    if completed.is_file():
        try:
            protocol = json.loads(completed.read_text()).get("scorer_protocol")
        except (OSError, json.JSONDecodeError):
            protocol = None
        if protocol == SCORER_PROTOCOL:
            return None
        archived = completed.with_name(
            "exact_all.legacy-pre-dual-sample.json"
        )
        suffix = 1
        while archived.exists():
            archived = completed.with_name(
                f"exact_all.legacy-pre-dual-sample-{suffix}.json"
            )
            suffix += 1
        os.replace(completed, archived)
        print(
            f"EXACT_SCREEN_PROTOCOL_REFRESH preserved={archived}",
            flush=True,
        )
    wave_size, batch_size = _screen_execution_shape(
        candidate_limit, len(gpus), predict_batch_size
    )
    return [
        sys.executable,
        "-m",
        "nanobody_opt.score_boltzgen_waves",
        "--sequences", str(source),
        "--config", str(config),
        "--target", target,
        "--output-dir", str(output_dir),
        "--gpus", ",".join(gpus),
        "--wave-size", str(wave_size),
        "--predict-batch-size", str(batch_size),
        "--limit", str(candidate_limit),
        "--scorer", str(scorer),
        "--capture-structures",
    ]


class ScreenJob:
    """One screen tranche whose scorer may still be folding on the GPUs.

    The scorer owns every GPU while it runs, so nothing else may be scheduled
    until ``wait`` returns.  Ranking, basin shrinkage and the mutation-database
    update are pure CPU work and are the only things overlapped with it.
    """

    def __init__(
        self,
        *,
        index: int,
        tranche: list[str],
        root: Path,
        process: subprocess.Popen | None,
        command: list[str] | None,
        started: float,
    ) -> None:
        self.index = index
        self.tranche = tranche
        self.root = root
        self.completed = root / "exact_all.json"
        self._process = process
        self._command = command
        self.started = started
        self.scoring_seconds = 0.0

    def wait(self) -> Path:
        if self._process is None:
            return self.completed
        return_code = self._process.wait()
        self.scoring_seconds = time.monotonic() - self.started
        self._process = None
        if return_code:
            raise subprocess.CalledProcessError(return_code, self._command or [])
        return self.completed


def _start_screen(
    *,
    index: int,
    tranche: list[str],
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    scorer: Path = DEFAULT_SCORER,
    predict_batch_size: int | None = None,
) -> ScreenJob:
    """Dispatch one tranche without waiting for its GPU work to finish."""
    command = _screen_command(
        source=source,
        config=config,
        target=target,
        output_dir=output_dir,
        gpus=gpus,
        candidate_limit=len(tranche),
        scorer=scorer,
        predict_batch_size=predict_batch_size,
    )
    started = time.monotonic()
    process = None if command is None else subprocess.Popen(command)
    return ScreenJob(
        index=index,
        tranche=tranche,
        root=output_dir,
        process=process,
        command=command,
        started=started,
    )



def _run_rounds_concurrently(
    *, groups, source, config, target, root, first_index, alive, scorer,
    predict_batch_size, timings, stage, all_gpus, chain_state, epoch,
    reserve_blocks,
):
    """Run one round per GPU group at the same time. Returns rounds completed.

    Each round is an ordinary scorer invocation into its own round directory,
    so nothing about isolation, shard layout or predict composition changes --
    only which GPUs each invocation is given. A group that fails leaves its
    round uncounted rather than poisoning the wave; the sequential path then
    retries it.
    """
    import concurrent.futures

    launched = []
    for offset, group in enumerate(groups):
        index = first_index + offset
        output = root / "rounds" / f"round_{index:02d}"
        if (output / "exact_all.json").is_file():
            continue
        if not _chain_allows_work(chain_state, epoch, reserve_blocks):
            break
        launched.append((index, group, output))
    if len(launched) < 2:
        return 0

    started = time.monotonic()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(launched)) as pool:
        futures = {
            pool.submit(
                _run_screen, source=source, config=config, target=target,
                output_dir=output, gpus=group, candidate_limit=len(alive),
                scorer=scorer, predict_batch_size=predict_batch_size,
            ): index
            for index, group, output in launched
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                future.result()
                results[index] = True
            except Exception as error:   # one bad group must not lose the wave
                print(
                    f"EXACT_REPLICATION_PARALLEL_ROUND_FAILED round={index} "
                    f"error={type(error).__name__}", flush=True)
                results[index] = False
    seconds = time.monotonic() - started

    done = sum(1 for ok in results.values() if ok)
    _append_timing(
        timings,
        {
            "phase": "replication_round_parallel",
            "stage": stage,
            "rounds": done,
            "concurrency": len(launched),
            "shortlist": len(alive),
            "candidates": len(alive),
            "gpu_count": len(all_gpus),
            "seconds": round(seconds, 1),
        },
    )
    print(
        f"EXACT_REPLICATION_PARALLEL rounds={done}/{len(launched)} "
        f"alive={len(alive)} groups={len(launched)} seconds={seconds:.0f}",
        flush=True)
    # Completed rounds must be a prefix: the sequential loop counts by index.
    ordered = sorted(results)
    prefix = 0
    for index in ordered:
        if results[index]:
            prefix += 1
        else:
            break
    return prefix



def _daemon_job(*, command: list[str], output_dir: Path) -> dict:
    """Thin wrapper over the canonical builder in scorer_daemon."""
    return _scorer_daemon.job_from_command(
        command, observation_id=output_dir.name)


def _run_screen(
    *,
    source: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpus: list[str],
    candidate_limit: int,
    scorer: Path = DEFAULT_SCORER,
    predict_batch_size: int | None = None,
) -> Path:
    command = _screen_command(
        source=source,
        config=config,
        target=target,
        output_dir=output_dir,
        gpus=gpus,
        candidate_limit=candidate_limit,
        scorer=scorer,
        predict_batch_size=predict_batch_size,
    )
    if command is None:
        return output_dir / "exact_all.json"
    # Prefer a warm daemon. It holds the imports, the model, the rotamer
    # library and the analysis pool across rounds, which is 33 s of the
    # measured 32.9 + 1.02*candidates per round. Each round is still its own
    # _worker call into its own output directory, so shard isolation and
    # predict composition -- and therefore the metric -- are unchanged.
    # The daemon is NOT hooked here. `command` runs score_boltzgen_waves,
    # which has no --out and no --batch-size, so the job builder rejected it
    # and every round fell back to a subprocess -- silently, because the
    # fallback is deliberately quiet. The hook lives in the wave loop, on the
    # inner scorer argv that actually does the work.
    subprocess.run(command, check=True)
    return output_dir / "exact_all.json"


def _repeat_command(
    *,
    input_path: Path,
    config: Path,
    target: str,
    output_dir: Path,
    gpu: str,
    scorer: Path = DEFAULT_SCORER,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "nanobody_opt.score_boltzgen_waves",
        "--sequences", str(input_path),
        "--config", str(config),
        "--target", target,
        "--output-dir", str(output_dir),
        "--gpus", gpu,
        "--wave-size", "1",
        "--predict-batch-size", "1",
        "--limit", "1",
        "--scorer", str(scorer),
        "--capture-structures",
    ]


def _run_parallel(commands: list[list[str]]) -> None:
    processes = [subprocess.Popen(command) for command in commands]
    failures = []
    for command, process in zip(commands, processes, strict=True):
        return_code = process.wait()
        if return_code:
            failures.append((return_code, command))
    if failures:
        raise subprocess.CalledProcessError(failures[0][0], failures[0][1])


def _score_repeats(
    *,
    sequence: str,
    repeat_numbers: list[int],
    root: Path,
    config: Path,
    target: str,
    gpus: list[str],
    scorer: Path = DEFAULT_SCORER,
) -> None:
    digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    candidate_root = root / "candidates" / digest
    input_path = candidate_root / "sequence.json"
    _atomic_json(input_path, {"target": target, "sequences": [sequence]})
    for start in range(0, len(repeat_numbers), len(gpus)):
        group = repeat_numbers[start : start + len(gpus)]
        commands = [
            _repeat_command(
                input_path=input_path,
                config=config,
                target=target,
                output_dir=candidate_root / f"repeat_{repeat:02d}",
                gpu=gpus[index],
                scorer=scorer,
            )
            for index, repeat in enumerate(group)
        ]
        _run_parallel(commands)


def _try_strict_rank(
    *,
    root: Path,
    pooled_root: Path | None = None,
    archive: Path,
    config: Path,
    target: str,
    epoch: int,
    output: Path,
    ledger: Path | None,
    strict_gate_overrides: dict[str, Any] | None = None,
    selection_metric_policy: str = "plain",
    consensus_ranking_policy: str = "tier",
) -> bool:
    try:
        subprocess.run(
            _rank_command(
                exact=_pooled_exact_paths(_exact_paths(root), pooled_root, root),
                archive=archive,
                config=config,
                target=target,
                epoch=epoch,
                output=output,
                ledger=ledger,
                strict=True,
                strict_gate_overrides=strict_gate_overrides,
                selection_metric_policy=selection_metric_policy,
                consensus_ranking_policy=consensus_ranking_policy,
            ),
            check=True,
        )
    except subprocess.CalledProcessError:
        return False
    return output.is_file()


def _ranking_by_sequence(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {
        str(candidate["sequence"]): candidate
        for candidate in payload.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("sequence")
    }


def _apply_basin_shrinkage(
    path: Path,
    metadata: dict[str, dict[str, Any]],
    *,
    prior_strength: float = 2.0,
    minimum_samples: int = 3,
) -> dict[str, Any]:
    """Stabilize one-shot terminal ranks while keeping every raw score intact.

    Only terminal candidates whose parent belongs to the same masked core are
    pooled.  Their one-shot p90 is shrunk toward the currently screened basin
    median.  This prevents a single lucky terminal sample from dominating a
    broad weak cohort; repeated exact evidence still decides qualification.
    """
    if prior_strength < 0 or minimum_samples < 1:
        raise ValueError("invalid basin shrinkage configuration")
    payload = json.loads(path.read_text())
    candidates = [
        candidate for candidate in payload.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("sequence")
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        sequence = str(candidate["sequence"])
        record = metadata.get(sequence, {})
        identifier = str(record.get("parent_basin_core_id") or "")
        if not identifier or record.get("proposal_lane") != "basin_terminal":
            continue
        grouped.setdefault(identifier, []).append(candidate)

    for identifier, items in grouped.items():
        if len(items) < minimum_samples:
            continue
        scores = sorted(float(item["score_p90"]) for item in items)
        midpoint = len(scores) // 2
        basin_median = (
            scores[midpoint]
            if len(scores) % 2
            else (scores[midpoint - 1] + scores[midpoint]) / 2
        )
        for item in items:
            raw = float(item["score_p90"])
            adjusted = (
                raw + prior_strength * basin_median
            ) / (1.0 + prior_strength)
            item["basin_adjusted_score"] = adjusted
            item["basin_screen_median"] = basin_median
            item["basin_screen_sample_count"] = len(items)
            item["basin_shrinkage_strength"] = prior_strength
            item["basin_core_id"] = identifier

    original_order = {
        str(candidate["sequence"]): index for index, candidate in enumerate(candidates)
    }
    candidates.sort(key=lambda item: (
        float(item.get("basin_adjusted_score", item["score_p90"])),
        float(item["score_p90"]),
        original_order[str(item["sequence"])],
    ))
    payload["candidates"] = candidates
    payload["sequences"] = [str(item["sequence"]) for item in candidates]
    payload["basin_ranking"] = {
        "enabled": True,
        "prior_strength": prior_strength,
        "minimum_samples": minimum_samples,
        "eligible_basin_count": sum(
            len(items) >= minimum_samples for items in grouped.values()
        ),
    }
    _atomic_json(path, payload)
    return payload


def _best_confirmed_p90(path: Path) -> float:
    scores = [
        float(candidate["score_p90"])
        for candidate in _ranking_by_sequence(path).values()
    ]
    if not scores:
        raise ValueError(f"qualified ranking {path} has no scored candidates")
    return min(scores)


def _challenger_can_overtake(
    *, screen_p90: float, incumbent_p90: float, margin: float
) -> bool:
    """Reject only challengers that clearly trail the confirmed incumbent."""
    return screen_p90 <= incumbent_p90 + margin


def _store_ranking(
    database: Path | None,
    ranking: Path,
    *,
    epoch: int,
    target: str,
    source: str,
    candidate_metadata: dict[str, dict[str, Any]] | None = None,
) -> None:
    if database is None or not ranking.is_file():
        return
    try:
        observations = ranking_observations(
            ranking,
            epoch=epoch,
            target=target,
            source=source,
        )
        for observation in observations:
            metadata = (candidate_metadata or {}).get(str(observation["sequence"]), {})
            observation.update(
                {
                    key: metadata[key]
                    for key in (
                        "parent_sequence",
                        "proposal_type",
                        "mutations",
                        "proposal_lane",
                        "parent_selection_lane",
                        "parent_selection_score",
                        "screen_priority",
                        "parent_basin_core_id",
                        "parent_basin_rank",
                        "basin_core_id",
                        "basin_allocation_reserved",
                    )
                    if metadata.get(key) is not None
                }
            )
        count = update_observation_store(
            database,
            observations,
        )
    except Exception as error:
        # Historical learning must never suppress a publishable result.
        print(
            f"MUTATION_DB_UPDATE_FAILED source={source} "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        # ...but it must not be silent either: this marker alone hid ~27
        # epochs of zero new observations (ARCHITECTURE_AUDIT §0.4). Traceback
        # to stderr plus a line in the box failure log the liveness checker
        # alerts on.
        from nanobody_opt.failure_log import record_failure
        record_failure("sequential_exact_race", "MUTATION_DB_UPDATE_FAILED",
                       error, source=source, database=database)
    else:
        print(
            f"MUTATION_DB_UPDATED source={source} observations={count}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-metric-policy",
        choices=("plain",),
        default="plain",
        help="compatibility flag; ranking uses the validator's plain dense-rank sum",
    )
    parser.add_argument("--consensus-ranking-policy", choices=("tier", "off"), default="tier")
    parser.add_argument("--minimum-selection-observations", type=int)
    parser.add_argument("--carry-pending-candidates", action="store_true")
    parser.add_argument("--maximum-screen-tranches", type=int, default=0,
                        help="zero keeps the legacy limit; one lets the producer regenerate after each observation wave")
    parser.add_argument("--adaptive-finish-guard", action="store_true")
    parser.add_argument("--replication-only", action="store_true")
    parser.add_argument("--replication-finish-guard-seconds", type=float, default=0.0)
    parser.add_argument("--maximum-current-epoch-observations", type=int, default=0,
                        help="zero preserves legacy repeats; positive values move confirmation to candidates not yet confirmed this epoch")
    parser.add_argument("--layer2", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--mutation-database", type=Path)
    parser.add_argument("--chain-state", type=Path)
    parser.add_argument("--reserve-blocks", type=int, default=55)
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated CUDA device ids; defaults to every device present",
    )
    parser.add_argument(
        "--scorer",
        type=Path,
        default=DEFAULT_SCORER,
        help=(
            "validator-protocol-compatible scorer entry point; replacement "
            "scorers must emit the same dual-sample artifact contract"
        ),
    )
    parser.add_argument("--candidate-limit", type=int, default=320)
    parser.add_argument(
        "--screen-tranche-size",
        type=int,
        default=20,
        help="publish and reconsider the leader after this many screen candidates",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=3,
        help="maximum finalists; fewer are repeated when an incumbent qualifies",
    )
    parser.add_argument(
        "--challenger-p90-margin",
        type=float,
        default=20.0,
        help=(
            "repeat a challenger after qualification only when its one-shot "
            "screen p90 is within this rank-sum margin of the incumbent"
        ),
    )
    parser.add_argument(
        "--maximum-initial-screen-p90",
        type=float,
        default=90.0,
        help=(
            "immediately repeat an initial candidate only when its one-shot "
            "p90 is at or below this strict qualification ceiling"
        ),
    )
    parser.add_argument(
        "--weak-initial-fallback-after",
        type=int,
        default=80,
        help=(
            "after screening this many candidates, permit one initial repeat "
            "above the p90 ceiling so first publication remains possible"
        ),
    )
    parser.add_argument("--qualification-observations", type=int, default=8)
    parser.add_argument(
        "--pooled-exact-root",
        type=Path,
        help=(
            "producer root whose earlier iterations' observations join ranking "
            "and replication, so an epoch can screen several generations and "
            "still confirm the best candidate from any of them"
        ),
    )
    parser.add_argument(
        "--layer2-gate-command",
        help=(
            "command template run over the replication shortlist to obtain full "
            "Layer-2 evidence, with {input} and {output} placeholders; the pool "
            "itself is screened on IgBLAST alone because full TNP over a "
            "breadth-first pool costs more than an epoch"
        ),
    )
    parser.add_argument(
        "--maximum-metric-percentile",
        type=float,
        default=1.0,
        help=(
            "drop shortlist candidates whose weakest metric is beaten by more "
            "than this fraction of the recent field; 1.0 disables the guard"
        ),
    )
    parser.add_argument(
        "--replication-rounds-per-stage",
        type=int,
        default=0,
        help=(
            "rounds between prunes; 0 keeps a flat allocation.  Stages compound, "
            "so the surviving candidate still accumulates the observations "
            "strict qualification requires even though pruned ones do not"
        ),
    )
    parser.add_argument(
        "--replication-keep-fraction",
        type=float,
        default=1.0,
        help="fraction of the alive set carried into the next stage",
    )
    parser.add_argument(
        "--screen-gpu-seconds-per-candidate",
        type=float,
        default=7.5,
        help=(
            "GPU-seconds to fold one screened candidate, for sizing a tranche "
            "against the window that will remain when it FINISHES.  Measured "
            "across seven live 700-candidate tranches: 5.16, 5.33, 5.43, 5.48, "
            "5.73, 6.39, 9.01 -- mean 6.08.  The previous default WAS that mean, "
            "which is the wrong statistic for a bound: a third of tranches "
            "overran the window it promised.  7.5 covers all but the outlier. "
            "The asymmetry justifies the margin -- over-estimating costs a few "
            "screened candidates out of a pool already holding ~27%% that break "
            "no settled position, while under-estimating costs the entire "
            "replication window"
        ),
    )
    parser.add_argument(
        "--replication-load-seconds",
        type=float,
        default=80.0,
        help=(
            "measured scorer start-up per invocation, for shortlist sizing.  "
            "Raised from 39 against 68-80s observed for the shortlist-1 rounds "
            "of epoch 25136: start-up is amortised over fewer candidates in "
            "exactly the narrowed regime this sizing exists to protect, so an "
            "optimistic figure here is worst where it matters most"
        ),
    )
    parser.add_argument(
        "--replication-seconds-per-observation",
        type=float,
        default=20.0,
        help=(
            "GPU-seconds per observation for shortlist sizing.  Deliberately "
            "conservative against the measured 6.23: the profile's own figure "
            "is about 94 seconds for a sixty-candidate round, not the 78 a "
            "6.23 model predicts, and an optimistic model returns a shortlist "
            "too wide exactly in the tight window this sizing exists for"
        ),
    )
    parser.add_argument(
        "--minimum-publishable-observations",
        type=int,
        default=8,
        help=(
            "observations strict qualification needs; a closing window that "
            "cannot reach this publishes nothing at all, so it is reported"
        ),
    )
    parser.add_argument(
        "--replication-reserve-blocks",
        type=int,
        default=0,
        help=(
            "stop screening this many blocks before the commitment reserve so "
            "the replication rounds still fit; screening that consumes the "
            "whole epoch leaves every candidate on one observation, which is "
            "the failure the shortlist exists to prevent"
        ),
    )
    parser.add_argument(
        "--strict-gates",
        default="",
        help=(
            "JSON object of re-derived strict qualification thresholds by flag "
            "name; these are calibrated per scorer, so a replacement scorer "
            "must not inherit the production scorer's values"
        ),
    )
    parser.add_argument(
        "--two-stage",
        type=int,
        default=0,
        help=(
            "disable the per-tranche greedy finalist repeats; the shortlist is "
            "replicated once instead.  Independent of --replication-rounds so a "
            "generation that is not the closing one does not silently fall back "
            "to greedy repeats with a shortlist-sized finalist budget"
        ),
    )
    parser.add_argument(
        "--replication-rounds",
        type=int,
        default=0,
        help=(
            "after screening, re-observe the best --finalists candidates this "
            "many times, each round one scorer invocation over the whole "
            "shortlist; 0 keeps the incumbent/challenger repeat loop"
        ),
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        default=1,
        help=(
            "dispatch this many screen tranches ahead so ranking, basin "
            "shrinkage and the mutation-database update run while the GPUs "
            "fold the next tranche; 0 restores fully serial behaviour"
        ),
    )
    parser.add_argument(
        "--screen-predict-batch-size",
        type=int,
        default=0,
        help=(
            "pin how many candidates share one BoltzGen predict call; scores "
            "depend on that grouping, so a larger tranche must not be allowed "
            "to change it implicitly.  0 derives the historical "
            "ceil(candidates / GPUs) shape"
        ),
    )
    parser.add_argument("--basin-shrinkage-strength", type=float, default=2.0)
    parser.add_argument("--minimum-basin-samples", type=int, default=3)
    args = parser.parse_args()
    if not args.gpus:
        # Resolved here rather than as an argparse default so --help stays
        # importable without torch, which the deploy gate depends on.
        args.gpus = visible_gpu_ids()
    if not 1 <= args.finalists <= args.candidate_limit:
        parser.error("require 1 <= --finalists <= --candidate-limit")
    if not 1 <= args.screen_tranche_size <= args.candidate_limit:
        parser.error("require 1 <= --screen-tranche-size <= --candidate-limit")
    if args.qualification_observations < 8:
        parser.error("qualification requires at least eight observations")
    if args.challenger_p90_margin < 0:
        parser.error("--challenger-p90-margin cannot be negative")
    if args.maximum_initial_screen_p90 < 0:
        parser.error("--maximum-initial-screen-p90 cannot be negative")
    if args.weak_initial_fallback_after < 1:
        parser.error("--weak-initial-fallback-after must be positive")
    if args.basin_shrinkage_strength < 0:
        parser.error("--basin-shrinkage-strength cannot be negative")
    if args.minimum_basin_samples < 1:
        parser.error("--minimum-basin-samples must be positive")
    if args.pipeline_depth not in (0, 1):
        parser.error("--pipeline-depth must be 0 or 1")
    if args.replication_rounds < 0:
        parser.error("--replication-rounds cannot be negative")
    if args.maximum_screen_tranches < 0 or args.replication_finish_guard_seconds < 0 or args.maximum_current_epoch_observations < 0:
        parser.error("screen tranche limit and finish guard cannot be negative")
    if args.replication_reserve_blocks < 0:
        parser.error("--replication-reserve-blocks cannot be negative")
    if args.replication_rounds_per_stage < 0:
        parser.error("--replication-rounds-per-stage cannot be negative")
    if not 0 < args.replication_keep_fraction <= 1:
        parser.error("--replication-keep-fraction must be within (0, 1]")
    if args.screen_predict_batch_size < 0:
        parser.error("--screen-predict-batch-size cannot be negative")
    args.screen_predict_batch_size = args.screen_predict_batch_size or None
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("at least one GPU is required")
    try:
        strict_gate_overrides = load_strict_gate_overrides(
            Path(args.strict_gates) if args.strict_gates else None
        )
    except (OSError, ValueError) as error:
        parser.error(f"unusable strict gate overrides: {error}")
    if strict_gate_overrides:
        print(
            "EXACT_STRICT_GATES_OVERRIDDEN "
            f"{json.dumps(strict_gate_overrides, sort_keys=True)}",
            flush=True,
        )
    candidate_records = _candidate_records(args.layer2)
    if args.carry_pending_candidates:
        candidate_records = _carry_pending_records(candidate_records, args.pooled_exact_root,
                                                  args.output_dir, args.target)
    # Confirmation can use the complete scored pool even when no pending case remains.
    if not candidate_records and args.replication_rounds:
        candidate_records = _candidate_records(args.layer2)
    candidate_records = candidate_records[: args.candidate_limit]
    sequences = [str(record["sequence"]) for record in candidate_records]
    candidate_metadata = {
        str(record["sequence"]): record for record in candidate_records
    }
    candidate_metadata = _pooled_candidate_metadata(
        candidate_metadata, args.pooled_exact_root
    )
    if not sequences:
        raise SystemExit("Layer-2 input has no passing candidates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        args.output_dir / "screen_candidates.json",
        {"target": args.target, "sequences": sequences},
    )
    if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
        raise SystemExit("current chain epoch is inside the scoring reserve window")

    screen_ranking = args.output_dir / "screen_ranking.json"
    screen_exact_paths: list[Path] = []
    qualified_outputs: list[Path] = []
    repeated_finalists: list[str] = []
    skipped_finalists: list[str] = []
    deferred_weak_leaders: list[str] = []
    incumbent_p90: float | None = None
    weak_initial_fallback_spent = False
    screened_parent_counts: Counter[str] = Counter()
    screened_source_counts: Counter[str] = Counter()
    screened = 0
    pending = list(sequences)
    tranche_index = 0
    timings_path = args.output_dir / "timings.jsonl"

    def _dispatch_next_tranche(minimum_tranche: int = 0) -> ScreenJob | None:
        """Start the next tranche's GPU work, or return None when none may run.

        Tranche composition depends only on the pending queue and how often each
        structural basin has already been screened, never on the ranking of an
        earlier tranche, so dispatching one tranche ahead selects exactly the
        candidates a fully serial run would have selected.
        """
        nonlocal tranche_index
        if args.replication_only:
            return None
        if args.maximum_screen_tranches and tranche_index >= args.maximum_screen_tranches:
            return None
        if not pending:
            return None
        # The replication reserve must hold on EVERY generation, not only the
        # one that spends it.  Gating it on this generation's round count let
        # non-closing generations screen until only the commitment reserve
        # remained, consuming the window the closing generation needs.
        screen_reserve = args.reserve_blocks + (
            args.replication_reserve_blocks
            if (args.two_stage or args.replication_rounds)
            else 0
        )
        if not _chain_allows_work(args.chain_state, args.epoch, screen_reserve):
            print(
                "EXACT_RACE_RESERVE_REACHED before next screen tranche "
                f"screen_reserve_blocks={screen_reserve}",
                flush=True,
            )
            return None
        # Room to START is not room to FINISH.  Size the tranche to the window
        # that will still exist when its fold completes, or the screen eats the
        # replication window and every candidate is published on one
        # observation -- see _affordable_tranche_size.
        window = max(
            _remaining_blocks(args.chain_state, args.epoch) - screen_reserve, 0.0
        )
        affordable = _affordable_tranche_size(
            args.screen_tranche_size,
            remaining_blocks=window,
            load_seconds=args.replication_load_seconds,
            gpu_seconds_per_candidate=args.screen_gpu_seconds_per_candidate,
            gpu_count=len(gpus),
        )
        affordable = max(affordable, minimum_tranche)
        if affordable < 1:
            print(
                "EXACT_SCREEN_WINDOW_TOO_SHORT "
                f"remaining_blocks={window:.0f} screen_reserve={screen_reserve} "
                f"tranche_size={args.screen_tranche_size}",
                flush=True,
            )
            return None
        if affordable < args.screen_tranche_size:
            print(
                f"EXACT_SCREEN_TRANCHE_NARROWED from={args.screen_tranche_size} "
                f"to={affordable} remaining_blocks={window:.0f}",
                flush=True,
            )
        tranche_index += 1
        tranche_root = args.output_dir / "screen" / f"tranche_{tranche_index:02d}"
        source = tranche_root / "candidates.json"
        completed = tranche_root / "exact_all.json"
        tranche = _resumable_screen_tranche(
            pending,
            candidate_metadata,
            screened_parent_counts,
            affordable,
            source=source,
            completed=completed,
            screened_source_counts=screened_source_counts,
        )
        if not completed.is_file():
            _atomic_json(source, {"target": args.target, "sequences": tranche})
        for sequence in tranche:
            pending.remove(sequence)
            record = candidate_metadata.get(sequence, {})
            parent = str(
                record.get("parent_basin_core_id")
                or record.get("basin_core_id")
                or record.get("parent_sequence")
                or "__unattributed__"
            )
            screened_parent_counts[parent] += 1
            screened_source_counts[_tranche_source_key(candidate_metadata, sequence)] += 1
        return _start_screen(
            index=tranche_index,
            tranche=tranche,
            source=source,
            config=args.config,
            target=args.target,
            output_dir=tranche_root,
            gpus=gpus,
            scorer=args.scorer,
            predict_batch_size=args.screen_predict_batch_size,
        )

    def _dispatch_bootstrap_tranche() -> ScreenJob | None:
        # The epoch has no observations at all.  One observation of something
        # beats none of nothing, so this pass screens against the commitment
        # reserve alone and never sizes itself down to nothing.
        saved = args.replication_reserve_blocks
        args.replication_reserve_blocks = 0
        try:
            return _dispatch_next_tranche(minimum_tranche=1)
        finally:
            args.replication_reserve_blocks = saved

    in_flight = _dispatch_next_tranche()
    if in_flight is None and not _pooled_exact_paths(
        [], args.pooled_exact_root, args.output_dir
    ):
        # The reserve blocked screening and the epoch has no observations at all,
        # so there is nothing to replicate and nothing to publish.  One
        # observation of something beats none of nothing: screen a single
        # tranche against the commitment reserve alone.
        print("EXACT_RACE_BOOTSTRAP_TRANCHE no pooled evidence", flush=True)
        if _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
            in_flight = _dispatch_bootstrap_tranche()
    while in_flight is not None:
        job = in_flight
        in_flight = None
        job.wait()
        # The scorer owns every GPU, so with pipelining the next tranche is
        # dispatched before the CPU-only bookkeeping below runs against it.
        # Without pipelining the next tranche is dispatched after bookkeeping,
        # which is serial -- but it is still dispatched, or the loop would screen
        # exactly one tranche.
        if args.pipeline_depth:
            in_flight = _dispatch_next_tranche()
        bookkeeping_started = time.monotonic()
        screen_exact_paths.append(job.completed)
        screened += len(job.tranche)
        subprocess.run(
            _rank_command(
                exact=_pooled_exact_paths(
                    screen_exact_paths, args.pooled_exact_root, args.output_dir
                ),
                archive=args.archive,
                config=args.config,
                target=args.target,
                epoch=args.epoch,
                output=screen_ranking,
                ledger=args.ledger,
                strict=False,
                selection_metric_policy=getattr(args, "selection_metric_policy", "plain"),
                consensus_ranking_policy=args.consensus_ranking_policy,
                minimum_selection_observations=args.minimum_selection_observations,
                observation_history=args.mutation_database,
            ),
            check=True,
        )
        _apply_basin_shrinkage(
            screen_ranking,
            candidate_metadata,
            prior_strength=args.basin_shrinkage_strength,
            minimum_samples=args.minimum_basin_samples,
        )
        _store_ranking(
            args.mutation_database,
            screen_ranking,
            epoch=args.epoch,
            target=args.target,
            source="live_exact_screen",
            candidate_metadata=candidate_metadata,
        )
        screen_candidates = _ranking_by_sequence(screen_ranking)
        ranked_sequences = json.loads(screen_ranking.read_text())["sequences"]
        print(
            f"EXACT_SCREEN_TRANCHE_COMPLETE tranche={job.index} "
            f"screened={screened}/{len(sequences)} leader={ranked_sequences[0]}",
            flush=True,
        )
        if not args.pipeline_depth:
            in_flight = _dispatch_next_tranche()
        bookkeeping_seconds = time.monotonic() - bookkeeping_started
        _append_timing(
            timings_path,
            {
                "phase": "screen_tranche",
                "tranche": job.index,
                "candidates": len(job.tranche),
                "scoring_seconds": round(job.scoring_seconds, 1),
                "bookkeeping_seconds": round(bookkeeping_seconds, 1),
                "overlapped": bool(args.pipeline_depth),
                "screened": screened,
            },
        )
        print(
            f"EXACT_TRANCHE_TIMING tranche={job.index} "
            f"scoring={job.scoring_seconds:.0f}s "
            f"bookkeeping={bookkeeping_seconds:.0f}s "
            f"candidates={len(job.tranche)}",
            flush=True,
        )
        if args.two_stage or args.replication_rounds:
            # Two-stage mode: the whole shortlist is replicated after screening,
            # so no repeat budget is spent on a per-tranche leader.
            continue
        if len(repeated_finalists) >= args.finalists:
            continue
        weak_fallback_boundary = min(args.weak_initial_fallback_after, len(sequences))
        selection = _initial_repeat_candidate(
            ranked_sequences,
            screen_candidates,
            repeated=set(repeated_finalists),
            maximum_screen_p90=args.maximum_initial_screen_p90,
            allow_weak_fallback=(
                incumbent_p90 is None
                and not weak_initial_fallback_spent
                and screened >= weak_fallback_boundary
            ),
        )
        if selection is None:
            leader = next(
                (item for item in ranked_sequences if item not in repeated_finalists),
                None,
            )
            if leader is not None:
                if leader not in deferred_weak_leaders:
                    deferred_weak_leaders.append(leader)
                print(
                    f"EXACT_INITIAL_REPEAT_DEFERRED screen_p90="
                    f"{float(screen_candidates[leader]['score_p90']):.1f} "
                    f"ceiling={args.maximum_initial_screen_p90:.1f} "
                    f"screened={screened}/{weak_fallback_boundary}",
                    flush=True,
                )
            continue
        sequence, weak_fallback = selection
        if weak_fallback:
            weak_initial_fallback_spent = True
        finalist_index = len(repeated_finalists) + 1
        screen_p90 = float(
            screen_candidates[sequence].get(
                "basin_adjusted_score", screen_candidates[sequence]["score_p90"]
            )
        )
        if incumbent_p90 is not None and not _challenger_can_overtake(
            screen_p90=screen_p90,
            incumbent_p90=incumbent_p90,
            margin=args.challenger_p90_margin,
        ):
            if sequence not in skipped_finalists:
                skipped_finalists.append(sequence)
            print(
                f"EXACT_CHALLENGER_SKIPPED index={finalist_index} "
                f"screen_p90={screen_p90:.1f} incumbent_p90={incumbent_p90:.1f} "
                f"margin={args.challenger_p90_margin:.1f}",
                flush=True,
            )
            continue
        if in_flight is not None:
            # Replication needs every GPU, and so does leaving.  Never break out
            # of this loop with a scorer still folding: the interpreter would
            # exit without reaping it, leaving all ten GPUs busy inside the
            # commitment reserve and OOM-ing the next generation.
            in_flight.wait()
            in_flight = None
        if not _chain_allows_work(args.chain_state, args.epoch, args.reserve_blocks):
            print("EXACT_RACE_RESERVE_REACHED before next finalist", flush=True)
            break
        repeats_started = time.monotonic()
        try:
            _score_repeats(
                sequence=sequence,
                repeat_numbers=list(range(2, args.qualification_observations + 1)),
                root=args.output_dir,
                config=args.config,
                target=args.target,
                gpus=gpus,
                scorer=args.scorer,
            )
        except subprocess.CalledProcessError as error:
            # A transient failure must not consume a finalist slot: five CUDA
            # OOMs would otherwise exhaust the repeat budget and publish
            # nothing despite a fully screened candidate bank.
            if sequence not in skipped_finalists:
                skipped_finalists.append(sequence)
            print(
                f"EXACT_REPEAT_FAILED index={finalist_index} "
                f"returncode={error.returncode}; slot not consumed",
                flush=True,
            )
            continue
        repeated_finalists.append(sequence)
        _append_timing(
            timings_path,
            {
                "phase": "finalist_repeats",
                "finalist": finalist_index,
                "observations": args.qualification_observations - 1,
                "seconds": round(time.monotonic() - repeats_started, 1),
            },
        )
        output = args.output_dir / f"qualified_after_{finalist_index:02d}.json"
        if _try_strict_rank(
            root=args.output_dir,
            archive=args.archive,
            config=args.config,
            target=args.target,
            epoch=args.epoch,
            output=output,
            ledger=args.ledger,
            pooled_root=args.pooled_exact_root,
            strict_gate_overrides=strict_gate_overrides,
            selection_metric_policy=getattr(args, "selection_metric_policy", "plain"),
            consensus_ranking_policy=args.consensus_ranking_policy,
        ):
            _store_ranking(
                args.mutation_database,
                output,
                epoch=args.epoch,
                target=args.target,
                source="live_exact_repeat",
                candidate_metadata=candidate_metadata,
            )
            qualified_outputs.append(output)
            incumbent_p90 = _best_confirmed_p90(output)
            print(
                f"EXACT_CANDIDATE_QUALIFIED index={finalist_index} "
                f"incumbent_p90={incumbent_p90:.1f} output={output}",
                flush=True,
            )
        else:
            print(f"EXACT_CANDIDATE_REJECTED index={finalist_index}", flush=True)

    replication_rounds_completed = 0
    replication_shortlist: list[str] = []
    if args.replication_rounds:
        pooled_for_ranking = _pooled_exact_paths(
            [], args.pooled_exact_root, args.output_dir
        )
        already_confirmed = set()
        if args.maximum_current_epoch_observations:
            from nanobody_opt.rank_exact import _observations
            already_confirmed = {
                sequence for sequence, observations in _observations(pooled_for_ranking, args.target).items()
                if len(observations) >= args.maximum_current_epoch_observations
            }
        if not screen_ranking.is_file() and pooled_for_ranking:
            # The closing iteration may screen nothing at all: it opens inside
            # the replication window on purpose.  Rank the epoch's pooled
            # observations so the shortlist is the best of every generation.
            subprocess.run(
                _rank_command(
                    exact=pooled_for_ranking,
                    archive=args.archive,
                    config=args.config,
                    target=args.target,
                    epoch=args.epoch,
                    output=screen_ranking,
                    ledger=args.ledger,
                    strict=False,
                    selection_metric_policy=getattr(args, "selection_metric_policy", "plain"),
                    consensus_ranking_policy=args.consensus_ranking_policy,
                    minimum_selection_observations=args.minimum_selection_observations,
                    observation_history=args.mutation_database,
                ),
                check=True,
            )
            # Order the pooled ranking the same way a screened tranche is
            # ordered.  Without this the closing shortlist is chosen on raw
            # one-shot p90, which is exactly the statistic that carries the
            # winner's curse: selecting the best of thousands of single draws
            # overstates quality by roughly the noise spread times the
            # selection depth, and the candidate regresses when confirmed.
            _apply_basin_shrinkage(
                screen_ranking,
                candidate_metadata,
                prior_strength=args.basin_shrinkage_strength,
                minimum_samples=args.minimum_basin_samples,
            )
        replication_shortlist = _replication_shortlist(
            screen_ranking, exclude=set(repeated_finalists) | already_confirmed, limit=args.finalists
        )
        replication_shortlist = _drop_metric_outliers(
            replication_shortlist,
            ranking=screen_ranking,
            archive=args.archive,
            maximum_percentile=args.maximum_metric_percentile,
        )
        replication_shortlist = _affordable_shortlist(
            replication_shortlist,
            rounds=args.replication_rounds,
            minimum_observations=args.minimum_publishable_observations,
            remaining_blocks=max(
                _remaining_blocks(args.chain_state, args.epoch)
                - args.reserve_blocks,
                0.0,
            ),
            load_seconds=args.replication_load_seconds,
            seconds_per_observation=args.replication_seconds_per_observation,
            gpu_count=len(gpus),
        )
        if args.layer2_gate_command:
            gate_started = time.monotonic()
            replication_shortlist = _layer2_gate(
                replication_shortlist,
                command=shlex.split(args.layer2_gate_command),
                root=args.output_dir,
                target=args.target,
            )
            _append_timing(timings_path, {"phase": "layer2_gate", "seconds": round(time.monotonic() - gate_started, 1)})
        finish_model = None
        if args.adaptive_finish_guard:
            from nanobody_opt.compute_budget import TimingModel
            finish_model = TimingModel(args.replication_load_seconds, args.replication_seconds_per_observation, 0)
            # TNP consumes time after initial sizing: use the clock again before folding.
            usable = max(_remaining_blocks(args.chain_state, args.epoch) - args.reserve_blocks, 0) * 12.0
            # Price the run the loop actually executes (it prunes between
            # stages), as confirmation_plan does. The flat rounds*width form
            # cut 96 finalists to ~51 at 20 rounds on 8 GPUs. The per-round
            # guard inside _run_replication_rounds still stops on overrun.
            while replication_shortlist and (
                finish_model.staged_seconds(
                    rounds=args.replication_rounds,
                    candidates=len(replication_shortlist),
                    gpus=len(gpus),
                    rounds_per_stage=args.replication_rounds_per_stage,
                    keep_fraction=args.replication_keep_fraction,
                )
                + args.replication_finish_guard_seconds > usable
            ):
                replication_shortlist.pop()
        def _prune(alive: list[str], keep: int) -> list[str]:
            """Keep the best of the alive set on all pooled observations."""
            ranking = args.output_dir / "rounds" / "stage_ranking.json"
            alive_file = args.output_dir / "rounds" / "stage_alive.txt"
            alive_file.parent.mkdir(parents=True, exist_ok=True)
            alive_file.write_text("".join(f"{item}\n" for item in alive))
            try:
                subprocess.run(
                    _rank_command(
                        exact=_pooled_exact_paths(
                            _exact_paths(args.output_dir),
                            args.pooled_exact_root,
                            args.output_dir,
                        ),
                        archive=args.archive,
                        config=args.config,
                        target=args.target,
                        epoch=args.epoch,
                        output=ranking,
                        ledger=args.ledger,
                        strict=False,
                        selection_metric_policy=getattr(args, "selection_metric_policy", "plain"),
                        consensus_ranking_policy=args.consensus_ranking_policy,
                        minimum_selection_observations=args.minimum_selection_observations,
                        observation_history=args.mutation_database,
                        only_sequences=alive_file,
                    ),
                    check=True,
                )
                order = json.loads(ranking.read_text()).get("sequences", [])
            except (subprocess.CalledProcessError, OSError, ValueError):
                # Failing to rank must not end the tournament; carrying the set
                # forward unpruned only costs efficiency.
                print("EXACT_REPLICATION_PRUNE_SKIPPED", flush=True)
                return alive
            alive_set = set(alive)
            ranked = [str(item) for item in order if str(item) in alive_set]
            missing = [item for item in alive if item not in set(ranked)]
            return _keep_a_family_slate(ranked + missing, keep)

        replication_rounds_completed = _run_replication_rounds(
            shortlist=replication_shortlist,
            rounds=args.replication_rounds,
            root=args.output_dir,
            config=args.config,
            target=args.target,
            gpus=gpus,
            scorer=args.scorer,
            chain_state=args.chain_state,
            epoch=args.epoch,
            reserve_blocks=args.reserve_blocks,
            predict_batch_size=args.screen_predict_batch_size,
            timings=timings_path,
            rounds_per_stage=args.replication_rounds_per_stage,
            keep_fraction=args.replication_keep_fraction,
            rerank=_prune,
            finish_model=finish_model,
            finish_guard_seconds=args.replication_finish_guard_seconds,
        )
        observations = replication_rounds_completed + 1
        if observations < args.minimum_publishable_observations:
            print(
                "EXACT_REPLICATION_UNDER_QUALIFIED "
                f"observations={observations} "
                f"required={args.minimum_publishable_observations}; "
                "the closing window was too short to confirm a candidate",
                flush=True,
            )
        if replication_rounds_completed:
            output = args.output_dir / "qualified_after_rounds.json"
            if _try_strict_rank(
                root=args.output_dir,
                archive=args.archive,
                config=args.config,
                target=args.target,
                epoch=args.epoch,
                output=output,
                ledger=args.ledger,
                pooled_root=args.pooled_exact_root,
                strict_gate_overrides=strict_gate_overrides,
                selection_metric_policy=getattr(args, "selection_metric_policy", "plain"),
                consensus_ranking_policy=args.consensus_ranking_policy,
            ):
                _store_ranking(
                    args.mutation_database,
                    output,
                    epoch=args.epoch,
                    target=args.target,
                    source="live_exact_repeat",
                    candidate_metadata=candidate_metadata,
                )
                qualified_outputs.append(output)
                incumbent_p90 = _best_confirmed_p90(output)
                print(
                    "EXACT_REPLICATION_QUALIFIED "
                    f"incumbent_p90={incumbent_p90:.1f} output={output}",
                    flush=True,
                )
            else:
                print("EXACT_REPLICATION_REJECTED", flush=True)

    _atomic_json(
        args.output_dir / "complete.json",
        {
            "epoch": args.epoch,
            "replication_rounds": args.replication_rounds,
            "replication_rounds_completed": replication_rounds_completed,
            "replication_shortlist": len(replication_shortlist),
            "target": args.target,
            "screened": screened,
            "candidate_limit": len(sequences),
            "screen_tranche_size": args.screen_tranche_size,
            "maximum_finalists": args.finalists,
            "repeated_finalists": repeated_finalists,
            "skipped_finalists": skipped_finalists,
            "deferred_weak_leaders": deferred_weak_leaders,
            "challenger_p90_margin": args.challenger_p90_margin,
            "maximum_initial_screen_p90": args.maximum_initial_screen_p90,
            "weak_initial_fallback_after": args.weak_initial_fallback_after,
            "weak_initial_fallback_spent": weak_initial_fallback_spent,
            "basin_shrinkage_strength": args.basin_shrinkage_strength,
            "minimum_basin_samples": args.minimum_basin_samples,
            "incumbent_p90": incumbent_p90,
            "qualified_rankings": [str(path) for path in qualified_outputs],
            "exact_files": [str(path) for path in _exact_paths(args.output_dir)],
        },
    )
    print(
        f"EXACT_RACE_COMPLETE repeated={len(repeated_finalists)} "
        f"skipped={len(skipped_finalists)} "
        f"qualified={len(qualified_outputs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
