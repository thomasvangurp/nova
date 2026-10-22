"""Scored vector index and mutation graph for historical VHH sequences."""
from __future__ import annotations

import json
import math
import pickle
import sys
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanobody_opt import db
from nanobody_opt.regions import find_regions
from nanobody_opt.simulate_round import METRICS


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}
AA_PROPERTY = {
    **dict.fromkeys("AVILMFWY", "hydrophobic"),
    **dict.fromkeys("STNQ", "polar"),
    **dict.fromkeys("KRH", "positive"),
    **dict.fromkeys("DE", "negative"),
    **dict.fromkeys("CGP", "special"),
}


@dataclass(frozen=True)
class MutationNode:
    sequence: str
    score: float
    observations: int
    sources: tuple[str, ...]
    epochs: tuple[int, ...]


@dataclass(frozen=True)
class MutationNeighbour:
    sequence: str
    score: float
    distance: int
    adjusted_score: float
    mutations: tuple[str, ...]
    observations: int
    sources: tuple[str, ...]
    supported_effect_count: int = 0
    effect_uncertainty: float = 0.0


def mutations(left: str, right: str) -> tuple[str, ...]:
    if len(left) != len(right):
        raise ValueError("mutation endpoints have different lengths")
    return tuple(
        f"{before}{position + 1}{after}"
        for position, (before, after) in enumerate(
            zip(left, right, strict=True)
        )
        if before != after
    )


def mutation_contexts(
    left: str,
    right: str,
    *,
    contact_positions: Iterable[int] = (),
) -> list[dict[str, Any]]:
    """Describe substitutions with stable VHH-region and chemistry features."""
    if len(left) != len(right):
        raise ValueError("mutation endpoints have different lengths")
    contacts = set(contact_positions)
    try:
        regions = find_regions(left)
        region_by_position = {
            position: name
            for name, bounds in (
                ("cdr1", regions.cdr1),
                ("cdr2", regions.cdr2),
                ("cdr3", regions.cdr3),
            )
            for position in range(*bounds)
        }
    except ValueError:
        region_by_position = {}
    contexts = []
    for position, (before, after) in enumerate(zip(left, right, strict=True)):
        if before == after:
            continue
        terminal = (
            "n_terminal" if position < 10 else
            "c_terminal" if position >= len(left) - 17 else
            "internal"
        )
        contexts.append({
            "position": position + 1,
            "from_residue": before,
            "to_residue": after,
            "region": region_by_position.get(position, "framework"),
            "terminal_class": terminal,
            "contact_class": "contact" if position in contacts else "unknown",
            "property_change": (
                "conservative"
                if AA_PROPERTY[before] == AA_PROPERTY[after]
                else f"{AA_PROPERTY[before]}_to_{AA_PROPERTY[after]}"
            ),
        })
    return contexts


class MutationNetwork:
    """Compact position-vector store with virtual-node nearest neighbours."""

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        *,
        mutation_effects: Iterable[dict[str, Any]] = (),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            sequence = str(record.get("sequence", "")).strip().upper()
            try:
                score = float(record["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not sequence
                or not math.isfinite(score)
                or any(residue not in AA_INDEX for residue in sequence)
            ):
                continue
            grouped[sequence].append({**record, "score": score})
        if not grouped:
            raise ValueError("mutation network has no valid scored sequences")
        lengths = {len(sequence) for sequence in grouped}
        if len(lengths) != 1:
            raise ValueError("mutation network requires one sequence length")
        self.length = lengths.pop()
        self.nodes: list[MutationNode] = []
        for sequence, observations in grouped.items():
            public = [
                item for item in observations
                if item.get("score_kind") == "validator_final_rank_sum"
            ]
            scoring_observations = public or observations
            scores = sorted(float(item["score"]) for item in scoring_observations)
            sources = tuple(sorted({
                str(item.get("source", "unknown")) for item in observations
            }))
            epochs = tuple(sorted({
                int(item["epoch"])
                for item in observations
                if item.get("epoch") is not None
            }))
            # Preserve an elite observation without letting a single lucky
            # score fully dominate repeated evidence.
            robust_score = 0.5 * scores[0] + 0.5 * float(np.median(scores))
            self.nodes.append(MutationNode(
                sequence=sequence,
                score=robust_score,
                observations=sum(
                    int(item.get("observation_count", 1))
                    for item in scoring_observations
                ),
                sources=sources,
                epochs=epochs,
            ))
        self.nodes.sort(key=lambda node: (node.score, node.sequence))
        self._by_sequence = {node.sequence: node for node in self.nodes}
        self._contacts: dict[str, set[int]] = {}
        for sequence, observations in grouped.items():
            contacts = set()
            for item in observations:
                pose = item.get("pose_stability") or {}
                contacts.update(
                    int(position) - 1
                    for position in pose.get(
                        "consensus_nanobody_contact_positions", []
                    )
                    if 1 <= int(position) <= len(sequence)
                )
            self._contacts[sequence] = contacts
        self._vectors = np.asarray([
            [AA_INDEX[residue] for residue in node.sequence]
            for node in self.nodes
        ], dtype=np.uint8)
        # Node scores as an array so a neighbour query can build its ranking key
        # with numpy instead of a per-node Python lambda.
        self._scores = np.asarray(
            [node.score for node in self.nodes], dtype=np.float64
        )
        self._effects: dict[tuple[int, str, str, str, str], dict[str, Any]] = {}
        for effect in mutation_effects:
            key = (
                int(effect["position"]),
                str(effect["from_residue"]),
                str(effect["to_residue"]),
                str(effect["region"]),
                str(effect["contact_class"]),
            )
            self._effects[key] = effect
        # The contribution a learned effect makes is a function of the effect
        # alone -- the shrinkage, the reciprocal-root-support term and the
        # squared uncertainty do not depend on the query.  Recomputing them per
        # lookup cost four hundred thousand ``sqrt``/``max``/divide triples per
        # generation.  Precomputed with exactly the same operations in exactly
        # the same order, so the accumulated float is identical.  A malformed
        # effect is simply left out and still takes the original path, which
        # keeps its exception behaviour where it was.
        self._effect_terms: dict[
            tuple[int, str, str, str, str], tuple[float, float]
        ] = {}
        for key, effect in self._effects.items():
            try:
                support = int(effect["support"])
                shrinkage = support / (support + 3.0)
                uncertainty = float(effect["uncertainty"])
                term = (
                    shrinkage * float(effect["median_score_delta"])
                    + uncertainty / math.sqrt(max(support, 1))
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            self._effect_terms[key] = (term, uncertainty**2)
        # Refining a shortlist asks "is there a learned effect for this exact
        # substitution?" once per differing position per candidate -- hundreds
        # of thousands of times per generation.  The region and contact class
        # are only ever a further *narrowing* of (position, from, to), so a
        # coarse (position, from, to) bitmap is a sound pre-filter: a position
        # it rejects cannot possibly hit ``self._effects``, and one it accepts
        # still goes through the exact key below.  Only a few percent of the
        # 127x20x20 substitution space is ever observed, so this skips the
        # region parse for the overwhelming majority of candidates.
        self._effect_screen = np.zeros(self.length * 400, dtype=bool)
        for position, before, after, _region, _contact in self._effects:
            index = position - 1
            if not 0 <= index < self.length:
                continue
            if before not in AA_INDEX or after not in AA_INDEX:
                continue
            self._effect_screen[
                index * 400 + AA_INDEX[before] * 20 + AA_INDEX[after]
            ] = True
        self._node_sequences = [node.sequence for node in self.nodes]
        self._node_contacts = [
            self._contacts.get(node.sequence, frozenset()) for node in self.nodes
        ]
        # ``find_regions`` is deterministic per node sequence and independent of
        # the query, so it is loop-invariant across a whole refine pass -- and
        # across every query the process ever makes.  Filled lazily, and only
        # for nodes the screen above could not already settle.
        self._node_region_bounds: list[Any] = [False] * len(self.nodes)

    def _region_bounds(self, index: int) -> tuple[tuple[int, int], ...] | None:
        """CDR spans for a node, cached; ``None`` when the parse fails."""
        bounds = self._node_region_bounds[index]
        if bounds is not False:
            return bounds
        try:
            regions = find_regions(self._node_sequences[index])
        except ValueError:
            bounds = None
        else:
            bounds = (regions.cdr1, regions.cdr2, regions.cdr3)
        self._node_region_bounds[index] = bounds
        return bounds

    def _prospective_adjustments(
        self,
        indices: list[int],
        candidate: str,
        vector: np.ndarray,
        *,
        distance_penalty: float,
    ) -> list[tuple[float, int, float]]:
        """Directional-effect refinement for a whole shortlist at once.

        The form this replaces walked all 127 positions of every candidate in
        Python and reparsed the node's CDR boundaries each time -- 700,000 calls
        and 89 million position comparisons per generation.  Here the
        substitutions are found once with numpy for the entire shortlist, the
        region parse is cached per node, the per-effect constants are
        precomputed, and a (position, from, to) bitmap skips the region work for
        candidates that cannot match any learned effect at all.

        The per-position accumulation still runs in ascending position order
        with the same operations on the same values, so the accumulated float is
        bit-identical, not merely close.  tests/test_mutation_network.py pins
        that against the original body.
        """
        if not indices:
            return []
        rows = self._vectors[indices]
        differing = rows != vector
        row_index, column_index = np.nonzero(differing)
        counts = np.count_nonzero(differing, axis=1).tolist()
        if row_index.size:
            screen = self._effect_screen[
                column_index * 400
                + rows[row_index, column_index].astype(np.intp) * 20
                + vector[column_index].astype(np.intp)
            ].tolist()
            columns = column_index.tolist()
        else:
            screen = []
            columns = []
        effects = self._effects
        terms = self._effect_terms
        results = []
        cursor = 0
        for offset, index in enumerate(indices):
            start = cursor
            cursor += counts[offset]
            adjustment = 0.0
            if True not in screen[start:cursor]:
                # No learned effect can match any of this candidate's
                # substitutions: every one of them takes the miss branch.
                for _ in range(start, cursor):
                    adjustment += distance_penalty
                results.append((adjustment, 0, 0.0))
                continue
            bounds = self._region_bounds(index)
            contacts = self._node_contacts[index]
            source = self._node_sequences[index]
            if bounds is None:
                first = second = third = (0, 0)
            else:
                first, second, third = bounds
            supported = 0
            uncertainty_squared = 0.0
            for cell in range(start, cursor):
                if not screen[cell]:
                    adjustment += distance_penalty
                    continue
                position = columns[cell]
                # ``mutation_contexts`` labels a position by a dict comprehension
                # over cdr1, cdr2 then cdr3, so a later span wins an overlap;
                # testing in reverse reproduces that.
                if third[0] <= position < third[1]:
                    region = "cdr3"
                elif second[0] <= position < second[1]:
                    region = "cdr2"
                elif first[0] <= position < first[1]:
                    region = "cdr1"
                else:
                    region = "framework"
                key = (
                    position + 1,
                    source[position],
                    candidate[position],
                    region,
                    "contact" if contacts and position in contacts else "unknown",
                )
                precomputed = terms.get(key)
                if precomputed is not None:
                    adjustment += precomputed[0]
                    supported += 1
                    uncertainty_squared += precomputed[1]
                    continue
                effect = effects.get(key)
                if effect is None:
                    adjustment += distance_penalty
                    continue
                support = int(effect["support"])
                shrinkage = support / (support + 3.0)
                uncertainty = float(effect["uncertainty"])
                adjustment += (
                    shrinkage * float(effect["median_score_delta"])
                    + uncertainty / math.sqrt(max(support, 1))
                )
                supported += 1
                uncertainty_squared += uncertainty**2
            results.append((adjustment, supported, math.sqrt(uncertainty_squared)))
        return results

    def neighbours(
        self,
        sequence: str,
        *,
        limit: int = 16,
        maximum_distance: int | None = 6,
        distance_penalty: float = 5.0,
    ) -> list[MutationNeighbour]:
        sequence = sequence.strip().upper()
        if len(sequence) != self.length or any(
            residue not in AA_INDEX for residue in sequence
        ):
            raise ValueError("query is incompatible with mutation network")
        if limit < 1 or distance_penalty < 0:
            raise ValueError("invalid neighbour query configuration")
        vector = np.asarray([AA_INDEX[residue] for residue in sequence], dtype=np.uint8)
        distances = np.count_nonzero(self._vectors != vector, axis=1)
        # This used to rank EVERY node with a Python lambda key and then keep
        # the first few hundred: on a 28,000-node network that is 28,000 key
        # evaluations per query, and the producer issues one query per accepted
        # proposal.  It was 89% of candidate generation -- 79 million lambda
        # calls in one epoch's run.  Same ordering, built the cheap way: the
        # primary key is vectorised, incompatible nodes are dropped before the
        # sort rather than after, and only the shortlist is ordered exactly.
        shortlist = max(256, limit * 16)
        primary = self._scores + distance_penalty * distances
        if maximum_distance is None:
            pool = np.arange(len(self.nodes))
        else:
            pool = np.flatnonzero(distances <= maximum_distance)
        if pool.size > shortlist:
            values = primary[pool]
            cut = float(np.partition(values, shortlist - 1)[shortlist - 1])
            # Keep every tie at the cut so the exact sort below sees the whole
            # boundary and cannot drop a node that outranks one it kept.
            pool = pool[values <= cut]
        compatible = sorted(
            pool.tolist(),
            key=lambda index: (
                float(primary[index]),
                int(distances[index]),
                self.nodes[index].sequence,
            ),
        )
        # Refine a generous vector shortlist with contextual directional
        # effects. This keeps virtual-node queries cheap even after thousands
        # of historical sequences have been ingested.
        shortlisted = compatible[:shortlist]
        adjustments = self._prospective_adjustments(
            shortlisted, sequence, vector, distance_penalty=distance_penalty
        )
        refined = []
        for index, (adjustment, supported, uncertainty) in zip(
            shortlisted, adjustments, strict=True
        ):
            node = self.nodes[index]
            refined.append((
                node.score + adjustment,
                int(distances[index]),
                node.sequence,
                index,
                supported,
                uncertainty,
            ))
        refined.sort()
        found = []
        for adjusted_score, distance, _, index, supported, uncertainty in refined:
            node = self.nodes[index]
            found.append(MutationNeighbour(
                sequence=node.sequence,
                score=node.score,
                distance=distance,
                adjusted_score=adjusted_score,
                mutations=mutations(node.sequence, sequence),
                observations=node.observations,
                sources=node.sources,
                supported_effect_count=supported,
                effect_uncertainty=uncertainty,
            ))
            if len(found) >= limit:
                break
        return found

    def evidence(
        self,
        sequence: str,
        *,
        limit: int = 16,
        maximum_distance: int | None = 6,
        distance_penalty: float = 5.0,
        consensus_size: int = 5,
    ) -> dict[str, Any]:
        neighbours = self.neighbours(
            sequence,
            limit=limit,
            maximum_distance=maximum_distance,
            distance_penalty=distance_penalty,
        )
        if not neighbours:
            neighbours = self.neighbours(
                sequence,
                limit=limit,
                maximum_distance=None,
                distance_penalty=distance_penalty,
            )
        consensus = neighbours[: max(1, consensus_size)]
        return {
            "predecessor_min_distance": min(item.distance for item in neighbours),
            "predecessor_potential_score": neighbours[0].adjusted_score,
            "prospective_score": neighbours[0].adjusted_score,
            "prospective_effect_support": (
                neighbours[0].supported_effect_count
            ),
            "prospective_uncertainty": neighbours[0].effect_uncertainty,
            "predecessor_consensus_score": float(np.mean([
                item.adjusted_score for item in consensus
            ])),
            "predecessor_support_within_six": sum(
                item.distance <= 6 for item in neighbours
            ),
            "predecessor_neighbours": [asdict(item) for item in neighbours],
        }

    def graph(self, *, edge_distance: int = 6, edge_limit: int = 16) -> dict[str, Any]:
        edges = []
        seen: set[tuple[str, str]] = set()
        for node in self.nodes:
            for neighbour in self.neighbours(
                node.sequence,
                limit=edge_limit + 1,
                maximum_distance=edge_distance,
            ):
                if neighbour.distance == 0:
                    continue
                endpoints = tuple(sorted((node.sequence, neighbour.sequence)))
                if endpoints in seen:
                    continue
                seen.add(endpoints)
                left = self._by_sequence[endpoints[0]]
                right = self._by_sequence[endpoints[1]]
                edges.append({
                    "left_sequence": endpoints[0],
                    "right_sequence": endpoints[1],
                    "distance": neighbour.distance,
                    "mutations_left_to_right": mutations(endpoints[0], endpoints[1]),
                    "score_delta_right_minus_left": right.score - left.score,
                })
        return {
            "schema_version": 1,
            "sequence_length": self.length,
            "node_count": len(self.nodes),
            "edge_count": len(edges),
            "nodes": [asdict(node) for node in self.nodes],
            "edges": edges,
        }

    def save(self, path: Path, *, edge_distance: int = 6, edge_limit: int = 16) -> None:
        payload = self.graph(edge_distance=edge_distance, edge_limit=edge_limit)
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


def _connect(
    path: Path | None = None, *, create: bool = False,
) -> "db.Connection | db.LocalConnection":
    """Open master's write connection: local SQLite primary, or central MySQL.

    Local sqlite is used whenever ``path`` names a file that already exists
    (ARCHITECTURE_REDESIGN new finding 8) -- the same path every caller
    already passes for historical sqlite3 call-site compatibility. A box
    without that file (the dashboard box, any box that is not master) falls
    through to central MySQL automatically, with no configuration needed.
    ``create=True`` is the one deliberate exception: bootstrapping a brand
    new local store (db.connect_local applies local_schema.sql on first
    open) has no file to check existence of yet, so the caller says so
    explicitly instead of the file silently not existing and this routing to
    MySQL by accident.

    User, 2026-10-11: this used to be gated on NOVA_LOCAL_SQLITE=1, an opt-in
    environment variable every process had to remember to export. The scorer
    daemon's own supervisor script never did -- every observation it wrote
    went straight to MySQL, silently orphaned from master's actual primary,
    for the ~20 epochs between the 2026-10-10 cutover and this fix (recovered
    via tools/sync_missing_from_mysql.py). Checking the file itself removes
    that whole failure mode: there is no flag left to forget. Replication to
    Hetzner MySQL (for cross-box visibility and durability) remains a
    separate, downstream process (nanobody_opt/replicate_local.py), not
    synchronous with this write.
    """
    if path is not None and (create or path.exists()):
        return db.connect_local(path)
    # Central MySQL is retired (user, 2026-10-11: "nothing writes to MySQL
    # anymore, nor should anything read from there"). A missing file is a
    # wrong path, never a cue to fall back to another store.
    raise FileNotFoundError(f"local observation store not found: {path}")


def _is_local(connection: "db.Connection | db.LocalConnection") -> bool:
    return isinstance(connection, db.LocalConnection)


_NODE_UPSERT_SQL = """
    INSERT INTO nodes(target, sequence, length, residue_vector)
    VALUES(%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        length=VALUES(length),
        residue_vector=VALUES(residue_vector)
"""
_OBSERVATION_UPSERT_SQL = """
    INSERT INTO observations(
        observation_id, node_id, epoch, source, score_kind, score,
        observation_count, metrics_json, pose_json, provenance_json
    ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        node_id=VALUES(node_id),
        epoch=VALUES(epoch),
        source=VALUES(source),
        score_kind=VALUES(score_kind),
        score=VALUES(score),
        observation_count=VALUES(observation_count),
        metrics_json=VALUES(metrics_json),
        pose_json=VALUES(pose_json),
        provenance_json=VALUES(provenance_json)
"""
_ARTIFACT_UPSERT_SQL = """
    INSERT INTO observation_artifacts(
        observation_id, confidence_path, writer_path,
        structure_json, sample_selection_json
    ) VALUES(%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        confidence_path=VALUES(confidence_path),
        writer_path=VALUES(writer_path),
        structure_json=VALUES(structure_json),
        sample_selection_json=VALUES(sample_selection_json)
"""
_LINEAGE_UPSERT_SQL = """
    INSERT INTO lineage_edges(
        parent_node_id, child_node_id, epoch, source,
        proposal_type, mutations_json, metadata_json
    ) VALUES(%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        proposal_type=VALUES(proposal_type),
        mutations_json=VALUES(mutations_json),
        metadata_json=VALUES(metadata_json)
"""
# SQLite dialect of the four upserts above: `?` placeholders, `ON CONFLICT`
# needs its target (the unique/primary key actually violated) spelled out --
# unlike MySQL's `ON DUPLICATE KEY UPDATE`, which infers it -- so these are
# hand-written per table, not auto-translated. Row semantics (which column
# wins, idempotency) are identical to the MySQL versions above.
_NODE_UPSERT_SQL_SQLITE = """
    INSERT INTO nodes(target, sequence, length, residue_vector)
    VALUES(?, ?, ?, ?)
    ON CONFLICT(target, sequence) DO UPDATE SET
        length=excluded.length,
        residue_vector=excluded.residue_vector
"""
_OBSERVATION_UPSERT_SQL_SQLITE = """
    INSERT INTO observations(
        observation_id, node_id, epoch, source, score_kind, score,
        observation_count, metrics_json, pose_json, provenance_json
    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(observation_id) DO UPDATE SET
        node_id=excluded.node_id,
        epoch=excluded.epoch,
        source=excluded.source,
        score_kind=excluded.score_kind,
        score=excluded.score,
        observation_count=excluded.observation_count,
        metrics_json=excluded.metrics_json,
        pose_json=excluded.pose_json,
        provenance_json=excluded.provenance_json
"""
_ARTIFACT_UPSERT_SQL_SQLITE = """
    INSERT INTO observation_artifacts(
        observation_id, confidence_path, writer_path,
        structure_json, sample_selection_json
    ) VALUES(?, ?, ?, ?, ?)
    ON CONFLICT(observation_id) DO UPDATE SET
        confidence_path=excluded.confidence_path,
        writer_path=excluded.writer_path,
        structure_json=excluded.structure_json,
        sample_selection_json=excluded.sample_selection_json
"""
_LINEAGE_UPSERT_SQL_SQLITE = """
    INSERT INTO lineage_edges(
        parent_node_id, child_node_id, epoch, source,
        proposal_type, mutations_json, metadata_json
    ) VALUES(?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(parent_node_id, child_node_id, epoch, source) DO UPDATE SET
        proposal_type=excluded.proposal_type,
        mutations_json=excluded.mutations_json,
        metadata_json=excluded.metadata_json
"""
_OBSERVATION_EXCLUDED = frozenset({
    "observation_id", "sequence", "target", "epoch", "source",
    "score_kind", "score", "observation_count", "metrics",
    "pose_stability", "structure",
})
_LINEAGE_EXCLUDED = frozenset({
    "observation_id", "sequence", "parent_sequence", "target",
    "epoch", "source", "proposal_type", "mutations",
    "score_kind", "score", "observation_count", "metrics",
    "pose_stability",
})
# One multi-row statement per table per chunk. A 1000-row chunk of
# live_exact_screen rows is ~1-3 MB of SQL (metrics/provenance JSON), so 8 MiB
# keeps it to one statement while staying far below the server's 64 MiB
# max_allowed_packet. mysqlclient's own default (64 KiB) would split the same
# chunk into ~40 statements, i.e. ~40 extra network round-trips.
_MAX_STATEMENT_BYTES = 8 * 1024 * 1024
# Bound for one ``sequence IN (...)`` id lookup (keeps the statement small).
_NODE_LOOKUP_BATCH = 1000


def _prepare_observation(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one record and build every row it writes, except the node ids.

    Pure Python, no DB access. Mirrors the per-row rules the writer has always
    applied; returns None for a record the writer skips.
    """
    observation_id = str(observation.get("observation_id", "")).strip()
    sequence = str(observation.get("sequence", "")).strip().upper()
    target = str(observation.get("target", "")).strip()
    try:
        score = float(observation["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not observation_id
        or not sequence
        or not target
        or not math.isfinite(score)
        or any(residue not in AA_INDEX for residue in sequence)
    ):
        return None
    provenance = {
        key: value for key, value in observation.items()
        if key not in _OBSERVATION_EXCLUDED
    }
    prepared: dict[str, Any] = {
        "target": target,
        "sequence": sequence,
        # observations row minus node_id (filled in after the id lookup).
        "observation": (
            observation_id,
            observation.get("epoch"),
            str(observation.get("source", "unknown")),
            str(observation.get("score_kind", "unknown")),
            score,
            int(observation.get("observation_count", 1)),
            json.dumps(observation.get("metrics") or {}, sort_keys=True),
            json.dumps(observation.get("pose_stability"), sort_keys=True)
            if observation.get("pose_stability") is not None else None,
            json.dumps(provenance, sort_keys=True),
        ),
        "artifact": None,
        "parent_sequence": None,
        "lineage": None,
    }
    structure = observation.get("structure")
    if isinstance(structure, dict):
        confidence_path = structure.get("confidence_path") or structure.get("path")
        writer_path = structure.get("writer_path") or structure.get("path")
        sample_selection = structure.get("sample_selection")
        prepared["artifact"] = (
            observation_id,
            str(confidence_path) if confidence_path else None,
            str(writer_path) if writer_path else None,
            json.dumps(structure, sort_keys=True),
            json.dumps(sample_selection, sort_keys=True)
            if isinstance(sample_selection, dict) else None,
        )
    parent_sequence = str(observation.get("parent_sequence", "")).strip().upper()
    if (
        parent_sequence
        and len(parent_sequence) == len(sequence)
        and parent_sequence != sequence
        and all(residue in AA_INDEX for residue in parent_sequence)
    ):
        lineage_metadata = {
            key: value for key, value in observation.items()
            if key not in _LINEAGE_EXCLUDED
        }
        supplied_mutations = observation.get("mutations")
        if isinstance(supplied_mutations, str):
            supplied_mutations = [item for item in supplied_mutations.split(";") if item]
        prepared["parent_sequence"] = parent_sequence
        # lineage_edges row minus (parent_node_id, child_node_id).
        prepared["lineage"] = (
            int(observation["epoch"]) if observation.get("epoch") is not None else -1,
            str(observation.get("source", "unknown")),
            str(observation.get("proposal_type", "unknown")),
            json.dumps(supplied_mutations or mutations(parent_sequence, sequence)),
            json.dumps(lineage_metadata, sort_keys=True),
        )
    return prepared


def _resolve_node_ids(
    connection: db.Connection, keys: Iterable[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    """Return ``{(target, sequence): nodes.id}``, inserting the missing nodes.

    Batched replacement for the old per-row ``INSERT ... ON DUPLICATE KEY
    UPDATE`` + ``SELECT id``. ``LAST_INSERT_ID()`` cannot replace the SELECT:
    after a multi-row insert it returns only the first generated id, and on
    the UPDATE branch of ``ON DUPLICATE KEY UPDATE`` it does not return the
    existing row's id at all. So ids are looked up by the unique key instead.

    1. One plain (non-locking) SELECT per <=1000 keys finds existing nodes.
       ``length``/``residue_vector`` are pure functions of ``sequence``, so the
       old upsert's UPDATE branch rewrote identical values; skipping it for
       existing rows changes no data and takes no row locks on them.
    2. The missing nodes go in one multi-row upsert, in first-encounter
       order (child, then its parent, row by row) -- the order the per-row
       loop inserted them, so new nodes get ids in the same relative order
       (``load_observations`` orders by ``n.id``).
    3. Their ids are read with ``FOR SHARE``, a locking read that sees the
       latest committed row. A plain read would use this transaction's
       REPEATABLE-READ snapshot and could miss a node another writer committed
       after that snapshot (our upsert then hit its UPDATE branch with
       identical values, which does not make the row visible to us).
    """
    ordered = list(dict.fromkeys(keys))  # de-duplicated, first-encounter order
    by_target: dict[str, list[str]] = defaultdict(list)
    for target, sequence in ordered:
        by_target[target].append(sequence)
    local = _is_local(connection)
    placeholder = "?" if local else "%s"
    node_upsert_sql = _NODE_UPSERT_SQL_SQLITE if local else _NODE_UPSERT_SQL

    def lookup(target: str, sequences: list[str], *, locking: bool) -> dict[tuple[str, str], int]:
        found: dict[tuple[str, str], int] = {}
        for start in range(0, len(sequences), _NODE_LOOKUP_BATCH):
            batch = sequences[start:start + _NODE_LOOKUP_BATCH]
            # SQLite has no FOR SHARE equivalent and doesn't need one here:
            # it's a single-writer-at-a-time database (WAL mode serialises
            # writers), so a committed INSERT from the missing-node upsert
            # just below is immediately visible to the very next plain SELECT
            # on this same connection -- there is no other writer's
            # in-flight, uncommitted row to race against.
            suffix = " FOR SHARE" if (locking and not local) else ""
            rows = connection.execute(
                f"SELECT id, sequence FROM nodes WHERE target={placeholder} "
                f"AND sequence IN ("
                + ",".join([placeholder] * len(batch)) + ")"
                + suffix,
                (target, *batch),
            ).fetchall()
            for node_id, sequence in rows:
                found[(target, str(sequence).upper())] = int(node_id)
        return found

    ids: dict[tuple[str, str], int] = {}
    for target, sequences in by_target.items():
        ids.update(lookup(target, sequences, locking=False))
    missing = [key for key in ordered if key not in ids]
    if missing:
        connection.executemany(
            node_upsert_sql,
            [
                (target, sequence, len(sequence),
                 bytes(AA_INDEX[residue] for residue in sequence))
                for target, sequence in missing
            ],
            max_stmt_length=_MAX_STATEMENT_BYTES,
        )
        missing_by_target: dict[str, list[str]] = defaultdict(list)
        for target, sequence in missing:
            missing_by_target[target].append(sequence)
        for target, sequences in missing_by_target.items():
            ids.update(lookup(target, sequences, locking=True))
    unresolved = [key for key in ordered if key not in ids]
    if unresolved:
        raise RuntimeError(
            f"node id lookup failed for {len(unresolved)} sequence(s), "
            f"e.g. {unresolved[0]}"
        )
    return ids


def _write_observation_chunk(
    connection: db.Connection, chunk: list[dict[str, Any]]
) -> None:
    """Write one chunk with a constant number of round-trips (~5-7)."""
    keys: list[tuple[str, str]] = []
    for row in chunk:
        keys.append((row["target"], row["sequence"]))
        if row["parent_sequence"]:
            keys.append((row["target"], row["parent_sequence"]))
    ids = _resolve_node_ids(connection, keys)
    local = _is_local(connection)
    observation_sql = _OBSERVATION_UPSERT_SQL_SQLITE if local else _OBSERVATION_UPSERT_SQL
    artifact_sql = _ARTIFACT_UPSERT_SQL_SQLITE if local else _ARTIFACT_UPSERT_SQL
    lineage_sql = _LINEAGE_UPSERT_SQL_SQLITE if local else _LINEAGE_UPSERT_SQL
    connection.executemany(
        observation_sql,
        [
            (row["observation"][0], ids[(row["target"], row["sequence"])],
             *row["observation"][1:])
            for row in chunk
        ],
        max_stmt_length=_MAX_STATEMENT_BYTES,
    )
    artifacts = [row["artifact"] for row in chunk if row["artifact"] is not None]
    if artifacts:
        connection.executemany(
            artifact_sql, artifacts, max_stmt_length=_MAX_STATEMENT_BYTES
        )
    lineage = [
        (
            ids[(row["target"], row["parent_sequence"])],
            ids[(row["target"], row["sequence"])],
            *row["lineage"],
        )
        for row in chunk if row["lineage"] is not None
    ]
    if lineage:
        connection.executemany(
            lineage_sql, lineage, max_stmt_length=_MAX_STATEMENT_BYTES
        )


@db.deadlock_retry()
def update_observation_store(
    path: Path,
    observations: Iterable[dict[str, Any]],
    commit_every: int = 1000,
    commit_interval: float = 5.0,
) -> int:
    """Transactionally upsert uniquely identified scored VHH observations.

    Rows are written in chunks of ``commit_every`` (default 1000), one
    transaction per chunk. Each chunk costs a constant ~5-7 network
    round-trips (node id lookup, missing-node upsert + locking re-read, one
    multi-row upsert each for observations / artifacts / lineage, commit)
    instead of 3-5 round-trips per row. Over the ~460 ms master->Hetzner
    tunnel the old per-row loop took ~1 s/row -- 10,000 rows was the measured
    9,600-10,400 s "bookkeeping" of epochs 25623/25627 (ARCHITECTURE_REDESIGN
    new finding 5). A 1000-row chunk holds its row locks for a few seconds,
    the same order as the old ``commit_interval`` bound, and its statements
    are a few MB, far below ``max_allowed_packet``. ``commit_interval`` is
    kept for call-site compatibility; a chunk is never split by time.

    Row semantics are unchanged: same tables, same ``ON DUPLICATE KEY
    UPDATE`` upserts keyed the same way, rows applied in input order within a
    statement (a later duplicate ``observation_id`` still wins), and the same
    per-row validation/skip rules. The upserts are idempotent, so an
    intermediate commit is safe to re-run.

    A MySQL deadlock (1213) rolls back the current chunk; the
    ``@db.deadlock_retry()`` decorator re-runs the entire function, so
    ``observations`` must be re-iterable. It is materialized up front because
    ``select_elite_parents`` passes a generator expression.
    """
    if not isinstance(observations, (list, tuple)):
        observations = list(observations)
    # Label registry (ARCHITECTURE_REDESIGN R3/P4): a score_kind names one
    # scale. Strict labels (multipass, closed legacy) refuse a wrong writer,
    # formula or source; the rest warn on stderr. Checked before any write, so
    # a refused batch writes nothing. A registry import/bug never blocks a
    # write; only a real LabelViolation does.
    try:
        from nanobody_opt import observation_labels
    except Exception as error:  # noqa: BLE001
        print(f"OBSERVATION_LABEL_REGISTRY_UNAVAILABLE {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
    else:
        try:
            observation_labels.check_observations(observations)
        except observation_labels.LabelViolation:
            raise
        except Exception as error:  # noqa: BLE001
            print(f"OBSERVATION_LABEL_CHECK_ERROR {type(error).__name__}: {error}",
                  file=sys.stderr, flush=True)
    prepared = [
        row for row in map(_prepare_observation, observations) if row is not None
    ]
    if not prepared:
        return 0
    chunk_size = commit_every if commit_every > 0 else 1000
    with _connect(path) as connection:
        for start in range(0, len(prepared), chunk_size):
            _write_observation_chunk(connection, prepared[start:start + chunk_size])
            connection.commit()
    return len(prepared)


# ---------------------------------------------------------------------------
# Fix 2 (2026-10-15): incremental per-(sequence, target, metric) aggregate.
#
# rank_exact._historical_observations used to re-run the four-way join and
# json.loads ~1.04M folds on every ranking cycle (~8 min before the pool even
# forks).  These helpers maintain ``exact_observation_aggregate`` -- SUM,
# SUM_SQ, COUNT per (sequence, target, metric) over the sealed local_exact_raw
# folds -- in the same transaction that rewrites ``exact_run_observations``, so
# rank_exact can read the numeric aggregate (cheap) instead of the raw folds.
# The aggregate is a derived cache of already-replicated observations +
# exact_runs and is deliberately NOT replicated to Hetzner.
# ---------------------------------------------------------------------------


def _aggregate_metric_values(blob: str) -> dict[str, float] | None:
    """One metrics_json blob -> finite per-metric floats, or None if skipped.

    Mirrors rank_exact._historical_observations' per-row validation exactly: a
    row contributes to the aggregate only when every METRIC is present and
    finite, which is precisely the set of folds the old full-pool read kept.
    """
    try:
        metrics = json.loads(blob)
    except (TypeError, ValueError):
        return None
    if not isinstance(metrics, dict) or not set(METRICS).issubset(metrics):
        return None
    try:
        values = {name: float(metrics[name]) for name in METRICS}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def _run_folds(connection: Any, target: str, run_id: str, p: str,
               ) -> list[tuple[str, dict[str, float]]]:
    """(sequence, metric values) for every valid fold of one exact run.

    Reads the run's ``exact_run_observations`` rows joined through the current
    ``observations`` / ``nodes`` rows, so a re-import that changed a fold's
    metrics (or dropped it) is reflected in the returned contribution.
    """
    rows = connection.execute(
        f"""
        SELECT n.sequence, o.metrics_json
        FROM exact_run_observations x
        JOIN observations o ON o.observation_id = x.observation_id
        JOIN nodes n ON n.id = o.node_id
        WHERE x.target={p} AND x.run_id={p} AND x.target=n.target
          AND o.source='local_exact_raw'
        """,
        (target, run_id),
    ).fetchall()
    folds: list[tuple[str, dict[str, float]]] = []
    for sequence, blob in rows:
        values = _aggregate_metric_values(blob)
        if values is not None:
            folds.append((str(sequence).strip().upper(), values))
    return folds


def _folds_net_delta(
    old_folds: list[tuple[str, dict[str, float]]],
    new_folds: list[tuple[str, dict[str, float]]],
) -> dict[tuple[str, str], tuple[float, float, int]]:
    """Net (sum, sum_sq, count) change per (sequence, metric) between two sets.

    Computed as a NET delta (new minus old) so an idempotent re-seal of an
    unchanged run produces an exactly-zero delta and no floating-point churn in
    the aggregate (subtract-then-add of the same float is not associative).
    """
    def accumulate(
        folds: list[tuple[str, dict[str, float]]],
    ) -> dict[tuple[str, str], list[float]]:
        acc: dict[tuple[str, str], list[float]] = {}
        for sequence, values in folds:
            for metric, value in values.items():
                row = acc.get((sequence, metric))
                if row is None:
                    acc[(sequence, metric)] = [value, value * value, 1.0]
                else:
                    row[0] += value
                    row[1] += value * value
                    row[2] += 1.0
        return acc

    old = accumulate(old_folds)
    new = accumulate(new_folds)
    delta: dict[tuple[str, str], tuple[float, float, int]] = {}
    for key in set(old) | set(new):
        o = old.get(key, [0.0, 0.0, 0.0])
        n = new.get(key, [0.0, 0.0, 0.0])
        dsum = n[0] - o[0]
        dsq = n[1] - o[1]
        dcount = int(round(n[2] - o[2]))
        if dcount or dsum or dsq:
            delta[key] = (dsum, dsq, dcount)
    return delta


def _apply_aggregate_delta(
    connection: Any, target: str,
    delta: dict[tuple[str, str], tuple[float, float, int]], p: str,
) -> None:
    """Add a net (sum, sum_sq, count) delta to the aggregate for `target`."""
    for (sequence, metric), (dsum, dsq, dcount) in delta.items():
        if dcount > 0:
            connection.execute(
                f"""
                INSERT INTO exact_observation_aggregate(
                    target, sequence, metric, sum_value, sum_sq, count_value
                ) VALUES({p}, {p}, {p}, {p}, {p}, {p})
                ON CONFLICT(target, sequence, metric) DO UPDATE SET
                    sum_value=sum_value + excluded.sum_value,
                    sum_sq=sum_sq + excluded.sum_sq,
                    count_value=count_value + excluded.count_value
                """,
                (target, sequence, metric, dsum, dsq, dcount),
            )
        else:
            connection.execute(
                f"""
                UPDATE exact_observation_aggregate
                SET sum_value=sum_value + {p},
                    sum_sq=sum_sq + {p},
                    count_value=count_value + {p}
                WHERE target={p} AND sequence={p} AND metric={p}
                """,
                (dsum, dsq, dcount, target, sequence, metric),
            )
            connection.execute(
                f"""
                DELETE FROM exact_observation_aggregate
                WHERE target={p} AND sequence={p} AND metric={p} AND count_value <= 0
                """,
                (target, sequence, metric),
            )


def _aggregate_ready(connection: Any) -> bool:
    """Only a backfilled, explicitly valid cache may be maintained or read."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='exact_observation_aggregate'").fetchone()
    if not exists:
        return False
    valid = connection.execute(
        "SELECT value FROM exact_observation_meta WHERE key='valid'").fetchone()
    return bool(valid and int(valid[0]) == 1)


def _bump_aggregate_meta(connection: Any) -> None:
    """Advance the change counter and pin the source MAX(rowid) watermark."""
    row = connection.execute(
        "SELECT MAX(rowid) FROM exact_run_observations"
    ).fetchone()
    max_rowid = int(row[0]) if row and row[0] is not None else 0
    connection.execute(
        """
        INSERT INTO exact_observation_meta(key, value)
        VALUES('change_counter', 1)
        ON CONFLICT(key) DO UPDATE SET value=value + 1
        """
    )
    connection.execute(
        """
        INSERT INTO exact_observation_meta(key, value)
        VALUES('source_max_rowid', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (max_rowid,),
    )


def rebuild_exact_observation_aggregate(path: Path, *, target: str) -> dict[str, int]:
    """One-time, idempotent backfill of the aggregate from existing history.

    Streams the same four-way join the old full-pool read used (sealed
    local_exact_raw folds), accumulates SUM/SUM_SQ/COUNT per (sequence, target,
    metric), then atomically replaces the aggregate table's rows for `target`
    and pins the change counter + source MAX(rowid) watermark.

    The scan runs on a read-only connection so the ~8-minute streaming read
    holds no write transaction (and cannot block or be blocked by the live
    producer/topup writers); the write phase is a short second transaction.
    Safe to re-run: the second pass overwrites the first with identical
    numbers.
    """
    import sqlite3 as _sqlite3

    stats: dict[tuple[str, str], list[float]] = {}
    seen: set[str] = set()
    folds = 0
    read_only = _sqlite3.connect(
        f"file:{path}?mode=ro", uri=True, isolation_level=None)
    try:
        read_only.execute("BEGIN")
        counter_row = read_only.execute("SELECT value FROM exact_observation_meta WHERE key='change_counter'").fetchone()
        snapshot_counter = int(counter_row[0]) if counter_row else 0
        snapshot_row = read_only.execute(
            "SELECT MAX(rowid) FROM exact_run_observations").fetchone()
        snapshot_max = int(snapshot_row[0]) if snapshot_row and snapshot_row[0] else 0
        rows = read_only.execute(
            """
            SELECT n.sequence, o.metrics_json
            FROM observations o
            JOIN nodes n ON n.id = o.node_id
            JOIN exact_run_observations x
              ON x.observation_id=o.observation_id AND x.target=n.target
            JOIN exact_runs r
              ON r.target=x.target AND r.run_id=x.run_id AND r.complete=1
            WHERE n.target=? AND o.source='local_exact_raw'
              AND o.metrics_json IS NOT NULL
            """,
            (target,),
        )
        for sequence, blob in rows:
            values = _aggregate_metric_values(blob)
            if values is None:
                continue
            seq = str(sequence).strip().upper()
            seen.add(seq)
            folds += 1
            for metric, value in values.items():
                row = stats.get((seq, metric))
                if row is None:
                    stats[(seq, metric)] = [value, value * value, 1.0]
                else:
                    row[0] += value
                    row[1] += value * value
                    row[2] += 1.0
        read_only.execute("COMMIT")
    finally:
        read_only.close()
    sequences = len(seen)
    with _connect(path) as connection:
        p = "?" if _is_local(connection) else "%s"
        live_row = connection.execute(
            "SELECT MAX(rowid) FROM exact_run_observations").fetchone()
        live_max = int(live_row[0]) if live_row and live_row[0] else 0
        counter_row = connection.execute("SELECT value FROM exact_observation_meta WHERE key='change_counter'").fetchone()
        live_counter = int(counter_row[0]) if counter_row else 0
        if live_max != snapshot_max or live_counter != snapshot_counter:
            raise RuntimeError(
                f"aggregate backfill raced a writer "
                f"(snapshot max rowid {snapshot_max}, live {live_max}); retry"
            )
        connection.execute(
            f"DELETE FROM exact_observation_aggregate WHERE target={p}", (target,)
        )
        connection.executemany(
            f"""
            INSERT INTO exact_observation_aggregate(
                target, sequence, metric, sum_value, sum_sq, count_value
            ) VALUES({p}, {p}, {p}, {p}, {p}, {p})
            """,
            (
                (target, seq, metric, row[0], row[1], int(round(row[2])))
                for (seq, metric), row in stats.items()
            ),
        )
        _bump_aggregate_meta(connection)
        connection.execute("INSERT INTO exact_observation_meta(key,value) VALUES('valid',1) ON CONFLICT(key) DO UPDATE SET value=1")
    return {"folds": folds, "sequences": sequences, "aggregate_rows": len(stats)}


@db.deadlock_retry()
def record_exact_run(
    path: Path,
    *,
    target: str,
    run_id: str,
    epoch: int | None,
    source_path: Path,
    file_sha256: str,
    scorer_protocol: str | None,
    observation_ids: Iterable[str],
) -> dict[str, Any]:
    """Seal one raw exact run only after every expected row is present.

    ``exact_runs.complete`` is the ingestion barrier used by publication.  A
    half-written import can leave useful observation rows behind, but it can
    never be mistaken for an authoritative run until a later retry seals it.
    """
    expected = list(dict.fromkeys(str(item) for item in observation_ids))
    with _connect(path) as connection:
        local = _is_local(connection)
        p = "?" if local else "%s"
        # Fix 2: capture the run's pre-update state so the aggregate delta is
        # computed against what the aggregate actually held.  ``old_complete``
        # must be read before the exact_runs UPSERT and ``old_folds`` before
        # the exact_run_observations DELETE below.
        old_complete = 0
        old_folds: list[tuple[str, dict[str, float]]] = []
        aggregate_ready = local and _aggregate_ready(connection)
        if aggregate_ready:
            old_row = connection.execute(
                f"SELECT complete FROM exact_runs WHERE target={p} AND run_id={p}",
                (target, run_id),
            ).fetchone()
            old_complete = int(old_row[0]) if old_row else 0
            if old_complete:
                old_folds = _run_folds(connection, target, run_id, p)
        present_ids = {
            str(row[0])
            for observation_id in expected
            for row in connection.execute(
                f"""
                SELECT o.observation_id
                FROM observations o
                JOIN nodes n ON n.id=o.node_id
                WHERE o.observation_id={p} AND n.target={p}
                """,
                (observation_id, target),
            ).fetchall()
        }
        present = len(present_ids)
        complete = int(present == len(expected) and bool(expected))
        if local:
            connection.execute(
                """
                INSERT INTO exact_runs(
                    target, run_id, epoch, source_path, file_sha256,
                    scorer_protocol, expected_rows, ingested_rows, complete,
                    ingested_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target, run_id) DO UPDATE SET
                    epoch=excluded.epoch,
                    source_path=excluded.source_path,
                    file_sha256=excluded.file_sha256,
                    scorer_protocol=excluded.scorer_protocol,
                    expected_rows=excluded.expected_rows,
                    ingested_rows=excluded.ingested_rows,
                    complete=excluded.complete,
                    ingested_at=excluded.ingested_at
                """,
                (
                    target, run_id, epoch, str(source_path), file_sha256,
                    scorer_protocol, len(expected), present, complete, time.time(),
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO exact_runs(
                    target, run_id, epoch, source_path, file_sha256,
                    scorer_protocol, expected_rows, ingested_rows, complete,
                    ingested_at
                ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    epoch=VALUES(epoch),
                    source_path=VALUES(source_path),
                    file_sha256=VALUES(file_sha256),
                    scorer_protocol=VALUES(scorer_protocol),
                    expected_rows=VALUES(expected_rows),
                    ingested_rows=VALUES(ingested_rows),
                    complete=VALUES(complete),
                    ingested_at=VALUES(ingested_at)
                """,
                (
                    target, run_id, epoch, str(source_path), file_sha256,
                    scorer_protocol, len(expected), present, complete, time.time(),
                ),
            )
        connection.execute(
            f"DELETE FROM exact_run_observations WHERE target={p} AND run_id={p}",
            (target, run_id),
        )
        connection.executemany(
            f"""
            INSERT INTO exact_run_observations(target, run_id, observation_id)
            VALUES({p}, {p}, {p})
            """,
            ((target, run_id, observation_id) for observation_id in present_ids),
        )
        # Fix 2: reflect this run's net contribution in the aggregate, in the
        # same transaction as the exact_run_observations rewrite, so a reader
        # never sees the two out of sync.  Net-delta (not subtract-then-add)
        # keeps an idempotent re-seal from drifting the floats.
        if aggregate_ready:
            new_folds = _run_folds(connection, target, run_id, p) if complete else []
            delta = _folds_net_delta(old_folds, new_folds)
            if delta:
                _apply_aggregate_delta(connection, target, delta, p)
        if local and connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='exact_observation_meta'").fetchone():
            _bump_aggregate_meta(connection)
    return {
        "target": target,
        "run_id": run_id,
        "expected_rows": len(expected),
        "ingested_rows": present,
        "complete": bool(complete),
    }


def exact_run_is_complete(
    path: Path,
    *,
    target: str,
    run_id: str,
    file_sha256: str,
    expected_rows: int,
) -> bool:
    """Whether the database holds a sealed, byte-matched raw exact run.

    Reads through _connect(path), same as the writer (record_exact_run), so
    under NOVA_LOCAL_SQLITE=1 this checks the barrier the producer's own
    write just sealed -- not a MySQL replica that may still be a few seconds
    behind it.
    """
    try:
        with _connect(path) as connection:
            p = "?" if _is_local(connection) else "%s"
            row = connection.execute(
                f"""
                SELECT r.file_sha256, r.expected_rows, r.ingested_rows, r.complete,
                       COUNT(x.observation_id)
                FROM exact_runs r
                LEFT JOIN exact_run_observations x
                  ON x.target=r.target AND x.run_id=r.run_id
                WHERE r.target={p} AND r.run_id={p}
                GROUP BY r.target, r.run_id
                """,
                (target, run_id),
            ).fetchone()
    except Exception:
        return False
    return bool(
        row
        and str(row[0]) == file_sha256
        and int(row[1]) == expected_rows
        and int(row[2]) == expected_rows
        and int(row[3]) == 1
        and int(row[4]) == expected_rows
    )


def load_exact_artifacts(
    path: Path, *, target: str, sequences: Iterable[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load retained structure/sample metadata for authoritative raw folds."""
    try:
        with _connect(path) as connection:
            p = "?" if _is_local(connection) else "%s"
            wanted = sorted(set(sequences)) if sequences is not None else None
            batches = [None] if wanted is None else [wanted[i:i + 500] for i in range(0, len(wanted), 500)]
            rows = []
            for batch in batches:
                restriction = "" if batch is None else " AND n.sequence IN (" + ",".join([p] * len(batch)) + ")"
                rows.extend(connection.execute(
                f"""
                SELECT n.sequence, o.observation_id, x.run_id, a.confidence_path,
                       a.writer_path, a.structure_json, a.sample_selection_json
                FROM nodes n CROSS JOIN observations o ON n.id=o.node_id
                JOIN exact_run_observations x
                  ON x.observation_id=o.observation_id AND x.target=n.target
                JOIN exact_runs r
                  ON r.target=x.target AND r.run_id=x.run_id AND r.complete=1
                JOIN observation_artifacts a ON a.observation_id=o.observation_id
                WHERE n.target={p} AND o.source='local_exact_raw'{restriction}
                ORDER BY o.observation_id
                """,
                (target, *(batch or [])),
                ).fetchall())
    except Exception:
        return {}
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (
        sequence, observation_id, run_id, confidence, writer, structure, selection
    ) in rows:
        try:
            structure_payload = json.loads(structure or "{}")
        except (TypeError, ValueError):
            structure_payload = {}
        try:
            selection_payload = json.loads(selection) if selection else None
        except (TypeError, ValueError):
            selection_payload = None
        found[str(sequence)].append({
            "observation_id": str(observation_id),
            "run_id": str(run_id),
            "confidence_path": confidence,
            "writer_path": writer,
            "structure": structure_payload,
            "sample_selection": selection_payload,
        })
    return dict(found)


def load_observations(path: Path, *, target: str) -> list[dict[str, Any]]:
    """Every observation for one target: a full-table scan plus a JSON decode
    of three columns per row (~392k rows and counting). Each producer
    iteration is a fresh subprocess and the epoch runs several of them
    (2-3 in-epoch, up to 2 more prescreening the next epoch), so this ran
    from scratch every time although the table barely changes between them
    -- measured 22% of a 239s generation. Cached on disk, keyed on the
    database's own (mtime, size): a real change is a miss, a rerun on an
    unchanged database is not.
    """
    return _cached_observations(
        path.with_name(f"{path.name}.obs_cache.{target}.pkl"),
        target=target,
        load=lambda: _load_observations_uncached(path, target=target),
        source_path=path,
    )


def _cached_observations(
    cache_path: Path, *, target: str, load: Any, source_path: Path | None = None
) -> list[dict[str, Any]]:
    """Serve ``load()`` from a pickle keyed on the target's (row count, max
    observation id); any write to the target's observations is a miss.

    ``source_path``, when given, makes the cache-key probe read through the
    SAME connection ``load()`` actually reads from (_connect(source_path):
    local SQLite under NOVA_LOCAL_SQLITE=1, MySQL otherwise). Without it the
    probe always hit MySQL -- correct for load_network_observations, whose
    _load_network_observations_uncached also always reads MySQL by design
    (see its docstring), but WRONG for load_observations once its own read
    moved to local SQLite: a fresh local write not yet replicated wouldn't
    change what MySQL reports, so the cache key wouldn't change either, and
    a stale pickle would be served past the write that should have missed it.
    """
    if source_path is not None:
        with _connect(source_path) as probe:
            p = "?" if _is_local(probe) else "%s"
            cache_key = probe.execute(
                f"""
                SELECT COUNT(*), COALESCE(MAX(o.observation_id), '')
                FROM observations o JOIN nodes n ON n.id=o.node_id
                WHERE n.target={p}
                """,
                (target,),
            ).fetchone()
    else:
        raise ValueError("_cached_observations needs source_path (local store)")
    try:
        with cache_path.open("rb") as handle:
            cached_key, cached_records = pickle.load(handle)
        if tuple(cached_key) == tuple(cache_key):
            return cached_records
    except (OSError, EOFError, pickle.UnpicklingError, ValueError):
        pass
    records = load()
    try:
        handle_fd, temporary = tempfile.mkstemp(
            prefix=".obs_cache_", suffix=".pkl", dir=cache_path.parent
        )
        with os.fdopen(handle_fd, "wb") as handle:
            pickle.dump((cache_key, records), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    except OSError:
        pass  # a cache that cannot write must not fail generation
    return records


def _load_observations_uncached(path: Path, *, target: str) -> list[dict[str, Any]]:
    with _connect(path) as connection:
        p = "?" if _is_local(connection) else "%s"
        rows = connection.execute(
            f"""
            SELECT n.sequence, o.score, o.epoch, o.source, o.score_kind,
                   o.observation_count, o.metrics_json, o.pose_json,
                   o.provenance_json
            FROM observations o
            JOIN nodes n ON n.id=o.node_id
            WHERE n.target={p}
            ORDER BY o.epoch, o.observation_id
            """,
            (target,),
        ).fetchall()
    return [
        {
            "sequence": row[0],
            "score": row[1],
            "epoch": row[2],
            "source": row[3],
            "score_kind": row[4],
            "observation_count": row[5],
            "metrics": json.loads(row[6]),
            "pose_stability": json.loads(row[7]) if row[7] else None,
            "provenance": json.loads(row[8]),
        }
        for row in rows
    ]


def load_network_observations(
    path: Path, *, target: str, length: int
) -> list[dict[str, Any]]:
    """The observations a MutationNetwork / terminal-prior build reads, and
    only those: one target, one sequence length, and only the columns those
    consumers use (sequence, score, epoch, source, score_kind,
    observation_count, pose_stability).

    ``load_observations`` returns every column of every row and orders the
    result server side. On the central MySQL store that is ~3.9M rows for
    P20809, ~3.6 GB of it ``provenance_json`` on ``local_exact_raw`` rows that
    no generator reads; the ORDER BY makes the server filesort all of it
    before sending the first byte (measured 182 s to first row), which
    exceeds ``db.connect``'s 60 s ``read_timeout`` -- every live producer
    iteration died in ``local_neighborhood`` with MySQL error 2013.

    Here the rows are streamed unordered on a server-side cursor (first row
    in well under a second; ~76 s and 3.1 GB RSS end to end for the whole
    target on master, 2026-10-09) and ordered by epoch client side. Row scope is unchanged on purpose: the
    network's per-sequence robust score, source/epoch sets and contact
    union are built from every historical observation, so a recency window
    would change the generator's evidence, not just its cost.
    """
    return _cached_observations(
        path.with_name(f"{path.name}.netobs_cache.{target}.{length}.pkl"),
        target=target,
        load=lambda: _load_network_observations_uncached(
            path, target=target, length=length
        ),
        source_path=path,
    )


def _load_network_observations_uncached(
    path: Path, *, target: str, length: int
) -> list[dict[str, Any]]:
    with _connect(path) as connection:
        sequences = {
            int(node_id): str(sequence)
            for node_id, sequence in connection.execute(
                "SELECT id, sequence FROM nodes WHERE target=? AND length=?",
                (target, length),
            ).fetchall()
        }
        rows = connection.execute(
            """
            SELECT o.node_id, o.score, o.epoch, o.source, o.score_kind,
                   o.observation_count, o.pose_json
            FROM observations o
            JOIN nodes n ON n.id=o.node_id
            WHERE n.target=? AND n.length=?
            """,
            (target, length),
        ).fetchall()
    # pose_json repeats heavily (~11k distinct strings over ~3.9M rows on
    # 2026-10-09); decode each once. The decoded dicts are shared, read-only.
    poses: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for node_id, score, epoch, source, score_kind, count, pose_json in rows:
        sequence = sequences.get(int(node_id))
        if sequence is None or len(sequence) != length:
            continue  # node inserted after the node read; picked up next time
        if pose_json:
            if pose_json not in poses:
                poses[pose_json] = json.loads(pose_json)
            pose = poses[pose_json]
        else:
            pose = None
        records.append({
            "sequence": sequence,
            "score": score,
            "epoch": epoch,
            "source": source,
            "score_kind": score_kind,
            "observation_count": count,
            "pose_stability": pose,
        })
    # Same coarse order as load_observations (MySQL sorts NULL epochs first).
    records.sort(key=lambda item: (
        item["epoch"] is not None,
        item["epoch"] if item["epoch"] is not None else 0,
    ))
    return records


def load_lineage_edges(path: Path, *, target: str) -> list[dict[str, Any]]:
    """Load observed proposal parent-child relationships without inference."""
    with _connect(path) as connection:
        p = "?" if _is_local(connection) else "%s"
        rows = connection.execute(
            f"""
            SELECT parent.sequence, child.sequence, edge.epoch, edge.source,
                   edge.proposal_type, edge.mutations_json, edge.metadata_json
            FROM lineage_edges edge
            JOIN nodes parent ON parent.id=edge.parent_node_id
            JOIN nodes child ON child.id=edge.child_node_id
            WHERE child.target={p}
            ORDER BY edge.epoch, parent.sequence, child.sequence
            """,
            (target,),
        ).fetchall()
    return [
        {
            "parent_sequence": row[0],
            "sequence": row[1],
            "epoch": row[2] if row[2] >= 0 else None,
            "source": row[3],
            "proposal_type": row[4],
            "mutations": json.loads(row[5]),
            "metadata": json.loads(row[6]),
        }
        for row in rows
    ]


def load_mutation_effects(path: Path, *, target: str) -> list[dict[str, Any]]:
    """Load the latest complete directional-effect snapshot for a target.

    mutation_effects/metadata are NOT replicated to Hetzner (a derived
    snapshot rebuild_nearest_edges recomputes locally from nodes +
    observations; pushing it would overwrite the graph_version pointer other
    boxes read -- see replicate_local.py). Under NOVA_LOCAL_SQLITE=1 this is
    local-only data, so a fresh box or one whose rebuild hasn't run yet
    legitimately has none: the empty-row path below is not an error.
    """
    with _connect(path) as connection:
        p = "?" if _is_local(connection) else "%s"
        row = connection.execute(
            f"SELECT value FROM metadata WHERE `key`={p}",
            (f"graph_version:{target}",),
        ).fetchone()
        if row is None:
            return []
        graph_version = str(row[0])
        rows = connection.execute(
            f"""
            SELECT position, from_residue, to_residue, region, contact_class,
                   support, median_score_delta, uncertainty, graph_version
            FROM mutation_effects
            WHERE target={p} AND graph_version={p}
            """,
            (target, graph_version),
        ).fetchall()
    return [
        {
            "position": row[0],
            "from_residue": row[1],
            "to_residue": row[2],
            "region": row[3],
            "contact_class": row[4],
            "support": row[5],
            "median_score_delta": row[6],
            "uncertainty": row[7],
            "graph_version": row[8],
        }
        for row in rows
    ]


def _robust_node_scores(
    connection: "db.Connection | db.LocalConnection",
    *,
    target: str,
    sequence_length: int,
) -> tuple[list[int], list[str], np.ndarray, dict[str, float]]:
    p = "?" if _is_local(connection) else "%s"
    rows = connection.execute(
        f"""
        SELECT n.id, n.sequence, n.residue_vector, o.score, o.score_kind,
               o.epoch, o.source, o.observation_count
        FROM nodes n
        JOIN observations o ON o.node_id=n.id
        WHERE n.target={p} AND n.length={p}
        ORDER BY n.id, o.observation_id
        """,
        (target, sequence_length),
    ).fetchall()
    records = [
        {
            "sequence": row[1],
            "score": row[3],
            "score_kind": row[4],
            "epoch": row[5],
            "source": row[6],
            "observation_count": row[7],
        }
        for row in rows
    ]
    if not records:
        raise ValueError(f"mutation database has no observations for {target}")
    network = MutationNetwork(records)
    score_by_sequence = {node.sequence: node.score for node in network.nodes}
    first_by_id: dict[int, tuple[str, bytes]] = {}
    for node_id, sequence, vector, *_ in rows:
        first_by_id.setdefault(int(node_id), (str(sequence), bytes(vector)))
    node_ids = sorted(first_by_id)
    sequences = [first_by_id[node_id][0] for node_id in node_ids]
    vectors = np.frombuffer(
        b"".join(first_by_id[node_id][1] for node_id in node_ids),
        dtype=np.uint8,
    ).reshape(len(node_ids), sequence_length)
    return node_ids, sequences, vectors, score_by_sequence


@db.deadlock_retry()
def rebuild_nearest_edges(
    path: Path,
    *,
    target: str,
    edge_limit: int = 16,
    maximum_distance: int = 6,
    workers: int = 16,
) -> dict[str, Any]:
    """Rebuild a bounded nearest-neighbour graph and empirical mutation effects."""
    if edge_limit < 1 or maximum_distance < 1 or workers < 1:
        raise ValueError("invalid nearest-edge rebuild configuration")
    with _connect(path) as connection:
        p = "?" if _is_local(connection) else "%s"
        edge_rows: dict[tuple[int, int], tuple[int, int, int, str, str]] = {}
        scores: dict[str, float] = {}
        contact_rows = connection.execute(
            f"""
            SELECT n.id, o.pose_json FROM nodes n
            JOIN observations o ON o.node_id=n.id
            WHERE n.target={p} AND o.pose_json IS NOT NULL
            """,
            (target,),
        ).fetchall()
        contacts_by_node_id: dict[int, set[int]] = defaultdict(set)
        for node_id, pose_json in contact_rows:
            pose = json.loads(pose_json)
            contacts_by_node_id[int(node_id)].update(
                int(position) - 1
                for position in pose.get(
                    "consensus_nanobody_contact_positions", []
                )
                if int(position) >= 1
            )
        lengths = connection.execute(
            f"""
            SELECT length, COUNT(*) FROM nodes
            WHERE target={p} GROUP BY length ORDER BY length
            """,
            (target,),
        ).fetchall()
        if not lengths:
            raise ValueError(f"mutation database has no observations for {target}")
        node_count = 0
        for sequence_length, _ in lengths:
            node_ids, sequences, vectors, partition_scores = _robust_node_scores(
                connection,
                target=target,
                sequence_length=int(sequence_length),
            )
            node_count += len(node_ids)
            scores.update(partition_scores)
            neighbour_count = min(edge_limit + 1, len(node_ids))

            # Each worker handles independent query rows. NumPy releases the
            # GIL in the comparison/count, while the query-row design bounds
            # peak RAM even for a large historical bank.
            def nearest(index: int) -> list[tuple[int, int]]:
                distances = np.count_nonzero(vectors != vectors[index], axis=1)
                if neighbour_count == len(node_ids):
                    candidates = np.arange(len(node_ids))
                else:
                    candidates = np.argpartition(
                        distances, neighbour_count - 1
                    )[:neighbour_count]
                ordered = sorted(
                    (
                        (int(distances[other]), int(other))
                        for other in candidates
                        if other != index
                        and int(distances[other]) <= maximum_distance
                    ),
                    key=lambda item: (item[0], sequences[item[1]]),
                )
                return ordered[:edge_limit]

            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(nearest, range(len(node_ids)))
                for index, neighbours in enumerate(results):
                    for distance, other in neighbours:
                        left_index, right_index = sorted((index, other))
                        key = (node_ids[left_index], node_ids[right_index])
                        left = sequences[left_index]
                        right = sequences[right_index]
                        edge_rows[key] = (
                            key[0],
                            key[1],
                            distance,
                            json.dumps(mutations(left, right)),
                            json.dumps(
                                mutation_contexts(
                                    left,
                                    right,
                                    contact_positions=(
                                        contacts_by_node_id[key[0]]
                                        | contacts_by_node_id[key[1]]
                                    ),
                                ),
                                sort_keys=True,
                            ),
                        )

        connection.execute(
            f"""
            DELETE FROM edges
            WHERE source_node_id IN (SELECT id FROM nodes WHERE target={p})
            """,
            (target,),
        )
        connection.executemany(
            f"""
            INSERT INTO edges(
                source_node_id, target_node_id, distance,
                mutations_json, context_json
            ) VALUES({p}, {p}, {p}, {p}, {p})
            """,
            edge_rows.values(),
        )

        observation_rows = connection.execute(
            f"""
            SELECT n.id, o.epoch, o.score_kind, o.source, o.score
            FROM nodes n
            JOIN observations o ON o.node_id=n.id
            WHERE n.target={p} AND o.epoch IS NOT NULL
            """,
            (target,),
        ).fetchall()
        matched_scores: dict[
            int, dict[tuple[int, str, str], list[float]]
        ] = defaultdict(lambda: defaultdict(list))
        for node_id, epoch, score_kind, source, score in observation_rows:
            matched_scores[int(node_id)][
                (int(epoch), str(score_kind), str(source))
            ].append(float(score))

        graph_material = "\n".join(
            f"{sequence}\t{scores[sequence]:.8f}"
            for sequence in sorted(scores)
        )
        import hashlib

        graph_version = hashlib.sha256(graph_material.encode()).hexdigest()[:16]
        effects: dict[tuple[int, str, str, str, str], list[float]] = defaultdict(list)
        for source_id, target_id, distance, _, context_json in edge_rows.values():
            if distance != 1:
                continue
            context = json.loads(context_json)[0]
            forward = (
                int(context["position"]),
                str(context["from_residue"]),
                str(context["to_residue"]),
                str(context["region"]),
                str(context["contact_class"]),
            )
            reverse = (forward[0], forward[2], forward[1], forward[3], forward[4])
            common = (
                matched_scores[source_id].keys()
                & matched_scores[target_id].keys()
            )
            for comparison in common:
                source_score = float(np.mean(
                    matched_scores[source_id][comparison]
                ))
                target_score = float(np.mean(
                    matched_scores[target_id][comparison]
                ))
                delta = target_score - source_score
                effects[forward].append(delta)
                effects[reverse].append(-delta)
        connection.execute(
            f"DELETE FROM mutation_effects WHERE target={p} AND graph_version={p}",
            (target, graph_version),
        )
        effect_rows = []
        for key, deltas in effects.items():
            median = float(np.median(deltas))
            uncertainty = float(np.median(np.abs(np.asarray(deltas) - median)))
            effect_rows.append((
                target, *key, len(deltas), median, uncertainty, graph_version
            ))
        connection.executemany(
            f"""
            INSERT INTO mutation_effects(
                target, position, from_residue, to_residue, region,
                contact_class, support, median_score_delta, uncertainty,
                graph_version
            ) VALUES({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})
            """,
            effect_rows,
        )
        connection.execute(
            f"REPLACE INTO metadata(`key`, value) VALUES({p}, {p})",
            (f"graph_version:{target}", graph_version),
        )
    return {
        "target": target,
        "graph_version": graph_version,
        "node_count": node_count,
        "edge_count": len(edge_rows),
        "mutation_effect_count": len(effect_rows),
    }


def ranking_observations(
    ranking: Path,
    *,
    epoch: int,
    target: str,
    source: str,
) -> list[dict[str, Any]]:
    """Convert a local exact ranking into provenance-preserving graph records."""
    payload = json.loads(ranking.read_text())
    observations = []
    for candidate in payload.get("candidates", []):
        sequence = str(candidate.get("sequence", "")).strip().upper()
        score = candidate.get("selection_score")
        if not sequence or score is None:
            continue
        observations.append({
            # This is a mutable derived snapshot, not an independent scorer
            # draw.  Its identity must therefore be stable as evidence depth
            # changes within an epoch.  Including observation_count here grew
            # production to >500k near-duplicate rows and was the reason the
            # raw/derived distinction became impossible to reason about.
            "observation_id": f"{source}:epoch{epoch}:{sequence}",
            "sequence": sequence,
            "score": float(score),
            "score_kind": "local_validator_replay_rank_sum",
            "source": source,
            "epoch": epoch,
            "target": target,
            "observation_count": int(candidate.get("observation_count", 0)),
            "score_median": candidate.get("score_median"),
            "score_p90": candidate.get("score_p90"),
            "metrics": candidate.get("metrics") or candidate.get("metric_medians", {}),
            "pose_stability": candidate.get("pose_stability"),
        })
    return observations
