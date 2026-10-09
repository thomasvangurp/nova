"""Scored vector index and mutation graph for historical VHH sequences."""
from __future__ import annotations

import json
import math
import pickle
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


def _connect(path: Path | None = None) -> db.Connection:
    """Open the central MySQL connection.

    ``path`` is retained for call-site compatibility with the previous
    sqlite3 backend and is ignored -- all nodes share the one ``sn68`` schema
    on Hetzner.
    """
    return db.connect()


@db.deadlock_retry()
def update_observation_store(
    path: Path,
    observations: Iterable[dict[str, Any]],
    commit_every: int = 1000,
    commit_interval: float = 5.0,
) -> int:
    """Transactionally upsert uniquely identified scored VHH observations.

    Commits every ``commit_every`` writes or ``commit_interval`` seconds
    (whichever comes first) so a slow/lazy ``observations`` iterable can never
    hold an open transaction -- and its row locks -- for minutes and stall
    every other writer.  The upserts are idempotent (``ON DUPLICATE KEY
    UPDATE``), so an intermediate commit is safe to re-run.

    A MySQL deadlock (1213) rolls back this whole transaction; the
    ``@db.deadlock_retry()`` decorator re-runs the entire function, so
    ``observations`` must be re-iterable. It is materialized up front because
    ``select_elite_parents`` passes a generator expression.
    """
    if not isinstance(observations, (list, tuple)):
        observations = list(observations)
    written = 0
    last_commit = time.monotonic()
    with _connect(path) as connection:
        for observation in observations:
            observation_id = str(observation.get("observation_id", "")).strip()
            sequence = str(observation.get("sequence", "")).strip().upper()
            target = str(observation.get("target", "")).strip()
            try:
                score = float(observation["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                not observation_id
                or not sequence
                or not target
                or not math.isfinite(score)
                or any(residue not in AA_INDEX for residue in sequence)
            ):
                continue
            vector = bytes(AA_INDEX[residue] for residue in sequence)
            connection.execute(
                """
                INSERT INTO nodes(target, sequence, length, residue_vector)
                VALUES(%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    length=VALUES(length),
                    residue_vector=VALUES(residue_vector)
                """,
                (target, sequence, len(sequence), vector),
            )
            node_id = int(connection.execute(
                "SELECT id FROM nodes WHERE target=%s AND sequence=%s",
                (target, sequence),
            ).fetchone()[0])
            excluded = {
                "observation_id", "sequence", "target", "epoch", "source",
                "score_kind", "score", "observation_count", "metrics",
                "pose_stability", "structure",
            }
            provenance = {
                key: value for key, value in observation.items() if key not in excluded
            }
            connection.execute(
                """
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
                """,
                (
                    observation_id,
                    node_id,
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
            )
            structure = observation.get("structure")
            if isinstance(structure, dict):
                confidence_path = (
                    structure.get("confidence_path") or structure.get("path")
                )
                writer_path = structure.get("writer_path") or structure.get("path")
                sample_selection = structure.get("sample_selection")
                connection.execute(
                    """
                    INSERT INTO observation_artifacts(
                        observation_id, confidence_path, writer_path,
                        structure_json, sample_selection_json
                    ) VALUES(%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        confidence_path=VALUES(confidence_path),
                        writer_path=VALUES(writer_path),
                        structure_json=VALUES(structure_json),
                        sample_selection_json=VALUES(sample_selection_json)
                    """,
                    (
                        observation_id,
                        str(confidence_path) if confidence_path else None,
                        str(writer_path) if writer_path else None,
                        json.dumps(structure, sort_keys=True),
                        json.dumps(sample_selection, sort_keys=True)
                        if isinstance(sample_selection, dict) else None,
                    ),
                )
            parent_sequence = str(
                observation.get("parent_sequence", "")
            ).strip().upper()
            if (
                parent_sequence
                and len(parent_sequence) == len(sequence)
                and parent_sequence != sequence
                and all(residue in AA_INDEX for residue in parent_sequence)
            ):
                parent_vector = bytes(AA_INDEX[residue] for residue in parent_sequence)
                connection.execute(
                    """
                    INSERT INTO nodes(target, sequence, length, residue_vector)
                    VALUES(%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        length=VALUES(length),
                        residue_vector=VALUES(residue_vector)
                    """,
                    (target, parent_sequence, len(parent_sequence), parent_vector),
                )
                parent_node_id = int(connection.execute(
                    "SELECT id FROM nodes WHERE target=%s AND sequence=%s",
                    (target, parent_sequence),
                ).fetchone()[0])
                lineage_excluded = {
                    "observation_id", "sequence", "parent_sequence", "target",
                    "epoch", "source", "proposal_type", "mutations",
                    "score_kind", "score", "observation_count", "metrics",
                    "pose_stability",
                }
                lineage_metadata = {
                    key: value
                    for key, value in observation.items()
                    if key not in lineage_excluded
                }
                supplied_mutations = observation.get("mutations")
                if isinstance(supplied_mutations, str):
                    supplied_mutations = [
                        item for item in supplied_mutations.split(";") if item
                    ]
                connection.execute(
                    """
                    INSERT INTO lineage_edges(
                        parent_node_id, child_node_id, epoch, source,
                        proposal_type, mutations_json, metadata_json
                    ) VALUES(%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        proposal_type=VALUES(proposal_type),
                        mutations_json=VALUES(mutations_json),
                        metadata_json=VALUES(metadata_json)
                    """,
                    (
                        parent_node_id,
                        node_id,
                        (
                            int(observation["epoch"])
                            if observation.get("epoch") is not None
                            else -1
                        ),
                        str(observation.get("source", "unknown")),
                        str(observation.get("proposal_type", "unknown")),
                        json.dumps(
                            supplied_mutations or mutations(parent_sequence, sequence)
                        ),
                        json.dumps(lineage_metadata, sort_keys=True),
                    ),
                )
            written += 1
            now = time.monotonic()
            if (commit_every > 0 and written % commit_every == 0) or (
                now - last_commit
            ) >= commit_interval:
                connection.commit()
                last_commit = now
    return written


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
        present_ids = {
            str(row[0])
            for observation_id in expected
            for row in connection.execute(
                """
                SELECT o.observation_id
                FROM observations o
                JOIN nodes n ON n.id=o.node_id
                WHERE o.observation_id=%s AND n.target=%s
                """,
                (observation_id, target),
            )
        }
        present = len(present_ids)
        complete = int(present == len(expected) and bool(expected))
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
            "DELETE FROM exact_run_observations WHERE target=%s AND run_id=%s",
            (target, run_id),
        )
        connection.executemany(
            """
            INSERT INTO exact_run_observations(target, run_id, observation_id)
            VALUES(%s, %s, %s)
            """,
            ((target, run_id, observation_id) for observation_id in present_ids),
        )
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
    """Whether the database holds a sealed, byte-matched raw exact run."""
    try:
        with db.connect() as connection:
            row = connection.execute(
                """
                SELECT r.file_sha256, r.expected_rows, r.ingested_rows, r.complete,
                       COUNT(x.observation_id)
                FROM exact_runs r
                LEFT JOIN exact_run_observations x
                  ON x.target=r.target AND x.run_id=r.run_id
                WHERE r.target=%s AND r.run_id=%s
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
    path: Path, *, target: str
) -> dict[str, list[dict[str, Any]]]:
    """Load retained structure/sample metadata for authoritative raw folds."""
    try:
        with db.connect() as connection:
            rows = connection.execute(
                """
                SELECT n.sequence, o.observation_id, x.run_id, a.confidence_path,
                       a.writer_path, a.structure_json, a.sample_selection_json
                FROM observations o
                JOIN nodes n ON n.id=o.node_id
                JOIN exact_run_observations x
                  ON x.observation_id=o.observation_id AND x.target=n.target
                JOIN exact_runs r
                  ON r.target=x.target AND r.run_id=x.run_id AND r.complete=1
                JOIN observation_artifacts a ON a.observation_id=o.observation_id
                WHERE n.target=%s AND o.source='local_exact_raw'
                ORDER BY o.observation_id
                """,
                (target,),
            ).fetchall()
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
    )


def _cached_observations(
    cache_path: Path, *, target: str, load: Any
) -> list[dict[str, Any]]:
    """Serve ``load()`` from a pickle keyed on the target's (row count, max
    observation id); any write to the target's observations is a miss."""
    with db.connect() as probe:
        cache_key = probe.execute(
            """
            SELECT COUNT(*), COALESCE(MAX(o.observation_id), '')
            FROM observations o JOIN nodes n ON n.id=o.node_id
            WHERE n.target=%s
            """,
            (target,),
        ).fetchone()
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
        rows = connection.execute(
            """
            SELECT n.sequence, o.score, o.epoch, o.source, o.score_kind,
                   o.observation_count, o.metrics_json, o.pose_json,
                   o.provenance_json
            FROM observations o
            JOIN nodes n ON n.id=o.node_id
            WHERE n.target=%s
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
            target=target, length=length
        ),
    )


def _load_network_observations_uncached(
    *, target: str, length: int
) -> list[dict[str, Any]]:
    with db.connect() as connection:
        sequences = {
            int(node_id): str(sequence)
            for node_id, sequence in connection.execute(
                "SELECT id, sequence FROM nodes WHERE target=%s AND length=%s",
                (target, length),
            ).fetchall()
        }
        # Drain the stream into plain tuples first: the server drops an
        # unbuffered result whose client stops reading for net_write_timeout.
        rows = list(connection.stream(
            """
            SELECT o.node_id, o.score, o.epoch, o.source, o.score_kind,
                   o.observation_count, o.pose_json
            FROM observations o
            JOIN nodes n ON n.id=o.node_id
            WHERE n.target=%s AND n.length=%s
            """,
            (target, length),
        ))
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
        rows = connection.execute(
            """
            SELECT parent.sequence, child.sequence, edge.epoch, edge.source,
                   edge.proposal_type, edge.mutations_json, edge.metadata_json
            FROM lineage_edges edge
            JOIN nodes parent ON parent.id=edge.parent_node_id
            JOIN nodes child ON child.id=edge.child_node_id
            WHERE child.target=%s
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
    """Load the latest complete directional-effect snapshot for a target."""
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE `key`=%s",
            (f"graph_version:{target}",),
        ).fetchone()
        if row is None:
            return []
        graph_version = str(row[0])
        rows = connection.execute(
            """
            SELECT position, from_residue, to_residue, region, contact_class,
                   support, median_score_delta, uncertainty, graph_version
            FROM mutation_effects
            WHERE target=%s AND graph_version=%s
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
    connection: db.Connection,
    *,
    target: str,
    sequence_length: int,
) -> tuple[list[int], list[str], np.ndarray, dict[str, float]]:
    rows = connection.execute(
        """
        SELECT n.id, n.sequence, n.residue_vector, o.score, o.score_kind,
               o.epoch, o.source, o.observation_count
        FROM nodes n
        JOIN observations o ON o.node_id=n.id
        WHERE n.target=%s AND n.length=%s
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
        edge_rows: dict[tuple[int, int], tuple[int, int, int, str, str]] = {}
        scores: dict[str, float] = {}
        contact_rows = connection.execute(
            """
            SELECT n.id, o.pose_json FROM nodes n
            JOIN observations o ON o.node_id=n.id
            WHERE n.target=%s AND o.pose_json IS NOT NULL
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
            """
            SELECT length, COUNT(*) FROM nodes
            WHERE target=%s GROUP BY length ORDER BY length
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
            """
            DELETE FROM edges
            WHERE source_node_id IN (SELECT id FROM nodes WHERE target=%s)
            """,
            (target,),
        )
        connection.executemany(
            """
            INSERT INTO edges(
                source_node_id, target_node_id, distance,
                mutations_json, context_json
            ) VALUES(%s, %s, %s, %s, %s)
            """,
            edge_rows.values(),
        )

        observation_rows = connection.execute(
            """
            SELECT n.id, o.epoch, o.score_kind, o.source, o.score
            FROM nodes n
            JOIN observations o ON o.node_id=n.id
            WHERE n.target=%s AND o.epoch IS NOT NULL
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
            "DELETE FROM mutation_effects WHERE target=%s AND graph_version=%s",
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
            """
            INSERT INTO mutation_effects(
                target, position, from_residue, to_residue, region,
                contact_class, support, median_score_delta, uncertainty,
                graph_version
            ) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            effect_rows,
        )
        connection.execute(
            "REPLACE INTO metadata(`key`, value) VALUES(%s, %s)",
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
