"""Coverage + row/column block lane for reaction 3.

Measured basis for the axis choice (epoch 25176 data, all from exact scores):

* Locality is real. Molecules sharing two components correlate at 0.39-0.61;
  random pairs correlate at ~0.00. Take a strong trio, change one component,
  and you keep a strong trio far more often than chance.
* The productive axis is role C. P(neighbour in top decile | anchor in top
  decile), base rate 0.100: varying C 0.4579 (4.58x), varying B 0.2852
  (2.85x), varying A 0.1113 (1.11x, the base rate). Role A is the molecule's
  identity; swapping it destroys quality.
* Winners do exactly this. Across 70 winning submissions the mean number of
  molecule pairs sharing (A,B) -- C-sweeps -- is 7.09 against 2.58 for rank
  11+, correlating with submission score at +0.4163. Pairs sharing (A,C)
  average only 1.71. Our own portfolios contained ZERO (A,B)-shared pairs.
* Only similar swaps help; dissimilar ones actively hurt. By Tanimoto of the
  swapped role-C reagent: 0.8-1.0 retains 3.29x, 0.7-0.8 2.48x, 0.6-0.7 1.58x,
  and below 0.6 it is 0.83x -- worse than random. Near swaps change ~1.6 heavy
  atoms and 23 Da while leaving aromatic ring count almost untouched (delta
  0.08), and aromatic count is the strongest chemical correlate of binding
  probability we measured.
* Ultra-similar swaps are scarce: per anchor the median role-C reagent has 0
  neighbours at >=0.9, 1 at >=0.7, 4 at >=0.6 and 16 at >=0.5. The usable band
  is roughly 0.5-0.7 and is small enough to enumerate exhaustively.

An earlier version anchored on (A,C) and enumerated role B in a fixed
pseudo-random order: the 2.85x axis, walked in the order least likely to
preserve quality. It produced 153 (A,C)-shared pairs and zero (A,B)-shared
pairs, exhausted its grid (block_count x 2731) by round four, and collapsed
portfolio diversity to 3 distinct role-A and 2 distinct role-C reagents -- an
AtomPair entropy of 0.2695 against a 0.25 gate. Sweeping C instead yields up to
20 distinct role-C values, relieving that entropy pressure rather than fighting
it.

Two bounded mechanisms:

1. Coverage: a fraction of proposals drawn uniformly from the full valid role
   space rather than from historical frequencies, making unexplored reagents
   reachable. The frozen proxy still ranks novel reagents on chemistry; only
   its six component target-encodings fall back to the global mean.
2. Block sweep: fix the (A, B) pair of a high-scoring molecule and walk role C
   in descending Tanimoto similarity to that molecule's own role-C reagent,
   stopping at `minimum_similarity` where the effect reverses.

Both lanes are tagged so a slice of the exact-scoring batch can be reserved for
them, spent inside the batch and never above it. Validity, the 0.5 historical
novelty gate and the AtomPair entropy floor are external and untouched.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from compound_miner.adaptive_strategy import FeedbackProposalPolicy

RECORD_COLUMNS = ("name", "smiles", "reaction", "components", "inchikey",
                  "heavy_atoms", "rotatable_bonds", "proxy_mean", "proxy_std")


@dataclass
class SweepState:
    """Lane progress shared across rounds; the engine rebuilds policies."""

    sweep_components: set[tuple[int, ...]] = field(default_factory=set)
    coverage_components: set[tuple[int, ...]] = field(default_factory=set)
    cursor: dict[tuple[int, int], int] = field(default_factory=dict)
    # A block is an (A, B) pair. reference_c carries the anchor molecule's own
    # role-C reagent, which defines the similarity ordering for that block.
    blocks: list[tuple[int, int]] = field(default_factory=list)
    reference_c: dict[tuple[int, int], int] = field(default_factory=dict)
    c_order: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    sweep_proposed: int = 0
    coverage_proposed: int = 0
    exhausted: int = 0
    exhausted_blocks: set[tuple[int, int]] = field(default_factory=set)
    retired_blocks: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> dict:
        return dict(sweep_proposed=self.sweep_proposed,
                    coverage_proposed=self.coverage_proposed,
                    distinct_sweep_tuples=len(self.sweep_components),
                    distinct_coverage_tuples=len(self.coverage_components),
                    blocks=[list(block) for block in self.blocks],
                    exhausted_blocks=[list(b) for b in sorted(self.exhausted_blocks)],
                    retired_blocks=[list(b) for b in self.retired_blocks],
                    block_axis='fix_AB_sweep_C',
                    depth_capped=True,
                    reference_c={str(list(k)): v for k, v in self.reference_c.items()},
                    candidates_per_block={str(list(k)): len(v) for k, v in self.c_order.items()},
                    grid_exhausted_draws=self.exhausted)


class ExplorationSweepPolicy(FeedbackProposalPolicy):
    """Feedback policy plus a coverage lane and an (A,B)-block, C-sweep lane."""

    def __init__(self, history, reaction, database, rng, exact_parents=None, *,
                 feedback: pd.DataFrame, state: SweepState,
                 coverage_fraction: float = 0.35, sweep_fraction: float = 0.30,
                 block_count: int = 6, similarity_ranker=None,
                 minimum_similarity: float = 0.5, depth_per_block: int = 4):
        super().__init__(history, reaction, database, rng, exact_parents, feedback=feedback)
        if not 0.0 <= coverage_fraction <= 1.0 or not 0.0 <= sweep_fraction <= 1.0:
            raise ValueError("invalid_lane_fraction")
        if coverage_fraction + sweep_fraction > 1.0:
            raise ValueError("invalid_lane_fraction")
        if block_count < 1:
            raise ValueError("invalid_block_count")
        if len(self.roles) != 3:
            raise ValueError("rowcolumn_requires_three_roles")
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("invalid_minimum_similarity")
        if depth_per_block < 1:
            raise ValueError("invalid_depth_per_block")
        self.coverage_fraction = coverage_fraction
        self.sweep_fraction = sweep_fraction
        self.minimum_similarity = minimum_similarity
        # Winners' largest (A,B) group averages 2.3 molecules and only 10% of
        # them sweep five or more deep; the previous lane put 153 pairs in one
        # block and squeezed AtomPair entropy to 0.2695 against a 0.25 gate.
        # Cap depth so the sweep spreads across blocks instead of collapsing.
        self.depth_per_block = depth_per_block
        self.state = state
        # Ranking role C by similarity needs reagent structures. It is injected
        # so the lane stays testable without the chemistry stack; see
        # tanimoto_reagent_ranker for the production implementation.
        self.similarity_ranker = similarity_ranker
        self.role_c_ids = [int(value) for value in database.role_ids[self.roles[2]]]
        # Anchor blocks on the best molecules observed so far, not on marginal
        # reagent means: a good anchor looks mediocre until its column is found.
        previous = list(state.blocks)
        state.blocks = self._blocks(feedback, block_count, state.exhausted_blocks,
                                    state.reference_c)
        for block in previous:
            if block not in state.blocks and block not in state.retired_blocks:
                state.retired_blocks.append(block)
        for block in state.blocks:
            state.cursor.setdefault(block, 0)
            if block not in state.c_order:
                state.c_order[block] = self._rank_role_c(state.reference_c[block])[:depth_per_block]

    def _rank_role_c(self, reference: int) -> list[int]:
        """Role-C reagents ordered by similarity to the anchor's own role C.

        Dissimilar swaps measured *worse than random* (0.83x against a 0.100
        base rate), so the list is truncated rather than merely reordered.
        """
        if self.similarity_ranker is None:
            return [i for i in self.role_c_ids if i != reference]
        ordered = self.similarity_ranker(reference, self.role_c_ids,
                                         self.minimum_similarity)
        return [int(i) for i in ordered if int(i) != reference]

    @staticmethod
    def _blocks(feedback: pd.DataFrame, block_count: int,
                exhausted: set[tuple[int, int]] | None = None,
                reference_c: dict[tuple[int, int], int] | None = None) -> list[tuple[int, int]]:
        """Best-molecule anchors, skipping blocks whose role-C list is spent.

        Without the exclusion the block set locks in: the sweep concentrates
        scoring into its own blocks, those keep producing the best molecules, the
        set never changes and its grid stays consumed. On the first trial
        `sweep_proposed` froze at 38,234 between rounds four and five for exactly
        this reason.
        """
        if feedback.empty:
            return []
        exhausted = exhausted or set()
        ordered = feedback.sort_values("score", ascending=False)
        blocks: list[tuple[int, int]] = []
        for _, row in ordered.iterrows():
            block = (int(row["component_a"]), int(row["component_b"]))
            if block in blocks or block in exhausted:
                continue
            blocks.append(block)
            if reference_c is not None:
                # The best molecule in this block defines its similarity centre.
                reference_c.setdefault(block, int(row["component_c"]))
            if len(blocks) >= block_count:
                break
        return blocks

    def _sweep(self) -> tuple[int, ...] | None:
        blocks = [b for b in self.state.blocks
                  if b not in self.state.exhausted_blocks and self.state.c_order.get(b)]
        if not blocks:
            return None
        block = self.rng.choice(blocks)
        order = self.state.c_order[block]
        start = self.state.cursor.get(block, 0)
        # Walk in descending similarity: the nearest untried role C first.
        for index in range(start, len(order)):
            candidate = (block[0], block[1], order[index])
            if candidate not in self.state.sweep_components:
                self.state.cursor[block] = index + 1
                return candidate
        # This block's similarity-filtered role-C list is spent. Retire it so
        # the next round anchors somewhere new rather than re-selecting it.
        self.state.exhausted_blocks.add(block)
        self.state.exhausted += 1
        return None

    def propose(self) -> tuple[int, ...]:
        draw = self.rng.random()
        if draw < self.coverage_fraction:
            candidate = tuple(self.database.random_components(self.reaction, self.rng))
            self.state.coverage_components.add(candidate)
            self.state.coverage_proposed += 1
            return candidate
        if draw < self.coverage_fraction + self.sweep_fraction and self.state.blocks:
            candidate = self._sweep()
            if candidate is not None:
                self.state.sweep_components.add(candidate)
                self.state.sweep_proposed += 1
                return candidate
        return super().propose()


def tanimoto_reagent_ranker(database, reaction: int = 3, role_index: int = 2,
                            fingerprint_bits: int = 2048, radius: int = 2):
    """Build the production similarity ranker over a role's reagent structures.

    Returns a callable (reference_id, candidate_ids, minimum_similarity) ->
    ids ordered by descending Tanimoto, truncated at minimum_similarity.
    Fingerprints are computed once for the whole role and reused.

    Truncation matters: swaps below ~0.6 similarity retained top-decile
    molecules at 0.83x the base rate, i.e. worse than choosing at random.
    """
    import sqlite3 as _sqlite3
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    roles = database.reactions[reaction].roles
    identifiers = [int(value) for value in database.role_ids[roles[role_index]]]
    with _sqlite3.connect("file:%s?mode=ro" % database.path, uri=True) as connection:
        structures = {int(i): s for i, s in
                      connection.execute("SELECT mol_id,smiles FROM molecules")}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fingerprint_bits)
    fingerprints: dict[int, object] = {}
    for identifier in identifiers:
        molecule = Chem.MolFromSmiles(structures.get(identifier, "") or "")
        if molecule is not None:
            fingerprints[identifier] = generator.GetFingerprint(molecule)

    def rank(reference: int, candidates, minimum_similarity: float = 0.5):
        if reference not in fingerprints:
            return []
        usable = [i for i in candidates if i in fingerprints and i != reference]
        if not usable:
            return []
        values = DataStructs.BulkTanimotoSimilarity(
            fingerprints[reference], [fingerprints[i] for i in usable])
        kept = [(i, v) for i, v in zip(usable, values) if v >= minimum_similarity]
        kept.sort(key=lambda item: (-item[1], item[0]))
        return [i for i, _ in kept]

    rank.coverage = len(fingerprints)
    return rank


def quota_records(archive, reaction: int, components, quota: int,
                  excluded: set[str] | None = None,
                  order_by: str = "proxy_mean") -> list[dict]:
    """Unscored archive rows built by a tagged lane, best-first.

    `order_by` matters. Ranking an *exploration* lane by `proxy_mean` is
    self-defeating: that is the statistic most biased against chemistry the
    model has no labels for, so coverage rows lose every slot to sweep rows.
    Rank exploration by `proxy_std` (tree disagreement) instead.
    """
    if order_by not in ("proxy_mean", "proxy_std"):
        raise ValueError("invalid_quota_order")
    if quota <= 0 or not components:
        return []
    wanted = {tuple(item) for item in components}
    excluded = excluded or set()
    with sqlite3.connect(archive) as connection:
        rows = connection.execute(
            "SELECT name,smiles,reaction,components,inchikey,heavy_atoms,rotatable_bonds,"
            "proxy_mean,proxy_std FROM candidates WHERE reaction=? AND score IS NULL "
            "ORDER BY " + order_by + " DESC, name ASC", (reaction,)).fetchall()
    chosen = []
    for row in rows:
        if len(chosen) >= quota:
            break
        if (row[4] or row[0]) in excluded:
            continue
        if tuple(json.loads(row[3])) not in wanted:
            continue
        chosen.append(dict(zip(RECORD_COLUMNS, row)))
    return chosen


def merge_with_quota(candidates: list[dict], extra: list[dict]) -> tuple[list[dict], dict]:
    """Substitute lane rows for the weakest proxy picks, keeping size fixed.

    The exact-score budget is the batch size, so spending a quota must displace
    the lowest-ranked entries rather than extend the batch.
    """
    present = {record["name"] for record in candidates}
    incoming = {record["name"] for record in extra}
    fresh = [record for record in extra if record["name"] not in present]
    already = sum(1 for record in candidates if record["name"] in incoming)
    coverage = dict(quota_requested=len(extra), rows_substituted=0,
                    rows_already_selected=already, batch_size=len(candidates))
    if not fresh:
        return list(candidates), coverage
    keep = candidates[:max(0, len(candidates) - len(fresh))]
    merged = keep + fresh[:len(candidates) - len(keep)]
    if len(merged) != len(candidates):
        raise ValueError("quota_changed_batch_size")
    coverage["rows_substituted"] = len(merged) - len(keep)
    return merged, coverage
