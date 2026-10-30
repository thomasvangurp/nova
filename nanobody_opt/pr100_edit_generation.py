"""Edit-based candidate generation ported from ``nanobody_opt/local_neighborhood.py``.

The PR100 scaffold lane (``pr100_proposal_pilot.propose``) generates de-novo
scaffolds and collapses ``design_iiptm`` to ~0.25. The v2 producer instead
generated edit-based children of *real public frameworks* via
``nanobody_opt/local_neighborhood``, reaching ``design_iiptm`` ~0.50. This
module ports the two pure edit-lane primitives that power that generator so the
PR100 worker's ``sequence_phase`` can run an edit-based lane in-process, fed by
rival parents, without importing ``local_neighborhood``'s top-level scientific
stack (which pulls in ``mutation_network`` -> ``db`` -> ``MySQLdb``).

Ported from the v2 source and dependency-free -- no pandas, no numpy, no
network, no DB. The one intentional deviation is that both haplotype paths skip
a haplotype carrying a residue outside ``gates.ALLOWED_AAS``, since a donor
residue is copied verbatim and would otherwise reach the annotator:

* ``supported_framework_candidates`` -- single/multi framework and terminal
  substitutions that PRESERVE the parent CDR3 (positions outside the CDRs).
* ``supported_lineage_candidates`` -- the CDR + terminal lineage grammar:
  single substitutions and co-observed haplotypes, the only lane that edits
  CDR positions (including the whole-CDR motif edits).

Both are pure functions of a parent string + a per-position allowed-residue
table + same-length background sequences, exactly as ``_generate_exact_length``
invokes them. The full v2 orchestrator (seed shortlisting, basin allocation,
mutation-network ranking) remains available as a subprocess:
``python -m nanobody_opt.local_neighborhood --layer2-seeds ... --archive ...
--config ... --top50-index ... --winner-priors ... --mutation-observations ...``
-- this module is the testable in-process lane, not a replacement for it.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import random
from itertools import zip_longest

from nanobody_opt import gates, rules
from nanobody_opt.regions import find_regions

#: Proposal record source tag emitted by the in-process edit lane.
SOURCE = "edit_generation"
BALANCED_POLICY = "balanced_v2"
CHEMISTRY_POLICY = "chemistry_conservative_v3"
EDIT_POLICIES = frozenset((BALANCED_POLICY, CHEMISTRY_POLICY))
CONSERVATIVE_GROUPS = ("DE", "KR", "NQST", "ILVM", "FYW")


def validate_policy(policy):
    if not isinstance(policy, str) or policy not in EDIT_POLICIES:
        raise ValueError("unknown edit generation policy")
    return policy


def conservative_substitutions(residue):
    """Keep charge/aromatic classes; do not move C/G/P/A/H positions.

    This is a bounded proposal heuristic, never a developability or fitness
    prediction. Every emitted child still requires the full scientific gates.
    """
    for group in CONSERVATIVE_GROUPS:
        if residue in group:
            return tuple(aa for aa in group if aa != residue)
    return ()


def supported_framework_candidates(
    parent: str,
    *,
    allowed: list[list[str]],
    background_sequences: list[str],
    cdr_positions: set[int],
    protected: set[int],
    terminal_width: int = 17,
    maximum_haplotype_mutations: int = 3,
) -> list[tuple[str, str]]:
    """Enumerate supported framework singles and coherent terminal haplotypes.

    Ported verbatim from ``nanobody_opt/local_neighborhood.py``. Framework
    positions are everything outside the CDRs and protected contacts, so these
    children inherit the parent's CDR3 unchanged (and therefore its V framework
    identity budget).
    """
    length = len(parent)
    if len(allowed) != length:
        raise ValueError("allowed residue table has the wrong length")
    if terminal_width < 1 or maximum_haplotype_mutations < 2:
        raise ValueError("invalid terminal haplotype configuration")
    framework = set(range(length)) - cdr_positions - protected
    c_terminal = (
        set(range(max(0, length - terminal_width), length)) & framework
    )
    n_terminal = set(range(0, min(10, length))) & framework
    candidates: dict[str, str] = {}

    position_order = sorted(
        framework,
        key=lambda position: (
            0 if position == length - 1 else
            1 if position in c_terminal else
            2 if position in n_terminal else 3,
            position,
        ),
    )
    for position in position_order:
        source = (
            "c_terminal_single_substitution"
            if position in c_terminal
            else "n_terminal_single_substitution"
            if position in n_terminal
            else "framework_single_substitution"
        )
        for residue in allowed[position]:
            if residue == parent[position]:
                continue
            candidate = parent[:position] + residue + parent[position + 1:]
            candidates.setdefault(candidate, source)

    compatible = [
        sequence
        for sequence in background_sequences
        if len(sequence) == length
    ]
    for donor in compatible:
        for positions, source in (
            (c_terminal, "c_terminal_haplotype"),
            (n_terminal, "n_terminal_haplotype"),
            (framework, "framework_haplotype"),
        ):
            changes = [
                position
                for position in positions
                if donor[position] != parent[position]
            ]
            if not 2 <= len(changes) <= maximum_haplotype_mutations:
                continue
            # A haplotype copies donor residues verbatim, so (unlike the
            # singles, which draw from ``allowed``) it can carry a residue
            # outside the standard alphabet; drop the whole haplotype.
            if any(donor[position] not in gates.ALLOWED_AAS for position in changes):
                continue
            candidate = list(parent)
            for position in changes:
                candidate[position] = donor[position]
            candidates.setdefault("".join(candidate), source)
    return list(candidates.items())


def supported_lineage_candidates(
    parent: str,
    *,
    allowed: list[list[str]],
    background_sequences: list[str],
    cdr_positions: set[int],
    protected: set[int],
    terminal_width: int = 17,
    maximum_haplotype_mutations: int = 6,
    priors: list[dict[str, float]] | None = None,
) -> list[tuple[str, str]]:
    """Enumerate the complete supported terminal/CDR lineage grammar.

    Ported verbatim from ``nanobody_opt/local_neighborhood.py``. This is the
    ONLY producer for CDR positions -- the framework lane subtracts them -- so
    it is what carries whole-CDR motif edits and CDR3-repair haplotypes.
    """
    length = len(parent)
    if len(allowed) != length:
        raise ValueError("allowed residue table has the wrong length")
    if terminal_width < 1 or maximum_haplotype_mutations < 2:
        raise ValueError("invalid lineage haplotype configuration")
    terminal = set(range(min(10, length))) | set(
        range(max(0, length - terminal_width), length)
    )
    mutable = (terminal | cdr_positions) - protected
    candidates: dict[str, str] = {}
    haplotypes: list[tuple[float, str]] = []

    singles: list[tuple[float, int, str, str, str]] = []
    for position in sorted(mutable):
        source = (
            "lineage_cdr_single_substitution"
            if position in cdr_positions
            else "lineage_terminal_single_substitution"
        )
        for residue in allowed[position]:
            if residue == parent[position]:
                continue
            candidate = parent[:position] + residue + parent[position + 1:]
            weight = 1.0
            if priors is not None and position < len(priors):
                weight = float(priors[position].get(residue, 1.0))
            # position and residue break ties so the order stays deterministic
            singles.append((-weight, position, residue, candidate, source))
    singles.sort()
    for _, _, _, candidate, source in singles:
        candidates.setdefault(candidate, source)

    for donor in background_sequences:
        if len(donor) != length:
            continue
        changes = [
            position
            for position in sorted(mutable)
            if donor[position] != parent[position]
        ]
        if not 2 <= len(changes) <= maximum_haplotype_mutations:
            continue
        # Same guard as the framework haplotypes: skip the whole haplotype if
        # any changed position carries a non-standard residue.
        if any(donor[position] not in gates.ALLOWED_AAS for position in changes):
            continue
        candidate = list(parent)
        for position in changes:
            candidate[position] = donor[position]
        haplotypes.append((
            -_mean_prior_weight(priors, changes, donor),
            "".join(candidate),
        ))
    haplotypes.sort()
    for _, candidate in haplotypes:
        candidates.setdefault(candidate, "lineage_haplotype")
    return list(candidates.items())


def _mean_prior_weight(
    priors: list[dict[str, float]] | None,
    positions: list[int],
    donor: str,
) -> float:
    """Mean residue weight a haplotype carries; 1.0 when there is no prior.

    Ported verbatim from ``nanobody_opt/local_neighborhood.py``.
    """
    if priors is None or not positions:
        return 1.0
    weights = [
        float(priors[position].get(donor[position], 1.0))
        for position in positions
        if position < len(priors)
    ]
    return sum(weights) / len(weights) if weights else 1.0


def allowed_table(sequences: list[str], length: int, *, min_support: int = 1) -> list[list[str]]:
    """Per-position residues observed across ``sequences``, filtered to the
    allowed amino-acid alphabet and a ``min_support`` count floor.

    Mirrors the ``allowed`` computation in ``_generate_exact_length`` (a
    ``value_counts`` over same-length donors, ``count >= min_support``, residue
    in ``gates.ALLOWED_AAS``), without pandas.
    """
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    table: list[list[str]] = []
    for position in range(length):
        counts = Counter(s[position] for s in sequences if len(s) == length)
        table.append(sorted(
            residue for residue, count in counts.items()
            if count >= min_support and residue in gates.ALLOWED_AAS
        ))
    return table


def _annotated_spans(parent, cdrs):
    if not isinstance(cdrs, dict):
        return None
    spans = {}
    for key in ('cdr1', 'cdr2', 'cdr3'):
        loop = cdrs.get(key)
        if not isinstance(loop, str) or not loop or parent.count(loop) != 1:
            return None
        start = parent.index(loop)
        spans[key] = list(range(start, start + len(loop)))
    if (max(spans['cdr1']) >= min(spans['cdr2'])
            or max(spans['cdr2']) >= min(spans['cdr3'])):
        return None
    return spans


def novelty_aware_candidates(parent, *, cdrs, allowed, limit, identity_budget,
                             policy=BALANCED_POLICY):
    """Bounded parent-guided multi-edits; parent-relative proxies are not novelty.

    CDR spans come from qualified annotation, not the heuristic region finder.
    Framework, conserved cysteines and the final three J-junction residues stay
    unchanged. The final pipeline must still annotate, profile and search the
    full current archive using the validator implementation.
    """
    policy = validate_policy(policy)
    spans = _annotated_spans(parent, cdrs)
    if (spans is None or len(allowed) != len(parent) or identity_budget is None
            or identity_budget.parent != parent):
        return []
    # Strict inequalities: 95% whole identity and 85% CDR3 identity are both
    # duplicate boundaries, so equality does not qualify this proposal arm.
    whole_edits = math.floor(len(parent) * 0.05) + 1
    cdr3_edits = math.floor(len(spans['cdr3']) * 0.15) + 1
    protected = identity_budget.protected_j | set(spans['cdr3'][-3:])
    cdr3_mutable = [i for i in spans['cdr3'] if parent[i] != 'C' and i not in protected]
    other_mutable = [i for key in ('cdr1', 'cdr2') for i in spans[key]
                     if parent[i] != 'C' and i not in protected]
    if cdr3_edits > len(cdr3_mutable) or whole_edits - cdr3_edits > len(other_mutable):
        return []
    from nanobody_opt.pr100_proposal_pilot import AMINO_ACID_WEIGHTS
    broad = [aa for aa, weight in AMINO_ACID_WEIGHTS.items() if weight > 0 and aa != 'C']
    candidates = []
    seen = set()
    for ordinal in range(limit):
        seed_prefix = ('pr100-parent-chemistry-v3:' if policy == CHEMISTRY_POLICY
                       else 'pr100-parent-v-budget-v2:')
        seed = int.from_bytes(hashlib.sha256(
            (seed_prefix + parent + ':' + str(ordinal)).encode()).digest(), 'big')
        rng = random.Random(seed)
        candidate = list(parent)
        used = set()

        def choose(positions):
            options = []
            for position in positions:
                if position in used:
                    continue
                observed = [aa for aa in allowed[position]
                            if aa in gates.ALLOWED_AAS and aa not in (parent[position], 'C')]
                if policy == CHEMISTRY_POLICY:
                    population = conservative_substitutions(parent[position])
                else:
                    population = observed if observed and ordinal % 2 == 0 else broad
                    restore = identity_budget.restoration(position)
                    population = sorted(set(population) | ({restore} if restore else set()))
                for aa in population:
                    if aa in (parent[position], 'C'):
                        continue
                    delta = identity_budget.delta(position, aa)
                    # Never spend a matching V residue merely to fill a quota.
                    # Parent mismatches, insertions and restoration edits supply
                    # novelty without consuming its narrow identity allowance.
                    if delta < 0:
                        continue
                    # Keep the minimum CDR3 edits, then use permitted
                    # CDR1/2 mismatches. Concentrating all edits in CDR3 failed
                    # real archive novelty despite whole-sequence Hamming<95%:
                    # the validator uses local alignment against every length.
                    if policy == CHEMISTRY_POLICY:
                        # Prefer observed class-preserving residues on alternate
                        # draws; a germline restoration is not a quality score.
                        priority = 0 if ordinal % 2 == 0 and aa in observed else 1
                    else:
                        priority = 0 if delta > 0 else 1
                    options.append((priority, rng.random(), position, aa))
            if not options:
                return False
            _, _, position, aa = min(options)
            candidate[position] = aa
            used.add(position)
            return True

        if not all(choose(cdr3_mutable) for _ in range(cdr3_edits)):
            continue
        if not all(choose(other_mutable)
                   for _ in range(whole_edits - cdr3_edits)):
            continue
        candidate = ''.join(candidate)
        differences = sum(a != b for a, b in zip(candidate, parent))
        changed_cdr3 = sum(candidate[i] != parent[i] for i in spans['cdr3'])
        if (differences < whole_edits or changed_cdr3 < cdr3_edits
                or candidate in seen or not identity_budget.accepts(candidate)):
            continue
        seen.add(candidate)
        lane = ('novelty_aware_multi_edit_chemistry_conservative_v3'
                if policy == CHEMISTRY_POLICY else 'novelty_aware_multi_edit')
        candidates.append((candidate, lane))
    return candidates


def generate(
    parents: list[str],
    *,
    background_sequences: list[str] | None = None,
    priors: list[dict[str, float]] | None = None,
    terminal_width: int = 17,
    maximum_haplotype_mutations: int = 6,
    framework_maximum_haplotype_mutations: int = 3,
    min_support: int = 1,
    limit: int | None = None,
    excluded_hashes: set[str] | frozenset[str] | None = None,
    stats: dict[str, int] | None = None,
    novelty_aware: bool = False,
    parent_cdrs: dict[str, dict[str, str]] | None = None,
    parent_identity_budgets: dict | None = None,
    policy: str = BALANCED_POLICY,
) -> list[dict[str, str]]:
    """Emit edit children round-robin, with separate contexts for each parent length.

    With ``novelty_aware=True``, only bounded multi-edits that pass strict
    parent-relative identity proxies are emitted, using supplied annotation
    CDRs and HMM identity budgets. These projections are never current-archive
    novelty or child annotation evidence.

    ``policy`` defaults to ``balanced_v2`` and reproduces its proposal order.
    The opt-in ``chemistry_conservative_v3`` preserves residue classes without
    prioritizing germline restoration. Its independent seed reproduces the
    measured prototype; it does not promise native or binding-score success.

    Parents are proposal CENTRES, never submissions. Each parent seeds the
    framework lane (CDR3-preserving singles/haplotypes) and the lineage lane
    (CDR + terminal singles/haplotypes), both drawn from the same per-position
    ``allowed`` table built from the parent + background pool. Within a parent
    the two lanes alternate, so a small ``limit`` cannot be filled by the
    framework lane alone. Children that are byte-identical to a parent (a cross
    single-edit) or to a previously emitted child are dropped. The result is
    deduped, deterministic for a fixed input, and ordered round-robin across the ranked parents.

    ``excluded_hashes`` holds ``rules.sequence_hash`` values the caller will
    discard anyway (already-published or already-tried sequences). They are
    skipped BEFORE ``limit`` is applied, so ``limit`` caps only the candidates
    that are actually emitted; a deterministic continuation that excludes every
    prior hash therefore advances to fresh candidates instead of re-selecting
    the same leading ``limit``.

    A parent whose regions cannot be located (``find_regions`` raises
    ``ValueError``) is skipped rather than aborting the pool, and does not seed
    the ``allowed`` table or the parent filter. Pass a ``stats`` dict to
    receive the count under ``malformed_parents``.
    """
    policy = validate_policy(policy)
    if policy != BALANCED_POLICY and not novelty_aware:
        raise ValueError("chemistry policy requires qualified novelty-aware parent contexts")
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('limit must be a nonnegative integer')
    if limit == 0:
        return []
    if stats is None:
        stats = {}
    stats.setdefault("malformed_parents", 0)
    located = []
    seen = set()
    for raw in parents:
        parent = str(raw).strip().upper()
        if not parent or parent in seen:
            continue
        seen.add(parent)
        if novelty_aware:
            spans = _annotated_spans(parent, (parent_cdrs or {}).get(parent))
            if spans is None:
                stats['malformed_parents'] += 1
                continue
            cdr_positions = set().union(*spans.values())
        else:
            try:
                regions = find_regions(parent)
            except ValueError:
                stats["malformed_parents"] += 1
                continue
            cdr_positions = set(regions.cdr_positions())
        located.append((parent, cdr_positions))
    backgrounds = {}
    for raw in background_sequences or []:
        sequence = str(raw).strip().upper()
        backgrounds.setdefault(len(sequence), []).append(sequence)
    parent_set = {parent for parent, _ in located}
    allowed_by_length = {}
    for parent, _ in located:
        length = len(parent)
        if length not in allowed_by_length:
            allowed_by_length[length] = allowed_table(
                [p for p, _ in located if len(p) == length] + backgrounds.get(length, []),
                length, min_support=min_support)
    streams = []
    for parent, cdr_positions in located:
        length = len(parent)
        background = backgrounds.get(length, [])
        allowed = allowed_by_length[length]
        if novelty_aware:
            framework = novelty_aware_candidates(parent,
                cdrs=(parent_cdrs or {}).get(parent), allowed=allowed,
                identity_budget=(parent_identity_budgets or {}).get(parent), policy=policy,
                limit=max(16, min(2048, (limit or 300) * 4)))
            lineage = []
        else:
            framework = supported_framework_candidates(
                parent, allowed=allowed, background_sequences=background,
                cdr_positions=cdr_positions, protected=set(), terminal_width=terminal_width,
                maximum_haplotype_mutations=framework_maximum_haplotype_mutations)
            lineage = supported_lineage_candidates(
                parent, allowed=allowed, background_sequences=background,
                cdr_positions=cdr_positions, protected=set(), terminal_width=terminal_width,
                maximum_haplotype_mutations=maximum_haplotype_mutations, priors=priors)
        records = []
        for pair in zip_longest(framework, lineage):
            for candidate_lane in pair:
                if candidate_lane is None:
                    continue
                candidate, lane = candidate_lane
                if candidate not in parent_set:
                    records.append({"sequence": candidate, "parent_sequence": parent,
                                    "proposal_lane": lane, "source": SOURCE,
                                    **({"edit_policy": policy} if novelty_aware else {})})
        # Dedup/filter before round-robin so a parent's redundant leading
        # records cannot starve the other parents under a small output limit.
        own_seen = set()
        stream = []
        for record in records:
            candidate = record['sequence']
            if candidate in own_seen or rules.sequence_hash(candidate) in (excluded_hashes or ()):
                continue
            own_seen.add(candidate)
            stream.append(record)
        streams.append(iter(stream))
    deduped = []
    emitted = set()
    while streams:
        pending = []
        for stream in streams:
            for record in stream:
                sequence = record['sequence']
                if sequence in emitted:
                    continue
                emitted.add(sequence)
                deduped.append(record)
                pending.append(stream)
                break
            if limit is not None and len(deduped) >= limit:
                return deduped
        streams = pending
    return deduped
