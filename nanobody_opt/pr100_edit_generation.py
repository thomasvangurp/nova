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

from nanobody_opt import gates
from nanobody_opt.regions import find_regions

#: Proposal record source tag emitted by the in-process edit lane.
SOURCE = "edit_generation"


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
    stats: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Emit edit-based children for a pool of same-length parent sequences.

    Parents are proposal CENTRES, never submissions. Each parent seeds the
    framework lane (CDR3-preserving singles/haplotypes) and the lineage lane
    (CDR + terminal singles/haplotypes), both drawn from the same per-position
    ``allowed`` table built from the parent + background pool. Children that
    are byte-identical to a parent (a cross single-edit) or to a previously
    emitted child are dropped. The result is deduped, deterministic for a fixed
    input, and ordered parent-first.

    A parent whose regions cannot be located (``find_regions`` raises
    ``ValueError``) is skipped rather than aborting the pool. Pass a ``stats``
    dict to receive the count under ``malformed_parents``.
    """
    if stats is None:
        stats = {}
    stats.setdefault("malformed_parents", 0)
    parent_list: list[str] = []
    seen: set[str] = set()
    for raw in parents:
        sequence = str(raw).strip().upper()
        if not sequence:
            continue
        if parent_list and len(sequence) != len(parent_list[0]):
            continue  # per-length context, like _generate_exact_length
        if sequence in seen:
            continue
        seen.add(sequence)
        parent_list.append(sequence)
    if not parent_list:
        return []
    length = len(parent_list[0])
    background = [
        str(sequence).strip().upper()
        for sequence in (background_sequences or [])
        if len(str(sequence).strip().upper()) == length
    ]
    allowed = allowed_table([*parent_list, *background], length, min_support=min_support)
    parent_set = set(parent_list)
    records: list[dict[str, str]] = []
    for parent in parent_list:
        try:
            regions = find_regions(parent)
        except ValueError:
            stats["malformed_parents"] += 1
            continue
        cdr_positions = set(regions.cdr_positions())
        protected: set[int] = set()
        for candidate, lane in supported_framework_candidates(
            parent,
            allowed=allowed,
            background_sequences=background,
            cdr_positions=cdr_positions,
            protected=protected,
            terminal_width=terminal_width,
            maximum_haplotype_mutations=framework_maximum_haplotype_mutations,
        ):
            if candidate in parent_set:
                continue
            records.append({"sequence": candidate, "parent_sequence": parent,
                            "proposal_lane": lane, "source": SOURCE})
        for candidate, lane in supported_lineage_candidates(
            parent,
            allowed=allowed,
            background_sequences=background,
            cdr_positions=cdr_positions,
            protected=protected,
            terminal_width=terminal_width,
            maximum_haplotype_mutations=maximum_haplotype_mutations,
            priors=priors,
        ):
            if candidate in parent_set:
                continue
            records.append({"sequence": candidate, "parent_sequence": parent,
                            "proposal_lane": lane, "source": SOURCE})
    deduped: list[dict[str, str]] = []
    emitted: set[str] = set()
    for record in records:
        if record["sequence"] in emitted:
            continue
        emitted.add(record["sequence"])
        deduped.append(record)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped
