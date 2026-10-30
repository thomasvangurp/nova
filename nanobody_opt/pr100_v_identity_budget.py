"""Proposal-only V identity accounting on the validator's HMM match states.

This predicts substitutions under the parent's alignment. It never replaces
re-annotation: a child can align differently or acquire a different top gene.
No IMGT/AbNumber position is used to index a germline reference here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property


@dataclass(frozen=True)
class VIdentityBudget:
    parent: str
    allowed_gene: str
    minimum: Fraction
    # Each allele is (name, non-gap reference positions as sequence-index/AA).
    # A missing query match has index None and still counts in the denominator.
    alleles: tuple
    anchor: tuple
    protected_j: frozenset[int]

    @cached_property
    def _projection(self):
        return tuple((name, sum(i is not None and self.parent[i] == aa for i, aa in positions),
                      len(positions), {i: aa for i, aa in positions if i is not None})
                     for name, positions in self.alleles)

    @cached_property
    def _anchor_residues(self):
        return dict(self.anchor)

    def scores(self, sequence):
        if len(sequence) != len(self.parent):
            raise ValueError('identity projection requires substitution-only edits')
        changes = [i for i, (old, new) in enumerate(zip(self.parent, sequence)) if old != new]
        return [(name, Fraction(base + sum(int(sequence[i] == reference[i])
                    - int(self.parent[i] == reference[i]) for i in changes if i in reference), length))
                for name, base, length, reference in self._projection]

    def accepts(self, sequence):
        scores = self.scores(sequence)
        best = max(score for _, score in scores)
        return (best >= self.minimum
                and any(name.split('*')[0] == self.allowed_gene and score == best
                        for name, score in scores)
                and all(sequence[i] == self.parent[i] for i in self.protected_j))

    def delta(self, index, residue):
        """Change in matches to a best allowed-gene allele; insertions cost zero."""
        reference = self._anchor_residues.get(index)
        if reference is None:
            return 0
        return int(residue == reference) - int(self.parent[index] == reference)

    def restoration(self, index):
        reference = self._anchor_residues.get(index)
        return reference if reference is not None and reference != self.parent[index] else None


def from_alignment(parent, *, allowed_gene, minimum_identity_pct,
                   state_vectors, details, panels):
    """Use precisely the run_hmmer/check_for_j state vectors scored upstream."""
    if (len(state_vectors) != 1 or len(details) != 1
            or details[0].get('chain_type') != 'H'):
        raise ValueError('expected exactly one heavy-chain HMM domain')
    minimum = Fraction(str(minimum_identity_pct)) / 100
    if not 0 <= minimum <= 1:
        raise ValueError('invalid minimum V identity')
    states = {tuple(state): index for state, index in state_vectors[0]}
    # dict(state_vectors[0]) is also the exact upstream treatment of repeated
    # insertion states; only match states enter the identity denominator.
    match_indices = {position: states.get((position, 'm')) for position in range(1, 129)}
    if any(i is not None and (type(i) is not int or not 0 <= i < len(parent))
           for i in match_indices.values()):
        raise ValueError('invalid HMM sequence index')
    present = [i for i in match_indices.values() if i is not None]
    if len(set(present)) != len(present):
        raise ValueError('duplicate HMM match sequence index')

    def references(region):
        output = []
        for name, reference in sorted(panels[region].items()):
            positions = tuple((match_indices[int(p)], aa)
                              for p, aa in reference.items() if aa != '-')
            if not positions:
                raise ValueError('empty germline reference')
            output.append((name, positions))
        if not output:
            raise ValueError('empty germline panel')
        return tuple(output)

    alleles = references('V')
    scores = [(name, Fraction(sum(i is not None and parent[i] == aa for i, aa in positions),
                             len(positions)), positions) for name, positions in alleles]
    best = max(score for _, score, _ in scores)
    eligible = [(name, positions) for name, score, positions in scores
                if name.split('*')[0] == allowed_gene and score == best and score >= minimum]
    if not eligible:
        raise ValueError('parent does not satisfy the current V identity challenge')
    # Preserve the matched query positions used by every tied best J allele.
    # This includes the J contribution inside CDR3, not an assumed IMGT span.
    j_scores = [(Fraction(sum(i is not None and parent[i] == aa for i, aa in positions),
                         len(positions)), positions) for _, positions in references('J')]
    j_best = max(score for score, _ in j_scores)
    protected = frozenset(i for score, positions in j_scores if score == j_best
                          for i, _ in positions if i is not None)
    anchor = tuple((i, aa) for i, aa in eligible[0][1] if i is not None)
    return VIdentityBudget(parent, allowed_gene, minimum, alleles, anchor, protected)


def build(parent, *, allowed_gene, minimum_identity_pct):
    """Extract one real parent alignment in the qualified annotation runtime."""
    from anarci.anarci import run_hmmer, check_for_j
    from nanobody_opt import pr100_filter

    pr100_filter.validate_annotation_runtime()
    records = [('query', parent)]
    alignments = run_hmmer(records, ncpu=1, hmmer_species=None)
    check_for_j(records, alignments, 'imgt')
    _, vectors, details = alignments[0]
    panels, _ = pr100_filter.helper('germline_identity')._anarci_panel('human')
    return from_alignment(parent, allowed_gene=allowed_gene,
                          minimum_identity_pct=minimum_identity_pct,
                          state_vectors=vectors, details=details, panels=panels)
