"""Real ANARCI germline references for the PR100 proposal tests.

The proposal module reads its germline from the ``panels`` argument, so the
tests must pass a panel that actually contains the ``WVRQ`` FR2 anchor and the
IMGT alignment gaps; an all-``A`` synthetic panel silently skips the CDR2-motif
and V-identity paths. These are the human IGHV3-* alleles in ANARCI's 128-state
reference alignment plus the two J alleles the scaffold reads.
"""

#: ANARCI 128-state reference alignment (dashes are IMGT gaps).
V_GERMLINES = {
    'IGHV3-23*01': 'EVQLLESGG-GLVQPGGSLRLSCAASGFTF----SSYAMSWVRQAPGKGLEWVSAISGS--GGSTYYADSVK-GRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAK----------------------',
    'IGHV3-66*01': 'EVQLVESGG-GLVQPGGSLRLSCAASGFTV----SSNYMSWVRQAPGKGLEWVSVIYSG---GSTYYADSVK-GRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR----------------------',
    'IGHV3-74*01': 'EVQLVESGG-GLVQPGGSLRLSCAASGFTF----SSYWMHWVRQAPGKGLVWVSRINSD--GSSTSYADSVK-GRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAR----------------------',
    'IGHV3-7*01': 'EVQLVESGG-GLVQPGGSLRLSCAASGFTF----SSYWMSWVRQAPGKGLEWVANIKQD--GSEKYYVDSVK-GRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR----------------------',
    'IGHV3-11*01': 'QVQLVESGG-GLVKPGGSLRLSCAASGFTF----SDYYMSWIRQAPGKGLEWVSYISSS--GSTIYYADSVK-GRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR----------------------',
    'IGHV3-43*01': 'EVQLVESGG-VVVQPGGSLRLSCAASGFTF----DDYTMHWVRQAPGKGLEWVSLISWD--GGSTYYADSVK-GRFTISRDNSKNSLYLQMNSLRTEDTALYYCAKD---------------------',
    'IGHV3-48*01': 'EVQLVESGG-GLVQPGGSLRLSCAASGFTF----SSYSMNWVRQAPGKGLEWVSYISSS--SSTIYYADSVK-GRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAR----------------------',
    'IGHV3-53*01': 'EVQLVESGG-GLIQPGGSLRLSCAASGFTV----SSNYMSWVRQAPGKGLEWVSVIYSG---GSTYYADSVK-GRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR----------------------',
    'IGHV3-64*01': 'EVQLVESGG-GLVQPGGSLRLSCAASGFTF----SSYAMHWVRQAPGKGLEYVSAISSN--GGSTYYANSVK-GRFTISRDNSKNTLYLQMGSLRAEDMAVYYCAR----------------------',
}

J_GERMLINES = {
    'IGHJ4*01': '------------------------------------------------------------------------------------------------------------------FDYWGQGTLVTVSS',
    'IGHJ6*01': '------------------------------------------------------------------------------------------------------------------MDVWGQGTTVTVSS',
}


def real_panel():
    """Build the {region: {allele: {position: aa}}} panel the module consumes."""
    def to_panel(seq):
        return {p: aa for p, aa in enumerate(seq, start=1)}
    return {
        'V': {allele: to_panel(seq) for allele, seq in V_GERMLINES.items()},
        'J': {allele: to_panel(seq) for allele, seq in J_GERMLINES.items()},
    }
