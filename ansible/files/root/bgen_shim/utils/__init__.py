"""The two symbols boltzgen_wrapper needs, without the validator import graph.

boltzgen_wrapper.py does exactly:

    from utils import get_sequence_from_protein_code, seq_hash

but nova's utils/__init__ eagerly imports molecules -> combinatorial_db,
proteins -> datasets, and nanobodies -> metanano (the whole NOVA-nanobody-filter
service: TNP, IgBLAST, ANARCI, mmseqs2). None of that is reachable from the two
functions actually used, and installing it would put a second heavy dependency
stack next to torch 2.7.1 in this env.

This box exists to run boltzgen as an EXACT SCORER. It is not a validator and
does not run Layer-2 filtering, so it ships these two functions faithfully --
copied from utils/proteins.py and utils/nanobodies.py -- and nothing else.

Lives in /root/bgen_shim, reached via PYTHONPATH, so the shadowing is explicit
and scoped rather than a package quietly dropped into site-packages.
"""
import hashlib

import requests


def seq_hash(seq: str) -> str:
    """Verbatim from utils/nanobodies.py."""
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def get_sequence_from_protein_code(protein_code: str,
                                   clip_interval: tuple[int, int] = None) -> str:
    """From utils/proteins.py, UniProt path only.

    The upstream version falls back to the HuggingFace "Metanova/Proteins"
    dataset when UniProt fails. That fallback is dropped here deliberately: it
    pulls in `datasets`, and a scoring run that silently proceeded on a WRONG
    target sequence would produce plausible-looking metrics that are entirely
    invalid. Raising is the safer failure for an offline scorer.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"UniProt returned {response.status_code} for {protein_code}. "
            "Refusing to guess the target sequence -- scoring against the wrong "
            "target yields metrics that look fine and mean nothing."
        )
    sequence = "".join(
        line.strip() for line in response.text.splitlines()
        if not line.startswith(">")
    )
    if not sequence:
        raise RuntimeError(f"UniProt returned an empty sequence for {protein_code}")
    if clip_interval:
        sequence = sequence[clip_interval[0]:clip_interval[1]]
    return sequence
