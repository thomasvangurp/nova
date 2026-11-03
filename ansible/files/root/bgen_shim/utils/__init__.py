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
import threading
import time

import requests

# The scorer calls this once per CHUNK on replicate jobs (batch_size is forced to
# 1, and _write_yaml_files() runs every chunk), i.e. once per sequence. A single
# transient UniProt failure used to lose that sequence: the scorer's chunk loop
# swallows the exception ("raised ... continuing") and the shard's out file comes
# back short, which the daemon reports as RETRY_SHORT_SHARDS. A UniProt outage
# during a round lost EVERY sequence ("produced 0/14" on all shards).
# Retrying transient failures and remembering a good answer for a while fixes
# both; a definitive answer (404 etc.) still fails closed immediately.
_FETCH_ATTEMPTS = 6
_FETCH_BACKOFF_SECONDS = 2.0          # 2, 4, 8, 16, 32 -> ~62 s of patience
_CACHE_TTL_SECONDS = 3600.0
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_cache: dict = {}
_cache_lock = threading.Lock()


def _fetch_fasta(url: str, protein_code: str) -> str:
    """GET the FASTA, retrying only failures that can plausibly be transient."""
    last = None
    for attempt in range(_FETCH_ATTEMPTS):
        if attempt:
            time.sleep(_FETCH_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        try:
            response = requests.get(url, timeout=60)
        except requests.RequestException as error:   # SSL EOF, DNS, reset, timeout
            last = error
            continue
        if response.status_code == 200:
            return response.text
        if response.status_code not in _RETRYABLE_STATUS:
            raise RuntimeError(
                f"UniProt returned {response.status_code} for {protein_code}. "
                "Refusing to guess the target sequence -- scoring against the wrong "
                "target yields metrics that look fine and mean nothing."
            )
        last = RuntimeError(f"UniProt returned {response.status_code} for {protein_code}")
    raise RuntimeError(
        f"UniProt unreachable for {protein_code} after {_FETCH_ATTEMPTS} attempts "
        f"({type(last).__name__}: {last}). Refusing to guess the target sequence."
    ) from last


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
    with _cache_lock:
        hit = _cache.get(protein_code)
    if hit is not None and time.monotonic() - hit[0] < _CACHE_TTL_SECONDS:
        sequence = hit[1]
    else:
        url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
        text = _fetch_fasta(url, protein_code)
        sequence = "".join(
            line.strip() for line in text.splitlines()
            if not line.startswith(">")
        )
        if not sequence:
            raise RuntimeError(f"UniProt returned an empty sequence for {protein_code}")
        with _cache_lock:
            _cache[protein_code] = (time.monotonic(), sequence)
    if clip_interval:
        sequence = sequence[clip_interval[0]:clip_interval[1]]
    return sequence
