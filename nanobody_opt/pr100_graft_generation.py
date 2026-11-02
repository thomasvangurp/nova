"""Exact-cell leaderboard CDR3 graft lane for PR100 proposal generation.

The validator restricts each epoch to one germline and one exact CDR3 length.
A winning CDR3 can be re-used on a different same-V framework with a few
neutral substitutions; this lane emits exactly those children. The framework
(fr1/cdr1/fr2/cdr2/fr3/fr4) comes from one exact-cell donor and the CDR3 comes
from either another exact-cell donor (verbatim + bounded neutral substitutions)
or a deterministic de-novo recipe sampled from the validated winner motifs.
Both modes share the same framework donor, divergence gate and downstream
gates.

Every emitted residue has an exact donor interval plus an explicit edit list
(for the verbatim mode) or a recorded de-novo CDR3, so history ingestion and
lineage analytics can reconstruct the child from the recorded provenance.
Children still require ordinary live gates.
"""
from __future__ import annotations

from collections import Counter
import random

from nanobody_opt import rules
from nanobody_opt import pr100_reference_donors as reference_donors
from nanobody_opt import pr100_segment_generation as segments
from nanobody_opt.pr100_reference_donors import merit

SOURCE = 'leaderboard_graft'
PROTOCOL = 'imgt-exact-cell-cdr3-graft-v1'
PROPOSAL_LANE = 'graft_cdr3'
DE_NOVO_PART_SOURCE = 'de_novo_recipe'

#: Bounded neutral residue swaps. Keys deliberately exclude charged residues
#: (K/R/D/E), histidine, aromatic residues (F/Y/W), cysteine and proline. None
#: of those are ever touched by the neutral substitution pass.
NEUTRAL_SWAPS = {'S': 'T', 'T': 'S', 'N': 'Q', 'Q': 'N', 'M': 'V',
                 'A': 'S', 'G': 'A', 'I': 'V', 'L': 'I', 'V': 'L'}

_DEFAULTS = {'enabled': False, 'limit': 300, 'cdr3_donors': 8,
             'framework_donors': 8, 'donor_slots': 64}
_ALLOWED_KEYS = frozenset(_DEFAULTS)
_BOUNDS = (('limit', 1, 4800), ('cdr3_donors', 1, 64),
           ('framework_donors', 1, 64), ('donor_slots', 2, 128))
_PROVENANCE_KEYS = ('source_hash', 'source_start', 'source_end', 'region',
                    'sequence', 'imgt_start', 'imgt_end')
_NEUTRAL_SUBSTITUTIONS = 3
_MODES = ('verbatim', 'de_novo')

#: Measured de-novo winner recipe components (121-epoch winner analysis). The
#: recipe is germline-independent: the exact-cell framework donor already
#: supplies the germline, so only CDR3 length conditions the template.
_ANCHORS = ('AR', 'AK', 'SK', 'SR')
_TAILS = ('FDY', 'FDV', 'FDP', 'MDV')
_SHORT_AA = ('T', 'S', 'A', 'G', 'Q', 'N')
_MEDIUM_CORES = ('YPYHF', 'HPYHFK', 'YPYR', 'PYHF')
_LONG_CORES = ('FQPR', 'FQPRY', 'YPYR', 'FPYR', 'PYRF')
_LINKERS = ('QGSDSG', 'QGSGD')
_POSITIVE = ('K', 'R')
_NEGATIVE = ('D', 'E')


def settings(job):
    """Validate and normalize ``job['graft_generation']`` (OFF by default)."""
    supplied = job.get('graft_generation', {})
    if (not isinstance(supplied, dict)
            or type(supplied.get('enabled', False)) is not bool
            or set(supplied) - _ALLOWED_KEYS):
        raise ValueError('invalid graft_generation settings')
    result = {**_DEFAULTS, **supplied}
    for key, low, high in _BOUNDS:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError('invalid graft_generation bound')
    return result


def select_exact_donors(bank, challenge, slots):
    """The exact-cell parent rows in merit order, never a missing-cell fallback."""
    exact, _chosen, _evidence = reference_donors.select(bank, challenge, slots,
                                                        fallback=False)
    return exact


def prepare_framework(row, numbered):
    """Thin wrapper over the shared exact-numbering preparation."""
    return segments.prepare(row, numbered)


def extract_cdr3(row, challenge):
    """The donor's IMGT CDR3, requiring the challenge's exact length."""
    from abnumber import Chain
    cdr3 = Chain(row['sequence'], scheme='imgt', cdr_definition='imgt',
                 use_anarcii=False).cdr3_seq
    if len(cdr3) != challenge['allowed_cdr3_length']:
        raise ValueError('cdr3 length mismatch')
    return cdr3


def whole_identity(left, right):
    """Positional identity over the longer length (zip over the shared prefix)."""
    size = max(len(left), len(right))
    return (sum(a == b for a, b in zip(left, right)) / size) if size else 1.0


def neutral_substitutions(cdr3, n=_NEUTRAL_SUBSTITUTIONS):
    """Deterministic, seedless neutral substitutions; never the last 3 residues.

    Returns the edited CDR3 and one edit record per applied substitution. Each
    edit's ``child_offset`` is relative to the CDR3 start and ``from`` is the
    verbatim donor residue.
    """
    indices = [i for i in range(0, len(cdr3) - 3) if cdr3[i] in NEUTRAL_SWAPS]
    edits = [{'child_offset': i, 'from': cdr3[i], 'to': NEUTRAL_SWAPS[cdr3[i]],
              'kind': 'neutral_cdr3_substitution'} for i in indices[:n]]
    edited = list(cdr3)
    for edit in edits:
        edited[edit['child_offset']] = edit['to']
    return ''.join(edited), edits


def _charge(sequence):
    """Counts of positively (K/R) and negatively (D/E) charged residues."""
    return (sum(1 for c in sequence if c in _POSITIVE),
            sum(1 for c in sequence if c in _NEGATIVE))


def de_novo_cdr3(length, seed):
    """Deterministic, length-conditioned de-novo CDR3 from the winner recipe.

    The invariant anchor (AR/AK/SK/SR) and J-tail (FDY/FDV/FDP/MDV) frame the
    CDR3; the middle is filled from a length-bucket template: minimal neutral
    filler (5-10), a PYR/PYHF core (11-14), an FQPR/PYR core plus an
    alternating K/R-D/E charge network (15-18), or an FQPR/PYR core plus a
    QGSDSG/QGSGD linker padding and charge network (19-23). The charge network
    is tuned so the total (K+R) and (D+E) counts stay within one of each other,
    which bounds the CDR3 charge content that the native developability
    charge-patch gates re-measure.
    """
    if type(length) is not int or length < 5:
        raise ValueError('de-novo CDR3 recipe requires length >= 5')
    rng = random.Random(seed)
    anchor = rng.choice(_ANCHORS)
    tail = rng.choice(_TAILS)
    middle_len = length - 5
    if length <= 10:
        middle = _short_middle(middle_len, rng)
    elif length <= 14:
        middle = _medium_middle(middle_len, rng)
    elif length <= 18:
        middle = _long_middle(middle_len, rng)
    else:
        middle = _xlong_middle(middle_len, rng)
    return anchor + middle + tail


def _fill_middle(prefix, middle_len, rng):
    """Append an alternating K/R-D/E network to hit ``middle_len``, balanced.

    The alternating network keeps the whole middle's (K+R) and (D+E) counts
    within one of each other when ``prefix`` itself is already within one.
    """
    prefix_pos, prefix_neg = _charge(prefix)
    net_len = middle_len - len(prefix)
    imbalance = prefix_pos - prefix_neg
    network = []
    if net_len > 0:
        if net_len % 2 == 0:
            start_positive = bool(rng.getrandbits(1))
        else:
            start_positive = imbalance < 0
        for i in range(net_len):
            positive = start_positive if i % 2 == 0 else not start_positive
            network.append(rng.choice(_POSITIVE if positive else _NEGATIVE))
    return prefix + ''.join(network)


def _short_middle(n, rng):
    """Minimal neutral filler; no aromatic/His core."""
    return ''.join(rng.choice(_SHORT_AA) for _ in range(n))


def _medium_middle(n, rng):
    """One PYR/PYHF core motif, extended with neutral filler to the exact length."""
    core = rng.choice(_MEDIUM_CORES)[:n]
    if len(core) < n:
        core += ''.join(rng.choice(_SHORT_AA) for _ in range(n - len(core)))
    return core


def _long_middle(n, rng):
    """An FQPR/PYR core followed by an alternating charge network."""
    core = rng.choice(_LONG_CORES)[:n]
    return _fill_middle(core, n, rng)


def _xlong_middle(n, rng):
    """An FQPR/PYR core plus a QGSDSG/QGSGD linker, then a charge network."""
    core = rng.choice(_LONG_CORES)
    linker = rng.choice(_LINKERS)
    prefix = (core + linker)[:n]
    return _fill_middle(prefix, n, rng)


def reconstruct_with_edits(row, references):
    """Exact residue-level provenance check for graft rows, edits included.

    Mirrors ``segments.reconstruct`` interval/contiguity checks for donor
    intervals, validates a de-novo CDR3 byte-identical when the cdr3 part
    carries ``source == 'de_novo_recipe'``, then applies each verbatim part's
    neutral substitutions and returns the edited join. Any interval,
    contiguity, source-identity or edit mismatch raises.
    """
    parts = row.get('segment_provenance')
    if not isinstance(parts, list) or len(parts) != 7:
        raise ValueError('invalid graft provenance')
    assembled, cursor, sources, slices = [], 0, set(), []
    for part in parts:
        c = part.get('child_start'); d = part.get('child_end')
        if part.get('source') == DE_NOVO_PART_SOURCE:
            piece = part.get('sequence')
            if (not isinstance(piece, str) or type(c) is not int or type(d) is not int
                    or c != cursor or d - c != len(piece)):
                raise ValueError('graft de-novo cdr3 mismatch')
            assembled.append(piece); cursor = d; slices.append((part, piece))
            continue
        sha = part['source_hash']; source = references.get(sha)
        a, b = part['source_start'], part['source_end']
        if (source is None or rules.sequence_hash(source) != sha or type(a) is not int
                or type(b) is not int or not 0 <= a < b <= len(source)
                or type(c) is not int or type(d) is not int or c != cursor
                or d - c != b - a):
            raise ValueError('graft provenance interval mismatch')
        piece = source[a:b]
        assembled.append(piece); sources.add(sha); cursor = d
        slices.append((part, piece))
    if not 1 <= len(sources) <= 2:
        raise ValueError('graft provenance source mismatch')
    result = ''.join(assembled)
    for part, piece in slices:
        edits = part.get('edits', [])
        if not isinstance(edits, list):
            raise ValueError('invalid graft edits')
        for edit in edits:
            offset = edit.get('child_offset')
            if (edit.get('kind') != 'neutral_cdr3_substitution'
                    or type(offset) is not int or not 0 <= offset < len(piece)
                    or edit.get('from') != piece[offset]
                    or edit.get('to') != NEUTRAL_SWAPS.get(edit['from'])):
                raise ValueError('graft edit mismatch')
            position = part['child_start'] + offset
            result = result[:position] + edit['to'] + result[position + 1:]
    if result != row['sequence']:
        raise ValueError('graft provenance reconstruction mismatch')
    return result


def _emit_row(child, sha, framework, cdr3_part, cdr3_source_hash, neutral_subs,
              shortfall, graft_mode, challenge):
    """Build the shared emitted row shape for both graft modes."""
    provenance, cursor = [], 0
    for name in ('fr1', 'cdr1', 'fr2', 'cdr2', 'fr3'):
        segment = framework['segments'][name]
        end = cursor + len(segment['sequence'])
        provenance.append({k: segment[k] for k in _PROVENANCE_KEYS}
                          | {'child_start': cursor, 'child_end': end})
        cursor = end
    end = cursor + len(cdr3_part['sequence'])
    provenance.append({**cdr3_part, 'child_start': cursor, 'child_end': end})
    cursor = end
    fr4 = framework['segments']['fr4']
    fr4_end = cursor + len(fr4['sequence'])
    provenance.append({k: fr4[k] for k in _PROVENANCE_KEYS}
                      | {'child_start': cursor, 'child_end': fr4_end})
    return {'sequence': child, 'sequence_hash': sha, 'source': SOURCE,
            'proposal_type': PROTOCOL, 'proposal_lane': PROPOSAL_LANE,
            'segment_layout': 'cdr3_graft', 'segment_provenance': provenance,
            'graft_cdr3_source_hash': cdr3_source_hash,
            'graft_framework_source_hash': framework['sequence_hash'],
            'graft_neutral_subs': neutral_subs, 'graft_neutral_subs_shortfall': shortfall,
            'graft_mode': graft_mode, 'requested_v': challenge['allowed_v'],
            'requested_cdr3_length': challenge['allowed_cdr3_length'],
            'acceptance_qualified': False, 'target_scored': False}


def _diverges(framework, cdr3_donor):
    """Hard, cheap divergence gate shared by both graft modes."""
    return (whole_identity(framework['sequence'], cdr3_donor['sequence']) < 0.95
            and (segments.cdr_similarity(framework['segments']['cdr1']['sequence'],
                                         cdr3_donor['segments']['cdr1']['sequence']) < 0.90
                 or segments.cdr_similarity(framework['segments']['cdr2']['sequence'],
                                            cdr3_donor['segments']['cdr2']['sequence']) < 0.90))


def generate(cdr3_donors, framework_donors, challenge, *, limit=300,
             excluded=(), stats=None, modes=_MODES):
    """Emit exact-cell CDR3 grafts, in deterministic merit order, up to ``limit``.

    ``cdr3_donors`` are raw exact-cell bank rows (prepared here). ``framework_donors``
    are already-prepared exact-cell rows from ``prepare_framework``. Each passing
    donor pair emits a verbatim graft (donor CDR3 + neutral substitutions) and,
    when ``de_novo`` is enabled, a de-novo graft (recipe-sampled CDR3). A graft
    needs at least two distinct exact-cell donors, so a shared framework/CDR3
    donor pool with fewer than two donors emits nothing.
    """
    from abnumber import Chain
    if (type(limit) is not int or not 1 <= limit <= 4800
            or not isinstance(cdr3_donors, list) or not isinstance(framework_donors, list)):
        raise ValueError('invalid graft generation input bound')
    modes = tuple(modes)
    if not modes or any(mode not in _MODES for mode in modes):
        raise ValueError('invalid graft mode')
    counters = Counter()
    distinct = {row['sequence_hash'] for row in [*cdr3_donors, *framework_donors]}
    if len(distinct) < 2:
        counters['stop_reason'] = 'insufficient_exact_cell_donors'
        counters['emitted'] = 0
        counters['filled'] = False
        if stats is not None:
            stats.update(counters)
        return []

    def key(row):
        rank, _kind = merit(row, challenge)
        return (*rank, row['sequence_hash'])

    cdr3_donors = sorted(cdr3_donors, key=key)
    framework_donors = sorted(framework_donors, key=key)
    references = {fd['sequence_hash']: fd['sequence'] for fd in framework_donors}
    prepared_cds = []
    for cd in cdr3_donors:
        try:
            chain = Chain(cd['sequence'], scheme='imgt', cdr_definition='imgt',
                          use_anarcii=False)
            numbered = [{'number': pos.number, 'letter': pos.letter, 'aa': aa}
                        for pos, aa in chain]
            prepared = prepare_framework(cd, numbered)
        except Exception:
            counters['donor_numbering_failed'] += 1
            continue
        cdr3 = prepared['segments']['cdr3']['sequence']
        if len(cdr3) != challenge['allowed_cdr3_length']:
            counters['cdr3_length_mismatch'] += 1
            continue
        references[cd['sequence_hash']] = cd['sequence']
        prepared_cds.append((prepared, cdr3))

    excluded = set(excluded)
    result = []
    stop_reason = 'donor_pairs_exhausted'
    de_novo_counter = 0
    for prepared, cdr3_verbatim in prepared_cds:
        edited_cdr3, edits = neutral_substitutions(cdr3_verbatim)
        for fd in framework_donors:
            if fd['sequence_hash'] == prepared['sequence_hash']:
                continue
            if _diverges(fd, prepared):
                counters['divergence_drops'] += 1
                continue
            if 'verbatim' in modes:
                child = (fd['segments']['fr1']['sequence'] + fd['segments']['cdr1']['sequence']
                         + fd['segments']['fr2']['sequence'] + fd['segments']['cdr2']['sequence']
                         + fd['segments']['fr3']['sequence'] + edited_cdr3
                         + fd['segments']['fr4']['sequence'])
                sha = rules.sequence_hash(child)
                if sha in excluded:
                    counters['exact_or_resume_collisions'] += 1
                else:
                    cdr3_part = {k: prepared['segments']['cdr3'][k] for k in _PROVENANCE_KEYS}
                    cdr3_part['edits'] = edits
                    row = _emit_row(child, sha, fd, cdr3_part,
                                    prepared['sequence_hash'], len(edits),
                                    max(0, _NEUTRAL_SUBSTITUTIONS - len(edits)),
                                    'verbatim', challenge)
                    if reconstruct_with_edits(row, references) != child:
                        raise ValueError('graft reconstruction mismatch')
                    result.append(row)
                    if len(result) == limit:
                        stop_reason = 'limit_filled'
                        break
            if 'de_novo' in modes and len(result) < limit:
                seed = challenge['epoch'] * 1_000_000 + de_novo_counter
                de_novo_counter += 1
                generated = de_novo_cdr3(challenge['allowed_cdr3_length'], seed)
                child = (fd['segments']['fr1']['sequence'] + fd['segments']['cdr1']['sequence']
                         + fd['segments']['fr2']['sequence'] + fd['segments']['cdr2']['sequence']
                         + fd['segments']['fr3']['sequence'] + generated
                         + fd['segments']['fr4']['sequence'])
                sha = rules.sequence_hash(child)
                if sha in excluded:
                    counters['exact_or_resume_collisions'] += 1
                else:
                    cdr3_part = {'source': DE_NOVO_PART_SOURCE, 'region': 'cdr3',
                                 'sequence': generated}
                    row = _emit_row(child, sha, fd, cdr3_part, None, 0, 0,
                                    'de_novo', challenge)
                    if reconstruct_with_edits(row, references) != child:
                        raise ValueError('graft reconstruction mismatch')
                    result.append(row)
                    if len(result) == limit:
                        stop_reason = 'limit_filled'
                        break
            if len(result) == limit:
                break
        if len(result) == limit:
            break
    counters['emitted'] = len(result)
    counters['filled'] = stop_reason == 'limit_filled'
    counters['stop_reason'] = stop_reason
    if stats is not None:
        stats.update(counters)
    return result
