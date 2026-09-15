"""Locate VHH framework/CDR regions so mutation targets only what matters.

The reference miner mutates random positions across all 117 residues
(miner.py:281, mutate_sequence). That is doubly wasteful:

  * Framework mutations cannot improve binding -- the paratope is the CDR loops --
    but they DO erode the two gates the validator applies
    (min_nativeness_score 0.45, min_human_framework_score 0.85). Every framework
    mutation is downside with no upside.
  * It searches ~117 positions when only ~35 can help.

Region boundaries come from conserved VHH anchors rather than a numbering
library, so this has no anarci/ImmuneBuilder dependency. Many accepted archive
sequences use WGxG or FGxG at the FR4 boundary, but elite variants can mutate
that motif. Immunoglobulin FR4 is eleven residues in the supported aligned VHH
family, which provides a bounded fallback after the other anchors agree:
    Cys22 ... Trp(WVRQ) ... Cys96(YYC) ... [WF]GxG or 11-residue FR4
"""
import re
from dataclasses import dataclass


@dataclass
class Regions:
    cdr1: tuple[int, int]
    cdr2: tuple[int, int]
    cdr3: tuple[int, int]

    def cdr_positions(self) -> list[int]:
        out = []
        for start, end in (self.cdr1, self.cdr2, self.cdr3):
            out.extend(range(start, end))
        return out


_FR4_ANCHOR = re.compile(r"[WF]G.G")


def find_regions(seq: str) -> Regions:
    """Anchor-based VHH parse. Raises if the sequence is not VHH-shaped."""
    # FR1 ends at the conserved cysteine (C) around position 22
    c1 = seq.find("C", 18, 28)
    if c1 < 0:
        raise ValueError("no FR1 cysteine anchor near position 22")
    # FR2 begins at the conserved W of the WVRQ motif
    w = seq.find("WVRQ", c1)
    if w < 0:
        w = seq.find("W", c1 + 8)
    if w < 0:
        raise ValueError("no FR2 tryptophan anchor")
    cdr1 = (c1 + 3, w)

    # FR3 ends at the second conserved cysteine, in the YYC / YFC motif
    c2 = max(seq.rfind("YYC"), seq.rfind("YFC"), seq.rfind("YHC"))
    if c2 < 0:
        raise ValueError("no FR3 cysteine (YYC) anchor")
    c2 += 2

    # CDR2 sits between FR2 and FR3; FR2 is ~14 residues after the W anchor
    cdr2 = (w + 14, w + 14 + 17)

    # FR4 starts at the WGxG/FGxG motif after CDR3. The original WG-only
    # heuristic rejects current sequences that the validator numbers as VHH.
    m = _FR4_ANCHOR.search(seq, c2)
    if m:
        fr4_start = m.start()
    else:
        # Validator-accepted P20809 leaders include FR4 variants such as
        # ``FAAGT...`` that no longer match [WF]GxG.  Their V-domain alignment
        # retains the standard 11-residue FR4.  Use that invariant only after
        # the independent FR1, FR2, and FR3 anchors above have all matched.
        fr4_start = len(seq) - 11
        if fr4_start <= c2 + 3:
            raise ValueError("no plausible 11-residue FR4 after CDR3")
    cdr3 = (c2 + 1, fr4_start)
    return Regions(cdr1=cdr1, cdr2=cdr2, cdr3=cdr3)
