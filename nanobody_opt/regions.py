"""Locate VHH framework/CDR regions so mutation targets only what matters.

The reference miner mutates random positions across all 117 residues
(miner.py:281, mutate_sequence). That is doubly wasteful:

  * Framework mutations cannot improve binding -- the paratope is the CDR loops --
    but they DO erode the two gates the validator applies
    (min_nativeness_score 0.45, min_human_framework_score 0.85). Every framework
    mutation is downside with no upside.
  * It searches ~117 positions when only ~35 can help.

Region boundaries come from conserved VHH anchors rather than a numbering
library, so this has no anarci/ImmuneBuilder dependency. Current accepted
archive sequences use both WGxG and FGxG at the FR4 boundary:
    Cys22 ... Trp(WVRQ) ... Cys96(YYC) ... [WF]GxG (FR4)
"""
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
    import re
    m = re.search(r"[WF]G.G", seq[c2:])
    if not m:
        raise ValueError("no FR4 [WF]GxG anchor")
    cdr3 = (c2 + 1, c2 + m.start())
    return Regions(cdr1=cdr1, cdr2=cdr2, cdr3=cdr3)
