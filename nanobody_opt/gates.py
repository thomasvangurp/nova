"""Bit-exact local replication of the validator's Layer-1 nanobody gates.

Ported verbatim from upstream utils/nanobodies.py and neurons/validator/
nanobody_validity.py (fetched 2026-09-07). We submit num_sequences=1 and a
single failing gate rejects the ENTIRE submission for the epoch, so every gate
is replicated locally and a candidate is never committed unchecked.

Thresholds are read from the LIVE upstream config, never hardcoded -- a stale
local threshold (0.7 vs 0.6) is what cost us compound epoch 24978.

Structure-dependent gates (TNP red regions, IgBLAST nativeness/humanness) are
NOT here; they need external tools and live in tnp.py. check_all() reports
which gates it could not evaluate rather than implying a pass.
"""
from __future__ import annotations

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")


def normalize_seq(seq: str) -> str:
    return seq.strip().upper()


def max_run_length(s: str) -> int:
    if not s:
        return 0
    best = cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def max_di_repeat_pairs(s: str) -> int:
    """Max AB pairs in a contiguous ABAB... region. 'GSGSGS' -> 3."""
    best = 0
    n = len(s)
    for i in range(n - 3):
        a, b = s[i], s[i + 1]
        if a == b:
            continue  # AA runs are the homopolymer gate's job
        pairs = 1
        j = i + 2
        while j + 1 < n and s[j] == a and s[j + 1] == b:
            pairs += 1
            j += 2
        if pairs > best:
            best = pairs
    return best


def has_plausible_cys_pair(seq: str, min_sep: int, max_sep: int) -> bool:
    pos = [i for i, aa in enumerate(seq) if aa == "C"]
    for i in range(len(pos) - 1):
        for j in range(i + 1, len(pos)):
            sep = pos[j] - pos[i]
            if sep < min_sep:
                continue
            if sep > max_sep:
                break
            return True
    return False


def looks_like_signal_peptide(seq: str, window: int, hydro_min: int,
                              scan_prefix: int) -> bool:
    prefix = seq[: min(len(seq), scan_prefix)]
    if len(prefix) < window:
        return False
    count = sum(1 for aa in prefix[:window] if aa in HYDROPHOBIC)
    if count >= hydro_min:
        return True
    for i in range(window, len(prefix)):
        if prefix[i - window] in HYDROPHOBIC:
            count -= 1
        if prefix[i] in HYDROPHOBIC:
            count += 1
        if count >= hydro_min:
            return True
    return False


# Gates the validator applies that this module cannot evaluate from sequence
# alone. Named explicitly so a caller can never mistake silence for a pass.
EXTERNAL_GATES = (
    "igblast_vhh_nativeness",     # min_nativeness_score
    "igblast_human_framework",    # min_human_framework_score
    "classify_vhh_vh",            # enforce_vhh_hallmarks: reject domain_type == VH
    "tnp_developability",         # 6 TNP red regions
    "historical_uniqueness",      # HF registry, all previous submissions
    "diversity_vs_top50",         # MinHash Jaccard >= 0.9 against top 50
)


def check_sequence(seq: str, cfg: dict) -> list[str]:
    """Return the list of Layer-1 gates this sequence FAILS. Empty == passes."""
    s = normalize_seq(seq)
    fails = []
    if not (cfg["min_sequence_length"] <= len(s) <= cfg["max_sequence_length"]):
        fails.append(f"length={len(s)}")
    if set(s) - ALLOWED_AAS:
        fails.append(f"invalid_aas={''.join(sorted(set(s) - ALLOWED_AAS))}")
    if max_run_length(s) > cfg["max_homopolymer_run"]:
        fails.append(f"homopolymer={max_run_length(s)}")
    if max_di_repeat_pairs(s) > cfg["max_di_repeat_pairs"]:
        fails.append(f"di_repeat={max_di_repeat_pairs(s)}")
    if s.count("C") < cfg["min_cysteines"]:
        fails.append(f"cysteines={s.count('C')}")
    if cfg["min_cysteines"] > 1 and not has_plausible_cys_pair(
            s, cfg["cys_pair_min_separation"], cfg["cys_pair_max_separation"]):
        fails.append("no_plausible_cys_pair")
    if cfg.get("reject_signal_peptides", True) and looks_like_signal_peptide(
            s, cfg["sp_window"], cfg["sp_hydro_min_in_window"],
            cfg["sp_scan_prefix"]):
        fails.append("signal_peptide")
    return fails
