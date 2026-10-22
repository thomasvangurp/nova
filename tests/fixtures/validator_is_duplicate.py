# Source: utils/nanobodies.py at b53923caab62d9ad6990ba453a7e17d18b8444c0
def is_duplicate(match):
    """
    Decide whether a search match represents a duplicate submission.
    Uses CDR-focused logic rather than whole-sequence identity alone.
    from: NOVA-nanobody-filter/blob/main/metanano/search/search_engine.py#L134
    """
    identity = match.identity
    cdr_sim = match.cdr_similarity  # may be None

    # Tier 1: Whole-sequence near-identity — definitely a duplicate
    if identity >= 0.95:
        bt.logging.debug(f"match is_duplicate on step 1: high whole-sequence identity {identity:.0%}")
        return True, "near-identical sequence"

    # Tier 2: CDR3-focused check (if CDRs available)
    if cdr_sim is not None:
        cdr3 = cdr_sim.get("CDR3", 0.0)

        # CDR3 identity >= 0.85 = same clonotype, likely same epitope
        if cdr3 >= 0.85:
            bt.logging.debug(f"match is_duplicate on step 2.1: high CDR3 identity {cdr3:.0%}")
            return True, f"CDR3 identity {cdr3:.0%} (same clonotype)"

        # CDR3 >= 0.80 with conserved CDR1+CDR2 = functional duplicate
        cdr1 = cdr_sim.get("CDR1", 0.0)
        cdr2 = cdr_sim.get("CDR2", 0.0)
        if cdr3 >= 0.80 and cdr1 >= 0.90 and cdr2 >= 0.90:
            bt.logging.debug(f"match is_duplicate on step 2.2: high CDR3 identity {cdr3:.0%} with conserved CDR1/CDR2")
            return True, f"CDR3={cdr3:.0%} with conserved CDR1/CDR2"

    # Tier 3: High whole-sequence identity without CDR data
    if identity >= 0.90 and cdr_sim is None:
        bt.logging.debug(f"match is_duplicate on step 3: high whole-sequence identity {identity:.0%} with no CDR data")
        return True, f"whole-sequence identity {identity:.0%} (no CDR data)"

    return False, "novel"
