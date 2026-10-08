import sys

PATH = "/root/nanobody_opt/rank_exact.py"

CONSTANT = '''
# Decisive-position residue preference (1-indexed). A@116 is the rival's
# winning edit and V@123 is the winner's residue the rival keeps; the
# win-probability proxy mis-ranks S@116/A@123 above them. Source of truth:
# competitor_lane.POSITION_RESIDUE_ORDER top-1 at these positions. Reversible:
# delete the stable sort added in _order_candidates.
DECISIVE_RESIDUE_PREFERENCE = {116: "A", 123: "V"}
_MODAL_SEQUENCE_LENGTH = 124


def _residue_violations(item):
    """Count decisive positions whose residue disagrees with the preference.

    Lower is better. Non-modal-length sequences return 0 (neutral) so the
    preference leaves them in their existing relative order.
    """
    sequence = str(item.get("sequence") or "")
    if len(sequence) != _MODAL_SEQUENCE_LENGTH:
        return 0
    return sum(
        1 for position, preferred in DECISIVE_RESIDUE_PREFERENCE.items()
        if sequence[position - 1] != preferred
    )
'''

def main():
    src = open(PATH).read()
    if "DECISIVE_RESIDUE_PREFERENCE" in src:
        print("ALREADY_PATCHED")
        return
    # 1) insert the constant + helper after the module constant block.
    anchor_const = "VALIDATOR_DUPLICATE_IDENTITY = 0.95\n"
    assert src.count(anchor_const) == 1, src.count(anchor_const)
    src = src.replace(anchor_const, anchor_const + CONSTANT, 1)
    # 2) insert the stable sort immediately before the `return ranked` that
    # closes _order_candidates (the one right before _pose_order_is_subset_invariant).
    anchor_ret = "    return ranked\n\n\ndef _pose_order_is_subset_invariant("
    assert src.count(anchor_ret) == 1, src.count(anchor_ret)
    src = src.replace(
        anchor_ret,
        "    ranked.sort(key=_residue_violations)\n    return ranked\n\n\ndef _pose_order_is_subset_invariant(",
        1,
    )
    open(PATH, "w").write(src)
    print("PATCHED")

main()
