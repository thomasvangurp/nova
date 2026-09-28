#!/usr/bin/env python3
"""Chain-backed money collector: read stake + free TAO + alpha prices from finney.

Replaces the Taostats API dependency entirely.  Outputs one JSON line:
  {"block": ..., "free_tao": ..., "positions": {netuid: {"alpha": ..., "alpha_price_tao": ...}}, "sampled_at": ...}
"""
import json
import time

import bittensor as bt

COLDKEY = "5EkCwcNTfxeNbbhNSEJMSaGbW2XMVuhzouZkEan5ozvSsJdQ"


def main() -> None:
    sub = bt.Subtensor(network="finney")
    block = sub.get_current_block()
    free_tao = float(sub.get_balance(COLDKEY).tao)

    positions: dict[int, float] = {}
    for s in sub.get_stake_info_for_coldkey(COLDKEY):
        netuid = int(s.netuid)
        positions[netuid] = positions.get(netuid, 0.0) + float(s.stake.tao)

    out_positions = {}
    for netuid, alpha in positions.items():
        try:
            price_tao = float(sub.get_subnet_price(netuid).tao)
        except Exception:
            price_tao = 0.0
        out_positions[str(netuid)] = {"alpha": alpha, "alpha_price_tao": price_tao}

    print(json.dumps({
        "block": block,
        "free_tao": free_tao,
        "positions": out_positions,
        "sampled_at": time.time(),
    }))


if __name__ == "__main__":
    main()
