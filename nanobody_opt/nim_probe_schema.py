#!/usr/bin/env python3
"""One-shot probe of the local NIM boltz2 predict response schema.

Dumps top-level keys, score array shapes, and the CIF header/length of the
first returned structure, so the backend can be written against the real
contract instead of a guess. Run on the NIM box (no boltzgen import needed).
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/biology/mit/boltz2/predict"


def main() -> int:
    target_seq = open(sys.argv[1]).read().strip()
    template_cif = open(sys.argv[2]).read()
    design_seq = open(sys.argv[3]).read().strip()

    payload = {
        "polymers": [
            {
                "id": "A",
                "molecule_type": "protein",
                "sequence": target_seq,
                "structural_templates": [
                    {"structure": template_cif, "format": "cif", "chain_id": "A"}
                ],
            },
            {"id": "B", "molecule_type": "protein", "sequence": design_seq},
        ],
        "diffusion_samples": 2,
        "recycling_steps": 3,
        "sampling_steps": 50,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print(f"elapsed={time.time() - t0:.1f}s", flush=True)
    print("TOP_KEYS=", sorted(body.keys()), flush=True)
    for k, v in body.items():
        if isinstance(v, list):
            print(f"  {k}: list len={len(v)}", flush=True)
            if v and isinstance(v[0], dict):
                print(f"    [0] keys={sorted(v[0].keys())}", flush=True)
                for kk, vv in v[0].items():
                    if isinstance(vv, str):
                        print(f"    [0].{kk}: str len={len(vv)}", flush=True)
                    elif isinstance(vv, list):
                        print(f"    [0].{kk}: list len={len(vv)}", flush=True)
                    else:
                        print(f"    [0].{kk}: {vv!r}", flush=True)
            elif v and isinstance(v[0], (int, float, str)):
                print(f"    [0]: {v[0]!r}", flush=True)
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={sorted(v.keys())}", flush=True)
        else:
            print(f"  {k}: {v!r}", flush=True)

    structs = body.get("structures") or []
    if structs:
        cif = structs[0].get("structure", "")
        lines = cif.splitlines()
        print(f"CIF[0] lines={len(lines)} bytes={len(cif)}", flush=True)
        # print lines that carry confidence / pae markers
        for ln in lines:
            if any(t in ln for t in ("_ma_qa", "_pae", "PAE", "plddt", "iptm", "ptm")):
                print("CIF_QA:", ln[:200], flush=True)
        print("CIF_HEAD:", flush=True)
        for ln in lines[:30]:
            print("  ", ln[:160], flush=True)

    json.dump(body, open(sys.argv[4], "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
