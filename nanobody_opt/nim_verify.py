#!/usr/bin/env python3
"""Drive ``nim_backend`` over a list of sequences and write mapped results.

Run on a host that can reach the NIM service (e.g. the NIM box itself):

    python3 nanobody_opt/nim_verify.py \
        --sequences seqs.json --target-seq target_seq.txt \
        --template-cif template.cif --out nim_results.json \
        --diffusion-samples 2

``--sequences`` accepts a bare JSON list of sequences or
``{"sequences": [...]}`` / ``{"cases": [{"sequence": ...}]}`` (the scorer
contract shapes).  For each sequence it writes, per diffusion sample, the
mapped BoltzGen confidence dict plus the raw CIF text, and the analyze-rule
sample selection (argmax 0.8*design_to_target_iptm + 0.2*design_ptm), which is
what the local scorer's analysis task reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanobody_opt.nim_backend import (  # noqa: E402
    NimBoltz2Client,
    predict_nanobodies,
)

ANALYZE_RULE = "argmax(0.8*design_to_target_iptm + 0.2*design_ptm)"


def _analyze_sample(conf: list[dict]) -> int:
    best, best_i = None, 0
    for i, c in enumerate(conf):
        dtti = c.get("design_to_target_iptm")
        dptm = c.get("design_ptm")
        if dtti is None or dptm is None:
            continue
        score = 0.8 * dtti + 0.2 * dptm
        if best is None or score > best:
            best, best_i = score, i
    return best_i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--target-seq", required=True)
    ap.add_argument("--template-cif", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--diffusion-samples", type=int, default=2)
    ap.add_argument("--max-sequences", type=int, default=0)
    a = ap.parse_args()

    raw = json.load(open(a.sequences))
    items = raw if isinstance(raw, list) else (raw.get("cases") or raw.get("sequences") or [])
    seqs = [it["sequence"] if isinstance(it, dict) else it for it in items]
    seqs = [s for s in seqs if isinstance(s, str) and s]
    if a.max_sequences:
        seqs = seqs[: a.max_sequences]

    target_seq = Path(a.target_seq).read_text().strip()
    template_cif = Path(a.template_cif).read_text()
    client = NimBoltz2Client(a.url)

    out = {}
    for i, seq in enumerate(seqs):
        result = predict_nanobodies(
            client, target_seq, [seq], template_cif,
            diffusion_samples=a.diffusion_samples,
        )[seq]
        conf = result.get("confidence") or []
        out[seq] = {
            "elapsed_s": result.get("elapsed_s"),
            "n_samples": len(result.get("structures") or []),
            "analyze_sample_index": _analyze_sample(conf),
            "analyze_rule": ANALYZE_RULE,
            "confidence_per_sample": conf,
            "cif_bytes": [len(c) for c in (result.get("structures") or [])],
        }
        print(f"{i + 1}/{len(seqs)} seq={seq[:8]}... n={out[seq]['n_samples']} "
              f"analyze_i={out[seq]['analyze_sample_index']} "
              f"dtti={conf[out[seq]['analyze_sample_index']]['design_to_target_iptm']}",
              flush=True)
        json.dump(out, open(a.out, "w"), indent=2)

    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
