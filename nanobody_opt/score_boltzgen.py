"""Exact scorer: run boltzgen over sequences and emit the 10 ranked metrics.

The counterpart of the compound side's score_boltz_batch.py. The proxy is a
cheap filter; THIS produces the numbers the validator actually ranks on.

Calls BoltzgenWrapper.run_nanobody_inference, which returns raw per-sequence
components. Deliberately NOT score_nanobodies(): that also ranks, and ranks are
relative to whatever batch you happen to pass, so they cannot be compared
against the archive's published values or across runs. Raw metrics can.

Must run under bgen_python.sh, which supplies the PYTHONPATH shim and
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.
"""
from __future__ import annotations
import argparse, json, time

METRICS = ("design_iiptm", "design_ptm", "design_to_target_iptm",
           "min_design_to_target_pae", "interaction_pae",
           "plip_hbonds_refolded", "plip_saltbridge_refolded",
           "delta_sasa_refolded", "liability_score", "liability_num_violations")


def build_subnet_config(config_yaml: str, boltzgen_yaml: str, target: str) -> dict:
    import yaml
    cfg = yaml.safe_load(open(config_yaml))
    bg = yaml.safe_load(open(boltzgen_yaml))
    # The wrapper expects the validator's flattened config. Keep the upstream
    # keys verbatim rather than inventing names: a silently ignored key would
    # change the scoring protocol without any error.
    merged = dict(cfg)
    merged.update(bg)
    merged["nanobody_target"] = [target]
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True,
                    help="JSON: a list of sequences, or {cases:[{sequence:...}]}")
    ap.add_argument("--config", required=True)
    ap.add_argument("--boltzgen-config", required=True)
    ap.add_argument("--target", default="P20809")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    raw = json.load(open(a.sequences))
    if isinstance(raw, dict):
        seqs = [c["sequence"] for c in raw.get("cases", raw.get("sequences", []))]
    else:
        seqs = list(raw)
    if not seqs:
        raise SystemExit("no sequences given")
    print(f"scoring {len(seqs)} sequences against {a.target}", flush=True)

    subnet_config = build_subnet_config(a.config, a.boltzgen_config, a.target)
    from boltzgen.boltzgen_wrapper import BoltzgenWrapper

    # one synthetic uid; the wrapper keys everything by uid then sequence
    by_uid = {0: seqs}
    started = time.time()
    wrapper = BoltzgenWrapper()
    components = wrapper.run_nanobody_inference(by_uid, subnet_config)
    elapsed = time.time() - started
    print(f"boltzgen finished in {elapsed:.1f}s "
          f"({elapsed/len(seqs):.1f}s per sequence)", flush=True)

    json.dump({"target": a.target, "n": len(seqs),
               "wall_seconds": elapsed,
               "per_nanobody_components": components},
              open(a.out, "w"), indent=2, default=str)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
