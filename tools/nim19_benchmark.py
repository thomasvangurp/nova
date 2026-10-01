#!/usr/bin/env python3
"""Cold/warm Boltz NIM smoke tests and explicitly configured affinity cases.

No raw HTTP bodies, exception messages, auth responses, or structures are logged.
HTTP 200 alone is not success: required predictions must be finite and present.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import requests

GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFCYGD"
    "QIQEQYKGIPLDGDQVQAVNGHEFEIEGEGEGRPYEGTQTAQLNKFCDKLPVMHYKQFFDSGNYNTLS"
    "AKAGFPFKVPHTYNNSSFVVKQKPGMVFKFIHGKDPGLNGQTVFLMVGGISQNLSGSSNLGVGYTFVQ"
    "KTSVLLESEIKKRLRGFHTRGAVTQGLHQFVNLPTLVTQVLDGDMSQLLQVT"
)
HEMOGLOBIN = (
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALT"
    "NAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTV"
    "LTSKYR"
)
P40261 = (
    "SGFTSKDTYLSHFNPRDYLEKYYKFGSRHSAESQILKHLLKNLFKIFCLDGVKGDLLIDIGSGPTIYQLLSACESFKEI"
    "VVTDYSDQNLQELEKWLKKEPEAFDWSPVVTYVCDLEGNRVKGPEKEEKLRQAVKQVLKCDVTQSQPLGAVPLPPADCVL"
    "STLCLDAACPDLPTYCRALRNLGSLLKPGGFLVIMDALKSSYYMIGEQKFSSLPLGREAVEAAVKEAGYTIEWFEVISQSY"
    "SSTMANNEGLFSLVARKL"
)
METRICS = {
    "total_time_seconds", "input_preparation_time_seconds",
    "dataloader_setup_time_seconds", "model_inference_time_seconds",
    "postprocessing_time_seconds", "response_construction_time_seconds",
    "affinity_prediction_time_seconds",
}
AFFINITY_FIELDS = {
    "affinity_pic50", "affinity_pred_value", "affinity_probability_binary",
    "model_1_affinity_pred_value", "model_1_affinity_probability_binary",
    "model_2_affinity_pred_value", "model_2_affinity_probability_binary",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def first_finite(data, key):
    values = data.get(key)
    return values[0] if isinstance(values, list) and values and finite(values[0]) else None


def smoke_cases():
    common = {
        "recycling_steps": 3, "sampling_steps": 50,
        "diffusion_samples": 1, "step_scale": 1.638,
        "output_format": "mmcif",
    }
    return [
        {"name": "user_gfp_structure", "purpose": "service_smoke_only", "request": {
            **common, "polymers": [{"id": "A", "molecule_type": "protein", "sequence": GFP}],
        }},
        {"name": "user_hemoglobin_affinity", "purpose": "service_smoke_only", "request": {
            **common, "polymers": [{"id": "A", "molecule_type": "protein", "sequence": HEMOGLOBIN}],
            "ligands": [{"id": "HEME", "smiles": "[Fe+2].C1=CC2=NC1=CC3=NC(=CC4=NC(=CC5=NC(=C2)C=C5)C=C4)C=C3", "predict_affinity": True}],
            "sampling_steps_affinity": 100, "diffusion_samples_affinity": 3,
        }},
    ]


def target_cases(msa_path):
    """Real target/MSA timing controls, not legal portfolios or parity proof."""
    alignment = msa_path.read_text()
    chunks = alignment.split(">")
    first = "".join(chunks[1].splitlines()[1:]) if len(chunks) > 1 else ""
    if "".join(x for x in first if not x.islower()).replace("-", "") != P40261:
        raise ValueError("msa_target_mismatch")
    request = {
        "polymers": [{"id": "A", "molecule_type": "protein", "sequence": P40261,
                      "msa": {"uniref90": {"a3m": {"format": "a3m", "alignment": alignment, "rank": 0}}}}],
        "ligands": [{"id": "L1", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "predict_affinity": True}],
        "recycling_steps": 3, "sampling_steps": 100, "diffusion_samples": 1,
        "sampling_steps_affinity": 100, "diffusion_samples_affinity": 3,
        "affinity_mw_correction": True, "step_scale": 1.638, "output_format": "mmcif",
    }
    return [{"name": f"P40261_aspirin_parallel_{parallel}",
             "purpose": "actual_target_timing_not_validator_parity",
             "unresolved_contract": "Separate affinity recycling_steps=5 must be verified in backend.",
             "request": {**request, "max_parallel_samples": parallel}}
            for parallel in [1, 3]]


def run_request(session, base_url, case, repetition):
    request = case["request"]
    started = time.perf_counter()
    record = {"case": case["name"], "repetition": repetition,
              "phase": "first_request" if repetition == 0 else "warm",
              "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
    try:
        response = session.post(base_url + "/biology/mit/boltz2/predict", json=request, timeout=(10, 1800))
        record.update(http_status=response.status_code, wall_seconds=time.perf_counter() - started)
        if response.status_code != 200:
            # Never print response.text or arbitrary API error strings.
            record.update(passed=False, failure="http_error")
            return record
        result = response.json()
        confidence = first_finite(result, "confidence_scores")
        structures = result.get("structures")
        structure_ok = isinstance(structures, list) and len(structures) == request.get("diffusion_samples", 1) and all(
            isinstance(x, dict) and x.get("format") == "mmcif" and isinstance(x.get("structure"), str) and x["structure"].strip().startswith("data_") for x in structures
        )
        record["confidence"] = confidence
        record["structure_present"] = structure_ok
        metrics = result.get("metrics", {})
        record["metrics"] = {k: v for k, v in metrics.items() if k in METRICS and finite(v)} if isinstance(metrics, dict) else {}
        expected = [x["id"] for x in request.get("ligands", []) if x.get("predict_affinity")]
        affinity_ok = True
        record["affinities"] = {}
        for ligand in expected:
            values = result.get("affinities", {}).get(ligand, {})
            allowlisted = {k: first_finite(values, k) for k in AFFINITY_FIELDS}
            probability = allowlisted["affinity_probability_binary"]
            if allowlisted["affinity_pred_value"] is None or probability is None or not 0 <= probability <= 1:
                affinity_ok = False
            record["affinities"][ligand] = allowlisted
        record["passed"] = structure_ok and confidence is not None and affinity_ok
        if not record["passed"]:
            record["failure"] = "missing_or_nonfinite_prediction"
    except requests.exceptions.Timeout:
        record.update(wall_seconds=time.perf_counter()-started, passed=False, failure="timeout")
    except requests.exceptions.RequestException:
        record.update(wall_seconds=time.perf_counter()-started, passed=False, failure="transport")
    except (ValueError, TypeError, AttributeError, KeyError):
        record.update(wall_seconds=time.perf_counter()-started, passed=False, failure="schema")
    return record


def summary(records):
    result = {}
    for name in dict.fromkeys(x["case"] for x in records):
        values = [x for x in records if x["case"] == name]
        warm = [x["wall_seconds"] for x in values if x["phase"] == "warm" and x["passed"]]
        result[name] = {
            "passed_requests": sum(x["passed"] for x in values),
            "total_requests": len(values),
            "first_request_seconds": values[0]["wall_seconds"],
            "warm_median_seconds": statistics.median(warm) if warm else None,
            "warm_predictions_per_hour": 3600/statistics.mean(warm) if warm else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--extra-cases", type=Path)
    parser.add_argument("--target-msa", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warm_repeats < 1:
        parser.error("--warm-repeats must be at least 1")
    cases = smoke_cases()
    if args.extra_cases:
        cases += json.loads(args.extra_cases.read_text())
    if args.target_msa:
        cases += target_cases(args.target_msa)
    records = []
    artifact = {
        "cases": cases, "records": records, "summary": {},
        "notes": ["First request per case is not necessarily a fresh-process cold start.",
                  "Smoke workloads do not establish validator parity or competitive search throughput.",
                  "Request settings, inputs and returned scalar metrics are retained; raw server bodies are not."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        artifact["summary"] = summary(records)
        temporary = args.output.with_suffix(".partial")
        temporary.write_text(json.dumps(artifact, indent=2))
        temporary.replace(args.output)

    with requests.Session() as session:
        for case in cases:
            for repetition in range(args.warm_repeats+1):
                record = run_request(session, args.base_url.rstrip("/"), case, repetition)
                records.append(record)
                persist()
                print(json.dumps({"event": "benchmark_request", **record}), flush=True)
                if not record["passed"]:
                    break
    persist()
    print(json.dumps({"event": "benchmark_summary", "summary": artifact["summary"]}), flush=True)
    return 0 if all(x["passed"] for x in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
