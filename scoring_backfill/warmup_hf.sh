#!/bin/bash
/root/bgen_python.sh - <<'PY'
from huggingface_hub import hf_hub_download
for name in ["boltzgen1_diverse.ckpt","boltzgen1_adherence.ckpt","boltzgen1_ifold.ckpt","boltz2_conf_final.ckpt","boltz2_aff.ckpt"]:
    p = hf_hub_download("boltzgen/boltzgen-1", name)
    print("OK", name, flush=True)
p = hf_hub_download("boltzgen/inference-data", "mols.zip", repo_type="dataset")
print("OK mols.zip", flush=True)
PY
echo "WARMUP_DONE"
