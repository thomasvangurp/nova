#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install \
  -c "${repo_root}/requirements/boltz-worker-constraints.txt" \
  -e "${repo_root}/external_tools/boltz[cuda]"
python -m pip install \
  pytorch-lightning==2.5.5 \
  rdkit==2024.9.4 \
  pandas==2.2.3

# Base PyTorch images may ship torchvision/torchaudio wheels tied to an older
# torch. Boltz does not use them, while torchmetrics imports an incompatible
# installed torchvision eagerly.
python -m pip uninstall -y torchvision torchaudio || true

# Populate the shared cache once. Starting multiple score workers against an
# empty cache races two multi-gigabyte downloads to the same destination.
if [[ "${NOVA_PREFETCH_BOLTZ_CACHE:-1}" == "1" ]]; then
  python - <<'PY'
from pathlib import Path

from boltz.main import download_boltz2

download_boltz2(Path.home() / ".boltz")
PY
fi

python - <<'PY'
import cuequivariance_torch
import pandas
import pytorch_lightning
import rdkit
import torch

assert torch.cuda.is_available()
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "rdkit": rdkit.__version__,
    "pandas": pandas.__version__,
    "gpus": torch.cuda.device_count(),
})
PY
