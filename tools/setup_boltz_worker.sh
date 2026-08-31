#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install -e "${repo_root}/external_tools/boltz[cuda]"
python -m pip install \
  pytorch-lightning==2.5.5 \
  rdkit==2024.9.4 \
  pandas==2.2.3

# Base PyTorch images may ship torchvision/torchaudio wheels tied to an older
# torch. Boltz does not use them, while torchmetrics imports an incompatible
# installed torchvision eagerly.
python -m pip uninstall -y torchvision torchaudio || true

python - <<'PY'
import cuequivariance_torch
import pandas
import pytorch_lightning
import rdkit
import torch

assert torch.cuda.is_available()
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "rdkit": rdkit.__version__,
    "pandas": pandas.__version__,
    "gpus": torch.cuda.device_count(),
})
PY
