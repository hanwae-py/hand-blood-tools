#!/usr/bin/env bash
# Install Blood inference deps into the current interpreter.
# Run after creating the `blood` (or `blood-ros` / `.venv-blood`) environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e "${ROOT}/third_party/rfdetr"
python -m pip install -e "${ROOT}/third_party/cutie" --no-deps
python -m pip install \
  "hydra-core>=1.3.2" \
  omegaconf \
  einops \
  "Pillow>=9.5" \
  "opencv-python>=4.8" \
  pyyaml \
  numpy \
  scipy \
  tqdm
python -m pip install -e "${ROOT}"
python -c "import torch, rfdetr, cutie, blood; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
