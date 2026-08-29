#!/usr/bin/env bash
# Install the TensorRT runtime matching the pinned RF-DETR PyTorch 2.7 stack.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config/system.env"
PYTHON="${RFDETR_PYTHON:?RFDETR_PYTHON is not configured}"
export PYTHONNOUSERSITE=1

"${PYTHON}" -m pip install \
  'torch-tensorrt==2.7.0' \
  'tensorrt==10.9.0.34' \
  'onnx>=1.16,<2.0'

"${PYTHON}" - <<'PY'
import tensorrt
import onnx
import torch
import torch_tensorrt

assert torch.__version__.startswith("2.7."), torch.__version__
assert torch_tensorrt.__version__ == "2.7.0", torch_tensorrt.__version__
assert tensorrt.__version__ == "10.9.0.34", tensorrt.__version__
assert torch.cuda.is_available()
print(
    "RF-DETR TensorRT environment ready:",
    f"torch={torch.__version__}",
    f"torch_tensorrt={torch_tensorrt.__version__}",
    f"tensorrt={tensorrt.__version__}",
    f"onnx={onnx.__version__}",
    f"gpu={torch.cuda.get_device_name(0)}",
)
PY
