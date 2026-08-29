#!/usr/bin/env bash
# Compile a fine-tuned RF-DETR checkpoint for dynamic multi-camera batching.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
TOOL="${ROOT}/components/tool_runtime_v1_6"
MODEL_SIZE="${1:-xlarge}"
if (( $# > 0 )); then
  shift
fi
MIN_BATCH="${TOOL_TRT_MIN_BATCH_SIZE:-1}"
OPT_BATCH="${TOOL_TRT_OPT_BATCH_SIZE:-3}"
MAX_BATCH="${TOOL_TRT_MAX_BATCH_SIZE:-3}"

case "${MODEL_SIZE}" in
  small) CHECKPOINT="${TOOL_CHECKPOINT_SMALL:-${TOOL_CHECKPOINT:-}}" ;;
  medium) CHECKPOINT="${TOOL_CHECKPOINT_MEDIUM:-}" ;;
  large) CHECKPOINT="${TOOL_CHECKPOINT_LARGE:-}" ;;
  xlarge) CHECKPOINT="${TOOL_CHECKPOINT_XLARGE:-}" ;;
  *)
    echo "usage: $0 small|medium|large|xlarge [output.plan]" >&2
    exit 2
    ;;
esac
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found for ${MODEL_SIZE}: ${CHECKPOINT:-<unset>}" >&2
  exit 2
fi
OUTPUT="${1:-${ROOT}/models/tensorrt/rfdetr_seg_${MODEL_SIZE}_fp16_b${MIN_BATCH}-${MAX_BATCH}_sm86.plan}"
if (( $# > 0 )); then
  shift
fi
mkdir -p "$(dirname "${OUTPUT}")"
export PYTHONPATH="${TOOL}/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${RFDETR_PYTHON}" \
  "${TOOL}/algorithm/tools/compile_rfdetr_tensorrt.py" \
  --checkpoint "${CHECKPOINT}" \
  --model-size "${MODEL_SIZE}" \
  --output "${OUTPUT}" \
  --minimum-batch-size "${MIN_BATCH}" \
  --optimum-batch-size "${OPT_BATCH}" \
  --maximum-batch-size "${MAX_BATCH}" \
  --workspace-gib "${TOOL_TRT_WORKSPACE_GIB:-3.0}" \
  "$@"
