#!/usr/bin/env bash
# Run one shared dynamic-batch TensorRT engine for all tool-camera workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/perception_runtime_env.sh" local-fast
TOOL="${ROOT}/components/tool_runtime_v1_6"
MIN_BATCH="${TOOL_TRT_MIN_BATCH_SIZE:-1}"
MAX_BATCH="${TOOL_TRT_MAX_BATCH_SIZE:-3}"
ENGINE="${TOOL_TRT_ENGINE:-${ROOT}/models/tensorrt/rfdetr_seg_xlarge_fp16_b${MIN_BATCH}-${MAX_BATCH}_sm86.plan}"
SOCKET="${TOOL_TRT_SERVER_SOCKET:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/pnu-rfdetr-trt.sock}"
STATS="${TOOL_TRT_STATS_PATH:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/pnu-rfdetr-trt-stats.json}"
CHECKPOINT="${TOOL_CHECKPOINT_XLARGE:-}"
if [[ ! -f "${ENGINE}" ]]; then
  echo "TensorRT engine not found: ${ENGINE}" >&2
  echo "Run: scripts/compile_tool_rfdetr_trt.sh xlarge" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "XLarge checkpoint not found: ${CHECKPOINT:-<unset>}" >&2
  exit 2
fi
export PYTHONPATH="${TOOL}/algorithm/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${RFDETR_PYTHON}" \
  "${TOOL}/algorithm/tools/run_rfdetr_trt_batch_server.py" \
  --engine "${ENGINE}" \
  --checkpoint "${CHECKPOINT}" \
  --model-size xlarge \
  --socket "${SOCKET}" \
  --stats "${STATS}" \
  --maximum-batch-size "${MAX_BATCH}" \
  --batch-window-ms "${TOOL_TRT_BATCH_WINDOW_MS:-0.0}"
